"""CPU-only independent audit of chronological M3 samples and their code mapping."""

import hashlib
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def resolve_project_output(root: Path, path: Path) -> Path:
    root = root.resolve()
    output = (root / path).resolve()
    allowed = (root / "outputs", root / "data/processed")
    if not any(output.is_relative_to(parent) and output != parent for parent in allowed):
        raise ValueError("Output must be a fresh subdirectory of project outputs or data/processed")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    return output


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"Expected an object: {path}:{number}")
                yield row


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def check_source_pins(source: dict, histories: Path, mapping: Path, metadata: dict) -> None:
    for key, path in (("expected_histories_sha256", histories),
                      ("expected_semantic_ids_sha256", mapping)):
        if source.get(key) is not None:
            _require(sha256(path) == source[key], f"Source fingerprint mismatch: {key}")
    for key, field in (("expected_semantic_id_model_type", "model_type"),
                       ("expected_semantic_id_variant", "variant")):
        if source.get(key) is not None:
            _require(metadata.get(field) == source[key], f"Semantic ID metadata mismatch: {field}")
    if source.get("require_catalog_quality_gate"):
        _require(metadata.get("catalog_quality_gate", {}).get("passed") is True,
                 "Semantic ID catalog quality gate did not pass")


def audit_m3_sequences(
    prepared: Path,
    sequences: Path,
    semantic_ids: Path,
    *,
    reference_sequences: Path | None = None,
    expected_mapping_sha256: str | None = None,
) -> dict:
    """Check all samples directly against raw histories; do not call the sample builder.

    This offline data audit reads held-out rows for integrity checks, not for fitting,
    model selection or filtering the evaluation denominator. Training does not call it.
    """
    manifest = _read_json(sequences / "manifest.json")
    dataset = _read_json(prepared / "manifest.json")
    mapping = _read_json(semantic_ids)
    assignments, metadata = mapping["assignments"], mapping["metadata"]
    mapping_hash = sha256(semantic_ids)
    _require(mapping_hash == manifest["semantic_ids_sha256"], "Mapping hash differs from samples")
    if expected_mapping_sha256 is not None:
        _require(mapping_hash == expected_mapping_sha256, "Mapping differs from frozen experiment")
    _require(sha256(prepared / "histories.jsonl") == manifest["histories_sha256"],
             "History hash differs from samples")
    for name, key in (("source_dataset", "dataset"), ("category", "category"),
                      ("dataset_revision", "revision")):
        _require(metadata[name] == dataset[key], "Mapping/prepared dataset mismatch")
    _require(manifest["dataset"] == dataset["dataset"]
             and manifest["category"] == dataset["category"]
             and manifest["dataset_revision"] == dataset["revision"],
             "Sample/prepared dataset mismatch")
    _require(len(assignments) == manifest["codebook_item_count"], "Catalog count mismatch")
    for code in assignments.values():
        _require(len(code) == 3 and all(isinstance(token, str) for token in code),
                 "M3 requires three string code tokens")
    histories = sorted(_rows(prepared / "histories.jsonl"), key=lambda row: row["user_id"])
    _require(len({row["user_id"] for row in histories}) == len(histories) == manifest["users"],
             "Duplicate/missing source users")
    max_history = manifest["max_history"]
    _require(isinstance(max_history, int) and max_history > 0, "Invalid history window")
    expected = {split: [] for split in ("train", "validation", "test")}
    for user in histories:
        items, times = user["train_items"], user["train_timestamps"]
        _require(bool(items) and len(items) == len(times) and sorted(times) == times,
                 "Invalid source training timeline")
        _require(times[-1] <= user["validation_timestamp"] <= user["test_timestamp"],
                 "Source held-out event precedes its history")
        for index in range(1, len(items)):
            expected["train"].append((user["user_id"], items[max(0, index-max_history):index],
                                      items[index]))
        expected["validation"].append((user["user_id"], items[-max_history:],
                                       user["validation_item"]))
        observed = [*items, user["validation_item"]]
        expected["test"].append((user["user_id"], observed[-max_history:], user["test_item"]))
    counts, hashes, train_users, train_codes = {}, {}, set(), set()
    validation_codes = []
    for split, expected_rows in expected.items():
        path = sequences / f"{split}.jsonl"
        hashes[split] = sha256(path)
        if manifest.get("example_file_sha256"):
            _require(hashes[split] == manifest["example_file_sha256"][split],
                     f"Recorded sample hash mismatch: {split}")
        count = 0
        for actual, wanted in zip_longest(_rows(path), expected_rows):
            _require(actual is not None and wanted is not None, f"Wrong sample count: {split}")
            user_id, context, target = wanted
            _require(actual.get("split") == split and actual.get("user_id") == user_id
                     and actual.get("history_items") == context
                     and actual.get("target_item") == target,
                     f"Chronological item boundary mismatch: {split} row {count}")
            _require(actual.get("history_codes") == [assignments[item] for item in context]
                     and actual.get("target_code") == assignments[target],
                     f"Item/code correspondence mismatch: {split} row {count}")
            if split == "train":
                train_users.add(user_id)
                train_codes.add(tuple(actual["target_code"]))
            elif split == "validation":
                validation_codes.append(tuple(actual["target_code"]))
            count += 1
        counts[split] = count
        _require(count == manifest["examples"][split], f"Manifest sample count mismatch: {split}")
        if reference_sequences is not None:
            for actual, old in zip_longest(_rows(path), _rows(reference_sequences / path.name)):
                _require(actual is not None and old is not None
                         and all(actual[key] == old[key] for key in
                                 ("user_id", "split", "history_items", "target_item")),
                         f"New samples changed original item/user protocol: {split}")
    no_prefix = sum(len(row["train_items"]) < 2 for row in histories)
    _require(no_prefix == manifest["users_with_no_train_prefix"], "No-prefix user count mismatch")
    groups = Counter(tuple(code) for code in assignments.values())
    return {
        "passed": True,
        "method": "All rows independently matched to chronological raw history positions and "
                  "the pinned code map; sample-builder implementation is not used by this audit.",
        "examples": counts, "users": len(histories), "train_users": len(train_users),
        "users_with_no_train_prefix": no_prefix, "max_history": max_history,
        "semantic_ids_sha256": mapping_hash, "example_file_sha256": hashes,
        "histories_sha256": manifest["histories_sha256"],
        "same_item_user_protocol_as_control": reference_sequences is not None,
        "validation_target_code_train_coverage": {
            "covered": sum(code in train_codes for code in validation_codes),
            "denominator": len(validation_codes),
        },
        "catalog_item_count": len(assignments), "valid_catalog_codes": len(groups),
        "colliding_catalog_items": sum(size for size in groups.values() if size > 1),
        "largest_collision_group": max(groups.values()),
        "model_training_started": False,
        "test_label_use": "Offline chronological integrity checks only; no fitting, selection "
                          "or test-code-coverage diagnostic/tuning in this preparation step.",
    }
