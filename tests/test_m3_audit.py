import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from genrecagent.data.m3_audit import (
    audit_m3_sequences,
    check_source_pins,
    resolve_project_output,
    sha256,
)
from genrecagent.data.models import UserHistory
from genrecagent.data.sequences import build_sequence_examples
from genrecagent.semantic_id.codebook import SemanticCodebook

ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_rows(path: Path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _fixture(tmp_path):
    prepared, sequences, control = (tmp_path / name for name in ("prepared", "sequences", "old"))
    for path in (prepared, sequences, control):
        path.mkdir()
    user = UserHistory("u", ("A", "B", "C"), "D", "E", (1, 2, 3), 4, 5)
    _write_rows(prepared / "histories.jsonl", [user.to_dict()])
    _write_json(prepared / "manifest.json", {"dataset": "fixture", "category": "test",
                                              "revision": "rev"})
    book = SemanticCodebook(assignments={
        item: (f"<s0:{max(0, index-1)}>", "<s1:0>", "<s2:0>")
        for index, item in enumerate("ABCDE")
    }, metadata={"source_dataset": "fixture", "category": "test", "dataset_revision": "rev",
                 "model_type": "trained_rqvae", "variant": "pca_ema",
                 "catalog_quality_gate": {"passed": True}})
    mapping = tmp_path / "mapping.json"
    book.save(mapping)
    examples = build_sequence_examples([user], book, max_history=2)
    for split, rows in examples.items():
        actual = [row.to_dict() for row in rows]
        _write_rows(sequences / f"{split}.jsonl", actual)
        # Reference codes may differ: only the original item/user protocol is fixed.
        old = [{**row, "target_code": ["old", "code", "tokens"]} for row in actual]
        _write_rows(control / f"{split}.jsonl", old)
    _write_json(sequences / "manifest.json", {
        "dataset": "fixture", "category": "test", "dataset_revision": "rev",
        "histories_sha256": sha256(prepared / "histories.jsonl"),
        "semantic_ids_sha256": sha256(mapping), "codebook_item_count": 5,
        "users": 1, "users_with_no_train_prefix": 0, "max_history": 2,
        "examples": {split: len(rows) for split, rows in examples.items()},
        "example_file_sha256": {
            split: sha256(sequences / f"{split}.jsonl") for split in examples
        },
    })
    return prepared, sequences, mapping, control


def test_independent_all_row_audit_and_unchanged_control_protocol(tmp_path):
    prepared, sequences, mapping, control = _fixture(tmp_path)
    result = audit_m3_sequences(prepared, sequences, mapping, reference_sequences=control)
    assert result["passed"] and not result["model_training_started"]
    assert result["examples"] == {"train": 2, "validation": 1, "test": 1}
    assert result["same_item_user_protocol_as_control"]
    assert result["train_users"] == 1
    assert result["valid_catalog_codes"] == 4
    assert result["colliding_catalog_items"] == 2


@pytest.mark.parametrize("change", ["future_context", "wrong_code", "missing_row", "wrong_split"])
def test_audit_rejects_tampering_even_with_updated_file_digest(tmp_path, change):
    prepared, sequences, mapping, _ = _fixture(tmp_path)
    path = sequences / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if change == "future_context":
        rows[0]["history_items"] = ["D"]
    elif change == "wrong_code":
        rows[0]["target_code"] = ["wrong", "code", "tokens"]
    elif change == "missing_row":
        rows.pop()
    else:
        rows[0]["split"] = "test"
    _write_rows(path, rows)
    manifest = json.loads((sequences / "manifest.json").read_text())
    manifest["example_file_sha256"]["train"] = sha256(path)
    _write_json(sequences / "manifest.json", manifest)
    with pytest.raises(ValueError, match="boundary|correspondence|count"):
        audit_m3_sequences(prepared, sequences, mapping)


def test_source_pins_reject_old_mapping_or_failed_quality(tmp_path):
    prepared, _, mapping, _ = _fixture(tmp_path)
    meta = json.loads(mapping.read_text())["metadata"]
    source = {"expected_semantic_ids_sha256": sha256(mapping),
              "expected_histories_sha256": sha256(prepared / "histories.jsonl"),
              "expected_semantic_id_model_type": "trained_rqvae",
              "expected_semantic_id_variant": "pca_ema", "require_catalog_quality_gate": True}
    check_source_pins(source, prepared / "histories.jsonl", mapping, meta)
    with pytest.raises(ValueError, match="fingerprint"):
        check_source_pins({**source, "expected_semantic_ids_sha256": "0" * 64},
                          prepared / "histories.jsonl", mapping, meta)
    with pytest.raises(ValueError, match="quality gate"):
        check_source_pins(source, prepared / "histories.jsonl", mapping,
                          {**meta, "catalog_quality_gate": {"passed": False}})


def test_output_boundary_and_no_overwrite(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    assert resolve_project_output(root, Path("data/processed/new")) == root / "data/processed/new"
    for path in (tmp_path / "outside", Path("outputs"), Path("data/processed"), Path("src/new")):
        with pytest.raises(ValueError, match="Output must"):
            resolve_project_output(root, path)
    populated = root / "outputs/existing"
    populated.mkdir(parents=True)
    (populated / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        resolve_project_output(root, populated)
    assert (populated / "keep.txt").read_text() == "keep"


def test_unapproved_training_rejected_before_inputs_or_torch_are_loaded(tmp_path):
    config = tmp_path / "approval.yaml"
    config.write_text(yaml.safe_dump({"transformer": {"approval_required": True}}))
    # A missing source section would crash if launch proceeded past the approval gate.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_semantic_transformer.py"),
         "--config", str(config)], cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert "requires separate training approval" in result.stderr
    assert "Traceback" not in result.stderr


def test_training_help_is_available_without_native_model_loading():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/train_semantic_transformer.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--approved" in result.stdout
