"""Verify returned learned-tokenizer/generator evidence without loading Torch."""

import json
import math
from collections import Counter
from itertools import zip_longest
from pathlib import Path

from genrecagent.delivery.readiness import sha256
from genrecagent.delivery.source_evidence import verify_source

M2_BASE = Path("outputs/m2_rqvae_stable_full_server")
M3_BASE = Path("outputs/m3_transformer_rqvae_stable_full")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _rows(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _gate(stats: dict, settings: dict) -> bool:
    return (
        all(row["utilization"] >= settings["min_utilization"]
            and row["perplexity"] >= settings["min_perplexity_fraction"] * row["codebook_size"]
            for row in stats["codebook_usage_by_level"])
        and stats["unique_ratio"] >= settings["min_unique_ratio"]
        and stats["largest_group_fraction"] <= settings["max_group_fraction"]
        and math.isfinite(stats["reconstruction_mse"])
        and stats["reconstruction_mse"] <= settings["max_reconstruction_mse"]
    )


def build_learned_report(root: Path, *, catalog_items: int, users: int) -> dict | None:
    """Missing experiments stay absent; partial or changed experiments fail closed.

    Tensor/CUDA checks are preserved independent audits tied to exact weight bytes.
    This CPU report additionally recomputes catalog code statistics and all retrieval
    ranks/metrics from saved outcomes; it never trains or performs model inference.
    """
    m2, m3 = root / M2_BASE, root / M3_BASE
    if not m2.exists() and not m3.exists():
        return None
    fingerprints = {}

    def fingerprint(path):
        _require(path.resolve().is_relative_to(root.resolve()), "Evidence escapes project root")
        digest = sha256(path)
        fingerprints[path.relative_to(root).as_posix()] = digest
        return digest

    def read(path):
        fingerprint(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def audited_files(directory, hashes):
        _require(bool(hashes), "Empty artifact fingerprint ledger")
        for name, digest in hashes.items():
            _require(Path(name).name == name, "Unsafe artifact filename")
            _require(fingerprint(directory / name) == digest, "Learned artifact hash mismatch")

    trained = m2 / "m2_rqvae_stable_full"
    metrics, run = read(trained / "metrics.json"), read(trained / "run_config.json")
    audit = read(m2 / "m2_full_verification.json")
    _require(audit["passed"] and audit["checks"] and all(audit["checks"].values()),
             "RQ-VAE independent audit failed")
    audited_files(trained, audit["artifact_sha256"])
    _require(metrics["status"] == "full_catalog_training" and metrics["variant"] == "pca_ema"
             and run["subset_sampling"] == "full" and not run["git_dirty"],
             "Full clean stabilized RQ-VAE evidence required")
    for name, digest in run["source_hashes"].items():
        verify_source(root, name, digest, run["git_commit"])
    source = root / "data/processed/All_Beauty/index"
    _require(fingerprint(source / "item_embeddings.npy") == run["item_embedding_sha256"]
             and fingerprint(source / "item_ids.json") == run["item_ids_sha256"],
             "RQ-VAE source index fingerprint mismatch")
    mapping_path = trained / "item_semantic_ids.json"
    book = read(mapping_path)
    assignments = book["assignments"]
    item_ids = json.loads((source / "item_ids.json").read_text(encoding="utf-8"))
    _require(len(assignments) == len(item_ids) == catalog_items == audit["item_count"]
             == run["selected_item_count"] and set(assignments) == set(item_ids),
             "RQ-VAE does not cover the complete baseline catalog")
    _require(audit["mapping_mismatches"] == 0 and book["metadata"]["config"] == run["model"],
             "RQ-VAE mapping/checkpoint config mismatch")
    levels, size = run["model"]["num_levels"], run["model"]["codebook_size"]
    counters = [Counter() for _ in range(levels)]
    groups = Counter()
    for code in assignments.values():
        _require(len(code) == levels, "Invalid code length")
        for level, token in enumerate(code):
            prefix = f"<s{level}:"
            _require(token.startswith(prefix) and token.endswith(">"), "Invalid learned token")
            number = int(token[len(prefix):-1])
            _require(0 <= number < size, "Learned token exceeds codebook")
            counters[level][number] += 1
        groups[tuple(code)] += 1
    usage = []
    for counts in counters:
        probabilities = [count / catalog_items for count in counts.values()]
        usage.append({"used_codes": len(counts), "codebook_size": size,
                      "utilization": len(counts) / size,
                      "perplexity": math.exp(-sum(p * math.log(p) for p in probabilities))})
    collisions = {"item_count": catalog_items, "unique_code_count": len(groups),
                  "collision_group_count": sum(n > 1 for n in groups.values()),
                  "colliding_item_count": sum(n for n in groups.values() if n > 1),
                  "largest_collision_group": max(groups.values())}
    _require(collisions == metrics["collisions"], "RQ-VAE collision statistics mismatch")
    for actual, saved in zip(usage, metrics["codebook_usage_by_level"], strict=True):
        _require(actual["used_codes"] == saved["used_codes"]
                 and abs(actual["perplexity"] - saved["perplexity"]) < 1e-7,
                 "RQ-VAE code usage statistics mismatch")
    stats = {"codebook_usage_by_level": usage, "unique_ratio": len(groups) / catalog_items,
             "largest_group_fraction": max(groups.values()) / catalog_items,
             "reconstruction_mse": metrics["catalog_reconstruction_mse"]}
    _require(metrics["catalog_quality_gate"]["passed"] is True
             and metrics["selected_validation_quality"]["gate"]["passed"] is True
             and _gate(stats, run["training"])
             and _gate(metrics["selected_validation_quality"]["statistics"], run["training"]),
             "Frozen RQ-VAE quality gates did not pass")
    result = {
        "scope": "checksum-verified frozen learned experiments; CPU evidence checks, no inference",
        "tokenizer": {"variant": "pca_ema", "items": catalog_items,
                      "best_epoch": metrics["best_epoch"],
                      "validation_reconstruction_mse":
                          metrics["best_validation_reconstruction_mse"],
                      "catalog_reconstruction_mse": metrics["catalog_reconstruction_mse"],
                      "codebook_usage_by_level": usage, "collisions": collisions,
                      "both_quality_gates_passed": True,
                      "independent_checks": len(audit["checks"]),
                      "checkpoint_sha256":
                          fingerprints[(trained / "rqvae.pt").relative_to(root).as_posix()],
                      "mapping_sha256": fingerprints[mapping_path.relative_to(root).as_posix()]},
        "generative": None,
        "source_sha256": fingerprints,
    }
    if not m3.exists():
        return result
    training, config = read(m3 / "metrics.json"), read(m3 / "run_config.json")
    train_audit = read(root / "outputs/m3_training_verification.json")
    grade_audit = read(root / "outputs/m3_retrieval_verification.json")
    execution = read(root / "outputs/m3_evaluation_execution.json")
    _require(train_audit["passed"] and train_audit["checks"]
             and all(train_audit["checks"].values())
             and grade_audit["passed"] and execution["passed"],
             "M3 independent verification failed")
    audited_files(m3, train_audit["artifact_sha256"])
    checkpoint_hash = fingerprint(m3 / "transformer.pt")
    _require(checkpoint_hash == train_audit["checkpoint_sha256"]
             == grade_audit["checkpoint_sha256"] == execution["checkpoint_sha256"],
             "M3 audited checkpoint mismatch")
    _require(training["status"] == "full_train_prefixes" and not config["git_dirty"]
             and config["semantic_ids_sha256"] == fingerprint(mapping_path),
             "M3 full training or tokenizer pairing mismatch")
    for name, digest in config["source_hashes"].items():
        verify_source(root, name, digest, config["git_commit"])
    for name, digest in execution["source_sha256"].items():
        verify_source(root, name, digest, execution["git_commit"])
    sequence = root / "data/processed/All_Beauty_m3_rqvae_stable"
    manifest = read(sequence / "manifest.json")
    _require(manifest["semantic_ids_sha256"] == config["semantic_ids_sha256"]
             and manifest["codebook_item_count"] == catalog_items,
             "M3 sample manifest differs from the tokenizer")
    _require(fingerprint(sequence / "train.jsonl") == config["train_examples_sha256"]
             and fingerprint(sequence / "validation.jsonl") == config["validation_examples_sha256"],
             "M3 train/validation examples changed")
    history = root / "data/processed/All_Beauty_full_optimized/histories.jsonl"
    _require(fingerprint(history) == manifest["histories_sha256"],
             "M3 history fingerprint mismatch")
    evaluation = root / "outputs/m3_transformer_rqvae_stable_evaluation"
    scored, eval_run = read(evaluation / "metrics.json"), read(evaluation / "run_config.json")
    audited_files(evaluation, grade_audit["evaluation_artifact_sha256"])
    _require(eval_run["status"] == "full_evaluation" and not eval_run["git_dirty"]
             and eval_run["checkpoint_sha256"] == checkpoint_hash
             and eval_run["dataset_manifest"] == manifest,
             "M3 evaluation is not the frozen full experiment")
    log = read(m3 / "training_log.json")
    _require([row["epoch"] for row in log] == list(range(1, config["training"]["epochs"] + 1)),
             "M3 training log is incomplete")
    selected = min(log, key=lambda row: row["validation_target_nll"])
    _require(selected["epoch"] == training["best_epoch"]
             and selected["validation_target_nll"] == training["best_validation_target_nll"],
             "M3 checkpoint was not selected on minimum validation NLL")
    for split in ("validation", "test"):
        _require(fingerprint(sequence / f"{split}.jsonl") == manifest["example_file_sha256"][split],
                 "M3 held-out sample hash mismatch")
        traces = root / "outputs/m3_retrieval_traces" / f"{split}.jsonl"
        _require(fingerprint(traces) == grade_audit["trace_sha256"][split]
                 == execution["trace_sha256"][split], "M3 candidate trace hash mismatch")
        hits, dcg, diagnostics, seen_users = Counter(), Counter(), Counter(), set()
        count = 0
        for trace, example in zip_longest(_rows(traces), _rows(sequence / f"{split}.jsonl")):
            _require(trace is not None and example is not None, "M3 missing user trace")
            _require(all(trace[key] == example[key] for key in
                         ("user_id", "split", "history_items", "target_item", "target_code")),
                     "M3 trace/sample alignment mismatch")
            _require(trace["user_id"] not in seen_users, "Duplicate M3 evaluation user")
            seen_users.add(trace["user_id"])
            ids = [hit["item_id"] for hit in trace["hits"]]
            _require(len(ids) == len(set(ids)) and not set(ids).intersection(trace["history_items"])
                     and set(ids).issubset(assignments), "Invalid/seen/duplicate M3 candidate")
            rank = ids.index(example["target_item"]) + 1 if example["target_item"] in ids else None
            for k in (5, 10, 20):
                if rank is not None and rank <= k:
                    hits[k] += 1
                    dcg[k] += 1 / math.log2(rank + 1)
            diagnostics.update(trace["diagnostics"])
            count += 1
        _require(count == users == scored[split]["users"] == manifest["examples"][split],
                 "M3 evaluation denominator differs from baselines")
        for k in (5, 10, 20):
            _require(abs(hits[k] / count - scored[split][f"Recall@{k}"]) < 1e-12
                     and abs(dcg[k] / count - scored[split][f"NDCG@{k}"]) < 1e-12,
                     "M3 Recall/NDCG differs from actual candidate ranks")
        _require(dict(diagnostics) == scored[split]["diagnostic_totals"], "M3 diagnostics mismatch")
    result["generative"] = {
        "status": "verified_full_train_and_evaluation", "best_epoch": training["best_epoch"],
        "train_examples": training["train_examples"], "parameters": training["parameters"],
        "best_validation_target_nll": training["best_validation_target_nll"],
        "validation": scored["validation"], "test": scored["test"],
        "checkpoint_sha256": checkpoint_hash, "mapping_sha256": config["semantic_ids_sha256"],
        "all_user_executions_verified": users * 2, "default_runtime_switched": False,
    }
    return result
