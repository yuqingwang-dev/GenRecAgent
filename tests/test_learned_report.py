"""Small synthetic evidence fixtures test validators, not real model performance."""

import json
import math
from collections import Counter

import pytest

from genrecagent.delivery.learned_report import M2_BASE, M3_BASE, build_learned_report
from genrecagent.delivery.readiness import sha256


def write(root, name, value):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def lines(root, name, values):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
    return path


def audit_hashes(root, directory):
    return {path.name: sha256(path) for path in (root / directory).iterdir() if path.is_file()}


def fixture(root, *, with_m3=False):
    source = write(root, "source.py", {"fixture": "no model training"})
    source_hashes = {"source.py": sha256(source)}
    codes = {"A": ["<s0:0>", "<s1:0>", "<s2:0>"],
             "B": ["<s0:1>", "<s1:1>", "<s2:1>"],
             "C": ["<s0:0>", "<s1:0>", "<s2:1>"]}
    model_config = {"variant": "pca_ema", "num_levels": 3, "codebook_size": 2}
    base = M2_BASE / "m2_rqvae_stable_full"
    mapping = write(root, base / "item_semantic_ids.json",
                    {"metadata": {"config": model_config}, "assignments": codes})
    vector = write(root, "data/processed/All_Beauty/index/item_embeddings.npy", {"fixture": True})
    ids = write(root, "data/processed/All_Beauty/index/item_ids.json", list(codes))
    usage = []
    for level in range(3):
        counts = Counter(code[level] for code in codes.values())
        probabilities = [count / 3 for count in counts.values()]
        usage.append({"used_codes": 2, "codebook_size": 2, "utilization": 1.,
                      "perplexity": math.exp(-sum(p * math.log(p) for p in probabilities))})
    statistics = {"codebook_usage_by_level": usage, "unique_ratio": 1.,
                  "largest_group_fraction": 1/3, "reconstruction_mse": .1}
    collisions = {"item_count": 3, "unique_code_count": 3, "collision_group_count": 0,
                  "colliding_item_count": 0, "largest_collision_group": 1}
    write(root, base / "rqvae.pt", {"fixture_only": True})
    write(root, base / "metrics.json", {
        "status": "full_catalog_training", "variant": "pca_ema", "best_epoch": 1,
        "catalog_reconstruction_mse": .1, "best_validation_reconstruction_mse": .1,
        "codebook_usage_by_level": usage, "collisions": collisions,
        "catalog_quality_gate": {"passed": True},
        "selected_validation_quality": {"gate": {"passed": True}, "statistics": statistics},
    })
    write(root, base / "run_config.json", {
        "subset_sampling": "full", "git_dirty": False, "git_commit": "fixture",
        "source_hashes": source_hashes, "item_embedding_sha256": sha256(vector),
        "item_ids_sha256": sha256(ids), "selected_item_count": 3, "model": model_config,
        "training": {"min_utilization": .5, "min_perplexity_fraction": .25,
                     "min_unique_ratio": .5, "max_group_fraction": .5,
                     "max_reconstruction_mse": .5},
    })
    write(root, M2_BASE / "m2_full_verification.json", {
        "passed": True, "checks": {"fixture_check": True}, "item_count": 3,
        "mapping_mismatches": 0, "artifact_sha256": audit_hashes(root, base),
    })
    if not with_m3:
        return
    sequences = "data/processed/All_Beauty_m3_rqvae_stable"
    examples = {
        split: [{"user_id": user, "split": split, "history_items": ["A"],
                 "target_item": "B", "target_code": codes["B"]} for user in ("u1", "u2")]
        for split in ("train", "validation", "test")
    }
    paths = {split: lines(root, f"{sequences}/{split}.jsonl", values)
             for split, values in examples.items()}
    history = lines(root, "data/processed/All_Beauty_full_optimized/histories.jsonl", [])
    manifest = {"semantic_ids_sha256": sha256(mapping), "codebook_item_count": 3,
                "histories_sha256": sha256(history),
                "examples": {split: 2 for split in examples},
                "example_file_sha256": {split: sha256(path) for split, path in paths.items()}}
    write(root, f"{sequences}/manifest.json", manifest)
    write(root, M3_BASE / "transformer.pt", {"fixture_only": True})
    checkpoint_hash = sha256(root / M3_BASE / "transformer.pt")
    write(root, M3_BASE / "metrics.json", {
        "status": "full_train_prefixes", "best_epoch": 1, "train_examples": 2,
        "parameters": 10, "best_validation_target_nll": 1.,
    })
    write(root, M3_BASE / "run_config.json", {
        "git_dirty": False, "git_commit": "fixture", "source_hashes": source_hashes,
        "semantic_ids_sha256": sha256(mapping), "train_examples_sha256": sha256(paths["train"]),
        "validation_examples_sha256": sha256(paths["validation"]), "training": {"epochs": 2},
    })
    write(root, M3_BASE / "training_log.json", [
        {"epoch": 1, "validation_target_nll": 1.}, {"epoch": 2, "validation_target_nll": 2.}])
    write(root, "outputs/m3_training_verification.json", {
        "passed": True, "checks": {"fixture_check": True}, "checkpoint_sha256": checkpoint_hash,
        "artifact_sha256": audit_hashes(root, M3_BASE),
    })
    trace_hashes, metrics = {}, {}
    for split in ("validation", "test"):
        traces = []
        for example in examples[split]:
            traces.append({**example, "hits": [{"item_id": "B"}],
                           "diagnostics": {"predicted_codes": 1}})
        path = lines(root, f"outputs/m3_retrieval_traces/{split}.jsonl", traces)
        trace_hashes[split] = sha256(path)
        metrics[split] = {"users": 2, "diagnostic_totals": {"predicted_codes": 2},
                          **{f"{metric}@{k}": 1. for metric in ("Recall", "NDCG")
                             for k in (5, 10, 20)}}
    evaluation = "outputs/m3_transformer_rqvae_stable_evaluation"
    write(root, f"{evaluation}/metrics.json", metrics)
    write(root, f"{evaluation}/run_config.json", {
        "status": "full_evaluation", "git_dirty": False, "checkpoint_sha256": checkpoint_hash,
        "dataset_manifest": manifest})
    write(root, "outputs/m3_retrieval_verification.json", {
        "passed": True, "checkpoint_sha256": checkpoint_hash, "trace_sha256": trace_hashes,
        "evaluation_artifact_sha256": audit_hashes(root, evaluation)})
    write(root, "outputs/m3_evaluation_execution.json", {
        "passed": True, "checkpoint_sha256": checkpoint_hash, "source_sha256": source_hashes,
        "git_commit": "fixture", "trace_sha256": trace_hashes})


def test_absent_and_partial_learned_outputs_are_distinguished(tmp_path):
    assert build_learned_report(tmp_path, catalog_items=3, users=2) is None
    (tmp_path / M3_BASE).mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        build_learned_report(tmp_path, catalog_items=3, users=2)


def test_valid_tokenizer_and_separate_generator_evidence(tmp_path):
    fixture(tmp_path, with_m3=True)
    report = build_learned_report(tmp_path, catalog_items=3, users=2)
    assert report["tokenizer"]["both_quality_gates_passed"]
    assert report["generative"]["test"]["Recall@20"] == 1.
    assert report["generative"]["all_user_executions_verified"] == 4
    assert not report["generative"]["default_runtime_switched"]


@pytest.mark.parametrize("mutation", ["weights", "quality", "collisions", "catalog"])
def test_tokenizer_corruption_is_not_reported_as_success(tmp_path, mutation):
    fixture(tmp_path)
    base = M2_BASE / "m2_rqvae_stable_full"
    if mutation == "weights":
        write(tmp_path, base / "rqvae.pt", {"changed": True})
    else:
        path = tmp_path / base / "metrics.json"
        metrics = json.loads(path.read_text())
        if mutation == "quality":
            metrics["catalog_quality_gate"]["passed"] = False
        elif mutation == "collisions":
            metrics["collisions"]["unique_code_count"] = 2
        else:
            metrics["codebook_usage_by_level"][0]["used_codes"] = 1
        write(tmp_path, base / "metrics.json", metrics)
        audit = json.loads((tmp_path / M2_BASE / "m2_full_verification.json").read_text())
        audit["artifact_sha256"] = audit_hashes(tmp_path, base)
        write(tmp_path, M2_BASE / "m2_full_verification.json", audit)
    with pytest.raises(ValueError):
        build_learned_report(tmp_path, catalog_items=3, users=2)


def test_recall_recomputed_even_if_metric_ledger_is_updated(tmp_path):
    fixture(tmp_path, with_m3=True)
    evaluation = "outputs/m3_transformer_rqvae_stable_evaluation"
    metrics = json.loads((tmp_path / evaluation / "metrics.json").read_text())
    metrics["test"]["Recall@20"] = .5
    write(tmp_path, f"{evaluation}/metrics.json", metrics)
    audit = json.loads((tmp_path / "outputs/m3_retrieval_verification.json").read_text())
    audit["evaluation_artifact_sha256"] = audit_hashes(tmp_path, evaluation)
    write(tmp_path, "outputs/m3_retrieval_verification.json", audit)
    with pytest.raises(ValueError, match="candidate ranks"):
        build_learned_report(tmp_path, catalog_items=3, users=2)


def test_new_checkpoint_cannot_be_paired_with_other_tokenizer(tmp_path):
    fixture(tmp_path, with_m3=True)
    config = json.loads((tmp_path / M3_BASE / "run_config.json").read_text())
    config["semantic_ids_sha256"] = "0" * 64
    write(tmp_path, M3_BASE / "run_config.json", config)
    audit = json.loads((tmp_path / "outputs/m3_training_verification.json").read_text())
    audit["artifact_sha256"] = audit_hashes(tmp_path, M3_BASE)
    write(tmp_path, "outputs/m3_training_verification.json", audit)
    with pytest.raises(ValueError, match="pairing"):
        build_learned_report(tmp_path, catalog_items=3, users=2)
