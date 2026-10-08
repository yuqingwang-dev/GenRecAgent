import json
from types import SimpleNamespace

import pytest

from genrecagent.delivery.demo import grade_demo
from genrecagent.delivery.readiness import (
    CORE_FILES,
    DEFAULT_ADAPTER,
    QWEN_REVISION,
    check_artifacts,
    resolve_output,
    sha256,
)
from genrecagent.delivery.report import build_report, render_report


def _write(root, name, payload):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_missing_or_empty_artifacts_fail_and_fixed_does_not_need_qwen(tmp_path):
    assert not check_artifacts(tmp_path)["ready"]
    for name in CORE_FILES:
        _write(tmp_path, name, {})
    assert check_artifacts(tmp_path)["ready"]
    assert not check_artifacts(tmp_path, policy="prompt")["ready"]
    (tmp_path / CORE_FILES[0]).write_text("", encoding="utf-8")
    assert not check_artifacts(tmp_path)["ready"]


def test_sft_smoke_or_changed_weights_are_refused(tmp_path):
    adapter = tmp_path / DEFAULT_ADAPTER
    _write(tmp_path, str(adapter.relative_to(tmp_path) / "adapter_config.json"), {})
    weights = _write(tmp_path, str(adapter.relative_to(tmp_path) / "adapter_model.safetensors"), {})
    _write(
        tmp_path,
        str(adapter.parent.relative_to(tmp_path) / "run_config.json"),
        {"base_revision": QWEN_REVISION},
    )
    metrics_name = str(adapter.parent.relative_to(tmp_path) / "metrics.json")
    _write(
        tmp_path, metrics_name, {"status": "software_smoke_only", "adapter_sha256": sha256(weights)}
    )
    assert any("full SFT" in text for text in check_artifacts(tmp_path, policy="sft")["problems"])
    _write(tmp_path, metrics_name, {"status": "full_sft", "adapter_sha256": "wrong"})
    assert any(
        "hash mismatch" in text for text in check_artifacts(tmp_path, policy="sft")["problems"]
    )


def test_output_boundary_and_old_run_are_protected(tmp_path):
    assert resolve_output(tmp_path, tmp_path / "outputs/new") == tmp_path / "outputs/new"
    for path in (tmp_path / "elsewhere", tmp_path / "outputs", tmp_path / "outputs/../../other"):
        with pytest.raises(ValueError, match="subdirectory"):
            resolve_output(tmp_path, path)
    _write(tmp_path, "outputs/old/trace.json", {})
    with pytest.raises(ValueError, match="fresh"):
        resolve_output(tmp_path, tmp_path / "outputs/old")


def test_demo_checks_feasibility_empty_success_seen_and_duplicate_items():
    items = {
        "A": SimpleNamespace(price=1),
        "B": SimpleNamespace(price=2),
        "C": SimpleNamespace(price=None),
    }

    def grade(status, ids, price=3):
        return grade_demo(
            {"status": status, "items": [{"item_id": i} for i in ids]},
            items,
            history=["A"],
            top_k=1,
            max_price=price,
        )

    assert grade("success", ["B"])["contract_pass"]
    assert not grade("success", [])["contract_pass"]
    assert not grade("success", ["A"])["contract_pass"]
    assert not grade("success", ["B", "B"])["contract_pass"]
    assert not grade("clarification", [])["contract_pass"]
    assert grade("clarification", [], 0)["contract_pass"]
    assert not grade("success", [], 0)["contract_pass"]


def _report_fixture(root):
    score = {"Recall@20": 0.25, "NDCG@20": 0.1}
    m1 = {
        s: {"users": 2, "items": 3, "methods": {"dense": score, "popularity": score}}
        for s in ("validation", "test")
    }
    _write(root, "outputs/m1_all_beauty_full/metrics.json", m1)
    for name in ("m3_transition_test", "m3_transformer_test_full_local"):
        _write(root, f"outputs/{name}/metrics.json", {"test": {"users": 2, **score}})
    _write(
        root,
        "outputs/fast_dual_validation/metrics.json",
        {"users": 2, "items": 3, "methods": {"hybrid_75": score}},
    )
    base = "outputs/tool_sft_lab95_server"
    weights = _write(root, f"{DEFAULT_ADAPTER}/adapter_model.safetensors", {"adapter": "fixture"})
    digest = sha256(weights)
    training = {
        "status": "full_sft",
        "assistant_only_labels": True,
        "adapter_sha256": digest,
        "train_actions": 3,
        "epochs": 1,
        "initial_validation_nll": 1,
        "best_validation_nll": 0.2,
    }
    _write(root, f"{base}/tool_sft_lora/metrics.json", training)
    _write(
        root,
        f"{base}/tool_sft_lora/run_config.json",
        {"git_commit_at_start": "fixture", "elapsed_seconds": 10},
    )
    common = {
        "tasks": 1,
        "task_ids": ["fixture-1"],
        "test_file_sha256": "test-hash",
        "source_hashes": {"tools": "tools-hash"},
        "git_commit_at_start": "fixture",
        "base_model_revision": QWEN_REVISION,
        "recommendation_checkpoint_sha256": "m3-hash",
        "contract_pass": 1,
        "feasible_filled": 1,
        "feasible_tasks": 1,
        "correct_clarifications": 0,
        "policy_illegal_calls": 0,
        "generation_tokens": 0,
        "adapter_sha256": digest,
    }
    for name in ("fixed", "prompt", "sft", "no_recovery", "no_finish_verification"):
        _write(root, f"{base}/lab95_comparison/{name}/metrics.json", common)


def test_report_separates_splits_and_preserves_metric_provenance(tmp_path):
    _report_fixture(tmp_path)
    report = build_report(tmp_path)
    assert "hybrid_75" not in report["retrieval_test"]
    assert report["dual_retrieval_validation_only"]["hybrid_75"]["Recall@20"] == 0.25
    assert len(report["source_metric_sha256"]) == 11
    assert report["recovery_v3"] is None
    text = render_report(report)
    assert "test only" in text and "validation only" in text
    assert "1 synthetic held-out tasks" in text


def test_report_refuses_unpaired_tasks_or_missing_metrics(tmp_path):
    _report_fixture(tmp_path)
    path = tmp_path / "outputs/tool_sft_lab95_server/lab95_comparison/sft/metrics.json"
    row = json.loads(path.read_text(encoding="utf-8"))
    row["task_ids"] = ["other-task"]
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="Unpaired"):
        build_report(tmp_path)
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path / "missing")


def test_partial_recovery_artifacts_are_not_silently_presented_as_completed(tmp_path):
    _report_fixture(tmp_path)
    _write(tmp_path, "outputs/tool_sft_recovery_v3_server/tool_sft_recovery_v3/metrics.json", {})
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path)
