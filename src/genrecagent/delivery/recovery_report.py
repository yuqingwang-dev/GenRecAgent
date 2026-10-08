"""Audit completed recovery training and raw paired outcomes without GPU inference."""

import hashlib
import json
from datetime import datetime
from math import isfinite
from pathlib import Path

from genrecagent.agent.recovery_preflight import validate_plan
from genrecagent.data.storage import load_items, read_jsonl
from genrecagent.delivery.readiness import check_artifacts, sha256
from genrecagent.delivery.source_evidence import verify_source
from genrecagent.evaluation.agent_evidence import audit_run, check_paired_runs

RECOVERY_BASE = Path("outputs/tool_sft_recovery_v3_server")


def selection_ids(tasks: list[dict]) -> list[str]:
    if len(tasks) != 72 or any(row["split"] != "validation" for row in tasks):
        raise ValueError("Recovery selection must use the complete frozen validation split")
    chosen = []
    for outcome in ("success", "clarification"):
        group = sorted(
            (row for row in tasks if row["expected"] == outcome),
            key=lambda row: hashlib.sha256(f"42:selection:{row['task_id']}".encode()).hexdigest(),
        )
        if len(group) < 8:
            raise ValueError("Recovery selection needs eight tasks of each outcome")
        chosen.extend(row["task_id"] for row in group[:8])
    return chosen


def validate_training(
    training: dict, run: dict, log: list[dict], manifest: dict, validation_tasks: list[dict]
) -> dict:
    validate_plan(run["config"], manifest, training["train_actions"])
    if (
        training["status"] != "full_sft"
        or run["smoke_test"]
        or run["dataset_manifest"] != manifest
        or run["device"] != "cuda"
        or run["base_revision"] != manifest["base_revision"]
    ):
        raise ValueError("Recovery is not the declared real CUDA training run")
    if (
        training["epochs"] != 2
        or len(log) != 2
        or [row["epoch"] for row in log] != [1, 2]
        or training["input_protocol"] != manifest["input_protocol"]
        or training["selection"] != "validation_rollout"
        or training["validation_actions"] != manifest["splits"]["validation"]["assistant_actions"]
        or training["action_name_weight"] != 4
        or not training["assistant_only_labels"]
        or not training["action_logits_only"]
    ):
        raise ValueError("Recovery training count, masks or selection protocol changed")
    expected_ids, scores = selection_ids(validation_tasks), []
    for row in log:
        rollout = row["validation_raw_rollout"]
        if (
            rollout["tasks"] != 16
            or rollout["task_ids"] != expected_ids
            or rollout["controller_actions"] != 0
            or not 0 <= rollout["feasible_filled"] <= 8
            or not rollout["feasible_filled"] <= rollout["contract_pass"] <= 16
            or not 0 <= rollout["policy_illegal_calls"] <= 160
        ):
            raise ValueError("Checkpoint selection used different tasks or a controller")
        for key in ("validation_nll", "train_weighted_action_loss"):
            if not isfinite(row[key]) or row[key] < 0:
                raise ValueError("Nonfinite or negative recovery training loss")
        scores.append(
            (
                rollout["feasible_filled"],
                rollout["contract_pass"],
                -rollout["policy_illegal_calls"],
                -row["validation_nll"],
            )
        )
    selected = max(range(2), key=lambda index: scores[index])
    if training["selected_epoch"] != selected + 1:
        raise ValueError("Saved checkpoint does not follow the predeclared raw selection score")
    if training["best_validation_nll"] != min(row["validation_nll"] for row in log):
        raise ValueError("Reported minimum NLL differs from the actual training log")
    if not isfinite(training["initial_validation_nll"]) or not isfinite(run["elapsed_seconds"]):
        raise ValueError("Nonfinite initial loss or elapsed time")
    return {
        "selected_epoch": selected + 1,
        "selected_validation_nll": log[selected]["validation_nll"],
        "minimum_validation_nll": training["best_validation_nll"],
        "selection_score": list(scores[selected]),
    }


def adapter_update(initial: Path, trained: Path) -> dict:
    import numpy as np
    from safetensors.numpy import load_file

    before, after = load_file(initial), load_file(trained)
    if set(before) != set(after) or not before:
        raise ValueError("Initial and trained LoRA tensor names differ")
    changed, elements, squared = 0, 0, 0.0
    for name, value in after.items():
        if (
            ".lora_A." not in name
            and ".lora_B." not in name
            or value.shape != before[name].shape
            or not np.isfinite(value).all()
            or not np.isfinite(before[name]).all()
        ):
            raise ValueError("Unexpected, reshaped or nonfinite LoRA tensor")
        changed += not np.array_equal(value, before[name])
        elements += value.size
        delta = value.astype(np.float64) - before[name].astype(np.float64)
        squared += float(np.square(delta).sum())
    if not changed or not isfinite(squared) or squared <= 0:
        raise ValueError("No finite, actual adapter parameter update")
    return {
        "tensors": len(after),
        "changed_tensors": changed,
        "tensor_elements": elements,
        "update_l2": squared**0.5,
        "all_finite": True,
    }


def build_recovery_report(root: Path, base: Path | None = None) -> dict:
    base = base or root / RECOVERY_BASE
    hashes = {}

    def read(path):
        hashes[str(path.relative_to(root))] = sha256(path)
        return json.loads(path.read_text(encoding="utf-8"))

    if (base / "recovery_job/exit_code.txt").read_text().strip() != "0":
        raise ValueError("Recovery job did not finish successfully")
    trained = base / "tool_sft_recovery_v3"
    training, run = read(trained / "metrics.json"), read(trained / "run_config.json")
    log = read(trained / "training_log.json")
    data = root / "data/processed/tool_sft_recovery_v3"
    manifest = read(data / "manifest.json")
    for split in ("train", "validation", "test"):
        actions, tasks = data / f"{split}_actions.jsonl", data / f"{split}_tasks.jsonl"
        if (
            sha256(actions) != manifest["splits"][split]["actions_sha256"]
            or sha256(tasks) != manifest["splits"][split]["task_sha256"]
        ):
            raise ValueError("Recovery input file changed after preparation")
    validation_tasks, test_tasks = (
        list(read_jsonl(data / "validation_tasks.jsonl")),
        list(read_jsonl(data / "test_tasks.jsonl")),
    )
    selection = validate_training(training, run, log, manifest, validation_tasks)
    if (
        sha256(root / "outputs/m3_transformer_full_server/transformer.pt")
        != manifest["semantic_checkpoint_sha256"]
    ):
        raise ValueError("Recommendation checkpoint differs from the frozen recovery input")
    for name, digest in run["source_hashes"].items():
        verify_source(root, name, digest, run["git_commit_at_start"])
    readiness = check_artifacts(root, policy="sft", adapter=trained / "adapter", verify_hashes=True)
    if not readiness["ready"]:
        raise ValueError(
            "Pinned base or returned adapter check failed: " + "; ".join(readiness["problems"])
        )
    initial = root / run["config"]["initial_adapter"] / "adapter_model.safetensors"
    if sha256(initial) != training["initial_adapter_sha256"]:
        raise ValueError("Recovery started from a different initial adapter")
    update = adapter_update(initial, trained / "adapter/adapter_model.safetensors")
    if update["tensor_elements"] != training["trainable_parameters"]:
        raise ValueError("Saved LoRA tensor elements differ from trainable parameter count")
    items, paired, results = load_items(root / "data/processed/All_Beauty_full_optimized"), [], {}
    locations = {
        "validation_compact_sft": base / "recovery_v3_sft_validation",
        **{
            policy: base / "recovery_v3_test" / policy
            for policy in ("fixed", "compact_prompt", "compact_sft")
        },
    }
    finished = datetime.fromisoformat(run["created_utc"])
    for label, directory in locations.items():
        metrics = read(directory / "metrics.json")
        for name, digest in metrics["source_hashes"].items():
            verify_source(root, name, digest, metrics["git_commit_at_start"])
        trace_file = directory / "traces.jsonl"
        hashes[str(trace_file.relative_to(root))] = sha256(trace_file)
        rows = list(read_jsonl(trace_file))
        tasks = validation_tasks if label.startswith("validation_") else test_tasks
        split = "validation" if label.startswith("validation_") else "test"
        policy = "compact_sft" if label.startswith("validation_") else label
        if (
            metrics["policy"] != policy
            or metrics["split"] != split
            or metrics["tasks"] != 72
            or metrics["ablation"] != "none"
            or metrics["guard_intervention"] is not None
            or metrics["task_file_sha256"] != sha256(data / f"{split}_tasks.jsonl")
            or metrics["recommendation_checkpoint_sha256"] != manifest["semantic_checkpoint_sha256"]
            or metrics["base_model_revision"] != manifest["base_revision"]
            or datetime.fromisoformat(metrics["created_utc"]) < finished
        ):
            raise ValueError("Recovery evaluation is not the frozen raw paired protocol")
        if policy != "fixed" and metrics["input_protocol"] != manifest["input_protocol"]:
            raise ValueError("Prompt and SFT received different input protocols")
        if policy == "compact_sft" and metrics["adapter_sha256"] != training["adapter_sha256"]:
            raise ValueError("Recovery evaluation used a different selected adapter")
        results[label] = {
            "metrics": metrics,
            "independent_audit": audit_run(metrics, rows, tasks, items),
        }
        if split == "test":
            paired.append((metrics, rows))
    check_paired_runs(paired)
    if any(metrics["source_hashes"] != paired[0][0]["source_hashes"] for metrics, _ in paired):
        raise ValueError("Paired test policies use different evaluation source fingerprints")
    return {
        "status": "verified_real_recovery_run",
        "training": training,
        "selection": selection,
        "adapter_update": update,
        "training_seconds": run["elapsed_seconds"],
        "training_commit": run["git_commit_at_start"],
        "raw_results": results,
        "source_sha256": hashes,
        "controller_used": False,
        "scope": "synthetic normalized tasks, fresh held-out users/templates/prices; "
        "not broad natural-language planning or recommendation relevance",
    }
