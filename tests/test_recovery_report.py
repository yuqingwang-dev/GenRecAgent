from copy import deepcopy

import numpy as np
import pytest
from safetensors.numpy import save_file

from genrecagent.agent.recovery_preflight import FROZEN_CONFIG
from genrecagent.delivery.recovery_report import adapter_update, selection_ids, validate_training


def _fixture():
    config = {**FROZEN_CONFIG, "target_modules": ["q_proj", "v_proj"]}
    manifest = {
        "base_revision": config["revision"],
        "input_protocol": config["input_protocol"],
        "assistant_only_labels": True,
        "observation_snapshots_immutable": True,
        "action_name_weight": 4.0,
        "splits": {
            split: {
                "assistant_actions": 1080 if split == "train" else 308,
                "tasks": 288 if split == "train" else 72,
                "user_groups": [split],
            }
            for split in ("train", "validation", "test")
        },
    }
    tasks = [
        {
            "task_id": f"validation:{i}",
            "split": "validation",
            "expected": "success" if i < 36 else "clarification",
        }
        for i in range(72)
    ]
    log = [
        {
            "epoch": epoch,
            "validation_nll": loss,
            "train_weighted_action_loss": 0.1,
            "validation_raw_rollout": {
                "tasks": 16,
                "task_ids": selection_ids(tasks),
                "feasible_filled": epoch,
                "contract_pass": 8,
                "policy_illegal_calls": 2,
                "controller_actions": 0,
            },
        }
        for epoch, loss in ((1, 0.2), (2, 0.3))
    ]
    run = {
        "config": config,
        "dataset_manifest": manifest,
        "device": "cuda",
        "base_revision": config["revision"],
        "smoke_test": False,
        "elapsed_seconds": 1,
    }
    training = {
        "status": "full_sft",
        "train_actions": 1080,
        "validation_actions": 308,
        "epochs": 2,
        "input_protocol": config["input_protocol"],
        "selection": "validation_rollout",
        "action_name_weight": 4.0,
        "assistant_only_labels": True,
        "action_logits_only": True,
        "initial_validation_nll": 0.7,
        "best_validation_nll": 0.2,
        "selected_epoch": 2,
    }
    return training, run, log, manifest, tasks


def test_selected_checkpoint_nll_is_not_confused_with_minimum_diagnostic_nll():
    report = validate_training(*_fixture())
    assert report["selected_epoch"] == 2
    assert report["selected_validation_nll"] == 0.3
    assert report["minimum_validation_nll"] == 0.2


@pytest.mark.parametrize("mutation", ["epoch", "mask", "test", "controller", "nll", "min"])
def test_training_audit_rejects_protocol_or_selection_corruption(mutation):
    training, run, log, manifest, tasks = deepcopy(_fixture())
    if mutation == "epoch":
        training["selected_epoch"] = 1
    elif mutation == "mask":
        training["assistant_only_labels"] = False
    elif mutation == "test":
        tasks[0]["split"] = "test"
    elif mutation == "controller":
        log[0]["validation_raw_rollout"]["controller_actions"] = 1
    elif mutation == "nll":
        log[0]["validation_nll"] = float("nan")
    else:
        training["best_validation_nll"] = 0.05
    with pytest.raises(ValueError):
        validate_training(training, run, log, manifest, tasks)


def test_adapter_requires_finite_actual_tensor_updates(tmp_path):
    before, after = tmp_path / "before.safetensors", tmp_path / "after.safetensors"
    name = "model.q_proj.lora_A.weight"
    value = np.zeros((2, 3), dtype=np.float32)
    save_file({name: value}, before)
    save_file({name: value}, after)
    with pytest.raises(ValueError, match="actual adapter"):
        adapter_update(before, after)
    save_file({name: value + 0.1}, after)
    report = adapter_update(before, after)
    assert report["changed_tensors"] == 1 and report["tensor_elements"] == 6
    save_file({name: value + float("nan")}, after)
    with pytest.raises(ValueError, match="nonfinite"):
        adapter_update(before, after)
