from copy import deepcopy

import pytest

from genrecagent.agent.recovery_preflight import (
    FROZEN_CONFIG,
    require_fresh_outputs,
    validate_plan,
)


def _fixture():
    config = {**FROZEN_CONFIG, "target_modules": ["q_proj", "v_proj"]}
    manifest = {
        "base_revision": config["revision"],
        "input_protocol": config["input_protocol"],
        "observation_snapshots_immutable": True,
        "assistant_only_labels": True,
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
    return config, manifest


def test_frozen_plan_uses_actual_action_cap_and_never_grants_permission():
    config, manifest = _fixture()
    report = validate_plan(config, manifest, 1080)
    assert report["maximum_train_action_visits"] == 2160
    assert report["maximum_optimizer_steps"] == 270
    assert "approval is still required" in report["permission"]


@pytest.mark.parametrize(
    "key,value",
    [
        ("epochs", 3),
        ("rank", 16),
        ("max_sequence_length", 2048),
        ("validation_rollout_tasks", 72),
        ("data_dir", "../RecAI"),
        ("selection", "test_rollout"),
        ("target_modules", ["q_proj", "k_proj", "v_proj"]),
    ],
)
def test_proposal_drift_is_rejected(key, value):
    config, manifest = _fixture()
    config[key] = value
    with pytest.raises(ValueError):
        validate_plan(config, manifest, 1080)


@pytest.mark.parametrize("actual", [0, 1081, 1201])
def test_manifest_cannot_hide_the_actual_train_size(actual):
    config, manifest = _fixture()
    with pytest.raises(ValueError, match="Actual train action count"):
        validate_plan(config, manifest, actual)


def test_split_leakage_and_changed_heldout_size_are_rejected():
    config, manifest = _fixture()
    changed = deepcopy(manifest)
    changed["splits"]["validation"]["user_groups"] = ["train"]
    with pytest.raises(ValueError, match="overlap"):
        validate_plan(config, changed, 1080)
    manifest["splits"]["test"]["tasks"] = 73
    with pytest.raises(ValueError, match="72 tasks"):
        validate_plan(config, manifest, 1080)


def test_existing_output_is_preserved_and_preflight_creates_no_files(tmp_path):
    require_fresh_outputs(tmp_path)
    assert not list(tmp_path.iterdir())
    log = tmp_path / "outputs/recovery_job/train.log"
    log.parent.mkdir(parents=True)
    log.write_text("original run", encoding="utf-8")
    with pytest.raises(ValueError, match="preserve existing run"):
        require_fresh_outputs(tmp_path)
    assert log.read_text(encoding="utf-8") == "original run"


def test_check_only_exits_before_gpu_import_or_training():
    from pathlib import Path

    script = (Path(__file__).parents[1] / "scripts/run_lab95_recovery.sh").read_text()
    assert script.index('[[ "$MODE" != "--check-only" ]] || exit 0') < script.index(
        "import json, torch"
    )
    assert script.index('[[ "$MODE" != "--check-only" ]] || exit 0') < script.index(
        "scripts/train_tool_sft.py"
    )
