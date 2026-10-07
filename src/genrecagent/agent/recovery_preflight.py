"""Read-only boundaries for the proposed recovery run, not permission to launch it."""

from itertools import combinations
from pathlib import Path

from genrecagent.delivery.readiness import QWEN_REVISION

FROZEN_CONFIG = {
    "model_name": "Qwen/Qwen2.5-1.5B-Instruct",
    "revision": QWEN_REVISION,
    "data_dir": "data/processed/tool_sft_recovery_v3",
    "cache_dir": "artifacts/model_cache",
    "initial_adapter": "outputs/tool_sft_lab95_server/tool_sft_lora/adapter",
    "output_dir": "outputs/tool_sft_recovery_v3",
    "input_protocol": "compact_recovery_v3",
    "epochs": 2,
    "rank": 8,
    "lora_alpha": 16,
    "learning_rate": 0.0001,
    "gradient_accumulation": 8,
    "max_sequence_length": 1024,
    "seed": 42,
    "device": "cuda",
    "dtype": "bfloat16",
    "action_name_weight": 4.0,
    "selection": "validation_rollout",
    "validation_rollout_tasks": 16,
}
RUN_OUTPUTS = (
    "outputs/recovery_job",
    "outputs/tool_sft_recovery_v3",
    "outputs/recovery_v3_sft_validation",
    "outputs/recovery_v3_test",
    "outputs/recovery_v3_results.tar.gz",
)


def validate_plan(config: dict, manifest: dict, actual_train_actions: int) -> dict:
    for key, value in FROZEN_CONFIG.items():
        if config.get(key) != value or isinstance(config.get(key), bool):
            raise ValueError(f"Recovery run differs from the proposed frozen plan: {key}")
    if set(config["target_modules"]) != {"q_proj", "v_proj"}:
        raise ValueError("Recovery adapter must remain rank-8 q/v LoRA")
    for key, value in (
        ("base_revision", QWEN_REVISION),
        ("input_protocol", config["input_protocol"]),
        ("observation_snapshots_immutable", True),
        ("assistant_only_labels", True),
        ("action_name_weight", 4.0),
    ):
        if manifest.get(key) != value:
            raise ValueError(f"Recovery manifest differs from the frozen protocol: {key}")
    declared = manifest["splits"]["train"]["assistant_actions"]
    if not 0 < actual_train_actions <= 1200 or actual_train_actions != declared:
        raise ValueError("Actual train action count differs from manifest or exceeds 1,200 cap")
    for split in ("validation", "test"):
        if manifest["splits"][split]["tasks"] != 72:
            raise ValueError("Recovery held-out split must remain the declared 72 tasks")
    groups = [
        set(manifest["splits"][split]["user_groups"]) for split in ("train", "validation", "test")
    ]
    if any(a & b for a, b in combinations(groups, 2)):
        raise ValueError("Recovery manifest user groups overlap")
    return {
        "train_actions": actual_train_actions,
        "epochs": 2,
        "maximum_train_action_visits": 2 * actual_train_actions,
        "maximum_optimizer_steps": 2 * ((actual_train_actions + 7) // 8),
        "selection_tasks_per_epoch": 16,
        "permission": "readiness only; explicit user approval is still required",
    }


def require_fresh_outputs(root: Path) -> None:
    root = root.resolve()
    boundary = (root / "outputs").resolve()
    if not boundary.is_relative_to(root) or boundary == root:
        raise ValueError("Recovery outputs boundary resolves outside this project")
    for name in RUN_OUTPUTS:
        path = root / name
        if not path.resolve().is_relative_to(boundary) or path.exists() or path.is_symlink():
            raise ValueError(f"Recovery output must be new; preserve existing run: {name}")
