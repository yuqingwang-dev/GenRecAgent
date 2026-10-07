"""Read-only preflight: frozen proposal, input hashes/counts, adapter and fresh outputs."""

import json
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.recovery_preflight import require_fresh_outputs, validate_plan
from genrecagent.data.storage import read_jsonl
from genrecagent.delivery.readiness import check_artifacts, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    config_path = ROOT / "configs/tool_sft_recovery_v3.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    # Validate before using any configuration path; no arbitrary project or model selection.
    manifest_path = ROOT / "data/processed/tool_sft_recovery_v3/manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data = manifest_path.parent
    report = validate_plan(
        config, manifest, sum(1 for _ in read_jsonl(data / "train_actions.jsonl"))
    )
    require_fresh_outputs(ROOT)
    for split in ("train", "validation"):
        actions = data / f"{split}_actions.jsonl"
        if sha256(actions) != manifest["splits"][split]["actions_sha256"]:
            raise ValueError(f"Recovery {split} action-file hash changed")
        if sum(1 for _ in read_jsonl(actions)) != manifest["splits"][split]["assistant_actions"]:
            raise ValueError(f"Recovery {split} action-file count changed")
    validation = data / "validation_tasks.jsonl"
    if sha256(validation) != manifest["splits"]["validation"]["task_sha256"]:
        raise ValueError("Recovery validation task hash changed")
    if sum(1 for _ in read_jsonl(validation)) != 72:
        raise ValueError("Recovery validation task count changed")
    # Test task/action content is not opened before training/checkpoint selection.
    adapter = ROOT / config["initial_adapter"]
    readiness = check_artifacts(ROOT, policy="sft", adapter=adapter)
    if not readiness["ready"]:
        raise ValueError("Recovery artifacts not ready: " + "; ".join(readiness["problems"]))
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if (
        adapter_config["r"] != 8
        or adapter_config["lora_alpha"] != 16
        or set(adapter_config["target_modules"]) != {"q_proj", "v_proj"}
    ):
        raise ValueError("Existing adapter architecture differs from the frozen plan")
    report.update(
        status="ready_without_training",
        config_sha256=sha256(config_path),
        manifest_sha256=sha256(manifest_path),
        initial_adapter_sha256=sha256(adapter / "adapter_model.safetensors"),
        test_content_opened=False,
        gpu_checked=False,
        output_files_created=False,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
