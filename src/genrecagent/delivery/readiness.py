"""Lightweight artifact checks; never download models or run training."""

import hashlib
import json
from pathlib import Path

QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
DEFAULT_ADAPTER = "outputs/tool_sft_lab95_server/tool_sft_lora/adapter"
CORE_FILES = (
    "data/processed/All_Beauty_full_optimized/items.jsonl",
    "data/processed/All_Beauty_full_optimized/histories.jsonl",
    "data/processed/All_Beauty_full_optimized/manifest.json",
    "data/processed/All_Beauty/index/item_embeddings.npy",
    "data/processed/All_Beauty/index/item_ids.json",
    "data/processed/All_Beauty/index/metadata.json",
    "data/processed/All_Beauty_m3/train.jsonl",
    "data/processed/All_Beauty_m3/manifest.json",
    "outputs/m2_rqkmeans_full/item_semantic_ids.json",
    "outputs/m3_transformer_full_server/transformer.pt",
    "outputs/m3_transformer_full_server/run_config.json",
    "outputs/m3_transformer_full_server/metrics.json",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def check_artifacts(
    root: Path, *, policy: str = "fixed", adapter: Path | None = None, verify_hashes: bool = False
) -> dict:
    policy = policy.removeprefix("guarded_")
    if policy not in {"fixed", "prompt", "sft"}:
        raise ValueError("Unknown policy")
    problems, checked = [], []

    def require(path: Path) -> bool:
        checked.append(str(path))
        if not path.is_file() or path.stat().st_size == 0:
            problems.append(f"Missing or empty file: {path}")
            return False
        return True

    for name in CORE_FILES:
        require(root / name)
    if policy != "fixed":
        snapshot = (
            root
            / "artifacts/model_cache/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots"
            / QWEN_REVISION
        )
        for name in (
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
        ):
            require(snapshot / name)
        if verify_hashes:
            path = root / "outputs/m4_qwen_download/manifest.json"
            if require(path):
                try:
                    manifest = json.loads(path.read_text(encoding="utf-8"))
                    if manifest["revision"] != QWEN_REVISION:
                        raise ValueError("Base revision mismatch")
                    for name, record in manifest["files"].items():
                        if (
                            not (snapshot / name).is_file()
                            or sha256(snapshot / name) != record["sha256"]
                        ):
                            raise ValueError(f"Base snapshot hash mismatch: {name}")
                except (ValueError, KeyError, OSError) as exc:
                    problems.append(f"Base verification failed: {exc}")
    if policy == "sft":
        adapter = adapter or root / DEFAULT_ADAPTER
        if not adapter.is_absolute():
            adapter = root / adapter
        weights = adapter / "adapter_model.safetensors"
        require(adapter / "adapter_config.json")
        present = require(weights)
        metrics_path = adapter.parent / "metrics.json"
        config_path = adapter.parent / "run_config.json"
        if require(metrics_path) and require(config_path):
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                config = json.loads(config_path.read_text(encoding="utf-8"))
                if metrics["status"] != "full_sft" or config["base_revision"] != QWEN_REVISION:
                    raise ValueError("A full SFT adapter with the pinned base is required")
                if present and sha256(weights) != metrics["adapter_sha256"]:
                    raise ValueError("Adapter hash mismatch")
            except (ValueError, KeyError, OSError) as exc:
                problems.append(f"Adapter verification failed: {exc}")
    return {"ready": not problems, "policy": policy, "problems": problems, "checked_files": checked}


def resolve_output(root: Path, value: Path) -> Path:
    """Do not accidentally write into another project or overwrite an old run."""
    output = (value if value.is_absolute() else root / value).resolve()
    boundary = (root / "outputs").resolve()
    if (
        not boundary.is_relative_to(root.resolve())
        or output == boundary
        or not output.is_relative_to(boundary)
    ):
        raise ValueError("Choose a fresh subdirectory inside this project's outputs/")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output directory must be fresh or empty")
    return output
