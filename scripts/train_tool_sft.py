"""Minimal assistant-action LoRA SFT; tiny CPU mode is a software check only."""

import argparse
import json
import math
import random
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.agent.trajectory_data import action_tokens
from genrecagent.data.storage import read_jsonl
from genrecagent.retrieval.hybrid_loader import file_sha256

ROOT = Path(__file__).resolve().parents[1]


def evaluate_loss(model, rows, device):
    total, tokens = 0.0, 0
    model.eval()
    with torch.no_grad():
        for row in rows:
            inputs = torch.tensor([row["input_ids"]], device=device)
            labels = torch.tensor([row["labels"]], device=device)
            count = sum(label != -100 for label in row["labels"])
            loss = float(model(input_ids=inputs, labels=labels, use_cache=False).loss)
            if not math.isfinite(loss):
                raise ValueError("Nonfinite validation loss")
            total += loss * count
            tokens += count
    model.train()
    return total / tokens


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/tool_sft.yaml")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_dir or ROOT / config["output_dir"]
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory must be empty")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    snapshot = (
        ROOT
        / config["cache_dir"]
        / ("models--" + config["model_name"].replace("/", "--"))
        / "snapshots"
        / config["revision"]
    )
    if not snapshot.is_dir():
        raise ValueError("Pinned model snapshot must already be cached")
    tokenizer = AutoTokenizer.from_pretrained(
        str(snapshot),
        revision=config["revision"],
        cache_dir=ROOT / config["cache_dir"],
        local_files_only=True,
        trust_remote_code=False,
    )
    manifest = None
    if args.smoke_test:
        device = "cpu"
        model = AutoModelForCausalLM.from_config(
            Qwen2Config(
                vocab_size=max(tokenizer.get_vocab().values()) + 1,
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                max_position_embeddings=256,
            )
        )
        messages = [
            {"role": "user", "content": "Choose a tool."},
            {"role": "assistant", "content": '{"name":"read_history","arguments":{}}'},
        ]
        train_rows = [action_tokens(tokenizer, messages) for _ in range(2)]
        validation_rows = train_rows[:1]
    else:
        device = config["device"]
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA required for the configured full run")
        data = ROOT / config["data_dir"]
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("observation_snapshots_immutable") is not True:
            raise ValueError("Use corrected v2 immutable-observation trajectories")
        if manifest["base_revision"] != config["revision"]:
            raise ValueError("SFT data and model revision differ")
        groups = {
            split: set(manifest["splits"][split]["user_groups"])
            for split in ("train", "validation", "test")
        }
        if any(
            groups[a] & groups[b]
            for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise ValueError("SFT user leakage")
        encoded = {}
        for split in ("train", "validation"):
            path = data / f"{split}_actions.jsonl"
            if file_sha256(path) != manifest["splits"][split]["actions_sha256"]:
                raise ValueError("SFT data hash changed")
            encoded[split] = [
                action_tokens(tokenizer, row["messages"], max_length=config["max_sequence_length"])
                for row in read_jsonl(path)
            ]
        train_rows, validation_rows = encoded["train"], encoded["validation"]
        model = AutoModelForCausalLM.from_pretrained(
            str(snapshot),
            revision=config["revision"],
            cache_dir=ROOT / config["cache_dir"],
            local_files_only=True,
            trust_remote_code=False,
            dtype=getattr(torch, config["dtype"]),
        )
    model = get_peft_model(
        model,
        LoraConfig(
            r=config["rank"],
            lora_alpha=config["lora_alpha"],
            target_modules=config["target_modules"],
            bias="none",
            task_type="CAUSAL_LM",
        ),
    ).to(device)
    if not args.smoke_test:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.config.use_cache = False
    model.train()
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"])
    accumulation = 1 if args.smoke_test else config["gradient_accumulation"]
    epochs = 1 if args.smoke_test else config["epochs"]
    started, log, best = perf_counter(), [], float("inf")
    output.mkdir(parents=True, exist_ok=True)
    before = evaluate_loss(model, validation_rows, device)
    print(
        f"train={len(train_rows)} val={len(validation_rows)} trainable="
        f"{sum(p.numel() for p in parameters)} initial_val_nll={before:.4f}",
        flush=True,
    )
    for epoch in range(1, epochs + 1):
        order = list(range(len(train_rows)))
        random.shuffle(order)
        total, count = 0.0, 0
        optimizer.zero_grad()
        for number, index in enumerate(order, 1):
            row = train_rows[index]
            inputs = torch.tensor([row["input_ids"]], device=device)
            labels = torch.tensor([row["labels"]], device=device)
            loss = model(input_ids=inputs, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            (loss / accumulation).backward()
            total += float(loss.detach())
            count += 1
            if number % accumulation == 0 or number == len(order):
                torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                optimizer.step()
                optimizer.zero_grad()
            if number % 50 == 0:
                print(
                    f"epoch={epoch} actions={number}/{len(order)} loss={total / count:.4f}",
                    flush=True,
                )
        validation = evaluate_loss(model, validation_rows, device)
        record = {"epoch": epoch, "train_nll": total / count, "validation_nll": validation}
        log.append(record)
        if validation < best:
            best = validation
            model.save_pretrained(output / "adapter", safe_serialization=True)
            tokenizer.save_pretrained(output / "adapter")
        print(json.dumps(record), flush=True)
    metrics = {
        "status": "software_smoke_only" if args.smoke_test else "full_sft",
        "train_actions": len(train_rows),
        "validation_actions": len(validation_rows),
        "trainable_parameters": sum(p.numel() for p in parameters),
        "initial_validation_nll": before,
        "best_validation_nll": best,
        "epochs": epochs,
        "assistant_only_labels": True,
        "adapter_sha256": file_sha256(output / "adapter/adapter_model.safetensors"),
    }
    run_config = {
        "config": config,
        "dataset_manifest": manifest,
        "smoke_test": args.smoke_test,
        "base_revision": config["revision"],
        "device": device,
        "seed": config["seed"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "versions": {package: version(package) for package in ("torch", "transformers", "peft")},
        "elapsed_seconds": perf_counter() - started,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    for name, payload in (
        ("metrics.json", metrics),
        ("training_log.json", log),
        ("run_config.json", run_config),
    ):
        (output / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
