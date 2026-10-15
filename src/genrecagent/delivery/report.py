"""Aggregate existing measurements without running or inventing experiments."""

import json
from pathlib import Path

from genrecagent.delivery.learned_report import build_learned_report
from genrecagent.delivery.readiness import DEFAULT_ADAPTER, sha256
from genrecagent.delivery.recovery_report import RECOVERY_BASE, build_recovery_report


def build_report(root: Path) -> dict:
    sources = {}

    def read(relative: str):
        path = root / relative
        payload = json.loads(path.read_text(encoding="utf-8"))
        sources[relative] = sha256(path)
        return payload

    m1 = read("outputs/m1_all_beauty_full/metrics.json")
    transition = read("outputs/m3_transition_test/metrics.json")["test"]
    transformer = read("outputs/m3_transformer_test_full_local/metrics.json")["test"]
    dual = read("outputs/fast_dual_validation/metrics.json")
    base = "outputs/tool_sft_lab95_server"
    training = read(f"{base}/tool_sft_lora/metrics.json")
    config = read(f"{base}/tool_sft_lora/run_config.json")
    if training["status"] != "full_sft" or not training["assistant_only_labels"]:
        raise ValueError("Cannot present software smoke or unmasked training as real SFT")
    if sha256(root / DEFAULT_ADAPTER / "adapter_model.safetensors") != training["adapter_sha256"]:
        raise ValueError("Returned adapter does not match the trained artifact")
    policies = {
        name: read(f"{base}/lab95_comparison/{name}/metrics.json")
        for name in ("fixed", "prompt", "sft", "no_recovery", "no_finish_verification")
    }
    reference = policies["fixed"]
    for name, row in policies.items():
        for key in (
            "tasks",
            "task_ids",
            "test_file_sha256",
            "source_hashes",
            "git_commit_at_start",
            "base_model_revision",
            "recommendation_checkpoint_sha256",
        ):
            if row[key] != reference[key]:
                raise ValueError(f"Unpaired policy result: {name}/{key}")
    if policies["sft"]["adapter_sha256"] != training["adapter_sha256"]:
        raise ValueError("Evaluated SFT adapter differs from the training artifact")
    users, items = m1["test"]["users"], m1["test"]["items"]
    if any(row["users"] != users for row in (m1["validation"], transition, transformer, dual)):
        raise ValueError("Retrieval runs have different user counts")
    if dual["items"] != items or m1["validation"]["items"] != items:
        raise ValueError("Retrieval runs have different catalog counts")
    retrieval_test = {
        **m1["test"]["methods"],
        "code_transition": transition,
        "transformer": transformer,
    }
    frozen_test_path = root / "outputs/dual_retrieval_frozen_test/metrics.json"
    if frozen_test_path.exists():
        frozen_test = read("outputs/dual_retrieval_frozen_test/metrics.json")
        if frozen_test["status"] != "full_test_dual_retrieval" or frozen_test["split"] != "test":
            raise ValueError("A complete test run is required for test fusion results")
        if frozen_test["users"] != users or frozen_test["items"] != items:
            raise ValueError("Frozen fusion test uses a different evaluation cohort")
        for name, reference in (
            ("dense", m1["test"]["methods"]["dense"]),
            ("generative", transformer),
        ):
            if any(
                frozen_test["methods"][name][key] != reference[key]
                for key in ("Recall@20", "NDCG@20")
            ):
                raise ValueError("Test source endpoints do not reproduce their original baselines")
        retrieval_test.update(
            {
                key: value
                for key, value in frozen_test["methods"].items()
                if key.startswith("hybrid_")
            }
        )
    recovery = build_recovery_report(root) if (root / RECOVERY_BASE).exists() else None
    learned = build_learned_report(root, catalog_items=items, users=users)
    if learned is not None:
        sources.update(learned["source_sha256"])
        if learned["generative"] is not None:
            retrieval_test["transformer_rqvae_stable"] = learned["generative"]["test"]
    return {
        "scope": "aggregation of frozen runs; no new training or benchmark evaluation",
        "users": users,
        "catalog_items": items,
        "retrieval_test": {
            name: {key: row[key] for key in row if key.startswith(("Recall@", "NDCG@"))}
            for name, row in retrieval_test.items()
        },
        "dual_retrieval_validation_only": dual["methods"],
        "policy_heldout_synthetic_subset": policies,
        "sft_training": training,
        "recovery_v3": recovery,
        "learned_chain": learned,
        "training_code_commit": config["git_commit_at_start"],
        "training_seconds": config["elapsed_seconds"],
        "source_metric_sha256": sources,
        "default_policy": "fixed",
        "limits": [
            "Legacy v2 policy tasks are synthetic with normalized constraints "
            "and reused templates.",
            f"Legacy v2 SFT filled {policies['sft']['feasible_filled']}/"
            f"{policies['sft']['feasible_tasks']} "
            "feasible requests and correctly clarified "
            f"{policies['sft']['correct_clarifications']}.",
            "Frozen hybrid test is reported when available; small differences are not robust gain, "
            "online impact or production-readiness claims.",
            "Per-user chronological split and static catalog "
            "do not guarantee global temporal safety.",
            "Recovery v3, when present, is a separate fresh synthetic split; "
            "the environment supplies observable flags and a precomputed static eligibility count.",
            "Raw Compact Prompt is an untuned schema-following baseline; "
            "this does not establish superiority over an optimized prompted agent.",
            "Learned RQ-VAE token quality is not next-item relevance; new Transformer "
            "results remain separate from the old model/fusion and do not switch defaults.",
            "The existing retrieval test was already observed for historical baselines; "
            "one seed and few hits do not establish a robust generative gain.",
        ],
    }


def render_report(data: dict) -> str:
    lines = [
        "# GenRecAgent delivery evidence",
        "",
        data["scope"],
        "",
        f"Catalog: {data['catalog_items']:,} items; retrieval users: {data['users']:,}.",
        "",
        "## Next-item retrieval — test only",
        "",
        "| Method | Recall@20 | NDCG@20 |",
        "| --- | ---: | ---: |",
    ]
    for name, row in data["retrieval_test"].items():
        lines.append(f"| {name} | {row['Recall@20']:.6f} | {row['NDCG@20']:.6f} |")
    learned = data.get("learned_chain")
    if learned is not None:
        token = learned["tokenizer"]
        lines.extend([
            "", "## Learned Semantic ID — verified full RQ-VAE", "",
            f"{token['items']:,} items; variant `{token['variant']}`; "
            f"selected epoch {token['best_epoch']}; validation reconstruction MSE "
            f"{token['validation_reconstruction_mse']:.8f}.", "",
            f"Used codes: {[row['used_codes'] for row in token['codebook_usage_by_level']]}; "
            f"unique codes {token['collisions']['unique_code_count']:,}; "
            f"colliding items {token['collisions']['colliding_item_count']:,}. "
            "Both frozen tokenizer quality gates pass; collisions remain explicit.", "",
            "The original collapsed RQ-VAE and separately named RQ-KMeans baseline "
            "are preserved, not relabeled as this learned model.",
        ])
        generator = learned["generative"]
        if generator is not None:
            lines.extend([
                "", "## New RQ-VAE-coded Transformer — validation, then frozen test", "",
                f"{generator['train_examples']:,} training prefixes; "
                f"{generator['parameters']:,} parameters; "
                f"selected epoch {generator['best_epoch']}.",
                "", "| Split | Users | Recall@20 | NDCG@20 |",
                "| --- | ---: | ---: | ---: |",
            ])
            for split in ("validation", "test"):
                row = generator[split]
                lines.append(f"| {split} | {row['users']} | {row['Recall@20']:.6f} | "
                             f"{row['NDCG@20']:.6f} |")
            lines.extend([
                "", "All saved user candidate ranks were CPU-regraded. "
                "The new model is not part of historical fusion/SFT comparisons. "
                "Mixed small changes do not establish overall superiority; "
                "old runtime defaults remain.",
            ])
    lines.extend(
        [
            "",
            "## Real dual retrieval — validation only",
            "",
            "| Method | Recall@20 | NDCG@20 |",
            "| --- | ---: | ---: |",
        ]
    )
    for name, row in data["dual_retrieval_validation_only"].items():
        lines.append(f"| {name} | {row['Recall@20']:.6f} | {row['NDCG@20']:.6f} |")
    lines.extend(
        [
            "",
            "## Same-lab95 policy comparison — "
            f"{data['policy_heldout_synthetic_subset']['fixed']['tasks']} synthetic held-out tasks",
            "",
            "| Policy | Contracts | Feasible filled | Clarifications | Illegal calls | Tokens |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, row in data["policy_heldout_synthetic_subset"].items():
        lines.append(
            f"| {name} | {row['contract_pass']}/{row['tasks']} | "
            f"{row['feasible_filled']}/{row['feasible_tasks']} | {row['correct_clarifications']} | "
            f"{row['policy_illegal_calls']} | {row['generation_tokens']} |"
        )
    training = data["sft_training"]
    lines.extend(
        [
            "",
            "## Historical v2 SFT — separate small-task protocol",
            "",
            f"{training['train_actions']} train actions, {training['epochs']} epoch; "
            f"validation action NLL {training['initial_validation_nll']:.6f} -> "
            f"{training['best_validation_nll']:.6f}; {data['training_seconds']:.2f} seconds.",
            "",
            f"Adapter SHA256: `{training['adapter_sha256']}`.",
            "",
            "Default demo: **Fixed**. Raw Prompt and SFT remain explicit experiments.",
            "",
        ]
    )
    recovery = data.get("recovery_v3")
    if recovery is not None:
        lines.extend(
            [
                "## Recovery v3 — fresh synthetic held-out split, raw models",
                "",
                recovery["scope"],
                "",
                "| Policy | Contracts | Feasible filled | Clarifications | "
                "Illegal calls | Mean ms |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for name in ("fixed", "compact_prompt", "compact_sft"):
            row = recovery["raw_results"][name]["metrics"]
            lines.append(
                f"| {name} | {row['contract_pass']}/{row['tasks']} | "
                f"{row['feasible_filled']}/{row['feasible_tasks']} | "
                f"{row['correct_clarifications']} | {row['policy_illegal_calls']} | "
                f"{row['avg_execution_latency_ms']:.2f} |"
            )
        lines.extend(
            [
                "",
                f"Selected epoch: {recovery['selection']['selected_epoch']}; "
                f"selected unweighted validation NLL: "
                f"{recovery['selection']['selected_validation_nll']:.6f}; "
                f"diagnostic minimum NLL: {recovery['selection']['minimum_validation_nll']:.6f}.",
                "",
                f"Changed finite LoRA tensors: {recovery['adapter_update']['changed_tensors']}/"
                f"{recovery['adapter_update']['tensors']}. No runtime controller is used.",
                "",
                "This protocol differs from v2; do not merge their rates "
                "or infer a cross-protocol gain.",
                "",
                f"Full raw validation: "
                f"{recovery['raw_results']['validation_compact_sft']['metrics']['contract_pass']}/"
                f"{recovery['raw_results']['validation_compact_sft']['metrics']['tasks']}. "
                "Fresh synthetic test success is not a universal reliability claim.",
                "",
            ]
        )
    lines.extend(["## Limits", ""])
    lines.extend(f"- {limit}" for limit in data["limits"])
    lines.extend(["", "## Input metric checksums", ""])
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in data["source_metric_sha256"].items())
    if recovery is not None:
        lines.extend(["", "### Recovery v3 fingerprints", ""])
        lines.extend(
            f"- `{path}`: `{digest}`" for path, digest in recovery["source_sha256"].items()
        )
    return "\n".join(lines) + "\n"
