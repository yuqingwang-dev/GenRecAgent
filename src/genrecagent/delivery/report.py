"""Aggregate existing measurements without running or inventing experiments."""

import json
from pathlib import Path

from genrecagent.delivery.readiness import DEFAULT_ADAPTER, sha256


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
    return {
        "scope": "aggregation of frozen runs; no new training or benchmark evaluation",
        "users": users,
        "catalog_items": items,
        "retrieval_test": {
            name: {key: row[key] for key in ("Recall@20", "NDCG@20")}
            for name, row in retrieval_test.items()
        },
        "dual_retrieval_validation_only": dual["methods"],
        "policy_heldout_synthetic_subset": policies,
        "sft_training": training,
        "training_code_commit": config["git_commit_at_start"],
        "training_seconds": config["elapsed_seconds"],
        "source_metric_sha256": sources,
        "default_policy": "fixed",
        "limits": [
            "Policy tasks are synthetic and provide normalized constraints; templates are reused.",
            f"SFT filled {policies['sft']['feasible_filled']}/{policies['sft']['feasible_tasks']} "
            "feasible requests and correctly clarified "
            f"{policies['sft']['correct_clarifications']}.",
            "No hybrid test result, generative relevance gain, online impact or production claim.",
            "Per-user chronological split and static catalog "
            "do not guarantee global temporal safety.",
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
            "## Actual SFT",
            "",
            f"{training['train_actions']} train actions, {training['epochs']} epoch; "
            f"validation action NLL {training['initial_validation_nll']:.6f} -> "
            f"{training['best_validation_nll']:.6f}; {data['training_seconds']:.2f} seconds.",
            "",
            f"Adapter SHA256: `{training['adapter_sha256']}`.",
            "",
            "Default demo: **Fixed**. Raw Prompt and SFT remain explicit experiments.",
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {limit}" for limit in data["limits"])
    lines.extend(["", "## Input metric checksums", ""])
    lines.extend(f"- `{path}`: `{digest}`" for path, digest in data["source_metric_sha256"].items())
    return "\n".join(lines) + "\n"
