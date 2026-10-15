"""Build a local reviewer bundle without publishing or including private training artifacts."""

import hashlib
import ipaddress
import json
import re
import subprocess
import zipfile
from pathlib import Path

from genrecagent.delivery.readiness import sha256
from genrecagent.delivery.report import build_report, render_report

_IP = re.compile(r"(?<![\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])")
_ASSIGNMENT = re.compile(
    r"(?im)^\s*(?:export\s+)?[\"']?"
    r"(?:[A-Z_]*(?:API_KEY|ACCESS_TOKEN|SECRET_KEY)|password|passwd|api_key)"
    r"[\"']?\s*[:=]\s*[\"']?([^\s\"'#,;]+)"
)
_TOKEN = re.compile(r"\b(?:sk-[\w-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})\b")
_PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----")


def credential_findings(name: str, text: str) -> list[dict]:
    """Report location/kind only, never the suspicious value."""
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        kind = None
        if _TOKEN.search(line) or _PRIVATE_KEY.search(line):
            kind = "credential_token_or_private_key"
        assignment = _ASSIGNMENT.search(line)
        if assignment:
            value = assignment[1]
            if value.lower() not in {"none", "null", "str"} and not value.startswith(
                ("os.", "self.", "${", "$env:", "<")
            ):
                kind = "nonempty_credential_assignment"
        if kind:
            findings.append({"file": name, "line": number, "kind": kind})
    return findings


def sanitize_environment(text: str) -> str:
    def redact_ip(match):
        try:
            address = ipaddress.ip_address(match[0])
        except ValueError:
            return match[0]
        return "<PRIVATE_IP>" if address.is_private and not address.is_loopback else match[0]

    text = _IP.sub(redact_ip, text)
    text = re.sub(r"/mnt/[^/\s`\"']+", "/mnt/<USER>", text)
    text = re.sub(r"(?i)C:/Users/[^/\s`\"']+", "C:/Users/<USER>", text)
    text = re.sub(r"(?i)C:\\Users\\[^\\\s`\"']+", r"C:\\Users\\<USER>", text)
    text = text.replace("F:/AIProjects/GenRecAgent", "<PROJECT_ROOT>")
    text = text.replace("F:\\AIProjects\\GenRecAgent", "<PROJECT_ROOT>")
    return re.sub(r"\b[\w.-]+@<PRIVATE_IP>", "<USER>@<PRIVATE_IP>", text)


def public_metrics(report: dict) -> dict:
    keys = (
        "tasks",
        "contract_pass",
        "feasible_tasks",
        "feasible_filled",
        "correct_clarifications",
        "constraint_violations",
        "policy_illegal_calls",
        "generation_tokens",
        "avg_execution_latency_ms",
        "ablation",
    )
    snapshot = {
        key: report[key]
        for key in (
            "scope",
            "users",
            "catalog_items",
            "retrieval_test",
            "dual_retrieval_validation_only",
            "sft_training",
            "training_code_commit",
            "training_seconds",
            "source_metric_sha256",
            "default_policy",
            "limits",
        )
    } | {
        "policy_heldout_synthetic_subset": {
            name: {key: row[key] for key in keys}
            for name, row in report["policy_heldout_synthetic_subset"].items()
        }
    }
    snapshot["learned_chain"] = report.get("learned_chain")
    recovery = report.get("recovery_v3")
    snapshot["recovery_v3"] = None if recovery is None else {
        "scope": recovery["scope"], "status": recovery["status"],
        "selection": recovery["selection"], "adapter_update": recovery["adapter_update"],
        "controller_used": recovery["controller_used"],
        "training_seconds": recovery["training_seconds"],
        "training_commit": recovery["training_commit"],
        "source_sha256": recovery["source_sha256"],
        "raw_results": {
            name: {key: row["metrics"][key] for key in keys if key in row["metrics"]}
            for name, row in recovery["raw_results"].items()
        },
    }
    return snapshot


def replay_payload(root: Path, demo_dir: Path, report: dict) -> dict:
    summary = json.loads((demo_dir / "demo_summary.json").read_text(encoding="utf-8"))
    if summary["policy"] != "fixed" or not summary["all_contracts_pass"]:
        raise ValueError("Use the validated Fixed archive, not relabeled model/failing runs")
    descriptions = {
        "hybrid": "有历史、$20 预算：读取历史，实际执行两种召回，再过滤、排序和校验。",
        "cold": "无历史、$20 预算：使用训练期热门度，保留同样的约束与校验。",
        "impossible": "$0 预算：没有合格商品时澄清，不放宽约束。",
    }
    scenarios = []
    for row in summary["scenarios"]:
        name = row["scenario"]
        if name not in descriptions:
            raise ValueError("Unexpected demo scenario")
        trace_path = (demo_dir / row["trace_file"]).resolve()
        if not trace_path.is_relative_to(demo_dir.resolve()):
            raise ValueError("Demo trace escapes its input directory")
        result = json.loads(trace_path.read_text(encoding="utf-8"))
        if result["status"] != row["status"] or len(result["trace"]) != row["steps"]:
            raise ValueError("Demo summary differs from archived trace")
        if [step["action"]["name"] for step in result["trace"]] != row["actions"]:
            raise ValueError("Demo actions differ from archived trace")
        if {p["item_id"] for p in row["products"]} != {p["item_id"] for p in result["items"]}:
            raise ValueError("Demo products differ from archived result")
        steps = []
        for step in result["trace"]:
            observation = step["observation"]
            tool = step["action"]["name"]
            visible = {
                key: observation[key]
                for key in (
                    "ok",
                    "source",
                    "count",
                    "source_counts",
                    "eligible_count",
                    "valid",
                    "sufficient",
                    "status",
                    "reason",
                )
                if key in observation
            }
            if tool == "read_history":
                visible["history_count"] = len(observation.get("history_items", []))
            steps.append({"tool": tool, "observation": json.dumps(visible, ensure_ascii=False)})
        price = result["trace"][0]["state"]["request"]["price_limit"]["amount"]
        scenarios.append(
            {
                "name": name,
                "description": descriptions[name],
                "status": row["status"],
                "max_price": price,
                "history_present": name != "cold",
                "steps": steps,
                "products": row["products"],
                "trace_sha256": sha256(trace_path),
            }
        )
    if {row["name"] for row in scenarios} != set(descriptions):
        raise ValueError("All three recorded scenarios are required")
    recovery = report.get("recovery_v3")
    policies = (report["policy_heldout_synthetic_subset"] if recovery is None else {
        "fixed": recovery["raw_results"]["fixed"]["metrics"],
        "sft": recovery["raw_results"]["compact_sft"]["metrics"],
    })
    return {
        "scope": "saved real-tool trace replay; not live inference, not a new benchmark",
        "catalog_items": report["catalog_items"],
        "users": report["users"],
        "policy_protocol": "legacy_v2" if recovery is None else "recovery_v3",
        "sft_validation_contracts": None if recovery is None else (
            f"{recovery['raw_results']['validation_compact_sft']['metrics']['contract_pass']}/"
            f"{recovery['raw_results']['validation_compact_sft']['metrics']['tasks']}"
        ),
        "fixed_contracts": f"{policies['fixed']['contract_pass']}/{policies['fixed']['tasks']}",
        "sft_contracts": f"{policies['sft']['contract_pass']}/{policies['sft']['tasks']}",
        "sft_feasible_fills": (
            f"{policies['sft']['feasible_filled']}/{policies['sft']['feasible_tasks']}"
        ),
        "sft_correct_clarifications": policies["sft"]["correct_clarifications"],
        "scenarios": scenarios,
    }


def render_replay(template: str, payload: dict) -> str:
    marker = "__GENREC_REPLAY_DATA__"
    if template.count(marker) != 1:
        raise ValueError("Replay template needs exactly one data marker")
    encoded = json.dumps(payload, ensure_ascii=True).replace("<", "\\u003c").replace("&", "\\u0026")
    return template.replace(marker, encoded)


def build_share_package(root: Path, output: Path, demo_dir: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=root).decode("utf-8").strip()

    if git("status", "--porcelain"):
        raise ValueError("Commit/review the working tree before building a source snapshot")
    commit = git("rev-parse", "HEAD")
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    files, findings, redacted, excluded = {}, [], [], []
    allowed_roots = {"src", "scripts", "tests", "configs", "docs", "assets"}
    root_files = {"README.md", "LICENSE", "pyproject.toml", ".gitignore", ".env.example"}
    for name in filter(None, names):
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            findings.append({"file": name, "kind": "symlink_or_escaped_path"})
            continue
        if path.suffix.lower() in {".pt", ".pth", ".npy", ".npz", ".safetensors", ".jsonl"}:
            findings.append({"file": name, "kind": "tracked_heavy_or_raw_data"})
            continue
        if Path(name).name == ".env":
            findings.append({"file": name, "kind": "tracked_environment_secret_file"})
            continue
        allowed = (
            name.split("/")[0] in allowed_roots
            or name in root_files
            or name.startswith("requirements")
            and name.endswith(".txt")
            or name in {"data/README.md", "artifacts/README.md", "outputs/README.md"}
        )
        if not allowed:
            excluded.append(name)
            continue
        original_bytes = path.read_bytes()
        original = original_bytes.decode("utf-8")
        findings.extend(credential_findings(name, original))
        # Keep executable sources byte-identical; redact environment locations in document copies.
        sanitized = sanitize_environment(original) if path.suffix.lower() == ".md" else original
        if sanitized != original:
            redacted.append(name)
        files[name] = sanitized.encode("utf-8") if sanitized != original else original_bytes
    if findings:
        # Only file/line/kind are exposed: never print or save matched secret values.
        raise ValueError("Source share audit failed: " + json.dumps(findings))
    report = build_report(root)
    payload = replay_payload(root, demo_dir, report)
    template = (root / "assets/delivery/replay.html").read_text(encoding="utf-8")
    files["demo_replay.html"] = render_replay(template, payload).encode("utf-8")
    files["evidence/replay_data.json"] = json.dumps(payload, ensure_ascii=False, indent=2).encode()
    files["evidence/metrics_summary.json"] = json.dumps(public_metrics(report), indent=2).encode()
    files["evidence/results.md"] = render_report(report).encode("utf-8")
    start_here = (
        "# GenRecAgent reviewer bundle\n\n"
        "Open demo_replay.html locally. It replays archived real-tool execution, "
        "not live inference.\n\n"
        "Read docs/resume_project_zh.md, docs/project_delivery_zh.md and evidence/results.md.\n\n"
        "No weights, prepared data, cache, credentials, raw user histories "
        "or Git history are included. "
        "Use run_project.py check to identify missing artifacts before live inference.\n\n"
        "Fixed is the archived demo default, not a model-policy result. "
        f"Raw SFT ({payload['policy_protocol']}): {payload['sft_contracts']} synthetic test "
        f"contracts, {payload['sft_feasible_fills']} feasible fills. "
        f"Raw validation: {payload['sft_validation_contracts'] or 'see historical report'}. "
        "These supplied normalized-state tasks do not establish broad planning ability.\n\n"
        "This is a local source/evidence snapshot, NOT an approval to publicly publish. "
        "Full Git history and third-party redistribution terms require separate review. "
        "Environment locations in bundled copies use placeholders; "
        "original project files are unchanged.\n"
    )
    files["START_HERE.md"] = start_here.encode("utf-8")
    manifest = {
        "source_commit": commit,
        "scope": "local lightweight reviewer snapshot; not published",
        "environment_redacted_files": redacted,
        "excluded_tracked_files": excluded,
        "credential_findings": [],
        "git_history_included": False,
        "third_party_terms_and_git_history_review": "not certified by this static bundle check",
        "files": {
            name: {"bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
            for name, value in files.items()
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "GenRecAgent_review_bundle.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for name, value in files.items():
            zipped.writestr("GenRecAgent/" + name, value)
        zipped.writestr("GenRecAgent/share_manifest.json", json.dumps(manifest, indent=2))
    (output / "demo_replay.html").write_bytes(files["demo_replay.html"])
    (output / "share_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "source_commit": commit,
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "files": len(files),
        "bytes": archive.stat().st_size,
        "environment_redacted_files": redacted,
        "published": False,
    }
