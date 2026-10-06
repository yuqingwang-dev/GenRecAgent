"""Single safe delivery entry: check assets, run real demos, or aggregate frozen metrics."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.delivery.demo import grade_demo
from genrecagent.delivery.readiness import DEFAULT_ADAPTER, check_artifacts, resolve_output
from genrecagent.delivery.report import build_report, render_report

ROOT = Path(__file__).resolve().parents[1]
SCENARIOS = {
    "hybrid": {"mode": "hybrid", "history": ["B01DNS649G"], "max_price": 20},
    "cold": {"mode": "popularity", "history": [], "max_price": 20},
    "impossible": {"mode": "hybrid", "history": ["B01DNS649G"], "max_price": 0},
}


def readiness(args) -> dict:
    result = check_artifacts(
        ROOT, policy=args.policy, adapter=args.adapter, verify_hashes=args.verify_hashes
    )
    modules = ["numpy", "yaml", "torch"]
    if args.policy != "fixed":
        modules += ["transformers"]
    if args.policy == "sft":
        modules += ["peft"]
    for name in modules:
        if importlib.util.find_spec(name) is None:
            result["problems"].append(f"Missing Python dependency: {name}")
    if sys.version_info[:2] != (3, 11):
        result["problems"].append("Use Python 3.11, as required by this project's environment")
    try:
        # Import Torch before NumPy/Transformers to avoid Windows DLL initialization errors.
        import torch

        result["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__}
        if not torch.cuda.is_available():
            result["problems"].append("CUDA unavailable: this entry reuses the GPU M3 checkpoint")
        else:
            result["runtime"]["gpu"] = torch.cuda.get_device_name(0)
    except (ImportError, OSError, RuntimeError) as exc:
        result["problems"].append(f"Torch/CUDA initialization failed: {exc}")
    result["ready"] = not result["problems"]
    return result


def run_demos(args, output: Path) -> dict:
    from genrecagent.data.storage import load_items

    items = load_items(ROOT / "data/processed/All_Beauty_full_optimized")
    summaries = []
    env = {**os.environ, "PYTHONNOUSERSITE": "1", "HF_HUB_OFFLINE": "1"}
    for name in args.scenario or SCENARIOS:
        scenario = SCENARIOS[name]
        trace = output / f"{name}.json"
        command = [
            sys.executable,
            str(ROOT / "scripts/run_tool_agent.py"),
            "--policy",
            args.policy,
            "--text",
            "Recommend popular products." if name == "cold" else "Recommend based on my history.",
            "--mode",
            scenario["mode"],
            "--max-price",
            str(scenario["max_price"]),
            "--top-k",
            "5",
            "--output",
            str(trace),
        ]
        for item_id in scenario["history"]:
            command += ["--history-item", item_id]
        if args.policy == "sft":
            command += ["--adapter", str(args.adapter)]
        with (output / f"{name}.log").open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=log, check=True)
        result = json.loads(trace.read_text(encoding="utf-8"))
        grade = grade_demo(
            result, items, history=scenario["history"], top_k=5, max_price=scenario["max_price"]
        )
        summary = {
            "scenario": name,
            "policy": args.policy,
            "status": result["status"],
            "steps": result["steps"],
            "invalid_calls": result["invalid_calls"],
            **grade,
            "trace_file": trace.name,
            "actions": [row["action"]["name"] for row in result["trace"]],
            "products": [
                {
                    "item_id": row["item_id"],
                    "title": items[row["item_id"]].title,
                    "metadata_price": items[row["item_id"]].price,
                }
                for row in result["items"]
            ],
        }
        summaries.append(summary)
        print(
            f"{name}: {result['status']}, products={len(result['items'])}, "
            f"steps={result['steps']}, contract_pass={grade['contract_pass']}",
            flush=True,
        )
    return {
        "scope": "real-catalog preset demos; not a new benchmark",
        "policy": args.policy,
        "all_contracts_pass": all(r["contract_pass"] for r in summaries),
        "scenarios": summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["check", "demo", "report"])
    parser.add_argument("--policy", choices=["fixed", "prompt", "sft"], default="fixed")
    parser.add_argument("--adapter", type=Path, default=ROOT / DEFAULT_ADAPTER)
    parser.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Verify every cached Qwen file for prompt/sft; SFT adapter hash is always checked",
    )
    parser.add_argument("--scenario", choices=list(SCENARIOS), action="append")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.adapter.is_absolute():
        args.adapter = ROOT / args.adapter
    try:
        check = None
        if args.action != "report":
            check = readiness(args)
            if args.action == "check" or not check["ready"]:
                print(json.dumps(check, ensure_ascii=False, indent=2))
                return 0 if check["ready"] else 1
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = resolve_output(
            ROOT, args.output_dir or Path(f"outputs/delivery/{args.action}_{stamp}")
        )
        # Read inputs first: missing evidence must never look like a completed report.
        if args.action == "report":
            report = build_report(ROOT)
            output.mkdir(parents=True, exist_ok=True)
            (output / "metrics_summary.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8"
            )
            (output / "results.md").write_text(render_report(report), encoding="utf-8")
        else:
            output.mkdir(parents=True, exist_ok=True)
            report = run_demos(args, output)
            (output / "demo_summary.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            (output / "readiness.json").write_text(json.dumps(check, indent=2), encoding="utf-8")
        print(f"Saved to: {output}")
        return 1 if args.action == "demo" and not report["all_contracts_pass"] else 0
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(
            f"Delivery failed: {exc}. No model download or training was started.", file=sys.stderr
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
