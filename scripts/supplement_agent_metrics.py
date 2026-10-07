"""Read-only regrading of archived results; no new inference, training or result overwrite."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.storage import load_items, read_jsonl
from genrecagent.delivery.readiness import resolve_output, sha256
from genrecagent.evaluation.agent_contracts import end_reason, grade_contract, summarize_contracts

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    source = args.input_dir if args.input_dir.is_absolute() else ROOT / args.input_dir
    traces_path = source / "traces.jsonl"
    metrics = json.loads((source / "metrics.json").read_text(encoding="utf-8"))
    catalog = load_items(ROOT / "data/processed/All_Beauty_full_optimized")
    rows = list(read_jsonl(traces_path))
    if len(rows) != metrics["tasks"]:
        raise ValueError("Archive task count mismatch")
    for row in rows:
        row["grade"] = grade_contract(row["result"], row["task"], catalog)
        row["result"]["end_reason"] = end_reason(row["result"])
    if sum(row["grade"]["contract_pass"] for row in rows) != metrics["contract_pass"]:
        raise ValueError("Regrading contradicts old contract result; do not quietly overwrite it")
    summary = summarize_contracts(rows)
    summary.update(
        {
            "scope": "supplement to archived run; no new inference or training",
            "source_trace_sha256": sha256(traces_path),
            "source_metrics_sha256": sha256(source / "metrics.json"),
            "source_policy": metrics["policy"],
            "source_tasks": len(rows),
            "legacy_provenance_limit": "older previews lack full pools; unknown is not proof",
        }
    )
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = resolve_output(ROOT, args.output_dir or Path(f"outputs/metric_supplements/{stamp}"))
    output.mkdir(parents=True, exist_ok=True)
    (output / "supplement.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
