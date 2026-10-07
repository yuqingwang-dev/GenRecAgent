"""Check returned full-pool agent artifacts against local frozen inputs, without inference."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.storage import load_items, read_jsonl
from genrecagent.delivery.readiness import resolve_output, sha256
from genrecagent.evaluation.agent_evidence import audit_run, check_paired_runs

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output = resolve_output(ROOT, args.output_dir)
    task_file = args.task_file.resolve()
    tasks = list(read_jsonl(task_file))
    items = load_items(ROOT / "data/processed/All_Beauty_full_optimized")
    runs, reports = [], []
    for directory in args.input_dir:
        metrics_file, trace_file = directory / "metrics.json", directory / "traces.jsonl"
        metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
        rows = list(read_jsonl(trace_file))
        if sha256(task_file) != metrics["task_file_sha256"]:
            raise ValueError("Frozen original task-file hash mismatch")
        if not metrics["source_hashes"]:
            raise ValueError("No recorded evaluation-source fingerprints")
        for name, digest in metrics["source_hashes"].items():
            source = (ROOT / name).resolve()
            if not source.is_relative_to(ROOT) or sha256(source) != digest:
                raise ValueError(f"Recorded evaluation source hash mismatch: {name}")
        checkpoint = ROOT / "outputs/m3_transformer_full_server/transformer.pt"
        if sha256(checkpoint) != metrics["recommendation_checkpoint_sha256"]:
            raise ValueError("Recommendation checkpoint hash mismatch")
        if metrics["adapter_sha256"]:
            adapter = args.adapter or Path(metrics["adapter"])
            if sha256(adapter / "adapter_model.safetensors") != metrics["adapter_sha256"]:
                raise ValueError("SFT adapter hash mismatch")
        report = audit_run(metrics, rows, tasks, items)
        reports.append(
            {
                "input_dir": str(directory),
                "policy": metrics["policy"],
                "ablation": metrics["ablation"],
                **report,
                "metrics_sha256": sha256(metrics_file),
                "traces_sha256": sha256(trace_file),
            }
        )
        runs.append((metrics, rows))
    check_paired_runs(runs)
    report = {
        "status": "passed",
        "scope": "recorded evidence consistency, not a new "
        "retrieval run, statistical quality proof or cryptographic origin attestation",
        "task_file_sha256": sha256(task_file),
        "runs": reports,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "evidence_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
