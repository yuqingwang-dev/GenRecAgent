"""Independently verify prepared M3 samples on CPU; never train a Transformer."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.m3_audit import (
    audit_m3_sequences,
    check_source_pins,
    resolve_project_output,
    sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/generative_rqvae_stable.yaml"
    )
    parser.add_argument("--reference-sequences", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = resolve_project_output(ROOT, args.output_dir)
    source = config["source"]
    prepared, sequences = ROOT / source["prepared_path"], ROOT / source["output_path"]
    mapping = ROOT / source["semantic_ids_path"]
    metadata = json.loads(mapping.read_text(encoding="utf-8"))["metadata"]
    check_source_pins(source, prepared / "histories.jsonl", mapping, metadata)
    report = audit_m3_sequences(
        prepared, sequences, mapping, reference_sequences=args.reference_sequences,
        expected_mapping_sha256=source.get("expected_semantic_ids_sha256"),
    )
    if source.get("expected_examples") and report["examples"] != source["expected_examples"]:
        raise ValueError("Audited sample counts differ from the frozen experiment")
    report.update(
        created_utc=datetime.now(UTC).isoformat(), config_sha256=sha256(args.config),
        source_hashes={name: sha256(ROOT / name) for name in (
            "scripts/prepare_m3_sequences.py", "scripts/audit_m3_sequences.py",
            "src/genrecagent/data/m3_audit.py", "src/genrecagent/data/sequences.py",
        )},
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
