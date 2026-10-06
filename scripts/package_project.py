"""Prepare a local reviewer ZIP and offline trace replay; no upload, training or deployment."""

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.delivery.readiness import resolve_output
from genrecagent.delivery.sharing import build_share_package

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo-dir", type=Path, default=ROOT / "outputs/project_delivery/demo")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        output = resolve_output(ROOT, args.output_dir or Path(f"outputs/share/{stamp}"))
        result = build_share_package(ROOT, output, args.demo_dir)
        (output / "package_receipt.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as exc:
        print(f"Package preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
