"""Freeze curated intent tasks with disjoint templates and train-history users."""

import argparse
import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, default=ROOT / "configs/m4_intent_tasks.json")
    parser.add_argument("--prepared-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_full_optimized")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "data/processed/All_Beauty_m4_tasks")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    specification = json.loads(args.templates.read_text(encoding="utf-8"))
    templates = specification["templates"]
    if len({template["id"] for template in templates}) != len(templates):
        raise ValueError("Template IDs must be unique")
    with (args.prepared_dir / "histories.jsonl").open(encoding="utf-8") as stream:
        users = [json.loads(line) for line in stream if line.strip()]
    users.sort(key=lambda row: hashlib.sha256(
        f"{args.seed}:m4:{row['user_id']}".encode()
    ).hexdigest())
    if len(users) < len(templates):
        raise ValueError("Not enough distinct train-history users")
    rows: dict[str, list[dict]] = {"development": [], "heldout": []}
    for template, user in zip(templates, users, strict=False):
        split = template["split"]
        if split not in rows:
            raise ValueError("Unknown task split")
        for context in specification["contexts"]:
            if context == "known_history":
                history = user["train_items"]
                group = hashlib.sha256(user["user_id"].encode()).hexdigest()[:20]
            elif context == "cold_start":
                history, group = [], f"{template['id']}:cold_start"
            elif context == "unknown_history":
                history, group = ["__unknown_catalog_item__"], f"{template['id']}:unknown_history"
            else:
                raise ValueError("Unknown history context")
            rows[split].append({
                "task_id": f"{template['id']}:{context}", "template_id": template["id"],
                "split": split, "user_group": group, "context": context,
                "text": template["text"], "history_items": history, "top_k": 5,
                "expected": template["expected"],
            })
    if {row["user_group"] for row in rows["development"]} & {
        row["user_group"] for row in rows["heldout"]
    }:
        raise ValueError("User groups overlap across splits")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split, tasks in rows.items():
        with (args.output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for task in tasks:
                stream.write(json.dumps(task, ensure_ascii=False) + "\n")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True,
        capture_output=True, check=False,
    ).stdout.strip())
    manifest = {
        "task_set": specification["version"], "seed": args.seed,
        "source": "curated controlled-language templates with real catalog and train histories",
        "split_rule": "disjoint template IDs and user groups; no recommendation targets used",
        "history_rule": "train_items only; cold/unknown contexts explicitly synthetic",
        "counts": {split: len(tasks) for split, tasks in rows.items()},
        "templates_sha256": _sha256(args.templates),
        "histories_sha256": _sha256(args.prepared_dir / "histories.jsonl"),
        "items_sha256": _sha256(args.prepared_dir / "items.jsonl"),
        "split_sha256": {
            split: _sha256(args.output_dir / f"{split}.jsonl") for split in rows
        },
        "git_commit": commit, "git_dirty": dirty,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
