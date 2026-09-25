"""Prepare the official All_Beauty Parquet shards without remote code execution."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.amazon import prepare_records
from genrecagent.data.storage import iter_parquet_rows, save_prepared

ROOT = Path(__file__).resolve().parents[1]
DATASET = "McAuley-Lab/Amazon-Reviews-2023"
CATEGORY = "All_Beauty"
META_COLUMNS = [
    "parent_asin", "title", "description", "features", "categories",
    "main_category", "price", "average_rating",
]
REVIEW_COLUMNS = [
    "parent_asin", "user_id", "rating", "timestamp", "verified_purchase",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path, str]:
    if args.download:
        if args.review_file or args.metadata_file:
            raise ValueError("Use either --download or both local Parquet file arguments")
        from huggingface_hub import HfApi, hf_hub_download

        revision = HfApi().repo_info(
            DATASET, repo_type="dataset", revision=args.revision or "main"
        ).sha
        cache_dir = ROOT / "data" / "raw" / "hf_cache"
        paths = []
        for prefix in ("raw_review", "raw_meta"):
            paths.append(
                Path(
                    hf_hub_download(
                        repo_id=DATASET,
                        repo_type="dataset",
                        filename=f"{prefix}_{CATEGORY}/full-00000-of-00001.parquet",
                        revision=revision,
                        cache_dir=cache_dir,
                    )
                )
            )
        return paths[0], paths[1], revision
    if not args.review_file or not args.metadata_file:
        raise ValueError("Provide --review-file and --metadata-file, or use --download")
    return args.review_file, args.metadata_file, args.revision or "unknown-local"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", choices=[CATEGORY], default=CATEGORY)
    parser.add_argument("--review-file", type=Path)
    parser.add_argument("--metadata-file", type=Path)
    parser.add_argument(
        "--download", action="store_true", help="Explicitly download both official shards"
    )
    parser.add_argument("--revision", help="Known source revision for local files")
    parser.add_argument(
        "--source-kind",
        choices=["local_unverified", "official_local", "synthetic_fixture"],
        default="local_unverified",
        help="Provenance of local Parquet files; downloads are marked official automatically",
    )
    parser.add_argument("--min-user-interactions", type=int, default=3)
    parser.add_argument("--max-users", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/processed/All_Beauty")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    started = perf_counter()
    review_file, metadata_file, revision = _resolve_paths(args)
    print(f"Dataset: {DATASET} / {CATEGORY}; revision: {revision}")
    print(f"Seed: {args.seed}; max_users: {args.max_users}; minimum interactions: "
          f"{args.min_user_interactions}")
    print(f"Review file: {review_file}; metadata file: {metadata_file}")
    data = prepare_records(
        iter_parquet_rows(metadata_file, META_COLUMNS),
        iter_parquet_rows(review_file, REVIEW_COLUMNS),
        min_user_interactions=args.min_user_interactions,
        max_users=args.max_users,
        seed=args.seed,
    )
    manifest = {
        "dataset": DATASET,
        "category": CATEGORY,
        "source_kind": "official_hf_parquet" if args.download else args.source_kind,
        "revision": revision,
        "review_sha256": _sha256(review_file),
        "metadata_sha256": _sha256(metadata_file),
        "created_utc": datetime.now(UTC).isoformat(),
        "command": "python scripts/prepare_amazon.py",
        "mode": "development_subset" if args.max_users else "full",
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    save_prepared(data, args.output_dir, manifest)
    print(json.dumps(data.stats, indent=2))
    print(f"Output: {args.output_dir}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
