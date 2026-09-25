"""Prepare official All_Beauty JSONL/Parquet files without remote code execution."""

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from genrecagent.data.amazon import prepare_records
from genrecagent.data.storage import iter_jsonl_rows, iter_parquet_rows, save_prepared

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
            raise ValueError("Use either --download or both local file arguments")
        from huggingface_hub import HfApi, hf_hub_download

        info = HfApi().repo_info(
            DATASET, repo_type="dataset", revision=args.revision or "main"
        )
        revision = info.sha
        available = {entry.rfilename for entry in info.siblings}
        cache_dir = ROOT / "data" / "raw" / "hf_cache"
        choices = (
            (
                f"raw_review_{CATEGORY}/full-00000-of-00001.parquet",
                f"raw/review_categories/{CATEGORY}.jsonl",
            ),
            (
                f"raw_meta_{CATEGORY}/full-00000-of-00001.parquet",
                f"raw/meta_categories/meta_{CATEGORY}.jsonl",
            ),
        )
        filenames = []
        for options in choices:
            filename = next((candidate for candidate in options if candidate in available), None)
            if filename is None:
                raise FileNotFoundError(f"No supported official All_Beauty file among {options}")
            filenames.append(filename)
        paths = [
            Path(
                hf_hub_download(
                    repo_id=DATASET,
                    repo_type="dataset",
                    filename=filename,
                    revision=revision,
                    cache_dir=cache_dir,
                )
            )
            for filename in filenames
        ]
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
        "--download", action="store_true", help="Explicitly download both official files"
    )
    parser.add_argument("--revision", help="Known source revision for local files")
    parser.add_argument(
        "--source-kind",
        choices=["local_unverified", "official_local", "synthetic_fixture"],
        default="local_unverified",
        help="Provenance of local files; downloads are marked official automatically",
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
    metadata_rows = (
        iter_parquet_rows(metadata_file, META_COLUMNS)
        if metadata_file.suffix == ".parquet"
        else iter_jsonl_rows(metadata_file)
    )
    review_rows = (
        iter_parquet_rows(review_file, REVIEW_COLUMNS)
        if review_file.suffix == ".parquet"
        else iter_jsonl_rows(review_file)
    )
    data = prepare_records(
        metadata_rows,
        review_rows,
        min_user_interactions=args.min_user_interactions,
        max_users=args.max_users,
        seed=args.seed,
        progress_every=100_000,
    )
    manifest = {
        "dataset": DATASET,
        "category": CATEGORY,
        "source_kind": "official_hf_files" if args.download else args.source_kind,
        "revision": revision,
        "review_sha256": _sha256(review_file),
        "metadata_sha256": _sha256(metadata_file),
        "review_format": review_file.suffix,
        "metadata_format": metadata_file.suffix,
        "created_utc": datetime.now(UTC).isoformat(),
        "command": "python scripts/prepare_amazon.py",
        "mode": "development_subset" if args.max_users else "full",
        "elapsed_seconds": round(perf_counter() - started, 3),
    }
    print("Saving prepared records", flush=True)
    save_prepared(data, args.output_dir, manifest, progress_every=25_000)
    print(json.dumps(data.stats, indent=2))
    print(f"Output: {args.output_dir}; elapsed: {perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
