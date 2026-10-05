"""Download a pinned public Qwen policy model and record immutable file hashes."""

import argparse
import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _resume_http(config: dict, partial: Path) -> Path:
    import httpx
    from huggingface_hub import get_hf_file_metadata, hf_hub_url

    cache = (ROOT / config["cache_dir"]).resolve()
    partial = partial.resolve()
    if not partial.is_relative_to(cache) or not partial.is_file():
        raise ValueError("Partial must be an existing file inside the project model cache")
    if not partial.name.startswith(config["weights_sha256"] + "."):
        raise ValueError("Partial filename differs from the pinned weight hash")
    snapshot = cache / ("models--" + config["model_name"].replace("/", "--")) / (
        "snapshots/" + config["revision"]
    )
    if not (snapshot / "config.json").is_file() or not (snapshot / "tokenizer.json").is_file():
        raise ValueError("Download the small model/tokenizer files before resuming weights")
    destination = snapshot / "model.safetensors"
    if destination.exists():
        raise FileExistsError(destination)
    metadata = get_hf_file_metadata(hf_hub_url(
        config["model_name"], "model.safetensors", revision=config["revision"]
    ), token=False)
    if metadata.etag.strip('"') != config["weights_sha256"]:
        raise ValueError("Official download etag differs from the pinned weights")
    offset, total = partial.stat().st_size, metadata.size
    if total is None or not 0 <= offset <= total:
        raise ValueError("Partial size is invalid")
    if offset < total:
        with httpx.Client(timeout=httpx.Timeout(30, connect=15), follow_redirects=True) as client:
            with client.stream(
                "GET", metadata.location, headers={"Range": f"bytes={offset}-"}
            ) as r:
                r.raise_for_status()
                if r.status_code != 206 or not re.match(
                    rf"bytes {offset}-\d+/{total}$", r.headers.get("content-range", "")
                ):
                    raise ValueError("Official host did not honor the verified byte range")
                last_print = monotonic()
                with partial.open("ab") as stream:
                    for chunk in r.iter_bytes(chunk_size=1024 * 1024):
                        stream.write(chunk)
                        offset += len(chunk)
                        if monotonic() - last_print >= 10:
                            print(f"Weights: {offset}/{total} bytes", flush=True)
                            last_print = monotonic()
    if partial.stat().st_size != total:
        raise ValueError("Downloaded size differs from official metadata")
    with partial.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if digest != config["weights_sha256"]:
        raise ValueError("Resumed weights failed SHA-256 verification")
    partial.replace(destination)
    return snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/llm_policy.yaml")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/m4_qwen_download")
    parser.add_argument("--use-xet", action="store_true", help="Use the Hub's chunked transport")
    parser.add_argument("--resume-http-partial", type=Path,
                        help="Resume a verified model-cache partial from the official HTTP host")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {args.output_dir}")
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    os.environ.setdefault("HF_HOME", str(ROOT / "artifacts/hf_home"))
    os.environ["HF_HUB_DISABLE_XET"] = "0" if args.use_xet else "1"
    from huggingface_hub import snapshot_download

    if args.resume_http_partial:
        snapshot = _resume_http(config, args.resume_http_partial)
    else:
        snapshot = Path(snapshot_download(
            repo_id=config["model_name"], revision=config["revision"],
            cache_dir=ROOT / config["cache_dir"], token=False,
            allow_patterns=["*.json", "*.txt", "*.safetensors", "LICENSE", "README.md"],
            max_workers=4,
        ))
    files = {}
    for path in sorted(snapshot.iterdir()):
        if path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            files[path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    if files["model.safetensors"]["sha256"] != config["weights_sha256"]:
        raise ValueError("Model weights do not match the pinned SHA-256")
    manifest = {
        "model_name": config["model_name"], "revision": config["revision"],
        "snapshot_path": str(snapshot), "files": files,
        "total_bytes": sum(record["bytes"] for record in files.values()),
        "transport": "resumed_http" if args.resume_http_partial else (
            "hub_xet" if args.use_xet else "hub_http"
        ),
        "created_utc": datetime.now(UTC).isoformat(),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2))


if __name__ == "__main__":
    main()
