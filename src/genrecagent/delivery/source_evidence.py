"""Check an experiment's recorded source bytes, including preserved historical commits."""

import hashlib
import re
import subprocess
from pathlib import Path

from genrecagent.delivery.readiness import sha256


def verify_source(root: Path, name: str, digest: str, commit: str) -> str:
    root = root.resolve()
    relative = Path(name)
    source = (root / relative).resolve()
    if (
        relative.is_absolute()
        or not source.is_relative_to(root)
        or not isinstance(digest, str)
        or not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError("Invalid recorded source path or SHA256")
    if source.is_file() and sha256(source) == digest:
        return "matching_worktree_bytes"
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("A full preserved commit is required for historical source verification")
    try:
        blob = subprocess.check_output(
            ["git", "cat-file", "blob", f"{commit}:{relative.as_posix()}"],
            cwd=root,
            stderr=subprocess.PIPE,
            timeout=15,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Recorded source commit cannot be verified: {name}") from exc
    if hashlib.sha256(blob).hexdigest() != digest:
        raise ValueError(f"Source bytes do not match worktree or recorded commit: {name}")
    return "matching_preserved_commit_bytes"
