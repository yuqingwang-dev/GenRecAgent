import hashlib
from pathlib import Path

import pytest

from genrecagent.delivery.source_evidence import verify_source


def test_changed_worktree_can_verify_unchanged_historical_experiment(tmp_path, monkeypatch):
    path = tmp_path / "src/tool.py"
    path.parent.mkdir()
    original = b"original code\n"
    digest = hashlib.sha256(original).hexdigest()
    path.write_bytes(original)
    assert verify_source(tmp_path, "src/tool.py", digest, "a" * 40) == "matching_worktree_bytes"
    path.write_text("post-experiment hardening\n", encoding="utf-8")

    def git_blob(command, *, cwd, **kwargs):
        assert command == ["git", "cat-file", "blob", f"{'a' * 40}:src/tool.py"]
        assert cwd == Path(tmp_path).resolve()
        return original

    monkeypatch.setattr("genrecagent.delivery.source_evidence.subprocess.check_output", git_blob)
    assert verify_source(tmp_path, "src/tool.py", digest, "a" * 40) == (
        "matching_preserved_commit_bytes"
    )
    with pytest.raises(ValueError, match="do not match"):
        verify_source(tmp_path, "src/tool.py", "b" * 64, "a" * 40)


def test_unsafe_source_paths_and_unpinned_commits_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid recorded source"):
        verify_source(tmp_path, "../not-this-project.py", "a" * 64, "b" * 40)
    with pytest.raises(ValueError, match="full preserved commit"):
        verify_source(tmp_path, "missing.py", "a" * 64, "HEAD")
