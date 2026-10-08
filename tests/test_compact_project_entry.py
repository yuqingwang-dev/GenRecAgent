import runpy
import subprocess
import sys
from pathlib import Path


def test_unified_entry_keeps_legacy_adapter_and_resolves_recovery_adapter():
    script = Path(__file__).parents[1] / "scripts/run_project.py"
    entry = runpy.run_path(str(script), run_name="entry_test_only")
    choose, root = entry["resolve_adapter"], entry["ROOT"]
    assert choose("sft", None) == root / "outputs/tool_sft_lab95_server/tool_sft_lora/adapter"
    assert choose("guarded_sft", None) == choose("sft", None)
    assert choose("compact_sft", None) == (
        root / "outputs/tool_sft_recovery_v3_server/tool_sft_recovery_v3/adapter"
    )
    assert choose("compact_sft", Path("outputs/custom/adapter")) == root / "outputs/custom/adapter"


def test_unified_help_exposes_raw_compact_modes_without_initializing_cuda():
    root = Path(__file__).parents[1]
    process = subprocess.run(
        [sys.executable, "scripts/run_project.py", "--help"],
        cwd=root,
        text=True,
        capture_output=True,
        check=True,
        timeout=20,
    )
    assert "compact_prompt" in process.stdout and "compact_sft" in process.stdout
