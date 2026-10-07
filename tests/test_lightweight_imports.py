import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_storage_and_recovery_checks_do_not_import_optional_model_runtime():
    root = Path(__file__).parents[1]
    program = """
import json, sys
sys.path.insert(0, 'src')
import genrecagent.data.storage
import genrecagent.agent.recovery_preflight
from genrecagent.semantic_id import SemanticCodebook
assert SemanticCodebook().assignments == {}
print(json.dumps({name: name in sys.modules for name in ('torch', 'numpy', 'transformers')}))
"""
    process = subprocess.run(
        [sys.executable, "-c", program],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    assert json.loads(process.stdout) == {"torch": False, "numpy": False, "transformers": False}


def test_lazy_exports_preserve_existing_codebook_api_and_unknown_attributes():
    import genrecagent.semantic_id as package
    from genrecagent.semantic_id.codebook import SemanticCodebook

    assert package.SemanticCodebook is SemanticCodebook
    assert set(package.__all__) <= set(dir(package))
    with pytest.raises(AttributeError, match="no attribute"):
        _ = package.nonexistent_semantic_api
