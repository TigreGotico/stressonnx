"""Permanent guard: every runtime module in stressonnx/ must import without torch.

stressonnx is documented as pure-onnxruntime (no torch, no silero_stress at
runtime).  This spawns a subprocess per module that blocks both packages via
``sys.modules`` before import, proving the import graph never reaches for them.
"""
import pathlib
import subprocess
import sys

import pytest

_PKG_ROOT = pathlib.Path(__file__).resolve().parent.parent / "stressonnx"


def _module_names() -> list:
    names = []
    for path in sorted(_PKG_ROOT.rglob("*.py")):
        rel = path.relative_to(_PKG_ROOT.parent)
        parts = list(rel.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


_MODULES = _module_names()


@pytest.mark.parametrize("module_name", _MODULES)
def test_module_imports_without_torch(module_name):
    code = (
        "import sys\n"
        "sys.modules['torch'] = None\n"
        "sys.modules['silero_stress'] = None\n"
        f"import {module_name}\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=str(_PKG_ROOT.parent),
    )
    assert result.returncode == 0, (
        f"importing {module_name!r} failed with torch/silero_stress blocked:\n"
        f"{result.stderr}"
    )
