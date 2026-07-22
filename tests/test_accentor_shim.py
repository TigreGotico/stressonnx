"""stressonnx.accentor is a deprecated backward-compat shim."""
import importlib

import pytest


def test_accentor_import_emits_deprecation_warning():
    import stressonnx.accentor  # noqa: F401 — ensure it's importable at all

    with pytest.warns(DeprecationWarning):
        importlib.reload(stressonnx.accentor)


def test_accentor_shim_exposes_stress_critical_names():
    import stressonnx.accentor as accentor

    assert accentor.STRESS_TOKEN == "́"
    assert callable(accentor._download_files)
    assert accentor.RuAccentStressor is not None
    assert accentor.SimpleStressor is not None
