"""Smoke tests for silero Russian stress backend."""
import pytest


def test_silero_ru_stress_basic():
    from stressonnx import stress
    result = stress("молоко", "ru", model="silero")
    assert "́" in result


def test_silero_ru_stress_sentence():
    from stressonnx import stress
    result = stress("красивый город", "ru", model="silero")
    assert "́" in result
    assert isinstance(result, str)


def test_silero_ru_plus_notation():
    from stressonnx import stress
    result = stress("привет", "ru", model="silero", notation="plus")
    assert "+" in result
    assert "́" not in result


def test_silero_ru_not_default():
    """ruaccent must remain the default for ru."""
    from stressonnx import DEFAULT_MODEL
    assert DEFAULT_MODEL["ru"] == "ruaccent"


def test_silero_ru_in_registry():
    from stressonnx import MODEL_REGISTRY
    assert "ru" in MODEL_REGISTRY["silero"].langs


def test_silero_ru_make_stressor():
    from stressonnx import make_stressor, _SileroStressor
    s = make_stressor(model="silero", lang="ru")
    assert isinstance(s, _SileroStressor)
    assert callable(s)
