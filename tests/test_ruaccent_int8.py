"""The ruaccent family ships int8 weights for its two large models as an opt-in.

``ruaccent`` and ``ruaccent-silero`` load ``model.onnx`` for the stress-usage
and omograph models; the ``-int8`` ids load ``model.int8.onnx``.
"""
import pytest

from stressonnx import make_stressor
from stressonnx.backends import ruaccent as backend
from stressonnx.backends import silero as silero_backend
from stressonnx.backends.ruaccent import PRECISIONS, RuAccentStressor
from stressonnx.errors import UnknownModelError
from stressonnx.registry import MODEL_REGISTRY


def requested_files(monkeypatch, stressor) -> list:
    """File names the stressor asks the download layer for, with no network."""
    asked = []

    def fake(subdir, filenames, cache_dir, model_id=None):
        asked.extend(filenames)
        raise RuntimeError("stop after the file list")

    monkeypatch.setattr(backend, "_download_files", fake)
    with pytest.raises(RuntimeError):
        stressor._load()
    return asked


@pytest.mark.parametrize("model_id, expected", [
    ("ruaccent", "fp32"),
    ("ruaccent-silero", "fp32"),
    ("ruaccent-big-poetry", "fp32"),
    ("ruaccent-int8", "int8"),
    ("ruaccent-silero-int8", "int8"),
])
def test_registry_precision(model_id, expected):
    assert MODEL_REGISTRY[model_id].precision == expected


@pytest.mark.parametrize("model_id, name", [
    ("ruaccent", "model.onnx"),
    ("ruaccent-big-poetry", "model.onnx"),
    ("ruaccent-int8", "model.int8.onnx"),
    ("ruaccent-silero-int8", "model.int8.onnx"),
])
def test_precision_selects_the_model_file(monkeypatch, model_id, name):
    stressor = make_stressor(model=model_id, lang="ru")
    asked = requested_files(monkeypatch, stressor)
    omograph_dir = backend.OMOGRAPH_MODELS[MODEL_REGISTRY[model_id].variant]
    assert f"nn_stress_usage/{name}" in asked
    assert f"{omograph_dir}/{name}" in asked
    other = "model.onnx" if name == "model.int8.onnx" else "model.int8.onnx"
    assert f"nn_stress_usage/{other}" not in asked
    assert f"{omograph_dir}/{other}" not in asked
    # the small models stay fp32 whatever the precision
    assert "nn_accent/model.onnx" in asked
    assert "nn_yo_homograph/model.onnx" in asked


@pytest.mark.parametrize("model_id", ["ruaccent", "ruaccent-silero", "ruaccent-int8", "ruaccent-silero-int8"])
def test_download_errors_name_the_model_id(monkeypatch, model_id):
    named = []

    def fake(subdir, filenames, cache_dir, model_id=None):
        named.append(model_id)
        raise RuntimeError("stop after the first download")

    monkeypatch.setattr(backend, "_download_files", fake)
    monkeypatch.setattr(silero_backend._SileroStressor, "_ensure_loaded", lambda self: None)
    with pytest.raises(RuntimeError):
        make_stressor(model=model_id, lang="ru")._load()
    assert named == [model_id]


def test_default_precision_is_fp32(monkeypatch):
    asked = requested_files(monkeypatch, RuAccentStressor())
    assert "nn_stress_usage/model.onnx" in asked
    assert "nn_stress_usage/model.int8.onnx" not in asked


def test_unknown_precision():
    with pytest.raises(UnknownModelError) as err:
        RuAccentStressor(precision="fp16")
    assert err.value.supported == sorted(PRECISIONS)


@pytest.mark.parametrize("fp32_id, int8_id", [
    ("ruaccent", "ruaccent-int8"),
    ("ruaccent-silero", "ruaccent-silero-int8"),
])
def test_int8_and_fp32_agree_on_a_homograph(model_cache_dir, fp32_id, int8_id):
    """Both precisions resolve the same reading of a context homograph."""
    text = "на двери висел большой замок"
    fp32 = make_stressor(model=fp32_id, cache_dir=model_cache_dir)(text)
    int8 = make_stressor(model=int8_id, cache_dir=model_cache_dir)(text)
    assert int8 == fp32
    assert "замо́к" in int8
