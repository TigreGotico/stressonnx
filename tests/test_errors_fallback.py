"""Failure-path behavior: typed errors and the opt-in fallback chain."""
import logging

import pytest

import stressonnx
import stressonnx.download as download
from stressonnx import (
    FALLBACK_PRIORITY,
    ModelDownloadError,
    StressonnxError,
    UnsupportedLanguageError,
    stress,
)
from stressonnx import _fallback_chain


def test_exception_hierarchy():
    assert issubclass(UnsupportedLanguageError, StressonnxError)
    assert issubclass(UnsupportedLanguageError, ValueError)
    assert issubclass(ModelDownloadError, StressonnxError)


def test_unsupported_language_names_lang_and_supported():
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        stress("hello", "xx")
    assert excinfo.value.lang == "xx"
    assert "ru" in excinfo.value.supported


def test_fallback_priority_order():
    assert FALLBACK_PRIORITY == ("ruaccent", "silero", "simple")
    assert _fallback_chain("ru") == [
        ("ruaccent", "ru"), ("silero", "ru"), ("simple", "ru")
    ]
    assert _fallback_chain("uk") == [("silero", "uk"), ("simple", "uk")]
    assert _fallback_chain("be") == [("silero", "be"), ("simple", "be")]
    assert _fallback_chain("kk") == [("simple", "kk")]
    assert _fallback_chain("xx") == []


def test_download_failure_propagates_without_fallback(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("simulated outage")

    monkeypatch.setattr(download, "hf_hub_download", _boom)
    stressonnx._SINGLETONS.clear()
    with pytest.raises(ModelDownloadError) as excinfo:
        stress("привет", "ru")
    assert excinfo.value.model_id == "ruaccent"  # user-facing id, not the HF subdir


def test_fallback_walks_chain_and_warns(monkeypatch, caplog):
    """ruaccent download fails → silero serves the request, with a warning."""
    real = download.hf_hub_download

    def _fail_ruaccent(*args, **kwargs):
        if kwargs.get("filename", "").startswith("ru_ruaccent/"):
            raise OSError("simulated outage")
        return real(*args, **kwargs)

    monkeypatch.setattr(download, "hf_hub_download", _fail_ruaccent)
    stressonnx._SINGLETONS.clear()
    with caplog.at_level(logging.WARNING, logger="stressonnx"):
        result = stress("красивый город", "ru", fallback=True)
    assert result == "краси́вый го́род"
    assert any("falling back" in r.message for r in caplog.records)


def test_fallback_exhaustion_raises(monkeypatch):
    def _boom(*args, **kwargs):
        raise OSError("simulated outage")

    monkeypatch.setattr(download, "hf_hub_download", _boom)
    stressonnx._SINGLETONS.clear()
    with pytest.raises(ModelDownloadError):
        stress("привет", "ru", fallback=True)


def test_fallback_false_is_default_behavior():
    """No fallback flag → single-model semantics, cached singleton reuse."""
    assert stress("красивый город", "ru", model="silero") == "краси́вый го́род"
