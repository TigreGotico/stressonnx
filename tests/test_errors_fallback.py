"""Failure-path behavior: typed errors and the opt-in fallback chain."""
import logging
import threading

import pytest

import stressonnx
import stressonnx.backends.simple as simple_backend
import stressonnx.download as download
import stressonnx.pipeline as pipeline_module
from stressonnx import (
    FALLBACK_PRIORITY,
    ModelDownloadError,
    ModelLoadError,
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
    assert FALLBACK_PRIORITY == ("ruaccent-silero", "ruaccent", "silero", "simple")
    assert _fallback_chain("ru") == [
        ("ruaccent-silero", "ru"), ("ruaccent", "ru"), ("silero", "ru"), ("simple", "ru")
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
    assert excinfo.value.model_id == "ruaccent-silero"  # user-facing id, not the HF subdir


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


def test_model_language_mismatch_is_typed():
    """A model asked for a language it does not serve fails typed."""
    with pytest.raises(UnsupportedLanguageError) as excinfo:
        stress("сәлем", "kk", model="silero")
    err = excinfo.value
    assert isinstance(err, ValueError)
    assert (err.lang, err.model, err.supported) == ("kk", "silero", ["be", "ru", "uk"])
    assert str(err) == (
        "Model 'silero' does not support language 'kk'.  "
        "Supported: ['be', 'ru', 'uk']."
    )


def test_unsupported_language_error_without_model():
    err = UnsupportedLanguageError("xx", ["ru", "kk"])
    assert err.model is None
    assert str(err) == "Unsupported language 'xx'.  Supported: ['kk', 'ru']."


class _Clock:
    """Stand-in for the ``time`` module the pipeline reads its clock from."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(pipeline_module, "time", fake)
    return fake


class _VocabSource:
    """Serves a corrupt vocabulary while ``corrupt`` is set; logs every fetch."""

    def __init__(self, bad_path, real) -> None:
        self.bad_path = str(bad_path)
        self.real = real
        self.corrupt = True
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs["filename"])
        if self.corrupt:
            return self.bad_path
        return self.real(*args, **kwargs)


@pytest.fixture
def vocab_source(tmp_path, monkeypatch):
    bad = tmp_path / "vocab.gz"
    bad.write_bytes(b"this is not gzip")
    source = _VocabSource(bad, download.hf_hub_download)
    monkeypatch.setattr(download, "hf_hub_download", source)
    return source


def test_load_failure_retry_delay_doubles_up_to_cap(clock, vocab_source):
    """A load failure is retried, after 30, 60, 120 … s, never more than 600 s."""
    pipeline = stressonnx.StressPipeline(failure_cooldown_s=30.0)
    attempts = []
    for t in (0, 29, 30, 89, 90, 209, 210, 449, 450, 929, 930,
              1529, 1530, 2129, 2130):
        clock.now = float(t)
        before = len(vocab_source.calls)
        with pytest.raises(ModelLoadError) as excinfo:
            pipeline.stress("Қазақстан", "kk")
        assert excinfo.value.model_id == "simple"
        if len(vocab_source.calls) > before:
            attempts.append(t)
    assert attempts == [0, 30, 90, 210, 450, 930, 1530, 2130]

    vocab_source.corrupt = False  # the operator deletes the corrupt file
    clock.now = 2729.0
    with pytest.raises(ModelLoadError):
        pipeline.stress("Қазақстан", "kk")
    clock.now = 2730.0
    assert pipeline.stress("Қазақстан", "kk") == "Қазақста́н"
    assert pipeline._failures == {}


def test_transient_load_failure_heals_after_cooldown(clock, monkeypatch):
    """A load failure that passes (MemoryError) heals without warm_up()."""
    real = simple_backend._load_vocab
    failures = [MemoryError("transient")]

    def _flaky(path):
        if failures:
            raise failures.pop()
        return real(path)

    monkeypatch.setattr(simple_backend, "_load_vocab", _flaky)
    pipeline = stressonnx.StressPipeline(failure_cooldown_s=30.0)
    with pytest.raises(ModelLoadError) as excinfo:
        pipeline.stress("Қазақстан", "kk")
    assert isinstance(excinfo.value.__cause__, MemoryError)
    clock.now = 30.0
    assert pipeline.stress("Қазақстан", "kk") == "Қазақста́н"


def test_warm_up_recovers_from_load_failure(clock, vocab_source):
    """warm_up() loads the pair again and clears the remembered failure."""
    pipeline = stressonnx.StressPipeline(failure_cooldown_s=30.0)
    with pytest.raises(ModelLoadError):
        pipeline.stress("Қазақстан", "kk")
    vocab_source.corrupt = False
    clock.now = 1.0
    pipeline.warm_up("kk")
    assert pipeline.stress("Қазақстан", "kk") == "Қазақста́н"


def test_failure_stored_after_warm_up_does_not_shadow_loaded_model(vocab_source):
    """A request stores its load failure after a concurrent warm_up()
    loaded the model; the stale failure must not refuse later calls."""

    class _LateStore(dict):
        def __init__(self) -> None:
            super().__init__()
            self.storing = threading.Event()
            self.release = threading.Event()

        def __setitem__(self, key, value):
            self.storing.set()
            assert self.release.wait(10)
            super().__setitem__(key, value)

    pipeline = stressonnx.StressPipeline(failure_cooldown_s=30.0)
    pipeline._failures = _LateStore()
    errors = []

    def request():
        try:
            pipeline.stress("Қазақстан", "kk")
        except ModelLoadError as exc:
            errors.append(exc)

    worker = threading.Thread(target=request)
    worker.start()
    assert pipeline._failures.storing.wait(10)
    vocab_source.corrupt = False
    pipeline.warm_up("kk")
    pipeline._failures.release.set()
    worker.join(10)
    assert not worker.is_alive()
    assert len(errors) == 1
    assert pipeline.stress("Қазақстан", "kk") == "Қазақста́н"
    assert dict(pipeline._failures) == {}


def test_download_failure_is_retried_after_flat_cooldown(clock, monkeypatch):
    """A download failure can be an outage: retried every 30 s, no doubling."""
    calls = []

    def _boom(*args, **kwargs):
        calls.append(clock.now)
        raise OSError("simulated outage")

    monkeypatch.setattr(download, "hf_hub_download", _boom)
    pipeline = stressonnx.StressPipeline(failure_cooldown_s=30.0)
    for t in (0, 29, 30, 59, 60, 89, 90):
        clock.now = float(t)
        with pytest.raises(ModelDownloadError):
            pipeline.stress("Қазақстан", "kk")
    assert calls == [0.0, 30.0, 60.0, 90.0]


def _traceback_frames(exc):
    frames, tb = [], exc.__traceback__
    while tb is not None:
        frames.append(tb.tb_frame.f_code.co_name)
        tb = tb.tb_next
    return frames


@pytest.mark.parametrize("fallback", [False, True])
def test_remembered_failure_traceback_stays_bounded(monkeypatch, fallback):
    """Raising a remembered failure again must not grow its traceback."""
    def _boom(*args, **kwargs):
        raise OSError("simulated outage")

    monkeypatch.setattr(download, "hf_hub_download", _boom)
    pipeline = stressonnx.StressPipeline()
    tracebacks = []
    for _ in range(20):
        try:
            pipeline.stress("Қазақстан", "kk", fallback=fallback)
        except ModelDownloadError as exc:
            tracebacks.append(_traceback_frames(exc))
    assert len(tracebacks) == 20
    first, *remembered = tracebacks
    assert all(frames == remembered[0] for frames in remembered)
    assert len(remembered[0]) <= len(first) + 1
    assert remembered[0][-1] == first[-1] == "_download_files"
