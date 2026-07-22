"""Regression tests for the adversarial-review fixes."""
import threading
import time
import unicodedata

import pytest

import stressonnx
import stressonnx.download as download
from stressonnx import (
    SIMPLE_LANGS,
    ModelDownloadError,
    ModelLoadError,
    StressonnxError,
    stress,
    warm_up,
)
from stressonnx.backends.simple import OOV_RULES, SimpleStressor
from stressonnx.registry import _OOV_RULES, HF_REPO_REVISION


def test_dotted_capital_i_keeps_mark_on_vowel():
    """'İ'.lower() is two code points; indices must not shift (İstanbul bug)."""
    s = SimpleStressor("az-Latn")
    s._ensure_loaded()
    out = s._accentuate_oov("İstanbul")
    mark = out.find("́")
    assert out[mark - 1].lower() in s._vowels, out


def test_unknown_notation_raises():
    with pytest.raises(ValueError, match="notation"):
        stress("Қазақстан", "kk", notation="pluss")


def test_every_simple_lang_has_registered_rule():
    """A typo'd rule name must never silently degrade to final stress."""
    for lang in SIMPLE_LANGS:
        assert _OOV_RULES[lang] in OOV_RULES, lang


def test_corrupt_cache_engages_fallback(tmp_path, monkeypatch):
    """A file that downloads fine but fails to parse must raise ModelLoadError
    (a StressonnxError) — and therefore participate in the fallback chain."""
    bad = tmp_path / "kaz" / "vocab.gz"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"this is not gzip")
    (tmp_path / "kaz" / "meta.json").write_text(
        '{"alpha": "аб", "vowels": "а", "oov_rule": "last"}'
    )
    s = SimpleStressor("kk", cache_dir=str(tmp_path))
    with pytest.raises(ModelLoadError) as excinfo:
        s._ensure_loaded()
    assert isinstance(excinfo.value, StressonnxError)
    assert excinfo.value.model_id == "simple"


def test_truncated_cache_file_is_refetched(tmp_path):
    """A zero-byte file in cache_dir must not be trusted as complete."""
    empty = tmp_path / "kaz" / "meta.json"
    empty.parent.mkdir(parents=True)
    empty.touch()
    paths = download._download_files("kaz", ["meta.json"], cache_dir=str(tmp_path))
    import os
    assert os.path.getsize(paths["meta.json"]) > 0


def test_revision_is_pinned():
    assert len(HF_REPO_REVISION) == 40  # commit hash, not a branch name


def test_warm_up_shares_singleton_with_stress():
    warm_up("kk")
    key = ("kk", "simple")
    assert key in stressonnx._SINGLETONS
    backend = stressonnx._SINGLETONS[key]
    assert backend._loaded
    assert stress("Қазақстан", "kk") == "Қазақста́н"
    assert stressonnx._SINGLETONS[key] is backend


def test_default_and_explicit_model_share_one_instance():
    stress("Қазақстан", "kk")
    stress("Қазақстан", "kk", model="simple")
    keys = [k for k in stressonnx._SINGLETONS if k[0] == "kk"]
    assert keys == [("kk", "simple")]


def test_failure_cooldown_prevents_immediate_retry(monkeypatch):
    calls = []

    def _boom(*args, **kwargs):
        calls.append(1)
        raise OSError("simulated outage")

    monkeypatch.setattr(download, "hf_hub_download", _boom)
    stressonnx._SINGLETONS.clear()
    with pytest.raises(ModelDownloadError):
        stress("привет", "ru")
    first = len(calls)
    with pytest.raises(ModelDownloadError):
        stress("привет", "ru")  # within cooldown → no new download attempt
    assert len(calls) == first


def test_precomposed_acute_input_not_double_marked():
    s = SimpleStressor("az-Latn")
    s._ensure_loaded()
    pre = unicodedata.normalize("NFC", "Bakı́")  # precomposed stressed input
    out = s(pre)
    assert unicodedata.normalize("NFD", out).count("́") == 1


def test_concurrent_first_load_is_consistent():
    """Racing threads on one instance must both see a fully loaded backend."""
    s = SimpleStressor("tt")
    results = []

    def worker():
        results.append(s("Казан"))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(results) == {"Каза́н"}
