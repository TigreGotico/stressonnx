"""The v2 surface: pipelines, structured analysis, capability selection."""
import pytest

from stressonnx import (
    StressPipeline,
    StressResult,
    StressedWord,
    analyze,
    stress,
    stress_batch,
)
from stressonnx.langs import canonicalize_lang, load_languages


def test_canonicalize():
    assert canonicalize_lang("ru") == ("ru", None)
    assert canonicalize_lang("kaz") == ("kk", None)
    assert canonicalize_lang("bel_simple") == ("be", "simple")
    assert canonicalize_lang("xx") == ("xx", None)  # unknown passes through


def test_language_files_complete():
    for tag, spec in load_languages().items():
        assert spec["tag"] == tag
        assert spec["script"] in ("cyrillic", "latin", "armenian", "georgian")
        assert spec["rule"]
        assert spec["sources"], tag
        assert "simple" in spec["hf"], tag


def test_analyze_offsets_and_rendering():
    r = analyze("старинный замок стоит на горе", "ru")
    assert isinstance(r, StressResult)
    assert r.original == "старинный замок стоит на горе"
    words = {w.text: w for w in r.words}
    assert words["замок"].stressed_char == "а"
    assert words["замок"].stressed_index == 1
    assert words["на"].stressed_index is None
    assert r.text == "стари́нный за́мок сто́ит на горе́"


def test_analyze_reports_yo_restoration():
    r = analyze("зеленый лес", "ru", model="silero")
    w = r.words[0]
    assert w.yo_restored and w.stressed_index == 3 and w.text == "зеленый"


def test_prefer_strategies():
    assert stress("привет", "ru", prefer="smallest") == stress("привет", "ru", model="simple")
    assert stress("привет", "ru", prefer="fast") == stress("привет", "ru", model="silero")
    with pytest.raises(ValueError, match="prefer"):
        stress("привет", "ru", prefer="bestest")


def test_batch():
    assert stress_batch(["вода", "молоко"], "ru", model="simple") == ["вода́", "молоко́"]


def test_private_pipeline_is_isolated():
    p = StressPipeline()
    assert p.stress("Қазақстан", "kk") == "Қазақста́н"
    assert ("kk", "simple") in p._singletons
    import stressonnx
    # the private pipeline's cache is not the default pipeline's
    assert p._singletons is not stressonnx._SINGLETONS
