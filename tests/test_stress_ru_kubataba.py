"""Tests for the kubataba Russian stress backend (requires network on first run).

Note: this model is sentence-level (trained on literary sentence pairs).
Single isolated words may not always produce correct results; use phrases.
"""
import pytest

# Basic stress placement — sentence context, no homograph disambiguation expected.
# Output uses combining acute U+0301 after the stressed vowel.
CASES = [
    ("привет мир", "приве́т мир"),
    ("молоко убежало", "молоко́ убежа́ло"),
    ("красивый город", "краси́вый го́род"),
    ("хлеб и вода", "хлеб и вода́"),
    ("Россия великая страна", "Росси́я вели́кая страна́"),
]


@pytest.mark.parametrize("inp,expected", CASES)
def test_kubataba_stress(inp, expected):
    from stressonnx import stress
    assert stress(inp, "ru", model="kubataba") == expected


def test_kubataba_returns_diacritic():
    from stressonnx import stress
    result = stress("красивый город", "ru", model="kubataba")
    assert "́" in result, f"No combining acute in {result!r}"


def test_kubataba_plus_notation():
    from stressonnx import stress
    result = stress("привет мир", "ru", model="kubataba", notation="plus")
    assert "+" in result
    assert "́" not in result


def test_kubataba_class_direct():
    from stressonnx import _KubatabaStressor
    s = _KubatabaStressor()
    result = s("красивый город")
    assert "́" in result


def test_kubataba_not_default_for_ru():
    """ruaccent must remain the default; kubataba is opt-in."""
    from stressonnx import DEFAULT_MODEL
    assert DEFAULT_MODEL["ru"] == "ruaccent"


def test_kubataba_in_registry():
    from stressonnx import MODEL_REGISTRY
    assert "kubataba" in MODEL_REGISTRY
    assert "ru" in MODEL_REGISTRY["kubataba"].langs


def test_kubataba_make_stressor():
    from stressonnx import make_stressor
    s = make_stressor(model="kubataba", lang="ru")
    assert callable(s)


def test_kubataba_wrong_lang_raises():
    from stressonnx import make_stressor
    with pytest.raises(ValueError, match="does not support"):
        make_stressor(model="kubataba", lang="ukr")
