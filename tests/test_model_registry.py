"""Tests for the model registry and model-selection API."""
import pytest
from stressonnx import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    make_stressor,
    stress,
    RUACCENT_LANGS,
    MAIN_LANGS,
    SIMPLE_LANGS,
    ALL_LANGS,
)


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------

def test_registry_keys():
    assert set(MODEL_REGISTRY.keys()) == {"ruaccent", "silero", "simple"}


def test_registry_families():
    for model_id, entry in MODEL_REGISTRY.items():
        assert hasattr(entry, "langs")
        assert hasattr(entry, "family")
        assert hasattr(entry, "description")
        assert isinstance(entry.langs, frozenset)


def test_default_model_covers_all_langs():
    for lang in ALL_LANGS:
        assert lang in DEFAULT_MODEL, f"{lang!r} missing from DEFAULT_MODEL"
        assert DEFAULT_MODEL[lang] in MODEL_REGISTRY


def test_default_model_ru():
    assert DEFAULT_MODEL["ru"] == "ruaccent"


def test_default_model_ukr_bel():
    assert DEFAULT_MODEL["uk"] == "silero"
    assert DEFAULT_MODEL["be"] == "silero"


def test_default_model_simple_langs():
    for lang in SIMPLE_LANGS - MAIN_LANGS - RUACCENT_LANGS:
        assert DEFAULT_MODEL[lang] == "simple"


# ---------------------------------------------------------------------------
# make_stressor factory
# ---------------------------------------------------------------------------

def test_make_stressor_no_args():
    with pytest.raises(ValueError, match="At least one of"):
        make_stressor()


def test_make_stressor_unknown_lang():
    with pytest.raises(ValueError, match="Unsupported language"):
        make_stressor(lang="xx")


def test_make_stressor_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        make_stressor(model="nonexistent", lang="ru")


def test_make_stressor_model_lang_mismatch():
    with pytest.raises(ValueError, match="does not support language"):
        make_stressor(model="ruaccent", lang="kk")


def test_make_stressor_silero_no_lang():
    with pytest.raises(ValueError, match="lang.*must be specified"):
        make_stressor(model="silero")


def test_make_stressor_simple_no_lang():
    with pytest.raises(ValueError, match="lang.*must be specified"):
        make_stressor(model="simple")


def test_make_stressor_simple_returns_simple_stressor():
    from stressonnx.backends.simple import SimpleStressor
    s = make_stressor(model="simple", lang="kk")
    assert isinstance(s, SimpleStressor)


def test_make_stressor_silero_returns_silero_stressor():
    from stressonnx.backends.silero import _SileroStressor
    s = make_stressor(model="silero", lang="uk")
    assert isinstance(s, _SileroStressor)


def test_make_stressor_ruaccent_returns_ruaccent_stressor():
    from stressonnx.backends.ruaccent import RuAccentStressor
    s = make_stressor(model="ruaccent", lang="ru")
    assert isinstance(s, RuAccentStressor)


def test_make_stressor_default_ru():
    """make_stressor(lang='ru') should use the ruaccent model."""
    from stressonnx.backends.ruaccent import RuAccentStressor
    s = make_stressor(lang="ru")
    assert isinstance(s, RuAccentStressor)


def test_make_stressor_default_ukr():
    from stressonnx.backends.silero import _SileroStressor
    s = make_stressor(lang="uk")
    assert isinstance(s, _SileroStressor)


# ---------------------------------------------------------------------------
# Model-aware selection through the public API
# ---------------------------------------------------------------------------

def test_default_model_for_lang():
    from stressonnx import stress
    assert stress("Алматы", "kk") == "Алматы́"


def test_explicit_model_selection():
    from stressonnx import stress
    assert stress("Привіт світ", "uk", model="silero") == "Приві́т сві́т"
    assert stress("красивый город", "ru", model="ruaccent") == "краси́вый го́род"
    assert stress("Прывітанне свет", "be", model="silero") == "Прывіта́нне све́т"

# ---------------------------------------------------------------------------

_LANG_SAMPLES = {
    "az-Cyrl": "Бакы",
    "az-Latn": "Bakı",
    "uz-Cyrl": "Тошкент",
    "uz-Latn": "Toshkent",
    "ba": "Өфе",
    "be": "свет",
    "cv": "Шупашкар",
    "myv": "Саранск",
    "hy": "Երեւան",
    "ka": "თბილისი",
    "kk": "Казан",
    "kbd": "Налшык",
    "ky": "Бишкек",
    "kjh": "Абакан",
    "mdf": "Саранск",
    "sah": "Дьокуускай",
    "tt": "Казан",
    "tg": "Душанбе",
    "udm": "Ижевск",
    "xal": "Элиста",
}


@pytest.mark.parametrize("lang,text", list(_LANG_SAMPLES.items()))
def test_all_simple_langs_via_stressor(lang, text):
    from stressonnx import make_stressor
    s = make_stressor(model="simple", lang=lang)
    result = s(text)
    assert isinstance(result, str)
    assert len(result) >= len(text)
