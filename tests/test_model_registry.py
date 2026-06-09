"""Tests for the model registry and model-selection API."""
import pytest
from stressonnx import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    make_stressor,
    Stressor,
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
    assert set(MODEL_REGISTRY.keys()) == {"ruaccent", "silero", "simple", "kubataba"}


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
    assert DEFAULT_MODEL["ukr"] == "silero"
    assert DEFAULT_MODEL["bel"] == "silero"


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
        make_stressor(model="ruaccent", lang="kaz")


def test_make_stressor_silero_no_lang():
    with pytest.raises(ValueError, match="lang.*must be specified"):
        make_stressor(model="silero")


def test_make_stressor_simple_no_lang():
    with pytest.raises(ValueError, match="lang.*must be specified"):
        make_stressor(model="simple")


def test_make_stressor_simple_returns_simple_stressor():
    from stressonnx.accentor import SimpleStressor
    s = make_stressor(model="simple", lang="kaz")
    assert isinstance(s, SimpleStressor)


def test_make_stressor_silero_returns_silero_stressor():
    from stressonnx.accentor import _SileroStressor
    s = make_stressor(model="silero", lang="ukr")
    assert isinstance(s, _SileroStressor)


def test_make_stressor_ruaccent_returns_ruaccent_stressor():
    from stressonnx.accentor import RuAccentStressor
    s = make_stressor(model="ruaccent", lang="ru")
    assert isinstance(s, RuAccentStressor)


def test_make_stressor_default_ru():
    """make_stressor(lang='ru') should use the ruaccent model."""
    from stressonnx.accentor import RuAccentStressor
    s = make_stressor(lang="ru")
    assert isinstance(s, RuAccentStressor)


def test_make_stressor_default_ukr():
    from stressonnx.accentor import _SileroStressor
    s = make_stressor(lang="ukr")
    assert isinstance(s, _SileroStressor)


# ---------------------------------------------------------------------------
# Stressor class — model-aware wrapper
# ---------------------------------------------------------------------------

def test_stressor_model_attr_simple():
    s = Stressor(lang="kaz")
    assert s.model == "simple"


def test_stressor_model_attr_silero():
    s = Stressor(model="silero", lang="ukr")
    assert s.model == "silero"


def test_stressor_model_attr_ruaccent():
    s = Stressor(model="ruaccent", lang="ru")
    assert s.model == "ruaccent"


def test_stressor_lang_attr():
    s = Stressor(lang="kaz")
    assert s.lang == "kaz"


def test_stressor_callable_simple():
    s = Stressor(lang="kaz")
    result = s("Казан")
    assert "́" in result  # combining acute U+0301


def test_stressor_callable_silero_ukr():
    s = Stressor(model="silero", lang="ukr")
    result = s("Привіт")
    assert "́" in result


def test_stressor_callable_silero_bel():
    s = Stressor(model="silero", lang="bel")
    result = s("свет")
    assert "́" in result


# ---------------------------------------------------------------------------
# stress() function — model parameter
# ---------------------------------------------------------------------------

def test_stress_model_param_simple():
    result = stress("Казан", "tat", model="simple")
    assert result == "Каза́н"


def test_stress_model_param_silero():
    result = stress("Привіт", "ukr", model="silero")
    assert "́" in result


def test_stress_model_none_equals_default():
    """stress(text, lang) == stress(text, lang, model=None)."""
    for lang, text in [("kaz", "Казан"), ("ukr", "Привіт"), ("tat", "Казан")]:
        assert stress(text, lang) == stress(text, lang, model=None)


def test_stress_backward_compat_no_model():
    """stress(text, lang) without model argument still works."""
    assert "́" in stress("Привіт", "ukr")
    assert "́" in stress("Казан", "tat")


def test_stress_model_wrong_for_lang():
    with pytest.raises(ValueError):
        stress("text", "kaz", model="ruaccent")


# ---------------------------------------------------------------------------
# All-langs coverage (smoke: no crash, returns string with '+' or unchanged)
# ---------------------------------------------------------------------------

_LANG_SAMPLES = {
    "aze_cyr": "Бакы",
    "aze_lat": "Bakı",
    "uzb_cyr": "Тошкент",
    "uzb_lat": "Toshkent",
    "bak": "Өфе",
    "bel_simple": "свет",
    "chv": "Шупашкар",
    "erz": "Саранск",
    "hye": "Երեւան",
    "kat": "თბილისი",
    "kaz": "Казан",
    "kbd": "Налшык",
    "kir": "Бишкек",
    "kjh": "Абакан",
    "mdf": "Саранск",
    "sah": "Дьокуускай",
    "tat": "Казан",
    "tgk": "Душанбе",
    "udm": "Ижевск",
    "xal": "Элиста",
}


@pytest.mark.parametrize("lang,text", list(_LANG_SAMPLES.items()))
def test_all_simple_langs_via_stressor(lang, text):
    s = Stressor(lang=lang)
    result = s(text)
    assert isinstance(result, str)
    assert len(result) >= len(text)
