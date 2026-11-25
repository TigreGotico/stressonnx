import pytest

import stressonnx


def test_script_vowels_public():
    from stressonnx import SCRIPT_VOWELS
    assert "ё" in SCRIPT_VOWELS["cyrillic"]
    assert "ը" in SCRIPT_VOWELS["armenian"]
    assert "к" not in SCRIPT_VOWELS["cyrillic"]
    with pytest.raises(AttributeError):
        SCRIPT_VOWELS["latin"].add("x")


def test_resolve_lang_public():
    from stressonnx import resolve_lang
    assert resolve_lang("uk-UA", stressonnx.ALL_LANGS) == "uk"


def test_public_names_have_no_underscore():
    assert not [n for n in stressonnx.__all__ if n.startswith("_")]
