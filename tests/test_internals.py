"""Tests for pure-Python utility functions in accentor.py that do not require
network access or loaded ONNX models.

Covers: _plus_to_diacritic, _ruaccent_norm, _delete_spaces_before_punc,
_fix_capital, _ruaccent_split_by_words, _ruaccent_split_by_sentences,
RuAccentStressor._has_punct, RuAccentStressor._count_vowels,
_apostrophe_to_diacritic.
"""
import pytest


# ---------------------------------------------------------------------------
# _plus_to_diacritic
# ---------------------------------------------------------------------------

def test_plus_to_diacritic_basic():
    from stressonnx.accentor import _plus_to_diacritic
    assert _plus_to_diacritic("з+амок") == "за́мок"


def test_plus_to_diacritic_word_end():
    from stressonnx.accentor import _plus_to_diacritic
    assert _plus_to_diacritic("молок+о") == "молоко́"


def test_plus_to_diacritic_no_stress():
    from stressonnx.accentor import _plus_to_diacritic
    assert _plus_to_diacritic("замок") == "замок"


def test_plus_to_diacritic_trailing_plus():
    from stressonnx.accentor import _plus_to_diacritic
    # a lone + at end of string — no following char, kept as-is
    assert _plus_to_diacritic("abc+") == "abc+"


def test_plus_to_diacritic_multiple():
    from stressonnx.accentor import _plus_to_diacritic
    # only one + per word in practice, but handle multiple
    result = _plus_to_diacritic("п+ривет м+ир")
    assert "́" in result
    assert "+" not in result


def test_plus_to_diacritic_roundtrip():
    """to_plus_notation(diacritic) → plus; _plus_to_diacritic(plus) → diacritic."""
    from stressonnx import to_plus_notation
    from stressonnx.accentor import _plus_to_diacritic
    diacritic = "за́мок"
    plus = to_plus_notation(diacritic)
    assert _plus_to_diacritic(plus) == diacritic


# ---------------------------------------------------------------------------
# _ruaccent_norm
# ---------------------------------------------------------------------------

def test_ruaccent_norm_strips_soft_sign():
    from stressonnx.accentor import _ruaccent_norm, _RE_RUACCENT_NORM
    # norm strips chars matched by _RE_RUACCENT_NORM (e.g. combining chars)
    clean = _ruaccent_norm("приве́т")   # combining acute stripped
    assert "́" not in clean


def test_ruaccent_norm_plain_unchanged():
    from stressonnx.accentor import _ruaccent_norm
    assert _ruaccent_norm("привет") == "привет"


# ---------------------------------------------------------------------------
# _delete_spaces_before_punc
# ---------------------------------------------------------------------------

def test_delete_spaces_before_punc_comma():
    from stressonnx.accentor import _delete_spaces_before_punc
    assert _delete_spaces_before_punc("привет , мир") == "привет, мир"


def test_delete_spaces_before_punc_period():
    from stressonnx.accentor import _delete_spaces_before_punc
    assert _delete_spaces_before_punc("конец .") == "конец."


def test_delete_spaces_before_punc_tilde_to_dash():
    from stressonnx.accentor import _delete_spaces_before_punc
    assert _delete_spaces_before_punc("кот~мяч") == "кот-мяч"


def test_delete_spaces_before_punc_no_change():
    from stressonnx.accentor import _delete_spaces_before_punc
    text = "нет знаков препинания"
    assert _delete_spaces_before_punc(text) == text


# ---------------------------------------------------------------------------
# _fix_capital
# ---------------------------------------------------------------------------

def test_fix_capital_all_lower():
    from stressonnx.accentor import _fix_capital
    # same length: ё substitution, no extra chars
    assert _fix_capital("желтый", "жёлтый") == "жёлтый"


def test_fix_capital_first_upper():
    from stressonnx.accentor import _fix_capital
    # same length: capital preserved
    assert _fix_capital("Желтый", "жёлтый") == "Жёлтый"


def test_fix_capital_all_upper():
    from stressonnx.accentor import _fix_capital
    assert _fix_capital("ЖЕЛТЫЙ", "жёлтый") == "ЖЁЛТЫЙ"


def test_fix_capital_length_mismatch_returns_target():
    """If lengths differ, return target unchanged (no capitalisation applied)."""
    from stressonnx.accentor import _fix_capital
    result = _fix_capital("кот", "котёнок")
    assert result == "котёнок"


# ---------------------------------------------------------------------------
# _ruaccent_split_by_words
# ---------------------------------------------------------------------------

def test_split_by_words_basic():
    from stressonnx.accentor import _ruaccent_split_by_words
    words, rem = _ruaccent_split_by_words("привет мир")
    assert "привет" in words
    assert "мир" in words


def test_split_by_words_reconstruct():
    from stressonnx.accentor import _ruaccent_split_by_words
    text = "старинный замок стоит"
    words, rem = _ruaccent_split_by_words(text)
    # Original text reconstructable from words + rem
    reconstructed = "".join(l + r for l, r in zip(rem, words)) + rem[-1]
    assert reconstructed == text


def test_split_by_words_empty():
    from stressonnx.accentor import _ruaccent_split_by_words
    words, rem = _ruaccent_split_by_words("")
    assert words == []


def test_split_by_words_dash_normalised():
    from stressonnx.accentor import _ruaccent_split_by_words
    # " - " (em-dash context) becomes " ~ " internally then back
    words, rem = _ruaccent_split_by_words("кот - собака")
    assert len(words) > 0


# ---------------------------------------------------------------------------
# _ruaccent_split_by_sentences
# ---------------------------------------------------------------------------

def test_split_by_sentences_no_razdel():
    """Without razdel installed, returns single-element list."""
    import sys, types
    # Temporarily hide razdel if present
    razdel_backup = sys.modules.pop("razdel", None)
    try:
        from stressonnx.accentor import _ruaccent_split_by_sentences
        result = _ruaccent_split_by_sentences("Привет мир. Пока мир.")
        assert isinstance(result, list)
        assert len(result) >= 1
    finally:
        if razdel_backup is not None:
            sys.modules["razdel"] = razdel_backup


def test_split_by_sentences_single():
    from stressonnx.accentor import _ruaccent_split_by_sentences
    result = _ruaccent_split_by_sentences("Привет мир")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "Привет мир" in " ".join(result)


# ---------------------------------------------------------------------------
# RuAccentStressor._has_punct / _count_vowels  (static methods)
# ---------------------------------------------------------------------------

def test_has_punct_true():
    from stressonnx.accentor import RuAccentStressor
    assert RuAccentStressor._has_punct("привет,")
    assert RuAccentStressor._has_punct("конец.")
    assert RuAccentStressor._has_punct("вопрос?")


def test_has_punct_false():
    from stressonnx.accentor import RuAccentStressor
    assert not RuAccentStressor._has_punct("привет")
    assert not RuAccentStressor._has_punct("замок")


def test_count_vowels_russian():
    from stressonnx.accentor import RuAccentStressor
    assert RuAccentStressor._count_vowels("замок") == 2    # а, о
    assert RuAccentStressor._count_vowels("молоко") == 3   # о, о, о
    assert RuAccentStressor._count_vowels("стресс") == 1   # е


def test_count_vowels_no_vowels():
    from stressonnx.accentor import RuAccentStressor
    assert RuAccentStressor._count_vowels("крст") == 0


def test_count_vowels_yo():
    from stressonnx.accentor import RuAccentStressor
    assert RuAccentStressor._count_vowels("ёж") == 1


# ---------------------------------------------------------------------------
# _apostrophe_to_diacritic
# ---------------------------------------------------------------------------

def test_apostrophe_to_diacritic_basic():
    from stressonnx.accentor import _apostrophe_to_diacritic
    # apostrophe after vowel → combine acute
    result = _apostrophe_to_diacritic("молоко'")
    assert "́" in result


def test_apostrophe_to_diacritic_mid_word():
    from stressonnx.accentor import _apostrophe_to_diacritic
    result = _apostrophe_to_diacritic("мо'локо")
    assert "́" in result
    assert "'" not in result


def test_apostrophe_to_diacritic_no_apostrophe():
    from stressonnx.accentor import _apostrophe_to_diacritic
    assert _apostrophe_to_diacritic("замок") == "замок"


def test_apostrophe_after_consonant_unchanged():
    from stressonnx.accentor import _apostrophe_to_diacritic
    # apostrophe after non-vowel (consonant) should NOT become a stress mark
    result = _apostrophe_to_diacritic("кот'")
    assert "́" not in result


# ---------------------------------------------------------------------------
# make_stressor — offline error paths
# ---------------------------------------------------------------------------

def test_make_stressor_kubataba_wrong_lang():
    from stressonnx import make_stressor
    with pytest.raises(ValueError, match="does not support language"):
        make_stressor(model="kubataba", lang="ukr")


def test_make_stressor_silero_wrong_lang():
    from stressonnx import make_stressor
    with pytest.raises(ValueError, match="does not support language"):
        make_stressor(model="silero", lang="kaz")


def test_make_stressor_simple_wrong_lang():
    from stressonnx import make_stressor
    with pytest.raises(ValueError, match="does not support language"):
        make_stressor(model="simple", lang="ru")


def test_stressor_invalid_notation():
    from stressonnx import Stressor
    with pytest.raises(ValueError, match="notation"):
        Stressor(lang="kaz", notation="invalid")
