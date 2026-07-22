"""Tests for pure-Python utility functions in accentor.py that do not require
network access or loaded ONNX models.

Covers: _plus_to_diacritic, _ruaccent_norm, _delete_spaces_before_punc,
_fix_capital, _ruaccent_split_by_words, _ruaccent_split_by_sentences,
RuAccentStressor._has_punct, RuAccentStressor._count_vowels.
"""
import pytest


# ---------------------------------------------------------------------------
# _plus_to_diacritic
# ---------------------------------------------------------------------------

def test_plus_to_diacritic_basic():
    from stressonnx.notation import _plus_to_diacritic
    assert _plus_to_diacritic("з+амок") == "за́мок"


def test_plus_to_diacritic_word_end():
    from stressonnx.notation import _plus_to_diacritic
    assert _plus_to_diacritic("молок+о") == "молоко́"


def test_plus_to_diacritic_no_stress():
    from stressonnx.notation import _plus_to_diacritic
    assert _plus_to_diacritic("замок") == "замок"


def test_plus_to_diacritic_trailing_plus():
    from stressonnx.notation import _plus_to_diacritic
    # a lone + at end of string — no following char, kept as-is
    assert _plus_to_diacritic("abc+") == "abc+"


def test_plus_to_diacritic_multiple():
    from stressonnx.notation import _plus_to_diacritic
    # only one + per word in practice, but handle multiple
    result = _plus_to_diacritic("п+ривет м+ир")
    assert "́" in result
    assert "+" not in result


def test_plus_to_diacritic_roundtrip():
    """to_plus_notation(diacritic) → plus; _plus_to_diacritic(plus) → diacritic."""
    from stressonnx import to_plus_notation
    from stressonnx.notation import _plus_to_diacritic
    diacritic = "за́мок"
    plus = to_plus_notation(diacritic)
    assert _plus_to_diacritic(plus) == diacritic


# ---------------------------------------------------------------------------
# _ruaccent_norm
# ---------------------------------------------------------------------------

def test_ruaccent_norm_strips_soft_sign():
    from stressonnx.backends.ruaccent import _ruaccent_norm, _RE_RUACCENT_NORM
    # norm strips chars matched by _RE_RUACCENT_NORM (e.g. combining chars)
    clean = _ruaccent_norm("приве́т")   # combining acute stripped
    assert "́" not in clean


def test_ruaccent_norm_plain_unchanged():
    from stressonnx.backends.ruaccent import _ruaccent_norm
    assert _ruaccent_norm("привет") == "привет"


# ---------------------------------------------------------------------------
# _delete_spaces_before_punc
# ---------------------------------------------------------------------------

def test_delete_spaces_before_punc_comma():
    from stressonnx.backends.ruaccent import _delete_spaces_before_punc
    assert _delete_spaces_before_punc("привет , мир") == "привет, мир"


def test_delete_spaces_before_punc_period():
    from stressonnx.backends.ruaccent import _delete_spaces_before_punc
    assert _delete_spaces_before_punc("конец .") == "конец."


def test_delete_spaces_before_punc_tilde_to_dash():
    from stressonnx.backends.ruaccent import _delete_spaces_before_punc
    assert _delete_spaces_before_punc("кот~мяч") == "кот-мяч"


def test_delete_spaces_before_punc_no_change():
    from stressonnx.backends.ruaccent import _delete_spaces_before_punc
    text = "нет знаков препинания"
    assert _delete_spaces_before_punc(text) == text


# ---------------------------------------------------------------------------
# _fix_capital
# ---------------------------------------------------------------------------

def test_fix_capital_all_lower():
    from stressonnx.backends.ruaccent import _fix_capital
    # same length: ё substitution, no extra chars
    assert _fix_capital("желтый", "жёлтый") == "жёлтый"


def test_fix_capital_first_upper():
    from stressonnx.backends.ruaccent import _fix_capital
    # same length: capital preserved
    assert _fix_capital("Желтый", "жёлтый") == "Жёлтый"


def test_fix_capital_all_upper():
    from stressonnx.backends.ruaccent import _fix_capital
    assert _fix_capital("ЖЕЛТЫЙ", "жёлтый") == "ЖЁЛТЫЙ"


def test_fix_capital_length_mismatch_returns_target():
    """If lengths differ, return target unchanged (no capitalisation applied)."""
    from stressonnx.backends.ruaccent import _fix_capital
    result = _fix_capital("кот", "котёнок")
    assert result == "котёнок"


# ---------------------------------------------------------------------------
# _ruaccent_split_by_words
# ---------------------------------------------------------------------------

def test_split_by_words_basic():
    from stressonnx.backends.ruaccent import _ruaccent_split_by_words
    words, rem = _ruaccent_split_by_words("привет мир")
    assert "привет" in words
    assert "мир" in words


def test_split_by_words_reconstruct():
    from stressonnx.backends.ruaccent import _ruaccent_split_by_words
    text = "старинный замок стоит"
    words, rem = _ruaccent_split_by_words(text)
    # Original text reconstructable from words + rem
    reconstructed = "".join(l + r for l, r in zip(rem, words)) + rem[-1]
    assert reconstructed == text


def test_split_by_words_empty():
    from stressonnx.backends.ruaccent import _ruaccent_split_by_words
    words, rem = _ruaccent_split_by_words("")
    assert words == []


def test_split_by_words_dash_normalised():
    from stressonnx.backends.ruaccent import _ruaccent_split_by_words
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
        from stressonnx.backends.ruaccent import _ruaccent_split_by_sentences
        result = _ruaccent_split_by_sentences("Привет мир. Пока мир.")
        assert isinstance(result, list)
        assert len(result) >= 1
    finally:
        if razdel_backup is not None:
            sys.modules["razdel"] = razdel_backup


def test_split_by_sentences_single():
    from stressonnx.backends.ruaccent import _ruaccent_split_by_sentences
    result = _ruaccent_split_by_sentences("Привет мир")
    assert isinstance(result, list)
    assert len(result) >= 1
    assert "Привет мир" in " ".join(result)


# ---------------------------------------------------------------------------
# RuAccentStressor._has_punct / _count_vowels  (static methods)
# ---------------------------------------------------------------------------

def test_has_punct_true():
    from stressonnx.backends.ruaccent import RuAccentStressor
    assert RuAccentStressor._has_punct("привет,")
    assert RuAccentStressor._has_punct("конец.")
    assert RuAccentStressor._has_punct("вопрос?")


def test_has_punct_false():
    from stressonnx.backends.ruaccent import RuAccentStressor
    assert not RuAccentStressor._has_punct("привет")
    assert not RuAccentStressor._has_punct("замок")


def test_count_vowels_russian():
    from stressonnx.backends.ruaccent import RuAccentStressor
    assert RuAccentStressor._count_vowels("замок") == 2    # а, о
    assert RuAccentStressor._count_vowels("молоко") == 3   # о, о, о
    assert RuAccentStressor._count_vowels("стресс") == 1   # е


def test_count_vowels_no_vowels():
    from stressonnx.backends.ruaccent import RuAccentStressor
    assert RuAccentStressor._count_vowels("крст") == 0


def test_count_vowels_yo():
    from stressonnx.backends.ruaccent import RuAccentStressor
    assert RuAccentStressor._count_vowels("ёж") == 1


# ---------------------------------------------------------------------------
# make_stressor — offline error paths
# ---------------------------------------------------------------------------

def test_make_stressor_silero_wrong_lang():
    from stressonnx import make_stressor
    with pytest.raises(ValueError, match="does not support language"):
        make_stressor(model="silero", lang="kk")


def test_make_stressor_ruaccent_wrong_lang():
    from stressonnx import make_stressor
    with pytest.raises(ValueError, match="does not support language"):
        make_stressor(model="ruaccent", lang="uk")


def test_invalid_notation_raises():
    from stressonnx import stress
    with pytest.raises(ValueError):
        stress("Алматы", "kk", notation="invalid")

