"""Decomposed (NFD) input stresses the same as its composed (NFC) form.

й is и + U+0306 and ё is е + U+0308 in NFD.  The expected output is derived
from the NFC call, so the test checks that the normalization form of the
input does not change the answer.
"""
import unicodedata

import pytest

from stressonnx import STRESS_TOKEN, analyze, stress

TEXT = "мой новый большой музей. ещё ёж и ёлка"
CASES = [("ru", "ruaccent"), ("ru", "silero"), ("ru", "simple"), ("uk", "silero")]


def nfd(s):
    return unicodedata.normalize("NFD", s)


def nfc(s):
    return unicodedata.normalize("NFC", s)


@pytest.mark.parametrize("lang,model", CASES)
def test_nfd_input_gets_the_nfc_answer(lang, model):
    expected = stress(TEXT, lang, model=model)
    got = stress(nfd(TEXT), lang, model=model)
    assert nfc(got) == expected


@pytest.mark.parametrize("lang,model", CASES)
def test_nfd_input_comes_back_nfd(lang, model):
    got = stress(nfd(TEXT), lang, model=model)
    assert got == nfd(got)


@pytest.mark.parametrize("lang,model", [("ru", "simple"), ("uk", "silero")])
def test_nfd_output_minus_marks_is_the_input(lang, model):
    # these backends never substitute е→ё
    text = nfd(TEXT)
    assert stress(text, lang, model=model).replace(STRESS_TOKEN, "") == text


def stressed_letter(word):
    """The stressed letter of *word* with its combining marks, in NFC."""
    if word.stressed_index is None:
        return None
    end = word.stressed_index + 1
    while end < len(word.text) and unicodedata.combining(word.text[end]):
        end += 1
    return nfc(word.text[word.stressed_index:end])


@pytest.mark.parametrize("lang,model", CASES)
def test_analyze_nfd_offsets_point_at_the_same_vowels(lang, model):
    composed = analyze(TEXT, lang, model=model)
    decomposed = analyze(nfd(TEXT), lang, model=model)
    want = [(w.text, stressed_letter(w)) for w in composed.words]
    got = [(nfc(w.text), stressed_letter(w)) for w in decomposed.words]
    assert got == want
