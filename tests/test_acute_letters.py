"""A U+0301 that is part of a letter is not a stress mark.

Macedonian ѓ and ќ decompose to г/к + U+0301, and Latin ś ć ń ź to s/c/n/z +
U+0301.  Only a U+0301 after a vowel marks stress.  The letters must pass
through every notation unchanged and must not stop the word from being
stressed.

Expected Macedonian outputs come from the antepenultimate rule (Friedman
2001, "Macedonian"; ``rule: antepenult`` in ``mk.json``): stress the third
vowel from the end, the first vowel in words with fewer than three vowels.
The vowels are ``аеиоу`` (``mk.json``).  None of the words below is in the
Macedonian vocabulary, so the rule alone decides.
"""
import unicodedata

import pytest

from stressonnx import STRESS_TOKEN, analyze, stress, to_plus_notation
from stressonnx.pipeline import _align_marks


def nfd(s):
    return unicodedata.normalize("NFD", s)


# word, vowels (by mk.json), stressed vowel by the rule, expected output
MK_WORDS = [
    ("куќа", "уа", "у", "ку́ќа"),                  # 2 vowels → first
    ("ноќ", "о", "о", "но́ќ"),                     # sole vowel
    ("ѓавол", "ао", "а", "ѓа́вол"),                # 2 vowels → first
    ("куќичка", "уиа", "у", "ку́ќичка"),            # 3 vowels → antepenult = first
    ("среќава", "еаа", "е", "сре́ќава"),            # 3 vowels → antepenult = first
    ("плаќање", "аае", "а", "пла́ќање"),            # 3 vowels → antepenult = first
    ("среќавање", "еаае", "а", "среќа́вање"),       # 4 vowels → antepenult = second
    ("ѓевречиња", "ееиа", "е", "ѓевре́чиња"),       # 4 vowels → antepenult = second
]

SENTENCE = "Ѓорѓи и Ќирил"
SENTENCE_MARKED = "Ѓо́рѓи и́ Ќи́рил"   # о of о-и, sole и, и of и-и
SENTENCE_PLUS = "Ѓ+орѓи +и Ќ+ирил"


@pytest.mark.parametrize("word,vowels,stressed,expected", MK_WORDS)
def test_mk_word_with_acute_letter_is_stressed(word, vowels, stressed, expected):
    assert "".join(c for c in word if c in "аеиоу") == vowels
    assert expected[expected.index(STRESS_TOKEN) - 1] == stressed
    assert stress(word, "mk") == expected


@pytest.mark.parametrize("word,vowels,stressed,expected", MK_WORDS)
def test_mk_word_with_acute_letter_nfd(word, vowels, stressed, expected):
    assert stress(nfd(word), "mk") == nfd(expected)


def test_mk_sentence_diacritic():
    assert stress(SENTENCE, "mk") == SENTENCE_MARKED


def test_mk_sentence_plus():
    assert stress(SENTENCE, "mk", notation="plus") == SENTENCE_PLUS


def test_mk_sentence_nfd_diacritic():
    assert stress(nfd(SENTENCE), "mk") == nfd(SENTENCE_MARKED)


def test_mk_sentence_nfd_plus():
    # the letters stay decomposed: Г + U+0301, К + U+0301
    assert stress(nfd(SENTENCE), "mk", notation="plus") == nfd(SENTENCE_PLUS)


@pytest.mark.parametrize("text", [SENTENCE, nfd(SENTENCE)])
def test_mk_analyze_points_at_the_vowels(text):
    result = analyze(text, "mk")
    # NFC: Ѓ о / и / Ќ и; NFD adds one code point for the acute of Ѓ and Ќ
    offset = 0 if text == SENTENCE else 1
    assert [w.stressed_index for w in result.words] == [1 + offset, 0, 1 + offset]
    assert [w.stressed_char for w in result.words] == ["о", "и", "и"]


@pytest.mark.parametrize("text", ["ку́ќа", nfd("ку́ќа")])
def test_mk_existing_stress_is_kept_and_converted(text):
    # the word already carries a stress on у: it is left as it is, and the
    # plus form moves only that mark
    assert stress(text, "mk") == text
    assert stress(text, "mk", notation="plus") == text.replace("у́", "+у")


@pytest.mark.parametrize("text,expected", [
    ("Ѓорѓи", "Ѓорѓи"),
    ("ќ ѓ Ќ Ѓ", "ќ ѓ Ќ Ѓ"),
    ("ку́ќа", "к+уќа"),
    ("Gdańsk", "Gdańsk"),
    ("ćma źle śnieg", "ćma źle śnieg"),
    ("Gdáńsk", "Gd+ańsk"),
    ("ćmá śnié", "ćm+a śni+e"),
])
def test_plus_notation_keeps_acute_letters(text, expected):
    assert to_plus_notation(text) == expected
    assert to_plus_notation(nfd(text)) == nfd(expected)


def test_latin_word_with_acute_letter_is_stressed():
    # a is the sole vowel of Gdańsk: a single-vowel word takes the stress on it
    assert stress("Gdańsk", "az-Latn") == "Gda" + STRESS_TOKEN + "ńsk"
    assert stress(nfd("Gdańsk"), "az-Latn") == nfd("Gda" + STRESS_TOKEN + "ńsk")


@pytest.mark.parametrize("original,marked,marks", [
    (nfd("Gdańsk"), nfd("Gdańsk"), []),
    (nfd("Gdańsk"), "Gda" + STRESS_TOKEN + nfd("ńsk"), [2]),
    (nfd("куќа"), "ку" + STRESS_TOKEN + nfd("ќа"), [1]),
])
def test_align_marks_skips_acute_letters(original, marked, marks):
    assert _align_marks(original, marked) == (marks, [])


@pytest.mark.parametrize("lang", ["ru", "uk"])
def test_silero_word_with_acute_letter_is_stressed(lang):
    out = stress("Ќирил", lang, model="silero")
    # exactly one stress mark, on one of the two и, and Ќ intact
    assert out in ("Ќи́рил", "Ќири́л")


# Slovene marks stress on a syllabic r with an acute: SSKJ² on fran.si gives
# the headword vŕh and the phrase "čéšnja na vŕhu tórte".  The acute on r is
# a stress mark, so vŕhu is already stressed and keeps its one mark.
SL_PHRASE = "čéšnja na vŕhu tórte"
# na is a one-vowel word, so the backend stresses its a with a combining acute
SL_MARKED = "čéšnja na" + STRESS_TOKEN + " vŕhu tórte"
SL_PLUS = "č+ešnja n+a v+rhu t+orte"


def test_sl_syllabic_r_acute_is_stress():
    assert stress(SL_PHRASE, "sl") == SL_MARKED
    assert stress(SL_PHRASE, "sl", notation="plus") == SL_PLUS


def test_sl_syllabic_r_acute_is_stress_nfd():
    assert stress(nfd(SL_PHRASE), "sl") == nfd(SL_MARKED)
    assert stress(nfd(SL_PHRASE), "sl", notation="plus") == nfd(SL_PLUS)


@pytest.mark.parametrize("text,plus", [
    ("vŕh", "v+rh"),
    (nfd("vŕh"), "v+rh"),
    ("ср́це", "с+рце"),
])
def test_plus_notation_moves_syllabic_r_stress(text, plus):
    assert to_plus_notation(text) == plus


def test_mk_cyrillic_syllabic_r_acute_is_stress():
    assert stress("ср́це", "mk") == "ср́це"


def test_align_marks_stress_after_another_combining_mark():
    # ё in NFD is е + U+0308; the stress mark follows the diaeresis and
    # belongs to the base letter е at offset 0
    assert _align_marks("ёж", "ё" + STRESS_TOKEN + "ж") == ([0], [])
