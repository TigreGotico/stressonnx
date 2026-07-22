"""Exact-output tests for the Wiktionary/RUAccent-dictionary languages.

Stress values verified against the source lexicons (English Wiktionary
stressed headwords; RUAccent pronunciation dictionary) and each language's
documented rule (citations in ``stressonnx/backends/simple.py``).
"""
import pytest

from stressonnx import stress
from stressonnx.backends.simple import SimpleStressor


@pytest.mark.parametrize("text,lang,model,expected", [
    # Bulgarian — free stress, vocabulary-driven (Scatton 1984)
    ("водата е студена", "bg", None, "вода́та е́ студена"),
    ("Добро утро", "bg", None, "Добро́ у́тро"),
    # Macedonian — fixed antepenultimate (Friedman 2001); OOV words follow it
    ("Добро утро Македонија", "mk", None, "До́бро у́тро Македо́нија"),
    ("телевизија работи", "mk", None, "телеви́зија ра́боти"),
    # Slovene — free stress, vocabulary-driven (Herrity 2000)
    ("voda je mrzla", "sl", None, "vóda jé mrzlá"),
    # Latvian — fixed initial stress (Nau 1998)
    ("Labdien, mani draugi", "lv", None, "Lábdien, máni dráugi"),
    ("saule spīd debesīs", "lv", None, "sáule spī́d débesīs"),
    # ru_simple / ukr_simple — dictionary lookup, no positional guessing
    ("вода холодная", "ru", "simple", "вода́ холо́дная"),
    ("вода холодна", "uk", "simple", "вода́ холо́дна"),
    ("Привіт, як справи сьогодні", "uk", "simple", "Приві́т, я́к спра́ви сього́дні"),
])
def test_new_language_sentences(text, lang, model, expected):
    assert stress(text, lang, model=model) == expected


def test_free_stress_langs_do_not_guess_oov():
    """A multi-vowel word absent from the vocabulary must stay unmarked."""
    for lang, word in [("bg", "студена"), ("sl", "prijatelji"),
                       ("ru", "абракадабрит"), ("uk", "абракадабрить")]:
        s = SimpleStressor(lang)
        s._ensure_loaded()
        assert s._accentuate_oov(word) == word


def test_macedonian_oov_antepenult():
    s = SimpleStressor("mk")
    s._ensure_loaded()
    assert s._accentuate_oov("библиотекарка") == "библиоте́карка"


def test_vocab_sizes_sanity():
    """Vocabularies exist and have the expected order of magnitude."""
    expected_min = {"bg": 40000, "mk": 1500, "sl": 4000,
                    "lv": 100, "ru": 100000, "uk": 45000}
    for lang, n in expected_min.items():
        s = SimpleStressor(lang)
        s._ensure_loaded()
        assert len(s._vocab) >= n, (lang, len(s._vocab))
