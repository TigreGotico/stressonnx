"""OOV positional rules — grammar-sourced example words per language.

Sources are cited in ``SimpleStressor._accentuate_oov``; the words below are
the examples those sources themselves use (transliterated to the working
orthography where the source quotes IPA/Latin).
"""
import pytest

from stressonnx.backends.simple import SimpleStressor


def oov(lang, word):
    s = SimpleStressor(lang)
    s._ensure_loaded()
    return s._accentuate_oov(word)


# Chuvash: "stress falls on last full vowel; if a word has only reduced
# vowels, stress falls on the first vowel" (Dobrovolsky 1999; Clark 1998).
# ӑ and ӗ are the reduced vowels.
@pytest.mark.parametrize("word,expected", [
    ("вӑрман", "вӑрма́н"),     # last full vowel а (Dobrovolsky: vărman 'forest')
    ("ҫирӗм", "ҫи́рӗм"),       # final ӗ reduced → stress и (sírĕm '20')
    ("шӑпчӑк", "шӑ́пчӑк"),     # all vowels reduced → first (šắpčăk 'nightingale')
    ("кӑнтӑр", "кӑ́нтӑр"),     # all vowels reduced → first (kắndăr 'south')
])
def test_chuvash_reduced_vowel_rule(word, expected):
    assert oov("chv", word) == expected


# Georgian: antepenultimate for words of 3+ syllables, initial otherwise
# (Akhvlediani 1949 / Gudava 1969 / Aronson 1990 tradition).
@pytest.mark.parametrize("word,expected", [
    ("კომპიუტერები", "კომპიუტე́რები"),  # 5 vowels → antepenult
    ("წიგნი", "წი́გნი"),                # 2 vowels → initial
])
def test_georgian_antepenultimate(word, expected):
    assert oov("kat", word) == expected


# Eastern Armenian: "stress occurs within the last non-schwa syllable"
# (Chakmakjian 2024) — ը never carries stress.
@pytest.mark.parametrize("word,expected", [
    ("գիրքը", "գի́րքը"),    # definite article -ը unstressed → stress retracts
    ("մարդիկ", "մարդի́կ"),  # plain final stress
])
def test_armenian_schwa_skip(word, expected):
    assert oov("hye", word) == expected


# Tajik: final stress; word-final -и is the unstressed izafet enclitic,
# stressed final /i/ is written ӣ (Perry 2005 §1.6).
@pytest.mark.parametrize("word,expected", [
    ("китоби", "кито́би"),   # izafet: китоби ман 'my book'
    ("моҳӣ", "моҳӣ́"),       # final ӣ IS stressed ('fish')
    ("китоб", "кито́б"),
])
def test_tajik_izafet(word, expected):
    assert oov("tgk", word) == expected


# Moksha: first syllable, but "stress is often assigned to a non-initial
# syllable if it contains /a/ and the initial syllable has a high vowel"
# (Hamari & Ajanki 2022 §23.2.3).
@pytest.mark.parametrize("word,expected", [
    ("кудса", "кудса́"),     # initial у (high) → retract to а
    ("мода", "мо́да"),       # initial о is not high → initial stress
])
def test_moksha_high_vowel_retraction(word, expected):
    assert oov("mdf", word) == expected


# Kabardian: final syllable, except word-final schwa э → penult (Jaimoukha).
@pytest.mark.parametrize("word,expected", [
    ("дадэ", "да́дэ"),       # 'grandpa' — Jaimoukha's own example
    ("далун", "далу́н"),     # consonant-final → plain final stress
])
def test_kabardian_final_schwa(word, expected):
    assert oov("kbd", word) == expected


def test_rules_table_prevails_over_meta():
    s = SimpleStressor("chv")
    s._ensure_loaded()
    assert s._oov_rule == "chv"  # meta.json says "last"; local table wins
