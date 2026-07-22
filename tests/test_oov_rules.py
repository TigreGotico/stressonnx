"""OOV positional rules — grammar-sourced example words per language.

Sources are cited in ``SimpleStressor._accentuate_oov``; the words below are
the examples those sources themselves use (transliterated to the working
orthography where the source quotes IPA/Latin).
"""
import pytest

from stressonnx.backends.simple import SimpleStressor


def oov(lang, word):
    """Apply only the positional rule (never the vocabulary) to one word."""
    from stressonnx._common import lower_preserving_length
    from stressonnx.notation import render_marks
    s = SimpleStressor(lang)
    idx = s._rule_offset(lower_preserving_length(word))
    return word if idx is None else render_marks(word, [idx])


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
    assert oov("cv", word) == expected


# Georgian: antepenultimate for words of 3+ syllables, initial otherwise
# (Akhvlediani 1949 / Gudava 1969 / Aronson 1990 tradition).
@pytest.mark.parametrize("word,expected", [
    ("კომპიუტერები", "კომპიუტე́რები"),  # 5 vowels → antepenult
    ("წიგნი", "წი́გნი"),                # 2 vowels → initial
])
def test_georgian_antepenultimate(word, expected):
    assert oov("ka", word) == expected


# Eastern Armenian: "stress occurs within the last non-schwa syllable"
# (Chakmakjian 2024) — ը never carries stress.
@pytest.mark.parametrize("word,expected", [
    ("գիրքը", "գի́րքը"),    # definite article -ը unstressed → stress retracts
    ("մարդիկ", "մարդի́կ"),  # plain final stress
])
def test_armenian_schwa_skip(word, expected):
    assert oov("hy", word) == expected


# Tajik: final stress; word-final -и is the unstressed izafet enclitic,
# stressed final /i/ is written ӣ (Perry 2005 §1.6).
@pytest.mark.parametrize("word,expected", [
    ("китоби", "кито́би"),   # izafet: китоби ман 'my book'
    ("моҳӣ", "моҳӣ́"),       # final ӣ IS stressed ('fish')
    ("китоб", "кито́б"),
])
def test_tajik_izafet(word, expected):
    assert oov("tg", word) == expected


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
    s = SimpleStressor("cv")
    s._ensure_loaded()
    assert s._oov_rule == "chv"  # meta.json says "last"; the language file wins


# Yakut: long vowels / diphthongs (vowel digraphs) attract stress; the mark
# sits on the second element of the heavy nucleus (Krueger 1962; see
# _rule_sah).  Words below are vocabulary-attested spellings.
@pytest.mark.parametrize("word,expected", [
    ("буолан", "буо́лан"),    # diphthong уо in the first syllable wins
    ("эрээри", "эрээ́ри"),    # long ээ wins over final short и
    ("кини", "кини́"),        # no heavy nucleus → final vowel
])
def test_yakut_heavy_nucleus(word, expected):
    assert oov("sah", word) == expected


# Tatar: unstressed suffixes/enclitics retract stress (Comrie 1997b); only
# the vocabulary-validated variants are enabled — see _rule_tat.
@pytest.mark.parametrize("word,expected", [
    ("татарча", "тата́рча"),   # adverbial -ча unstressed
    ("бармыйсың", "бармыйсы́ң"),  # -сың variant NOT enabled (evidence < 0.7)
    ("китапме", "кита́пме"),   # interrogative -ме unstressed
])
def test_tatar_unstressed_suffixes(word, expected):
    assert oov("tt", word) == expected
