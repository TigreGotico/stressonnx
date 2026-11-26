"""Accent parsing of Sloleks forms.

Expected stresses are the SSKJ headwords (Slovar slovenskega knjižnega jezika,
fran.si): otròk, kolêno, mêsto, lopáta, vŕh.
"""
import pytest
from gold.sloleks_slovene import stress_ordinal


@pytest.mark.parametrize(("form", "want"), [
    ("otròk", ("otrok", 1)),
    ("kolêno", ("koleno", 1)),
    ("mêsto", ("mesto", 0)),
    ("lopáta", ("lopata", 1)),
])
def test_the_one_mark_gives_the_vowel_ordinal(form, want):
    assert stress_ordinal(form) == want


@pytest.mark.parametrize(("form", "reason"), [
    ("vŕh", "mark on the syllabic r or a consonant (rows)"),
    ("hiša", "no mark or more than one mark (rows)"),
    ("nádstrôpje", "no mark or more than one mark (rows)"),
    ("pès", "one vowel letter (rows)"),
])
def test_a_form_the_set_cannot_score_is_dropped(form, reason):
    assert stress_ordinal(form) == (None, reason)
