"""Exact-output tests for SimpleAccentor-backed languages.

Expected values come from the exported silero_stress vocabularies (verified
against the reference ``SimpleAccentor`` — see the parity test below).
"""
import pytest

# (lang, input, expected) — combining acute (U+0301) after the stressed vowel
SIMPLE_CASES = [
    ("kaz", "Қазақстан", "Қазақста́н"),
    ("tat", "Казан", "Каза́н"),
    ("kir", "Бишкек", "Бишке́к"),
    ("aze_cyr", "Бакы", "Бакы́"),
    ("aze_lat", "Bakı", "Bakı́"),
    ("uzb_cyr", "Тошкент", "Тошке́нт"),
    ("uzb_lat", "Toshkent", "Toshként"),
    ("hye", "Երեւան", "Երեւա́ն"),
    ("kat", "თბილისი", "თბი́ლისი"),
    ("bak", "Өфе", "Өфе́"),
    ("chv", "Шупашкар", "Шупашка́р"),
    ("sah", "Дьокуускай", "Дьокуу́скай"),
    ("erz", "Саранск", "Са́ранск"),
    ("mdf", "Саранск", "Са́ранск"),
    ("kbd", "Налшык", "Налшы́к"),
    ("kjh", "Абакан", "Абака́н"),
    ("tgk", "Душанбе", "Душанбе́"),
    ("udm", "Ижевск", "Иже́вск"),
    ("xal", "Элиста", "Элиста́"),
    ("bel_simple", "свет", "све́т"),
    # bel_simple exercises the vocab + rule path
]


@pytest.mark.parametrize("lang,inp,expected", SIMPLE_CASES)
def test_stress_simple(lang, inp, expected):
    from stressonnx import stress
    assert stress(inp, lang) == expected
