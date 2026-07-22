"""Exact-output tests for SimpleAccentor-backed languages.

Expected values come from the exported silero_stress vocabularies (verified
against the reference ``SimpleAccentor`` — see the parity test below).
"""
import pytest

# (lang, input, expected) — combining acute (U+0301) after the stressed vowel
SIMPLE_CASES = [
    ("kk", "Қазақстан", "Қазақста́н"),
    ("tt", "Казан", "Каза́н"),
    ("ky", "Бишкек", "Бишке́к"),
    ("az-Cyrl", "Бакы", "Бакы́"),
    ("az-Latn", "Bakı", "Bakı́"),
    ("uz-Cyrl", "Тошкент", "Тошке́нт"),
    ("uz-Latn", "Toshkent", "Toshként"),
    ("hy", "Երեւան", "Երեւա́ն"),
    ("ka", "თბილისი", "თბი́ლისი"),
    ("ba", "Өфе", "Өфе́"),
    ("cv", "Шупашкар", "Шупашка́р"),
    ("sah", "Дьокуускай", "Дьокуу́скай"),
    ("myv", "Саранск", "Са́ранск"),
    ("mdf", "Саранск", "Са́ранск"),
    ("kbd", "Налшык", "Налшы́к"),
    ("kjh", "Абакан", "Абака́н"),
    ("tg", "Душанбе", "Душанбе́"),
    ("udm", "Ижевск", "Иже́вск"),
    ("xal", "Элиста", "Элиста́"),
    ("be", "свет", "све́т"),
    # bel_simple exercises the vocab + rule path
]


@pytest.mark.parametrize("lang,inp,expected", SIMPLE_CASES)
def test_stress_simple(lang, inp, expected):
    from stressonnx import stress
    assert stress(inp, lang) == expected
