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


def test_stress_simple_matches_silero():
    """Exact match against SimpleAccentor for a sample of langs (plus notation)."""
    from stressonnx import stress, to_plus_notation
    from silero_stress.simple_accentor import SimpleAccentor

    sample = {
        "kaz": "Сәлем Қазақстан",
        "tat": "Сәлам Казан",
        "kat": "გამარჯობა თბილისი",
        "hye": "Բարեւ Երեւան",
        "aze_lat": "Salam Bakı",
        "uzb_lat": "Salom Toshkent",
        "sah": "Дорообо Дьокуускай",
    }
    for lang, sentence in sample.items():
        ref = SimpleAccentor(lang=lang)(sentence)  # silero emits + notation
        got = to_plus_notation(stress(sentence, lang))  # convert to + for comparison
        assert ref == got, f"[{lang}] ref={ref!r} got={got!r}"
