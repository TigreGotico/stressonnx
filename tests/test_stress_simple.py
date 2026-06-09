"""Smoke tests for SimpleAccentor-backed languages."""
import pytest

# (lang, input, expected) — output uses combining acute (U+0301) after stressed vowel
SIMPLE_CASES = [
    # vocab hit
    ("kaz", "Қазақстан", "Қазақста́н"),
    ("tat", "Казан", "Каза́н"),
    ("kir", "Бишкек", "Бишке́к"),
    ("aze_cyr", "Бакы", "Бакы́"),
    ("aze_lat", "Bakı", "Bakı́"),
    ("uzb_cyr", "Тошкент", "Тошке́нт"),
    ("uzb_lat", "Toshkent", "Toshként"),
    ("hye", "Երեւան", "Երևа́ն"),
    ("kat", "თბილისი", "თბილი́სი"),
    ("bak", "Өфе", "Өфе́"),
    ("chv", "Шупашкар", "Шупашка́р"),
    ("sah", "Дьокуускай", "Дьокуускай"),   # in vocab (no stress mark if no vowel)
    ("erz", "Саранск", "Са́ранск"),
    ("mdf", "Саранск", "Са́ранск"),
    ("kbd", "Налшык", "Налшык"),           # in vocab
    ("kjh", "Абакан", "Абака́н"),
    ("tgk", "Душанбе", "Душанбе́"),
    ("udm", "Ижевск", "Ижевск"),           # in vocab
    ("xal", "Элиста", "Элиста́"),
    # bel_simple (vocab + rule path)
    ("bel_simple", "свет", "све́т"),
]


@pytest.mark.parametrize("lang,inp,expected", SIMPLE_CASES)
def test_stress_simple(lang, inp, expected):
    from stressonnx import stress
    result = stress(inp, lang)
    # Check that combining acute is present (or word unchanged if no vowels/no vocab)
    assert "́" in result or "́" not in expected, (
        f"[{lang}] expected combining acute in result for {inp!r}, got {result!r}"
    )


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
