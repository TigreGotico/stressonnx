"""Smoke tests for SimpleAccentor-backed languages."""
import pytest

# (lang, input, expected) — just verify '+' is inserted somewhere
SIMPLE_CASES = [
    # vocab hit
    ("kaz", "Қазақстан", "Қазақст+ан"),
    ("tat", "Казан", "Каз+ан"),
    ("kir", "Бишкек", "Бишк+ек"),
    ("aze_cyr", "Бакы", "Бак+ы"),
    ("aze_lat", "Bakı", "Bak+ı"),
    ("uzb_cyr", "Тошкент", "Тошк+ент"),
    ("uzb_lat", "Toshkent", "Toshk+ent"),
    ("hye", "Երեւան", "Երև+ան"),
    ("kat", "თბილისი", "თბილ+ისი"),
    ("bak", "Өфе", "+Өфе"),
    ("chv", "Шупашкар", "Шупашк+ар"),
    ("sah", "Дьокуускай", "Дьокуускай"),   # in vocab
    ("erz", "Саранск", "С+аранск"),
    ("mdf", "Саранск", "С+аранск"),
    ("kbd", "Налшык", "Налшык"),           # in vocab
    ("kjh", "Абакан", "Абак+ан"),
    ("tgk", "Душанбе", "Душанб+е"),
    ("udm", "Ижевск", "Ижевск"),           # in vocab
    ("xal", "Элиста", "Элист+а"),
    # bel_simple (vocab + rule path)
    ("bel_simple", "свет", "св+ет"),
]


@pytest.mark.parametrize("lang,inp,expected", SIMPLE_CASES)
def test_stress_simple(lang, inp, expected):
    from stressonnx import stress
    result = stress(inp, lang)
    # Check that a stress token was inserted (or word unchanged if no vowels/no vocab)
    assert "+" in result or "+" not in expected, (
        f"[{lang}] expected '+' in result for {inp!r}, got {result!r}"
    )


def test_stress_simple_matches_silero():
    """Exact match against SimpleAccentor for a sample of langs."""
    from stressonnx import stress
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
        ref = SimpleAccentor(lang=lang)(sentence)
        got = stress(sentence, lang)
        assert ref == got, f"[{lang}] ref={ref!r} got={got!r}"
