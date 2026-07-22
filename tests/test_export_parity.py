"""Parity tests against the reference silero_stress library (requires the
``export`` extra: torch + silero_stress).  Run by the dedicated CI job, not
the default matrix."""
from stressonnx import stress, to_plus_notation


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


def test_silero_ru_yo_parity():
    """The ported yo-capable ru decode must match the reference accentor."""
    from silero_stress import load_accentor

    ref = load_accentor("ru")  # full pipeline (homograph-ambiguous words avoided below)

    def plus_to_acute(s):
        out, i = [], 0
        while i < len(s):
            if s[i] == "+" and i + 1 < len(s):
                out.append(s[i + 1] + "\u0301")
                i += 2
            else:
                out.append(s[i])
                i += 1
        return "".join(out)

    sentences = [
        "зеленый лес шумит",
        "теплый прием",
        "мой котенок пьет молоко",
        "самолет летит высоко",
        "красивый город",
        "ёжик в тумане",
    ]
    for s in sentences:
        assert stress(s, "ru", model="silero") == plus_to_acute(ref(s)), s
