"""Parity tests against the reference silero_stress library (requires the
``export`` extra: torch + silero_stress).  Run by the dedicated CI job, not
the default matrix."""
from stressonnx import stress, to_plus_notation


def test_stress_simple_matches_silero():
    """Exact match against SimpleAccentor for a sample of langs (plus notation)."""
    from stressonnx import stress, to_plus_notation
    from silero_stress.simple_accentor import SimpleAccentor

    # (stressonnx tag, silero_stress code).  Georgian is deliberately
    # absent: for OOV words stressonnx applies the antepenultimate rule
    # (Akhvlediani/Gudava/Aronson), which the upstream vocabulary itself
    # follows on 100% of its entries, while upstream's own OOV fallback
    # contradicts its vocabulary on 55% of them — see tests/test_oov_rules.py
    # and benchmarks/oov_rules_eval.py.
    sample = {
        ("kk", "kaz"): "Сәлем Қазақстан",
        ("tt", "tat"): "Сәлам Казан",
        ("hy", "hye"): "Բարեւ Երեւան",
        ("az-Latn", "aze_lat"): "Salam Bakı",
        ("uz-Latn", "uzb_lat"): "Salom Toshkent",
        ("sah", "sah"): "Дорообо Дьокуускай",
    }
    for (lang, ref_code), sentence in sample.items():
        ref = SimpleAccentor(lang=ref_code)(sentence)  # silero emits + notation
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
