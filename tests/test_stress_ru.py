"""End-to-end tests for Russian stress via RuAccentStressor (requires network on first run)."""
import pytest

# Basic accentuation cases — output is combining acute (U+0301) after stressed vowel
CASES = [
    ("Привет мир", "Приве́т мир"),
    ("Замок на горе котёнок", "За́мок на горе́ котёнок"),
    ("Молоко убежало", "Молоко́ убежа́ло"),
]

# Homograph disambiguation cases — the core motivation for the RUAccent backend.
# замок: за́мок = castle (на горе), замо́к = lock (дверной)
# мука: муко́ = flour (для хлеба), му́ка = torment (невыносима)
# белок: бело́к = egg-white / protein (яйца), бе́лок = squirrel-genitive (белка)
HOMOGRAPH_CASES = [
    ("старинный замок стоит на горе", "стари́нный за́мок сто́ит на горе́"),
    ("дверной замок надёжен", "дверно́й замо́к надёжен"),
    ("белок яйца полезен", "бело́к яйца́ поле́зен"),
    ("мука для хлеба", "мука́ для хле́ба"),
    ("мука была невыносима", "му́ка была́ невыноси́ма"),
    # коса́ 'scythe/braid' per Зализняк; the mermaid-braid variant is a
    # known model miss shown honestly in examples/homograph_demo.py
    ("острая коса на лугу", "о́страя коса́ на лугу́"),
]


@pytest.mark.parametrize("inp,expected", CASES)
def test_stress_ru(inp, expected):
    from stressonnx import stress
    assert stress(inp, "ru") == expected


@pytest.mark.parametrize("inp,expected", HOMOGRAPH_CASES)
def test_stress_ru_homographs(inp, expected):
    """RUAccent homograph disambiguation smoke tests."""
    from stressonnx import stress
    assert stress(inp, "ru") == expected


def test_ruaccent_stressor_class():
    from stressonnx import RuAccentStressor
    s = RuAccentStressor()
    result = s("привет мир")
    # Combining acute U+0301 must be present
    assert "́" in result


def test_ru_default_is_ruaccent():
    """ru must be routed to RuAccentStressor by default, not the silero backend."""
    from stressonnx import DEFAULT_MODEL, RUACCENT_LANGS
    assert DEFAULT_MODEL["ru"] == "ruaccent"
    assert "ru" in RUACCENT_LANGS


def test_ukr_bel_unaffected():
    """uk and be still use the main_accentor (silero) pipeline."""
    from stressonnx import MAIN_LANGS
    assert "uk" in MAIN_LANGS
    assert "be" in MAIN_LANGS


def test_to_plus_notation_roundtrip():
    """to_plus_notation converts combining-acute back to + notation."""
    from stressonnx import stress, to_plus_notation
    diacritic = stress("привет", "ru")
    assert "́" in diacritic
    plus_form = to_plus_notation(diacritic)
    assert "+" in plus_form
    assert "́" not in plus_form


def test_notation_plus_kwarg():
    """stress(..., notation='plus') returns legacy + form."""
    from stressonnx import stress
    result = stress("привет", "ru", notation="plus")
    assert "+" in result
    assert "́" not in result
