"""End-to-end tests for Russian stress via RuAccentStressor (requires network on first run)."""
import pytest

# Basic accentuation cases
CASES = [
    ("Привет мир", "Прив+ет мир"),
    ("Замок на горе котёнок", "З+амок на гор+е котёнок"),
    ("Молоко убежало", "Молок+о убеж+ало"),
]

# Homograph disambiguation cases — the core motivation for the RUAccent backend.
# замок: з+амок = castle (на горе), зам+ок = lock (дверной)
# мука: мук+а = flour (для хлеба), м+ука = torment (невыносима)
# белок: бел+ок = egg-white / protein (яйца), б+елок = squirrel-genitive (белка)
HOMOGRAPH_CASES = [
    ("старинный замок стоит на горе", "стар+инный з+амок ст+оит на гор+е"),
    ("дверной замок надёжен", "дверн+ой зам+ок надёжен"),
    ("белок яйца полезен", "бел+ок яйц+а пол+езен"),
    ("мука для хлеба", "мук+а для хл+еба"),
    ("мука была невыносима", "м+ука был+а невынос+има"),
    ("острая коса на лугу", "+острая к+оса на луг+у"),
    ("коса русалки", "кос+а рус+алки"),
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
    assert "+" in result


def test_ru_not_in_main_langs():
    """ru must be routed to RuAccentStressor, not the silero-derived Stressor."""
    from stressonnx import MAIN_LANGS, RUACCENT_LANGS
    assert "ru" not in MAIN_LANGS
    assert "ru" in RUACCENT_LANGS


def test_ukr_bel_unaffected():
    """ukr and bel still use the main_accentor (silero) pipeline."""
    from stressonnx import MAIN_LANGS
    assert "ukr" in MAIN_LANGS
    assert "bel" in MAIN_LANGS
