"""Basic end-to-end tests for Russian stress (requires network on first run)."""
import pytest

CASES = [
    ("Привет мир", "Прив+ет м+ир"),
    ("Замок на горе котёнок", "З+амок на гор+е кот+ёнок"),
    ("Молоко убежало", "Молок+о убеж+ало"),
]


@pytest.mark.parametrize("inp,expected", CASES)
def test_stress_ru(inp, expected):
    from stressonnx import stress
    assert stress(inp, "ru") == expected


def test_stressor_class():
    from stressonnx import Stressor
    s = Stressor("ru")
    assert "+" in s("Привет")
