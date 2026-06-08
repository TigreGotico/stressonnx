"""End-to-end smoke tests for Ukrainian and Belarusian neural accentors."""
import pytest

# Output uses combining acute (U+0301) placed after the stressed vowel.
UKR_CASES = [
    ("Привіт світ", "Приві́т сві́т"),
    ("Молоко втекло", "Молоко втекло́"),
    ("Голова болить", "Голова́ боли́ть"),
]

BEL_CASES = [
    ("Прывітанне свет", "Прывіта́нне све́т"),
    ("Малако ўцякло", "Малако́ ўцякло́"),
    ("Вада цячэ", "Вада́ цячэ́"),
]


@pytest.mark.parametrize("inp,expected", UKR_CASES)
def test_stress_ukr(inp, expected):
    from stressonnx import stress
    assert stress(inp, "ukr") == expected


@pytest.mark.parametrize("inp,expected", BEL_CASES)
def test_stress_bel(inp, expected):
    from stressonnx import stress
    assert stress(inp, "bel") == expected
