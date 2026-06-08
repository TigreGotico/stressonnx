"""End-to-end smoke tests for Ukrainian and Belarusian neural accentors."""
import pytest

UKR_CASES = [
    ("Привіт світ", "Прив+іт св+іт"),
    ("Молоко втекло", "Молоко втекл+о"),
    ("Голова болить", "Голов+а бол+ить"),
]

BEL_CASES = [
    ("Прывітанне свет", "Прывіт+анне св+ет"),
    ("Малако ўцякло", "Малак+о ўцякл+о"),
    ("Вада цячэ", "Вад+а цяч+э"),
]


@pytest.mark.parametrize("inp,expected", UKR_CASES)
def test_stress_ukr(inp, expected):
    from stressonnx import stress
    assert stress(inp, "ukr") == expected


@pytest.mark.parametrize("inp,expected", BEL_CASES)
def test_stress_bel(inp, expected):
    from stressonnx import stress
    assert stress(inp, "bel") == expected
