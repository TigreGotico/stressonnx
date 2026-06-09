"""Tests for notation types, conversion helpers, and the _apply_notation boundary."""
import pytest


# ---------------------------------------------------------------------------
# StressNotation enum
# ---------------------------------------------------------------------------

def test_notation_enum_values():
    from stressonnx import StressNotation
    assert StressNotation.DIACRITIC == "diacritic"
    assert StressNotation.PLUS == "plus"


def test_notation_enum_is_str():
    from stressonnx import StressNotation
    assert isinstance(StressNotation.DIACRITIC, str)
    assert isinstance(StressNotation.PLUS, str)


def test_notation_enum_from_string():
    from stressonnx import StressNotation
    assert StressNotation("diacritic") is StressNotation.DIACRITIC
    assert StressNotation("plus") is StressNotation.PLUS


def test_notation_enum_invalid():
    from stressonnx import StressNotation
    with pytest.raises(ValueError):
        StressNotation("invalid")


def test_notation_string_literal_compat():
    """StressNotation members compare equal to their string equivalents."""
    from stressonnx import StressNotation
    assert StressNotation.PLUS == "plus"
    assert StressNotation.DIACRITIC == "diacritic"
    assert "plus" == StressNotation.PLUS


# ---------------------------------------------------------------------------
# to_plus_notation
# ---------------------------------------------------------------------------

def test_to_plus_simple():
    from stressonnx import to_plus_notation
    assert to_plus_notation("приве́т") == "прив+ет"


def test_to_plus_multiple():
    from stressonnx import to_plus_notation
    assert to_plus_notation("стари́нный за́мок") == "стари́нный за́мок".replace("и́", "+и").replace("а́", "+а")
    # explicit expectation
    result = to_plus_notation("за́мок")
    assert result == "з+амок"


def test_to_plus_no_stress():
    from stressonnx import to_plus_notation
    assert to_plus_notation("привет") == "привет"


def test_to_plus_idempotent_on_plus_form():
    """Already-plus text is unchanged (no combining acute present)."""
    from stressonnx import to_plus_notation
    assert to_plus_notation("прив+ет") == "прив+ет"


def test_to_plus_preserves_non_stressed():
    from stressonnx import to_plus_notation
    assert to_plus_notation("мир") == "мир"


def test_to_plus_mixed_script():
    from stressonnx import to_plus_notation
    result = to_plus_notation("Bakı́")
    assert "+" in result
    assert "́" not in result


# ---------------------------------------------------------------------------
# stress() — notation parameter
# ---------------------------------------------------------------------------

def test_stress_diacritic_default():
    from stressonnx import stress
    result = stress("Казан", "tat")
    assert "́" in result
    assert "+" not in result


def test_stress_notation_diacritic_explicit():
    from stressonnx import stress
    result = stress("Казан", "tat", notation="diacritic")
    assert "́" in result


def test_stress_notation_plus():
    from stressonnx import stress
    result = stress("Казан", "tat", notation="plus")
    assert "+" in result
    assert "́" not in result


def test_stress_notation_enum_plus():
    from stressonnx import stress, StressNotation
    result = stress("Казан", "tat", notation=StressNotation.PLUS)
    assert "+" in result


def test_stress_notation_enum_diacritic():
    from stressonnx import stress, StressNotation
    result = stress("Казан", "tat", notation=StressNotation.DIACRITIC)
    assert "́" in result


def test_stress_diacritic_plus_roundtrip():
    """plus form of stress() == to_plus_notation(diacritic form)."""
    from stressonnx import stress, to_plus_notation
    diacritic = stress("Казан", "tat", notation="diacritic")
    plus_direct = stress("Казан", "tat", notation="plus")
    assert to_plus_notation(diacritic) == plus_direct


# ---------------------------------------------------------------------------
# Stressor — notation attribute and output
# ---------------------------------------------------------------------------

def test_stressor_notation_default():
    from stressonnx import Stressor
    s = Stressor(lang="kaz")
    assert s.notation == "diacritic"


def test_stressor_notation_plus():
    from stressonnx import Stressor
    s = Stressor(lang="kaz", notation="plus")
    result = s("Казан")
    assert "+" in result
    assert "́" not in result


def test_stressor_notation_enum():
    from stressonnx import Stressor, StressNotation
    s = Stressor(lang="kaz", notation=StressNotation.PLUS)
    result = s("Казан")
    assert "+" in result
