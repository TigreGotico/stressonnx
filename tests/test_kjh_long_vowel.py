import pytest

import stressonnx

# Баскаков 1975, Грамматика хакасского языка, §49, p. 43.
@pytest.mark.parametrize("word, expected", [
    ("хызыл", "хызы́л"),
    ("хузурух", "хузуру́х"),
    ("хара", "хара́"),
    ("чорга", "чорга́"),
    ("хараа", "хараа́"),
    ("чоорған", "чоо́рған"),
])
def test_long_stem_vowel_takes_the_stress_else_the_last_syllable(word, expected):
    assert stressonnx.stress(word, "kjh", model="simple") == expected
