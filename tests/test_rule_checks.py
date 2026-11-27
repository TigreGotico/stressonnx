"""Each OOV rule against the claim of its cited source, on words outside the vocabulary."""
import pytest

from stressonnx import stress

ACUTE = "́"


@pytest.mark.parametrize("lang, word", [
    # McCollum & Chen 2021, JIPA: no evidence for final word stress in Kazakh
    ("kk", "қаламсаптар"),
    ("kk", "мектептеріміз"),
    # Санжеев 1940: Kalmyk has no stress of the Russian kind
    ("xal", "бичгүдмүдин"),
])
def test_rule_none_leaves_oov_words_unmarked(lang, word):
    assert stress(word, lang) == word


@pytest.mark.parametrize("word, expected", [
    # Ghaffarvand Mokari & Werner 2017, JIPA: stress mostly falls on the final syllable
    ("kitabxanaçılar", "kitabxanaçıla\u0301r"),
    ("qələmdanlar", "qələmdanla\u0301r"),
])
def test_azerbaijani_oov_final_stress(word, expected):
    assert stress(word, "az-Latn") == expected
