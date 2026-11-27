"""kk and xal words take no stress mark, in the vocabulary or out of it.

McCollum & Chen 2021 (JIPA 51(2)) find no final word stress in Kazakh, and
Санжеев 1940 finds no stress of the Russian kind in Kalmyk.
"""
import pytest

import stressonnx


@pytest.mark.parametrize("lang, word", [("kk", "Қазақстан"), ("xal", "аавасн")])
def test_a_vocabulary_word_takes_no_mark(lang, word):
    assert stressonnx.stress(word, lang, model="simple") == word
