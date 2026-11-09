"""Hyphenated по- adverbs: the prefix carries no stress, and по-мо́ему stresses the pronoun on its first vowel."""
import re

import pytest

from stressonnx import stress
from stressonnx._common import _RE_RU_COND, tokenize

_UK = re.compile("[^абвгґдеєжзиіїйклмнопрстуфхцчшщьюяАБВГҐДЕЄЖЗИІЇЙКЛМНОПРСТУФХЦЧШЩЬЮЯ]")


def test_po_prefix_is_masked_in_ru_and_uk():
    assert tokenize("По-русски", _RE_RU_COND, lang="ru").predict == [False, True]
    assert tokenize("по-українськи", _UK, lang="uk").predict == [False, True]
    # the prefix only, not a final по
    assert tokenize("что-по", _RE_RU_COND, lang="ru").predict == [True, True]


@pytest.mark.parametrize("model", ["silero", "simple"])
def test_po_moemu_stresses_the_pronoun_on_its_first_vowel(model):
    assert stress("по-моему, По-твоему, ПО-МОЕМУ", "ru", model=model) == "по-мо́ему, По-тво́ему, ПО-МО́ЕМУ"
    # the dative on its own keeps its stress
    assert stress("моему брату", "ru", model=model) == "моему́ бра́ту"
