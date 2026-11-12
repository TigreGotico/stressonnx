"""The ruaccent character model marks one vowel per word.

The model labels every character on its own, so a word outside the accent
dictionary can get STRESS_PRIMARY on two vowels.  The word keeps the vowel
with the highest probability.  Expected marks come from the gold rows of the
scoring campaign (ruslan-stressed, Wiktionary and russtress, as the vowel
ordinal the gold marks) or from the accent dictionary file, never from the
model output.
"""
import gzip
import json

import pytest

from stressonnx import STRESS_TOKEN, stress
from stressonnx.backends.ruaccent import RuAccentStressor
from stressonnx.download import _download_files
from stressonnx.registry import MODEL_REGISTRY

VOWELS = "аеёиоуыэюя"


@pytest.fixture(scope="module")
def accents():
    files = _download_files(
        MODEL_REGISTRY["ruaccent"].hf_subdir,
        ["dictionary/accents_nn.json.gz"], None, model_id="ruaccent",
    )
    return json.load(gzip.open(files["dictionary/accents_nn.json.gz"]))


@pytest.fixture(scope="module")
def stressor(model_cache_dir):
    s = RuAccentStressor(cache_dir=model_cache_dir)
    s._ensure_loaded()
    return s


def on_vowel(word: str, ordinal: int) -> str:
    """*word* with the acute after its vowel number *ordinal*."""
    seen = -1
    for i, ch in enumerate(word):
        if ch.lower() in VOWELS:
            seen += 1
            if seen == ordinal:
                return word[:i + 1] + STRESS_TOKEN + word[i + 1:]
    raise ValueError(word)


# (word, gold vowel ordinal); each word is absent from accents_nn.json.gz
GOLD = [
    ("дарья", 0),     # ruslan-stressed, 9 rows; Wiktionary, 10 rows
    ("вела", 1),      # ruslan-stressed, 8 rows; Wiktionary, 4 rows
    ("сумки", 0),     # ruslan-stressed, 5 rows; Wiktionary, 3 rows
    ("австрии", 0),   # ruslan-stressed, 7 rows
    ("стыда", 1),     # ruslan-stressed, 3 rows; Wiktionary, 4 rows
    ("редьки", 0),    # russtress, 1 row
    ("перловую", 1),  # Wiktionary, 1 row
]


@pytest.mark.parametrize("word,ordinal", GOLD)
def test_character_model_marks_the_gold_vowel_once(stressor, accents, word, ordinal):
    assert word not in accents
    assert stressor._put_accent(word) == on_vowel(word, ordinal)


@pytest.mark.parametrize("text,expected", [
    ("Дарья вела сумки.", "Да́рья вела́ су́мки."),
    ("Он сгорел со стыда.", "Он сгоре́л со стыда́."),
])
def test_sentence_words_carry_one_mark_each(accents, text, expected):
    # сгорел is in the dictionary; the other marked words are not
    out = stress(text, "ru", model="ruaccent")
    assert out == expected
    assert all(tok.count(STRESS_TOKEN) <= 1 for tok in out.split())


def test_dictionary_compound_keeps_its_dictionary_mark(stressor, accents):
    # the dictionary holds one mark for this compound; the character model
    # is not consulted
    word = "самолётостроение"
    plus = accents[word]
    vowel = plus.index("+")
    expected = word[:vowel + 1] + STRESS_TOKEN + word[vowel + 1:]
    assert stressor._process_accent([word], ["STRESS"]) == [expected]
