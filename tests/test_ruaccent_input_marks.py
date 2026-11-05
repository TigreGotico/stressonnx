"""ruaccent and the marks and letters already in its input.

Expected stress for plain Russian words comes from the RUAccent accent
dictionary file, read here directly.  A word written with ё takes its stress
on the ё: Russian orthography always stresses ё, and ruaccent labels such a
word STRESS.  The mark position itself comes from the dictionary or the
character model, so this is a convention the tests rely on, not a rule the
code enforces.
"""
import gzip
import json
import unicodedata

import pytest

from stressonnx import STRESS_TOKEN, analyze, stress
from stressonnx.download import _download_files
from stressonnx.pipeline import _align_marks
from stressonnx.registry import MODEL_REGISTRY


def nfd(text: str) -> str:
    return unicodedata.normalize("NFD", text)


@pytest.fixture(scope="module")
def accents():
    files = _download_files(
        MODEL_REGISTRY["ruaccent"].hf_subdir,
        ["dictionary/accents_nn.json.gz"], None, model_id="ruaccent",
    )
    return json.load(gzip.open(files["dictionary/accents_nn.json.gz"]))


def _from_dictionary(accents, word: str) -> str:
    plus = accents[word.lower()]
    vowel = plus.index("+")
    return word[:vowel + 1] + STRESS_TOKEN + word[vowel + 1:]


def _on_yo(word: str) -> str:
    yo = word.lower().index("ё")
    return word[:yo + 1] + STRESS_TOKEN + word[yo + 1:]


def test_acute_on_a_consonant_stays_and_its_word_stays_unmarked(accents):
    # Montenegrin з́ and с́ have no precomposed form; the models read зема
    # and секира, which are other words
    text = "з́ема с́екира молоко"
    expected = "з́ема с́екира " + _from_dictionary(accents, "молоко")
    assert stress(text, "ru", model="ruaccent") == expected
    assert stress(nfd(text), "ru", model="ruaccent") == nfd(expected)


@pytest.mark.parametrize("text,expected", [
    ("куќа Gdańsk", "куќа Gdańsk"),
    ("Ки́їв", "Київ"),
    ("Ње́гош", "Његош"),
])
def test_acute_letters_and_dropped_stress_marks(text, expected):
    assert stress(text, "ru", model="ruaccent") == expected


@pytest.mark.parametrize("text", ["желтый", "жёлтый", "жё́лтый", "Жё́лтый"])
def test_yo_word_is_stressed_on_the_yo_however_it_is_written(text):
    expected = _on_yo(text.replace(STRESS_TOKEN, "").replace("е", "ё"))
    assert stress(text, "ru", model="ruaccent") == expected
    assert stress(nfd(text), "ru", model="ruaccent") == nfd(expected)


@pytest.mark.parametrize("form", ["NFC", "NFD"])
@pytest.mark.parametrize("text", [
    "Желтый",
    "все еще идет дождь",
    "зеленый лес шумит",
    "мой котенок пьет молоко",
    "самолет летит высоко",
    "теплый прием",
    "кр́ов и дом",
])
def test_stressing_twice_gives_the_same_text(text, form):
    once = stress(unicodedata.normalize(form, text), "ru", model="ruaccent")
    assert stress(once, "ru", model="ruaccent") == once


def test_soft_hyphen_does_not_hide_a_dropped_letter(accents):
    # the models read молоко, but the word is молоко + soft hyphen + і
    text = "молоко­і корова"
    expected = "молоко­і " + _from_dictionary(accents, "корова")
    assert stress(text, "ru", model="ruaccent") == expected
    text = "корова і­молоко"
    expected = _from_dictionary(accents, "корова") + " і­молоко"
    assert stress(text, "ru", model="ruaccent") == expected


@pytest.mark.parametrize("original,marked,expected", [
    # restored ё: зеленый → зелёный, offset 3 holds the е
    ("зеленый", "зелё́ный", ([3], [3])),
    ("зеленый", "зелё́ный", ([3], [3])),
    # ё already in the input is not a restoration
    (nfd("жёлтый"), nfd("жё́лтый"), ([1], [])),
    ("жёлтый", "жё́лтый", ([1], [])),
])
def test_align_marks_finds_yo_in_both_forms(original, marked, expected):
    assert _align_marks(original, marked) == expected


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_analyze_reports_restored_yo_in_both_forms(form):
    # зеленый is written with е for зелёный; жёлтый already has its ё
    text = unicodedata.normalize(form, "зеленый лес и жёлтый дом")
    words = analyze(text, "ru", model="ruaccent").words
    assert [w.yo_restored for w in words] == [True, False, False, False, False]


@pytest.mark.parametrize("text,expected", [
    # the words without ё keep the marks of origin/dev and of upstream RUAccent
    (
        "Таким образом, жёсткость была наделена особой святостью.",
        "Таки́м о́бразом, жё́сткость была́ наделена́ осо́бой свя́тостью.",
    ),
    (
        "Пробовал он играть и на скачках, купив самый дешёвый билет.",
        "Про́бовал он игра́ть и на ска́чках, купи́в са́мый дешё́вый биле́т.",
    ),
])
def test_yo_in_a_sentence_does_not_move_the_marks_of_other_words(text, expected):
    assert stress(text, "ru", model="ruaccent") == expected
