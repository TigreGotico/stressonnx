"""ruaccent gives back every character of its input.

The models read a normalized copy of the text, but the output is the input
itself with stress marks added (and е→ё restored).  Expected stress for the
Russian words comes from the RUAccent accent dictionary file, read here
directly.
"""
import gzip
import json
import unicodedata

import pytest
from hypothesis import given, settings, strategies as st

from stressonnx import STRESS_TOKEN, stress
from stressonnx.download import _download_files
from stressonnx.registry import MODEL_REGISTRY

_RU_VOWELS = set("аеёиоуыэюяАЕЁИОУЫЭЮЯ")


def _unstressed(out: str, text: str) -> str:
    """*out* without added stress marks and with restored ё written as the
    input wrote it, compared in NFC so a decomposed Latin é keeps its acute."""
    out = unicodedata.normalize("NFC", out)
    text = unicodedata.normalize("NFC", text)
    marks = [i for i, c in enumerate(out) if c == STRESS_TOKEN]
    for i in marks:
        assert i > 0 and out[i - 1] in _RU_VOWELS, (out, i)
    bare = out.replace(STRESS_TOKEN, "")
    if len(bare) != len(text):
        return bare
    return "".join(
        t if b in "ёЁ" and t in "еЕ" else b for b, t in zip(bare, text)
    )


CASES = [
    "Він живе в Києві, а я в Москве.",
    "Беларуская мова: ў і ї є ґ — в русском тексте.",
    "Кафе «Café» и ресторан über alles, señor.",
    "Греческий: αλφάβητο, китайский: 中文 и эмодзи 😀 тут.",
    "Многоточие… и знаки № 5, § 3, 50% и $10.",
    "Неразрывный пробел и слово ( в скобках ) .",
    "Он сказал : « да » , и ушёл .",
    "Тильда ~ остаётся тильдой.",
    "Два  пробела,\tтаб и\nперевод строки…",
    unicodedata.normalize("NFD", "Кафе «Café», ёжик и йогурт; Київ."),
]


@pytest.mark.parametrize("text", CASES)
def test_output_is_input_plus_marks(text):
    out = stress(text, "ru", model="ruaccent")
    assert _unstressed(out, text) == unicodedata.normalize("NFC", text)


def test_decomposed_input_comes_back_decomposed():
    text = unicodedata.normalize("NFD", "Кафе «Café» и ресторан.")
    out = stress(text, "ru", model="ruaccent")
    assert unicodedata.is_normalized("NFD", out)
    assert "Café" in unicodedata.normalize("NFC", out)


_MIXED = st.text(
    alphabet=(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАЕКМОРСТ"
        "іїєґўІЇЄҐЎ" "aeéüñçAÉ" "αλφβΩ" "中文字" "😀👍"
        "«»„“”‘’—–-~…№§%$ ,.!?:;() \t\n"
    ),
    max_size=40,
)


@settings(max_examples=60, deadline=None)
@given(_MIXED, st.booleans())
def test_no_character_is_lost(text, decompose):
    if decompose:
        text = unicodedata.normalize("NFD", text)
    out = stress(text, "ru", model="ruaccent")
    assert _unstressed(out, text) == unicodedata.normalize("NFC", text)


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


@pytest.mark.parametrize("text", [
    "Корова «Café» даёт молоко — хорошая страна, ў і ґ, 中文 😀…",
    "Корова даёт  молоко\tи\nхорошая   страна ( über ) .",
])
def test_russian_words_keep_their_stress(accents, text):
    expected = text
    for word in ("Корова", "молоко", "хорошая", "страна"):
        expected = expected.replace(word, _from_dictionary(accents, word))
    # ё is always stressed
    expected = expected.replace("даёт", "даё" + STRESS_TOKEN + "т")
    assert stress(text, "ru", model="ruaccent") == expected


def test_word_with_a_letter_the_models_cannot_read_stays_unmarked():
    # the models see Київ as Кив and ґанок as анок
    assert stress("Київ и ґанок", "ru", model="ruaccent") == "Київ и ґанок"


def test_zero_width_space_is_a_word_boundary():
    # UAX #29: U+200B ends a word, so the models (which read
    # "молококорова") did not see either word
    text = "молоко​корова"
    assert stress(text, "ru", model="ruaccent") == text


@pytest.mark.parametrize("joiner", ["­", "‌", "‍", "⁠"])
def test_format_character_inside_a_word_keeps_its_mark(accents, joiner):
    # UAX #29 rule WB4: these sit inside the word they are in
    marked = _from_dictionary(accents, "молоко")
    text = "моло" + joiner + "ко"
    assert stress(text, "ru", model="ruaccent") == marked[:4] + joiner + marked[4:]


@pytest.mark.parametrize("prefix", ["віцэ-", "і’"])
def test_compound_parts_are_checked_on_their_own(accents, prefix):
    # the part with a letter the models cannot read stays unmarked; the
    # Russian part after the hyphen or apostrophe keeps its mark
    out = stress(prefix + "корова", "ru", model="ruaccent")
    assert out == prefix + _from_dictionary(accents, "корова")


def test_input_marks_are_dropped_even_where_none_comes_back():
    assert stress("Ки́їв", "ru", model="ruaccent") == "Київ"
