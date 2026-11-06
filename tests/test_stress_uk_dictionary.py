"""Ukrainian silero: forms of the ukrainian-word-stress dictionary win over the model.

The expected values of the dictionary tests are the output of
ukrainian-word-stress 2.1.0 (lang-uk, MIT) in dictionary mode for forms the
neural model alone stresses differently.  ``STRESSONNX_TEST_CACHE_DIR``
points the stressor at a local model tree (``<dir>/ukr/<file>``) so a bundle
can be tested before it is published; unset, the bundle comes from the
pinned Hugging Face revision.
"""
import hashlib
import json
import os

import pytest

from stressonnx import make_stressor
from stressonnx.download import _download_files
from stressonnx.registry import hf_dir

CACHE_DIR = os.environ.get("STRESSONNX_TEST_CACHE_DIR")


@pytest.fixture(scope="module")
def silero_uk():
    return make_stressor(model="silero", lang="uk", cache_dir=CACHE_DIR)


def test_layer_was_built_against_the_bundled_model():
    """The layer holds only the forms this accentor.onnx stresses differently."""
    files = _download_files(hf_dir("uk", "silero"), ["accentor.onnx", "meta.json"], CACHE_DIR)
    with open(files["meta.json"], encoding="utf-8") as fh:
        built_against = json.load(fh)["exceptions_source"]["accentor_onnx_sha256"]
    with open(files["accentor.onnx"], "rb") as fh:
        assert built_against == hashlib.sha256(fh.read()).hexdigest()


@pytest.mark.parametrize("text,expected", [
    ("були", "були́"),
    ("навіть", "на́віть"),
    ("просто", "про́сто"),
    ("проти", "про́ти"),
    ("людей", "люде́й"),
    ("багатьох", "багатьо́х"),
    ("влада", "вла́да"),
    ("потім", "по́тім"),
])
def test_dictionary_form_wins(silero_uk, text, expected):
    assert silero_uk(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("перемир'я", "переми́р'я"),
    ("перемир’я", "переми́р’я"),
    ("перемирʼя", "переми́рʼя"),
])
def test_dictionary_form_with_any_apostrophe(silero_uk, text, expected):
    assert silero_uk(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("перемиря", "переми́ря"),
    ("попамятаємо", "попамята́ємо"),
    ("'перемир'я'", "'переми́р'я'"),
    ("пере\u00adмир'я", "пере\u00adми́р'я"),
])
def test_dictionary_form_keeps_its_letter_in_every_spelling(silero_uk, text, expected):
    assert silero_uk(text) == expected


def test_every_entry_indexes_a_vowel_of_its_key(silero_uk):
    silero_uk._ensure_loaded()
    entries = silero_uk._exceptions
    assert len(entries) > 90_000
    assert [k for k, (stress, _) in entries.items() if k[stress] not in silero_uk._vowels] == []


@pytest.mark.parametrize("text,expected", [
    ("Навіть були проти.", "На́віть були́ про́ти."),
    ("НАВІТЬ", "НА́ВІТЬ"),
    ("Влада", "Вла́да"),
])
def test_dictionary_form_keeps_case_in_a_sentence(silero_uk, text, expected):
    assert silero_uk(text) == expected


# ukrainian-word-stress lists only йо́го and ко́му for these spellings; the
# Wiktionary vocabulary of the uk simple model stresses the pronouns його́ and
# кому́, so the forms stay out of the layer and the model keeps them.
@pytest.mark.parametrize("text,expected", [
    ("його", "його́"),
    ("кому", "кому́"),
])
def test_form_wiktionary_contradicts_stays_with_the_model(silero_uk, text, expected):
    assert silero_uk(text) == expected
