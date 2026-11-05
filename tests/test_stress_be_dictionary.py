"""Belarusian silero: forms of the GrammarDB dictionary layer win over the model.

The expected values are the stressed forms of GrammarDB RELEASE-202601
(CC-BY-SA-4.0), read from the release's own ``spellchecker/slovy-2008-stress.txt``
and ``<Form>`` texts, for spellings the neural model alone stresses
differently.  ``STRESSONNX_TEST_CACHE_DIR`` points the stressor at a local
model tree (``<dir>/bel/<file>``) so a bundle can be tested before it is
published; unset, the bundle comes from the pinned Hugging Face revision.
"""
import hashlib
import json
import os
import unicodedata

import pytest

from stressonnx import make_stressor
from stressonnx.download import _download_files
from stressonnx.registry import hf_dir

CACHE_DIR = os.environ.get("STRESSONNX_TEST_CACHE_DIR")


@pytest.fixture(scope="module")
def silero_be():
    return make_stressor(model="silero", lang="be", cache_dir=CACHE_DIR)


def test_layer_was_built_against_the_bundled_pipeline():
    """The layer holds only the forms this accentor and exceptions table get wrong."""
    files = _download_files(hf_dir("be", "silero"),
                            ["accentor.onnx", "exceptions.txt.gz", "meta.json"], CACHE_DIR)
    with open(files["meta.json"], encoding="utf-8") as fh:
        source = json.load(fh)["dictionary_source"]
    assert source["license"] == "CC-BY-SA-4.0"
    for name, key in (("accentor.onnx", "accentor_onnx_sha256"),
                      ("exceptions.txt.gz", "exceptions_sha256")):
        with open(files[name], "rb") as fh:
            assert source[key] == hashlib.sha256(fh.read()).hexdigest()


@pytest.mark.parametrize("text,expected", [
    ("эканаміста", "эканамі́ста"),
    ("хлушу", "хлушу́"),
    ("аднаўляецца", "аднаўля́ецца"),
    ("пастаўшчыкоў", "пастаўшчыко́ў"),
    ("белуджаў", "белу́джаў"),
])
def test_dictionary_form_wins(silero_be, text, expected):
    assert silero_be(text) == expected


# GrammarDB writes the ASCII apostrophe; the silero alphabet keeps ’ and
# drops the others, so every spelling must find the entry
@pytest.mark.parametrize("text,expected", [
    ("з'ядка", "з'я́дка"),
    ("з’ядка", "з’я́дка"),
    ("зʼядка", "зʼя́дка"),
    ("раз'ятрывае", "раз'я́трывае"),
    ("раз’ятрывае", "раз’я́трывае"),
    ("разʼятрывае", "разʼя́трывае"),
])
def test_dictionary_form_with_any_apostrophe(silero_be, text, expected):
    assert silero_be(text) == expected


@pytest.mark.parametrize("text,expected", [
    ("аднаўляецца", "аднаўля́ецца"),
    ("пастаўшчыкоў", "пастаўшчыко́ў"),
])
def test_dictionary_form_from_decomposed_input(silero_be, text, expected):
    """NFD input (ў as у + U+0306) keeps the layer's vowel and the input's letters."""
    nfd = unicodedata.normalize("NFD", text)
    assert nfd != text
    assert unicodedata.normalize("NFC", silero_be(nfd)) == expected


@pytest.mark.parametrize("text,expected", [
    ("Эканаміста хлушу.", "Эканамі́ста хлушу́."),
    ("ЭКАНАМІСТА", "ЭКАНАМІ́СТА"),
    ("Эканаміста", "Эканамі́ста"),
    ("'эканаміста'", "'эканамі́ста'"),
    ("«эканаміста»", "«эканамі́ста»"),
])
def test_dictionary_form_keeps_case_and_quotes(silero_be, text, expected):
    """The stored vowel ordinal lands on the right vowel whatever the token carries."""
    assert silero_be(text) == expected


def test_homograph_stays_with_the_model(silero_be):
    """GrammarDB lists за́мак and зама́к, so the key is not in the layer."""
    silero_be._ensure_loaded()
    assert "замак" not in silero_be._dictionary
    assert silero_be("замак") == "за́мак"
