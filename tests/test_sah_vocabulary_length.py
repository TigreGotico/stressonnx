"""Yakut words in the simple-sah vocabulary take final stress, like OOV words.

Böhtlingk 1851, Einleitung p. XXVII: "Der Ton ruht auf der letzten Silbe."
The vocabulary marks the second letter of a long vowel or diphthong instead.
"""
import pytest

import stressonnx


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="https://github.com/TigreGotico/stressonnx/issues/151")
def test_a_vocabulary_word_takes_final_stress():
    assert stressonnx.stress("тэлэбиисэрэ", "sah", model="simple") == "тэлэбиисэрэ́"
