"""Yakut words in the simple-sah vocabulary take final stress, like OOV words.

Böhtlingk 1851, Einleitung p. XXVII: "Der Ton ruht auf der letzten Silbe."
The Hub vocabulary marks vowel length, so the simple model does not use it.
"""
import stressonnx


def test_a_vocabulary_word_takes_final_stress():
    assert stressonnx.stress("тэлэбиисэрэ", "sah", model="simple") == "тэлэбиисэрэ́"
