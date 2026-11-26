"""Yakut OOV words take final stress (#142).

ru Wikipedia, «Якутский язык» (oldid 156229264): stress always falls on the
last syllable and does not depend on diphthongs or vowel length.  The `sah`
rule stresses the last long vowel or diphthong instead, with no source.
"""
import pytest

import stressonnx


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason="https://github.com/TigreGotico/stressonnx/issues/142")
def test_an_oov_word_with_a_diphthong_takes_final_stress():
    # уоттарга is not in the simple-sah vocabulary
    assert stressonnx.stress("уоттарга", "sah", model="simple") == "уоттарга́"
