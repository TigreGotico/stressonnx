"""ruaccent on a sentence longer than the 512-token limit of its models.

Without razdel a whole paragraph is one sentence.  The expected marks come
from stressing the short sentence on its own.
"""
from stressonnx import stress

SHORT = "Старинный город стоит на высокой горе над рекой. "
HOMOGRAPH = "Старинный замок стоит на высокой горе над рекой. "




def test_every_repetition_is_stressed_past_512_tokens():
    expected = stress(SHORT, "ru")
    out = stress(SHORT * 80, "ru")
    assert out == expected * 80


def test_long_sentence_with_a_homograph_does_not_crash():
    expected = stress(HOMOGRAPH, "ru")
    assert "за́мок" in expected.lower()
    out = stress(HOMOGRAPH * 80, "ru")
    assert out == expected * 80
