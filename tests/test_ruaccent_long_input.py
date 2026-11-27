"""ruaccent on a sentence longer than the 512-token limit of its models.

Without razdel a whole paragraph is one sentence.  The expected marks come
from stressing the short sentences on their own.
"""
from stressonnx import stress

# no word of FILLER is in the homograph dictionary, so the label models
# alone decide it; 80 copies overflow their 512-token limit
FILLER = "Дети читали новые книги в библиотеке. "
HOMOGRAPH = "Дверной замок надёжен."


def test_every_repetition_is_stressed_past_512_tokens():
    expected = stress(FILLER, "ru")
    out = stress(FILLER * 80, "ru")
    assert out == expected * 80


def test_long_sentence_with_a_homograph_does_not_crash():
    expected = stress(FILLER, "ru") * 80 + stress(HOMOGRAPH, "ru")
    assert stress(FILLER * 80 + HOMOGRAPH, "ru") == expected


def test_homograph_window_shrinks_to_fit_long_words():
    # each filler word is 7 omograph tokens, so the full 129-word window
    # is about 900 tokens and must be halved to fit 512
    filler = "сверхвысокочастотный " * 70
    expected = stress("дверной замок надёжен", "ru").split()[1]
    out = stress(filler + "дверной замок надёжен " + filler, "ru")
    assert out.split()[71] == expected


def test_sentence_within_the_window_is_scored_whole():
    # a homograph at the start of a 100-word sentence still sees the
    # sentence end: the window shifts instead of shrinking
    tail = " ".join(["слово"] * 96 + ["конец"])
    from stressonnx.backends import ruaccent
    seen = []
    stressor = ruaccent.RuAccentStressor()
    stressor._ensure_loaded()
    real = stressor._omograph_tok.encode

    def spy(txt, pair=None):
        seen.append(txt)
        return real(txt, pair=pair)

    stressor._omograph_tok = type("Spy", (), {"encode": staticmethod(spy)})()
    stressor("замок стоит. " + tail)
    assert seen and all("конец" in t for t in seen)
