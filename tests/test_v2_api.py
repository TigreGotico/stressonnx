"""The v2 surface: pipelines, structured analysis, capability selection."""
import dataclasses
import pickle

import pytest

from stressonnx import (
    StressPipeline,
    StressResult,
    StressedWord,
    analyze,
    stress,
    stress_batch,
)
from stressonnx.langs import load_languages


def test_language_files_complete():
    for tag, spec in load_languages().items():
        assert spec["tag"] == tag
        assert spec["script"] in ("cyrillic", "latin", "armenian", "georgian")
        assert spec["rule"]
        assert spec["sources"], tag
        assert "simple" in spec["hf"], tag


def test_analyze_offsets_and_rendering():
    r = analyze("старинный замок стоит на горе", "ru")
    assert isinstance(r, StressResult)
    assert r.original == "старинный замок стоит на горе"
    words = {w.text: w for w in r.words}
    assert words["замок"].stressed_char == "а"
    assert words["замок"].stressed_index == 1
    assert words["на"].stressed_index is None
    assert r.text == "стари́нный за́мок сто́ит на горе́"


def test_analyze_reports_yo_restoration():
    r = analyze("зеленый лес", "ru", model="silero")
    w = r.words[0]
    assert w.yo_restored and w.stressed_index == 3 and w.text == "зеленый"


def test_prefer_strategies():
    assert stress("привет", "ru", prefer="smallest") == stress("привет", "ru", model="simple")
    assert stress("привет", "ru", prefer="fast") == stress("привет", "ru", model="silero")
    with pytest.raises(ValueError, match="prefer"):
        stress("привет", "ru", prefer="bestest")


def test_batch():
    assert stress_batch(["вода", "молоко"], "ru", model="simple") == ["вода́", "молоко́"]


def test_private_pipeline_is_isolated():
    p = StressPipeline()
    assert p.stress("Қазақстан", "kk") == "Қазақста́н"
    assert ("kk", "simple") in p._singletons
    import stressonnx
    # the private pipeline's cache is not the default pipeline's
    assert p._singletons is not stressonnx._SINGLETONS


def _independent_mark_offsets(marked: str) -> list:
    """Offsets of U+0301 base characters in *marked* with the marks removed."""
    offsets = []
    for i, ch in enumerate(marked):
        if ch == "́":
            offsets.append(i - 1 - len(offsets))
    return offsets


def test_analyze_keeps_every_mark_of_a_hyphenated_compound():
    text = "Қазақстан-Ресей"
    marked = stress(text, "kk")
    assert marked == "Қазақста́н-Ресе́й"
    (w,) = analyze(text, "kk").words
    assert w.stressed_indices == tuple(_independent_mark_offsets(marked)) == (7, 13)
    assert w.stressed_index == 7
    assert [w.text[i] for i in w.stressed_indices] == ["а", "е"]


def test_analyze_compound_marks_per_word():
    r = analyze("северо-западный ветер", "ru", model="silero")
    assert r.text == "се́веро-за́падный ве́тер"
    assert [(w.text, w.stressed_index, w.stressed_indices) for w in r.words] == [
        ("северо-западный", 1, (1, 8)),
        ("ветер", 1, (1,)),
    ]


def test_stressed_word_single_index_construction_still_compares_equal():
    (w,) = analyze("Қазақстан", "kk").words
    assert w == StressedWord("Қазақстан", 0, 9, 7)
    assert StressedWord("на", 0, 2, None).stressed_indices == ()


@pytest.mark.parametrize("index, indices", [(1, (2,)), (None, (2,)), (2, (7, 13))])
def test_stressed_word_rejects_contradictory_indices(index, indices):
    with pytest.raises(ValueError, match="stressed_index"):
        StressedWord("Қазақстан-Ресей", 0, 15, index, False, indices)


def test_stressed_word_replace_keeps_indices_consistent():
    (compound,) = analyze("Қазақстан-Ресей", "kk").words
    with pytest.raises(ValueError):
        dataclasses.replace(compound, stressed_index=13)
    with pytest.raises(ValueError):
        dataclasses.replace(StressedWord("Қазақстан", 0, 9, 7), stressed_index=None)
    cleared = dataclasses.replace(compound, stressed_index=None, stressed_indices=())
    assert (cleared.stressed_index, cleared.stressed_indices) == (None, ())


#: ``pickle.dumps(StressedWord("Қазақстан", 0, 9, 7), protocol=2)`` made by
#: a stressonnx version whose StressedWord had no ``stressed_indices``.
_PICKLE_WITHOUT_STRESSED_INDICES = (
    b"\x80\x02cstressonnx.pipeline\nStressedWord\nq\x00)\x81q\x01}q\x02(X\x04"
    b"\x00\x00\x00textq\x03X\x12\x00\x00\x00\xd2\x9a\xd0\xb0\xd0\xb7\xd0\xb0"
    b"\xd2\x9b\xd1\x81\xd1\x82\xd0\xb0\xd0\xbdq\x04X\x05\x00\x00\x00startq\x05"
    b"K\x00X\x03\x00\x00\x00endq\x06K\tX\x0e\x00\x00\x00stressed_indexq\x07K\x07"
    b"X\x0b\x00\x00\x00yo_restoredq\x08\x89ub."
)


def test_stressed_word_unpickles_state_without_stressed_indices():
    w = pickle.loads(_PICKLE_WITHOUT_STRESSED_INDICES)
    assert w.stressed_indices == (7,)
    assert w == StressedWord("Қазақстан", 0, 9, 7)


def test_stressed_word_pickle_round_trip():
    (compound,) = analyze("Қазақстан-Ресей", "kk").words
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        restored = pickle.loads(pickle.dumps(compound, protocol=protocol))
        assert restored == compound
        assert restored.stressed_indices == (7, 13)
