"""Tests for the kubataba Russian backend (downloads the bundle on first run).

Expected strings come from the upstream torch package (ruaccent-predictor
1.2.0, ``load_accentor()(text, format="apostrophe")``) with each ``'``
rewritten as U+0301 after the vowel.  The model is sentence-level and marks
neither ё nor single-vowel words.
"""
import unicodedata

import pytest

from stressonnx import (
    DEFAULT_MODEL,
    FALLBACK_PRIORITY,
    MODEL_REGISTRY,
    KubatabaStressor,
    analyze,
    make_stressor,
    stress,
)
from stressonnx.pipeline import StressPipeline

CASES = [
    ("привет мир", "приве́т мир"),
    ("молоко убежало", "молоко́ убежа́ло"),
    ("красивый город", "краси́вый го́род"),
    ("хлеб и вода", "хлеб и вода́"),
    ("Россия великая страна", "Росси́я вели́кая страна́"),
    ("В лесу родилась ёлочка", "В лесу́ родила́сь ёлочка"),
    ("Я купил замок и повесил его на замок.", "Я купи́л за́мок и пове́сил его́ на за́мок."),
    ("кто-то пришёл", "кто́-то пришёл"),
    ("зелёный замок", "зелёный за́мок"),
]


@pytest.mark.parametrize("inp,expected", CASES)
def test_kubataba_matches_upstream(inp, expected):
    assert stress(inp, "ru", model="kubataba") == expected


def test_kubataba_keeps_characters_the_model_rewrites():
    # upstream generates "A земе'льных дол" for this window: the model has
    # no "%" and invents a character; the output keeps the input's "%"
    assert stress("% земельных дол", "ru", model="kubataba") == "% земе́льных дол"


def test_kubataba_drops_marks_on_invented_text():
    # upstream generates "ких докуме'нтах во' Го'нв": the invented "Го'нв"
    # shares one "о" with "воо" and must not give that word a second mark
    assert stress("ких документах воо", "ru", model="kubataba") == "ких докуме́нтах во́о"


def test_kubataba_windows_cut_on_any_whitespace():
    from stressonnx.backends.kubataba import _windows
    text = "\n".join(["слово"] * 60)
    chunks = [chunk for _, chunk in _windows(text)]
    assert "".join(chunks) == text
    assert len(chunks) == 2
    assert all(len(c) <= 200 for c in chunks)
    assert all(set(c.split()) == {"слово"} for c in chunks)


def test_kubataba_leaves_text_without_prediction_unchanged():
    assert stress("Hello мир 123", "ru", model="kubataba") == "Hello мир 123"


def test_kubataba_skips_words_already_marked():
    assert stress("молоко́ убежало", "ru", model="kubataba") == "молоко́ убежа́ло"


def test_kubataba_nfd_in_nfd_out():
    inp = unicodedata.normalize("NFD", "зелёный замок")
    out = stress(inp, "ru", model="kubataba")
    assert out == unicodedata.normalize("NFD", "зелёный за́мок")


def test_kubataba_long_text_is_windowed():
    sentence = "Я купил замок и повесил его на замок."
    marked = "Я купи́л за́мок и пове́сил его́ на за́мок."
    text = " ".join([sentence] * 12)
    assert len(text) > 254
    assert stress(text, "ru", model="kubataba") == " ".join([marked] * 12)


def test_kubataba_plus_notation():
    assert stress("привет мир", "ru", model="kubataba", notation="plus") == "прив+ет мир"


def test_kubataba_analyze_spans():
    result = analyze("привет мир", "ru", model="kubataba")
    assert result.marked == "приве́т мир"
    assert [(w.text, w.stressed_index) for w in result.words] == [("привет", 4), ("мир", None)]


def test_kubataba_registry_entry():
    entry = MODEL_REGISTRY["kubataba"]
    assert entry.langs == frozenset({"ru"})
    assert entry.family == "kubataba"
    assert entry.hf_subdir == "ru_kubataba"


def test_kubataba_never_selected_by_default_prefer_or_fallback():
    assert DEFAULT_MODEL["ru"] == "ruaccent"
    assert "kubataba" not in FALLBACK_PRIORITY
    pipeline = StressPipeline()
    for prefer in ("best", "fast", "smallest"):
        assert pipeline._resolve("ru", None, prefer)[2] != "kubataba"
    assert "kubataba" not in [m for m, _ in pipeline._chain("ru")]


def test_kubataba_make_stressor():
    s = make_stressor(model="kubataba", lang="ru")
    assert isinstance(s, KubatabaStressor)
    assert isinstance(make_stressor(model="kubataba"), KubatabaStressor)


def test_kubataba_wrong_lang_raises():
    with pytest.raises(ValueError, match="does not support"):
        make_stressor(model="kubataba", lang="uk")
