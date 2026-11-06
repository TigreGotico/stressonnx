"""Belarusian silero: the homograph layer picks the reading the gold set marks.

The sentences and the expected readings are rows of ``10x10.tsv`` in
alex73/benchmarks-stress-bel (CC-BY-SA-4.0, revision 94bb5a86), ten expert
sentences per reading of each homograph; the neural model alone marks the
other reading of these words.  ``STRESSONNX_TEST_CACHE_DIR`` points the
stressor at a local model tree (``<dir>/bel/<file>``) so a bundle can be
tested before it is published; unset, the bundle comes from the pinned
Hugging Face revision.
"""
import os

import pytest

from stressonnx import make_stressor
from stressonnx.backends._homograph_rules import MINED

CACHE_DIR = os.environ.get("STRESSONNX_TEST_CACHE_DIR")


@pytest.fixture(scope="module")
def silero_be():
    return make_stressor(model="silero", lang="be", cache_dir=CACHE_DIR)


def test_bundle_declares_the_homograph_layer(silero_be):
    silero_be._ensure_loaded()
    assert silero_be._homographs is not None
    assert "бацькі" in silero_be._homographs
    for name in MINED:
        assert name in ("agree_ending", "subject_pronoun")


def tokens(marked: str) -> list:
    return [t.strip(".,!?") for t in marked.split()]


# gold row, the marked token the row expects, the rule that reads it
@pytest.mark.parametrize("text,expected,stage", [
    ("У складзе нашай футбольнай каманды адбыліся значныя змены перад пачаткам новага сезона.",
     "скла́дзе", "rule:loc"),
    ("Дызайнер падышоў да стала і расклаў на ім свае новыя чарцяжы.", "стала́", "rule:gen_prep"),
    ("Мы купілі гэты прыгожы і карысны падарунак спецыяльна для майго бацькі.", "ба́цькі", "rule:agree"),
    ("Кавалер пачціва пахіліўся і пацалаваў руку сваёй дамы пасля заканчэння вальса.", "да́мы", "rule:agree"),
    ("Перад пачаткам новай гульні трэба ўважліва перамяшаць усе карты.", "гульні́", "rule:agree_ending"),
    ("Яна вучыцца на медыцынскім факультэце, бо з дзяцінства марыць стаць доктарам.",
     "ву́чыцца", "rule:subject_pronoun"),
    ("Ён любіць гуляць у шахматы па вечарах са сваім суседам.", "лю́біць", "rule:subject_pronoun"),
])
def test_rule_picks_the_gold_reading(silero_be, text, expected, stage):
    out = silero_be(text)
    assert expected in tokens(out), out
    silero_be._ensure_loaded()
    _, clean, _ = silero_be._tokenize(text)
    words = [w for w in clean if w]
    target = expected.replace("́", "")
    assert silero_be._homographs.resolve(words, words.index(target))[1] == stage


@pytest.mark.parametrize("text,expected", [
    ("Аб баках перамоў мы мала што ведалі.", "бака́х"),
    ("Па баках гэтай машыны тырчэлі нейкія дзіўныя штукі.", "бака́х"),
])
def test_prior_picks_the_frequent_reading_without_a_rule(silero_be, text, expected):
    """бака́х (locative plural of бо́к) is the only reading of the prior corpora."""
    out = silero_be(text)
    assert expected in tokens(out), out
    silero_be._ensure_loaded()
    _, clean, _ = silero_be._tokenize(text)
    words = [w for w in clean if w]
    assert silero_be._homographs.resolve(words, words.index("баках")) == (1, "prior")


def test_tied_prior_leaves_the_word_with_the_model(silero_be):
    """вучыцца without a subject pronoun or a modal: the corpora tie 3 to 3."""
    silero_be._ensure_loaded()
    assert silero_be._homographs.resolve(["добра", "вучыцца"], 1) == (None, None)
    assert silero_be("добра вучыцца") == "до́бра вучы́цца"


def test_dictionary_layer_still_wins_before_the_rules(silero_be):
    silero_be._ensure_loaded()
    assert "эканаміста" in silero_be._dictionary and "эканаміста" not in silero_be._homographs
    assert silero_be("для эканаміста") == "для́ эканамі́ста"


def test_apostrophe_spellings_find_the_entry(silero_be):
    silero_be._ensure_loaded()
    assert "зявіцца" in silero_be._homographs and "з’явіцца" in silero_be._homographs
    for text in ("яна з'явіцца", "яна з’явіцца", "яна зʼявіцца"):
        _, clean, _ = silero_be._tokenize(text)
        words = [w for w in clean if w]
        assert silero_be._homographs.resolve(words, 1)[1] == "rule:subject_pronoun"
