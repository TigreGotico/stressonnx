"""Key and ordinal rules of export/build_be_simple_vocab.py."""
import sys
from pathlib import Path

import pytest

from stressonnx.backends.simple import SimpleStressor

EXPORT = Path(__file__).resolve().parents[1] / "export"


@pytest.fixture(scope="module")
def build():
    sys.path.insert(0, str(EXPORT))
    try:
        import build_be_simple_vocab
    finally:
        sys.path.remove(str(EXPORT))
    return build_be_simple_vocab


@pytest.fixture(scope="module")
def simple():
    return SimpleStressor("be")


VOWELS = set("аоуіэыяеёю")


def test_superset_ordinal_counts_the_non_syllabic_u(build):
    # наўчы+ць: language vowels а ы, stressed ы is the second; the
    # superset counts ў, so the runtime ordinal is 2
    assert build.superset_ordinal("наўчыць", 1, VOWELS) == 2
    assert build.superset_ordinal("воўк", 0, VOWELS) == 0
    with pytest.raises(ValueError):
        build.superset_ordinal("воўк", 1, VOWELS)


def test_entries_fold_to_simple_keys(build, simple):
    entries = {
        "Іосіф": (1, "іосіф"), "іосіф": (1, "іосіф"),
        "зява": (0, "з'ява"), "з’ява": (0, "з’ява"),
        "наўчыць": (1, "наўчыць"),
        "Алена": (0, "алена"),
    }
    vocab, dropped = build.simple_entries(entries, simple, VOWELS)
    assert vocab == {"іосіф": 1, "зява": 0, "наўчыць": 2, "алена": 0}
    assert dropped == {}
    assert simple._usable(vocab) == vocab


def test_form_outside_the_alphabet_is_dropped(build, simple):
    vocab, dropped = build.simple_entries({"ван": (0, "иван")}, simple, VOWELS)
    assert vocab == {} and dropped == {"outside alphabet": 1}


def test_name_folds_under_the_common_word(build, simple):
    # В'етнаму (name) and ве+тнаму (adjective) share the simple key
    vocab, dropped = build.simple_entries(
        {"В’етнаму": (1, "в’етнаму"), "ветнаму": (0, "ветнаму")}, simple, VOWELS)
    assert vocab == {"ветнаму": 0}
    assert dropped == {"name under a common word": 1}


def test_common_words_that_disagree_fail(build, simple):
    with pytest.raises(ValueError):
        build.simple_entries({"зява": (0, "з'ява"), "з’ява": (1, "з’ява")}, simple, VOWELS)


def test_wiktionary_kept_only_outside_the_database(build):
    vocab = {"хлушу": 1}
    counts = build.merge_wiktionary(
        vocab, {"хлушу": 1, "замак": 0, "кніга": 0, "выявы": 1}, {"хлушу", "замак", "выявы"})
    assert vocab == {"хлушу": 1, "кніга": 0}
    assert counts == {"in vocabulary, same": 1, "spelling in GrammarDB without an entry": 2, "kept": 1}
