"""The ruaccent tokenizers encode the same on tokenizers 0.x and 1.x.

The expected ids and word ids are what tokenizers 0.23 returns from
``Tokenizer.from_file(...).encode(text)`` with its native
``Encoding.word_ids``, the API that 1.x removed.
"""
import pytest

from stressonnx.backends.ruaccent import RuAccentStressor


@pytest.fixture(scope="module")
def stressor():
    stressor = RuAccentStressor()
    stressor._ensure_loaded()
    return stressor


# [CLS] привет [SEP] мир [UNK] кто - то [SEP]: a literal special token and
# [UNK] in the text are words; the [CLS]/[SEP] the template adds are not
SPECIALS_TEXT = "привет [SEP] мир [UNK] кто-то"


def test_stress_usage_word_ids(stressor):
    tok = stressor._stress_usage_tok
    ids, attention_mask = tok.encode(SPECIALS_TEXT)
    assert ids == [2, 32840, 3, 9311, 1, 11352, 17, 1619, 3]
    assert attention_mask == [1] * 9
    assert tok.word_ids(ids) == [None, 0, 1, 2, 3, 4, 5, 6, None]


def test_yo_homograph_word_ids_of_subwords(stressor):
    tok = stressor._yo_hom_tok
    ids, _ = tok.encode(SPECIALS_TEXT)
    assert ids == [2, 4762, 3, 2040, 1, 740, 45, 409, 3]
    assert tok.word_ids(ids) == [None, 0, 1, 2, 3, 4, 5, 6, None]
    # a ## continuation token stays in the word it continues
    ids, _ = tok.encode("ab")
    assert ids == [2, 49, 297, 3]
    assert tok.word_ids(ids) == [None, 0, 0, None]


def test_yo_homograph_tokenizer_deletes_nul(stressor):
    # the vocab keys holding NUL are unreachable: the normalizer deletes
    # NUL before the vocab lookup, so these encode as if it were absent
    tok = stressor._yo_hom_tok
    assert tok.encode("при\x00вет мир") == ([2, 4762, 2040, 3], [1, 1, 1, 1])
    assert tok.encode("a\x00\x00b") == tok.encode("ab")


def test_encode_does_not_truncate(stressor):
    # 600 words, each one token, plus [CLS] and [SEP]
    ids, _ = stressor._stress_usage_tok.encode("слово " * 600)
    assert len(ids) == 602


def test_omograph_pair(stressor):
    ids, attention_mask = stressor._omograph_tok.encode(" <w>замок</w> стоит", pair="з+амок")
    assert ids == [0, 225, 60256, 304, 4512, 60257, 2418, 2, 2, 51981, 2]
    assert attention_mask == [1] * 11


def test_run_without_spaces_past_512_tokens(stressor):
    # each "!" is one word; the head that fits is 512 - [CLS] - [SEP]
    labels = stressor._predict_word_labels(
        "!" * 600, stressor._stress_usage_sess, stressor._stress_usage_tok,
        stressor._stress_id2label,
    )
    assert len(labels) == 510
    assert stressor("!" * 600) == "!" * 600
