"""The Pravopis apostrophe parser, with examples from §26–§35 (pp. 27–28)."""
import pytest

from gold.pravopis_macedonian import MARKED, stress_ordinal


@pytest.mark.parametrize("marked, word, ordinal", [
    ("вним'ателно", "внимателно", 1),
    ("'ученик", "ученик", 0),
    ("уч'еникот", "ученикот", 1),
    ("југоз'апад", "југозапад", 2),
    ("со'обраќај", "сообраќај", 1),
])
def test_ordinal_counts_vowels_before_the_mark(marked, word, ordinal):
    assert stress_ordinal(marked) == (word, ordinal)


@pytest.mark.parametrize("marked, reason", [
    ("п'ортр'ет", "more than one mark (words)"),
    ("г'рд", "mark on the syllabic р or a consonant (words)"),
    ("б'ел", "one vowel letter (words)"),
])
def test_drops(marked, reason):
    assert stress_ordinal(marked) == (None, reason)


def test_a_token_in_a_hyphenated_compound_is_not_a_word():
    assert MARKED.findall("ког'о-годе, к'ој-\n-годе, шт'о‑годе и 'однатре") == ["'однатре"]


def test_a_word_before_a_clitic_is_not_a_word():
    from gold.pravopis_macedonian import CLITIC
    text = "нос'ете ја, лет'уваат"
    kept = [m.group() for m in MARKED.finditer(text) if not CLITIC.match(text, m.end())]
    assert kept == ["лет'уваат"]
