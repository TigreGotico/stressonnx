"""stressrnn: the StressRNN character BiLSTM as a selectable Russian model.

The expected strings are the output of upstream StressRNN 0.2.2
(``StressRNN().put_stress(text, stress_symbol="+", accuracy_threshold=0.75)``
on Python 3.8 with onnxruntime 1.19.2 and pymorphy2 0.9.1), with each ``+``
after a vowel turned into U+0301 after that vowel.  They are parity values,
not gold stress: upstream stresses monosyllables, prepositions and hyphen
particles, and ``го́ре`` for ``горе`` is upstream's reading.

``STRESSONNX_TEST_CACHE_DIR`` points the stressor at a local model tree
(``<dir>/ru_stressrnn/<file>``) so the bundle can be tested before it is
published; unset, the bundle comes from the pinned Hugging Face revision.
"""
import os
import unicodedata

import pytest

from stressonnx import DEFAULT_MODEL, MODEL_REGISTRY, StressRnnStressor, make_stressor
from stressonnx.backends.stressrnn import _tokenize
from stressonnx.pipeline import FALLBACK_PRIORITY, _PREFER_ORDERS

CACHE_DIR = os.environ.get("STRESSONNX_TEST_CACHE_DIR")

CASES = [
    ("Проставь, пожалуйста, ударения", "Проста́вь, пожа́луйста, ударе́ния"),
    ("старинный замок стоит на горе", "стари́нный за́мок стои́т на́ го́ре"),
    ("Мама мыла раму. Какой-нибудь кто-то пришёл-таки.",
     "Ма́ма мы́ла ра́му. Како́й-нибу́дь кто́-то́ пришё́л-таки́."),
    ("В лесу родилась ёлочка, в лесу она росла",
     "В лесу́ родила́сь ё́лочка, в лесу́ она́ росла́"),
    ("Дойдете сами? Купили каких-нибудь экзотермов вчера.",
     "Дойде́те са́ми? Купи́ли каки́х-нибу́дь экзоте́рмов вчера́."),
    ("Я иду домой, а ты?", "Я́ иду́ домо́й, а́ ты́?"),
    ("% земельных дол", "% земе́льных до́л"),
    ("1 января 188", "1 января́ 188"),
    ("10 минут зах", "10 мину́т за́х"),
]
CASE_IDS = ["readme", "zamok", "hyphens", "yolochka", "doydete", "mono", "pct", "jan", "minut"]


@pytest.fixture(scope="module")
def stressrnn():
    return make_stressor(model="stressrnn", lang="ru", cache_dir=CACHE_DIR)


@pytest.mark.parametrize("text,expected", CASES, ids=CASE_IDS)
def test_matches_upstream(stressrnn, text, expected):
    assert stressrnn(text) == expected


def test_registry_entry():
    entry = MODEL_REGISTRY["stressrnn"]
    assert entry.langs == frozenset({"ru"})
    assert entry.family == "stressrnn"
    assert entry.hf_subdir == "ru_stressrnn"


def test_never_selected_by_default_or_prefer():
    assert DEFAULT_MODEL["ru"] == "ruaccent-silero"
    assert "stressrnn" not in FALLBACK_PRIORITY
    for order in _PREFER_ORDERS.values():
        assert "stressrnn" not in order


def test_make_stressor_returns_stressrnn():
    s = make_stressor(model="stressrnn", lang="ru")
    assert isinstance(s, StressRnnStressor)
    assert s.accuracy_threshold == 0.75


def test_make_stressor_rejects_other_language():
    with pytest.raises(ValueError, match="does not support language"):
        make_stressor(model="stressrnn", lang="uk")


def test_threshold_zero_marks_low_confidence_words(stressrnn):
    confident = stressrnn("ния красноярские ави")
    assert confident == "ния красноя́рские ави́"
    every = StressRnnStressor(cache_dir=CACHE_DIR, accuracy_threshold=0.0)
    assert every("ния красноярские ави") == "ния́ красноя́рские ави́"


def test_punctuation_digits_and_latin_pass_through(stressrnn):
    assert stressrnn("Привет, мир!  «Да»… 42 — ok; ё/Ё") == "Приве́т, ми́р!  «Да́»… 42 — ok; ё́/Ё́"


def test_keeps_every_input_character(stressrnn):
    text = "Привет,\tмир!  «Да»… 42 — ok; ё/Ё\n+ъ'"
    out = stressrnn(text)
    assert out.replace("́", "") == text
    assert out.startswith("Приве́т,")


def test_underscore_does_not_silence_later_words(stressrnn):
    assert stressrnn("а_б дом привет") == "а́_б до́м приве́т"


def test_dictionary_consonant_entry_falls_to_the_network(stressrnn):
    text = "щегольков либо"
    marks, _ = stressrnn.mark_offsets(text)
    assert all(text[o] in "аеиоуэюяыё" for o in marks), [text[o] for o in marks]
    assert 18 not in marks


def test_premarked_word_is_left_alone(stressrnn):
    assert stressrnn("за́мок стоит") == "за́мок стои́т"


def test_decomposed_input_comes_back_decomposed(stressrnn):
    nfd = unicodedata.normalize("NFD", "ёлочка росла")
    out = stressrnn(nfd)
    assert unicodedata.is_normalized("NFD", out)
    assert unicodedata.normalize("NFC", out) == "ё́лочка росла́"


def test_mark_offsets_point_at_vowels(stressrnn):
    text = "старинный замок"
    marks, yo = stressrnn.mark_offsets(text)
    assert yo == []
    assert [text[o] for o in marks] == ["и", "а"]
    assert marks == [4, 11]


def test_tokenize_round_trips():
    text = "Как-то 10 раз, «ok»!\n+ё"
    tokens = _tokenize(text)
    assert "".join(t for _, t in tokens) == text
    assert [t for _, t in tokens] == ["Как", "-", "то", " ", "10", " ", "раз", ",", " ",
                                      "«", "o", "k", "»!", "\n", "+", "ё"]
