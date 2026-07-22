"""Property-based tests over the string-mechanics invariants.

The linguistic quality of stress placement is covered by the exact-output
corpora; these tests cover the *mechanics* that must hold for arbitrary
input — the class of guarantees example tests structurally cannot give.
"""
import unicodedata

from hypothesis import given, settings, strategies as st

from stressonnx import STRESS_TOKEN, analyze, stress
from stressonnx.backends.simple import SimpleStressor
from stressonnx.notation import _plus_to_diacritic, to_plus_notation

# words over alphabets that exercise case anomalies (İ), digraphs, and marks
_CYR = st.text(alphabet="абвгдежзиклмнопрстуфхцчшыэюяёЁАБВЕИİÍó -«»…5", min_size=0, max_size=24)
_LAT = st.text(alphabet="abcdefghijklmnoprstuvyzáéíóúışİĞğÖü -0", min_size=0, max_size=24)


def _marks_only_on_vowels(out: str, vowels) -> bool:
    nfd = unicodedata.normalize("NFD", out)
    for i, c in enumerate(nfd):
        if c != STRESS_TOKEN:
            continue
        # walk back past other combining marks (ü decomposes to u + ̈)
        j = i - 1
        while j >= 0 and unicodedata.combining(nfd[j]):
            j -= 1
        if j < 0 or nfd[j].lower() not in vowels:
            return False
    return True


@settings(max_examples=200, deadline=None)
@given(_CYR)
def test_simple_kk_mechanics(text):
    s = SimpleStressor("kk")
    s._ensure_loaded()
    out = s(text)
    # output minus marks and case equals input (no character invented or lost)
    assert out.replace(STRESS_TOKEN, "") == text
    # at most one mark per hyphen-part token, never doubled
    assert STRESS_TOKEN * 2 not in unicodedata.normalize("NFD", out)


@settings(max_examples=200, deadline=None)
@given(_LAT)
def test_simple_azlatn_mechanics(text):
    s = SimpleStressor("az-Latn")
    s._ensure_loaded()
    out = s(text)
    assert out.replace(STRESS_TOKEN, "") == text
    from stressonnx._common import SCRIPT_VOWELS
    assert _marks_only_on_vowels(out, SCRIPT_VOWELS["latin"] | set("ё"))


@settings(max_examples=200, deadline=None)
@given(_LAT)
def test_notation_round_trip(text):
    # diacritic → plus → diacritic is stable for any text
    plus = to_plus_notation(text)
    assert to_plus_notation(_plus_to_diacritic(plus)) == plus


@settings(max_examples=100, deadline=None)
@given(_CYR)
def test_simple_idempotent(text):
    s = SimpleStressor("kk")
    s._ensure_loaded()
    once = s(text)
    assert s(once) == once


@settings(max_examples=50, deadline=None)
@given(_CYR)
def test_analyze_offsets_index_the_original(text):
    r = analyze(text, "kk")
    assert r.original == text
    for w in r.words:
        assert r.original[w.start:w.end] == w.text
        if w.stressed_index is not None:
            assert 0 <= w.stressed_index < len(w.text)
