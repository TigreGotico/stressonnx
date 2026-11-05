"""The bounded context cleaning of the silero homosolver equals cleaning the
whole text, as upstream does."""
from hypothesis import given, settings
from hypothesis import strategies as st

import stressonnx.backends._silero_homosolver as H

# letters, digits and every character class the cleaning rules act on
_ALPHABET = st.sampled_from(list("абвЖЁё zZ09 .!?,-—;:«»\t\n  😀") + ["  ", "--", "!!", ",,"])


@settings(max_examples=400, deadline=None)
@given(st.lists(_ALPHABET, max_size=120).map("".join),
       st.integers(min_value=0, max_value=120), st.integers(min_value=1, max_value=12))
def test_bounded_context_equals_whole_text_cleaning(text, cut, size):
    cut = min(cut, len(text))
    span = H._CONTEXT_SPAN
    H._CONTEXT_SPAN = 3  # force short tails and heads, and the regrowth
    try:
        before = H._context_before(text, cut, size)
        after = H._context_after(text, cut, size)
    finally:
        H._CONTEXT_SPAN = span
    assert before == H._clean_context(text[:cut], is_start=True)[-size:]
    assert after == H._clean_context(text[cut:], is_start=False)[:size]
