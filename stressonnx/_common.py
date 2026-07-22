"""Private low-level helpers shared by several backends."""
import re

import numpy as np

# Russian-specific vowel set used for the full accentuate pipeline.
_RU_VOWELS = "аоуыэиеяёю"

# Word-boundary characters shared by every tokenizing backend.  Extends the
# upstream silero_stress set (\s.,!?;:<>=()/\\) with typographic punctuation
# and digits so that glued tokens («дом», текст—текст, дом5) are split into a
# clean word plus punctuation instead of falling through to OOV handling.
# Apostrophes are deliberately NOT boundaries: ' is part of the aze_lat /
# uzb_lat alphabets and ’ of the bel alphabet.
_RE_SPLIT = re.compile(r'([\s.,!?;:<>=()/\\«»„“”"…—–%№*@\[\]{}0-9]+)')
_RE_RU_COND = re.compile(r"[^А-Яа-яёЁ]")

# Russian hyphenated enclitic particles that never carry word stress
# (кто́-то, како́й-нибудь, кто́-либо, пришёл-таки, скажи́-ка) — the part after
# the hyphen is masked out of stress prediction.  See e.g. Русская
# грамматика (АН СССР, 1980) §§ on particles; matches upstream silero_stress
# behavior for "-то" and extends it to the remaining standard clitics.
_UNSTRESSED_HYPHEN_CLITICS = frozenset({"то", "нибудь", "либо", "таки", "ка"})


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)
