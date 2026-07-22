"""Private low-level helpers shared by several backends."""
import re
from typing import NamedTuple

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

#: Per-script vowel supersets used for the vowel-ordinal vocabulary format
#: ("stress the Nth vowel").  The set per script is the union of every
#: supported language's vowel inventory plus loanword vowels observed in the
#: curated vocabularies; the converter in export/convert_vocabs_to_ordinals.py
#: validates every entry against it and reports anything outside.
SCRIPT_VOWELS = {
    "cyrillic": set("аеёиоуыэюя" "іїє" "әөүұ" "ӑӗӳ" "ӥӧ" "ў" "ъ" "ӣӯ" "ӱ"),
    "latin": set("aeiouy" "áéíóúàèìòùâêîôû" "äëïöü" "åæøœãõ" "āēīōū" "əı"),
    "armenian": set("աեէըիուօ"),
    "georgian": set("აეიოუ"),
}


def lower_preserving_length(text: str) -> str:
    """Lowercase *text* without changing its length.

    Stress indices are computed on the lowercased word and then spliced into
    the original — which requires ``len(lower) == len(original)``.  Plain
    ``str.lower()`` breaks that for the Turkic dotted capital İ (U+0130 →
    ``i`` + combining dot, two code points), live in the aze_lat / uzb_lat /
    tat alphabets.  Any character whose lowercase form expands keeps only
    the first code point of that form.
    """
    return "".join(c.lower() if len(c.lower()) == 1 else c.lower()[0] for c in text)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


class Tokenized(NamedTuple):
    """A sentence split for word-level stress prediction.

    The three sequences are index-aligned; joining ``raw`` reconstructs the
    input exactly.  ``clean[i]`` is the lowercased, alphabet-filtered lookup
    key for ``raw[i]``; ``predict[i]`` is False for separators, tokens with
    no alphabet characters, and masked hyphen clitics.
    """

    raw: list
    clean: list
    predict: list


def tokenize(sentence: str, clean_re: re.Pattern, mask_clitics: bool = False) -> Tokenized:
    """Split *sentence* into stressable tokens (the one shared tokenizer).

    Words are cut at the shared boundary set (:data:`_RE_SPLIT`) and then at
    hyphens, each hyphen part predicted independently.  With
    ``mask_clitics=True`` a final hyphen part in
    :data:`_UNSTRESSED_HYPHEN_CLITICS` (кто́-то, како́й-нибудь …) is excluded
    from prediction.
    """
    raw, clean, predict = [], [], []
    for word in _RE_SPLIT.split(sentence):
        parts = word.split("-")
        if len(parts) == 1:
            tokens = parts
            mask = [True]
        else:
            tokens = [p + "-" for p in parts[:-1]] + [parts[-1]]
            mask = [True] * (len(parts) - 1) + [
                not (mask_clitics and parts[-1] in _UNSTRESSED_HYPHEN_CLITICS)
            ]
        keys = [clean_re.sub("", t.lower()) for t in tokens]
        raw.extend(tokens)
        clean.extend(keys)
        predict.extend(bool(k) and m for k, m in zip(keys, mask))
    return Tokenized(raw, clean, predict)
