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

# The prefix of Russian and Ukrainian hyphenated adverbs (по-ру́сски,
# по-пре́жнему, по-украї́нськи) carries no stress: en-Wiktionary marks stress on
# 181 Russian and 5 Ukrainian по- tokens, none of them on по.
_PO_PREFIX = frozenset({"по"})

# The possessive pronoun after по-: stressed on its first vowel (по-мо́ему,
# по-сво́ему; uk по-мо́єму, по-сво́єму per СУМ), where the dative on its own is
# моему́, моє́му.  lang → word → vowel ordinal, used after a по- part.
_PO_ADVERBS = {
    "ru": {"моему": 0, "твоему": 0, "своему": 0},
    "uk": {"моєму": 0, "твоєму": 0, "своєму": 0},
}

# Belarusian hyphen parts that never carry word stress, from GrammarDB
# RELEASE-202601 (spellchecker/slovy-2008-stress.txt).  The prefix is
# unstressed in all 452 па- adverbs (па-белару́ску) and 39 абы- pronouns
# (абы-хто́).  The particles -та and -такі are unstressed only after the hosts
# the list gives them (такі́-та, то́-та and 11 more forms of такі; усё-такі,
# та́к-такі, даво́лі-такі) and stressed elsewhere (та-та́, сякі́-такі́,
# про́ста-такі́); зноў-такі is listed both ways and is not masked.
# en-Wiktionary agrees: па-белару́ску [pa bʲeɫaˈrusku], усё-такі [uˈsʲo takʲi].
# -не́будзь carries its own stress (en-Wiktionary які́-не́будзь, [ˈnʲebut͡sʲ];
# GrammarDB не+будзь).
_BE_UNSTRESSED_HYPHEN_PREFIXES = frozenset({"па", "абы"})
_BE_HOSTED_HYPHEN_CLITICS = {
    "та": frozenset({"то", "такі", "такая", "такое", "такія", "такога", "такой", "такому",
                     "такую", "такою", "такім", "такіх", "такімі"}),
    "такі": frozenset({"усё", "ўсё", "так", "даволі"}),
}

#: lang → (hyphen parts never stressed at the start of a word, at its end,
#: at its end after one of the given preceding parts)
_UNSTRESSED_HYPHEN_PARTS = {
    "ru": (_PO_PREFIX, _UNSTRESSED_HYPHEN_CLITICS, {}),
    "uk": (_PO_PREFIX, frozenset(), {}),
    "be": (_BE_UNSTRESSED_HYPHEN_PREFIXES, frozenset(), _BE_HOSTED_HYPHEN_CLITICS),
}

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


#: Turkic case pairs: İ ↔ i and I ↔ ı.  ``str.lower()`` maps I to i, which
#: is a different letter in an alphabet with ı.
_TURKIC_I = str.maketrans({"İ": "i", "I": "ı"})


def lower_preserving_length(text: str, turkic: bool = False) -> str:
    """Lowercase *text* without changing its length.

    Stress indices are computed on the lowercased word and then spliced into
    the original — which requires ``len(lower) == len(original)``.  Plain
    ``str.lower()`` breaks that for the Turkic dotted capital İ (U+0130 →
    ``i`` + combining dot, two code points), live in the aze_lat / uzb_lat /
    tat alphabets.  Any character whose lowercase form expands keeps only
    the first code point of that form.  With ``turkic=True`` the pairs
    İ → i and I → ı apply before the generic lowering.
    """
    if turkic:
        text = text.translate(_TURKIC_I)
    lowered = text.lower()
    # str.lower() writes a word-final capital sigma as ς, by context; the
    # per-character form below gives σ
    if len(lowered) == len(text) and "Σ" not in text:
        return lowered
    return "".join(c.lower() if len(c.lower()) == 1 else c.lower()[0] for c in text)


def _encoder(tokenizer):
    """The object that encodes for a ``tokenizers.Tokenizer``.

    From tokenizers 1.0.0rc4 on, only the ``CompiledTokenizer`` that
    ``Tokenizer.compile()`` builds can encode, with no truncation and no
    padding.  Older versions, 1.0.0rc1 to rc3 among them, have no
    ``compile()`` and encode on the ``Tokenizer``.
    """
    if not hasattr(tokenizer, "compile"):
        return tokenizer
    # compile() freezes the settings: edits after it do not reach the result
    tokenizer.truncation = None
    tokenizer.padding = None
    return tokenizer.compile()


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


def tokenize(sentence: str, clean_re: re.Pattern, lang: str | None = None,
             turkic: bool = False) -> Tokenized:
    """Split *sentence* into stressable tokens (the one shared tokenizer).

    Words are cut at the shared boundary set (:data:`_RE_SPLIT`) and then at
    hyphens, each hyphen part predicted independently.  For a *lang* in
    :data:`_UNSTRESSED_HYPHEN_PARTS` a first or final hyphen part listed
    there (кто́-то, како́й-нибудь, па-белару́ску) is excluded from prediction,
    and so is a final part listed with the part before it (такі́-та).  ``turkic=True`` lowercases I to ı and İ to i.
    """
    prefixes, clitics, hosted = _UNSTRESSED_HYPHEN_PARTS.get(lang, (frozenset(), frozenset(), {}))
    raw, clean, predict = [], [], []
    for word in _RE_SPLIT.split(sentence):
        parts = word.split("-")
        if len(parts) == 1:
            tokens = parts
            mask = [True]
        else:
            tokens = [p + "-" for p in parts[:-1]] + [parts[-1]]
            mask = [True] * len(parts)
            mask[0] = parts[0].lower() not in prefixes
            last = parts[-1].lower()
            mask[-1] = last not in clitics and clean_re.sub("", parts[-2].lower()) not in hosted.get(last, ())
        keys = [clean_re.sub("", (t.translate(_TURKIC_I) if turkic else t).lower()) for t in tokens]
        raw.extend(tokens)
        clean.extend(keys)
        predict.extend(bool(k) and m for k, m in zip(keys, mask))
    return Tokenized(raw, clean, predict)
