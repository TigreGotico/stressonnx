"""Simple-accentor family (vocabulary + rules).

``simple_accentor`` (model id ``"simple"``, languages ``aze_cyr``,
``aze_lat``, ``uzb_cyr``, ``uzb_lat``, ``bak``, ``bel_simple``, ``chv``,
``erz``, ``hye``, ``kat``, ``kaz``, ``kbd``, ``kir``, ``kjh``, ``mdf``,
``sah``, ``tat``, ``tgk``, ``udm``, ``xal``):
Vocabulary + rule-based pipeline.
1. Tokenise sentence.
2. Look up clean token in vocabulary dict (word → stress char index).
3. OOV fall-back: language-specific positional rule (last/first/none/kat).
4. Insert '+' at the determined character index.
"""
import gzip
import json
import re
from typing import Callable, Optional

from stressonnx._common import tokenize
from stressonnx.download import _download_files
from stressonnx.errors import UnsupportedLanguageError
from stressonnx.notation import STRESS_TOKEN, _insert_stress
from stressonnx.registry import SIMPLE_LANGS, _OOV_RULES, _SIMPLE_FILES


def _load_vocab(path: str) -> dict:
    """Load SimpleAccentor vocab: word → stress char index."""
    with gzip.open(path, "rb") as fh:
        lines = [x.decode().strip() for x in fh.readlines()]
    return {x.rsplit(maxsplit=1)[0]: int(x.rsplit(maxsplit=1)[1]) for x in lines if x}


# ---------------------------------------------------------------------------
# OOV positional rules
#
# Each rule maps (lowercased word, indices of its vowels) → the char index to
# stress, or None for "leave unmarked".  Rules are only consulted for words
# with two or more vowels — single-vowel words are always stressed on their
# sole vowel (upstream SimpleAccentor semantics).  Sources per rule are in
# the docstrings; measured accuracy per language lives in
# benchmarks/RESULTS.md (reproduce: python benchmarks/oov_rules_eval.py).
# ---------------------------------------------------------------------------

OovRule = Callable[[str, list], Optional[int]]


def _rule_last(word: str, vowels: list) -> Optional[int]:
    """Final-syllable stress — the Turkic default (Kirchner 1998, Poppe 1964
    via Schiering & van der Hulst 2010, "Word accent systems in the
    languages of Asia")."""
    return vowels[-1]


def _rule_first(word: str, vowels: list) -> Optional[int]:
    """Initial-syllable stress — Mordvinic: "in both languages the stress
    most commonly falls on the first syllable" (Hamari & Ajanki 2022, The
    Oxford Guide to the Uralic Languages §23.2.3)."""
    return vowels[0]


def _rule_none(word: str, vowels: list) -> Optional[int]:
    """No mark — Belarusian stress is free and lexically governed, not
    positionally predictable ("nominal stress in Ukrainian, Russian, and
    Belarusian … seems to be unpredictable")."""
    return None


def _rule_chv(word: str, vowels: list) -> Optional[int]:
    """Chuvash: last full vowel; the reduced vowels ӑ/ӗ never carry stress,
    and all-reduced words stress the first syllable (Clark 1998: 435-436 and
    Krueger 1961 via Schiering & van der Hulst 2010; Dobrovolsky 1999,
    ICPhS: "If a word has only reduced vowels, stress falls on the first
    vowel")."""
    return next((i for i in reversed(vowels) if word[i] not in "ӑӗ"), vowels[0])


def _rule_kat(word: str, vowels: list) -> Optional[int]:
    """Georgian: antepenultimate vowel, initial for shorter words
    (Akhvlediani 1949, Gudava 1969, Aronson 1990).  Georgian stress is weak
    and contested — Borise 2020 argues fixed initial — but this is the
    tradition the curated vocabulary follows exactly."""
    return vowels[-3] if len(vowels) >= 3 else vowels[0]


def _rule_hye(word: str, vowels: list) -> Optional[int]:
    """Eastern Armenian: "stress occurs within the last non-schwa syllable"
    (Chakmakjian 2024, Speech Prosody) — the schwa ը never carries stress."""
    return next((i for i in reversed(vowels) if word[i] != "ը"), vowels[0])


def _rule_tgk(word: str, vowels: list) -> Optional[int]:
    """Tajik: final syllable, except unstressed word-final ``-и`` — the
    izafet enclitic; Perry 2005 (A Tajik Persian Reference Grammar): long ӣ
    "distinguish[es] accented word-final -i from unstressed final -i, which
    occurs only … as the syntactic izofat enclitic"."""
    if word[vowels[-1]] == "и" and vowels[-1] == len(word) - 1:
        return vowels[-2]
    return vowels[-1]


def _rule_mdf(word: str, vowels: list) -> Optional[int]:
    """Moksha: first syllable, but "stress is often assigned to a
    non-initial syllable if it contains /a/ (or /æ/) and if there is a high
    vowel (/i/ or /u/) in the initial syllable" (Hamari & Ajanki 2022)."""
    if word[vowels[0]] in "иу":
        return next((i for i in vowels[1:] if word[i] in "ая"), vowels[0])
    return vowels[0]


def _rule_kbd(word: str, vowels: list) -> Optional[int]:
    """Kabardian: final syllable, except words ending in the schwa letter э,
    which stress the penult (Jaimoukha, Grammar of the Kabardian-Cherkess
    Language; consistent with Colarusso 1992's final-stress default).
    Low-confidence: sources are paraphrase-level only."""
    return vowels[-2] if word.endswith("э") else vowels[-1]


OOV_RULES: dict = {
    "last": _rule_last,
    "first": _rule_first,
    "none": _rule_none,
    "chv": _rule_chv,
    "kat": _rule_kat,
    "hye": _rule_hye,
    "tgk": _rule_tgk,
    "mdf": _rule_mdf,
    "kbd": _rule_kbd,
}


class SimpleStressor:
    """Lazy-load vocabulary + rule-based accentor.

    Supports all ``SIMPLE_LANGS``.  The vocab maps known words to a stress
    character index; unknown words fall back to a per-language positional rule.

    Parameters
    ----------
    lang:
        One of the supported simple-accentor language tags.
    cache_dir:
        Override the default cache location.
    """

    def __init__(self, lang: str, cache_dir: str | None = None) -> None:
        # Treat "bel" as main accentor; "bel_simple" routes here.
        if lang not in SIMPLE_LANGS:
            raise UnsupportedLanguageError(lang, SIMPLE_LANGS)
        self.lang = lang
        # HF artefacts live under the canonical lang name (strip _simple suffix)
        self._hf_lang = lang.removesuffix("_simple") if lang.endswith("_simple") else lang
        self._cache_dir = cache_dir
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data = _download_files(self._hf_lang, _SIMPLE_FILES, self._cache_dir)

        with open(data["meta.json"], encoding="utf-8") as fh:
            meta = json.load(fh)

        self._vocab: dict = _load_vocab(data["vocab.gz"])
        self._vowels: str = meta["vowels"]
        # The local rule table wins over meta.json: several languages carry
        # linguistically refined rules (see _accentuate_oov) that the exported
        # metadata predates.
        self._oov_rule: str = _OOV_RULES.get(
            self.lang, meta.get("oov_rule", "last")
        )

        alpha = meta["alpha"]
        alpha_set = "".join(sorted(set(alpha + alpha.upper())))
        # Escape characters that are special in regex character classes
        escaped = re.escape(alpha_set)
        self._re_cond = re.compile(f"[^{escaped}]")

        self._loaded = True

    def _accentuate_vocab(self, clean_word: str, raw_word: str) -> str:
        # vowels=None: the curated vocab may stress loanword vowels outside
        # the language's core set (e.g. ю/я in aze_cyr дюнья́)
        return _insert_stress(raw_word, self._vocab[clean_word])

    def _accentuate_oov(self, raw_word: str) -> str:
        """Stress an out-of-vocabulary word by the language's positional rule.

        Single-vowel words are always stressed on their sole vowel (upstream
        SimpleAccentor semantics); otherwise the named rule from
        :data:`OOV_RULES` decides — see each rule's docstring for its
        linguistic source and benchmarks/RESULTS.md for measured accuracy.
        """
        lower = raw_word.lower()
        vowel_ids = [i for i, c in enumerate(lower) if c in self._vowels]
        if not vowel_ids:
            return raw_word
        if len(vowel_ids) == 1:
            idx = vowel_ids[0]
        else:
            rule = OOV_RULES.get(self._oov_rule, _rule_last)
            idx = rule(lower, vowel_ids)
            if idx is None:
                return raw_word
        return raw_word[:idx + 1] + STRESS_TOKEN + raw_word[idx + 1:]

    def __call__(self, sentence: str) -> str:
        self._ensure_loaded()
        tokens = tokenize(sentence, self._re_cond)
        out = []
        for raw_word, clean_word, need in zip(*tokens):
            if not need:
                out.append(raw_word)
                continue
            if STRESS_TOKEN in raw_word:
                out.append(raw_word)
                continue
            if clean_word in self._vocab:
                out.append(self._accentuate_vocab(clean_word, raw_word))
            else:
                out.append(self._accentuate_oov(raw_word))
        return "".join(out)
