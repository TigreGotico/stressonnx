"""Simple-accentor family: vocabulary + rules, span-native.

``simple_accentor`` (model id ``"simple"``) serves every language in
``SIMPLE_LANGS``: dictionary lookup with a per-language positional rule for
out-of-vocabulary words.  Positions are computed as **offsets into the
original input** (:meth:`SimpleStressor.mark_offsets`); the marked string is
a rendering of those offsets, never the other way around.
"""
import gzip
import re
import threading
import unicodedata
from typing import Callable, Optional

from stressonnx._common import SCRIPT_VOWELS, lower_preserving_length, tokenize
from stressonnx.download import LOG, _download_files
from stressonnx.errors import ModelDownloadError, ModelLoadError
from stressonnx.langs import resolve_lang
from stressonnx.notation import STRESS_TOKEN, render_marks
from stressonnx.registry import LANGUAGES, SIMPLE_LANGS, _OOV_RULES, _SIMPLE_FILES


def _load_vocab(path: str) -> dict:
    """Load a vocabulary (word → stressed vowel ordinal).

    Malformed lines are skipped and counted instead of aborting the whole
    language — vocabularies are regenerated from external sources and one
    bad row must not take a language down.
    """
    vocab: dict = {}
    bad = 0
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                word, ordinal = line.rsplit(None, 1)
                vocab[word] = int(ordinal)
            except ValueError:
                bad += 1
    if bad:
        LOG.warning("%s: skipped %d malformed vocabulary line(s)", path, bad)
    return vocab


# ---------------------------------------------------------------------------
# OOV positional rules
#
# Each rule maps (lowercased word, char indices of its language vowels) →
# the position in that vowel list to stress, or None for "leave unmarked".
# Rules are only consulted for words with two or more vowels — single-vowel
# words are always stressed on their sole vowel.  Sources per rule are in
# the docstrings; measured accuracy per language lives in each language's
# data file (``rule_accuracy``) and benchmarks/RESULTS.md (reproduce:
# python benchmarks/oov_rules_eval.py).
# ---------------------------------------------------------------------------

OovRule = Callable[[str, list], Optional[int]]


def _rule_last(word: str, vowels: list) -> Optional[int]:
    """Final-syllable stress — the Turkic default (Kirchner 1998, Poppe 1964
    via Schiering & van der Hulst 2010, "Word accent systems in the
    languages of Asia")."""
    return len(vowels) - 1


def _rule_first(word: str, vowels: list) -> Optional[int]:
    """Initial-syllable stress — Mordvinic: "in both languages the stress
    most commonly falls on the first syllable" (Hamari & Ajanki 2022, The
    Oxford Guide to the Uralic Languages §23.2.3)."""
    return 0


def _rule_none(word: str, vowels: list) -> Optional[int]:
    """No mark — for languages whose stress is free and lexically governed
    (East Slavic, Bulgarian, Slovene): a positional guess is never
    defensible, so unknown words stay unmarked."""
    return None


def _rule_chv(word: str, vowels: list) -> Optional[int]:
    """Chuvash: last full vowel; the reduced vowels ӑ/ӗ never carry stress,
    and all-reduced words stress the first syllable (Clark 1998: 435-436 and
    Krueger 1961 via Schiering & van der Hulst 2010; Dobrovolsky 1999,
    ICPhS: "If a word has only reduced vowels, stress falls on the first
    vowel")."""
    for n in range(len(vowels) - 1, -1, -1):
        if word[vowels[n]] not in "ӑӗ":
            return n
    return 0


def _rule_antepenult(word: str, vowels: list) -> Optional[int]:
    """Antepenultimate vowel, initial for shorter words.

    Georgian: Akhvlediani 1949, Gudava 1969, Aronson 1990 — the tradition
    the curated vocabulary follows exactly (Georgian stress is weak and
    contested; Borise 2020 argues fixed initial).  Macedonian: fixed
    antepenultimate stress, first syllable in shorter words (Friedman 2001,
    "Macedonian"); words with exceptional stress are in the vocabulary,
    which is exactly the set Wiktionary marks with an explicit accent."""
    return len(vowels) - 3 if len(vowels) >= 3 else 0


def _rule_hye(word: str, vowels: list) -> Optional[int]:
    """Eastern Armenian: "stress occurs within the last non-schwa syllable"
    (Chakmakjian 2024, Speech Prosody) — the schwa ը never carries stress."""
    for n in range(len(vowels) - 1, -1, -1):
        if word[vowels[n]] != "ը":
            return n
    return 0


def _rule_tgk(word: str, vowels: list) -> Optional[int]:
    """Tajik: final syllable, except unstressed word-final ``-и`` — the
    izafet enclitic; Perry 2005 (A Tajik Persian Reference Grammar): long ӣ
    "distinguish[es] accented word-final -i from unstressed final -i, which
    occurs only … as the syntactic izofat enclitic"."""
    if word[vowels[-1]] == "и" and vowels[-1] == len(word) - 1:
        return len(vowels) - 2
    return len(vowels) - 1


def _rule_mdf(word: str, vowels: list) -> Optional[int]:
    """Moksha: first syllable, but "stress is often assigned to a
    non-initial syllable if it contains /a/ (or /æ/) and if there is a high
    vowel (/i/ or /u/) in the initial syllable" (Hamari & Ajanki 2022)."""
    if word[vowels[0]] in "иу":
        for n in range(1, len(vowels)):
            if word[vowels[n]] in "ая":
                return n
    return 0


def _rule_tat(word: str, vowels: list) -> Optional[int]:
    """Tatar: final stress, except before the unstressed suffixes/enclitics
    Comrie 1997b documents (interrogative -мы/-ме, 2sg -сең, adverbial
    -ча/-чә).  Only the variants whose retraction the vocabulary itself
    supports at ≥0.7 are enabled — surface-string matching cannot see
    morphology, and the remaining variants (e.g. -ма) are dominated by
    ordinary final-stressed nouns (the алма́ apple / а́лма "don't take"
    problem).  The equivalent Kazakh/Kyrgyz/Azerbaijani lists (Kirchner
    1998) are documented but NOT applied: measured against those
    vocabularies, blind retraction loses more than it gains."""
    for suf in ("чә", "ча", "сең", "ме"):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            pre = [n for n, i in enumerate(vowels) if i < len(word) - len(suf)]
            if pre:
                return pre[-1]
    return len(vowels) - 1


def _rule_sah(word: str, vowels: list) -> Optional[int]:
    """Yakut/Sakha: stress the last long vowel or diphthong, else the final
    vowel.  Yakut writes long vowels and diphthongs as vowel digraphs (аа,
    ыы, уо, иэ …) and they attract stress (Krueger 1962: default final
    stress, with weight-sensitivity; the curated vocabulary marks the second
    element of the heavy nucleus — бии́р, буо́лан, эрээ́ри — and this rule
    scores 0.994 against it vs 0.178 for naive final stress)."""
    groups: list = []
    for n, i in enumerate(vowels):
        if groups and i == vowels[groups[-1][-1]] + 1:
            groups[-1].append(n)
        else:
            groups.append([n])
    heavy = [g for g in groups if len(g) >= 2]
    return (heavy[-1] if heavy else groups[-1])[-1]


def _rule_kbd(word: str, vowels: list) -> Optional[int]:
    """Kabardian: final syllable, except words ending in the schwa letter э,
    which stress the penult (Jaimoukha, Grammar of the Kabardian-Cherkess
    Language; consistent with Colarusso 1992's final-stress default).
    Low-confidence: sources are paraphrase-level only."""
    return len(vowels) - 2 if word.endswith("э") else len(vowels) - 1


OOV_RULES: dict = {
    "last": _rule_last,
    "first": _rule_first,
    "none": _rule_none,
    "chv": _rule_chv,
    "antepenult": _rule_antepenult,
    "hye": _rule_hye,
    "tgk": _rule_tgk,
    "mdf": _rule_mdf,
    "tat": _rule_tat,
    "sah": _rule_sah,
    "kbd": _rule_kbd,
}


class SimpleStressor:
    """Vocabulary + rule accentor; positions first, strings second.

    Parameters
    ----------
    lang:
        A tag from ``SIMPLE_LANGS``.
    cache_dir:
        Override the model storage directory (default: the standard
        Hugging Face cache).
    """

    def __init__(self, lang: str, cache_dir: str | None = None) -> None:
        lang = resolve_lang(lang, SIMPLE_LANGS)
        self.lang = lang
        spec = LANGUAGES[lang]
        self._hf_lang = spec["hf"]["simple"]
        self._ordinal_vowels = SCRIPT_VOWELS[spec["script"]]
        self._vowels: str = spec["vowels"]
        self._oov_rule: str = _OOV_RULES[lang]
        alpha = spec["alpha"]
        self._re_cond = re.compile(
            f"[^{re.escape(''.join(sorted(set(alpha + alpha.upper()))))}]"
        )
        self._cache_dir = cache_dir
        self._loaded = False
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Thread-safe lazy load (double-checked locking).

        Download failures surface as :class:`ModelDownloadError`; anything
        that fails while parsing files already on disk is wrapped in
        :class:`ModelLoadError` so the fallback chain can engage on a
        corrupt cache too.  ``self._loaded`` flips only after every
        attribute is fully initialized.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                data = _download_files(self._hf_lang, _SIMPLE_FILES,
                                       self._cache_dir, model_id="simple")
                self._vocab: dict = _load_vocab(data["vocab.gz"])
            except (ModelDownloadError, ModelLoadError):
                raise
            except Exception as exc:
                raise ModelLoadError("simple", exc) from exc
            self._loaded = True

    # ------------------------------------------------------------------
    # Span-native core
    # ------------------------------------------------------------------

    def _rule_offset(self, raw_lower: str) -> Optional[int]:
        """Char offset the positional rule stresses in *raw_lower*, or None.

        Single-vowel words are always stressed on their sole vowel;
        otherwise the language's rule from :data:`OOV_RULES` decides.
        """
        vowel_ids = [i for i, c in enumerate(raw_lower) if c in self._vowels]
        if not vowel_ids:
            return None
        if len(vowel_ids) == 1:
            return vowel_ids[0]
        n = OOV_RULES[self._oov_rule](raw_lower, vowel_ids)
        return None if n is None else vowel_ids[n]

    def _word_ordinal(self, clean_word: str, raw_lower: str) -> Optional[int]:
        """Superset-vowel ordinal to stress in one token, or None."""
        if clean_word in self._vocab:
            return self._vocab[clean_word]
        idx = self._rule_offset(raw_lower)
        if idx is None:
            return None
        # language-vowel position → superset ordinal, so vocabulary and rule
        # answers share one currency
        return sum(1 for c in raw_lower[:idx] if c in self._ordinal_vowels)

    def mark_offsets(self, text: str):
        """Offsets (into *text*) of every stressed vowel — the primitive.

        Returns ``(mark_offsets, yo_offsets)``; the simple family never
        substitutes е→ё, so the second list is always empty.
        """
        self._ensure_loaded()
        tokens = tokenize(text, self._re_cond)
        offsets = []
        pos = 0
        for raw_word, clean_word, need in zip(*tokens):
            start = pos
            pos += len(raw_word)
            if not need:
                continue
            if STRESS_TOKEN in unicodedata.normalize("NFD", raw_word):
                continue
            raw_lower = lower_preserving_length(raw_word)
            ordinal = self._word_ordinal(clean_word, raw_lower)
            if ordinal is None:
                continue
            seen = -1
            for i, c in enumerate(raw_lower):
                if c in self._ordinal_vowels:
                    seen += 1
                    if seen == ordinal:
                        offsets.append(start + i)
                        break
        return offsets, []

    def __call__(self, sentence: str) -> str:
        marks, yo = self.mark_offsets(sentence)
        return render_marks(sentence, marks, yo)
