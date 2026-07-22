"""Simple-accentor family (vocabulary + rules).

``simple_accentor`` (model id ``"simple"`` — every language in
``SIMPLE_LANGS``, from Turkic final-stress languages to the Wiktionary- and
dictionary-backed ``bul``/``mkd``/``slv``/``lav``/``ru_simple``/``ukr_simple``):
Vocabulary + rule-based pipeline.
1. Tokenise sentence.
2. Look up clean token in vocabulary dict (word → stress char index).
3. OOV fall-back: language-specific positional rule (last/first/none/kat).
4. Insert '+' at the determined character index.
"""
import threading
import gzip
import json
import unicodedata
import re
from typing import Callable, Optional

from stressonnx._common import SCRIPT_VOWELS, lower_preserving_length, tokenize
from stressonnx.download import LOG, _download_files
from stressonnx.errors import ModelDownloadError, ModelLoadError, UnsupportedLanguageError
from stressonnx.notation import STRESS_TOKEN, _insert_stress
from stressonnx.registry import LANGUAGES, SIMPLE_LANGS, _OOV_RULES, _SIMPLE_FILES, hf_dir


def _load_vocab(path: str) -> dict:
    """Load a SimpleAccentor vocab (word → stress char index).

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
                word, idx = line.rsplit(None, 1)
                vocab[word] = int(idx)
            except ValueError:
                bad += 1
    if bad:
        LOG.warning("%s: skipped %d malformed vocabulary line(s)", path, bad)
    return vocab


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


def _rule_antepenult(word: str, vowels: list) -> Optional[int]:
    """Antepenultimate vowel, initial for shorter words.

    Georgian: Akhvlediani 1949, Gudava 1969, Aronson 1990 — the tradition
    the curated vocabulary follows exactly (Georgian stress is weak and
    contested; Borise 2020 argues fixed initial).  Macedonian: fixed
    antepenultimate stress, first syllable in shorter words (Friedman 2001,
    "Macedonian"); words with exceptional stress are in the vocabulary,
    which is exactly the set Wiktionary marks with an explicit accent."""
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


def _rule_tat(word: str, vowels: list) -> Optional[int]:
    """Tatar: final stress, except before the unstressed suffixes/enclitics
    Comrie 1997b documents (interrogative -мы/-ме, 2sg -сең, adverbial
    -ча/-чә).  Only the variants whose retraction the vocabulary itself
    supports at ≥0.7 are enabled — surface-string matching cannot see
    morphology, and the remaining variants (e.g. -ма) are dominated by
    ordinary final-stressed nouns (the алма́ apple / а́лма "don't take"
    problem).  Same reason the equivalent Kazakh/Kyrgyz/Azerbaijani lists
    (Kirchner 1998) are documented but NOT applied: measured against those
    vocabularies, blind retraction loses more than it gains."""
    for suf in ("чә", "ча", "сең", "ме"):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            pre = [i for i in vowels if i < len(word) - len(suf)]
            if pre:
                return pre[-1]
    return vowels[-1]


def _rule_sah(word: str, vowels: list) -> Optional[int]:
    """Yakut/Sakha: stress the last long vowel or diphthong, else the final
    vowel.  Yakut writes long vowels and diphthongs as vowel digraphs (аа,
    ыы, уо, иэ …) and they attract stress (Krueger 1962: default final
    stress, with weight-sensitivity; the curated vocabulary marks the second
    element of the heavy nucleus — бии́р, буо́лан, эрээ́ри — and this rule
    scores 0.994 against it vs 0.178 for naive final stress)."""
    groups: list = []
    for i in vowels:
        if groups and i == groups[-1][-1] + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    heavy = [g for g in groups if len(g) >= 2]
    return (heavy[-1] if heavy else groups[-1])[-1]


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
    "antepenult": _rule_antepenult,
    "hye": _rule_hye,
    "tgk": _rule_tgk,
    "mdf": _rule_mdf,
    "sah": _rule_sah,
    "tat": _rule_tat,
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
        if lang not in SIMPLE_LANGS:
            raise UnsupportedLanguageError(lang, SIMPLE_LANGS)
        self.lang = lang
        self._hf_lang = hf_dir(lang, "simple")
        self._ordinal_vowels = SCRIPT_VOWELS[LANGUAGES[lang]["script"]]
        self._cache_dir = cache_dir
        self._loaded = False
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
        """Thread-safe lazy load (double-checked locking).

        Download failures surface as :class:`ModelDownloadError`; anything
        that fails while parsing or building sessions from files already on
        disk is wrapped in :class:`ModelLoadError` so the fallback chain can
        engage on a corrupt cache too.  ``self._loaded`` flips only after
        every attribute is fully initialized.
        """
        if self._loaded:
            return
        with self._load_lock:
            if self._loaded:
                return
            try:
                self._load()
            except (ModelDownloadError, ModelLoadError):
                raise
            except Exception as exc:
                raise ModelLoadError('simple', exc) from exc
            self._loaded = True

    def _load(self) -> None:
        data = _download_files(self._hf_lang, _SIMPLE_FILES, self._cache_dir, model_id="simple")

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


    def _accentuate_vocab(self, clean_word: str, raw_word: str) -> str:
        """Vocabulary hit: mark the Nth vowel of the raw word.

        Vocabularies store vowel ordinals counted over the script-wide
        superset (:data:`SCRIPT_VOWELS`) — orthography-robust where raw
        character indices were not (case folding, digraphs).
        """
        ordinal = self._vocab[clean_word]
        lower = lower_preserving_length(raw_word)
        vowel_ids = [i for i, c in enumerate(lower) if c in self._ordinal_vowels]
        if ordinal >= len(vowel_ids):
            return raw_word  # raw form diverges from the vocab spelling
        return _insert_stress(raw_word, vowel_ids[ordinal])

    def _accentuate_oov(self, raw_word: str) -> str:
        """Stress an out-of-vocabulary word by the language's positional rule.

        Single-vowel words are always stressed on their sole vowel (upstream
        SimpleAccentor semantics); otherwise the named rule from
        :data:`OOV_RULES` decides — see each rule's docstring for its
        linguistic source and benchmarks/RESULTS.md for measured accuracy.
        """
        lower = lower_preserving_length(raw_word)
        vowel_ids = [i for i, c in enumerate(lower) if c in self._vowels]
        if not vowel_ids:
            return raw_word
        if len(vowel_ids) == 1:
            idx = vowel_ids[0]
        else:
            rule = OOV_RULES[self._oov_rule]  # unknown names fail loudly
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
            if STRESS_TOKEN in unicodedata.normalize("NFD", raw_word):
                out.append(raw_word)
                continue
            if clean_word in self._vocab:
                out.append(self._accentuate_vocab(clean_word, raw_word))
            else:
                out.append(self._accentuate_oov(raw_word))
        return "".join(out)
