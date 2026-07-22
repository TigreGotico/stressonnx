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

from stressonnx._common import _RE_SPLIT
from stressonnx.download import _download_files
from stressonnx.errors import UnsupportedLanguageError
from stressonnx.notation import STRESS_TOKEN, _insert_stress
from stressonnx.registry import SIMPLE_LANGS, _OOV_RULES, _SIMPLE_FILES


def _load_vocab(path: str) -> dict:
    """Load SimpleAccentor vocab: word → stress char index."""
    with gzip.open(path, "rb") as fh:
        lines = [x.decode().strip() for x in fh.readlines()]
    return {x.rsplit(maxsplit=1)[0]: int(x.rsplit(maxsplit=1)[1]) for x in lines if x}


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

    def _tokenize(self, sentence: str):
        tokens, model_inputs, prediction_mask = [], [], []
        for word in _RE_SPLIT.split(sentence):
            parts = word.split("-")
            if len(parts) == 1:
                cur_tokens = parts
                cur_pred_mask = [True]
            else:
                cur_tokens = [p + "-" for p in parts[:-1]] + [parts[-1]]
                cur_pred_mask = [True for _ in parts]
            cur_inputs = [self._re_cond.sub("", t.lower()) for t in cur_tokens]
            cur_pred_mask = [
                (len(x) > 0) and bool(m)
                for x, m in zip(cur_inputs, cur_pred_mask)
            ]
            tokens.extend(cur_tokens)
            model_inputs.extend(cur_inputs)
            prediction_mask.extend(cur_pred_mask)
        return tokens, model_inputs, prediction_mask

    def _accentuate_vocab(self, clean_word: str, raw_word: str) -> str:
        # vowels=None: the curated vocab may stress loanword vowels outside
        # the language's core set (e.g. ю/я in aze_cyr дюнья́)
        return _insert_stress(raw_word, self._vocab[clean_word])

    def _accentuate_oov(self, raw_word: str) -> str:
        """Stress an out-of-vocabulary word by the language's positional rule.

        Each named rule is grounded in the descriptive literature and scored
        against the language's own vocabulary in
        ``benchmarks/oov_rules_eval.py`` (accuracy on multi-vowel words):

        ``last`` / ``first`` / ``none``
            Final-syllable stress (Turkic default: Kirchner 1998, Poppe 1964
            via Schiering & van der Hulst 2010, "Word accent systems in the
            languages of Asia"), initial-syllable stress (Mordvinic: "in both
            languages the stress most commonly falls on the first syllable",
            Hamari & Ajanki 2022, The Oxford Guide to the Uralic Languages
            §23.2.3), and no mark (Belarusian: stress is free and lexically
            governed, not positionally predictable — "nominal stress in
            Ukrainian, Russian, and Belarusian … seems to be unpredictable").
        ``chv`` (0.60 → 0.94)
            "Stress falls on the last syllable with a full vowel, else on the
            first syllable" — the reduced vowels ӑ/ӗ never carry stress
            (Clark 1998: 435-436 and Krueger 1961 via Schiering & van der
            Hulst 2010; independently Dobrovolsky 1999, ICPhS: "If a word has
            only reduced vowels, stress falls on the first vowel").
        ``kat`` (0.46 → 1.00)
            Antepenultimate vowel, initial for shorter words (Akhvlediani
            1949, Gudava 1969, Aronson 1990).  Georgian stress is weak and
            contested — Borise 2020 argues fixed initial — but this is the
            tradition the curated vocabulary follows exactly.
        ``hye`` (0.74 → 0.78)
            "Stress occurs within the last non-schwa syllable" (Chakmakjian
            2024, Speech Prosody) — the schwa ը never carries stress.
        ``tgk`` (0.45 → 0.74)
            Final syllable, but unstressed word-final ``-и`` — the izafet
            enclitic; Perry 2005 (A Tajik Persian Reference Grammar): long ӣ
            "distinguish[es] accented word-final -i from unstressed final -i,
            which occurs only … as the syntactic izofat enclitic".
        ``mdf`` (0.72 → 0.76)
            First syllable; "stress is often assigned to a non-initial
            syllable if it contains /a/ (or /æ/) and if there is a high vowel
            (/i/ or /u/) in the initial syllable" (Hamari & Ajanki 2022).
        ``kbd`` (0.32 → 0.40)
            Final syllable, except words ending in the schwa letter э, which
            stress the penult (Jaimoukha, Grammar of the Kabardian-Cherkess
            Language; consistent with Colarusso 1992's final-stress default).
            Low-confidence: sources are paraphrase-level only.
        """
        lower = raw_word.lower()
        vowel_ids = [i for i, c in enumerate(lower) if c in self._vowels]
        if not vowel_ids:
            return raw_word
        rule = self._oov_rule
        if len(vowel_ids) == 1:
            idx = vowel_ids[0]
        elif rule == "last":
            idx = vowel_ids[-1]
        elif rule == "first":
            idx = vowel_ids[0]
        elif rule == "none":
            return raw_word
        elif rule == "chv":
            idx = next(
                (i for i in reversed(vowel_ids) if lower[i] not in "ӑӗ"),
                vowel_ids[0],
            )
        elif rule == "kat":
            idx = vowel_ids[-3] if len(vowel_ids) >= 3 else vowel_ids[0]
        elif rule == "hye":
            idx = next(
                (i for i in reversed(vowel_ids) if lower[i] != "ը"),
                vowel_ids[0],
            )
        elif rule == "tgk":
            if lower[vowel_ids[-1]] == "и" and vowel_ids[-1] == len(lower) - 1:
                idx = vowel_ids[-2]
            else:
                idx = vowel_ids[-1]
        elif rule == "mdf":
            idx = vowel_ids[0]
            if lower[idx] in "иу":
                idx = next((i for i in vowel_ids[1:] if lower[i] in "ая"), idx)
        elif rule == "kbd":
            idx = vowel_ids[-2] if lower.endswith("э") else vowel_ids[-1]
        else:
            idx = vowel_ids[-1]
        return raw_word[:idx + 1] + STRESS_TOKEN + raw_word[idx + 1:]

    def __call__(self, sentence: str) -> str:
        self._ensure_loaded()
        raw_tokens, clean_tokens, prediction_mask = self._tokenize(sentence)
        out = []
        for raw_word, clean_word, need in zip(
            raw_tokens, clean_tokens, prediction_mask
        ):
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
