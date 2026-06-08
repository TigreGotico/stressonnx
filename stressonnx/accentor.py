"""Per-language stress accentor backed by ONNX + numpy (main langs) or
vocabulary + rules (SimpleAccentor langs).

Models and data are downloaded from HuggingFace on first use and cached under
``~/.local/share/stressonnx/<lang>/``.

Two families
------------
``main_accentor`` langs (``ru``, ``ukr``, ``bel``):
    Pipeline:
    1. Tokenise sentence → (raw tokens, clean tokens, prediction mask).
    2. Compute fastText-style n-gram embeddings for each clean token by
       mean-pooling the rows selected from the embedding matrix.
    3. Run the ONNX MLP heads → stress_logits [N, K] (+ yo_logits for ``ru``).
    4. Decode: exceptions dict → skip sets → argmax position → insert '+'.

``simple_accentor`` langs (``aze_cyr``, ``aze_lat``, ``uzb_cyr``, ``uzb_lat``,
    ``bak``, ``bel``, ``chv``, ``erz``, ``hye``, ``kat``, ``kaz``, ``kbd``,
    ``kir``, ``kjh``, ``mdf``, ``sah``, ``tat``, ``tgk``, ``udm``, ``xal``):
    Pipeline:
    1. Tokenise sentence.
    2. Look up clean token in vocabulary dict (word → stress char index).
    3. OOV fall-back: language-specific positional rule (last/first/none/kat).
    4. Insert '+' at the determined character index.

``bel`` is available in *both* families; pass ``bel`` for the neural version
and ``bel_simple`` (alias handled internally) for the rule/vocab version —
or simply use ``bel`` which dispatches to the neural accentor by default.
"""
import gzip
import json
import os
import re

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# HF repo that hosts all per-language runtime artefacts
# ---------------------------------------------------------------------------
HF_REPO_ID = "TigreGotico/stressonnx-models"

# Files for main_accentor family
_MAIN_FILES = [
    "accentor.onnx",
    "embedding.npy",
    "ngram_dict.txt.gz",
    "exceptions.txt.gz",
    "skip_stress_words.txt.gz",
    "skip_yo_words.txt.gz",
    "meta.json",
]

# Files for simple_accentor family
_SIMPLE_FILES = [
    "vocab.gz",
    "meta.json",
]

STRESS_TOKEN = "+"

# ---------------------------------------------------------------------------
# Language routing tables
# ---------------------------------------------------------------------------
#: Languages backed by the neural ONNX pipeline (main_accentor).
MAIN_LANGS = {"ru", "ukr", "bel"}

#: Languages backed by vocabulary + rules (simple_accentor).
SIMPLE_LANGS = {
    "aze_cyr", "aze_lat",
    "uzb_cyr", "uzb_lat",
    "bak",
    "bel_simple",   # alias — same vocab as bel but always rule-path
    "chv", "erz", "hye", "kat", "kaz", "kbd", "kir",
    "kjh", "mdf", "sah", "tat", "tgk", "udm", "xal",
}

ALL_LANGS = MAIN_LANGS | SIMPLE_LANGS

# OOV stress-position rule per simple lang.
# "last"  → last vowel
# "first" → first vowel
# "none"  → skip (return unchanged)
# "kat"   → ≤3 vowels → first, else penultimate
_OOV_RULES = {
    "aze_cyr": "last", "aze_lat": "last",
    "uzb_cyr": "last", "uzb_lat": "last",
    "bak": "last", "bel": "none", "bel_simple": "none",
    "chv": "last", "erz": "first",
    "hye": "last", "kat": "kat",
    "kaz": "last", "kbd": "last", "kir": "last",
    "kjh": "last", "mdf": "first",
    "sah": "last", "tat": "last", "tgk": "last",
    "udm": "last", "xal": "last",
}

# Russian-specific vowel set used for the full accentuate pipeline.
_RU_VOWELS = "аоуыэиеяёю"
_RE_RU_SPLIT = re.compile(r"([\s.,!?;:<>=()/\\]+)")
_RE_RU_COND = re.compile(r"[^А-Яа-яёЁ]")


# ---------------------------------------------------------------------------
# Shared data loaders
# ---------------------------------------------------------------------------

def _load_ngram_dict(path: str) -> dict:
    d = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            gram, idx = line.rsplit("\t", 1)
            d[gram] = int(idx)
    return d


def _load_exceptions(path: str) -> dict:
    d = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            w, s, y = line.split("\t")
            d[w] = (int(s), int(y))
    return d


def _load_set(path: str) -> set:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return {x for x in fh.read().split("\n") if x}


def _load_vocab(path: str) -> dict:
    """Load SimpleAccentor vocab: word → stress char index."""
    with gzip.open(path, "rb") as fh:
        lines = [x.decode().strip() for x in fh.readlines()]
    return {x.rsplit(maxsplit=1)[0]: int(x.rsplit(maxsplit=1)[1]) for x in lines if x}


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# HF download helpers
# ---------------------------------------------------------------------------

def _download_files(lang: str, cache_dir: str, filenames: list) -> dict:
    """Download *filenames* for *lang* into *cache_dir*; return {fname: path}."""
    os.makedirs(cache_dir, exist_ok=True)
    paths = {}
    for fname in filenames:
        local = os.path.join(cache_dir, fname)
        if os.path.exists(local):
            paths[fname] = local
            continue
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=f"{lang}/{fname}",
            local_dir=cache_dir,
        )
        # hf_hub_download may nest under lang/ — normalise
        if os.path.basename(downloaded) == fname:
            actual = downloaded
        else:
            actual = os.path.join(cache_dir, lang, fname)
        paths[fname] = actual
    return paths


# ===========================================================================
# Main-accentor family (ru / ukr / bel)
# ===========================================================================

class Stressor:
    """Lazy-load neural accentor for ``ru``, ``ukr``, or ``bel``.

    Parameters
    ----------
    lang:
        Language tag: ``"ru"``, ``"ukr"``, or ``"bel"``.
    cache_dir:
        Override the default cache location
        (``~/.local/share/stressonnx/<lang>``).
    """

    def __init__(self, lang: str = "ru", cache_dir: str | None = None) -> None:
        if lang not in MAIN_LANGS:
            raise ValueError(
                f"Stressor only supports main-accentor langs {sorted(MAIN_LANGS)}; "
                f"got {lang!r}.  Use SimpleStressor for other langs."
            )
        self.lang = lang
        if cache_dir is None:
            base = os.path.join(
                os.path.expanduser("~"), ".local", "share", "stressonnx"
            )
            cache_dir = os.path.join(base, lang)
        self._cache_dir = cache_dir
        self._loaded = False

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data = _download_files(self.lang, self._cache_dir, _MAIN_FILES)

        with open(data["meta.json"]) as fh:
            meta = json.load(fh)

        self._W: np.ndarray = np.load(data["embedding.npy"]).astype(np.float32)
        self._ngram_dict: dict = _load_ngram_dict(data["ngram_dict.txt.gz"])
        self._unk: int = self._ngram_dict["UNK"]
        self._exceptions: dict = _load_exceptions(data["exceptions.txt.gz"])
        self._skip_stress: set = _load_set(data["skip_stress_words.txt.gz"])
        self._skip_yo: set = _load_set(data["skip_yo_words.txt.gz"])
        self._sess = ort.InferenceSession(
            data["accentor.onnx"],
            providers=["CPUExecutionProvider"],
        )

        # Per-language settings from meta.json (set by export script)
        self._has_yo: bool = meta.get("has_yo", self.lang == "ru")
        alpha = meta.get("alpha")
        if alpha:
            escaped = re.escape("".join(sorted(set(alpha))))
            self._re_cond = re.compile(f"[^{escaped}]")
        else:
            self._re_cond = _RE_RU_COND
        self._vowels: str = meta.get("vowels", _RU_VOWELS)

        self._loaded = True

    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, sentence: str):
        """Tokenise for ru (hyphen-aware) or ukr/bel (simple split)."""
        if self.lang == "ru":
            return self._tokenize_ru(sentence)
        return self._tokenize_simple(sentence)

    @staticmethod
    def _tokenize_ru(sentence: str):
        tokens, model_inputs, prediction_mask = [], [], []
        for word in _RE_RU_SPLIT.split(sentence):
            parts = word.split("-")
            if len(parts) == 1:
                cur_tokens = parts
                cur_pred_mask = [True]
            else:
                cur_tokens = [p + "-" for p in parts[:-1]] + [parts[-1]]
                cur_pred_mask = [True for _ in parts[:-1]] + [parts[-1] != "то"]
            cur_inputs = [_RE_RU_COND.sub("", t.lower()) for t in cur_tokens]
            cur_pred_mask = [
                (len(x) > 0) and bool(m)
                for x, m in zip(cur_inputs, cur_pred_mask)
            ]
            tokens.extend(cur_tokens)
            model_inputs.extend(cur_inputs)
            prediction_mask.extend(cur_pred_mask)
        return tokens, model_inputs, prediction_mask

    def _tokenize_simple(self, sentence: str):
        tokens, model_inputs, prediction_mask = [], [], []
        for word in re.split(r"([\s.,!?;:<>=()/\\]+)", sentence):
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

    # ------------------------------------------------------------------
    # Embedding pool
    # ------------------------------------------------------------------

    @staticmethod
    def _word_ngrams(text: str) -> list:
        grams = []
        t = "<" + text + ">"
        for i in range(1, len(text) + 3):
            for j in range(len(t) - i + 1):
                grams.append(t[j: j + i])
        if len(text) < 1:
            grams.append(text)
        return grams

    def _pool(self, words: list) -> np.ndarray:
        out = np.zeros((len(words), self._W.shape[1]), dtype=np.float32)
        for k, w in enumerate(words):
            ids = [
                self._ngram_dict[g]
                for g in self._word_ngrams(w)
                if g in self._ngram_dict
            ]
            if not ids:
                ids = [self._unk]
            out[k] = self._W[ids].mean(axis=0)
        return out

    # ------------------------------------------------------------------
    # ONNX inference
    # ------------------------------------------------------------------

    def _predict(self, clean_tokens: list):
        n = len(clean_tokens)
        stress_pred = np.zeros(n, dtype=np.int64)
        stress_prob = np.zeros(n, dtype=np.float32)
        yo_pred = np.zeros(n, dtype=np.int64)
        yo_prob = np.zeros(n, dtype=np.float32)
        idx = [i for i, w in enumerate(clean_tokens) if w]
        if not idx:
            return stress_pred, stress_prob, yo_pred, yo_prob
        pooled = self._pool([clean_tokens[i] for i in idx])
        outs = self._sess.run(None, {"pooled": pooled})
        s_logits = outs[0]
        y_logits = outs[1] if len(outs) > 1 else None
        s_prob = _softmax(s_logits)
        s_arg = s_prob.argmax(axis=1)
        for row, i in enumerate(idx):
            stress_pred[i] = s_arg[row]
            stress_prob[i] = s_prob[row, s_arg[row]]
        if y_logits is not None:
            y_prob = _softmax(y_logits)
            y_arg = y_prob.argmax(axis=1)
            for row, i in enumerate(idx):
                yo_pred[i] = y_arg[row]
                yo_prob[i] = y_prob[row, y_arg[row]]
        return stress_pred, stress_prob, yo_pred, yo_prob

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    def _get_positions(self, word: str, stressed_vowel_ids, yo_vowel_ids):
        vowel_ids = [i for i, c in enumerate(word) if c in self._vowels]
        ye_ids = [i for i, c in enumerate(word) if c == "е"]
        stress_positions = [
            vowel_ids[ix]
            for ix in stressed_vowel_ids
            if (ix < len(vowel_ids)) and vowel_ids
        ]
        yo_positions = [
            ye_ids[ix - 1]
            for ix in yo_vowel_ids
            if (ix > 0) and (ix - 1 < len(ye_ids)) and ye_ids
        ]
        num_vowels = len(vowel_ids)
        first_vowel_pos = vowel_ids[0] if vowel_ids else -1
        return stress_positions, yo_positions, num_vowels, first_vowel_pos

    def _get_positions_nyo(self, word: str, stressed_vowel_ids):
        """Simplified _get_positions for langs without yo logic."""
        vowel_ids = [i for i, c in enumerate(word) if c in self._vowels]
        stress_positions = [
            vowel_ids[ix]
            for ix in stressed_vowel_ids
            if (ix < len(vowel_ids)) and vowel_ids
        ]
        num_vowels = len(vowel_ids)
        first_vowel_pos = vowel_ids[0] if vowel_ids else -1
        return stress_positions, num_vowels, first_vowel_pos

    def _accentuate_exception(self, clean_word: str, raw_word: str) -> str:
        exc_stress, exc_yo = self._exceptions[clean_word]
        if self._has_yo and exc_yo != -1:
            raw_word = (
                raw_word[:exc_yo]
                + ("ё" if raw_word[exc_yo].islower() else "Ё")
                + raw_word[exc_yo + 1:]
            )
        return raw_word[:exc_stress] + STRESS_TOKEN + raw_word[exc_stress:]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, sentence: str) -> str:
        self._ensure_loaded()
        if self.lang == "ru":
            return self._process_ru(sentence)
        return self._process_ukr_bel(sentence)

    def _process_ukr_bel(self, sentence: str) -> str:
        """Decode path for ukr / bel (no yo logic)."""
        raw_tokens, clean_tokens, prediction_mask = self._tokenize(sentence)
        stress_preds, stress_probs, _yo_preds, _yo_probs = self._predict(clean_tokens)

        out = []
        for wi, (raw_word, clean_word, need) in enumerate(
            zip(raw_tokens, clean_tokens, prediction_mask)
        ):
            if not need:
                out.append(raw_word)
                continue

            raw_lower = raw_word.lower()
            have_stress = STRESS_TOKEN in raw_lower
            if have_stress:
                out.append(raw_word)
                continue

            if clean_word in self._exceptions:
                out.append(self._accentuate_exception(clean_word, raw_word))
                continue

            stressed_vowel_ids = [int(stress_preds[wi])]
            passed_stress = stress_probs[wi] > 0.5
            set_stress = (
                passed_stress
                and not have_stress
                and (clean_word not in self._skip_stress)
            )

            stress_positions, num_vowels, first_vowel_pos = (
                self._get_positions_nyo(raw_lower, stressed_vowel_ids)
            )

            if num_vowels == 0:
                out.append(raw_word)
                continue

            if num_vowels == 1:
                stress_positions = [first_vowel_pos]
                set_stress = True

            if not have_stress and set_stress:
                for i, sp in enumerate(stress_positions):
                    raw_word = raw_word[: sp + i] + STRESS_TOKEN + raw_word[sp + i:]

            out.append(raw_word)
        return "".join(out)

    def _process_ru(self, sentence: str) -> str:
        """Full decode path for Russian (includes yo logic)."""
        raw_tokens, clean_tokens, prediction_mask = self._tokenize(sentence)
        stress_preds, stress_probs, yo_preds, yo_probs = self._predict(clean_tokens)

        out = []
        for wi, (raw_word, clean_word, need) in enumerate(
            zip(raw_tokens, clean_tokens, prediction_mask)
        ):
            raw_lower = raw_word.lower()
            if not need:
                out.append(raw_word)
                continue

            have_stress = STRESS_TOKEN in raw_lower
            have_yo = "ё" in raw_lower
            if have_stress and have_yo:
                out.append(raw_word)
                continue
            if (not have_stress) and have_yo:
                if (
                    sum(c in self._vowels for c in raw_lower) == 1
                ) or clean_word.replace("ё", "е") in self._skip_stress:
                    out.append(raw_word)
                    continue
                user_yo = [i for i, x in enumerate(raw_lower) if x == "ё"]
                for i, yo_pos in enumerate(user_yo):
                    raw_word = (
                        raw_word[: yo_pos + i] + STRESS_TOKEN + raw_word[yo_pos + i:]
                    )
                out.append(raw_word)
                continue

            if clean_word in self._exceptions:
                out.append(self._accentuate_exception(clean_word, raw_word))
                continue

            stressed_vowel_ids = [int(stress_preds[wi])]
            passed_stress = stress_probs[wi] > 0.5
            set_stress = (
                passed_stress
                and not have_stress
                and (clean_word.replace("ё", "е") not in self._skip_stress)
            )

            yo_vowel_ids = [int(yo_preds[wi])]
            passed_yo = yo_probs[wi] > 0.5
            set_yo = passed_yo and (
                clean_word.replace("ё", "е") not in self._skip_yo
            )

            if have_stress:
                stressed_vowel_ids = [
                    sum(part.count(v) for v in self._vowels)
                    for part in raw_lower.split(STRESS_TOKEN)
                ]

            stress_positions, yo_positions, num_vowels, first_vowel_pos = (
                self._get_positions(raw_lower, stressed_vowel_ids, yo_vowel_ids)
            )

            if num_vowels == 0:
                out.append(raw_word)
                continue

            for yo_pos in yo_positions:
                if yo_pos in stress_positions and set_yo:
                    if raw_lower[yo_pos] == "е":
                        raw_word = (
                            raw_word[:yo_pos]
                            + ("ё" if raw_word[yo_pos].islower() else "Ё")
                            + raw_word[yo_pos + 1:]
                        )

            if num_vowels == 1:
                stress_positions = [first_vowel_pos]
                set_stress = True

            if not have_stress and set_stress:
                for i, sp in enumerate(stress_positions):
                    raw_word = raw_word[: sp + i] + STRESS_TOKEN + raw_word[sp + i:]

            out.append(raw_word)

        return "".join(out)


# ===========================================================================
# Simple-accentor family (vocabulary + rules)
# ===========================================================================

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
            raise ValueError(
                f"SimpleStressor supports {sorted(SIMPLE_LANGS)}; got {lang!r}."
            )
        self.lang = lang
        # HF artefacts live under the canonical lang name (strip _simple suffix)
        self._hf_lang = lang.removesuffix("_simple") if lang.endswith("_simple") else lang
        if cache_dir is None:
            base = os.path.join(
                os.path.expanduser("~"), ".local", "share", "stressonnx"
            )
            cache_dir = os.path.join(base, lang)
        self._cache_dir = cache_dir
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data = _download_files(self._hf_lang, self._cache_dir, _SIMPLE_FILES)

        with open(data["meta.json"], encoding="utf-8") as fh:
            meta = json.load(fh)

        self._vocab: dict = _load_vocab(data["vocab.gz"])
        self._vowels: str = meta["vowels"]
        self._oov_rule: str = meta.get("oov_rule", _OOV_RULES.get(self.lang, "last"))

        alpha = meta["alpha"]
        alpha_set = "".join(sorted(set(alpha + alpha.upper())))
        # Escape characters that are special in regex character classes
        escaped = re.escape(alpha_set)
        self._re_cond = re.compile(f"[^{escaped}]")

        self._loaded = True

    def _tokenize(self, sentence: str):
        tokens, model_inputs, prediction_mask = [], [], []
        for word in re.split(r"([\s.,!?;:<>=()/\\]+)", sentence):
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
        idx = self._vocab[clean_word]
        return raw_word[:idx] + STRESS_TOKEN + raw_word[idx:]

    def _accentuate_oov(self, raw_word: str) -> str:
        vowel_ids = [i for i, c in enumerate(raw_word.lower()) if c in self._vowels]
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
        elif rule == "kat":
            if len(vowel_ids) <= 3:
                idx = vowel_ids[0]
            else:
                idx = vowel_ids[-2]
        else:
            idx = vowel_ids[-1]
        return raw_word[:idx] + STRESS_TOKEN + raw_word[idx:]

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
            if STRESS_TOKEN in raw_word.lower():
                out.append(raw_word)
                continue
            if clean_word in self._vocab:
                out.append(self._accentuate_vocab(clean_word, raw_word))
            else:
                out.append(self._accentuate_oov(raw_word))
        return "".join(out)
