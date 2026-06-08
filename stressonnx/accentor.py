"""Per-language stress accentor backed by ONNX + numpy.

Models and data are downloaded from HuggingFace on first use and cached under
``~/.local/share/stressonnx/<lang>/``.

The pipeline (Russian / silero-style)
--------------------------------------
1. Tokenise sentence → (raw tokens, clean tokens, prediction mask).
2. Compute fastText-style n-gram embeddings for each clean token by
   mean-pooling the rows selected from the embedding matrix.
3. Run the ONNX MLP heads → (stress_logits [N, K], yo_logits [N, K]).
4. Decode: exceptions dict → skip sets → argmax position → insert '+'.
"""
import gzip
import os
import re

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# HF repo that hosts the per-language runtime artefacts
# ---------------------------------------------------------------------------
HF_REPO_ID = "TigreGotico/stressonnx-models"

# Files expected under <repo>/<lang>/
_LANG_FILES = [
    "accentor.onnx",
    "embedding.npy",
    "ngram_dict.txt.gz",
    "exceptions.txt.gz",
    "skip_stress_words.txt.gz",
    "skip_yo_words.txt.gz",
    "meta.json",
]

STRESS_TOKEN = "+"
VOWELS = "аоуыэиеяёю"
_RE_COND = re.compile(r"[^А-Яа-яёЁ]")
_RE_SPLIT = re.compile(r"([\s.,!?;:<>=()/\\]+)")


# ---------------------------------------------------------------------------
# Data loaders
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


# ---------------------------------------------------------------------------
# Softmax helper
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Stressor
# ---------------------------------------------------------------------------

class Stressor:
    """Lazy-load accentor for a single language.

    Parameters
    ----------
    lang:
        BCP-47-like language tag, e.g. ``"ru"``, ``"uk"``, ``"be"``.
    cache_dir:
        Override the default cache location
        (``~/.local/share/stressonnx/<lang>``).
    """

    def __init__(self, lang: str = "ru", cache_dir: str | None = None) -> None:
        self.lang = lang
        if cache_dir is None:
            base = os.path.join(
                os.path.expanduser("~"), ".local", "share", "stressonnx"
            )
            cache_dir = os.path.join(base, lang)
        self._cache_dir = cache_dir
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data = self._download_lang(self.lang, self._cache_dir)
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
        self._loaded = True

    @staticmethod
    def _download_lang(lang: str, cache_dir: str) -> dict:
        """Download all artefacts for *lang* and return a {filename: local_path} map."""
        os.makedirs(cache_dir, exist_ok=True)
        paths = {}
        for fname in _LANG_FILES:
            local = os.path.join(cache_dir, fname)
            if os.path.exists(local):
                paths[fname] = local
                continue
            downloaded = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=f"{lang}/{fname}",
                local_dir=cache_dir,
                local_dir_use_symlinks=False,
            )
            # hf_hub_download may nest under lang/ — normalise
            if os.path.basename(downloaded) == fname:
                actual = downloaded
            else:
                actual = os.path.join(cache_dir, lang, fname)
            paths[fname] = actual
        return paths

    # ------------------------------------------------------------------
    # Tokenisation (mirrors AccentorNgram._tokenize)
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(sentence: str):
        tokens, model_inputs, prediction_mask = [], [], []
        for word in _RE_SPLIT.split(sentence):
            parts = word.split("-")
            if len(parts) == 1:
                cur_tokens = parts
                cur_pred_mask = [True]
            else:
                cur_tokens = [p + "-" for p in parts[:-1]] + [parts[-1]]
                cur_pred_mask = [True for _ in parts[:-1]] + [parts[-1] != "то"]
            cur_inputs = [_RE_COND.sub("", t.lower()) for t in cur_tokens]
            cur_pred_mask = [
                (len(x) > 0) and bool(m)
                for x, m in zip(cur_inputs, cur_pred_mask)
            ]
            tokens.extend(cur_tokens)
            model_inputs.extend(cur_inputs)
            prediction_mask.extend(cur_pred_mask)
        return tokens, model_inputs, prediction_mask

    # ------------------------------------------------------------------
    # Embedding pool (mirrors NewFastTextEmbeddingBag + word_ngrams)
    # ------------------------------------------------------------------

    @staticmethod
    def _word_ngrams(text: str) -> list:
        grams = []
        t = "<" + text + ">"
        for i in range(1, len(text) + 3):
            for j in range(len(t) - i + 1):
                grams.append(t[j : j + i])
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
        s_logits, y_logits = self._sess.run(None, {"pooled": pooled})
        s_prob = _softmax(s_logits)
        y_prob = _softmax(y_logits)
        s_arg = s_prob.argmax(axis=1)
        y_arg = y_prob.argmax(axis=1)
        for row, i in enumerate(idx):
            stress_pred[i] = s_arg[row]
            stress_prob[i] = s_prob[row, s_arg[row]]
            yo_pred[i] = y_arg[row]
            yo_prob[i] = y_prob[row, y_arg[row]]
        return stress_pred, stress_prob, yo_pred, yo_prob

    # ------------------------------------------------------------------
    # Decoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_positions(word: str, stressed_vowel_ids, yo_vowel_ids):
        vowel_ids = [i for i, c in enumerate(word) if c in VOWELS]
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

    def _accentuate_exception(self, clean_word: str, raw_word: str) -> str:
        exc_stress, exc_yo = self._exceptions[clean_word]
        if exc_yo != -1:
            raw_word = (
                raw_word[:exc_yo]
                + ("ё" if raw_word[exc_yo].islower() else "Ё")
                + raw_word[exc_yo + 1 :]
            )
        return raw_word[:exc_stress] + STRESS_TOKEN + raw_word[exc_stress:]

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, sentence: str) -> str:
        self._ensure_loaded()
        raw_tokens, clean_tokens, prediction_mask = self._tokenize(sentence)
        stress_preds, stress_probs, yo_preds, _yo_probs = self._predict(clean_tokens)

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
                    sum(c in VOWELS for c in raw_lower) == 1
                ) or clean_word.replace("ё", "е") in self._skip_stress:
                    out.append(raw_word)
                    continue
                user_yo = [i for i, x in enumerate(raw_lower) if x == "ё"]
                for i, yo_pos in enumerate(user_yo):
                    raw_word = raw_word[: yo_pos + i] + STRESS_TOKEN + raw_word[yo_pos + i :]
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
            passed_yo = _yo_probs[wi] > 0.5
            set_yo = passed_yo and (
                clean_word.replace("ё", "е") not in self._skip_yo
            )

            if have_stress:
                stressed_vowel_ids = [
                    sum(part.count(v) for v in VOWELS)
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
                            + raw_word[yo_pos + 1 :]
                        )

            if num_vowels == 1:
                stress_positions = [first_vowel_pos]
                set_stress = True

            if not have_stress and set_stress:
                for i, sp in enumerate(stress_positions):
                    raw_word = raw_word[: sp + i] + STRESS_TOKEN + raw_word[sp + i :]

            out.append(raw_word)

        return "".join(out)
