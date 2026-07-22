"""Main-accentor family (ukr / bel / ru) — silero neural pipeline.

Neural ONNX pipeline exported from silero_stress.  Embedding-bag + MLP
heads. Pipeline:
1. Tokenise sentence → (raw tokens, clean tokens, prediction mask).
2. Compute fastText-style n-gram embeddings for each clean token by
   mean-pooling the rows selected from the embedding matrix.
3. Run the ONNX MLP heads → stress_logits [N, K] (+ yo_logits for ``ru``).
4. Decode: exceptions dict → skip sets → argmax position → insert '+'.
"""
import gzip
import json
import re

import numpy as np
import onnxruntime as ort

from stressonnx._common import _RE_RU_COND, _RE_SPLIT, _RU_VOWELS, _UNSTRESSED_HYPHEN_CLITICS, _softmax
from stressonnx.download import _download_files
from stressonnx.errors import UnsupportedLanguageError
from stressonnx.notation import STRESS_TOKEN, _insert_stress
from stressonnx.registry import MAIN_LANGS, _MAIN_FILES


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


class _SileroStressor:
    """Internal: lazy-load neural (silero) accentor for ``ukr``, ``bel``, or ``ru``.

    Use :class:`Stressor` (the public wrapper) instead of this class directly.

    Parameters
    ----------
    lang:
        Language tag: ``"ukr"``, ``"bel"``, or ``"ru"``.
    cache_dir:
        Override the model storage directory (default: the standard
        Hugging Face cache).
    """

    def __init__(self, lang: str, cache_dir: str | None = None) -> None:
        if lang not in MAIN_LANGS:
            raise UnsupportedLanguageError(lang, MAIN_LANGS)
        self.lang = lang
        self._cache_dir = cache_dir
        self._loaded = False

    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data = _download_files(self.lang, _MAIN_FILES, self._cache_dir)

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
        for word in _RE_SPLIT.split(sentence):
            parts = word.split("-")
            if len(parts) == 1:
                cur_tokens = parts
                cur_pred_mask = [True]
            else:
                cur_tokens = [p + "-" for p in parts[:-1]] + [parts[-1]]
                cur_pred_mask = [True for _ in parts[:-1]] + [
                    parts[-1] not in _UNSTRESSED_HYPHEN_CLITICS
                ]
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
        if (
            self._has_yo
            and exc_yo != -1
            and 0 <= exc_yo < len(raw_word)
            and raw_word[exc_yo].lower() == "е"
        ):
            raw_word = (
                raw_word[:exc_yo]
                + ("ё" if raw_word[exc_yo].islower() else "Ё")
                + raw_word[exc_yo + 1:]
            )
        return _insert_stress(raw_word, exc_stress, self._vowels)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def __call__(self, sentence: str) -> str:
        self._ensure_loaded()
        if self._has_yo:
            return self._process_yo_lang(sentence)
        return self._process_ukr_bel(sentence)

    def _process_yo_lang(self, sentence: str) -> str:
        """Decode path for languages with е→ё restoration (``ru``).

        Port of silero_stress's ``Accentor.__call__`` (default flags), with
        the stress mark adapted from ``+``-before-vowel to combining acute
        after the vowel.  In Russian orthography ``ё`` is always stressed, so
        a word that already contains ``ё`` is stressed on it directly, and
        the model's yo prediction is only applied when it agrees with the
        predicted stress position.
        """
        raw_tokens, clean_tokens, prediction_mask = self._tokenize(sentence)
        stress_preds, stress_probs, yo_preds, yo_probs = self._predict(clean_tokens)

        out = []
        for wi, (raw_word, clean_word, need) in enumerate(
            zip(raw_tokens, clean_tokens, prediction_mask)
        ):
            if not need:
                out.append(raw_word)
                continue

            raw_lower = raw_word.lower()
            have_stress = STRESS_TOKEN in raw_word
            if have_stress:
                out.append(raw_word)
                continue

            # skip/yo dictionaries are keyed on the е-spelling
            base_word = clean_word.replace("ё", "е")

            if "ё" in raw_lower:
                # ё is inherently stressed — mark each ё the input already has
                if base_word in self._skip_stress:
                    out.append(raw_word)
                    continue
                yo_char_pos = [i for i, c in enumerate(raw_lower) if c == "ё"]
                for i, p in enumerate(yo_char_pos):
                    raw_word = raw_word[: p + i + 1] + STRESS_TOKEN + raw_word[p + i + 1:]
                out.append(raw_word)
                continue

            if clean_word in self._exceptions:
                out.append(self._accentuate_exception(clean_word, raw_word))
                continue

            stressed_vowel_ids = [int(stress_preds[wi])]
            set_stress = (
                stress_probs[wi] > 0.5 and base_word not in self._skip_stress
            )
            yo_vowel_ids = [int(yo_preds[wi])]
            set_yo = yo_probs[wi] > 0.5 and base_word not in self._skip_yo

            stress_positions, yo_positions, num_vowels, first_vowel_pos = (
                self._get_positions(raw_lower, stressed_vowel_ids, yo_vowel_ids)
            )

            if num_vowels == 0:
                out.append(raw_word)
                continue

            # е→ё restoration: only where the yo prediction lands on the
            # predicted stress position (ё must carry the stress)
            if set_yo:
                for yo_pos in yo_positions:
                    if yo_pos in stress_positions and raw_lower[yo_pos] == "е":
                        raw_word = (
                            raw_word[:yo_pos]
                            + ("ё" if raw_word[yo_pos].islower() else "Ё")
                            + raw_word[yo_pos + 1:]
                        )

            if num_vowels == 1:
                stress_positions = [first_vowel_pos]
                set_stress = True

            if set_stress:
                for i, sp in enumerate(stress_positions):
                    raw_word = raw_word[: sp + i + 1] + STRESS_TOKEN + raw_word[sp + i + 1:]

            out.append(raw_word)
        return "".join(out)

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
            have_stress = STRESS_TOKEN in raw_word
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
                    raw_word = raw_word[: sp + i + 1] + STRESS_TOKEN + raw_word[sp + i + 1:]

            out.append(raw_word)
        return "".join(out)
