"""Main-accentor family (ukr / bel / ru) — silero neural pipeline.

Neural ONNX pipeline exported from silero_stress.  Embedding-bag + MLP
heads. Pipeline:
1. Tokenise sentence → (raw tokens, clean tokens, prediction mask).
2. Compute fastText-style n-gram embeddings for each clean token by
   mean-pooling the rows selected from the embedding matrix.
3. Run the ONNX MLP heads → stress_logits [N, K] (+ yo_logits for ``ru``).
4. Decode: exceptions dict → skip sets → argmax position → insert '+'.
"""
import threading
import unicodedata
import gzip
import json
import re

import numpy as np
import onnxruntime as ort

from stressonnx._common import _RE_RU_COND, _RU_VOWELS, _softmax, lower_preserving_length, tokenize
from stressonnx.download import _download_files
from stressonnx.errors import ModelDownloadError, ModelLoadError, UnsupportedLanguageError
from stressonnx.notation import STRESS_TOKEN, render_marks
from stressonnx.registry import MAIN_LANGS, _MAIN_FILES, hf_dir


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
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
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
                raise ModelLoadError('silero', exc) from exc
            self._loaded = True

    def _load(self) -> None:
        data = _download_files(hf_dir(self.lang, "silero"), _MAIN_FILES, self._cache_dir, model_id="silero")

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


    # ------------------------------------------------------------------
    # Tokenisation
    # ------------------------------------------------------------------

    def _tokenize(self, sentence: str):
        """Tokenise via the shared tokenizer; ru masks hyphen clitics."""
        if self.lang == "ru":
            return tokenize(sentence, _RE_RU_COND, mask_clitics=True)
        return tokenize(sentence, self._re_cond)

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

    def _exception_offsets(self, clean_word: str, raw_word: str, with_yo: bool):
        """(mark, yo) offsets within *raw_word* for an exceptions-dict entry."""
        exc_stress, exc_yo = self._exceptions[clean_word]
        yo = []
        if (
            with_yo
            and exc_yo != -1
            and 0 <= exc_yo < len(raw_word)
            and raw_word[exc_yo].lower() == "е"
        ):
            yo.append(exc_yo)
        lower = lower_preserving_length(raw_word)
        effective = dict(enumerate(lower))
        for o in yo:
            effective[o] = "ё"
        if 0 <= exc_stress < len(raw_word) and effective[exc_stress] in self._vowels:
            return [exc_stress], yo
        return [], yo

    def __call__(self, sentence: str) -> str:
        marks, yo = self.mark_offsets(sentence)
        return render_marks(sentence, marks, yo)

    def mark_offsets(self, text: str):
        """Offsets (into *text*) of stressed vowels and е→ё substitutions.

        With yo enabled (``ru``) this is a port of silero_stress's
        ``Accentor.__call__`` (default flags): ё in the input is stressed
        directly (ё is inherently stressed in Russian orthography), the
        model's е→ё prediction is applied only where it agrees with the
        predicted stress position, and skip/yo dictionaries are keyed on the
        е-spelling.  For ``uk``/``be`` every ё branch is inert and the same
        loop reduces to the plain decode.
        """
        self._ensure_loaded()
        with_yo = self._has_yo
        raw_tokens, clean_tokens, prediction_mask = self._tokenize(text)
        stress_preds, stress_probs, yo_preds, yo_probs = self._predict(clean_tokens)

        marks, yo_out = [], []
        pos = 0
        for wi, (raw_word, clean_word, need) in enumerate(
            zip(raw_tokens, clean_tokens, prediction_mask)
        ):
            start = pos
            pos += len(raw_word)
            if not need:
                continue

            raw_lower = lower_preserving_length(raw_word)
            if STRESS_TOKEN in unicodedata.normalize("NFD", raw_word):
                continue

            base_word = clean_word.replace("ё", "е") if with_yo else clean_word

            if with_yo and "ё" in raw_lower:
                # ё is inherently stressed — mark each ё the input already has
                if base_word not in self._skip_stress:
                    marks.extend(
                        start + i for i, c in enumerate(raw_lower) if c == "ё"
                    )
                continue

            if clean_word in self._exceptions:
                m, y = self._exception_offsets(clean_word, raw_word, with_yo)
                marks.extend(start + o for o in m)
                yo_out.extend(start + o for o in y)
                continue

            stressed_vowel_ids = [int(stress_preds[wi])]
            set_stress = (
                stress_probs[wi] > 0.5 and base_word not in self._skip_stress
            )
            yo_vowel_ids = [int(yo_preds[wi])] if with_yo else []

            stress_positions, yo_positions, num_vowels, first_vowel_pos = (
                self._get_positions(raw_lower, stressed_vowel_ids, yo_vowel_ids)
            )

            if num_vowels == 0:
                continue

            # е→ё restoration: only where the yo prediction lands on the
            # predicted stress position (ё must carry the stress)
            if with_yo and yo_probs[wi] > 0.5 and base_word not in self._skip_yo:
                yo_out.extend(
                    start + p
                    for p in yo_positions
                    if p in stress_positions and raw_lower[p] == "е"
                )

            if num_vowels == 1:
                stress_positions = [first_vowel_pos]
                set_stress = True

            if set_stress:
                marks.extend(start + p for p in stress_positions)
        return marks, yo_out
