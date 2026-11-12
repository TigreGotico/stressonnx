"""Kubataba family (ru): character-level encoder-decoder Transformer.

Port of kubataba/Russian-Stress-Accent-Predictor (MIT).  The model reads a
sentence as a character sequence and generates a copy with an apostrophe
after each stressed vowel.  Two ONNX graphs replace the torch model: the
encoder (``src [1, 256] → memory``) and one decoder step
(``memory, tgt [1, t] → logits``); the greedy loop runs here in numpy.

A generated copy is not the input: the model drops or replaces characters
it has never seen and never emits a mark on ё or on a single-vowel word.
The generated text is aligned back onto the input, and only an apostrophe
that follows a vowel the input has at that place becomes a mark, so every
input character comes back and the output never contains model-invented
text.
"""
import difflib
import json
import threading

import numpy as np
import onnxruntime as ort

from stressonnx._common import _RE_RU_COND, _RU_VOWELS, tokenize
from stressonnx.download import _download_files
from stressonnx.errors import ModelDownloadError, ModelLoadError
from stressonnx.notation import STRESS_TOKEN, composed, has_stress_mark, in_form_of, render_marks
from stressonnx.registry import MODEL_REGISTRY, _KUBATABA_FILES

#: Model sequence length: <s> + up to 254 characters + </s>.
_MAX_LEN = 256
_MAX_CHARS = _MAX_LEN - 2
#: Input characters per window.  The output shares the 256-token budget
#: with the marks it adds (a marked word has at least two vowels, so at
#: most one mark per four characters): 200 characters plus their marks
#: always fit, where a 254-character window loses its tail.
_WINDOW_CHARS = 200
_MARK = "'"


def _windows(text: str, limit: int = _WINDOW_CHARS):
    """``(start, chunk)`` pieces of *text* at most *limit* characters long.

    Cuts fall on whitespace where one exists inside the limit; a run of
    non-space characters longer than *limit* is cut hard.
    """
    pos = 0
    while pos < len(text):
        end = min(pos + limit, len(text))
        if end < len(text):
            cut = next((i for i in range(end, pos, -1) if text[i].isspace()), pos)
            if cut > pos:
                end = cut
        yield pos, text[pos:end]
        pos = end


class KubatabaStressor:
    """Lazy-loading Russian stressor backed by the kubataba Transformer.

    Use :class:`Stressor` or :func:`stress` with ``model="kubataba"``
    instead of this class directly.

    Parameters
    ----------
    cache_dir:
        Override the model storage directory (default: the standard
        Hugging Face cache).
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        self.lang = "ru"
        self._cache_dir = cache_dir
        self._loaded = False
        self._load_lock = threading.Lock()

    def _ensure_loaded(self) -> None:
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
                raise ModelLoadError("kubataba", exc) from exc
            self._loaded = True

    def _load(self) -> None:
        data = _download_files(
            MODEL_REGISTRY["kubataba"].hf_subdir, _KUBATABA_FILES,
            self._cache_dir, model_id="kubataba",
        )
        self._enc = ort.InferenceSession(data["encoder.onnx"], providers=["CPUExecutionProvider"])
        self._dec = ort.InferenceSession(data["decoder_step.onnx"], providers=["CPUExecutionProvider"])
        with open(data["vocab.json"], encoding="utf-8") as fh:
            self._vocab: dict = json.load(fh)
        self._idx2char = {v: k for k, v in self._vocab.items()}
        self._pad = self._vocab["<pad>"]
        self._bos = self._vocab["<s>"]
        self._eos = self._vocab["</s>"]
        self._unk = self._vocab["<unk>"]

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    def _generate(self, text: str) -> str:
        """The model's own output for one window: text with ``'`` marks."""
        ids = [self._bos] + [self._vocab.get(c, self._unk) for c in text[:_MAX_CHARS]] + [self._eos]
        ids += [self._pad] * (_MAX_LEN - len(ids))
        memory = self._enc.run(["memory"], {"src": np.array([ids], dtype=np.int64)})[0]
        tgt = np.array([[self._bos]], dtype=np.int64)
        out = []
        while len(out) < _MAX_LEN - 1:
            logits = self._dec.run(["logits"], {"memory": memory, "tgt": tgt})[0]
            tok = int(logits[0].argmax())
            if tok == self._eos:
                break
            out.append(tok)
            tgt = np.concatenate([tgt, [[tok]]], axis=1)
        return "".join(self._idx2char.get(t, "") for t in out)

    # ------------------------------------------------------------------
    # Alignment
    # ------------------------------------------------------------------

    @staticmethod
    def _aligned_marks(window: str, generated: str):
        """Offsets into *window* of the vowels the model marked.

        The generated copy without its marks is aligned to *window*; a mark
        counts only where the generated vowel and the input vowel are the
        same character at matched positions, and the input itself has no
        apostrophe right after it (the model echoes input apostrophes).
        A matched run of one character is noise: the model's invented tail
        can share a letter with a real word and give that word a second
        mark.
        """
        plain, marked_at = [], []
        for ch in generated:
            if ch == _MARK and plain and plain[-1].lower() in _RU_VOWELS:
                marked_at.append(len(plain) - 1)
            else:
                plain.append(ch)
        plain = "".join(plain)
        matcher = difflib.SequenceMatcher(None, window, plain, autojunk=False)
        gen_to_win = {}
        for a, b, n in matcher.get_matching_blocks():
            if n < 2:
                continue
            for k in range(n):
                gen_to_win[b + k] = a + k
        offsets = []
        for j in marked_at:
            i = gen_to_win.get(j)
            if i is None or (i + 1 < len(window) and window[i + 1] == _MARK):
                continue
            offsets.append(i)
        return offsets

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> str:
        marks, yo = self.mark_offsets(text)
        return in_form_of(text, render_marks(text, marks, yo))

    def mark_offsets(self, text: str):
        """Offsets into *text* of stressed vowels; the ё list is always empty.

        The work happens on the NFC form of *text*; each offset points at
        the base character in *text* itself.
        """
        nfc, origin = composed(text)
        marks = self._composed_mark_offsets(nfc)
        return [origin[o] for o in marks], []

    def _composed_mark_offsets(self, text: str):
        """Offsets into NFC *text* of the vowels to mark.

        Stress marks already in the input are removed before the model
        reads the text, and a word that carries one keeps it and gets no
        new mark, the same contract as ``silero``.
        """
        self._ensure_loaded()
        plain, plain_to_text = [], []
        for i, ch in enumerate(text):
            if ch != STRESS_TOKEN:
                plain.append(ch)
                plain_to_text.append(i)
        plain = "".join(plain)

        protected = set()
        pos = 0
        for raw in tokenize(text, _RE_RU_COND).raw:
            if has_stress_mark(raw):
                protected.update(range(pos, pos + len(raw)))
            pos += len(raw)

        offsets = []
        for start, window in _windows(plain):
            for o in self._aligned_marks(window, self._generate(window)):
                t = plain_to_text[start + o]
                if t not in protected:
                    offsets.append(t)
        return sorted(offsets)
