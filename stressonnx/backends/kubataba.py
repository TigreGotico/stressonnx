"""Kubataba family — char-level encoder-decoder Transformer (Russian)."""
import json

import numpy as np
import onnxruntime as ort

from stressonnx.download import _download_files
from stressonnx.notation import STRESS_TOKEN, _apostrophe_to_diacritic
from stressonnx.registry import _KUBATABA_FILES


class _KubatabaStressor:
    """Char-level encoder-decoder Transformer for Russian stress (kubataba, MIT).

    Alternative to :class:`RuAccentStressor`.  Simpler pipeline; no homograph
    disambiguation.  Uses two ONNX graphs (encoder + single decoder step) with
    the autoregressive greedy loop running in pure Python/numpy.

    Parameters
    ----------
    cache_dir:
        Override the model storage directory (default: the standard
        Hugging Face cache).
    """

    _MAX_LEN = 256

    def __init__(self, cache_dir: str | None = None) -> None:
        self._cache_dir = cache_dir
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        data = _download_files("ru_kubataba", _KUBATABA_FILES, self._cache_dir)
        self._enc_sess = ort.InferenceSession(
            data["encoder.onnx"], providers=["CPUExecutionProvider"]
        )
        self._dec_sess = ort.InferenceSession(
            data["decoder_step.onnx"], providers=["CPUExecutionProvider"]
        )
        with open(data["vocab.json"], encoding="utf-8") as fh:
            self._vocab: dict = json.load(fh)
        self._idx2char: dict = {v: k for k, v in self._vocab.items()}
        self._pad = self._vocab.get("<pad>", 0)
        self._bos = self._vocab.get("<s>", 1)
        self._eos = self._vocab.get("</s>", 2)
        self._unk = self._vocab.get("<unk>", 3)
        self._loaded = True

    def _encode_text(self, text: str) -> np.ndarray:
        indices = [self._bos]
        for ch in text[:254]:
            indices.append(self._vocab.get(ch, self._unk))
        indices.append(self._eos)
        indices += [self._pad] * (self._MAX_LEN - len(indices))
        return np.array([indices[:self._MAX_LEN]], dtype=np.int64)

    def _decode_tokens(self, tgt: np.ndarray) -> str:
        chars = []
        for idx in tgt[0, 1:].tolist():
            if idx == self._eos:
                break
            chars.append(self._idx2char.get(idx, ""))
        return _apostrophe_to_diacritic("".join(chars))

    def _accent_text(self, text: str) -> str:
        """Run encoder-decoder inference on a full sentence/phrase."""
        src = self._encode_text(text)
        memory = self._enc_sess.run(["memory"], {"src": src})[0]
        tgt = np.array([[self._bos]], dtype=np.int64)
        for _ in range(self._MAX_LEN):
            logits = self._dec_sess.run(["logits"], {"memory": memory, "tgt": tgt})[0]
            next_tok = int(np.argmax(logits[0]))
            tgt = np.concatenate([tgt, [[next_tok]]], axis=1)
            if next_tok == self._eos:
                break
        return self._decode_tokens(tgt)

    def __call__(self, text: str) -> str:
        self._ensure_loaded()
        # Re-derive semantics: existing marks are stripped (the char-level
        # model has no vocab entry for U+0301 and would emit garbage), then
        # stress is predicted from scratch — same contract as RuAccentStressor.
        return self._accent_text(text.replace(STRESS_TOKEN, ""))
