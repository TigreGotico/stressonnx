"""uk_accentor family (uk) — the ukrainian-accentor-transformer seq2seq model.

Attribution: ukrainian-accentor-transformer by Bohdan Mykhailenko
(https://github.com/Theodotus1243/ukrainian-accentor-transformer, MIT), a
4+4-layer Transformer with a shared SentencePiece vocabulary that rewrites a
Ukrainian sentence with U+0301 after each stressed vowel.  It was trained on
the lang-uk News corpus stressed with ukrainian-word-stress.  Runtime:
onnxruntime + numpy + tokenizers.

Pipeline, a port of ``Accentor.__call__`` with its defaults (beam 2,
repetition penalty 1.2, length penalty 1):

1. Strip U+0301 from the text and encode it with the SentencePiece model.
2. Cut the piece sequence after every ``.``, ``,``, ``!`` and ``?`` piece,
   then halve every chunk of 30 or more pieces at the word start nearest
   its middle until every chunk is shorter than 30.
3. Translate each chunk with ``encoder.onnx`` and ``decoder.onnx``, beam
   search in numpy; join the hypotheses and decode them to text.
4. Align the decoded text with the stripped input (``difflib``): a U+0301
   the model inserted becomes a mark on the input character before it when
   that character is a vowel; every other difference is dropped.  The
   result is the input with marks inserted, never a letter the model
   rewrote.

A word that already carries a stress mark keeps it, and the model's marks on
that word are dropped; upstream strips and re-marks every word.  Upstream
also reads the text in its original form, where this port works on the NFC
form and maps every offset back.  Upstream decodes up to 256 pieces per
chunk whatever the input; this port stops at ``2 * len(pieces) + 20``, which
only cuts short the inputs the model never ends with an end token.
"""
import difflib
import json
import threading
from importlib.metadata import version

import numpy as np
import onnxruntime as ort

from stressonnx.download import _download_files
from stressonnx.errors import ModelDownloadError, ModelLoadError
from stressonnx.notation import STRESS_TOKEN, composed, has_stress_mark, in_form_of, render_marks
from stressonnx.registry import _UK_ACCENTOR_FILES, LANGUAGES, hf_dir

_MAX_CHUNK = 30
_SPLIT_PIECES = frozenset({".", ",", "!", "?"})
_BEAM = 2
_REPETITION_PENALTY = 1.2
_MAX_DECODING_LENGTH = 256
_MIN_DECODING_LENGTH = 1
_DECODING_LENGTH_MARGIN = 20
_UNK_SURFACE = " ⁇ "
# the model also writes U+0301 after a consonant (хр́е́ще́ною́); only a
# vowel can carry stress, and ``is_stress_base`` would keep р (syllabic in
# Slovene, never in Ukrainian)
_VOWELS = frozenset(LANGUAGES["uk"]["vowels"])


def _split_at_punctuation(pieces: list) -> list:
    """Chunks that end after a sentence-splitting piece; upstream
    ``_split_punctuation_sentence``."""
    chunks = []
    start = 0
    for i, piece in enumerate(pieces, start=1):
        if piece in _SPLIT_PIECES:
            chunks.append(pieces[start:i])
            start = i
    if start < len(pieces):
        chunks.append(pieces[start:])
    return chunks


def _middle_word_start(chunk: list) -> int:
    middle = len(chunk) // 2
    for shift in range(len(chunk) // 10):
        if chunk[middle - shift][0] == "▁":
            return middle - shift
        if chunk[middle + shift][0] == "▁":
            return middle + shift
    return middle


def _split_long(chunks: list) -> list:
    """Halve chunks of ``_MAX_CHUNK`` or more pieces until none is left;
    upstream ``_split_long``."""
    while True:
        short = []
        for chunk in chunks:
            if len(chunk) < _MAX_CHUNK:
                short.append(chunk)
            else:
                middle = _middle_word_start(chunk)
                short.append(chunk[:middle])
                short.append(chunk[middle:])
        if short == chunks:
            return short
        chunks = short


def _log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


class UkAccentorStressor:
    """Lazy-loading port of ukrainian-accentor-transformer for ``uk``.

    Use :class:`Stressor` (the public wrapper) instead of this class directly.

    Parameters
    ----------
    cache_dir:
        Override the model storage directory (default: the standard
        Hugging Face cache).
    """

    lang = "uk"

    def __init__(self, cache_dir: str | None = None) -> None:
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
                raise ModelLoadError("uk-accentor", exc) from exc
            self._loaded = True

    def _load(self) -> None:
        from tokenizers import Tokenizer

        data = _download_files(hf_dir(self.lang, "uk_accentor"), _UK_ACCENTOR_FILES,
                               self._cache_dir, model_id="uk-accentor")
        tokenizer = Tokenizer.from_file(data["tokenizer.json"])
        # on tokenizers 1.x only the CompiledTokenizer that compile() builds can encode
        if int(version("tokenizers").split(".")[0]) >= 1:
            tokenizer = tokenizer.compile()
        self._tokenizer = tokenizer
        with open(data["tokenizer.json"], encoding="utf-8") as fh:
            self._tokenizer_pieces = [piece for piece, _ in json.load(fh)["model"]["vocab"]]
        # decoder rows follow vocab.txt (CTranslate2 order: <blank> first,
        # <unk> last); tokenizer ids follow the SentencePiece order.  Both
        # are joined through the piece string.
        with open(data["vocab.txt"], encoding="utf-8") as fh:
            self._pieces = fh.read().split("\n")[:-1]
        self._rows = {piece: row for row, piece in enumerate(self._pieces)}
        self._bos = self._rows["<s>"]
        self._eos = self._rows["</s>"]
        self._unk = self._rows["<unk>"]
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        self._encoder = ort.InferenceSession(data["encoder.onnx"], sess_options=opts,
                                             providers=["CPUExecutionProvider"])
        self._decoder = ort.InferenceSession(data["decoder.onnx"], sess_options=opts,
                                             providers=["CPUExecutionProvider"])

    # ------------------------------------------------------------------
    # Pieces
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> list:
        """SentencePiece pieces of *text*; an unknown character is ``<unk>``."""
        return [self._tokenizer_pieces[i] for i in self._tokenizer.encode(text).ids]

    def _decode(self, pieces: list) -> str:
        """SentencePiece decoding of *pieces*: ``▁`` is a space, the dummy
        prefix goes, ``<unk>`` prints its surface, control pieces print
        nothing."""
        out = []
        for piece in pieces:
            if piece == "<unk>":
                out.append(_UNK_SURFACE)
            elif piece in ("<s>", "</s>", "<blank>"):
                continue
            else:
                out.append(piece)
        text = "".join(out).replace("▁", " ")
        return text[1:] if text.startswith(" ") else text

    # ------------------------------------------------------------------
    # Beam search, as CTranslate2 runs it: beam 2, patience 1, length
    # penalty 1 over the length including the end token, repetition
    # penalty 1.2 on the logits, one hypothesis, no sampling
    # ------------------------------------------------------------------

    def _translate(self, pieces: list) -> list:
        """Output pieces for one chunk of input pieces."""
        src = np.array([[self._rows.get(p, self._unk) for p in pieces]], dtype=np.int64)
        memory = self._encoder.run(None, {"src_ids": src})[0]
        memory = np.repeat(memory, _BEAM, axis=0)
        alive = np.zeros((_BEAM, 0), dtype=np.int64)
        scores = np.array([0.0] + [-np.inf] * (_BEAM - 1), dtype=np.float32)
        finished = []
        # the decoder reads the whole prefix at every step, so a chunk the
        # model never ends costs 256**2 / 2 token positions without this
        # bound; the output of a chunk the model does end is unchanged
        max_length = min(_MAX_DECODING_LENGTH, 2 * len(pieces) + _DECODING_LENGTH_MARGIN)
        for step in range(max_length):
            prefix = np.concatenate([np.full((_BEAM, 1), self._bos, dtype=np.int64), alive], axis=1)
            logits = self._decoder.run(None, {"prefix_ids": prefix, "memory": memory})[0]
            if step < _MIN_DECODING_LENGTH:
                logits[:, self._eos] = -np.inf
            if alive.shape[1]:
                for b in range(_BEAM):
                    previous = logits[b, alive[b]]
                    logits[b, alive[b]] = np.where(previous < 0, previous * _REPETITION_PENALTY,
                                                   previous / _REPETITION_PENALTY)
            flat = (_log_softmax(logits) + scores[:, None]).reshape(-1)
            candidates = np.argsort(-flat, kind="stable")[: 2 * _BEAM]
            vocab = logits.shape[1]
            cand_beam, cand_piece, cand_score = candidates // vocab, candidates % vocab, flat[candidates]
            last = step + 1 == max_length
            kept = []
            spare = _BEAM
            for k in range(_BEAM):
                choice = k
                if cand_piece[k] == self._eos or last:
                    hypothesis = list(alive[cand_beam[k]])
                    if cand_piece[k] != self._eos:
                        hypothesis.append(int(cand_piece[k]))
                    # the length penalty counts the end token (CTranslate2
                    # include_eos_in_hypotheses), so every hypothesis that
                    # ends at this step has length step + 1
                    finished.append((float(cand_score[k]) / (step + 1), hypothesis))
                    for j in range(spare, 2 * _BEAM):
                        if cand_piece[j] != self._eos:
                            choice = j
                            spare = j + 1
                            break
                kept.append(choice)
            if len(finished) >= _BEAM or last:
                best = max(finished, key=lambda f: f[0])
                return [self._pieces[row] for row in best[1]]
            alive = np.stack([np.append(alive[cand_beam[j]], cand_piece[j]) for j in kept])
            scores = cand_score[kept].astype(np.float32)
        raise AssertionError("unreachable")

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def __call__(self, sentence: str) -> str:
        marks, yo = self.mark_offsets(sentence)
        return in_form_of(sentence, render_marks(sentence, marks, yo))

    def mark_offsets(self, text: str):
        """Offsets into *text* of stressed characters, and an empty list:
        this model never rewrites е as ё.

        The work happens on the NFC form of *text*; each offset points at
        the base character in *text* itself.
        """
        nfc, origin = composed(text)
        marks = self._composed_mark_offsets(nfc)
        return [origin[o] for o in marks], []

    def _composed_mark_offsets(self, text: str) -> list:
        self._ensure_loaded()
        clean_to_text = [i for i, ch in enumerate(text) if ch != STRESS_TOKEN]
        clean = "".join(text[i] for i in clean_to_text)
        chunks = _split_long(_split_at_punctuation(self._encode(clean)))
        output = self._decode([piece for chunk in chunks for piece in self._translate(chunk)])
        marks = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(
                a=output, b=clean, autojunk=False).get_opcodes():
            if (tag == "delete" and i2 - i1 == 1 and output[i1] == STRESS_TOKEN and j1 > 0
                    and clean[j1 - 1].lower() in _VOWELS):
                marks.append(j1 - 1)
        return [clean_to_text[o] for o in marks if not self._word_is_marked(text, clean_to_text[o])]

    @staticmethod
    def _word_is_marked(text: str, offset: int) -> bool:
        """True when the whitespace-delimited word around *offset* already
        carries a stress mark."""
        start = offset
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        end = offset
        while end < len(text) and not text[end].isspace():
            end += 1
        return has_stress_mark(text[start:end])
