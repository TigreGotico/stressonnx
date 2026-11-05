"""StressRNN family (ru) — character BiLSTM with a Zaliznyak exception dictionary.

Attribution: StressRNN by Vladislav Klim (https://github.com/dbklim/StressRNN,
Apache-2.0), a maintained fork of RusStress by Maria Ponomareva and Kirill
Milintsevich (Ponomareva et al. 2017, "Automated Word Stress Detection in
Russian").  The ONNX graph is the upstream ``models/model.onnx`` verbatim; the
exception dictionary is the upstream ``dicts/exception_dictionary.txt``, built
from A. A. Zaliznyak's Grammatical Dictionary.  Runtime: onnxruntime + numpy.

Pipeline, a port of ``StressRNN.put_stress`` with its defaults
(``accuracy_threshold=0.75``, ``replace_similar_symbols=False``,
``lemmatize_words=False``):

1. Split the text twice: into *tokens* (runs of letters, digits, spaces or
   punctuation) that keep every character, and into lowercase *words* with
   clause marks (``_``) where the text had ``.,:?!…-`` or a newline.
2. Each word with two or more vowels gets the last three letters of the
   previous word as context: ``ние_привет``.
3. A word with one vowel is stressed on it.  A word with two or more vowels
   is looked up in the exception dictionary, and otherwise one-hot encoded
   (40 positions, 36 symbols) and scored by the BiLSTM.  A prediction below
   the threshold leaves the word unmarked.
4. The stressed words are transferred back onto the tokens in order, so the
   output keeps the input's casing, punctuation and spacing.

Upstream reads ``+`` and ``'`` as stress marks already in the text; here only
U+0301 is, and ``+`` and ``'`` are word boundaries like any other symbol.
Any whitespace separates words; upstream splits on the space only, so a tab
desynchronises its transfer step and the rest of the text stays unmarked.  A
prediction that lands in the padding before the word is read modulo the word
length, the vowel upstream's negative index names, instead of the mark
upstream puts before the first letter.  Words longer than 40 characters with
their context are left unmarked.  Hyphen parts and monosyllables, including
particles and prepositions, are stressed as upstream does.
"""
import gzip
import re
import threading

import numpy as np
import onnxruntime as ort

from stressonnx.download import _download_files
from stressonnx.errors import ModelDownloadError, ModelLoadError
from stressonnx.notation import STRESS_TOKEN, composed, in_form_of, render_marks
from stressonnx.registry import _STRESSRNN_FILES, hf_dir

_VOWELS = "аеиоуэюяыё"
_MAX_INPUT_LEN = 40
_CHARS = "'-_абвгдежзийклмнопрстуфхцчшщъыьэюяё"
_CHAR_INDICES = {c: i for i, c in enumerate(_CHARS)}

_RE_CLAUSE = re.compile(r"[…:,.?!\-\n]")
_RE_NOT_WORD = re.compile(rf"[^а-яё_ {STRESS_TOKEN}]")
_RE_TWO_VOWELS = re.compile(f"[{_VOWELS}].*[{_VOWELS}]")

_TOKEN_CATEGORIES = (
    "0123456789",
    " ",
    ',.;:!?()"[]@#$%^&*_-=«»',
    "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ" + STRESS_TOKEN,
)


def _category(ch: str):
    for cat in _TOKEN_CATEGORIES:
        if ch in cat:
            return cat
    return None


def _tokenize(text: str) -> list:
    """``(start, token)`` runs of one category; a symbol outside every
    category is a token of its own.  Joining the tokens gives *text* back."""
    tokens = []
    start = 0
    category = None
    for i, ch in enumerate(text):
        if i > start and category is not None and ch in category:
            continue
        if i > start:
            tokens.append((start, text[start:i]))
        start = i
        category = _category(ch)
    if text:
        tokens.append((start, text[start:]))
    return tokens


def _vowel_indices(word: str) -> list:
    return [i for i, c in enumerate(word) if c in _VOWELS]


def _load_exceptions(path: str) -> dict:
    """``word → stressed vowel index`` from the upstream ``word+`` / ``word'``
    lines; the first line of a repeated word wins, as upstream."""
    d = {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            for symbol in ("'", "+"):
                if symbol in line:
                    word = line.replace(symbol, "")
                    d.setdefault(word, line.find(symbol) - 1)
                    break
    return d


class StressRnnStressor:
    """Lazy-loading port of StressRNN for ``ru``.

    Use :class:`Stressor` (the public wrapper) instead of this class directly.

    Parameters
    ----------
    cache_dir:
        Override the model storage directory (default: the standard
        Hugging Face cache).
    accuracy_threshold:
        A word whose best BiLSTM probability is below this value stays
        unmarked.  Upstream's default is 0.75; 0.0 marks every word.
    """

    lang = "ru"

    def __init__(self, cache_dir: str | None = None, accuracy_threshold: float = 0.75) -> None:
        self._cache_dir = cache_dir
        self.accuracy_threshold = accuracy_threshold
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
                raise ModelLoadError("stressrnn", exc) from exc
            self._loaded = True

    def _load(self) -> None:
        data = _download_files(hf_dir(self.lang, "stressrnn"), _STRESSRNN_FILES,
                               self._cache_dir, model_id="stressrnn")
        self._exceptions = _load_exceptions(data["exception_dictionary.txt.gz"])
        # upstream runs the session sequentially on one thread; the graph is
        # small enough that more threads only add contention
        opts = ort.SessionOptions()
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._sess = ort.InferenceSession(
            data["model.onnx"], sess_options=opts, providers=["CPUExecutionProvider"],
        )
        self._input_name = self._sess.get_inputs()[0].name

    # ------------------------------------------------------------------
    # Words for the model
    # ------------------------------------------------------------------

    @staticmethod
    def _words_with_context(text: str) -> list:
        """Lowercase words of *text*, each two-vowel word prefixed with the
        last three letters of the previous word and ``_``."""
        words = _RE_NOT_WORD.sub(" ", _RE_CLAUSE.sub(" _ ", text).lower()).split(" ")
        out = []
        for i, word in enumerate(words):
            if not _RE_TWO_VOWELS.search(word):
                out.append(word)
            elif i == 0 or words[i - 1] == "_":
                out.append("_" + word)
            else:
                context = words[i - 1].replace(STRESS_TOKEN, "")
                out.append(context[-3:] + "_" + word)
        return out

    def _predict(self, words_with_context: list) -> list:
        """``(stressed vowel index, probability)`` per word, upstream's decode:
        the argmax position is shifted to the next vowel, or the previous one,
        when it lands on a consonant."""
        if not words_with_context:
            return []
        inputs = np.zeros((len(words_with_context), _MAX_INPUT_LEN, len(_CHARS)), dtype=np.float32)
        for i, word in enumerate(words_with_context):
            for j, ch in enumerate(word):
                inputs[i, _MAX_INPUT_LEN - len(word) + j, _CHAR_INDICES[ch]] = 1
        probs = self._sess.run(None, {self._input_name: inputs})[0]
        out = []
        for word_with_context, row in zip(words_with_context, probs):
            word = word_with_context[word_with_context.index("_") + 1:]
            idx = len(word) - _MAX_INPUT_LEN + int(row.argmax())
            if idx < -len(word) or word[idx] not in _VOWELS:
                vowels = _vowel_indices(word)
                later = [v for v in vowels if idx < v]
                idx = later[0] if later else max(v for v in vowels if idx > v)
            if idx < 0:
                idx += len(word)
            out.append((idx, float(row.max())))
        return out

    def _stressed_words(self, text: str) -> list:
        """``(lowercase word, stressed vowel index)`` for every word that gets
        a mark, in text order."""
        results = []
        batch, batch_slots = [], []
        for word_with_context in self._words_with_context(text):
            if STRESS_TOKEN in word_with_context:
                continue
            word = word_with_context[word_with_context.index("_") + 1:] \
                if "_" in word_with_context else word_with_context
            vowels = _vowel_indices(word_with_context)
            if len(vowels) == 1:
                results.append((word_with_context, vowels[-1]))
            elif not _RE_TWO_VOWELS.search(word_with_context):
                continue
            elif word in self._exceptions or word.replace("ё", "е") in self._exceptions:
                idx = self._exceptions.get(word)
                if idx is None:
                    idx = self._exceptions[word.replace("ё", "е")]
                results.append((word, idx))
            elif len(word_with_context) <= _MAX_INPUT_LEN:
                batch.append(word_with_context)
                batch_slots.append(len(results))
                results.append(None)
        for slot, word_with_context, (idx, prob) in zip(
            batch_slots, batch, self._predict(batch)
        ):
            word = word_with_context[word_with_context.index("_") + 1:]
            if prob >= self.accuracy_threshold:
                results[slot] = (word, idx)
        return [r for r in results if r is not None]

    # ------------------------------------------------------------------
    # Public contract
    # ------------------------------------------------------------------

    def __call__(self, sentence: str) -> str:
        marks, yo = self.mark_offsets(sentence)
        return in_form_of(sentence, render_marks(sentence, marks, yo))

    def mark_offsets(self, text: str):
        """Offsets into *text* of stressed vowels, and an empty list: this
        model never rewrites е as ё.

        The work happens on the NFC form of *text*; each offset points at
        the base character in *text* itself.
        """
        nfc, origin = composed(text)
        marks = self._composed_mark_offsets(nfc)
        return [origin[o] for o in marks], []

    def _composed_mark_offsets(self, text: str) -> list:
        self._ensure_loaded()
        stressed = self._stressed_words(text)
        marks = []
        k = 0
        for start, token in _tokenize(text):
            if k == len(stressed) or not _vowel_indices(token.lower()):
                continue
            word, idx = stressed[k]
            if word == token.lower():
                marks.append(start + idx)
                k += 1
        return marks
