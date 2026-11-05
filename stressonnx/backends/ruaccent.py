"""RuAccent family (ru) — homograph-aware neural pipeline.

Attribution: RUAccent by Den4ikAI (https://github.com/Den4ikAI/ruaccent),
MIT (https://github.com/Den4ikAI/ruaccent/blob/main/LICENSE and the
model card of https://huggingface.co/ruaccent/accentuator).
Models: HuggingFace ruaccent/accentuator (turbo3.1 or big_poetry omograph
model).  Mirrored to TigreGotico/stressonnx-models under ru_ruaccent/.
Runtime: onnxruntime + numpy + tokenizers (no torch, no transformers).

Pipeline:
1. Predict per-word stress usage (STRESS / NO_STRESS) via a BERT-family
   token-classifier ONNX model.
2. Resolve yo-homographs (е→ё disambiguation) via a DistilBERT ONNX model.
3. Resolve omographs (context-sensitive stress variants) via an NLI ONNX
   model (RoBERTa turbo3.1 or BERT big_poetry); each omograph candidate is
   scored against the sentence context.
4. Accent non-omograph words: look up in the accent dictionary or run the
   character-level RoFormer ONNX accent model.
Runtime deps: onnxruntime, numpy, tokenizers (no torch, no transformers).
"""
import threading
import gzip
import json
import os
import re
import tempfile
import unicodedata
from importlib.metadata import version

import numpy as np
import onnxruntime as ort

from stressonnx._common import _RU_VOWELS, _UNSTRESSED_HYPHEN_CLITICS, _softmax
from stressonnx.download import _download_files
from stressonnx.registry import MODEL_REGISTRY
from stressonnx.errors import ModelDownloadError, ModelLoadError, UnknownModelError
from stressonnx.notation import (
    STRESS_TOKEN, _plus_to_diacritic, composed, in_form_of, is_stress_base,
    render_marks,
)

# Files to download for the ru_ruaccent bundle, besides the omograph model
_RUACCENT_FILES = [
    "nn_accent/model.onnx",
    "nn_accent/vocab.txt",
    "nn_accent/config.json",
    "nn_stress_usage/model.onnx",
    "nn_stress_usage/tokenizer.json",
    "nn_stress_usage/config.json",
    "nn_yo_homograph/model.onnx",
    "nn_yo_homograph/tokenizer.json",
    "nn_yo_homograph/config.json",
    "dictionary/omographs.json.gz",
    "dictionary/yo_words.json.gz",
    "dictionary/yo_homographs.json.gz",
    "dictionary/accents_nn.json.gz",
    "meta.json",
]

#: Omograph models: RUAccent name → directory under ru_ruaccent/ holding its
#: model.onnx and tokenizer.json.
OMOGRAPH_MODELS = {
    "turbo3.1": "nn_omograph",
    "big_poetry": "nn_omograph_big_poetry",
}

# Characters the models never see (matches RUAccent normalize regex).  They
# stay in the output: marks found on the normalized text map back to the input.
_RE_RUACCENT_NORM = re.compile(
    r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;\"“”‘’"
    r"(){}\[\]«»„“\"\-]"
)

# Whitespace to Python's split that the tokenizer deletes.  The models read a
# space there, so the word list and the model labels stay the same length.
_DELETED_WHITESPACE = str.maketrans(dict.fromkeys("\x0b\x0c\x1c\x1d\x1e\x1f\x85", " "))

# Position limit of the BERT-family models.
_MAX_TOKENS = 512

# Words on each side of a homograph that the omograph model sees.  The
# window keeps its full size (2 * this + 1 words) near the ends of a
# sentence, so a sentence of up to that many words is scored whole.
_OMOGRAPH_CONTEXT_WORDS = 64

# Punctuation characters used by delete_spaces_before_punc
_PUNC_CHARS = '!"#%&\'()*,./:;<=>?@[\\]^_`{|}-'

# Matches word tokens; ́ = combining acute (diacritic stress mark) is
# treated as part of the word since it attaches to the preceding vowel.
_RU_SPLIT_RE = re.compile(r"[\w\u0301]+|[^\w\s\u0301]+")


def _ruaccent_norm(text: str) -> str:
    return _RE_RUACCENT_NORM.sub("", text)


def _whole_words(text: str, kept: list, norm: str) -> set:
    """Offsets in *norm* that lie in a word the models saw as *text* spells it.

    *norm* is *text* without the characters outside the allow-list, and
    ``kept[i]`` is the offset in *text* of ``norm[i]``.  A word loses this
    status when a letter, digit or combining mark was dropped from it or at
    its edge (``Київ`` reaches the models as ``Кив``), or when a dropped
    visible character or a ZERO WIDTH SPACE joined two words into one.
    Other dropped format characters (soft hyphen, ZWJ, ZWNJ, WORD JOINER)
    sit inside the word they were in, so the edge check looks past them.
    """
    dropped = set(range(len(text))).difference(kept)

    def in_word(i: int) -> bool:
        # UAX #29 rule WB4 ignores format characters inside a word; of the
        # Cf characters only U+200B is a word boundary
        return unicodedata.category(text[i]) == "Cf" and text[i] != "\u200b"

    whole = set()
    for m in re.finditer(r"\w+", norm):
        first, last = kept[m.start()], kept[m.end() - 1]
        while first - 1 in dropped and in_word(first - 1):
            first -= 1
        while last + 1 in dropped and in_word(last + 1):
            last += 1
        if any(
            not in_word(i) for i in dropped.intersection(range(first, last))
        ) or any(
            text[i].isalnum() or unicodedata.combining(text[i])
            for i in dropped.intersection((first - 1, last + 1))
        ):
            continue
        whole.update(range(m.start(), m.end()))
    return whole


def _delete_spaces_before_punc(text: str) -> str:
    for char in _PUNC_CHARS:
        if char == "-":
            text = text.replace(" " + char, char).replace(char + " ", char)
        text = text.replace(" " + char, char)
    return text.replace("~", "-")


def _fix_capital(source: str, target: str) -> str:
    if len(source) != len(target):
        return target
    return "".join(
        t.upper() if s.isupper() else t.lower()
        for s, t in zip(source, target)
    )


def _match_case(source: str, stressed: str) -> str:
    """*stressed*, a lowercase marked form of *source*, in the case of *source*."""
    letters = iter(source)
    return "".join(
        ch if ch == STRESS_TOKEN or not next(letters, "").isupper() else ch.upper()
        for ch in stressed
    )


def _ruaccent_split_by_words(string: str):
    """Split *string* into (words, remaining_text_parts) preserving whitespace/punc.

    Returns ``(valid_words, rem)`` where ``rem`` has ``len(valid_words) + 1``
    elements: the text before the first word, between consecutive words, and
    after the last word.  The original string can be reconstructed as::

        "".join(l + r for l, r in zip(rem, valid_words)) + rem[-1]

    Punctuation tokens (matched by the non-word alternative of *_RU_SPLIT_RE*)
    are kept as-is and included in ``valid_words``; spaces fall into ``rem``.
    """
    string = string.replace(" - ", " ~ ")
    all_matches = list(_RU_SPLIT_RE.finditer(string.lower()))
    if not all_matches:
        return [], ["", ""]

    # Build valid_words (non-empty matches) and rem (gaps between them)
    raw_words = [string[m.start():m.end()] for m in all_matches]
    mask = [i for i, w in enumerate(raw_words) if w]
    if not mask:
        return [], ["", ""]

    valid_words = [raw_words[i] for i in mask]
    valid_spans = [all_matches[i] for i in mask]

    # rem[0] = text before first valid word
    # rem[k] = text between valid_spans[k-1] and valid_spans[k]
    # rem[-1] = text after last valid word
    rem = [string[:valid_spans[0].start()]]
    rem += [
        string[valid_spans[k - 1].end():valid_spans[k].start()]
        for k in range(1, len(valid_spans))
    ]
    rem.append(string[valid_spans[-1].end():])
    return valid_words, rem


def _ruaccent_split_by_sentences(text: str) -> list:
    """Split *text* into sentences using razdel if available, else return as-is."""
    try:
        from razdel import sentenize
        from razdel.substring import Substring
        sentences = list(sentenize(text))
        if not sentences:
            return []
        result = [
            text[l.stop:r.start] + r.text if l.stop != r.start else r.text
            for l, r in zip([Substring(0, 0, "")] + sentences, sentences)
        ]
        result[-1] = result[-1] + text[sentences[-1].stop:]
        return result
    except ImportError:
        return [text]


class _Tokenizer:
    """A ``tokenizer.json`` with truncation off, on tokenizers 0.x and 1.x.

    tokenizers 1.x has no ``Encoding.word_ids`` and refuses vocab keys that
    hold NUL, so this class reads the file itself.  :meth:`word_ids` needs a
    WordPiece model with a template post-processor, as the BERT tokenizers of
    this pipeline have.
    """

    def __init__(self, path: str) -> None:
        from tokenizers import Tokenizer

        with open(path, encoding="utf-8") as fh:
            spec = json.load(fh)
        model = spec["model"]
        self._tokenizers_1x = int(version("tokenizers").split(".")[0]) >= 1
        if self._tokenizers_1x:
            # NUL keys never match: the BertNormalizer (clean_text) deletes
            # NUL from the input before the vocab lookup
            model["vocab"] = {k: i for k, i in model["vocab"].items() if "\0" not in k}
            with tempfile.TemporaryDirectory() as tmp:
                sanitized = os.path.join(tmp, "tokenizer.json")
                with open(sanitized, "w", encoding="utf-8") as fh:
                    json.dump(spec, fh)
                self._tok = Tokenizer.from_file(sanitized)
            self._tok.truncation = None
        else:
            self._tok = Tokenizer.from_file(path)
            self._tok.no_truncation()

        if model["type"] == "WordPiece":
            prefix = model["continuing_subword_prefix"]
            self._continuation = frozenset(
                i for k, i in model["vocab"].items() if k.startswith(prefix)
            )
            template = [next(iter(item)) for item in spec["post_processor"]["single"]]
            self._lead = template.index("Sequence")
            self._trail = len(template) - self._lead - 1

    def encode(self, text: str, pair: str | None = None) -> tuple[list, list]:
        """Token ids and attention mask of *text*, or of the pair (*text*, *pair*)."""
        if self._tokenizers_1x:
            enc = self._tok.encode(text if pair is None else (text, pair))
        else:
            enc = self._tok.encode(text, pair=pair)
        return enc.ids, enc.attention_mask

    def type_ids(self, text: str, pair: str) -> list:
        """Segment id of each token of the pair (*text*, *pair*)."""
        if self._tokenizers_1x:
            return self._tok.encode((text, pair)).type_ids
        return self._tok.encode(text, pair=pair).type_ids

    def word_ids(self, ids: list) -> list:
        """Word index of each token of one sequence, ``None`` for added specials.

        A WordPiece token starts a new word unless it carries the
        continuation prefix (``##``).
        """
        word_ids: list = [None] * len(ids)
        word = -1
        for i in range(self._lead, len(ids) - self._trail):
            if ids[i] not in self._continuation:
                word += 1
            word_ids[i] = word
        return word_ids


class RuAccentStressor:
    """Homograph-aware Russian stress accentor backed by RUAccent ONNX models.

    Uses four ONNX models (no torch, no transformers):

    * **stress_usage** (BERT-family token classifier) — predicts STRESS / NO_STRESS
      per word in context.
    * **yo_homograph** (DistilBERT token classifier) — resolves е→ё substitutions
      for yo-homographs.
    * **omograph** (NLI classifier: RoBERTa turbo3.1 or ruBERT big_poetry) —
      picks the correct stressed variant of context-dependent homographs
      (замок castle/lock, мука flour/torment, белок protein/squirrel …).
    * **accent** (RoFormer char-level token classifier) — accentuates words not
      found in the accent dictionary.

    All tokenizers are loaded via the ``tokenizers`` library (HuggingFace
    *fast tokenizer* format) without requiring ``transformers`` or ``torch``.

    Attribution
    -----------
    Derived from RUAccent by Den4ikAI
    (https://github.com/Den4ikAI/ruaccent), MIT.  Models sourced from
    ``ruaccent/accentuator`` on HuggingFace and mirrored to
    ``TigreGotico/stressonnx-models`` under ``ru_ruaccent/``.

    Parameters
    ----------
    cache_dir:
        Override the model storage directory (default: the standard
        Hugging Face cache).
    omograph:
        Omograph model, a key of :data:`OMOGRAPH_MODELS`.  Only this model
        is downloaded and loaded.

    Raises
    ------
    UnknownModelError
        If *omograph* is not a key of :data:`OMOGRAPH_MODELS`.
    """

    def __init__(self, cache_dir: str | None = None, omograph: str = "turbo3.1") -> None:
        if omograph not in OMOGRAPH_MODELS:
            raise UnknownModelError(omograph, OMOGRAPH_MODELS)
        self._omograph_dir = OMOGRAPH_MODELS[omograph]
        self._cache_dir = cache_dir
        self._loaded = False
        self._load_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lazy loading
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
                raise ModelLoadError('ruaccent', exc) from exc
            self._loaded = True

    def _load(self) -> None:

        omograph_files = [f"{self._omograph_dir}/model.onnx", f"{self._omograph_dir}/tokenizer.json"]
        data = _download_files(MODEL_REGISTRY['ruaccent'].hf_subdir,
                               _RUACCENT_FILES + omograph_files,
                               self._cache_dir, model_id="ruaccent")

        try:
            import tokenizers  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "stressonnx 'ru' (RuAccentStressor) requires the 'tokenizers' "
                "package.  Install it with: pip install tokenizers"
            ) from exc

        # ONNX sessions
        self._omograph_sess = ort.InferenceSession(
            data[omograph_files[0]],
            providers=["CPUExecutionProvider"],
        )
        # the BERT omograph model reads the segment of each token, the
        # RoBERTa one has no segments
        self._omograph_has_tti = any(
            i.name == "token_type_ids" for i in self._omograph_sess.get_inputs()
        )
        self._accent_sess = ort.InferenceSession(
            data["nn_accent/model.onnx"],
            providers=["CPUExecutionProvider"],
        )
        self._stress_usage_sess = ort.InferenceSession(
            data["nn_stress_usage/model.onnx"],
            providers=["CPUExecutionProvider"],
        )
        self._yo_hom_sess = ort.InferenceSession(
            data["nn_yo_homograph/model.onnx"],
            providers=["CPUExecutionProvider"],
        )

        # Tokenizers (no transformers)
        self._omograph_tok = _Tokenizer(data[omograph_files[1]])
        self._stress_usage_tok = _Tokenizer(data["nn_stress_usage/tokenizer.json"])
        self._yo_hom_tok = _Tokenizer(data["nn_yo_homograph/tokenizer.json"])

        # Char vocab for accent model
        with open(data["nn_accent/vocab.txt"], encoding="utf-8") as fh:
            self._char_vocab = {line.rstrip("\n"): i for i, line in enumerate(fh)}
        with open(data["nn_accent/config.json"], encoding="utf-8") as fh:
            self._accent_id2label = json.load(fh)["id2label"]
        # Label maps
        with open(data["nn_stress_usage/config.json"], encoding="utf-8") as fh:
            self._stress_id2label = json.load(fh)["id2label"]
        with open(data["nn_yo_homograph/config.json"], encoding="utf-8") as fh:
            self._yo_id2label = json.load(fh)["id2label"]

        # Dictionaries
        self._omographs: dict = json.load(gzip.open(data["dictionary/omographs.json.gz"]))
        # Extra entry matching RUAccent's hardcoded update
        self._omographs["коса"] = ["к+оса", "кос+а"]
        self._yo_words: dict = json.load(gzip.open(data["dictionary/yo_words.json.gz"]))
        self._yo_homographs: dict = json.load(gzip.open(data["dictionary/yo_homographs.json.gz"]))
        self._accents: dict = json.load(gzip.open(data["dictionary/accents_nn.json.gz"]))
        # Single-vowel mappings from RUAccent
        self._accents.update({"о": "+о", "О": "+О"})


    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _predict_word_labels(
        self,
        text: str,
        sess: "ort.InferenceSession",
        tok,
        id2label: dict,
        has_tti: bool = True,
    ) -> list:
        """Run a token-classification model and aggregate subword → word labels.

        A text past the model's position limit is split in half at a space
        and each half is labelled on its own, so no word loses its label to
        truncation.
        """
        ids, attention_mask = tok.encode(text)
        if len(ids) > _MAX_TOKENS:
            words = text.split(" ")
            half = len(words) // 2
            if half:
                return [
                    label
                    for part in (words[:half], words[half:])
                    for label in self._predict_word_labels(
                        " ".join(part), sess, tok, id2label, has_tti
                    )
                ]
            # one run without spaces: keep its head and the closing [SEP]
            ids = ids[:_MAX_TOKENS - 1] + ids[-1:]
            attention_mask = attention_mask[:_MAX_TOKENS - 1] + attention_mask[-1:]
        input_ids = np.array([ids], dtype=np.int64)
        attn = np.array([attention_mask], dtype=np.int64)
        feed: dict = {"input_ids": input_ids, "attention_mask": attn}
        if has_tti:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        logits = sess.run(None, feed)[0][0]
        probs = _softmax(logits)

        word_probs: dict = {}
        for i, wid in enumerate(tok.word_ids(ids)):
            if wid is None:
                continue
            word_probs.setdefault(wid, []).append(probs[i])

        return [
            id2label[str(int(np.stack(word_probs[wid]).mean(0).argmax()))]
            for wid in sorted(word_probs)
        ]

    def _put_accent(self, word: str) -> str:
        """Accentuate a single word with the char-level RoFormer model."""
        lower = word.lower()
        # BOS=2, EOS=3, UNK=1
        ids = [2] + [self._char_vocab.get(c, 1) for c in lower] + [3]
        input_ids = np.array([ids], dtype=np.int64)
        logits = self._accent_sess.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": np.ones_like(input_ids),
                "token_type_ids": np.zeros_like(input_ids),
            },
        )[0][0]
        probs = _softmax(logits)
        result = list(word)
        for i, (label_id, score) in enumerate(
            zip(logits.argmax(axis=-1), probs.max(axis=-1))
        ):
            # position 0 is BOS and position len(word)+1 is EOS — a stress
            # label there must not wrap around onto a real character
            if not 0 < i <= len(result):
                continue
            label = self._accent_id2label[str(int(label_id))]
            if (
                label not in ("NO", "STRESS_SECONDARY")
                and score >= 0.55
                and lower[i - 1] in _RU_VOWELS
            ):
                result[i - 1] = result[i - 1] + STRESS_TOKEN
        return "".join(result)

    @staticmethod
    def _has_punct(text: str) -> bool:
        return any(c in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~' for c in text)

    @staticmethod
    def _count_vowels(text: str) -> int:
        return sum(1 for c in text if c in "аеёиоуыэюяАЕЁИОУЫЭЮЯ")

    def _process_yo(self, words: list, sentence: str) -> list:
        lower = sentence.lower()
        yo_preds = None
        if "е" in lower:
            yo_preds = self._predict_word_labels(
                lower, self._yo_hom_sess, self._yo_hom_tok,
                self._yo_id2label, has_tti=False,
            )
        for i, word in enumerate(words):
            lw = word.lower()
            words[i] = _fix_capital(word, self._yo_words.get(lw, word))
            if yo_preds and i < len(yo_preds) and yo_preds[i] == "YO":
                words[i] = _fix_capital(word, self._yo_homographs.get(lw, word))
        return words

    def _process_omographs(self, splitted_text: list) -> list:
        """Resolve homographs using the NLI model.

        Each omograph is scored against the *original* (pre-modification) context
        to avoid leaking prior stress decisions into subsequent classifications.
        """
        # Snapshot original words so every omograph sees the same context.
        # Lookup and context are lowercase: the dictionary keys are, and the
        # model reads a capitalized context word differently.
        original = list(splitted_text)
        context = [w.lower() for w in original]
        found = [
            (i, self._omographs[w])
            for i, w in enumerate(context)
            if w in self._omographs
        ]
        for pos, variants in found:
            probs = []
            for hyp in variants:
                logits = self._omograph_sess.run(
                    None, self._omograph_feed(context, pos, hyp)
                )[0][0]
                e = np.exp(logits - logits.max())
                probs.append(float(e[1] / e.sum()))
            splitted_text[pos] = _match_case(
                original[pos], _plus_to_diacritic(variants[int(np.argmax(probs))])
            )
        return splitted_text

    def _omograph_feed(self, words: list, pos: int, hyp: str) -> dict:
        """Model inputs for the homograph at *pos* in its context, with hypothesis *hyp*.

        The context is a window of words around the homograph, halved until
        the pair fits the model's position limit.
        """
        span = _OMOGRAPH_CONTEXT_WORDS
        while True:
            size = 2 * span + 1
            lo = max(0, min(pos - span, len(words) - size))
            tmp = words[lo:lo + size]
            tmp[pos - lo] = " <w>" + tmp[pos - lo] + "</w> "
            txt = _delete_spaces_before_punc(" ".join(tmp))
            ids, attention_mask = self._omograph_tok.encode(txt, pair=hyp)
            if len(ids) <= _MAX_TOKENS or span == 0:
                break
            span //= 2
        feed = {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.array([attention_mask], dtype=np.int64),
        }
        if self._omograph_has_tti:
            feed["token_type_ids"] = np.array(
                [self._omograph_tok.type_ids(txt, hyp)], dtype=np.int64
            )
        return feed

    def _process_accent(self, words: list, stress_usages: list) -> list:
        for i, word in enumerate(words):
            if STRESS_TOKEN in word:
                continue
            # кто́-то, что́-либо, пришёл-таки: the post-hyphen enclitic
            # particle never carries word stress
            if (
                word.lower() in _UNSTRESSED_HYPHEN_CLITICS
                and i > 0
                and words[i - 1] == "-"
            ):
                continue
            if i < len(stress_usages) and stress_usages[i] == "STRESS":
                lower = word.lower()
                stressed = self._accents.get(lower, lower)
                if (
                    stressed == lower
                    and not self._has_punct(lower)
                    and self._count_vowels(lower) > 1
                ):
                    words[i] = self._put_accent(word)
                else:
                    # 'stressed' uses '+'-before-vowel notation from the dict.
                    # Convert: for each '+' at pos p in 'stressed', the vowel
                    # in the original word is at p - (number of '+' seen so far).
                    # Place the combining acute AFTER that vowel.
                    matches = list(re.finditer(r"\+", stressed))
                    result_chars = list(word)
                    # Apply insertions in reverse order to keep indices stable.
                    for j, m in reversed(list(enumerate(matches))):
                        vowel_idx = m.start() - j  # position in original word
                        if (
                            0 <= vowel_idx < len(word)
                            and word[vowel_idx].lower() in _RU_VOWELS
                        ):
                            result_chars.insert(vowel_idx + 1, STRESS_TOKEN)
                    words[i] = "".join(result_chars)
        return words

    def _process_sentence(self, sentence: str) -> tuple[list, list]:
        """Offsets into *sentence* of stressed vowels and е→ё substitutions."""
        words, remaining = _ruaccent_split_by_words(sentence)
        if not words:
            return [], []
        stress_usages = self._predict_word_labels(
            sentence, self._stress_usage_sess, self._stress_usage_tok,
            self._stress_id2label, has_tti=True,
        )
        # The usage model leaves a word written with ё unmarked but marks
        # the same word written with е, whose ё the yo step then restores.
        # ё carries the stress, so a word written with ё is always marked.
        stress_usages = [
            "STRESS" if i < len(words) and "ё" in words[i].lower() else label
            for i, label in enumerate(stress_usages)
        ]
        original = list(words)
        words = self._process_yo(words, sentence)
        words = self._process_omographs(words)
        words = self._process_accent(words, stress_usages)

        marks, yo = [], []
        pos = 0
        for gap, word, result in zip(remaining, original, words):
            pos += len(gap)
            # every step keeps the letters of a word and may only turn е
            # into ё; a word that comes back otherwise contributes nothing
            if len(result.replace(STRESS_TOKEN, "")) == len(word):
                i = pos
                for ch in result:
                    if ch == STRESS_TOKEN:
                        marks.append(i - 1)
                        continue
                    if ch in "ёЁ" and sentence[i] in "еЕ":
                        yo.append(i)
                    i += 1
            pos += len(word)
        return marks, yo

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> str:
        """Stress *text*; every character except input stress marks comes back.

        The models read *text* the way upstream RUAccent feeds them: composed
        (so NFD й and ё survive), without the characters outside the
        allow-list.  Marks found there map back onto the full text.  Stress
        marks already in the input are dropped and derived again.  A U+0301
        on a consonant is part of that letter and comes back unchanged.
        """
        self._ensure_loaded()
        base, letter = [], ""
        for ch in composed(text)[0]:
            if ch == STRESS_TOKEN and is_stress_base(letter):
                continue
            if not unicodedata.combining(ch):
                letter = ch
            base.append(ch)
        base = "".join(base)
        kept = [i for i, c in enumerate(base) if _ruaccent_norm(c)]
        norm = "".join(base[i] for i in kept).translate(_DELETED_WHITESPACE)
        marks, yo = [], []
        start = 0
        for sentence in _ruaccent_split_by_sentences(norm):
            sentence_marks, sentence_yo = self._process_sentence(sentence)
            marks += [start + o for o in sentence_marks]
            yo += [start + o for o in sentence_yo]
            start += len(sentence)
        whole = _whole_words(base, kept, norm)
        return in_form_of(text, render_marks(
            base,
            [kept[o] for o in marks if o in whole],
            [kept[o] for o in yo if o in whole],
        ))
