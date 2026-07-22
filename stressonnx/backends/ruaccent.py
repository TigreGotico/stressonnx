"""RuAccent family (ru) — homograph-aware neural pipeline.

Attribution: RUAccent by Den4ikAI (https://github.com/Den4ikAI/ruaccent),
licensed Apache-2.0 per upstream setup.py classifiers.
Models: HuggingFace ruaccent/accentuator (turbo3.1 omograph model).
Mirrored to TigreGotico/stressonnx-models under ru_ruaccent/.
Runtime: onnxruntime + numpy + tokenizers (no torch, no transformers).

Pipeline:
1. Predict per-word stress usage (STRESS / NO_STRESS) via a BERT-family
   token-classifier ONNX model.
2. Resolve yo-homographs (е→ё disambiguation) via a DistilBERT ONNX model.
3. Resolve omographs (context-sensitive stress variants) via a RoBERTa NLI
   ONNX model; each omograph candidate is scored against the sentence
   context.
4. Accent non-omograph words: look up in the accent dictionary or run the
   character-level RoFormer ONNX accent model.
Runtime deps: onnxruntime, numpy, tokenizers (no torch, no transformers).
"""
import threading
import gzip
import json
import re

import numpy as np
import onnxruntime as ort

from stressonnx._common import _RU_VOWELS, _UNSTRESSED_HYPHEN_CLITICS, _softmax
from stressonnx.download import _download_files
from stressonnx.registry import MODEL_REGISTRY
from stressonnx.errors import ModelDownloadError, ModelLoadError
from stressonnx.notation import STRESS_TOKEN, _plus_to_diacritic

# Files to download for the ru_ruaccent variant
_RUACCENT_FILES = [
    "nn_omograph/model.onnx",
    "nn_omograph/tokenizer.json",
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

# Characters to strip from text before processing (matches RUAccent normalize regex)
_RE_RUACCENT_NORM = re.compile(
    r"[^a-zA-Z0-9\sа-яА-ЯёЁ—.,!?:;\"“”‘’"
    r"(){}\[\]«»„“\"\-]"
)

# Punctuation characters used by delete_spaces_before_punc
_PUNC_CHARS = '!"#%&\'()*,./:;<=>?@[\\]^_`{|}-'

# Matches word tokens; ́ = combining acute (diacritic stress mark) is
# treated as part of the word since it attaches to the preceding vowel.
_RU_SPLIT_RE = re.compile(r"[\w\u0301]+|[^\w\s\u0301]+")


def _ruaccent_norm(text: str) -> str:
    return _RE_RUACCENT_NORM.sub("", text)


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


class RuAccentStressor:
    """Homograph-aware Russian stress accentor backed by RUAccent ONNX models.

    Uses four ONNX models (no torch, no transformers):

    * **stress_usage** (BERT-family token classifier) — predicts STRESS / NO_STRESS
      per word in context.
    * **yo_homograph** (DistilBERT token classifier) — resolves е→ё substitutions
      for yo-homographs.
    * **omograph** (RoBERTa NLI classifier, turbo3.1 variant) — picks the correct
      stressed variant of context-dependent homographs (замок castle/lock,
      мука flour/torment, белок protein/squirrel …).
    * **accent** (RoFormer char-level token classifier) — accentuates words not
      found in the accent dictionary.

    All tokenizers are loaded via the ``tokenizers`` library (HuggingFace
    *fast tokenizer* format) without requiring ``transformers`` or ``torch``.

    Attribution
    -----------
    Derived from RUAccent by Den4ikAI
    (https://github.com/Den4ikAI/ruaccent), licensed Apache-2.0 per upstream
    setup.py and PyPI classifiers.  Models sourced from
    ``ruaccent/accentuator`` on HuggingFace and mirrored to
    ``TigreGotico/stressonnx-models`` under ``ru_ruaccent/``.

    Parameters
    ----------
    cache_dir:
        Override the model storage directory (default: the standard
        Hugging Face cache).
    """

    def __init__(self, cache_dir: str | None = None) -> None:
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

        data = _download_files(MODEL_REGISTRY['ruaccent'].hf_subdir, _RUACCENT_FILES, self._cache_dir, model_id="ruaccent")

        try:
            from tokenizers import Tokenizer as _Tokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "stressonnx 'ru' (RuAccentStressor) requires the 'tokenizers' "
                "package.  Install it with: pip install tokenizers"
            ) from exc

        # ONNX sessions
        self._omograph_sess = ort.InferenceSession(
            data["nn_omograph/model.onnx"],
            providers=["CPUExecutionProvider"],
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
        self._omograph_tok = _Tokenizer.from_file(data["nn_omograph/tokenizer.json"])
        self._stress_usage_tok = _Tokenizer.from_file(data["nn_stress_usage/tokenizer.json"])
        self._yo_hom_tok = _Tokenizer.from_file(data["nn_yo_homograph/tokenizer.json"])

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
        """Run a token-classification model and aggregate subword → word labels."""
        enc = tok.encode(text)
        input_ids = np.array([enc.ids], dtype=np.int64)
        attn = np.array([enc.attention_mask], dtype=np.int64)
        feed: dict = {"input_ids": input_ids, "attention_mask": attn}
        if has_tti:
            feed["token_type_ids"] = np.zeros_like(input_ids)
        logits = sess.run(None, feed)[0][0]
        probs = _softmax(logits)

        word_probs: dict = {}
        for i, wid in enumerate(enc.word_ids):
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
        # Snapshot original words so every omograph sees the same context
        original = list(splitted_text)
        found = [
            (i, self._omographs[w])
            for i, w in enumerate(splitted_text)
            if w in self._omographs
        ]
        for pos, variants in found:
            probs = []
            for hyp in variants:
                tmp = list(original)
                tmp[pos] = " <w>" + tmp[pos] + "</w> "
                txt = _delete_spaces_before_punc(" ".join(tmp))
                enc = self._omograph_tok.encode(txt, pair=hyp)
                input_ids = np.array([enc.ids], dtype=np.int64)
                attn = np.array([enc.attention_mask], dtype=np.int64)
                logits = self._omograph_sess.run(
                    None, {"input_ids": input_ids, "attention_mask": attn}
                )[0][0]
                e = np.exp(logits - logits.max())
                probs.append(float(e[1] / e.sum()))
            splitted_text[pos] = _plus_to_diacritic(variants[int(np.argmax(probs))])
        return splitted_text

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

    def _process_sentence(self, sentence: str) -> str:
        words, remaining = _ruaccent_split_by_words(sentence)
        if not words:
            return "".join(remaining)
        stress_usages = self._predict_word_labels(
            sentence, self._stress_usage_sess, self._stress_usage_tok,
            self._stress_id2label, has_tti=True,
        )
        words = self._process_yo(words, sentence)
        words = self._process_omographs(words)
        words = self._process_accent(words, stress_usages)
        result = "".join(l + r for l, r in zip(remaining, words)) + remaining[-1]
        return _delete_spaces_before_punc(result)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(self, text: str) -> str:
        self._ensure_loaded()
        text = _ruaccent_norm(text)
        sentences = _ruaccent_split_by_sentences(text)
        if not sentences:
            return text
        outputs = [self._process_sentence(s) for s in sentences]
        return "".join(outputs)
