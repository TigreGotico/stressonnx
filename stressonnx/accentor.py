"""Per-language stress accentor backed by ONNX + numpy (main langs) or
vocabulary + rules (SimpleAccentor langs).

Models and data are downloaded from HuggingFace on first use and cached under
``~/.local/share/stressonnx/<lang>/``.

Model registry
--------------
Every stress backend is registered in ``MODEL_REGISTRY`` keyed by a
string model-id.  The registry entry declares which languages the model
serves and which internal family it belongs to.  ``DEFAULT_MODEL`` maps each
supported language tag to the model-id that is used when ``model=None``.

Three families
--------------
``ruaccent`` (model id ``"ruaccent"``, language ``ru``):
    Homograph-aware neural pipeline derived from RUAccent (Den4ikAI/ruaccent),
    licensed Apache-2.0 per upstream classifiers and setup.py.
    Models sourced from HuggingFace: ruaccent/accentuator.
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

``silero`` (model id ``"silero"``, languages ``ukr``, ``bel``):
    Neural ONNX pipeline exported from silero_stress.  Embedding-bag + MLP
    heads. Pipeline:
    1. Tokenise sentence → (raw tokens, clean tokens, prediction mask).
    2. Compute fastText-style n-gram embeddings for each clean token by
       mean-pooling the rows selected from the embedding matrix.
    3. Run the ONNX MLP heads → stress_logits [N, K] (+ yo_logits for ``ru``).
    4. Decode: exceptions dict → skip sets → argmax position → insert '+'.

``simple_accentor`` (model id ``"simple"``, languages ``aze_cyr``,
    ``aze_lat``, ``uzb_cyr``, ``uzb_lat``, ``bak``, ``bel``, ``chv``,
    ``erz``, ``hye``, ``kat``, ``kaz``, ``kbd``, ``kir``, ``kjh``, ``mdf``,
    ``sah``, ``tat``, ``tgk``, ``udm``, ``xal``):
    Vocabulary + rule-based pipeline.
    1. Tokenise sentence.
    2. Look up clean token in vocabulary dict (word → stress char index).
    3. OOV fall-back: language-specific positional rule (last/first/none/kat).
    4. Insert '+' at the determined character index.

``bel`` is available in both the ``"silero"`` and ``"simple"`` models; the
default for ``bel`` is the neural model (``"silero"``).  Pass
``model="simple"`` (or the ``bel_simple`` language alias) to use the
vocabulary-only path.
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
#: Languages backed by the RUAccent homograph-aware pipeline.
RUACCENT_LANGS = {"ru"}

#: Languages backed by the neural ONNX pipeline (main_accentor).
MAIN_LANGS = {"ukr", "bel"}

#: Languages backed by vocabulary + rules (simple_accentor).
SIMPLE_LANGS = {
    "aze_cyr", "aze_lat",
    "uzb_cyr", "uzb_lat",
    "bak",
    "bel_simple",   # alias — same vocab as bel but always rule-path
    "chv", "erz", "hye", "kat", "kaz", "kbd", "kir",
    "kjh", "mdf", "sah", "tat", "tgk", "udm", "xal",
}

ALL_LANGS = RUACCENT_LANGS | MAIN_LANGS | SIMPLE_LANGS

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: Registry mapping model-id → metadata dict.
#:
#: Keys per entry:
#:   ``langs``       – frozenset of language tags supported by this model.
#:   ``family``      – internal family string: ``"ruaccent"``, ``"silero"``,
#:                     or ``"simple"``.
#:   ``hf_subdir``   – sub-directory prefix inside ``TigreGotico/stressonnx-models``
#:                     (``None`` for families that compute the sub-dir from the
#:                     language tag at runtime).
#:   ``description`` – one-line human-readable description.
MODEL_REGISTRY: dict = {
    "ruaccent": {
        "langs": frozenset(RUACCENT_LANGS),
        "family": "ruaccent",
        "hf_subdir": "ru_ruaccent",
        "description": (
            "Homograph-aware Russian accentor (RUAccent by Den4ikAI, "
            "Apache-2.0). Four-model ONNX pipeline: stress-usage classifier, "
            "yo-homograph resolver, omograph resolver, char-level accent model."
        ),
    },
    "silero": {
        "langs": frozenset(MAIN_LANGS),
        "family": "silero",
        "hf_subdir": None,  # per-language: <lang>/
        "description": (
            "Neural ONNX accentor exported from silero_stress (MIT). "
            "Fasttext-style n-gram embedding-bag + MLP heads for "
            "Ukrainian and Belarusian."
        ),
    },
    "simple": {
        "langs": frozenset(SIMPLE_LANGS),
        "family": "simple",
        "hf_subdir": None,  # per-language: <lang>/
        "description": (
            "Vocabulary + rule-based accentor (silero_stress, MIT). "
            "Dictionary lookup with per-language OOV positional fallback. "
            "Supports 20 languages across Cyrillic/Latin scripts."
        ),
    },
}

#: Default model-id for each language tag.
#:
#: Languages that appear in multiple families (``bel``) map to their
#: highest-quality (neural) model.  Access the alternative with
#: ``model="simple"`` or the ``bel_simple`` alias.
DEFAULT_MODEL: dict = {}
# Build defaults: RUACCENT_LANGS → ruaccent, MAIN_LANGS → silero,
# SIMPLE_LANGS → simple (do not overwrite a higher-priority entry).
for _lang in RUACCENT_LANGS:
    DEFAULT_MODEL[_lang] = "ruaccent"
for _lang in MAIN_LANGS:
    DEFAULT_MODEL[_lang] = "silero"
for _lang in SIMPLE_LANGS:
    if _lang not in DEFAULT_MODEL:
        DEFAULT_MODEL[_lang] = "simple"


def make_stressor(
    model: str | None = None,
    lang: str | None = None,
    cache_dir: str | None = None,
):
    """Factory: return the appropriate stressor instance for *model* and *lang*.

    Parameters
    ----------
    model:
        Model-id string — one of ``"ruaccent"``, ``"silero"``, or ``"simple"``.
        When *None*, the default model for *lang* is used (see
        :data:`DEFAULT_MODEL`).
    lang:
        Language tag (e.g. ``"ru"``, ``"ukr"``, ``"kaz"``).  Required when
        *model* is *None* so the default can be looked up.  Optional when
        *model* is given and the model supports only one language, but
        **required** when the model covers multiple languages (``"silero"``
        covers both ``"ukr"`` and ``"bel"``).
    cache_dir:
        Override the HF download cache directory.

    Returns
    -------
    Callable ``(str) -> str``
        A stressor instance ready to be called with a text string.

    Raises
    ------
    ValueError
        If the combination of *model* and *lang* is unsupported.
    """
    if model is None:
        if lang is None:
            raise ValueError("At least one of 'model' or 'lang' must be provided.")
        model = DEFAULT_MODEL.get(lang)
        if model is None:
            all_supported = sorted(DEFAULT_MODEL.keys())
            raise ValueError(
                f"Unsupported language {lang!r}.  Supported: {all_supported}."
            )

    entry = MODEL_REGISTRY.get(model)
    if entry is None:
        raise ValueError(
            f"Unknown model {model!r}.  "
            f"Available models: {sorted(MODEL_REGISTRY.keys())}."
        )

    family = entry["family"]

    if family == "ruaccent":
        if lang is not None and lang not in entry["langs"]:
            raise ValueError(
                f"Model {model!r} does not support language {lang!r}.  "
                f"Supported: {sorted(entry['langs'])}."
            )
        return RuAccentStressor(cache_dir=cache_dir)

    if family == "silero":
        if lang is None:
            raise ValueError(
                f"Model {model!r} supports multiple languages "
                f"({sorted(entry['langs'])}); 'lang' must be specified."
            )
        if lang not in entry["langs"]:
            raise ValueError(
                f"Model {model!r} does not support language {lang!r}.  "
                f"Supported: {sorted(entry['langs'])}."
            )
        return _SileroStressor(lang=lang, cache_dir=cache_dir)

    if family == "simple":
        if lang is None:
            raise ValueError(
                f"Model {model!r} supports multiple languages "
                f"({sorted(entry['langs'])}); 'lang' must be specified."
            )
        if lang not in entry["langs"]:
            raise ValueError(
                f"Model {model!r} does not support language {lang!r}.  "
                f"Supported: {sorted(entry['langs'])}."
            )
        return SimpleStressor(lang=lang, cache_dir=cache_dir)

    raise ValueError(f"Internal error: unknown family {family!r}.")


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
# Main-accentor family (ukr / bel) — silero neural pipeline
# ===========================================================================

class _SileroStressor:
    """Internal: lazy-load neural (silero) accentor for ``ukr`` or ``bel``.

    Use :class:`Stressor` (the public wrapper) instead of this class directly.

    Parameters
    ----------
    lang:
        Language tag: ``"ukr"`` or ``"bel"``.
    cache_dir:
        Override the default cache location
        (``~/.local/share/stressonnx/<lang>``).
    """

    def __init__(self, lang: str, cache_dir: str | None = None) -> None:
        if lang not in MAIN_LANGS:
            raise ValueError(
                f"_SileroStressor only supports {sorted(MAIN_LANGS)}; got {lang!r}."
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


# ===========================================================================
# Public Stressor wrapper — model-aware entry point
# ===========================================================================

class Stressor:
    """Model-aware stress accentor wrapper.

    ``Stressor`` is the recommended high-level class.  It accepts an explicit
    *model* parameter (mirroring ``text2tashkeel``'s ``Diacritizer(model=…)``
    ergonomics) and delegates to the appropriate backend.

    Parameters
    ----------
    model:
        Model-id string — one of ``"ruaccent"``, ``"silero"``, or ``"simple"``.
        When *None* (the default), the best model for *lang* is selected
        automatically via :data:`DEFAULT_MODEL`.
    lang:
        Language tag (e.g. ``"ru"``, ``"ukr"``, ``"kaz"``).  Required when
        *model* is *None* or when the chosen model covers multiple languages.
    cache_dir:
        Override the HF download cache directory passed to the backend.

    Examples
    --------
    >>> s = Stressor(lang="ru")                # default: ruaccent
    >>> s("старинный замок стоит на горе")
    'стар+инный з+амок ст+оит на гор+е'

    >>> s = Stressor(model="silero", lang="ukr")
    >>> s("Привіт світ")
    'Прив+іт св+іт'

    >>> s = Stressor(model="simple", lang="kaz")
    >>> s("Сәлем Қазақстан")
    'Сәл+ем Қазақст+ан'
    """

    def __init__(
        self,
        model: str | None = None,
        lang: str | None = None,
        cache_dir: str | None = None,
    ) -> None:
        self._backend = make_stressor(model=model, lang=lang, cache_dir=cache_dir)
        # Expose for inspection
        self.lang = getattr(self._backend, "lang", lang)
        self.model = model or DEFAULT_MODEL.get(lang or "")

    def __call__(self, text: str) -> str:
        """Accentuate *text*; inserts ``+`` before each stressed vowel."""
        return self._backend(text)


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


# ===========================================================================
# RuAccent family (ru) — homograph-aware neural pipeline
#
# Attribution: RUAccent by Den4ikAI (https://github.com/Den4ikAI/ruaccent),
# licensed Apache-2.0 per upstream setup.py classifiers.
# Models: HuggingFace ruaccent/accentuator (turbo2 omograph model).
# Mirrored to TigreGotico/stressonnx-models under ru_ruaccent/.
# Runtime: onnxruntime + numpy + tokenizers (no torch, no transformers).
# ===========================================================================

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

_RU_SPLIT_RE = re.compile(r"\w*(?:\+\w+)*|[^\w\s]+")


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
    """Split *string* into (words, remaining_text_parts) preserving punctuation."""
    string = string.replace(" - ", " ~ ")
    match = list(_RU_SPLIT_RE.finditer(string.lower()))
    remaining = [string[l.end():r.start()] for l, r in zip(match, match[1:])]
    raw_words = [string[x.start():x.end()] for x in match]
    mask = [i for i, w in enumerate(raw_words) if w]
    valid_words = [raw_words[i] for i in mask]
    if not mask:
        return valid_words, ["", ""]
    rem = ["".join(remaining[:mask[0]])]
    rem += ["".join(remaining[l + 1:r]) for l, r in zip(mask, mask[1:])]
    rem.append("".join(remaining[mask[-1] + 1:]))
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
    * **omograph** (RoBERTa NLI classifier, turbo2 variant) — picks the correct
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
        Override the default cache location
        (``~/.local/share/stressonnx/ru_ruaccent``).
    """

    def __init__(self, cache_dir: str | None = None) -> None:
        if cache_dir is None:
            base = os.path.join(
                os.path.expanduser("~"), ".local", "share", "stressonnx"
            )
            cache_dir = os.path.join(base, "ru_ruaccent")
        self._cache_dir = cache_dir
        self._loaded = False

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        cache = self._cache_dir
        os.makedirs(cache, exist_ok=True)

        # Download all files
        for fname in _RUACCENT_FILES:
            local = os.path.join(cache, fname)
            if not os.path.exists(local):
                os.makedirs(os.path.dirname(local), exist_ok=True)
                hf_hub_download(
                    repo_id=HF_REPO_ID,
                    filename=f"ru_ruaccent/{fname}",
                    local_dir=cache,
                )

        try:
            from tokenizers import Tokenizer as _Tokenizer  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "stressonnx 'ru' (RuAccentStressor) requires the 'tokenizers' "
                "package.  Install it with: pip install tokenizers"
            ) from exc

        # ONNX sessions
        self._omograph_sess = ort.InferenceSession(
            os.path.join(cache, "nn_omograph", "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._accent_sess = ort.InferenceSession(
            os.path.join(cache, "nn_accent", "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._stress_usage_sess = ort.InferenceSession(
            os.path.join(cache, "nn_stress_usage", "model.onnx"),
            providers=["CPUExecutionProvider"],
        )
        self._yo_hom_sess = ort.InferenceSession(
            os.path.join(cache, "nn_yo_homograph", "model.onnx"),
            providers=["CPUExecutionProvider"],
        )

        # Tokenizers (no transformers)
        self._omograph_tok = _Tokenizer.from_file(
            os.path.join(cache, "nn_omograph", "tokenizer.json")
        )
        self._stress_usage_tok = _Tokenizer.from_file(
            os.path.join(cache, "nn_stress_usage", "tokenizer.json")
        )
        self._yo_hom_tok = _Tokenizer.from_file(
            os.path.join(cache, "nn_yo_homograph", "tokenizer.json")
        )

        # Char vocab for accent model
        with open(os.path.join(cache, "nn_accent", "vocab.txt"), encoding="utf-8") as fh:
            self._char_vocab = {line.rstrip("\n"): i for i, line in enumerate(fh)}
        with open(os.path.join(cache, "nn_accent", "config.json"), encoding="utf-8") as fh:
            self._accent_id2label = json.load(fh)["id2label"]

        # Label maps
        with open(os.path.join(cache, "nn_stress_usage", "config.json"), encoding="utf-8") as fh:
            self._stress_id2label = json.load(fh)["id2label"]
        with open(os.path.join(cache, "nn_yo_homograph", "config.json"), encoding="utf-8") as fh:
            self._yo_id2label = json.load(fh)["id2label"]

        # Dictionaries
        import gzip as _gzip
        self._omographs: dict = json.load(
            _gzip.open(os.path.join(cache, "dictionary", "omographs.json.gz"))
        )
        # Extra entry matching RUAccent's hardcoded update
        self._omographs["коса"] = ["к+оса", "кос+а"]
        self._yo_words: dict = json.load(
            _gzip.open(os.path.join(cache, "dictionary", "yo_words.json.gz"))
        )
        self._yo_homographs: dict = json.load(
            _gzip.open(os.path.join(cache, "dictionary", "yo_homographs.json.gz"))
        )
        self._accents: dict = json.load(
            _gzip.open(os.path.join(cache, "dictionary", "accents_nn.json.gz"))
        )
        # Single-vowel mappings from RUAccent
        self._accents.update({"о": "+о", "О": "+О"})

        self._loaded = True

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        return e / e.sum(axis=-1, keepdims=True)

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
        probs = self._softmax(logits)

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
        probs = self._softmax(logits)
        result = list(word)
        for i, (label_id, score) in enumerate(
            zip(logits.argmax(axis=-1), probs.max(axis=-1))
        ):
            label = self._accent_id2label[str(int(label_id))]
            if label not in ("NO", "STRESS_SECONDARY") and score >= 0.55:
                result[i - 1] = STRESS_TOKEN + result[i - 1]
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
            splitted_text[pos] = variants[int(np.argmax(probs))]
        return splitted_text

    def _process_accent(self, words: list, stress_usages: list) -> list:
        for i, word in enumerate(words):
            if STRESS_TOKEN in word:
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
                    matches = list(re.finditer(r"\+", stressed))
                    word_fixed = list(word)
                    for j, m in enumerate(matches):
                        word_fixed = (
                            word_fixed[: m.start() + j]
                            + [STRESS_TOKEN]
                            + list(word)[m.end() - 1:]
                        )
                    words[i] = "".join(word_fixed)
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
