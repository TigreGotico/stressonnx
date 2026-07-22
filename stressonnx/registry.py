"""Model registry, language routing tables, and script taxonomy.

Every stress backend is registered in ``MODEL_REGISTRY`` keyed by a
string model-id.  The registry entry declares which languages the model
serves and which internal family it belongs to.  ``DEFAULT_MODEL`` maps each
supported language tag to the model-id that is used when ``model=None``.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from stressonnx.errors import UnsupportedLanguageError

# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------

class Script(str, Enum):
    """Writing system / input script of the text.

    Mirrors the script taxonomy used in phoonnx's ``Alphabet`` enum so that
    phoonnx can query which script a stressor accepts before delegating text
    to stressonnx.

    Inherits :class:`str` so values compare equal to their string form
    (``Script.CYRILLIC == "cyrillic"``).
    """

    CYRILLIC = "cyrillic"
    LATIN = "latin"
    ARMENIAN = "armenian"
    GEORGIAN = "georgian"


class StressNotation(str, Enum):
    """Output notation for stress marks.

    ``DIACRITIC`` (default): combining acute U+0301 placed after the stressed
    vowel — ``"приве́т"``.  Standard Unicode; compatible with
    ``russian_text_stresser`` and Chatterbox-Multilingual.

    ``PLUS``: ``+``-before-vowel form — ``"прив+ет"``.  Used by some TTS
    models trained on that format.

    Inherits :class:`str` so plain string literals ``"diacritic"`` /
    ``"plus"`` are accepted wherever :class:`StressNotation` is expected.
    """

    DIACRITIC = "diacritic"
    PLUS = "plus"


@dataclass(frozen=True)
class ModelEntry:
    """Registry entry describing one stress model.

    Attributes
    ----------
    langs:
        Frozenset of language tags supported by this model.
    family:
        Internal family string used by :func:`make_stressor` to select the
        backend class.
    hf_subdir:
        Sub-directory inside ``TigreGotico/stressonnx-models`` where the
        model's runtime artefacts are stored.  ``None`` for families that
        derive the sub-directory from the language tag at runtime.
    description:
        One-line human-readable description.
    input_scripts:
        Frozenset of :class:`Script` values describing which writing systems
        this model accepts.  Used by phoonnx (and other callers) to verify
        that the input text is in a script the model can handle before
        dispatching.
    """

    langs: frozenset
    family: str
    hf_subdir: str | None
    description: str
    input_scripts: frozenset  # frozenset[Script]


@runtime_checkable
class StressorBackend(Protocol):
    """Protocol satisfied by all stressor backend classes.

    Every backend must be callable: ``backend(text: str) -> str``, returning
    text with combining-acute stress marks (U+0301) inserted.
    """

    def __call__(self, text: str) -> str: ...


# ---------------------------------------------------------------------------
# HF repo that hosts all per-language runtime artefacts
# ---------------------------------------------------------------------------
HF_REPO_ID = "TigreGotico/stressonnx-models"

#: Pinned commit of the model repository.  Every file of every bundle is
#: fetched from exactly this revision, so releases are reproducible and an
#: upstream force-push cannot change what users run.  Bump deliberately when
#: models are uploaded (see export/ADDING_A_LANGUAGE.md).
HF_REPO_REVISION = "b8ba7afdb78534afa1a7f4794b98d019c3866aef"

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

# Files for simple_accentor family (language metadata ships in
# stressonnx/languages/, so only the vocabulary is fetched)
_SIMPLE_FILES = [
    "vocab.gz",
]

# ---------------------------------------------------------------------------
# Language routing tables — built from stressonnx/languages/*.json
# ---------------------------------------------------------------------------
from stressonnx.langs import load_languages

LANGUAGES = load_languages()

#: Languages backed by the RUAccent homograph-aware pipeline.
RUACCENT_LANGS = {t for t, spec in LANGUAGES.items() if "ruaccent" in spec["hf"]}

#: Languages backed by the neural ONNX pipeline (main_accentor).
MAIN_LANGS = {t for t, spec in LANGUAGES.items() if "silero" in spec["hf"]}

#: Languages backed by vocabulary + rules (simple_accentor) — every language.
SIMPLE_LANGS = {t for t, spec in LANGUAGES.items() if "simple" in spec["hf"]}

ALL_LANGS = RUACCENT_LANGS | MAIN_LANGS | SIMPLE_LANGS


def hf_dir(lang: str, family: str) -> str:
    """HF subdirectory holding *family*'s artefacts for canonical *lang*."""
    return LANGUAGES[lang]["hf"][family]

# ---------------------------------------------------------------------------
# Script routing
# ---------------------------------------------------------------------------

#: Canonical mapping: language tag → :class:`Script`.
#: Used by :func:`lang_to_script` and to populate :attr:`ModelEntry.input_scripts`.
LANG_SCRIPT: dict[str, Script] = {
    tag: Script(spec["script"]) for tag, spec in LANGUAGES.items()
}


def lang_to_script(lang: str) -> Script:
    """Return the :class:`Script` for a supported language tag.

    Parameters
    ----------
    lang:
        A language tag from :data:`ALL_LANGS` (e.g. ``"ru"``, ``"aze_lat"``).

    Returns
    -------
    Script
        The writing system used by *lang*.

    Raises
    ------
    ValueError
        If *lang* is not in :data:`LANG_SCRIPT`.

    Examples
    --------
    ::

        >>> lang_to_script("ru")
        <Script.CYRILLIC: 'cyrillic'>
        >>> lang_to_script("kat")
        <Script.GEORGIAN: 'georgian'>
        >>> lang_to_script("aze_lat")
        <Script.LATIN: 'latin'>
    """
    try:
        return LANG_SCRIPT[lang]
    except KeyError:
        raise UnsupportedLanguageError(lang, LANG_SCRIPT) from None


# Convenience sets — scripts present in each model family
_CYRILLIC_ONLY = frozenset({Script.CYRILLIC})
_ALL_SCRIPTS = frozenset({Script.CYRILLIC, Script.LATIN, Script.ARMENIAN, Script.GEORGIAN})

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

#: Registry mapping model-id → :class:`ModelEntry`.
MODEL_REGISTRY: dict[str, ModelEntry] = {
    "ruaccent": ModelEntry(
        langs=frozenset(RUACCENT_LANGS),
        family="ruaccent",
        hf_subdir="ru_ruaccent",
        description=(
            "Homograph-aware Russian accentor (RUAccent by Den4ikAI, "
            "Apache-2.0). Four-model ONNX pipeline: stress-usage classifier, "
            "yo-homograph resolver, omograph resolver, char-level accent model."
        ),
        input_scripts=_CYRILLIC_ONLY,
    ),
    "silero": ModelEntry(
        langs=frozenset(MAIN_LANGS),
        family="silero",
        hf_subdir=None,  # per-language: <lang>/
        description=(
            "Neural ONNX accentor exported from silero_stress (MIT). "
            "Fasttext-style n-gram embedding-bag + MLP heads. "
            "Supports Ukrainian, Belarusian, and Russian."
        ),
        input_scripts=_CYRILLIC_ONLY,
    ),
    "simple": ModelEntry(
        langs=frozenset(SIMPLE_LANGS),
        family="simple",
        hf_subdir=None,  # per-language: <lang>/
        description=(
            "Vocabulary + rule-based accentor (silero_stress, MIT). "
            "Dictionary lookup with per-language OOV positional fallback. "
            "Supports 26 languages across Cyrillic, Latin, Armenian, and Georgian scripts."
        ),
        input_scripts=_ALL_SCRIPTS,
    ),
}

#: Default model-id for each language tag: the highest-quality model that
#: serves the language (ruaccent > silero > simple).
DEFAULT_MODEL: dict = {}
# Build defaults: RUACCENT_LANGS → ruaccent, MAIN_LANGS → silero,
# SIMPLE_LANGS → simple (do not overwrite a higher-priority entry).
for _lang in RUACCENT_LANGS:
    DEFAULT_MODEL[_lang] = "ruaccent"
for _lang in MAIN_LANGS:
    if _lang not in DEFAULT_MODEL:
        DEFAULT_MODEL[_lang] = "silero"
for _lang in SIMPLE_LANGS:
    if _lang not in DEFAULT_MODEL:
        DEFAULT_MODEL[_lang] = "simple"


# OOV stress-position rule per simple lang.
# "last"  → last vowel
# "first" → first vowel
# "none"  → skip (return unchanged)
# "kat"   → ≤3 vowels → first, else penultimate
# Per-language OOV stress rule, from the language data files — see
# SimpleStressor._accentuate_oov for the rule implementations and
# benchmarks/RESULTS.md for measured accuracy.
_OOV_RULES = {tag: spec["rule"] for tag, spec in LANGUAGES.items()}
