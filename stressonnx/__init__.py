"""stressonnx — pure-onnxruntime multi-language word-stress / accentuation.

Runtime dependencies: onnxruntime, numpy, huggingface_hub (no torch).
Russian (``ru``) additionally requires: tokenizers.

Model families
--------------
``"ruaccent"`` (default for ``ru``):
    Homograph-aware neural pipeline derived from RUAccent (Den4ikAI/ruaccent,
    Apache-2.0).  Resolves context-dependent homographs (замок castle/lock,
    мука flour/torment, белок protein/squirrel …) via four ONNX models.
    Runtime deps: onnxruntime, numpy, tokenizers (no torch, no transformers).

``"silero"`` (default for ``ukr``, ``bel``):
    Neural ONNX pipeline (embedding-bag + MLP heads) exported from
    silero_stress (MIT).

``"simple"`` (default for the other rule/vocabulary languages):
    Vocabulary + rules; no ONNX inference.  Languages: ``aze_cyr``,
    ``aze_lat``, ``uzb_cyr``, ``uzb_lat``, ``bak``, ``bul``, ``chv``,
    ``erz``, ``hye``, ``kat``, ``kaz``, ``kbd``, ``kir``, ``kjh``, ``lav``,
    ``mdf``, ``mkd``, ``sah``, ``slv``, ``tat``, ``tgk``, ``udm``, ``xal``
    (plus the ``bel_simple``, ``ru_simple``, ``ukr_simple`` aliases).

Pass ``bel`` to use the neural accentor; ``bel_simple`` for the vocab path.

Output notation
---------------
All backends emit the **combining acute accent** (U+0301) placed immediately
after the stressed vowel: ``"приве́т"``.  This is the standard Unicode stress
notation compatible with ``russian_text_stresser`` and Chatterbox-Multilingual.

For models trained on the legacy ``+``-before-vowel notation (``"прив+ет"``),
use :func:`to_plus_notation` or pass ``notation="plus"`` to :func:`stress` /
:class:`Stressor`.

Script model
------------
Every language tag is associated with a :class:`Script` (writing system)
via :func:`lang_to_script` and :data:`LANG_SCRIPT`.  Each
:class:`ModelEntry` in :data:`MODEL_REGISTRY` declares ``input_scripts`` —
the set of writing systems the model accepts.  phoonnx and other callers can
use this to verify compatibility before dispatching text to stressonnx::

    from stressonnx import lang_to_script, MODEL_REGISTRY, Script

    if lang_to_script("ru") in MODEL_REGISTRY["ruaccent"].input_scripts:
        # safe to call stress(text, "ru", model="ruaccent")
        ...
"""
import unicodedata as _unicodedata

from stressonnx.errors import (
    StressonnxError,
    UnsupportedLanguageError,
    ModelDownloadError,
)
from stressonnx.registry import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    RUACCENT_LANGS,
    MAIN_LANGS,
    SIMPLE_LANGS,
    ALL_LANGS,
    LANG_SCRIPT,
    ModelEntry,
    Script,
    StressNotation,
    StressorBackend,
    lang_to_script,
)
from stressonnx.notation import STRESS_TOKEN, _apply_notation
from stressonnx.backends import _SileroStressor, _KubatabaStressor, SimpleStressor, RuAccentStressor
from stressonnx.stressor import Stressor, make_stressor

_SINGLETONS: dict = {}

# Combining acute U+0301
_COMBINING_ACUTE = "́"


def _decompose_acute(text: str) -> str:
    """Split precomposed acute-accented characters into base + U+0301.

    Latin stress output (e.g. ``aze_lat``) may reach a consumer NFC-composed
    (``"á"`` instead of ``"a" + U+0301``); only characters whose canonical
    decomposition ends in U+0301 are expanded — everything else (``ё``,
    ``ö``, ``й`` …) is left untouched, so this is NOT a general NFD pass.
    """
    out = []
    for ch in text:
        decomp = _unicodedata.normalize("NFD", ch)
        if len(decomp) > 1 and decomp[-1] == _COMBINING_ACUTE:
            out.append(decomp)
        else:
            out.append(ch)
    return "".join(out)


def to_plus_notation(text: str) -> str:
    """Convert combining-acute stress notation to legacy ``+``-before-vowel.

    ``"приве́т"`` → ``"прив+ет"``

    Useful for models that were trained on ``+``-marked text.  Operates on any
    script — it simply moves every U+0301 (combining acute) from after its base
    character to a ``+`` before it.

    Parameters
    ----------
    text:
        Text containing U+0301 combining-acute stress marks.

    Returns
    -------
    str
        Text with each stressed vowel written as ``+<vowel>``.
    """
    text = _decompose_acute(text)
    result = []
    i = 0
    while i < len(text):
        ch = text[i]
        if i + 1 < len(text) and text[i + 1] == _COMBINING_ACUTE:
            result.append("+")
            result.append(ch)
            i += 2  # skip the combining acute
        else:
            result.append(ch)
            i += 1
    return "".join(result)


import logging as _logging

_LOG = _logging.getLogger("stressonnx")

#: Model priority used when ``stress(..., fallback=True)`` walks down after a
#: download failure: highest-quality first.  ``kubataba`` is deliberately not
#: in the chain — it is an explicit-opt-in alternative for ``ru``.
FALLBACK_PRIORITY = ("ruaccent", "silero", "simple")


def _fallback_chain(lang: str) -> list:
    """``(model, lang)`` pairs able to serve *lang*, best first.

    Follows :data:`FALLBACK_PRIORITY`; a ``<lang>_simple`` alias counts as
    ``simple`` support for *lang* (e.g. ``bel`` falls back to ``bel_simple``).
    """
    chain = []
    for m in FALLBACK_PRIORITY:
        langs = MODEL_REGISTRY[m].langs
        if lang in langs:
            chain.append((m, lang))
        elif f"{lang}_simple" in langs:
            chain.append((m, f"{lang}_simple"))
    return chain


def stress(
    text: str,
    lang: str = "ru",
    model: str | None = None,
    notation: str = "diacritic",
    fallback: bool = False,
) -> str:
    """Insert stress marks into *text*.

    Parameters
    ----------
    text:
        Input text to accentuate.
    lang:
        Language tag (e.g. ``"ru"``, ``"ukr"``, ``"kaz"``).  Defaults to
        ``"ru"`` for backward compatibility.
    model:
        Model-id string — one of ``"ruaccent"``, ``"silero"``, or
        ``"simple"``.  When *None* (the default), the best model for *lang*
        is selected automatically via :data:`DEFAULT_MODEL`:

        * ``ru`` → ``"ruaccent"`` (homograph-aware, context-sensitive)
        * ``ukr`` / ``bel`` → ``"silero"`` (neural ONNX)
        * all other supported languages → ``"simple"`` (vocab + rules)
    notation:
        Output notation.  ``"diacritic"`` (default) returns the combining
        acute accent placed after the stressed vowel (``"приве́т"``).
        ``"plus"`` returns the legacy ``+``-before-vowel form (``"прив+ет"``),
        e.g. for models trained on that format.
    fallback:
        When *True* and the selected model's files cannot be fetched
        (:class:`ModelDownloadError`), walk down the documented priority
        chain (:data:`FALLBACK_PRIORITY`, e.g. ``ru``: ruaccent → silero)
        with a logged warning per hop, raising only when the chain is
        exhausted.  Default *False*: the error propagates immediately.

    Returns
    -------
    str
        Text with stress marks inserted according to *notation*.

    Raises
    ------
    UnsupportedLanguageError
        If *lang* is not supported (also catchable as ``ValueError``).
    ModelDownloadError
        If model files cannot be fetched and *fallback* is *False* (or the
        fallback chain is exhausted).

    Examples
    --------
    ::

        >>> from stressonnx import stress
        # Russian — homograph-aware (замок castle vs lock)
        >>> stress("старинный замок стоит на горе", "ru")
        'стари́нный за́мок сто́ит на горе́'
        >>> stress("дверной замок надёжен", "ru")
        'дверно́й замо́к надёжен'

        # Legacy plus notation for models trained on it
        >>> stress("привет", "ru", notation="plus")
        'прив+ет'

        # Explicit model selection
        >>> stress("Привіт світ", "ukr", model="silero")
        'Приві́т сві́т'
        >>> stress("Сәлем Қазақстан", "kaz", model="simple")
        'Сәле́м Қазақста́н'
    """
    attempts = [(model, lang)]
    if fallback:
        attempts += [p for p in _fallback_chain(lang) if p[0] != model]
        if model is None and len(attempts) > 1:
            attempts = attempts[1:]  # (None, lang) resolves to the chain head

    last_error = None
    for try_model, try_lang in attempts:
        key = (try_lang, try_model)
        if key not in _SINGLETONS:
            _SINGLETONS[key] = make_stressor(model=try_model, lang=try_lang)
        try:
            result = _SINGLETONS[key](text)
        except ModelDownloadError as exc:
            if not fallback:
                raise
            last_error = exc
            _LOG.warning(
                "Model %r unavailable for lang %r (%s) — falling back.",
                try_model, try_lang, exc,
            )
            continue
        return _apply_notation(result, notation)
    raise last_error


__all__ = [
    "stress",
    "to_plus_notation",
    "StressonnxError",
    "UnsupportedLanguageError",
    "ModelDownloadError",
    "FALLBACK_PRIORITY",
    "Stressor",
    "_SileroStressor",
    "_KubatabaStressor",
    "SimpleStressor",
    "RuAccentStressor",
    "make_stressor",
    "MODEL_REGISTRY",
    "DEFAULT_MODEL",
    "RUACCENT_LANGS",
    "MAIN_LANGS",
    "SIMPLE_LANGS",
    "ALL_LANGS",
    "STRESS_TOKEN",
    "ModelEntry",
    "Script",
    "StressNotation",
    "StressorBackend",
    "LANG_SCRIPT",
    "lang_to_script",
]
