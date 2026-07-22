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

``"silero"`` (default for ``uk``, ``be``):
    Neural ONNX pipeline (embedding-bag + MLP heads) exported from
    silero_stress (MIT).

``"simple"`` (default for the other rule/vocabulary languages):
    Vocabulary + rules; no ONNX inference.  Languages: ``az-Cyrl``,
    ``az-Latn``, ``uz-Cyrl``, ``uz-Latn``, ``ba``, ``bg``, ``cv``,
    ``myv``, ``hy``, ``ka``, ``kk``, ``kbd``, ``ky``, ``kjh``, ``lv``,
    ``mdf``, ``mk``, ``sah``, ``sl``, ``tat``, ``tgk``, ``udm``, ``xal``
    and the dictionary path for ``ru``/``uk``/``be``.

For ``ru``/``uk``/``be`` the neural models are the defaults; pass
``model="simple"`` for the pure dictionary path.

Output notation
---------------
All backends emit the **combining acute accent** (U+0301) placed immediately
after the stressed vowel: ``"приве́т"``.  This is the standard Unicode stress
notation compatible with ``russian_text_stresser`` and Chatterbox-Multilingual.

For models trained on the ``+``-before-vowel notation (``"прив+ет"``),
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
import logging as _logging
import time as _time

from stressonnx.errors import (
    StressonnxError,
    UnsupportedLanguageError,
    ModelDownloadError,
    ModelLoadError,
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
from stressonnx.notation import STRESS_TOKEN, _apply_notation, to_plus_notation
from stressonnx.backends import _SileroStressor, SimpleStressor, RuAccentStressor
from stressonnx.stressor import make_stressor
from stressonnx.pipeline import (
    DEFAULT_PIPELINE,
    FALLBACK_PRIORITY,
    StressPipeline,
    StressResult,
    StressedWord,
)

#: Shared state of the default pipeline, exposed for tests/introspection.
_SINGLETONS = DEFAULT_PIPELINE._singletons
_RECENT_FAILURES = DEFAULT_PIPELINE._failures


def _fallback_chain(lang: str) -> list:
    """``(model, lang)`` pairs able to serve *lang*, best first."""
    return DEFAULT_PIPELINE._chain(lang)


def stress(
    text: str,
    lang: str = "ru",
    model: str | None = None,
    notation: str = "diacritic",
    fallback: bool = False,
    prefer: str | None = None,
) -> str:
    """Insert stress marks into *text*.

    Thin wrapper over the shared :data:`DEFAULT_PIPELINE` —
    ``StressPipeline().stress(...)`` gives you the same behavior with a
    private cache.  Parameters:

    - ``lang``: BCP-47 tag (``"ru"``, ``"uk"``, ``"az-Latn"`` …).
    - ``model``: explicit model id (``"ruaccent"``, ``"silero"``,
      ``"simple"``) — default: the best model for *lang*.
    - ``prefer``: capability strategy instead of a model id — ``"best"``
      (quality), ``"fast"`` (latency), ``"smallest"`` (footprint).
    - ``notation``: ``"diacritic"`` (combining acute after the vowel) or
      ``"plus"``; unknown values raise :class:`ValueError`.
    - ``fallback``: walk down the quality chain on
      :class:`ModelDownloadError` / :class:`ModelLoadError`.

    For structured, offset-based results use :func:`analyze`.

        >>> stress("старинный замок стоит на горе", "ru")
        'стари́нный за́мок сто́ит на горе́'
    """
    return DEFAULT_PIPELINE.stress(text, lang, model=model, notation=notation,
                                   fallback=fallback, prefer=prefer)


def stress_batch(texts, lang: str = "ru", **kwargs) -> list:
    """Stress a sequence of texts (see :meth:`StressPipeline.stress_batch`)."""
    return DEFAULT_PIPELINE.stress_batch(texts, lang, **kwargs)


def analyze(text: str, lang: str = "ru", **kwargs) -> StressResult:
    """Structured stressing: per-word spans with offsets into *text* itself.

        >>> r = analyze("замок стоит", "ru")
        >>> [(w.text, w.stressed_index) for w in r.words]
        [('замок', 1), ('стоит', 2)]
        >>> r.text                      # the marked rendering
        'за́мок сто́ит'
    """
    return DEFAULT_PIPELINE.analyze(text, lang, **kwargs)


def warm_up(lang: str, model: str | None = None) -> None:
    """Download and load the model for *lang* ahead of the first request.

    Call once at service startup so no synthesis request ever blocks on a
    model download (the ``ru`` default is ≈500 MB cold).
    """
    DEFAULT_PIPELINE.warm_up(lang, model)


__all__ = [
    "stress",
    "stress_batch",
    "analyze",
    "warm_up",
    "StressPipeline",
    "StressResult",
    "StressedWord",
    "to_plus_notation",
    "StressonnxError",
    "UnsupportedLanguageError",
    "ModelDownloadError",
    "ModelLoadError",
    "FALLBACK_PRIORITY",
    "_SileroStressor",
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
