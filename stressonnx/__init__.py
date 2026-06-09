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

``"simple"`` (default for 20 other Slavic/Turkic/Caucasian languages):
    Vocabulary + rules; no ONNX inference.  Languages: ``aze_cyr``,
    ``aze_lat``, ``uzb_cyr``, ``uzb_lat``, ``bak``, ``chv``, ``erz``,
    ``hye``, ``kat``, ``kaz``, ``kbd``, ``kir``, ``kjh``, ``mdf``, ``sah``,
    ``tat``, ``tgk``, ``udm``, ``xal`` (plus ``bel_simple`` alias).

Pass ``bel`` to use the neural accentor; ``bel_simple`` for the vocab path.

Output notation
---------------
All backends emit the **combining acute accent** (U+0301) placed immediately
after the stressed vowel: ``"приве́т"``.  This is the standard Unicode stress
notation compatible with ``russian_text_stresser`` and Chatterbox-Multilingual.

For models trained on the legacy ``+``-before-vowel notation (``"прив+ет"``),
use :func:`to_plus_notation` or pass ``notation="plus"`` to :func:`stress` /
:class:`Stressor`.
"""
import re as _re

from stressonnx.accentor import (
    Stressor,
    _SileroStressor,
    SimpleStressor,
    RuAccentStressor,
    make_stressor,
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    RUACCENT_LANGS,
    MAIN_LANGS,
    SIMPLE_LANGS,
    ALL_LANGS,
    STRESS_TOKEN,
)

_SINGLETONS: dict = {}

# Combining acute U+0301
_COMBINING_ACUTE = "́"

# Vowel classes for plus-notation conversion
_VOWEL_RE = _re.compile(r"([аАеЕёЁиИоОуУыЫэЭюЮяЯіІїЇєЄаАеЕёЁ])́")


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


def stress(
    text: str,
    lang: str = "ru",
    model: str | None = None,
    notation: str = "diacritic",
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

    Returns
    -------
    str
        Text with stress marks inserted according to *notation*.

    Examples
    --------
    ::

        >>> from stressonnx import stress
        # Russian — homograph-aware (замок castle vs lock)
        >>> stress("старинный замок стоит на горе", "ru")
        'стари́нный за́мок стои́т на горе́'
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
    key = (lang, model)
    if key not in _SINGLETONS:
        _SINGLETONS[key] = make_stressor(model=model, lang=lang)
    result = _SINGLETONS[key](text)
    if notation == "plus":
        return to_plus_notation(result)
    return result


__all__ = [
    "stress",
    "to_plus_notation",
    "Stressor",
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
]
