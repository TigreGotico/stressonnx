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
"""
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
)

_SINGLETONS: dict = {}


def stress(text: str, lang: str = "ru", model: str | None = None) -> str:
    """Insert stress marks (``+`` before the stressed vowel) into *text*.

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

    Returns
    -------
    str
        Text with ``+`` inserted before each stressed vowel.

    Examples
    --------
    ::

        >>> from stressonnx import stress
        # Russian — homograph-aware (замок castle vs lock)
        >>> stress("старинный замок стоит на горе", "ru")
        'стар+инный з+амок ст+оит на гор+е'
        >>> stress("дверной замок надёжен", "ru")
        'дверн+ой зам+ок надёжен'

        # Explicit model selection
        >>> stress("Привіт світ", "ukr", model="silero")
        'Прив+іт св+іт'
        >>> stress("Сәлем Қазақстан", "kaz", model="simple")
        'Сәл+ем Қазақст+ан'
    """
    key = (lang, model)
    if key not in _SINGLETONS:
        _SINGLETONS[key] = make_stressor(model=model, lang=lang)
    return _SINGLETONS[key](text)


__all__ = [
    "stress",
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
]
