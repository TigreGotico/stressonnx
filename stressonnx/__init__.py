"""stressonnx — pure-onnxruntime multi-language word-stress / accentuation.

Runtime dependencies: onnxruntime, numpy, huggingface_hub (no torch).
Russian (``ru``) additionally requires: tokenizers.

Three accentor families
-----------------------
RUAccent homograph-aware (``ru``):
    Neural pipeline derived from RUAccent (Den4ikAI/ruaccent, Apache-2.0).
    Resolves context-dependent homographs (замок castle/lock, мука flour/
    torment, белок protein/squirrel …) via a RoBERTa NLI ONNX model.
    Runtime deps: onnxruntime, numpy, tokenizers (no torch, no transformers).

Neural ONNX (main_accentor): ``ukr``, ``bel``
Vocabulary + rules (simple_accentor): ``aze_cyr``, ``aze_lat``, ``uzb_cyr``,
    ``uzb_lat``, ``bak``, ``chv``, ``erz``, ``hye``, ``kat``, ``kaz``,
    ``kbd``, ``kir``, ``kjh``, ``mdf``, ``sah``, ``tat``, ``tgk``, ``udm``,
    ``xal``

Pass ``bel`` to use the neural accentor; ``bel_simple`` for the vocab path.
"""
from stressonnx.accentor import (
    Stressor,
    SimpleStressor,
    RuAccentStressor,
    RUACCENT_LANGS,
    MAIN_LANGS,
    SIMPLE_LANGS,
)

_SINGLETONS: dict = {}


def stress(text: str, lang: str = "ru") -> str:
    """Insert stress marks (``+`` before the stressed vowel) into *text*.

    Dispatches to:

    * ``ru``: RUAccent homograph-aware pipeline (resolves замок/мука/белок …).
    * ``ukr`` / ``bel``: neural ONNX pipeline (embedding-bag + MLP).
    * all other supported languages: vocabulary + rule-based pipeline.

    Example::

        >>> from stressonnx import stress
        >>> stress("старинный замок стоит на горе", "ru")
        'стар+инный з+амок ст+оит на гор+е'
        >>> stress("дверной замок надёжен", "ru")
        'дверн+ой зам+ок надёжен'
        >>> stress("Привіт світ", "ukr")
        'Прив+іт св+іт'
    """
    if lang not in _SINGLETONS:
        if lang in RUACCENT_LANGS:
            _SINGLETONS[lang] = RuAccentStressor()
        elif lang in MAIN_LANGS:
            _SINGLETONS[lang] = Stressor(lang)
        elif lang in SIMPLE_LANGS:
            _SINGLETONS[lang] = SimpleStressor(lang)
        else:
            raise ValueError(
                f"Unsupported language {lang!r}.  "
                f"Homograph-aware langs: {sorted(RUACCENT_LANGS)}.  "
                f"Neural langs: {sorted(MAIN_LANGS)}.  "
                f"Rule/vocab langs: {sorted(SIMPLE_LANGS)}."
            )
    return _SINGLETONS[lang](text)


__all__ = [
    "stress",
    "Stressor",
    "SimpleStressor",
    "RuAccentStressor",
    "RUACCENT_LANGS",
    "MAIN_LANGS",
    "SIMPLE_LANGS",
]
