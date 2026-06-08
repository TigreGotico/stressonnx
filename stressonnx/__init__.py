"""stressonnx — pure-onnxruntime multi-language word-stress / accentuation.

Runtime dependencies: onnxruntime, numpy, huggingface_hub (no torch).

Two accentor families
---------------------
Neural ONNX (main_accentor): ``ru``, ``ukr``, ``bel``
Vocabulary + rules (simple_accentor): ``aze_cyr``, ``aze_lat``, ``uzb_cyr``,
    ``uzb_lat``, ``bak``, ``chv``, ``erz``, ``hye``, ``kat``, ``kaz``,
    ``kbd``, ``kir``, ``kjh``, ``mdf``, ``sah``, ``tat``, ``tgk``, ``udm``,
    ``xal``

Pass ``bel`` to use the neural accentor; ``bel_simple`` for the vocab path.
"""
from stressonnx.accentor import Stressor, SimpleStressor, MAIN_LANGS, SIMPLE_LANGS

_SINGLETONS: dict = {}


def stress(text: str, lang: str = "ru") -> str:
    """Insert stress marks (``+`` before the stressed vowel) into *text*.

    Dispatches to the neural ONNX pipeline for ``ru``/``ukr``/``bel`` and to
    the vocabulary + rule-based pipeline for all other supported languages.

    Example::

        >>> from stressonnx import stress
        >>> stress("Привет мир", "ru")
        'Прив+ет м+ир'
        >>> stress("Привіт світ", "ukr")
        'Прив+іт св+іт'
    """
    if lang not in _SINGLETONS:
        if lang in MAIN_LANGS:
            _SINGLETONS[lang] = Stressor(lang)
        elif lang in SIMPLE_LANGS:
            _SINGLETONS[lang] = SimpleStressor(lang)
        else:
            raise ValueError(
                f"Unsupported language {lang!r}.  "
                f"Neural langs: {sorted(MAIN_LANGS)}.  "
                f"Rule/vocab langs: {sorted(SIMPLE_LANGS)}."
            )
    return _SINGLETONS[lang](text)


__all__ = ["stress", "Stressor", "SimpleStressor", "MAIN_LANGS", "SIMPLE_LANGS"]
