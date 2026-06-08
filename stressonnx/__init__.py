"""stressonnx — pure-onnxruntime multi-language word-stress / accentuation.

Runtime dependencies: onnxruntime, numpy, huggingface_hub (no torch).
"""
from stressonnx.accentor import Stressor

_SINGLETONS: dict = {}


def stress(text: str, lang: str = "ru") -> str:
    """Insert stress marks into *text* for the given *lang*.

    Supported langs: ``ru`` (Ukrainian/Belarusian/SimpleAccentor variants planned).
    Stressed vowels are marked with '+' inserted before them.

    Example::

        >>> from stressonnx import stress
        >>> stress("Привет мир", "ru")
        'Прив+ет м+ир'
    """
    if lang not in _SINGLETONS:
        _SINGLETONS[lang] = Stressor(lang)
    return _SINGLETONS[lang](text)


__all__ = ["stress", "Stressor"]
