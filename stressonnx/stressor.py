"""make_stressor: the model-aware backend factory."""
from stressonnx.backends import RuAccentStressor, _SileroStressor, SimpleStressor
from stressonnx.registry import ALL_LANGS, DEFAULT_MODEL, MODEL_REGISTRY
from stressonnx.errors import UnsupportedLanguageError
from stressonnx.langs import resolve_lang


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
    if lang is not None:
        lang = resolve_lang(lang, ALL_LANGS)

    if model is None:
        if lang is None:
            raise ValueError("At least one of 'model' or 'lang' must be provided.")
        model = DEFAULT_MODEL.get(lang)
        if model is None:
            raise UnsupportedLanguageError(lang, DEFAULT_MODEL)

    entry = MODEL_REGISTRY.get(model)
    if entry is None:
        raise ValueError(
            f"Unknown model {model!r}.  "
            f"Available models: {sorted(MODEL_REGISTRY.keys())}."
        )

    # One validation for every family: a given lang must be one the model
    # serves, and multi-language families cannot infer the language.
    if lang is not None and lang not in entry.langs:
        raise ValueError(
            f"Model {model!r} does not support language {lang!r}.  "
            f"Supported: {sorted(entry.langs)}."
        )
    if lang is None and len(entry.langs) > 1:
        raise ValueError(
            f"Model {model!r} supports multiple languages "
            f"({sorted(entry.langs)}); 'lang' must be specified."
        )

    if entry.family == "ruaccent":
        return RuAccentStressor(cache_dir=cache_dir)
    if entry.family == "silero":
        return _SileroStressor(lang=lang, cache_dir=cache_dir)
    if entry.family == "simple":
        return SimpleStressor(lang=lang, cache_dir=cache_dir)
    raise ValueError(f"Internal error: unknown family {entry.family!r}.")
