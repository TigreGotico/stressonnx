"""Public Stressor wrapper and the make_stressor factory — model-aware entry points."""
from stressonnx.backends import _KubatabaStressor, RuAccentStressor, _SileroStressor, SimpleStressor
from stressonnx.notation import _apply_notation
from stressonnx.registry import DEFAULT_MODEL, MODEL_REGISTRY, StressNotation
from stressonnx.errors import UnsupportedLanguageError


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
    if entry.family == "kubataba":
        return _KubatabaStressor(cache_dir=cache_dir)
    raise ValueError(f"Internal error: unknown family {entry.family!r}.")


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

    notation:
        Output notation.  ``"diacritic"`` (default) places the combining
        acute accent (U+0301) after each stressed vowel (``"приве́т"``).
        ``"plus"`` emits the legacy ``+``-before-vowel form (``"прив+ет"``).

    Examples
    --------
    >>> s = Stressor(lang="ru")                # default: ruaccent
    >>> s("старинный замок стоит на горе")
    'стари́нный за́мок сто́ит на горе́'

    >>> s = Stressor(model="silero", lang="ukr")
    >>> s("Привіт світ")
    'Приві́т сві́т'

    >>> s = Stressor(model="simple", lang="kaz")
    >>> s("Сәлем Қазақстан")
    'Сәле́м Қазақста́н'
    """

    def __init__(
        self,
        model: str | None = None,
        lang: str | None = None,
        cache_dir: str | None = None,
        notation: str = "diacritic",
    ) -> None:
        self._backend = make_stressor(model=model, lang=lang, cache_dir=cache_dir)
        # Expose for inspection
        self.lang = getattr(self._backend, "lang", lang)
        self.model = model or DEFAULT_MODEL.get(lang or "")
        try:
            self.notation = StressNotation(notation)
        except ValueError:
            raise ValueError(
                f"notation must be 'diacritic' or 'plus'; got {notation!r}."
            )

    def __call__(self, text: str) -> str:
        """Accentuate *text*; returns the combining-acute form by default."""
        return _apply_notation(self._backend(text), self.notation)
