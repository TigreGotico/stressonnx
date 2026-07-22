"""Language definitions: loaded from ``stressonnx/languages/*.json``.

Each supported language is one JSON file carrying everything the engine
needs — writing system, OOV rule name, HF subdirectories per model family,
and the linguistic sources behind the rule.  The engine is generic; the
language specifics live in data.

Canonical tags are BCP-47 (``ru``, ``uk``, ``az-Latn``); the historical
tags the library shipped with remain accepted everywhere via
:func:`canonicalize_lang`.
"""
import json
from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=1)
def load_languages() -> dict:
    """All language definitions, keyed by canonical tag."""
    langs = {}
    pkg = resources.files("stressonnx.languages")
    for entry in pkg.iterdir():
        if entry.name.endswith(".json"):
            spec = json.loads(entry.read_text(encoding="utf-8"))
            langs[spec["tag"]] = spec
    return langs


#: Historical tag → (canonical tag, forced model or None).  The ``*_simple``
#: aliases were (language, model) pairs pretending to be languages; they map
#: to the canonical language with the ``simple`` model forced.
LEGACY_ALIASES: dict = {
    "ukr": ("uk", None),
    "bel": ("be", None),
    "kaz": ("kk", None),
    "kir": ("ky", None),
    "tat": ("tt", None),
    "bak": ("ba", None),
    "chv": ("cv", None),
    "tgk": ("tg", None),
    "erz": ("myv", None),
    "hye": ("hy", None),
    "kat": ("ka", None),
    "bul": ("bg", None),
    "mkd": ("mk", None),
    "slv": ("sl", None),
    "lav": ("lv", None),
    "aze_lat": ("az-Latn", None),
    "aze_cyr": ("az-Cyrl", None),
    "uzb_lat": ("uz-Latn", None),
    "uzb_cyr": ("uz-Cyrl", None),
    "bel_simple": ("be", "simple"),
    "ru_simple": ("ru", "simple"),
    "ukr_simple": ("uk", "simple"),
}


def canonicalize_lang(lang: str) -> tuple:
    """Return ``(canonical_tag, forced_model_or_None)`` for any accepted tag.

    Unknown tags pass through unchanged — the registry lookup that follows
    raises the typed :class:`~stressonnx.errors.UnsupportedLanguageError`.
    """
    if lang in load_languages():
        return lang, None
    return LEGACY_ALIASES.get(lang, (lang, None))
