"""Language definitions: loaded from ``stressonnx/languages/*.json``.

Each supported language is one JSON file carrying everything the engine
needs — writing system, OOV rule name, HF subdirectories per model family,
and the linguistic sources behind the rule.  The engine is generic; the
language specifics live in data.  Tags are BCP-47 (``ru``, ``uk``,
``az-Latn``).
"""
import json
from functools import lru_cache
from importlib import resources

from stressonnx.errors import UnsupportedLanguageError


@lru_cache(maxsize=1)
def load_languages() -> dict:
    """All language definitions, keyed by tag."""
    langs = {}
    pkg = resources.files("stressonnx.languages")
    for entry in pkg.iterdir():
        if entry.name.endswith(".json"):
            spec = json.loads(entry.read_text(encoding="utf-8"))
            langs[spec["tag"]] = spec
    return langs


def resolve_lang(lang: str, supported) -> str:
    """Normalize a BCP-47 tag to the canonical key used by *supported*.

    Every stress-relevant lookup table is keyed by a canonical BCP-47 tag
    (``"ru"``, ``"az-Cyrl"``, ``"uz-Latn"`` …).  Callers routinely pass
    regional variants (``"ru-RU"``, ``"uk_UA"``) or different casing
    (``"RU"``), so every public lookup goes through this resolver instead of
    an exact-match ``dict.get``/``in`` check.

    Resolution order, first match wins:

    1. exact tag (case-insensitive; ``_`` treated as ``-``);
    2. language + script subtag, when the tag carries a 4-letter script
       subtag (``"uz-Latn-UZ"`` → ``"uz-Latn"``) — the script is never
       guessed, only read off the tag itself.  When the tag names a script
       this way, that script decides the outcome: it either resolves to the
       matching language+script entry or the tag is unsupported outright
       (``"ru-Latn"`` must not silently fall back to the Cyrillic ``"ru"``
       model);
    3. bare language subtag (``"ru-RU"`` → ``"ru"``) — only reached when the
       tag carried no script subtag at all.

    Parameters
    ----------
    lang:
        The tag as given by the caller.
    supported:
        A mapping keyed by canonical tags, or an iterable of canonical tags.

    Returns
    -------
    str
        The canonical tag as it appears in *supported*.

    Raises
    ------
    UnsupportedLanguageError
        If *lang* is not a string, or no candidate resolves against
        *supported*.
    """
    tags = supported.keys() if hasattr(supported, "keys") else supported
    if not isinstance(lang, str):
        raise UnsupportedLanguageError(lang, tags)

    lookup = {tag.lower(): tag for tag in tags}

    normalized = lang.replace("_", "-")
    parts = normalized.split("-")
    lang_sub = parts[0].lower()

    hit = lookup.get(normalized.lower())
    if hit is not None:
        return hit

    if len(parts) >= 2 and len(parts[1]) == 4 and parts[1].isalpha():
        hit = lookup.get(f"{lang_sub}-{parts[1].title()}".lower())
        if hit is not None:
            return hit
        # The tag explicitly names a script; that script must decide the
        # outcome rather than being silently discarded in favor of the
        # bare language (which may use a different, wrong, script).
        raise UnsupportedLanguageError(lang, tags)

    hit = lookup.get(lang_sub)
    if hit is not None:
        return hit

    raise UnsupportedLanguageError(lang, tags)
