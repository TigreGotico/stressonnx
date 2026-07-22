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
