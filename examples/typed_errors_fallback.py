"""Typed errors and the fallback chain.

stressonnx raises two typed exceptions instead of generic ones:

- ``UnsupportedLanguageError`` (a ``ValueError`` subclass) — the language
  tag is not registered anywhere in the library.  Existing code that catches
  plain ``ValueError`` keeps working unchanged.
- ``ModelDownloadError`` — a required model file could not be fetched from
  the Hugging Face Hub (offline, network failure, or a missing file for
  that model/language pair).

``stress(..., fallback=True)`` uses ``ModelDownloadError`` internally: if the
selected model's files cannot be fetched, it walks down
``FALLBACK_PRIORITY`` — ``("ruaccent", "silero", "simple")`` — trying the
next-best model for the same language, logging a warning at each hop.  A
``<lang>_simple`` alias counts as ``simple`` support for ``<lang>``, so
``bel`` falls back to ``bel_simple`` rather than failing outright.
"""
from stressonnx import (
    FALLBACK_PRIORITY,
    ModelDownloadError,
    UnsupportedLanguageError,
    stress,
)

# --- UnsupportedLanguageError ------------------------------------------------

try:
    stress("hello", "klingon")
except UnsupportedLanguageError as exc:
    print(f"UnsupportedLanguageError: lang={exc.lang!r}")
    print(f"  supported (first 5): {exc.supported[:5]}...")

# It is also a plain ValueError, so old code keeps working:
try:
    stress("hello", "klingon")
except ValueError as exc:
    print(f"caught as ValueError too: {type(exc).__name__}")

print()

# --- ModelDownloadError + fallback=True -------------------------------------

print(f"FALLBACK_PRIORITY: {FALLBACK_PRIORITY}")

# Simulate an unreachable Hub for a single call, to show the fallback chain
# walking ruaccent -> silero -> simple for Russian without crashing.  Each
# backend module imported ``_download_files`` by name
# (``from stressonnx.download import _download_files``), so every backend
# module's local binding must be patched — patching
# ``stressonnx.download._download_files`` alone would not affect them.
import stressonnx.backends.kubataba as _kubataba_mod
import stressonnx.backends.ruaccent as _ruaccent_mod
import stressonnx.backends.silero as _silero_mod
import stressonnx.backends.simple as _simple_mod


def _always_fail(hf_subdir, filenames, cache_dir=None):
    raise ModelDownloadError(hf_subdir, f"{hf_subdir}/{filenames[0]}", OSError("offline"))


_patched_modules = [_ruaccent_mod, _silero_mod, _simple_mod, _kubataba_mod]
_originals = [m._download_files for m in _patched_modules]
for m in _patched_modules:
    m._download_files = _always_fail
try:
    try:
        stress("привет", "ru", model="ruaccent", fallback=False)
    except ModelDownloadError as exc:
        print(f"fallback=False: raised immediately -> {exc.model_id!r} / {exc.hf_path!r}")

    # fallback=True walks ruaccent -> silero -> simple; "simple" does not
    # cover "ru" (only ukr/bel/ru via silero and 20 other langs), so once
    # silero also fails there is nowhere left to fall to and it raises.
    stress("привет", "ru", model="ruaccent", fallback=True)
except ModelDownloadError as exc:
    print(f"  chain exhausted: last failure was {exc.model_id!r}")
finally:
    for m, orig in zip(_patched_modules, _originals):
        m._download_files = orig

# Now demonstrate a real (successful) fallback: silero is unreachable, so
# stress() drops to the next model in the chain for the same language.
print()
print("Real call with fallback=True (network available):")
print(" ", stress("Прывітанне свет", "bel", fallback=True))
print(" ", stress("Прывітанне свет", "bel_simple", fallback=True))
