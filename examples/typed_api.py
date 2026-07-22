"""Demonstrate the typed API: Script, ModelEntry, StressNotation, StressorBackend.

The Script / lang_to_script / input_scripts triad is designed for phoonnx
integration: phoonnx queries the writing system of a language before
delegating text to stressonnx, avoiding script mismatches.
"""
from stressonnx import (
    MODEL_REGISTRY,
    DEFAULT_MODEL,
    LANG_SCRIPT,
    ModelEntry,
    Script,
    StressNotation,
    StressorBackend,
    lang_to_script,
    make_stressor,
    stress,
)

# --- ModelEntry introspection -----------------------------------------------

print("=== Script enum ===")
print(f"  CYRILLIC = {Script.CYRILLIC!r}  (ru, uk, be, kk, …)")
print(f"  LATIN    = {Script.LATIN!r}   (az-Latn, uz-Latn)")
print(f"  ARMENIAN = {Script.ARMENIAN!r}  (hy)")
print(f"  GEORGIAN = {Script.GEORGIAN!r}  (ka)")
print()

print("=== lang_to_script ===")
for lang in ("ru", "uk", "ka", "hy", "az-Latn", "uz-Latn", "kk"):
    print(f"  lang_to_script({lang!r:10}) → {lang_to_script(lang).value}")
print()

print("=== MODEL_REGISTRY (with input_scripts) ===")
for model_id, entry in MODEL_REGISTRY.items():
    assert isinstance(entry, ModelEntry)
    langs_str = ", ".join(sorted(entry.langs))
    scripts_str = ", ".join(s.value for s in sorted(entry.input_scripts, key=lambda s: s.value))
    print(f"  {model_id:<12}  input_scripts=[{scripts_str}]")
    print(f"              langs={langs_str}")
    print(f"              {entry.description}")
    print()

print("=== phoonnx guard pattern ===")
for lang in ("ru", "uk", "ka"):
    model_id = DEFAULT_MODEL[lang]
    entry = MODEL_REGISTRY[model_id]
    script = lang_to_script(lang)
    can_stress = script in entry.input_scripts
    print(f"  lang={lang!r:10} model={model_id!r:12} script={script.value!r:12} → can_stress={can_stress}")
print()

# --- StressNotation ---------------------------------------------------------

print("=== StressNotation ===")
print(f"  DIACRITIC value: {StressNotation.DIACRITIC!r}")
print(f"  PLUS value:      {StressNotation.PLUS!r}")
print(f"  str inheritance: {isinstance(StressNotation.PLUS, str)}")

# Backwards compatibility: string literals still work
result_str = stress("Казан", "tt", notation="plus")
result_enum = stress("Казан", "tt", notation=StressNotation.PLUS)
assert result_str == result_enum
print(f"  string literal == enum: {result_str!r} == {result_enum!r}  ✓")
print()

# --- StressorBackend Protocol -----------------------------------------------

print("=== StressorBackend Protocol ===")

# All make_stressor outputs satisfy the Protocol
for lang in ("kk", "uk"):
    s = make_stressor(lang=lang)
    assert isinstance(s, StressorBackend), f"make_stressor({lang!r}) not a StressorBackend"
    print(f"  make_stressor({lang!r}) → isinstance StressorBackend ✓")

# Custom callable also satisfies it
class MyCustomStressor:
    def __call__(self, text: str) -> str:
        return text   # passthrough

custom = MyCustomStressor()
assert isinstance(custom, StressorBackend)
print(f"  custom callable → isinstance StressorBackend ✓")

# factory-produced backends satisfy it
s = make_stressor(model="simple", lang="kk")
assert isinstance(s, StressorBackend)
print(f"  Stressor(lang='kk') → isinstance StressorBackend ✓")
