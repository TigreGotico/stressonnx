# stressonnx

Pure-onnxruntime, multi-language word-stress / accentuation library.
No torch at runtime — only `onnxruntime`, `numpy`, `huggingface_hub`, and
`tokenizers` (the last is required only for `ru` via the `ruaccent` model).

All models are hosted on
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
and downloaded + cached automatically on first use.

---

## Supported languages

| Family | Model id | Languages | Notes |
|--------|----------|-----------|-------|
| ruaccent | `"ruaccent"` | `ru` | Default for Russian; homograph-aware |
| silero | `"silero"` | `ukr`, `bel`, `ru` | Neural ONNX; `ru` is an alternative to ruaccent |
| kubataba | `"kubataba"` | `ru` | Seq2seq Transformer; sentence-level |
| simple | `"simple"` | 20 languages | Vocabulary + OOV rules; no ONNX inference |

**Default model per language:** `ru` → `ruaccent`, `ukr`/`bel` → `silero`, all others → `simple`.

---

## Output notation

All backends emit the **combining acute accent** (U+0301) placed immediately
after the stressed vowel:

```
приве́т   =  п р и в е ́ т
                     ^
                  U+0301 (combining acute)
```

The legacy `+`-before-vowel format (`прив+ет`) is available via `notation="plus"` or
`to_plus_notation()` for consumers trained on that format.

---

## Quick start

```python
from stressonnx import stress

# Russian — homograph-aware (замок castle vs lock, мука flour vs torment, …)
stress("старинный замок стоит на горе", "ru")   # → 'стари́нный за́мок сто́ит на горе́'
stress("дверной замок надёжен", "ru")           # → 'дверно́й замо́к надёжен'

# Ukrainian and Belarusian — neural ONNX
stress("Привіт світ", "ukr")         # → 'Приві́т сві́т'
stress("Прывітанне свет", "bel")     # → 'Прывіта́нне све́т'

# Turkic / Caucasian — vocabulary + rules
stress("Salam dünya", "aze_lat")     # → 'Salam дюнья́'
stress("Сәлем Қазақстан", "kaz")    # → 'Сәле́м Қазақста́н'

# Explicit alternative model for Russian
stress("красивый город", "ru", model="silero")    # → 'краси́вый го́род'
stress("красивый город", "ru", model="kubataba")  # → 'краси́вый го́род'

# Legacy notation for downstream TTS models trained on + format
stress("привет", "ru", notation="plus")   # → 'прив+ет'
```

---

## API reference

### `stress(text, lang="ru", model=None, notation="diacritic") → str`

Insert stress marks into *text*.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Input text. |
| `lang` | `str` | Language tag (e.g. `"ru"`, `"ukr"`, `"kaz"`). |
| `model` | `str \| None` | Model id. `None` selects the best model for `lang` automatically. |
| `notation` | `str \| StressNotation` | `"diacritic"` (default) or `"plus"`. |

Singletons are loaded lazily; subsequent calls for the same `(lang, model)` pair
reuse the cached backend.

### `to_plus_notation(text) → str`

Convert combining-acute output to the legacy `+`-before-vowel form.

```python
from stressonnx import to_plus_notation

to_plus_notation("приве́т")   # → 'прив+ет'
to_plus_notation("за́мок")    # → 'з+амок'
```

### `Stressor(model=None, lang=None, notation="diacritic", cache_dir=None)`

Stateful wrapper with `.model`, `.lang`, and `.notation` attributes.

```python
from stressonnx import Stressor

s = Stressor(lang="ru")
s("белок яйца полезен")     # → 'бело́к яйца́ поле́зен'

s = Stressor(model="silero", lang="ukr")
s("Привіт світ")            # → 'Приві́т сві́т'

s = Stressor(lang="kaz")
s("Сәлем Қазақстан")       # → 'Сәле́м Қазақста́н'

s = Stressor(lang="ru", notation="plus")
s("привет")                 # → 'прив+ет'
```

### `make_stressor(model=None, lang=None, cache_dir=None) → StressorBackend`

Low-level factory. Returns a bare backend callable `(text: str) → str`
that always emits combining-acute output. Useful when you need multiple
independent instances or a custom `cache_dir`.

```python
from stressonnx import make_stressor

s = make_stressor(lang="ukr")
s = make_stressor(model="simple", lang="kaz")
```

### Typed API

The following types are exported from `stressonnx` for static typing,
runtime introspection, and phoonnx integration:

```python
from stressonnx import (
    ModelEntry, Script, StressNotation, StressorBackend,
    lang_to_script, LANG_SCRIPT,
)

# Script — str enum, writing system of the input text
Script.CYRILLIC   # == "cyrillic"   (ru, ukr, bel, kaz, …)
Script.LATIN      # == "latin"      (aze_lat, uzb_lat)
Script.ARMENIAN   # == "armenian"   (hye)
Script.GEORGIAN   # == "georgian"   (kat)

# lang_to_script — map language tag to writing system
lang_to_script("ru")      # Script.CYRILLIC
lang_to_script("aze_lat") # Script.LATIN
lang_to_script("kat")     # Script.GEORGIAN

# ModelEntry — frozen dataclass with input_scripts field
entry = MODEL_REGISTRY["ruaccent"]
entry.langs          # frozenset({'ru'})
entry.family         # 'ruaccent'
entry.hf_subdir      # 'ru_ruaccent'
entry.input_scripts  # frozenset({Script.CYRILLIC})
entry.description    # '...'

# Guard pattern (used by phoonnx before delegating to stressonnx):
if lang_to_script(lang) in MODEL_REGISTRY[model_id].input_scripts:
    text = stress(text, lang, model=model_id)

# StressNotation — str enum, backwards-compatible with string literals
StressNotation.DIACRITIC   # == "diacritic"
StressNotation.PLUS        # == "plus"

# StressorBackend — runtime-checkable Protocol
isinstance(make_stressor(lang="ru"), StressorBackend)  # True
```

### Low-level backend classes

| Class | Family | Languages |
|-------|--------|-----------|
| `RuAccentStressor(cache_dir=None)` | ruaccent | `ru` |
| `_SileroStressor(lang, cache_dir=None)` | silero | `ukr`, `bel`, `ru` |
| `_KubatabaStressor(cache_dir=None)` | kubataba | `ru` |
| `SimpleStressor(lang, cache_dir=None)` | simple | 20 languages |

All are callable: `instance(text) → str` (combining-acute output).

---

## Model registry

```python
from stressonnx import MODEL_REGISTRY, DEFAULT_MODEL

list(MODEL_REGISTRY.keys())  # ['ruaccent', 'silero', 'simple', 'kubataba']
DEFAULT_MODEL["ru"]           # 'ruaccent'
DEFAULT_MODEL["ukr"]          # 'silero'
DEFAULT_MODEL["kaz"]          # 'simple'
```

### `"ruaccent"` — homograph-aware Russian (default for `ru`)

Four-model ONNX pipeline derived from
[RUAccent](https://github.com/Den4ikAI/ruaccent) (Apache-2.0).

| Stage | Architecture | Purpose |
|-------|-------------|---------|
| Stress-usage classifier | BERT token classifier | STRESS / NO_STRESS per word in context |
| Yo-homograph resolver | DistilBERT token classifier | Disambiguate е→ё |
| Omograph resolver | RoBERTa NLI (turbo2) | Pick correct stressed variant from homograph dict |
| Accent model | RoFormer char-level | Accentuate words not in the accent dictionary |

### `"silero"` — neural ONNX (`ukr`, `bel`, `ru`)

Fasttext-style n-gram embedding-bag + MLP classifier exported from
[silero_stress](https://github.com/snakers4/silero-models) (MIT).
Supports exceptions dictionaries and skip sets.

### `"kubataba"` — seq2seq Transformer (`ru`)

Encoder-decoder Transformer derived from
[kubataba/Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor).
Sentence-level; performs best on full phrases rather than isolated words.

The PyTorch model is exported as two ONNX graphs (`encoder.onnx` +
`decoder_step.onnx`) with a greedy-decode loop in Python.

### `"simple"` — vocabulary + rules (default for 20 other languages)

Dictionary lookup with per-language OOV positional fallback.

| Tag | Language | OOV rule |
|-----|----------|----------|
| `aze_cyr` | Azerbaijani (Cyrillic) | last vowel |
| `aze_lat` | Azerbaijani (Latin) | last vowel |
| `uzb_cyr` | Uzbek (Cyrillic) | last vowel |
| `uzb_lat` | Uzbek (Latin) | last vowel |
| `bak` | Bashkir | last vowel |
| `bel_simple` | Belarusian (vocab only) | none |
| `chv` | Chuvash | last vowel |
| `erz` | Erzya | first vowel |
| `hye` | Armenian | last vowel |
| `kat` | Georgian | ≤3 vowels → first, else penultimate |
| `kaz` | Kazakh | last vowel |
| `kbd` | Kabardino-Balkarian | last vowel |
| `kir` | Kyrgyz | last vowel |
| `kjh` | Khakas | last vowel |
| `mdf` | Moksha | first vowel |
| `sah` | Yakut | last vowel |
| `tat` | Tatar | last vowel |
| `tgk` | Tajik | last vowel |
| `udm` | Udmurt | last vowel |
| `xal` | Kalmyk | last vowel |

> **Belarusian note:** `bel` routes to `"silero"` by default.
> Use `model="simple"` or the `bel_simple` alias for the vocabulary-only path.

---

## Installation

```bash
pip install stressonnx
```

The `tokenizers` package is included in the default dependencies (required by
the `ruaccent` model for Russian).

### Export / dev tools (torch required)

```bash
pip install "stressonnx[export]"
```

See [`stressonnx/export/ADDING_A_LANGUAGE.md`](stressonnx/export/ADDING_A_LANGUAGE.md)
for instructions on adding a new language, exporting ONNX artefacts, and
uploading to HuggingFace.

---

## HuggingFace model repository

Runtime artefacts: [TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)

```
TigreGotico/stressonnx-models/
├── ru_ruaccent/          ← RUAccent pipeline (4 ONNX models + dicts)
├── ru_kubataba/          ← kubataba encoder + decoder_step ONNX + vocab
├── ukr/                  ← silero neural
├── bel/                  ← silero neural
├── ru/                   ← silero neural (alternative for ru)
├── kaz/                  ← simple vocab
├── tat/                  ← simple vocab
└── …                     ← one directory per simple-accentor language
```

---

## Attribution

| Component | Upstream | License |
|-----------|----------|---------|
| `"ruaccent"` (`ru`) | [Den4ikAI/ruaccent](https://github.com/Den4ikAI/ruaccent) | Apache-2.0 |
| `"silero"` (`ukr`, `bel`, `ru`) | [snakers4/silero-models](https://github.com/snakers4/silero-models) | MIT |
| `"simple"` (20 langs) | [snakers4/silero-models](https://github.com/snakers4/silero-models) | MIT |
| `"kubataba"` (`ru`) | [kubataba/Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor) | MIT |
