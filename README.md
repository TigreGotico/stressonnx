# stressonnx

Pure-onnxruntime, multi-language word-stress / accentuation library.
No torch at runtime — only `onnxruntime`, `numpy`, `huggingface_hub`, and
`tokenizers` (the last is required for `ru` only).

All models are hosted on
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
and downloaded + cached automatically on first use.

---

## Output notation

All backends emit the **combining acute accent** (U+0301) placed immediately
after the stressed vowel:

```
приве́т   =  п р и в е ́ т
               ^^^^^
               е + U+0301
```

This matches the convention used by `russian_text_stresser` and models such as
Chatterbox-Multilingual.  The legacy `+`-before-vowel format (`прив+ет`, used
internally by silero and ruaccent dictionaries) is exposed as a conversion
utility for downstream consumers that were trained on it.

---

## Quick start

```python
from stressonnx import stress

# Russian — homograph-aware (замок castle vs lock, мука flour vs torment, …)
stress("старинный замок стоит на горе", "ru")   # → 'стари́нный за́мок стои́т на горе́'
stress("дверной замок надёжен", "ru")           # → 'дверно́й замо́к надёжен'
stress("мука для хлеба", "ru")                  # → 'муко́ для хле́ба'
stress("мука была невыносима", "ru")            # → 'му́ка была́ невыноси́ма'

# Other languages
stress("Привіт світ", "ukr")      # → 'Приві́т сві́т'
stress("Прывітанне свет", "bel")  # → 'Прывіта́нне све́т'
stress("Salam dünya", "aze_lat")  # → 'Salam дюнья́'
```

---

## API

### `stress(text, lang="ru", model=None, notation="diacritic") -> str`

Insert stress marks into *text*.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Input text. |
| `lang` | `str` | Language tag (e.g. `"ru"`, `"ukr"`, `"kaz"`). Defaults to `"ru"`. |
| `model` | `str \| None` | Model id — `"ruaccent"`, `"silero"`, or `"simple"`. `None` selects the default model for `lang` (see [model registry](#model-registry)). |
| `notation` | `str` | `"diacritic"` (default) — combining acute after stressed vowel. `"plus"` — legacy `+`-before-vowel form for models trained on it. |

```python
from stressonnx import stress

# Default — combining acute
stress("привет", "ru")                           # → 'приве́т'

# Legacy plus notation
stress("привет", "ru", notation="plus")          # → 'прив+ет'

# Explicit model selection
stress("Привіт світ", "ukr", model="silero")    # → 'Приві́т сві́т'
stress("Казан",       "tat", model="simple")    # → 'Каза́н'
stress("замок",       "ru",  model="ruaccent")  # → 'за́мок' or 'замо́к' (context-dependent)
```

Singletons are loaded lazily on first call; subsequent calls for the same
`(lang, model)` pair reuse the cached instance.

### `to_plus_notation(text) -> str`

Convert combining-acute stress notation to the legacy `+`-before-vowel form.

```python
from stressonnx import to_plus_notation

to_plus_notation("приве́т")   # → 'прив+ет'
to_plus_notation("за́мок")    # → 'з+амок'
```

Useful when feeding output into models that were trained on `+`-marked text
(silero TTS, some ruaccent consumers).  `stress(..., notation="plus")` is a
one-step shortcut.

### `Stressor(model=None, lang=None, notation="diacritic", cache_dir=None)`

High-level class.  Accepts the same parameters as `stress()`.

```python
from stressonnx import Stressor

# Russian (default model: ruaccent)
s = Stressor(lang="ru")
s("белок яйца полезен")     # → 'бело́к яйца́ поле́зен'

# Ukrainian — neural
s = Stressor(model="silero", lang="ukr")
s("Привіт світ")            # → 'Приві́т сві́т'

# Kazakh — vocab + rules
s = Stressor(model="simple", lang="kaz")
s("Сәлем Қазақстан")        # → 'Сәле́м Қазақста́н'

# Legacy plus notation
s = Stressor(lang="ru", notation="plus")
s("привет")                 # → 'прив+ет'
```

`Stressor` exposes:
- `s.model` — the model-id string selected at construction.
- `s.lang` — the language tag.
- `s.notation` — `"diacritic"` or `"plus"`.
- `s(text)` — accentuate a string.

### `make_stressor(model=None, lang=None, cache_dir=None)`

Factory function: returns a backend stressor instance (always emits the
combining-acute form).  Useful when you need multiple independent instances or
want a custom `cache_dir`.

```python
from stressonnx import make_stressor

s = make_stressor(lang="ukr")        # → _SileroStressor("ukr")
s = make_stressor(model="simple", lang="kaz")  # → SimpleStressor("kaz")
```

### Low-level classes

| Class | Family | Use |
|-------|--------|-----|
| `RuAccentStressor(cache_dir=None)` | ruaccent | Russian homograph-aware pipeline |
| `_SileroStressor(lang, cache_dir=None)` | silero | Neural ONNX (ukr, bel) |
| `SimpleStressor(lang, cache_dir=None)` | simple | Vocabulary + rules |

All three are callable: `instance(text) -> str` (combining-acute output).

---

## Model registry

`MODEL_REGISTRY` maps model-id → metadata.  `DEFAULT_MODEL` maps language
tag → model-id.

```python
from stressonnx import MODEL_REGISTRY, DEFAULT_MODEL

print(MODEL_REGISTRY.keys())   # dict_keys(['ruaccent', 'silero', 'simple'])
print(DEFAULT_MODEL["ru"])     # 'ruaccent'
print(DEFAULT_MODEL["ukr"])    # 'silero'
print(DEFAULT_MODEL["kaz"])    # 'simple'
```

### Available models

#### `"ruaccent"` — homograph-aware Russian (default for `ru`)

Four-model ONNX pipeline derived from
[RUAccent](https://github.com/Den4ikAI/ruaccent) by Den4ikAI
(Apache-2.0 per upstream `setup.py` and PyPI classifiers).

| Stage | Model architecture | Size |
|-------|--------------------|------|
| Stress-usage classifier | BERT token classifier | 111 MB |
| Yo-homograph resolver | DistilBERT token classifier | 14 MB |
| Omograph resolver | RoBERTa NLI (turbo2) | 343 MB |
| Accent model | RoFormer char-level | 0.8 MB |

Runtime deps: `onnxruntime`, `numpy`, `tokenizers` (no torch, no
transformers).

Model files in HF repo: `ru_ruaccent/` sub-directory.

#### `"silero"` — neural ONNX (default for `ukr`, `bel`)

Fasttext-style n-gram embedding-bag + MLP heads exported from
`silero_stress` (MIT).  Supports exceptions dictionaries and skip sets.

| Tag | Language |
|-----|----------|
| `ukr` | Ukrainian |
| `bel` | Belarusian |

Model files per language: `<lang>/` sub-directory in HF repo.

#### `"simple"` — vocabulary + rules (default for 20 other languages)

Dictionary lookup with per-language OOV positional fallback.  No ONNX
inference, no runtime dependency beyond `numpy`.

| Tag | Language | OOV rule |
|-----|----------|----------|
| `aze_cyr` | Azerbaijani (Cyrillic) | last vowel |
| `aze_lat` | Azerbaijani (Latin) | last vowel |
| `uzb_cyr` | Uzbek (Cyrillic) | last vowel |
| `uzb_lat` | Uzbek (Latin) | last vowel |
| `bak` | Bashkir | last vowel |
| `bel_simple` | Belarusian (vocab only) | none (OOV unstressed) |
| `chv` | Chuvash | last vowel |
| `erz` | Erzya | first vowel |
| `hye` | Armenian | last vowel |
| `kat` | Georgian | ≤3 vowels→first, else penultimate |
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

**Note on Belarusian:** `bel` routes to the neural model (`"silero"`) by
default.  Use `model="simple"` or the `bel_simple` language alias for the
vocabulary-only path.

---

## Per-language defaults

| Language | Default model | Reasoning |
|----------|---------------|-----------|
| `ru` | `"ruaccent"` | Homograph resolution required for Russian |
| `ukr` | `"silero"` | Neural model available |
| `bel` | `"silero"` | Neural model available |
| all `SIMPLE_LANGS` | `"simple"` | Only vocab model exported for these languages |

---

## Installation

```bash
pip install stressonnx
```

The `tokenizers` package is required for `ru` (included in the package
dependencies).  `razdel` is an optional dependency that improves sentence
splitting for Russian; without it, text is treated as a single sentence.

### Export / dev tools (torch required)

```bash
pip install "stressonnx[export]"
```

See `stressonnx/export/ADDING_A_LANGUAGE.md` for instructions on adding a
new language or updating an existing model and uploading artefacts to
HuggingFace.

---

## HuggingFace model repo

Runtime artefacts: [TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)

Layout:

```
TigreGotico/stressonnx-models/
├── ru_ruaccent/          ← RUAccent pipeline (4 ONNX models + dicts)
│   ├── nn_omograph/
│   ├── nn_accent/
│   ├── nn_stress_usage/
│   ├── nn_yo_homograph/
│   └── dictionary/
├── ukr/                  ← silero neural (accentor.onnx + embedding.npy + …)
├── bel/                  ← silero neural
├── kaz/                  ← simple (vocab.gz + meta.json)
├── tat/                  ← simple
└── …                     ← one directory per simple-accentor language
```

---

## Attribution and licenses

| Component | Upstream source | License |
|-----------|----------------|---------|
| silero neural accentor (`"silero"` model, `ukr`/`bel`) | [silero-models/silero_stress](https://github.com/snakers4/silero-models) | MIT |
| silero simple accentor (`"simple"` model, 20 langs) | [silero-models/silero_stress](https://github.com/snakers4/silero-models) | MIT |
| RUAccent (`"ruaccent"` model, `ru`) | [Den4ikAI/ruaccent](https://github.com/Den4ikAI/ruaccent) | Apache-2.0 |

Russian homograph-aware accentuation is powered by
[RUAccent](https://github.com/Den4ikAI/ruaccent) by Den4ikAI, licensed
Apache-2.0 per upstream `setup.py` and PyPI classifiers.
