# stressonnx

Pure-onnxruntime, multi-language word-stress / accentuation library.
No torch at runtime — only `onnxruntime`, `numpy`, `huggingface_hub`, and
`tokenizers` (the last is required for `ru` only).

All models are hosted on
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
and downloaded + cached automatically on first use.

---

## Quick start

```python
from stressonnx import stress

# Russian — homograph-aware (замок castle vs lock, мука flour vs torment, …)
stress("старинный замок стоит на горе", "ru")   # → 'стар+инный з+амок ст+оит на гор+е'
stress("дверной замок надёжен", "ru")           # → 'дверн+ой зам+ок надёжен'
stress("мука для хлеба", "ru")                  # → 'мук+а для хл+еба'
stress("мука была невыносима", "ru")            # → 'м+ука был+а невынос+има'

# Other languages
stress("Привіт світ", "ukr")      # → 'Прив+іт св+іт'
stress("Прывітанне свет", "bel")  # → 'Прывіт+анне св+ет'
stress("Salam dünya", "aze_lat")  # → 'Salam düny+a'
```

---

## API

### `stress(text, lang="ru", model=None) -> str`

Insert `+` before the stressed vowel of each word.

| Parameter | Type | Description |
|-----------|------|-------------|
| `text` | `str` | Input text. |
| `lang` | `str` | Language tag (e.g. `"ru"`, `"ukr"`, `"kaz"`). Defaults to `"ru"`. |
| `model` | `str \| None` | Model id — `"ruaccent"`, `"silero"`, or `"simple"`. `None` selects the default model for `lang` (see [model registry](#model-registry)). |

```python
from stressonnx import stress

# Explicit model selection
stress("Привіт світ", "ukr", model="silero")    # neural
stress("Казан",       "tat", model="simple")    # vocabulary + rules
stress("замок",       "ru",  model="ruaccent")  # homograph-aware
```

Singletons are loaded lazily on first call; subsequent calls for the same
`(lang, model)` pair reuse the cached instance.

### `Stressor(model=None, lang=None, cache_dir=None)`

High-level class.  Accepts the same `model` and `lang` parameters as
`stress()`.  Use this when you want to hold a reference to the accentor
rather than calling the module-level function.

```python
from stressonnx import Stressor

# Russian (default model: ruaccent)
s = Stressor(lang="ru")
s("белок яйца полезен")     # → 'бел+ок яйц+а пол+езен'

# Ukrainian — neural
s = Stressor(model="silero", lang="ukr")
s("Привіт світ")            # → 'Прив+іт св+іт'

# Kazakh — vocab + rules
s = Stressor(model="simple", lang="kaz")
s("Сәлем Қазақстан")        # → 'Сәл+ем Қазақст+ан'
```

`Stressor` exposes:
- `s.model` — the model-id string selected at construction.
- `s.lang` — the language tag.
- `s(text)` — accentuate a string.

### `make_stressor(model=None, lang=None, cache_dir=None)`

Factory function: returns a backend stressor instance without the singleton
caching.  Useful when you need multiple independent instances or want to
pass a custom `cache_dir`.

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

All three are callable: `instance(text) -> str`.

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
