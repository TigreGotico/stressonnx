# stressonnx

Pure-onnxruntime, multi-language word-stress / accentuation library.
No torch at runtime — only `onnxruntime`, `numpy`, and `huggingface_hub`.

## API

```python
from stressonnx import stress

stress("Привет мир", "ru")        # → 'Прив+ет м+ир'
stress("Привіт світ", "ukr")      # → 'Прив+іт св+іт'
stress("Прывітанне свет", "bel")  # → 'Прывіт+анне св+ет'
stress("Salam dünya", "aze_lat")  # → 'Salam düny+a'
```

`stress(text, lang="ru") -> str` inserts `+` before each stressed vowel.
A per-language singleton is loaded lazily on first call; model files are
downloaded from HuggingFace and cached under `~/.local/share/stressonnx/<lang>/`.

```python
from stressonnx import Stressor, SimpleStressor

s = Stressor("bel")
s("Вада цячэ")          # → 'Вад+а цяч+э'

ss = SimpleStressor("kaz")
ss("Сәлем Қазақстан")   # → 'Сәл+ем Қазақст+ан'
```

## Supported languages

Two accentor families:

### Neural ONNX (main_accentor)

Embedding-bag + MLP heads, exported from silero_stress.  Supports exceptions
dictionaries and homograph skip sets.

| Tag | Language |
|-----|----------|
| `ru` | Russian |
| `ukr` | Ukrainian |
| `bel` | Belarusian |

### Vocabulary + rules (simple_accentor)

Dictionary lookup with per-language OOV positional rules.  No ONNX inference.

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

## Installation

```bash
pip install stressonnx
```

### Export / dev tools (torch required)

```bash
pip install "stressonnx[export]"
```

See `stressonnx/export/ADDING_A_LANGUAGE.md` for instructions on exporting a new
language from silero and uploading its artefacts to HuggingFace.

## HuggingFace model repo

Runtime artefacts: [TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
