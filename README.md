# stressonnx

Pure-onnxruntime, multi-language word-stress / accentuation library.
No torch at runtime — only `onnxruntime`, `numpy`, `huggingface_hub`, and
`tokenizers`.

## API

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

`stress(text, lang="ru") -> str` inserts `+` before each stressed vowel.
Per-language singletons are loaded lazily on first call; model files are
downloaded from HuggingFace and cached under `~/.local/share/stressonnx/<lang>/`.

```python
from stressonnx import RuAccentStressor, Stressor, SimpleStressor

# Russian — homograph-aware
ra = RuAccentStressor()
ra("белок яйца полезен")    # → 'бел+ок яйц+а пол+езен'   (egg-white / protein)
ra("белка белок ела орехи") # → 'б+елка б+елок +ела ор+ехи' (squirrel / squirrel's)

s = Stressor("bel")
s("Вада цячэ")              # → 'Вад+а цяч+э'

ss = SimpleStressor("kaz")
ss("Сәлем Қазақстан")       # → 'Сәл+ем Қазақст+ан'
```

## Supported languages

### Russian — homograph-aware (RUAccent)

`ru` uses a four-model neural pipeline derived from
[RUAccent](https://github.com/Den4ikAI/ruaccent) (Den4ikAI), licensed
Apache-2.0 per upstream setup.py and PyPI classifiers.  Models sourced from
[`ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator) (turbo2
omograph model) and mirrored to
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
under `ru_ruaccent/`.

Pipeline (all pure onnxruntime + numpy + tokenizers, no torch):

1. **Stress-usage classifier** (BERT, 111 MB) — predicts STRESS / NO_STRESS per
   word in sentence context, so function words and abbreviations are left
   unstressed.
2. **Yo-homograph resolver** (DistilBERT, 14 MB) — disambiguates е→ё for words
   with context-dependent yo substitution.
3. **Omograph resolver** (RoBERTa NLI, turbo2, 343 MB) — picks the correct
   stressed variant from the homograph dictionary (e.g. з+амок castle vs
   зам+ок lock) by scoring each candidate against the sentence context.
4. **Accent model** (RoFormer char-level, 0.8 MB) — accentuates any remaining
   words not found in the accent dictionary.

### Neural ONNX (main_accentor)

Embedding-bag + MLP heads, exported from silero_stress.  Supports exceptions
dictionaries and homograph skip sets.

| Tag | Language |
|-----|----------|
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

The `tokenizers` package is required for `ru` (included in dependencies).
`razdel` is an optional dependency that improves sentence splitting for Russian;
if not installed, text is treated as a single sentence.

### Export / dev tools (torch required)

```bash
pip install "stressonnx[export]"
```

See `stressonnx/export/ADDING_A_LANGUAGE.md` for instructions on exporting a new
language from silero and uploading its artefacts to HuggingFace.

## HuggingFace model repo

Runtime artefacts: [TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)

Russian (`ru`) model files live under `ru_ruaccent/` in that repo and are
mirrored from [`ruaccent/accentuator`](https://huggingface.co/ruaccent/accentuator).

## Attribution

Russian homograph-aware accentuation is powered by
[RUAccent](https://github.com/Den4ikAI/ruaccent) by Den4ikAI, licensed
Apache-2.0 per upstream `setup.py` and PyPI classifiers.
