# stressonnx

Pure-onnxruntime, multi-language word-stress / accentuation library.
No torch at runtime — only `onnxruntime`, `numpy`, and `huggingface_hub`.

## API

```python
from stressonnx import stress

stress("Привет мир", "ru")
# → 'Прив+ет м+ир'

stress("Замок на горе котёнок", "ru")
# → 'З+амок на гор+е кот+ёнок'
```

`stress(text, lang="ru") -> str` inserts `+` before each stressed vowel.
A per-language `Stressor` singleton is loaded lazily on first call; model files
are downloaded from HuggingFace and cached under `~/.local/share/stressonnx/<lang>/`.

```python
from stressonnx import Stressor

s = Stressor("ru")
s("Молоко убежало")   # → 'Молок+о убеж+ало'
```

## Supported languages

| Tag | Description |
|-----|-------------|
| `ru` | Russian (silero accentor, non-homograph path) |
| `uk` | Ukrainian — *planned* |
| `be` | Belarusian — *planned* |
| … | ~18 SimpleAccentor languages — *planned* |

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
