# stressonnx documentation

<!-- agent-hint: start at agents.md for layout, test command and pitfalls -->

stressonnx marks the stressed vowel of each word, so that a text-to-speech front end can pronounce text whose spelling does not show stress.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="img/gold-accuracy-dark.svg">
  <img alt="Default-model word accuracy on human-labelled sets: ru 0.951 and 0.990, be 0.924 to 0.977, uk 0.843 and 0.905; be homographs 0.543." src="img/gold-accuracy.svg">
</picture>

`img/plot_gold.py` makes this figure from `benchmarks/RESULTS.md`. The 23 languages other than ru, uk and be have no human-labelled test set.

## Quickstart

```bash
uv pip install stressonnx
```

```python
from stressonnx import stress

stress("старинный замок стоит на горе", "ru")   # 'стари́нный за́мок сто́ит на горе́'
stress("дверной замок надёжен", "ru")           # 'дверно́й замо́к надё́жен'
```

The output is the combining acute accent (U+0301) after the stressed vowel. The library supports 26 language tags and runs on `onnxruntime` and `numpy`, with no torch at runtime.

## Pages

| Page | Read it to |
| :--- | :--- |
| [Supported languages and models](supported-languages.md) | find the model for a language and its measured accuracy |
| [Usage](usage.md) | call `stress`, `analyze`, `stress_batch`, and choose a notation |
| [Offline use and failure behaviour](offline-and-errors.md) | run without a network, and handle download and load errors |
| [Language guide](languages.md) | learn why each language needs stress marks, with citations |
| [Model reference](models.md) | learn how each backend works and where it fails |
| [Architecture](architecture.md) | learn the package layout, registry, data flow and cache |
| [Add a language](adding-a-language.md) | add a language file and its sources |
| [Credits and licences](credits.md) | find the licence of each model and data set |
| [Agent hints](agents.md) | work on this repo as a coding agent |
