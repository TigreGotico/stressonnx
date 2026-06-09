# stressonnx — Architecture and model-wrapper design

## Overview

stressonnx is a pure-onnxruntime word-stress / accentuation library for
Russian, Ukrainian, Belarusian, and 20 other Slavic, Turkic, and Caucasian
languages.  The runtime stack is: `onnxruntime` + `numpy` +
`huggingface_hub` + `tokenizers` (for Russian only).  No torch at runtime.

All ONNX models and vocabulary files are hosted at
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
and downloaded to `~/.local/share/stressonnx/<lang>/` on first use.

---

## Model-wrapper architecture

stressonnx follows the same ergonomics as `text2tashkeel`'s
`Diacritizer(model=…)` pattern in phoonnx: the **model is a first-class
input**, not a hidden dispatch detail.

```
stress("замок", "ru", model="ruaccent")
     │
     ▼
  stress()  ──→  _SINGLETONS cache (lang, model) key
     │
     ▼
  make_stressor(model="ruaccent", lang="ru")
     │
     ├── lookup MODEL_REGISTRY["ruaccent"]
     │         {"family": "ruaccent", "langs": {"ru"}, …}
     │
     └──→ RuAccentStressor()   (lazy-loads ONNX on first call)
```

Three layers:

1. **`stress(text, lang, model=None)`** — module-level convenience function.
   Singletons keyed by `(lang, model)`.
2. **`Stressor(model=None, lang=None)`** — public class.  Wraps the backend
   returned by `make_stressor()`.  Exposes `.model` and `.lang` attributes
   for introspection.
3. **`make_stressor(model, lang)`** — factory.  Consults `MODEL_REGISTRY`,
   validates the `(model, lang)` pair, and returns the appropriate backend
   instance.

---

## Model registry

`MODEL_REGISTRY` (dict, module-level) maps model-id → entry:

```python
MODEL_REGISTRY = {
    "ruaccent": {
        "langs":       frozenset({"ru"}),
        "family":      "ruaccent",
        "hf_subdir":   "ru_ruaccent",
        "description": "…",
    },
    "silero": {
        "langs":       frozenset({"ukr", "bel"}),
        "family":      "silero",
        "hf_subdir":   None,   # per-language: <lang>/
        "description": "…",
    },
    "simple": {
        "langs":       frozenset({…20 langs…}),
        "family":      "simple",
        "hf_subdir":   None,
        "description": "…",
    },
}
```

`DEFAULT_MODEL` (dict) maps every supported language tag to its preferred
model-id.  It is derived from `MODEL_REGISTRY` at import time: RUAccent
languages get `"ruaccent"`, silero neural languages get `"silero"`, and
remaining languages get `"simple"`.

---

## Backend classes

### `RuAccentStressor` — ruaccent family

Homograph-aware Russian pipeline.  Four ONNX models loaded lazily from
`ru_ruaccent/` in the HF repo:

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `nn_stress_usage` | BERT token classifier | Predict STRESS / NO_STRESS per word in sentence context |
| `nn_yo_homograph` | DistilBERT token classifier | Disambiguate е→ё for yo-homographs |
| `nn_omograph` | RoBERTa NLI (turbo2) | Pick correct stressed variant from homograph dict |
| `nn_accent` | RoFormer char-level | Accentuate words not found in the accent dictionary |

Tokenizers are loaded via the `tokenizers` library (HuggingFace fast
tokenizer JSON format), without requiring `transformers` or `torch`.

All four dictionaries (`omographs.json.gz`, `yo_words.json.gz`,
`yo_homographs.json.gz`, `accents_nn.json.gz`) are also downloaded from
the HF repo.

Attribution: RUAccent by Den4ikAI (<https://github.com/Den4ikAI/ruaccent>),
Apache-2.0 per upstream `setup.py` and PyPI classifiers.

### `_SileroStressor` — silero family

Neural ONNX pipeline exported from silero_stress (MIT).  Supports
`ukr` and `bel`.

Pipeline per word:
1. Tokenise sentence → `(raw_tokens, clean_tokens, prediction_mask)`.
2. Compute fastText-style n-gram embeddings (mean-pool rows from the
   embedding matrix).
3. Run ONNX MLP heads → `stress_logits [N, K]`.
4. Decode: exceptions dict → skip sets → argmax position → insert `+`.

Files downloaded: `accentor.onnx`, `embedding.npy`, `ngram_dict.txt.gz`,
`exceptions.txt.gz`, `skip_stress_words.txt.gz`, `skip_yo_words.txt.gz`,
`meta.json` — all under `<lang>/` in the HF repo.

### `SimpleStressor` — simple family

Vocabulary + rule-based pipeline.  No ONNX inference.  Supports 20
languages.

Pipeline per word:
1. Tokenise sentence (split on whitespace and punctuation, handle hyphens).
2. Look up clean lowercase token in `vocab` dict (word → stress char index).
3. OOV fallback: per-language positional rule (`"last"`, `"first"`,
   `"none"`, or `"kat"`).
4. Insert `+` at the determined character index in the raw token.

Files downloaded: `vocab.gz`, `meta.json` — under `<lang>/` in HF repo.

---

## Comparison with `text2tashkeel` in phoonnx

| Aspect | stressonnx | phoonnx / text2tashkeel |
|--------|-----------|------------------------|
| Model parameter name | `model=` | `model=` |
| Registry | `MODEL_REGISTRY` dict | text2tashkeel's own registry |
| Default selection | `DEFAULT_MODEL[lang]` | `diacritizer_model` config key |
| Public callable | `Stressor(model, lang)` | `Diacritizer(model)` |
| Lazy loading | yes (per-backend) | yes |
| Runtime | onnxruntime + numpy | onnxruntime |

---

## Adding a new model

1. Export ONNX artefacts (see `stressonnx/export/ADDING_A_LANGUAGE.md` for
   the full guide).
2. Upload to `TigreGotico/stressonnx-models/<new_lang>/`.
3. Add the new language tag to the appropriate set (`MAIN_LANGS` or
   `SIMPLE_LANGS`) in `accentor.py`.
4. Add a `MODEL_REGISTRY` entry (or extend an existing one's `langs`
   frozenset) and set `DEFAULT_MODEL[new_lang]`.
5. Add tests in `tests/`.

To add a **new model family** (e.g. a third-party accentor for a different
language group):
1. Implement a class with `__call__(self, text: str) -> str`.
2. Register it in `MODEL_REGISTRY` with `"family": "your_family"`.
3. Add a branch in `make_stressor()` for the new family.

---

## Cache layout

```
~/.local/share/stressonnx/
├── ru_ruaccent/
│   ├── nn_omograph/model.onnx
│   ├── nn_omograph/tokenizer.json
│   ├── nn_accent/model.onnx
│   ├── nn_stress_usage/model.onnx
│   ├── nn_yo_homograph/model.onnx
│   └── dictionary/omographs.json.gz  …
├── ukr/
│   ├── accentor.onnx
│   ├── embedding.npy
│   └── …
├── bel/
│   └── …
├── kaz/
│   ├── vocab.gz
│   └── meta.json
└── …
```

Override with `cache_dir=` on any constructor or `make_stressor()`.
