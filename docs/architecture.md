# stressonnx — architecture and design

## Overview

stressonnx is a pure-onnxruntime word-stress / accentuation library for
Russian, Ukrainian, Belarusian, and 20 other Slavic, Turkic, and Caucasian
languages.

Runtime dependencies: `onnxruntime`, `numpy`, `huggingface_hub`,
`tokenizers` (Russian only).  No torch at runtime.

All ONNX models and vocabulary files are hosted at
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
and downloaded on first use.

---

## Layered API

```
stress("замок", "ru", model="ruaccent")
       │
       ▼
  stress()  ──→  _SINGLETONS cache keyed by (lang, model)
       │
       ▼
  make_stressor(model="ruaccent", lang="ru")
       │
       ├── MODEL_REGISTRY["ruaccent"]  →  ModelEntry(family="ruaccent", …)
       │
       └──→ RuAccentStressor()   (lazy-loads ONNX on first call)
```

Three layers:

1. **`stress(text, lang, model=None, notation="diacritic")`** — module-level
   convenience function. Singletons keyed by `(lang, model)`. Notation
   conversion via `_apply_notation()`.
2. **`Stressor(model=None, lang=None, notation=…)`** — public class. Wraps
   the backend returned by `make_stressor()`. Exposes `.model`, `.lang`, and
   `.notation` for introspection.
3. **`make_stressor(model, lang)`** — factory. Consults `MODEL_REGISTRY`,
   validates the `(model, lang)` pair, and returns the appropriate backend.

---

## Typed data model

### `ModelEntry` (frozen dataclass)

```python
@dataclass(frozen=True)
class ModelEntry:
    langs: frozenset[str]          # supported language tags
    family: str                    # dispatch key used by make_stressor
    hf_subdir: str | None          # fixed HF subdirectory, or None (per-lang)
    description: str
    input_scripts: frozenset[Script]  # writing systems the model accepts
```

`input_scripts` enables phoonnx (and other callers) to guard against script
mismatches before delegating to stressonnx:

```python
if lang_to_script(lang) in MODEL_REGISTRY[model_id].input_scripts:
    text = stress(text, lang, model=model_id)
```

### `Script` (str enum)

```python
class Script(str, Enum):
    CYRILLIC = "cyrillic"   # ru, ukr, bel, kaz, tat, bak, …
    LATIN    = "latin"      # aze_lat, uzb_lat
    ARMENIAN = "armenian"   # hye
    GEORGIAN = "georgian"   # kat
```

Mirrors the script taxonomy used in phoonnx's `Alphabet` enum.  New
writing systems can be added here when new language families are registered.

### `StressNotation` (str enum)

```python
class StressNotation(str, Enum):
    DIACRITIC = "diacritic"   # combining acute U+0301 after stressed vowel
    PLUS      = "plus"        # legacy + before stressed vowel
```

Inheriting `str` keeps backward compatibility: `notation="plus"` still
accepted wherever `StressNotation` is expected.

### `StressorBackend` (runtime-checkable Protocol)

```python
@runtime_checkable
class StressorBackend(Protocol):
    def __call__(self, text: str) -> str: ...
```

All four backend classes satisfy this protocol.  `make_stressor()` return
type is `StressorBackend`.

---

## Script routing

`LANG_SCRIPT: dict[str, Script]` maps every language tag to its writing
system.  `lang_to_script(lang)` is the public accessor — it raises
`ValueError` for unknown tags.

```python
lang_to_script("ru")      # Script.CYRILLIC
lang_to_script("aze_lat") # Script.LATIN
lang_to_script("hye")     # Script.ARMENIAN
lang_to_script("kat")     # Script.GEORGIAN
```

### Relationship to phoonnx

phoonnx uses an `Alphabet` enum to describe both phonetic representations
(IPA, ARPA, SAMPA, …) and writing scripts (UNICODE, HANGUL, KANA, PINYIN,
…).  When phoonnx needs to insert stress marks before phonemisation, it will:

1. Resolve the language's writing system with `stressonnx.lang_to_script(lang)`.
2. Verify the script appears in `ModelEntry.input_scripts` for the chosen model.
3. Call `stressonnx.stress(text, lang)` — receiving combining-acute output.
4. Feed the stressed text to its phonemiser.

The `Script` values (`"cyrillic"`, `"latin"`, …) are kept compatible with
phoonnx's naming so the mapping layer in phoonnx is trivial.

---

## Model registry

`MODEL_REGISTRY` maps model-id → `ModelEntry`.
`DEFAULT_MODEL` maps language tag → model-id, derived at import time with
priority: `ruaccent > silero > simple`.

```python
MODEL_REGISTRY: dict[str, ModelEntry] = {
    "ruaccent": ModelEntry(
        langs=frozenset({"ru"}),
        family="ruaccent",
        hf_subdir="ru_ruaccent",
        description="…",
    ),
    "silero": ModelEntry(
        langs=frozenset({"ukr", "bel", "ru"}),
        family="silero",
        hf_subdir=None,      # per-language: <lang>/
        description="…",
    ),
    "simple": ModelEntry(
        langs=frozenset({…20 langs…}),
        family="simple",
        hf_subdir=None,
        description="…",
    ),
    "kubataba": ModelEntry(
        langs=frozenset({"ru"}),
        family="kubataba",
        hf_subdir="ru_kubataba",
        description="…",
    ),
}
```

---

## Backend classes

### `RuAccentStressor` — ruaccent family

Homograph-aware Russian pipeline.  Four ONNX models loaded lazily from
`ru_ruaccent/` in the HF repo:

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `nn_stress_usage` | BERT token classifier | STRESS / NO_STRESS per word |
| `nn_yo_homograph` | DistilBERT token classifier | Disambiguate е→ё |
| `nn_omograph` | RoBERTa NLI (turbo2) | Pick stressed variant from homograph dict |
| `nn_accent` | RoFormer char-level | Accentuate words not in the accent dict |

Tokenizers use the `tokenizers` library (HuggingFace fast tokenizer JSON
format) — no torch, no transformers required.

### `_SileroStressor` — silero family

Neural ONNX pipeline exported from silero_stress (MIT).
Supports `ukr`, `bel`, and `ru`.

Pipeline per word:
1. Tokenise sentence → `(raw_tokens, clean_tokens, prediction_mask)`.
2. Compute fastText-style n-gram embeddings (mean-pool from the embedding
   matrix).
3. Run ONNX MLP heads → `stress_logits [N, K]`.
4. Decode: exceptions dict → skip sets → argmax position → insert U+0301.

Files: `accentor.onnx`, `embedding.npy`, `ngram_dict.txt.gz`,
`exceptions.txt.gz`, `skip_stress_words.txt.gz`, `skip_yo_words.txt.gz`,
`meta.json` — under `<lang>/` in the HF repo.

### `_KubatabaStressor` — kubataba family

Sentence-level encoder-decoder Transformer derived from
[kubataba/Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor).

The PyTorch model uses a character-level vocabulary.
During ONNX export (see `export/export_kubataba.py`) it is split into:

- **`encoder.onnx`**: `src (1, 256) int64 → memory (1, 256, 256) float32`
- **`decoder_step.onnx`**: `(memory, tgt) → logits (1, vocab_size) float32`

The autoregressive greedy-decode loop runs in Python.  The critical insight
that made the export possible: PyTorch's `nn.MultiheadAttention` is
incompatible with TorchScript and dynamo exporters when sequence length is
dynamic (`_detect_is_causal_mask` calls `torch.ones(t, t)` with a
data-dependent `t`).  The decoder step wrapper reimplements the decoder
forward pass manually using `F.scaled_dot_product_attention` with
`is_causal=True`, bypassing `nn.Transformer` entirely.

Files: `encoder.onnx`, `decoder_step.onnx`, `vocab.json` — under
`ru_kubataba/` in the HF repo.

### `SimpleStressor` — simple family

Vocabulary + rule-based pipeline.  No ONNX inference.  Supports 20 languages.

Pipeline per word:
1. Tokenise sentence (whitespace + punctuation, hyphen-aware).
2. Look up clean lowercase token in `vocab` dict → stress character index.
3. OOV fallback: per-language positional rule (`"last"`, `"first"`,
   `"none"`, or `"kat"`).
4. Insert U+0301 at the determined character index.

Files: `vocab.gz`, `meta.json` — under `<lang>/` in HF repo.

---

## Notation pipeline

All backends produce combining-acute output internally.  The notation
conversion happens at the boundary layers only:

```
backend(text)            →  text with U+0301   (always)
_apply_notation(result)  →  diacritic or plus  (per user request)
```

`_apply_notation()` is the single implementation called from both
`stress()` and `Stressor.__call__()`.

---

## File download helper

`_download_files(hf_subdir, cache, filenames)` wraps `hf_hub_download` for
all backends.  It handles nested HF subdirectories and caches to
`~/.local/share/stressonnx/` by default.  Override with `cache_dir=`.

---

## Adding a new model family

1. Implement a class with `__call__(self, text: str) -> str` (output:
   combining-acute).
2. Add a `ModelEntry` to `MODEL_REGISTRY` with `family="your_family"`.
3. Add a branch in `make_stressor()` dispatching on `entry.family`.
4. Export ONNX artefacts and upload to HF (see
   `export/ADDING_A_LANGUAGE.md`).
5. Add tests in `tests/`.

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
├── ru_kubataba/
│   ├── encoder.onnx
│   ├── decoder_step.onnx
│   └── vocab.json
├── ukr/
│   ├── accentor.onnx
│   ├── embedding.npy
│   └── …
├── ru/
│   └── …           ← silero Russian
├── bel/
│   └── …
├── kaz/
│   ├── vocab.gz
│   └── meta.json
└── …
```

Override with `cache_dir=` on any constructor or `make_stressor()`.
