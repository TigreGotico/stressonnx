# stressonnx — architecture and design

## Overview

stressonnx is a pure-onnxruntime word-stress / accentuation library for
Russian, Ukrainian, Belarusian, and 20 other Slavic, Turkic, Uralic,
Caucasian, and Mongolic languages.

Runtime dependencies: `onnxruntime`, `numpy`, `huggingface_hub`,
`tokenizers` (Russian only).  No torch at runtime.

All ONNX models and vocabulary files are hosted at
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
(a public HF repo) and downloaded on first use.

---

## Package layout

The library is split into small, single-purpose modules instead of one
monolithic file:

| Module | Responsibility |
|--------|---------------|
| `stressonnx/registry.py` | `Script`, `StressNotation`, `ModelEntry`, `StressorBackend` Protocol; `MODEL_REGISTRY`, `DEFAULT_MODEL`; language-set constants (`RUACCENT_LANGS`, `MAIN_LANGS`, `SIMPLE_LANGS`, `ALL_LANGS`); `LANG_SCRIPT` + `lang_to_script()`; per-family file lists (`_MAIN_FILES`, `_SIMPLE_FILES`, `_KUBATABA_FILES`); `_OOV_RULES`. |
| `stressonnx/download.py` | The single download layer: `_download_files()` resolves model files via `huggingface_hub.hf_hub_download`, honoring the standard HF cache (`HF_HOME`, `HF_HUB_OFFLINE`) or an explicit `cache_dir` override. |
| `stressonnx/errors.py` | Typed exceptions: `StressonnxError` (base), `UnsupportedLanguageError` (`ValueError` subclass), `ModelDownloadError`. |
| `stressonnx/notation.py` | `STRESS_TOKEN` (U+0301); `_insert_stress()`, `_apply_notation()`, `_plus_to_diacritic()`, `_apostrophe_to_diacritic()` — all notation/format conversions. |
| `stressonnx/_common.py` | Small helpers shared by more than one backend: `_softmax`, `_RU_VOWELS`, `_RE_SPLIT` (shared tokenizer boundary regex), `_RE_RU_COND`, `_UNSTRESSED_HYPHEN_CLITICS`. |
| `stressonnx/stressor.py` | Public entry points: `make_stressor()` (factory) and the `Stressor` class. |
| `stressonnx/backends/ruaccent.py` | `RuAccentStressor` — the `ruaccent` family. |
| `stressonnx/backends/silero.py` | `_SileroStressor` — the `silero` family (`ukr`, `bel`, `ru`). |
| `stressonnx/backends/simple.py` | `SimpleStressor` — the `simple` family (20 languages). |
| `stressonnx/backends/kubataba.py` | `_KubatabaStressor` — the `kubataba` family (`ru` alternative). |
| `stressonnx/backends/__init__.py` | Re-exports the four backend classes. |
| `stressonnx/__init__.py` | Assembles the public API: `stress()`, `to_plus_notation()`, and re-exports from every module above. |
| `stressonnx/accentor.py` | Deprecated compatibility shim — re-exports everything that used to live in this single module so old imports (`from stressonnx.accentor import ...`) keep working. New code should import from `stressonnx` directly; this module carries no logic of its own. |

### Data flow

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
       └──→ RuAccentStressor()   (lazy-loads ONNX on first call, via
                                   stressonnx.download._download_files)
```

Three layers:

1. **`stress(text, lang, model=None, notation="diacritic", fallback=False)`**
   — module-level convenience function.  Singletons keyed by `(lang, model)`.
   Notation conversion via `_apply_notation()`.  When `fallback=True` and the
   selected model's files cannot be fetched (`ModelDownloadError`), walks
   `FALLBACK_PRIORITY` (`"ruaccent"`, `"silero"`, `"simple"`) for the same
   language, logging a warning at each hop; `bel` falls back to `bel_simple`.
2. **`Stressor(model=None, lang=None, notation=…)`** — public class.  Wraps
   the backend returned by `make_stressor()`.  Exposes `.model`, `.lang`, and
   `.notation` for introspection.
3. **`make_stressor(model, lang, cache_dir=None)`** — factory.  Consults
   `MODEL_REGISTRY`, validates the `(model, lang)` pair, and returns the
   appropriate backend.

---

## Typed error model

### `StressonnxError` hierarchy (`stressonnx/errors.py`)

```python
class StressonnxError(Exception):
    """Base class for all stressonnx errors."""

class UnsupportedLanguageError(StressonnxError, ValueError):
    """lang tag not recognized anywhere in the library.
    Also a ValueError, so existing except ValueError: code keeps working."""

class ModelDownloadError(StressonnxError):
    """A required model file could not be fetched from the HF Hub.
    Wraps the underlying huggingface_hub exception as __cause__ and names
    the exact repo path that failed."""
```

`stress(..., fallback=True)` catches `ModelDownloadError` internally and
retries with the next model in `FALLBACK_PRIORITY` for the same language,
re-raising only once the chain is exhausted.  With `fallback=False` (the
default) the error propagates on the first failure.

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
`UnsupportedLanguageError` (also catchable as plain `ValueError`) for
unknown tags.

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

`MODEL_REGISTRY` (in `stressonnx/registry.py`) maps model-id → `ModelEntry`.
`DEFAULT_MODEL` maps language tag → model-id, derived at import time with
priority: `ruaccent > silero > simple` (the same order as
`FALLBACK_PRIORITY`).

```python
MODEL_REGISTRY: dict[str, ModelEntry] = {
    "ruaccent": ModelEntry(
        langs=frozenset({"ru"}),
        family="ruaccent",
        hf_subdir="ru_ruaccent",
        description="…",
        input_scripts=frozenset({Script.CYRILLIC}),
    ),
    "silero": ModelEntry(
        langs=frozenset({"ukr", "bel", "ru"}),
        family="silero",
        hf_subdir=None,      # per-language: <lang>/
        description="…",
        input_scripts=frozenset({Script.CYRILLIC}),
    ),
    "simple": ModelEntry(
        langs=frozenset({…20 langs, including bel_simple…}),
        family="simple",
        hf_subdir=None,
        description="…",
        input_scripts=frozenset({Script.CYRILLIC, Script.LATIN, Script.ARMENIAN, Script.GEORGIAN}),
    ),
    "kubataba": ModelEntry(
        langs=frozenset({"ru"}),
        family="kubataba",
        hf_subdir="ru_kubataba",
        description="…",
        input_scripts=frozenset({Script.CYRILLIC}),
    ),
}
```

`bel` is served by both `silero` (the default, neural) and `simple` (via the
`bel_simple` alias, vocabulary-only).  Both are counted in `ALL_LANGS`; only
`bel` (not `bel_simple`) is a key in `DEFAULT_MODEL`, since `bel_simple` is
always explicit.

---

## Backend classes

### `RuAccentStressor` — ruaccent family (`stressonnx/backends/ruaccent.py`)

Homograph-aware Russian pipeline.  Four ONNX models loaded lazily from
`ru_ruaccent/` in the HF repo:

| Model | Architecture | Purpose |
|-------|-------------|---------|
| `nn_stress_usage` | BERT token classifier | STRESS / NO_STRESS per word |
| `nn_yo_homograph` | DistilBERT token classifier | е→ё restoration (not все/всё disambiguation — see `docs/models.md`) |
| `nn_omograph` | RoBERTa NLI (turbo2) | Pick stressed variant from homograph dict |
| `nn_accent` | RoFormer char-level | Accentuate words not in the accent dict, for words with 2+ vowels |

Tokenizers use the `tokenizers` library (HuggingFace fast tokenizer JSON
format) — no torch, no transformers required.  Input passes through an
aggressive normalization regex first (`_ruaccent_norm`), so unsupported
symbols are dropped rather than preserved.

### `_SileroStressor` — silero family (`stressonnx/backends/silero.py`)

Neural ONNX pipeline exported from silero_stress (MIT).
Supports `ukr`, `bel`, and `ru`.

Pipeline per word:
1. Tokenise sentence → `(raw_tokens, clean_tokens, prediction_mask)`.
2. Compute fastText-style n-gram embeddings (mean-pool from the embedding
   matrix).
3. Run ONNX MLP heads → `stress_logits [N, K]` (+ `yo_logits` for `ru`).
4. Decode: exceptions dict → skip sets → argmax position → insert U+0301
   (+ е→ё restoration for `ru`, via `_process_yo_lang`; `ukr`/`bel` use the
   simpler `_process_ukr_bel` path with no yo logic).

Single-vowel words are always force-stressed on that vowel, and a word
already containing U+0301 is returned unchanged (idempotent).

Files: `accentor.onnx`, `embedding.npy`, `ngram_dict.txt.gz`,
`exceptions.txt.gz`, `skip_stress_words.txt.gz`, `skip_yo_words.txt.gz`,
`meta.json` — under `<lang>/` in the HF repo.

### `_KubatabaStressor` — kubataba family (`stressonnx/backends/kubataba.py`)

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

### `SimpleStressor` — simple family (`stressonnx/backends/simple.py`)

Vocabulary + rule-based pipeline.  No ONNX inference.  Supports 20 languages.

Pipeline per word:
1. Tokenise sentence (shared boundary regex, hyphen-aware).
2. Look up clean lowercase token in `vocab` dict → stress character index.
3. OOV fallback: if the word has exactly one vowel, always stress it;
   otherwise apply the per-language positional rule (`"last"`, `"first"`,
   `"none"`, or `"kat"`).
4. Insert U+0301 at the determined character index.

A word already containing U+0301 is returned unchanged (idempotent).

`bel_simple` routes here: `SimpleStressor.__init__` strips the `_simple`
suffix to resolve the HF subdirectory (`bel`), so it downloads the same
vocab as the `bel` entry but always takes the rule-based path instead of
dispatching to `_SileroStressor`.

Files: `vocab.gz`, `meta.json` — under `<lang>/` in HF repo (`<lang>` is
`bel` for both `bel` and `bel_simple`).

---

## Notation pipeline

All backends produce combining-acute output internally.  The notation
conversion happens at the boundary layers only:

```
backend(text)            →  text with U+0301   (always)
_apply_notation(result)  →  diacritic or plus  (per user request)
```

`_apply_notation()` (in `stressonnx/notation.py`) is the single
implementation called from both `stress()` and `Stressor.__call__()`.

---

## File download and cache layout

`stressonnx.download._download_files(hf_subdir, filenames, cache_dir=None)`
is the single download layer every backend goes through — no backend
constructs a model path itself.

- **`cache_dir=None` (default):** files live in the standard Hugging Face
  cache.  `hf_hub_download` handles reuse, `HF_HOME` relocation, and
  `HF_HUB_OFFLINE` semantics, and the models are shared with every other HF
  consumer on the machine — there is no stressonnx-specific cache directory.
- **Explicit `cache_dir=...`:** the layout is exactly
  `cache_dir/<hf_subdir>/<file>` (never a doubled `hf_subdir/hf_subdir/`
  path).  Existing files under that path are used as-is without touching the
  network.

```
<cache_dir>/                      (only with an explicit cache_dir=)
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
│   └── …            ← shared by both bel (silero) and bel_simple (vocab)
├── kaz/
│   ├── vocab.gz
│   └── meta.json
└── …
```

---

## Adding a new model family

1. Implement a class with `__call__(self, text: str) -> str` (output:
   combining-acute) in its own module under `stressonnx/backends/`.
2. Add a `ModelEntry` to `MODEL_REGISTRY` in `stressonnx/registry.py` with
   `family="your_family"`.
3. Add a branch in `make_stressor()` (`stressonnx/stressor.py`) dispatching
   on `entry.family`.
4. Export ONNX artefacts and upload to
   `TigreGotico/stressonnx-models` (see `export/ADDING_A_LANGUAGE.md`).
5. Add tests in `tests/`.
