# Adding a language or model to stressonnx

This guide assumes no prior familiarity with stressonnx, ONNX export, or the
`silero_stress` project.  Every term is defined the first time it is used, so
you can follow it end-to-end even if this is your first contribution.

## Glossary

- **Stress** (or **accent**): in these languages, the syllable of a word that
  is pronounced with more emphasis.  It is usually not written in ordinary
  text, so a "stress accentor" is a program that adds a mark showing where
  it falls.
- **Combining acute (U+0301)**: the Unicode character stressonnx places
  *after* the stressed vowel to mark it, e.g. `приве́т`.  It is the notation
  every backend in this library produces internally.
- **Vocabulary (vocab)**: a lookup table mapping known words to where their
  stress falls.  Words not in the vocabulary are called **OOV**.
- **OOV (out-of-vocabulary)**: a word the vocabulary doesn't contain.  Since
  the correct stress can't be looked up, a fallback rule guesses a position
  instead (e.g. "stress the last vowel").
- **`vocab.gz`**: the on-disk vocabulary format used by the `simple` family —
  a gzip-compressed text file, one entry per line, formatted
  `word<space>index`, where `index` is the character position (0-based,
  counting from the start of the word) of the letter that should carry the
  stress mark.  Example line: `привет 4` (the `е` at position 4 is stressed:
  `прив`**`е`**`т`).
- **`meta.json`**: a small JSON file shipped alongside `vocab.gz` describing
  the language: its alphabet (`alpha`), which characters count as vowels
  (`vowels`), and the OOV fallback rule (`oov_rule`).
- **ONNX** (Open Neural Network Exchange): a portable file format for trained
  neural networks.  stressonnx runs ONNX models with `onnxruntime` — it never
  needs `torch` (PyTorch) at *runtime*, only at *export* time, to convert an
  existing PyTorch model into an `.onnx` file.
- **ONNX export**: the one-time conversion step (`torch.onnx.export(...)`)
  that turns a PyTorch model into a `.onnx` file so it can run without
  PyTorch afterward.
- **HF subdir**: the sub-directory inside the
  [`TigreGotico/stressonnx-models`](https://huggingface.co/TigreGotico/stressonnx-models)
  Hugging Face repository where one model's runtime files live (e.g.
  `ru_ruaccent/`, `kaz/`).  This is the `hf_subdir` field of a `ModelEntry`
  in `stressonnx/registry.py`.
- **`silero_stress`**: the upstream MIT-licensed project stressonnx exports
  its `silero` and `simple` model families from.  It ships pre-trained
  accentors and vocabularies for many languages as a Python package.

---

## The two ways to add a language

| You have… | Use… |
|-----------|------|
| A word-list with each word's stressed syllable marked (a stressed wordlist) | **Path A: simple-accentor language** — no PyTorch/ONNX involved, pure data. |
| A neural model you want to export to ONNX (or you're adding a whole new model family) | **Path B: neural / custom ONNX path**. |

**Important:** all 20 upstream `silero_stress` `SimpleAccentor` vocabularies
are already exported and registered in stressonnx (`aze_cyr`, `aze_lat`,
`bak`, `bel_simple`, `chv`, `erz`, `hye`, `kat`, `kaz`, `kbd`, `kir`, `kjh`,
`mdf`, `sah`, `tat`, `tgk`, `udm`, `uzb_cyr`, `uzb_lat`, `xal`).  Adding a new
simple-accentor language means finding or building a **new** stressed
wordlist from some other source — running
`export/export_simple_accentors.py` again for a language already in that
list will just re-export the same data.

---

## Path A: adding a simple-accentor language from a stressed wordlist

Use this path when you have — or can build — a list of words in the target
language, each with its stress position marked, but no neural model.

### 1. Prepare `vocab.gz`

Build a gzip text file with one entry per line:

```
word<space>index
```

`index` is the 0-based character position of the stressed vowel, counting
characters (not bytes) from the start of the word.  Lowercase the words —
lookup in stressonnx is case-insensitive.

```python
import gzip

entries = [
    ("привет", 4),   # прив-Е-т, е at index 4 is stressed
    ("мама", 1),     # м-А-ма
    # ... thousands more, one line per known word
]

with gzip.open("vocab.gz", "wt", encoding="utf-8") as fh:
    for word, idx in entries:
        fh.write(f"{word} {idx}\n")
```

### 2. Write `meta.json`

```json
{
  "family": "simple_accentor",
  "lang": "xyz",
  "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюя",
  "vowels": "аеиоуыэюя",
  "oov_rule": "last",
  "n_vocab": 12345
}
```

- `alpha`: every letter that can appear in the language, lowercase (the
  regex tokenizer strips anything outside this set before lookup).
- `vowels`: the subset of `alpha` that counts as a vowel — used by the OOV
  fallback rule to pick a stress position.
- `oov_rule`: what to do for a word not found in `vocab.gz` (**only** applied
  when the word has 2+ vowels — a single-vowel OOV word is *always* stressed
  on that one vowel, matching upstream `SimpleAccentor` behavior; see
  `docs/models.md` for the monosyllable-policy table):
  - `"last"` — stress the rightmost vowel (the most common rule; most
    Turkic languages follow this).
  - `"first"` — stress the leftmost vowel.
  - `"none"` — leave the word unmarked (used by `bel_simple`).
  - `"kat"` — Georgian's special rule: ≤3 vowels → first vowel, else
    penultimate vowel.

### 3. Upload to Hugging Face

```bash
pip install huggingface_hub
huggingface-cli login   # needs write access to TigreGotico/stressonnx-models
huggingface-cli upload TigreGotico/stressonnx-models ./xyz/ xyz/
```

This uploads `vocab.gz` and `meta.json` to the `xyz/` HF subdir — the same
name you will use as the language tag.

### 4. Register the language in `stressonnx/registry.py`

Three edits, all in `stressonnx/registry.py`:

```python
# 1. Add the tag to SIMPLE_LANGS
SIMPLE_LANGS = {
    "aze_cyr", "aze_lat", ...,
    "xyz",   # <-- new
}

# 2. Add its writing system to LANG_SCRIPT
LANG_SCRIPT: dict[str, Script] = {
    ...,
    "xyz": Script.CYRILLIC,   # or LATIN / ARMENIAN / GEORGIAN
}

# 3. Add its OOV rule to _OOV_RULES
_OOV_RULES = {
    ...,
    "xyz": "last",
}
```

`ALL_LANGS`, `MODEL_REGISTRY["simple"].langs`, and `DEFAULT_MODEL["xyz"]` are
all derived automatically from `SIMPLE_LANGS` — no further registry edits
needed.

### 5. Add tests

Add a case to `tests/test_stress_simple.py` (or a new
`tests/test_stress_<xyz>.py`) covering: a known-vocabulary word, an OOV word
that exercises the fallback rule, and a single-vowel OOV word (must always be
stressed regardless of `oov_rule`).

```python
from stressonnx import stress

def test_xyz_known_word():
    assert stress("привет", "xyz") == "прив́ет"  # adjust to your data

def test_xyz_oov_fallback():
    result = stress("совершенноновоеслово", "xyz")
    assert "́" in result  # or assert no mark, if oov_rule == "none"

def test_xyz_single_vowel_oov_always_stressed():
    result = stress("та", "xyz")   # OOV monosyllable
    assert "́" in result
```

### 6. Try it

```bash
source ~/.venvs/ovos/bin/activate   # or your own venv with stressonnx installed
python -c "from stressonnx import stress; print(stress('привет мир', 'xyz'))"
```

Run the full suite before opening a PR:

```bash
pytest tests/ -q
```

---

## Path B: neural / custom ONNX path

Use this path to export a new neural accentor — either extending the
`silero` family for `ukr`/`bel`/`ru` (unlikely — those three are already
covered), or bringing in a genuinely new model architecture (the reference
implementation for this is `kubataba`, a seq2seq Transformer).

`export/` requires `torch` (only at export time — the runtime library never
imports it).  Install it with `pip install "stressonnx[export]"`.

### B.1 — silero-family export (`main_accentor`)

Each language in this family ships one ONNX file (an MLP classifier head)
plus supporting numpy/gzip data files.

**Export:**

```bash
pip install "stressonnx[export]"
python export/export_main_accentors.py --lang ukr --out_dir /tmp/ukr
```

The script:
- Loads the silero accentor for the target language from `silero_stress`.
- Rebuilds the scripted `stress_clf` head as a plain `nn.Sequential` (ONNX
  export needs a plain module graph, not a TorchScript-compiled one).
- Exports it to `accentor.onnx` via `torch.onnx.export`.
- Dumps `embedding.npy` (the n-gram embedding matrix — `onnxruntime` has no
  native `EmbeddingBag` op, so the pooling step runs in NumPy instead, not
  inside the ONNX graph), `ngram_dict.txt.gz`, `exceptions.txt.gz`,
  `skip_stress_words.txt.gz`, `skip_yo_words.txt.gz`, and `meta.json`
  (containing `family`, `has_yo`, `alpha`, `vowels`, and embedding stats).
- Verifies the ONNX classifier's output matches the original PyTorch output
  (max absolute difference < 1e-3).

**Upload:**

```bash
huggingface-cli upload TigreGotico/stressonnx-models /tmp/ukr/ <lang>/
```

**Register** in `stressonnx/registry.py`: add the tag to `MAIN_LANGS`.
`DEFAULT_MODEL` and `MODEL_REGISTRY["silero"].langs` are derived
automatically from `MAIN_LANGS`.

**Test:**

```python
from stressonnx import stress
print(stress("Привіт світ", "ukr"))
```

**Verify end-to-end match** against the original silero output:

```bash
python export/verify_e2e.py --lang ukr
# Should report: EXACT MATCH: N/N = 100.0%
```

### B.2 — custom ONNX family (e.g. seq2seq Transformer, like `kubataba`)

Use this when the model architecture does not fit the silero pipeline — for
example a sentence-to-sentence Transformer.  `kubataba` is the reference
implementation; read `export/export_kubataba.py` alongside this section.

**1. Export to ONNX.**  For a seq2seq model with dynamic sequence length,
PyTorch's standard exporters (TorchScript and dynamo) often fail when the
model computes data-dependent shapes internally — e.g.
`nn.MultiheadAttention` building a causal mask of shape `(t, t)` for a
data-dependent `t`.  The `kubataba` solution: split the model into an
**encoder** and a **one-step decoder**, and reimplement the decoder forward
pass manually with `F.scaled_dot_product_attention(is_causal=True)`,
bypassing `nn.Transformer` (and its problematic `_detect_is_causal_mask`
call) entirely.

```python
# encoder.onnx: static input shape is fine
torch.onnx.export(
    encoder_wrapper,
    (dummy_src,),
    "encoder.onnx",
    input_names=["src"],
    output_names=["memory"],
    dynamic_axes={"src": {1: "src_len"}, "memory": {1: "src_len"}},
)

# decoder_step.onnx: one step at a time (tgt grows by 1 each step)
torch.onnx.export(
    decoder_step_wrapper,    # custom wrapper, NOT nn.Transformer
    (dummy_memory, dummy_tgt),
    "decoder_step.onnx",
    input_names=["memory", "tgt"],
    output_names=["logits"],
    dynamic_axes={
        "memory": {1: "src_len"},
        "tgt":    {1: "tgt_len"},
    },
)
```

**Merging external data:** the dynamo exporter may produce
`decoder_step.onnx` + a separate `decoder_step.onnx.data` file.  Merge them
into one self-contained file before uploading:

```python
import onnx
from onnx.external_data_helper import load_external_data_for_model

model = onnx.load("decoder_step.onnx", load_external_data=False)
load_external_data_for_model(model, "/tmp/export_dir")
onnx.save_model(model, "decoder_step_merged.onnx", save_as_external_data=False)
```

**2. Upload to HuggingFace:**

```bash
huggingface-cli upload TigreGotico/stressonnx-models /tmp/ru_mymodel/ ru_mymodel/
```

The subdirectory name becomes `hf_subdir` on the `ModelEntry`.

**3. Register the model.**  Four edits in the codebase:

```python
# stressonnx/registry.py

# a. List the required files
_MYMODEL_FILES = ["encoder.onnx", "decoder_step.onnx", "vocab.json"]

# b. Add a ModelEntry to MODEL_REGISTRY
MODEL_REGISTRY["mymodel"] = ModelEntry(
    langs=frozenset({"ru"}),
    family="mymodel",
    hf_subdir="ru_mymodel",
    description="Short description.",
    input_scripts=frozenset({Script.CYRILLIC}),
)
```

```python
# stressonnx/backends/mymodel.py  (new file)

import onnxruntime as ort
import json

from stressonnx.download import _download_files
from stressonnx.registry import _MYMODEL_FILES


class _MyModelStressor:
    def __init__(self, cache_dir=None):
        self._cache_dir = cache_dir
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        data = _download_files("ru_mymodel", _MYMODEL_FILES, self._cache_dir)
        self._enc = ort.InferenceSession(data["encoder.onnx"])
        self._dec = ort.InferenceSession(data["decoder_step.onnx"])
        with open(data["vocab.json"]) as fh:
            self._vocab = json.load(fh)
        self._loaded = True

    def __call__(self, text: str) -> str:
        self._ensure_loaded()
        # encode + greedy decode + return combining-acute text
        ...
```

```python
# stressonnx/stressor.py — add a branch in make_stressor()

if family == "mymodel":
    from stressonnx.backends.mymodel import _MyModelStressor
    return _MyModelStressor(cache_dir=cache_dir)
```

**4. Test:**

- Verify ONNX numerical parity against the original PyTorch model (as
  `kubataba` does: max |diff| between ONNX and PyTorch logits, and identical
  argmax on a held-out sentence set).
- Add `tests/test_stress_<lang>_<model>.py`.
- Run `pytest tests/ -q`.

---

## Checklist

- [ ] ONNX artefacts exported and numerically verified (Path B), or
      `vocab.gz` + `meta.json` prepared and validated (Path A)
- [ ] Files uploaded to `TigreGotico/stressonnx-models/<hf_subdir>/`
- [ ] For Path A: tag added to `SIMPLE_LANGS`, `LANG_SCRIPT`, `_OOV_RULES`
      in `stressonnx/registry.py`
- [ ] For Path B: `_<MODEL>_FILES` list + `ModelEntry` added to
      `stressonnx/registry.py`; backend class in `stressonnx/backends/`;
      branch added in `make_stressor()` (`stressonnx/stressor.py`)
- [ ] Backend class satisfies the `StressorBackend` Protocol
      (`__call__(self, text: str) -> str`)
- [ ] Tests added and passing (`pytest tests/ -q`)
- [ ] `docs/models.md` updated with the new language/model
