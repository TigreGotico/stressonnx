# Adding a language or model to stressonnx

Languages ship in one of three families: **main_accentor** (silero neural ONNX),
**simple_accentor** (vocabulary + rules), or a **custom family** (any ONNX
model that does not fit the silero format, e.g. a seq2seq Transformer).

---

## Family 1: main_accentor — silero neural ONNX (`ukr`, `bel`, `ru`)

Each language ships as one ONNX file (MLP classifier head) plus a set of
numpy/gzip data files.

### 1. Export from silero_stress

```bash
pip install "stressonnx[export]"
python stressonnx/export/export_main_accentors.py --lang ukr --out_dir /tmp/ukr
```

The script:
- Loads the silero accentor for the target language.
- Rebuilds the scripted `stress_clf` as a plain `nn.Sequential`.
- Exports to `accentor.onnx` via `torch.onnx.export`.
- Dumps `embedding.npy`, `ngram_dict.txt.gz`, `exceptions.txt.gz`,
  `skip_stress_words.txt.gz`, `skip_yo_words.txt.gz`, and `meta.json`.
- `meta.json` contains `family`, `has_yo`, `alpha`, `vowels`, and embedding
  stats.
- Verifies classifiers match torch output (max abs diff < 1e-3).

### 2. Upload to HuggingFace

```bash
huggingface-cli upload TigreGotico/stressonnx-models /tmp/ukr/ <lang>/
```

### 3. Register in accentor.py

Add the tag to `MAIN_LANGS` in `stressonnx/accentor.py`.  The `DEFAULT_MODEL`
and `MODEL_REGISTRY["silero"].langs` are derived automatically from
`MAIN_LANGS`.

### 4. Test

```python
from stressonnx import stress
print(stress("Привіт світ", "ukr"))
```

### 5. Verify end-to-end match

```bash
python stressonnx/export/verify_e2e.py --lang ukr
# Should report: EXACT MATCH: N/N = 100.0%
```

---

## Family 2: simple_accentor — vocabulary + rules (20+ languages)

These languages use a word-level vocabulary (word → stress char index) with
a per-language OOV positional rule.  No ONNX; no torch at any stage.

### 1. Export from silero_stress

```bash
pip install silero_stress
python stressonnx/export/export_simple_accentors.py --lang kaz --out_dir /tmp/kaz
# or all at once:
python stressonnx/export/export_simple_accentors.py --all --out_base /tmp/out
```

The script copies `vocab-<lang>.gz` from the silero package and writes
`meta.json` with `family`, `lang`, `alpha`, `vowels`, and `oov_rule`.

### 2. Upload to HuggingFace

```bash
huggingface-cli upload TigreGotico/stressonnx-models /tmp/kaz/ <lang>/
```

### 3. Register in accentor.py

- Add the tag to `SIMPLE_LANGS`.
- Add the OOV rule to `_OOV_RULES` dict.

### 4. Test

```python
from stressonnx import stress
print(stress("Сәлем Қазақстан", "kaz"))
```

---

## Family 3: custom ONNX (e.g. seq2seq Transformer, kubataba)

Use this path when the model architecture does not fit the silero pipeline.
The kubataba Russian seq2seq is the reference implementation.

### 1. Export to ONNX

For a **seq2seq model with dynamic sequence length**, PyTorch's standard
exporters (TorchScript and dynamo) often fail when the model computes
data-dependent shapes internally (e.g. `nn.MultiheadAttention` creating a
causal mask of shape `(t, t)` for data-dependent `t`).

The kubataba solution: split the model into **encoder** and **one-step
decoder** and reimplement the decoder forward pass manually using
`F.scaled_dot_product_attention(is_causal=True)` to avoid the problematic
`_detect_is_causal_mask` call inside `nn.Transformer`.

See `export_kubataba.py` for the full implementation.

General recipe:

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
`decoder_step.onnx` + `decoder_step.onnx.data`.  Merge them into a
single self-contained file before uploading:

```python
import onnx
from onnx.external_data_helper import load_external_data_for_model

model = onnx.load("decoder_step.onnx", load_external_data=False)
load_external_data_for_model(model, "/tmp/export_dir")
onnx.save_model(model, "decoder_step_merged.onnx", save_as_external_data=False)
```

### 2. Upload to HuggingFace

```bash
huggingface-cli upload TigreGotico/stressonnx-models /tmp/ru_mymodel/ ru_mymodel/
```

The subdirectory name becomes `hf_subdir` in the `ModelEntry`.

### 3. Register in accentor.py

```python
# 1. List the required files
_MYMODEL_FILES = ["encoder.onnx", "decoder_step.onnx", "vocab.json"]

# 2. Add a ModelEntry to MODEL_REGISTRY
MODEL_REGISTRY["mymodel"] = ModelEntry(
    langs=frozenset({"ru"}),
    family="mymodel",
    hf_subdir="ru_mymodel",
    description="Short description.",
)

# 3. Implement the backend class
class _MyModelStressor:
    def __init__(self, cache_dir=None):
        self._cache = cache_dir
        self._enc = self._dec = self._vocab = None

    def _ensure_loaded(self):
        if self._enc is not None:
            return
        cache = _resolve_cache(self._cache)
        paths = _download_files("ru_mymodel", cache, _MYMODEL_FILES)
        import onnxruntime as ort, json
        self._enc = ort.InferenceSession(paths["encoder.onnx"])
        self._dec = ort.InferenceSession(paths["decoder_step.onnx"])
        with open(paths["vocab.json"]) as f:
            self._vocab = json.load(f)

    def __call__(self, text: str) -> str:
        self._ensure_loaded()
        # encode + greedy decode + return combining-acute text
        ...

# 4. Add a branch in make_stressor()
elif entry.family == "mymodel":
    return _MyModelStressor(cache_dir=cache_dir)
```

### 4. Test

- Verify ONNX numerical parity with the original PyTorch model.
- Add `tests/test_stress_<lang>_<model>.py`.
- Run `pytest tests/ -q`.

---

## Checklist

- [ ] ONNX artefacts exported and numerically verified
- [ ] Files uploaded to `TigreGotico/stressonnx-models/<subdir>/`
- [ ] `_<MODEL>_FILES` list added to `accentor.py`
- [ ] `ModelEntry` added to `MODEL_REGISTRY`
- [ ] Backend class satisfies `StressorBackend` Protocol
- [ ] Branch added in `make_stressor()`
- [ ] Language tag added to the appropriate `*_LANGS` set
- [ ] Tests added and passing
- [ ] `docs/models.md` updated
- [ ] `README.md` model table updated