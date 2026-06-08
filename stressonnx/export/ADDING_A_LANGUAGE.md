# Adding a new language

Languages ship in one of two families: **main_accentor** (neural ONNX, like `ru`,
`ukr`, `bel`) or **simple_accentor** (vocabulary + rules, like `kaz`, `kat`, etc.).

---

## Main-accentor family (`ru`, `ukr`, `bel`)

Each language ships as a small ONNX file (MLP classifier heads) plus a set of
numpy/gzip data files loaded at runtime without torch.

### 1. Export from silero_stress

```bash
pip install silero_stress torch onnxruntime numpy
python stressonnx/export/export_main_accentors.py --lang ukr --out_dir /tmp/ukr
```

The script:
- Loads the silero accentor for the target language.
- Rebuilds the scripted stress_clf (and yo_clf for `ru`) as a plain `nn.Sequential`.
- Exports to `accentor.onnx` via `torch.onnx.export`.
- Dumps `embedding.npy` (float32), `ngram_dict.txt.gz`, `exceptions.txt.gz`,
  `skip_stress_words.txt.gz`, `skip_yo_words.txt.gz`, and `meta.json`.
- `meta.json` contains `family`, `has_yo`, `alpha`, `vowels`, and embedding stats.
- Verifies classifiers match torch output (max abs diff < 1e-3).

### 2. Upload to HuggingFace

```bash
hf upload TigreGotico/stressonnx-models /tmp/ukr/ ukr/
```

### 3. Register in accentor.py

Add the tag to `MAIN_LANGS` in `stressonnx/accentor.py`.

### 4. Test

```python
from stressonnx import stress
print(stress("Привіт світ", "ukr"))  # Прив+іт св+іт
```

---

## Simple-accentor family

These languages use a word-level vocabulary (word → stress char index) with a
per-language OOV positional rule.  No ONNX; no torch at any stage.

### 1. Export from silero_stress

```bash
pip install silero_stress
python stressonnx/export/export_simple_accentors.py --lang kaz --out_dir /tmp/kaz
# or all at once:
python stressonnx/export/export_simple_accentors.py --all --out_base /tmp/out
```

The script copies `vocab-<lang>.gz` from the silero package and writes `meta.json`
with `family`, `lang`, `alpha`, `vowels`, and `oov_rule`.

### 2. Upload to HuggingFace

```bash
hf upload TigreGotico/stressonnx-models /tmp/kaz/ kaz/
```

### 3. Register in accentor.py

Add the tag to `SIMPLE_LANGS` and `_OOV_RULES` in `stressonnx/accentor.py`.

### 4. Test

```python
from stressonnx import stress
print(stress("Сәлем Қазақстан", "kaz"))
```

---

## Verify end-to-end match against silero reference

```bash
python stressonnx/export/verify_e2e.py --lang ukr
```

Should report `EXACT MATCH: N/N = 100.0%`.
