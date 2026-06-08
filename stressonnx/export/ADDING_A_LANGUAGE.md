# Adding a new language

Each language ships as a small ONNX file (the MLP classifier heads) plus a set of
numpy/gzip data files that are loaded at runtime without torch.

## Steps

### 1. Export from silero_stress (or compatible source)

```
pip install silero_stress torch onnxruntime numpy
python stressonnx/export/export_ru.py   # adapt for the new lang tag
```

The export script:
- Loads the silero accentor for the target language.
- Rebuilds the scripted MLP heads as a plain `nn.Sequential`.
- Exports them to `accentor.onnx` via `torch.onnx.export`.
- Dumps `embedding.npy` (float32), `ngram_dict.txt.gz`, `exceptions.txt.gz`,
  `skip_stress_words.txt.gz`, `skip_yo_words.txt.gz`, and `meta.json`.

### 2. Verify

```
python stressonnx/export/verify_e2e.py --lang <tag>
```

Should report `EXACT MATCH: N/N = 100.0%` against the silero reference.

### 3. Upload to HuggingFace

Upload all files under a `<lang>/` folder in the
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
HF model repo:

```
huggingface-cli upload TigreGotico/stressonnx-models \
    /path/to/artifacts/<lang>/ <lang>/
```

### 4. Test the runtime

```python
from stressonnx import stress
print(stress("Привет мир", "ru"))   # existing lang
print(stress("Привіт світ", "uk"))  # new lang
```

No torch needed at runtime — only `onnxruntime`, `numpy`, `huggingface_hub`.
