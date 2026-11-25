# Offline use and failure behaviour

 **First call per language downloads models** into the standard Hugging Face
  cache (`~/.cache/huggingface`, relocatable via `HF_HOME`). Sizes: `ru`
  ruaccent about 500 MB, silero tens of MB, kubataba about 31 MB. The
  `simple` vocabularies are 10 MB for `ru`, 6 MB for `be` and under 1 MB
  for the other languages.
- **Warm up ahead of serving.** Call `warm_up(lang)` once at startup so no
  synthesis request ever blocks on a model download. It loads the same
  cached instance later `stress()` calls use.
- **Fully offline.** After a warm run, set `HF_HUB_OFFLINE=1`. Cached models
  keep working and the network is never touched.
- **Typed failures:**

```python
from stressonnx import stress, UnsupportedLanguageError, ModelDownloadError

try:
    out = stress(text, lang, fallback=True)   # opt-in: walk ruaccent-silero→ruaccent→silero→simple
except UnsupportedLanguageError as e:         # also catchable as ValueError
    out = text                                # e.supported lists valid tags
except ModelDownloadError as e:               # names the exact missing HF path
    out = text
```

`UnsupportedLanguageError` also covers a model that does not serve the
language, for example `stress(text, "kk", model="silero")`. Then `e.model`
names the model, and `e.supported` lists the languages that model serves.

`fallback=True` degrades down the quality chain with a logged warning per
hop (it also engages on `ModelLoadError`, a corrupt cache, not just failed
downloads). The default (`False`) raises immediately. A pipeline remembers
each failed (lang, model) pair and raises the remembered error again without
a new attempt:

- After a `ModelDownloadError`, the pair is retried after 30 s, so an outage
  never triggers a download attempt per call.
- After a `ModelLoadError`, the pair is retried after 30 s. Each consecutive
  load failure doubles the delay (60 s, 120 s …) up to 10 minutes, because a
  reload rebuilds the model. Delete the corrupt files and the next retry
  loads fresh ones. Call `warm_up(lang, model)` to load the pair at once and
  clear the failure.

A successful load clears the failure record.

**Supply-chain pinning:** model files come from a commit-pinned revision of
the HF repo (`HF_REPO_REVISION` in `stressonnx/registry.py`), so releases are
reproducible and upstream changes never reach users implicitly.

**Thread safety:** `stress()` and the backends use double-checked locking
for lazy loads. Calling from multiple threads is supported (onnxruntime
sessions are thread-safe for inference).

## Contracts worth knowing

- `simple`/`silero`/`kubataba` **skip** words already carrying U+0301.
  `ruaccent` **strips and re-derives** (wrong input marks get corrected).
- Monosyllables: `simple`/`silero` always stress them. `ruaccent` usually
  leaves bare single-vowel words unmarked; `kubataba` never marks them.
- Every backend gives back the input unchanged except for the added marks
  and е→ё restorations. `ruaccent` also drops the stress marks already in
  the input and derives them again. A word that the models cannot read
  whole comes back with no mark: `Ки́їв` gives `Київ`. A word gets the
  same marks whether it is written with ё or е. A second call gives back
  the output of the first unless a restored ё changes how the models read
  a neighbouring word.
- Russian hyphenated clitics (`кто́-то`, `како́й-нибудь`, `-либо`, `-таки`,
  `-ка`) never receive a mark on the particle.

---

[Index](index.md)
