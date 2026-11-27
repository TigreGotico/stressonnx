# stressonnx

Multi-language **word-stress placement** ("accentuation") for text, built for
TTS front-ends. Pure `onnxruntime + numpy` at runtime. No torch, ever.

```python
from stressonnx import stress

stress("старинный замок стоит на горе", "ru")   # 'стари́нный за́мок сто́ит на горе́'
stress("дверной замок надёжен", "ru")           # 'дверно́й замо́к надё́жен'
```

Same spelling, different word: за́мок is a castle, замо́к is a lock.
stressonnx reads the sentence, decides which one you meant, and marks the
stressed vowel.

---

## Install

```bash
uv pip install stressonnx
```

Optional: `razdel` gives better Russian sentence splitting in the `ruaccent` pipeline. Without it, the pipeline reads the full input as one span.

## Documentation

See [`docs/index.md`](docs/index.md): supported languages and measured
accuracy, usage, offline behaviour, models, architecture and how to add a
language. The scoreboard is in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md).

## Related projects

- [phoonnx](https://github.com/TigreGotico/phoonnx): ONNX TTS that calls stressonnx for Russian voices.
- [bifonia](https://github.com/TigreGotico/bifonia): European Portuguese homographs by meaning.
- [scriptconv](https://github.com/TigreGotico/scriptconv): script detection and phoneme-notation conversion.
- [silabificador](https://github.com/TigreGotico/silabificador): Portuguese syllables and stress by rule.

## License

Apache-2.0. Model and data licences are in [`docs/credits.md`](docs/credits.md).
