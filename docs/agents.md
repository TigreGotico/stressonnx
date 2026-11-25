# Agent hints

## Layout

- `stressonnx/__init__.py`: the public API. A name that is not in `__all__` is private.
- `stressonnx/pipeline.py`: `StressPipeline`, model choice, fallback and the failure cache.
- `stressonnx/registry.py`: `MODEL_REGISTRY`, `DEFAULT_MODEL`, scripts.
- `stressonnx/backends/`: one module for each model family.
- `stressonnx/languages/*.json`: one file for each language. Language facts go here, not into Python.
- `benchmarks/run.py`: writes the scoreboard in `benchmarks/RESULTS.md`.
- `export/`: scripts that convert upstream models. They need the `export` extra and are not used at runtime.

## Tests

```bash
uv venv && uv pip install --prerelease=allow -e . pytest
.venv/bin/python -m pytest -q tests/<the files that your change touches>
```

GitHub CI runs the full suite. Some tests download models from the Hugging Face Hub on the first run.

## Pitfalls

- Consumers (udarnik, scriptconv) use only the names in `__all__`. Before you change one, search the org for consumers.
- `SCRIPT_VOWELS` values are `frozenset`. Do not change them at runtime.
- A change to a default model needs a win on a GOLD set in `benchmarks/RESULTS.md`. SILVER rows do not count.
- `tokenizers` 1.0.0rc4 and later encode only through `Tokenizer.compile()`. The CI matrix tests three `tokenizers` ranges.

---

[Index](index.md)
