# Usage

```python
from stressonnx import StressPipeline, analyze, stress, to_plus_notation

# One-shot function (caches model instances internally)
stress("Привіт світ", "uk")                   # 'Приві́т сві́т'
stress("Сәлем Қазақстан", "kk")               # 'Сәле́м Қазақста́н'

# Pick a specific model, or a capability instead of a model id
stress("красивый город", "ru", model="silero")    # 'краси́вый го́род'
stress("старинный замок", "ru", model="simple")   # 'стари́нный замок'  (dictionary only: a homograph such as за́мок/замо́к stays unmarked)
stress("красивый город", "ru", prefer="fast")     # 'краси́вый го́род'  (silero)

# An isolated engine with its own cache and failure policy
pipeline = StressPipeline()
pipeline.stress("замок стоит на горе", "ru")                      # 'за́мок сто́ит на горе́'
```

## Structured results: `analyze()`

For TTS pipelines that need to reason about individual words rather than a
marked string, `analyze()` returns a `StressResult`: per-word spans with
offsets into the original, untouched input.

```python
result = analyze("замок стоит на горе", "ru")
result.text     # 'за́мок сто́ит на горе́'  (same as stress())

for w in result.words:
    print(w.text, w.start, w.end, w.stressed_index, w.yo_restored)
# замок 0 5 1 False
# стоит 6 11 2 False
# на 12 14 None False
# горе 15 19 3 False
```

`w.stressed_index` is the offset of the stressed vowel *within the word*
(`None` if the word carries no mark). A hyphenated compound can carry more
than one mark. `w.stressed_indices` lists every marked offset in order, and
`w.stressed_index` is the first of them:

```python
w, = analyze("Қазақстан-Ресей", "kk").words
w.stressed_indices   # (7, 13)  'Қазақста́н-Ресе́й'
w.stressed_index     # 7
```

`w.yo_restored` is `True` when the backend rewrote е→ё inside that word. Offsets refer to `result.original`
exactly as passed in, so callers never need to re-parse the marked string to
locate a word.

## Batches: `stress_batch()`

```python
from stressonnx import stress_batch

stress_batch(["привет", "мир"], "ru")   # ['приве́т', 'мир']
```

## Output notation

All backends emit the **combining acute accent** (U+0301) *after* the
stressed vowel: `приве́т` is `п р и в е U+0301 т`. For TTS models trained on
the `+`-before-vowel format:

```python
stress("привет", "ru", notation="plus")   # 'прив+ет'
to_plus_notation("приве́т")                # 'прив+ет'  (handles NFC-composed á too)
```

## ё restoration (Russian)

Both Russian neural backends restore ё that writers commonly type as е:
`зеленый → зелё́ный`. Genuinely ambiguous ё-homographs (все/всё) are resolved
from context by `ruaccent` and by the homograph resolver of `silero`.

## Guarding by writing system

Feeding Cyrillic to the Georgian model (or vice versa) is a silent no-op: the
input never matches the model's alphabet. Check first:

```python
from stressonnx import lang_to_script, MODEL_REGISTRY

if lang_to_script(lang) in MODEL_REGISTRY[model_id].input_scripts:
    text = stress(text, lang, model=model_id)
```

`Script` values (`"cyrillic"`, `"latin"`, `"armenian"`, `"georgian"`) are
plain strings shared with phoonnx's `Alphabet` enum for direct comparison.

---

[Index](index.md)
