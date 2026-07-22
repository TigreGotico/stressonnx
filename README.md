# stressonnx

Multi-language **word-stress placement** ("accentuation") for text, built for
TTS front-ends.  Pure `onnxruntime + numpy` at runtime — **no torch, ever**.

```python
from stressonnx import stress

stress("старинный замок стоит на горе", "ru")   # 'стари́нный за́мок сто́ит на горе́'
stress("дверной замок надёжен", "ru")           # 'дверно́й замо́к надёжен'
```

Same spelling, different word: за́мок is a castle, замо́к is a lock.  stressonnx
reads the sentence, decides which one you meant, and marks the stressed vowel.

---

## Why word stress?

**Word stress** (lexical stress) is which syllable of a word is pronounced
prominently: English *REcord* (noun) vs *reCORD* (verb).  A text-to-speech
system must know it before it can pick the right sounds — in Russian,
unstressed vowels *reduce* (о sounds like а), so getting the stress wrong
changes every vowel in the word, not just the melody.

Whether **you** need this library depends on how your language writes stress:

| Orthography type | Examples | What you need |
|---|---|---|
| Stress is free/mobile and **not written** | Russian, Ukrainian, Belarusian, most languages here | **A stress model — this library.**  Nothing in the spelling of `замок` tells you which syllable to stress; only context does. |
| Stress is **predictable by rule** | Kazakh/Turkic (final syllable), Armenian, Georgian | A positional rule covers most words — that is the `simple` backend: a curated exception vocabulary plus a per-language default rule. |
| Stress is **already written** | Spanish, Greek, Portuguese | You do **not** need stressonnx — the orthography (accent rules) already encodes it.  But you may still need *sense* disambiguation, see below. |

**The Portuguese case (why "stress is written" isn't the end of the story):**
Portuguese spelling pins down the stressed syllable, yet pairs like *sede*
(thirst /ˈsedɨ/ vs headquarters /ˈsɛdɨ/) share spelling *and* stress position
while differing in vowel **quality** — resolvable only from meaning.  Our
sibling library [bifonia](https://github.com/TigreGotico/bifonia) solves that
with the same "disambiguate before G2P" idea used here: its
`add_extra_diacritics(text)` rewrites the homograph with an explicit
open/closed-vowel diacritic (*séde*/*sêde*) so any downstream phonemizer gets
it right.  Rule of thumb: **unwritten stress → stressonnx; written stress but
sense-dependent pronunciation → a bifonia-style diacritic restorer.**

---

## Supported languages

| Model id | Languages | What it is | Measured quality |
|----------|-----------|------------|------------------|
| `ruaccent` | `ru` (default) | Homograph-aware 4-model ONNX pipeline (derived from [RUAccent](https://github.com/Den4ikAI/ruaccent), Apache-2.0) | 0.938 word accuracy, **0.820 on homographs** |
| `silero` | `uk`, `be` (defaults), `ru` | Neural ONNX pipeline exported from [silero_stress](https://github.com/snakers4/silero-stress) (MIT); the `ru` variant also restores е→ё | ru 0.914 / uk 0.785 / be 0.873 |
| `simple` | 26 languages¹ | Curated vocabulary + per-language positional rule; no neural inference | parity-locked to upstream / sourced rules, see scoreboard |

¹ `ru uk be bg mk sl lv hy ka kk ky tt ba cv sah kjh tg udm mdf myv kbd xal
az-Latn az-Cyrl uz-Latn uz-Cyrl` — Russian, Ukrainian, Belarusian (all
dictionary-path), Bulgarian, Macedonian, Slovene, Latvian, Armenian,
Georgian, Kazakh, Kyrgyz, Tatar, Bashkir, Chuvash, Yakut, Khakas, Tajik,
Udmurt, Moksha, Erzya, Kabardian, Kalmyk, Azerbaijani (both scripts), Uzbek
(both scripts).  Every language — including ru/uk/be — has a torch-free,
ONNX-free rule/vocabulary path (`model="simple"`), so `fallback=True` always
bottoms out in a model that needs nothing but a small vocabulary file.

Numbers come from the committed, reproducible
[benchmark scoreboard](benchmarks/RESULTS.md) (annotated UD-treebank gold;
read its noise-ceiling note before quoting absolutes).  Defaults per
language: `ru → ruaccent`, `uk/be → silero`, everything else → `simple`.

**Legacy tags.**  Historical tags accepted before the switch to BCP-47 —
`ukr`, `bel`, `kaz`, `kir`, `tat`, `bak`, `chv`, `tgk`, `erz`, `hye`, `kat`,
`bul`, `mkd`, `slv`, `lav`, `aze_lat`, `aze_cyr`, `uzb_lat`, `uzb_cyr`, and
the `bel_simple`/`ru_simple`/`ukr_simple` (language, model) pairs — remain
accepted everywhere a `lang` is expected; they resolve to the canonical tag
above (the `*_simple` forms additionally force `model="simple"`).

Models are hosted on
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
and downloaded automatically on first use (see *Offline & failure behavior*).

---

## Install

```bash
pip install stressonnx
```

Runtime dependencies: `onnxruntime`, `numpy`, `huggingface_hub`, and
`tokenizers` (used only by the `ru` ruaccent pipeline).  Optional:

- `razdel` — better Russian sentence splitting inside the ruaccent pipeline
  (`pip install razdel`); without it the whole input is processed as one span.
- `pip install stressonnx[export]` — torch + silero_stress, **only** for
  re-exporting models from a checkout (never needed at runtime).

---

## Usage

```python
from stressonnx import stress, analyze, Stressor, to_plus_notation

# One-shot function (caches model instances internally)
stress("Привіт світ", "uk")                   # 'Приві́т сві́т'
stress("Сәлем Қазақстан", "kk")               # 'Сәле́м Қазақста́н'

# Pick a specific model, or a capability instead of a model id
stress("красивый город", "ru", model="silero")    # 'краси́вый го́род'
stress("красивый город", "ru", model="simple")    # 'краси́вый го́род'
stress("красивый город", "ru", prefer="fast")     # 'краси́вый го́род'  (silero)

# Reusable object (same API, explicit lifecycle)
s = Stressor(lang="ru")
s("замок стоит на горе")                      # 'за́мок сто́ит на горе́'
```

### Structured results: `analyze()`

For TTS pipelines that need to reason about individual words rather than a
marked string, `analyze()` returns a `StressResult`: per-word spans with
offsets into the **original, untouched input**.

```python
result = analyze("замок стоит на горе", "ru")
result.text     # 'за́мок сто́ит на горе́'  (same as stress())

for w in result.words:
    print(w.text, w.start, w.end, w.stressed_index, w.yo_restored)
# замок 0 5 1 False
# стоит 6 11 2 False
# на 12 14 None False
# горе 15 19 2 False
```

`w.stressed_index` is the offset of the stressed vowel *within the word*
(`None` if the word carries no mark); `w.yo_restored` is `True` when the
backend rewrote е→ё inside that word.  Because offsets refer to `result.original`
exactly as passed in, callers never need to re-parse the marked string to
locate a word.

### Batches: `stress_batch()`

```python
from stressonnx import stress_batch

stress_batch(["привет", "мир"], "ru")   # ['приве́т', 'мир']
```

### Output notation

All backends emit the **combining acute accent** (U+0301) *after* the
stressed vowel — `приве́т` is `п р и в е U+0301 т`.  For TTS models trained on
the legacy `+`-before-vowel format:

```python
stress("привет", "ru", notation="plus")   # 'прив+ет'
to_plus_notation("приве́т")                # 'прив+ет'  (handles NFC-composed á too)
```

### ё restoration (Russian)

Both Russian neural backends restore ё that writers commonly type as е:
`зеленый → зелё́ный`.  Genuinely ambiguous ё-homographs (все/всё) are resolved
by `ruaccent` from context and deliberately left untouched by `silero`.

---

## Offline & failure behavior

- **First call per language downloads models** into the standard Hugging Face
  cache (`~/.cache/huggingface`, relocatable via `HF_HOME`).  Sizes: `ru`
  ruaccent ≈ 500 MB, silero ≈ tens of MB, `simple` languages ≈ 1 MB.
- **Warm-up ahead of serving:** call `warm_up(lang)` once at startup so no
  synthesis request ever blocks on a model download; it loads the same
  cached instance later `stress()` calls use.
- **Fully offline:** after a warm run, set `HF_HUB_OFFLINE=1` — cached models
  keep working, network is never touched.
- **Typed failures:**

```python
from stressonnx import stress, UnsupportedLanguageError, ModelDownloadError

try:
    out = stress(text, lang, fallback=True)   # opt-in: walk ruaccent→silero→simple
except UnsupportedLanguageError as e:         # also catchable as ValueError
    out = text                                # e.supported lists valid tags
except ModelDownloadError as e:               # names the exact missing HF path
    out = text
```

`fallback=True` degrades down the quality chain with a logged warning per
hop (it also engages on `ModelLoadError` — a corrupt cache — not just
failed downloads); the default (`False`) raises immediately.  A failed
(lang, model) pair is not retried for 30 s, so an outage never triggers a
download attempt per call.

**Supply-chain pinning:** model files are fetched from a commit-pinned
revision of the HF repo (`HF_REPO_REVISION` in `stressonnx/registry.py`),
so releases are reproducible and upstream changes never reach users
implicitly.

**Thread safety:** `stress()` and the backends use double-checked locking
for lazy loads; calling from multiple threads is supported (onnxruntime
sessions are thread-safe for inference).

### Contracts worth knowing

- `simple`/`silero` **skip** words already carrying U+0301; `ruaccent`
  **strips and re-derives** (wrong input marks get corrected).
- Monosyllables: `simple`/`silero` always stress them; `ruaccent` usually
  leaves bare single-vowel words unmarked.
- `ruaccent` normalizes its input (drops symbols like `…`, collapses runs of
  whitespace) — it is not byte-layout-preserving; the other backends are.
- Russian hyphenated clitics (`кто́-то`, `како́й-нибудь`, `-либо`, `-таки`,
  `-ка`) never receive a mark on the particle.

---

## Guarding by writing system

Feeding Cyrillic to the Georgian model (or vice versa) is a silent no-op — the
input never matches the model's alphabet.  Check first:

```python
from stressonnx import lang_to_script, MODEL_REGISTRY

if lang_to_script(lang) in MODEL_REGISTRY[model_id].input_scripts:
    text = stress(text, lang, model=model_id)
```

`Script` values (`"cyrillic"`, `"latin"`, `"armenian"`, `"georgian"`) are
plain strings shared with phoonnx's `Alphabet` enum for direct comparison.

---

## Adding a language

Two paths, both documented step-by-step for newcomers in
[`export/ADDING_A_LANGUAGE.md`](export/ADDING_A_LANGUAGE.md):

1. **You have a stressed wordlist** → ship a `simple` language: package the
   vocabulary, declare alphabet/vowels/OOV rule, upload to the HF repo,
   register the language tag.  No training, no torch.
2. **You have (or train) a neural accentor** → export it to ONNX with the
   scripts in `export/` and add a backend entry.

All 20 upstream silero_stress vocabularies are already shipped, and
`export/build_wiktionary_vocab.py` turns any language whose Wiktionary
headwords carry stress marks into a `simple` language (that is how
Bulgarian, Macedonian, Slovene, Latvian and Ukrainian were built; Russian
came from the RUAccent pronunciation dictionary).

---

## Documentation

Pick your entry point:

- **New to all of this?**  The [Why word stress?](#why-word-stress) section
  above, then [`docs/languages.md`](docs/languages.md) — a plain-language,
  per-language guide to why stress marking is needed and what we do about it.
- **Developer integrating stressonnx?**  [Usage](#usage) above, then
  [`docs/models.md`](docs/models.md) for backend contracts and
  [`docs/architecture.md`](docs/architecture.md) for the package internals.
- **Linguist checking our homework?**  [`docs/languages.md`](docs/languages.md)
  carries the typology and per-rule citations;
  [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) the measurements; every
  OOV rule's source is quoted in `stressonnx/backends/simple.py`.

- [`docs/languages.md`](docs/languages.md) — per-language guide: stress
  system, why TTS needs it, what stressonnx does, with citations.
- [`docs/models.md`](docs/models.md) — every backend in depth: pipeline
  stages, per-language rules, quality numbers, contracts.
- [`docs/architecture.md`](docs/architecture.md) — package layout, the single
  download layer, data flow.
- [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) — the scoreboard and how to
  reproduce it.
- [`examples/`](examples/) — runnable scripts, from basics to homograph demos.

## Related projects

- [phoonnx](https://github.com/TigreGotico/phoonnx) — ONNX TTS engine; calls
  stressonnx before phonemization for Russian voices.
- [bifonia](https://github.com/TigreGotico/bifonia) — European-Portuguese
  homograph disambiguation by meaning (the "written stress, unwritten vowel
  quality" counterpart to this library).
- [scriptconv](https://github.com/TigreGotico/scriptconv) — script detection
  and phoneme-notation conversion.
- [silabificador](https://github.com/TigreGotico/silabificador) — Portuguese
  syllabification and stress by rule.

## License

Apache-2.0.  Model attributions: RUAccent (Den4ikAI, Apache-2.0),
silero_stress (snakers4, MIT).
