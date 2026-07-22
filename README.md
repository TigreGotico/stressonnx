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
| `ruaccent` | `ru` (default) | Homograph-aware 4-model ONNX pipeline (derived from [RUAccent](https://github.com/Den4ikAI/ruaccent), Apache-2.0) | 0.908 word accuracy, **0.742 on homographs** |
| `silero` | `ukr`, `bel` (defaults), `ru` | Neural ONNX pipeline exported from [silero_stress](https://github.com/snakers4/silero-stress) (MIT); the `ru` variant also restores е→ё | ru 0.886 / ukr 0.767 / bel 0.859 |
| `kubataba` | `ru` | Char-level seq2seq Transformer ([kubataba](https://huggingface.co/kubataba), MIT); sentence-in, sentence-out | 0.884 (slow: ~60 ms/row) |
| `simple` | 26 languages¹ | Curated vocabulary + per-language positional rule; no neural inference | parity-locked to upstream / sourced rules, see scoreboard |

¹ `aze_cyr aze_lat uzb_cyr uzb_lat bak bel_simple bul chv erz hye kat kaz
kbd kir kjh lav mdf mkd ru_simple sah slv tat tgk udm ukr_simple xal` —
Azerbaijani (both scripts), Uzbek (both scripts), Bashkir, Belarusian
(rule-path alias), Bulgarian, Chuvash, Erzya, Armenian, Georgian, Kazakh,
Kabardian, Kyrgyz, Khakas, Latvian, Moksha, Macedonian, Russian
(dictionary-path alias), Yakut, Slovene, Tatar, Tajik, Udmurt, Ukrainian
(dictionary-path alias), Kalmyk.  Every language — including ru/ukr/bel —
has a torch-free, ONNX-free rule/vocabulary path, so `fallback=True` always
bottoms out in a model that needs nothing but a small vocabulary file.

Numbers come from the committed, reproducible
[benchmark scoreboard](benchmarks/RESULTS.md) (annotated UD-treebank gold;
read its noise-ceiling note before quoting absolutes).  Defaults per
language: `ru → ruaccent`, `ukr/bel → silero`, everything else → `simple`.

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
from stressonnx import stress, Stressor, to_plus_notation

# One-shot function (caches model instances internally)
stress("Привіт світ", "ukr")                  # 'Приві́т сві́т'
stress("Сәлем Қазақстан", "kaz")              # 'Сәле́м Қазақста́н'

# Pick a specific model
stress("красивый город", "ru", model="silero")    # 'краси́вый го́род'
stress("красивый город", "ru", model="kubataba")  # 'краси́вый го́род'

# Reusable object (same API, explicit lifecycle)
s = Stressor(lang="ru")
s("замок стоит на горе")                      # 'за́мок сто́ит на горе́'
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
  ruaccent ≈ 500 MB, silero/kubataba ≈ tens of MB, `simple` languages ≈ 1 MB.
- **Warm-up ahead of serving:** call `stress("тест", lang)` once at startup so
  no download ever happens mid-synthesis.
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
hop; the default (`False`) raises immediately.

### Contracts worth knowing

- `simple`/`silero` **skip** words already carrying U+0301; `ruaccent`/
  `kubataba` **strip and re-derive** (wrong input marks get corrected).
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
silero_stress (snakers4, MIT), kubataba (MIT).
