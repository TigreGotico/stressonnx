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

## Why word stress?

**Word stress** (lexical stress) is which syllable of a word is pronounced
prominently: English *REcord* (noun) vs *reCORD* (verb). A text-to-speech
system must know it before it can pick the right sounds. In Russian,
unstressed vowels *reduce* (о sounds like а), so a wrong stress changes
every vowel in the word, not just the melody.

Whether you need this library depends on how your language writes stress:

| Orthography type | Examples | What you need |
|---|---|---|
| Stress is free/mobile and **not written** | Russian, Ukrainian, Belarusian, most languages here | **A stress model — this library.** Nothing in the spelling of `замок` tells you which syllable to stress; only context does. |
| Stress is **predictable by rule** | Kazakh/Turkic (final syllable), Armenian, Georgian | A positional rule covers most words. That is the `simple` backend: a curated exception vocabulary plus a per-language default rule. |
| Stress is **already written** | Spanish, Greek, Portuguese | You do not need stressonnx. The orthography (accent rules) already encodes it. You may still need *sense* disambiguation, see below. |

**The Portuguese case (why "stress is written" is not the end of the story):**
Portuguese spelling pins down the stressed syllable, yet pairs like *sede*
(thirst /ˈsedɨ/ vs headquarters /ˈsɛdɨ/) share spelling and stress position
while differing in vowel **quality**, resolvable only from meaning. Our
sibling library [bifonia](https://github.com/TigreGotico/bifonia) solves that
with the same "disambiguate before G2P" idea used here: its
`add_extra_diacritics(text)` rewrites the homograph with an explicit
open/closed-vowel diacritic (*séde*/*sêde*) so any downstream phonemizer gets
it right. Rule of thumb: unwritten stress needs stressonnx. Written stress
with sense-dependent pronunciation needs a bifonia-style diacritic restorer.

---

## Supported languages

| Model id | Languages | What it is | Measured quality |
|----------|-----------|------------|------------------|
| `ruaccent` | `ru` | Homograph-aware 4-model ONNX pipeline (derived from [RUAccent](https://github.com/Den4ikAI/ruaccent), MIT) | 0.947 word accuracy, **0.833 on homographs** |
| `ruaccent-big-poetry` | `ru` | The `ruaccent` pipeline with the RUAccent `big_poetry` homograph model in place of `turbo3.1` | 0.942 word accuracy, 0.753 on homographs; below `ruaccent`, see [docs/models.md](docs/models.md#the-big_poetry-omograph-model) |
| `ruaccent-silero` | `ru` (default) | The `ruaccent` pipeline, then the `silero` mark on every word of two or more vowels that `ruaccent` leaves unmarked (`и́ли`, `э́то`, `бы́ло`, `что́-то`) | 0.951 word accuracy, 0.833 on homographs. 0.990 on the Wiktionary sentences, where `ruaccent` has 0.964 and `silero` 0.984. See [docs/models.md](docs/models.md#ruaccent-silero-the-ruaccent-pipeline-with-silero-marks-on-its-unmarked-words) |
| `silero` | `uk`, `be` (defaults), `ru` | Neural ONNX pipeline exported from [silero_stress](https://github.com/snakers4/silero-stress) (MIT); the `ru` variant also restores е→ё and resolves homographs; the `uk` variant looks words up in a dictionary layer from [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) (MIT) first, and the `be` variant in one from [GrammarDB](https://github.com/Belarus/GrammarDB) (CC-BY-SA-4.0) | ru 0.944 (0.753 on homographs) / uk 0.843 / be 0.908² |
| `stressrnn` | `ru` (selectable) | Character BiLSTM with a Zaliznyak exception dictionary, ported from [StressRNN](https://github.com/dbklim/StressRNN) (Apache-2.0; a fork of RusStress). No sentence context, no е→ё; 0.6 MB | ru 0.892, scored on the model's own training corpus |
| `uk-accentor` | `uk` (selectable) | Seq2seq Transformer ported from [ukrainian-accentor-transformer](https://github.com/Theodotus1243/ukrainian-accentor-transformer) (MIT); rewrites the sentence with marks, the port keeps every input letter; 45 MB | uk 0.795 (silero 0.843) |
| `simple` | 26 languages¹ | Curated vocabulary + per-language positional rule; no neural inference | parity-locked to upstream / sourced rules, see scoreboard |
| `kubataba` | `ru` (opt-in) | Character-level seq2seq Transformer exported from [Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor) (MIT); sentence-level, no homograph resolution, no ё restoration | 0.888 word accuracy (0.816 on homographs) where `silero` gets 0.944 and `ruaccent` 0.947; see docs/models.md |

¹ `ru uk be bg mk sl lv hy ka kk ky tt ba cv sah kjh tg udm mdf myv kbd xal
az-Latn az-Cyrl uz-Latn uz-Cyrl`: Russian, Ukrainian, Belarusian (all
dictionary-path), Bulgarian, Macedonian, Slovene, Latvian, Armenian,
Georgian, Kazakh, Kyrgyz, Tatar, Bashkir, Chuvash, Yakut, Khakas, Tajik,
Udmurt, Moksha, Erzya, Kabardian, Kalmyk, Azerbaijani (both scripts), Uzbek
(both scripts). Every language, including ru/uk/be, has a torch-free,
ONNX-free rule/vocabulary path (`model="simple"`), so `fallback=True` always
bottoms out in a model that needs nothing but a small vocabulary file.

² Word accuracy on the russtress UD-treebank gold set, scored by
`benchmarks/run.py`: the scorer compares letters in NFC and counts a target
as a hit when the model marks exactly the gold vowel. uk uses the bundle
with the dictionary layer. be is the neural model alone; the Belarusian
dictionary layer is measured in [docs/models.md](docs/models.md#belarusian-dictionary-layer).

Numbers come from the committed, reproducible
[benchmark scoreboard](benchmarks/RESULTS.md), which also scores the
Wiktionary example sentences, the Belarusian hand-marked sets and the
silver sets; read its label-tier note before quoting absolutes. Defaults per
language: `ru → ruaccent`, `uk/be → silero`, everything else → `simple`.
`kubataba` is reached only with `model="kubataba"`: on russtress it is
below both Russian defaults (McNemar p = 7e-91 against `silero`, p = 2e-104
against `ruaccent`) and took about 0.4 s per row in the scoreboard run (two
onnxruntime threads on a shared host), so no default, `prefer=` strategy or
fallback picks it. The scoreboard does not run it on the two large sets
(Wiktionary, ruslan-stressed) and says so in their tables. Its port is
byte-identical to the upstream torch model on 1011 inputs.

Models are hosted on
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
and download automatically on first use (see *Offline & failure behavior*).

---

## Install

```bash
pip install stressonnx
```

Runtime dependencies: `onnxruntime`, `numpy`, `huggingface_hub`, and
`tokenizers` (used only by the `ru` ruaccent pipeline). Optional:

- `razdel` for better Russian sentence splitting inside the ruaccent pipeline
  (`pip install razdel`). Without it the whole input is processed as one span.
- `pip install stressonnx[export]` adds torch, onnx, silero_stress and
  ukrainian-word-stress, only for re-exporting models from a checkout.
  Never needed at runtime.

---

## Usage

```python
from stressonnx import StressPipeline, analyze, stress, to_plus_notation

# One-shot function (caches model instances internally)
stress("Привіт світ", "uk")                   # 'Приві́т сві́т'
stress("Сәлем Қазақстан", "kk")               # 'Сәле́м Қазақста́н'

# Pick a specific model, or a capability instead of a model id
stress("красивый город", "ru", model="silero")    # 'краси́вый го́род'
stress("старинный замок", "ru", model="simple")   # 'старинный за́мок'  (dictionary only: a word outside it stays unmarked)
stress("красивый город", "ru", prefer="fast")     # 'краси́вый го́род'  (silero)

# An isolated engine with its own cache and failure policy
pipeline = StressPipeline()
pipeline.stress("замок стоит на горе", "ru")                      # 'за́мок сто́ит на горе́'
```

### Structured results: `analyze()`

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

### Batches: `stress_batch()`

```python
from stressonnx import stress_batch

stress_batch(["привет", "мир"], "ru")   # ['приве́т', 'мир']
```

### Output notation

All backends emit the **combining acute accent** (U+0301) *after* the
stressed vowel: `приве́т` is `п р и в е U+0301 т`. For TTS models trained on
the `+`-before-vowel format:

```python
stress("привет", "ru", notation="plus")   # 'прив+ет'
to_plus_notation("приве́т")                # 'прив+ет'  (handles NFC-composed á too)
```

### ё restoration (Russian)

Both Russian neural backends restore ё that writers commonly type as е:
`зеленый → зелё́ный`. Genuinely ambiguous ё-homographs (все/всё) are resolved
from context by `ruaccent` and by the homograph resolver of `silero`.

---

## Offline & failure behavior

- **First call per language downloads models** into the standard Hugging Face
  cache (`~/.cache/huggingface`, relocatable via `HF_HOME`). Sizes: `ru`
  ruaccent about 500 MB, silero tens of MB, kubataba about 31 MB, `simple`
  languages about 1 MB.
- **Warm up ahead of serving.** Call `warm_up(lang)` once at startup so no
  synthesis request ever blocks on a model download. It loads the same
  cached instance later `stress()` calls use.
- **Fully offline.** After a warm run, set `HF_HUB_OFFLINE=1`. Cached models
  keep working and the network is never touched.
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

### Contracts worth knowing

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

## Adding a language

Two paths, both documented step-by-step for newcomers in
[`export/ADDING_A_LANGUAGE.md`](export/ADDING_A_LANGUAGE.md):

1. **You have a stressed wordlist.** Ship a `simple` language: package the
   vocabulary, declare alphabet/vowels/OOV rule, upload to the HF repo,
   register the language tag. No training, no torch.
2. **You have (or train) a neural accentor.** Export it to ONNX with the
   scripts in `export/` and add a backend entry.

All 20 upstream silero_stress vocabularies are already shipped, and
`export/build_wiktionary_vocab.py` turns any language whose Wiktionary
headwords carry stress marks into a `simple` language. That is how
Bulgarian, Macedonian, Slovene, Latvian and Ukrainian were built. Russian
came from the RUAccent pronunciation dictionary.

---

## Documentation

Pick your entry point:

- **New to all of this?** Read [Why word stress?](#why-word-stress) above,
  then [`docs/languages.md`](docs/languages.md), a plain-language,
  per-language guide to why stress marking is needed and what we do about it.
- **Developer integrating stressonnx?** Read [Usage](#usage) above, then
  [`docs/models.md`](docs/models.md) for backend contracts and
  [`docs/architecture.md`](docs/architecture.md) for the package internals.
- **Linguist checking our homework?** [`docs/languages.md`](docs/languages.md)
  carries the typology and per-rule citations.
  [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md) carries the measurements.
  Every OOV rule's source is quoted in `stressonnx/backends/simple.py`.

- [`docs/languages.md`](docs/languages.md): per-language guide to the stress
  system, why TTS needs it, what stressonnx does, with citations.
- [`docs/models.md`](docs/models.md): every backend in depth, with pipeline
  stages, per-language rules, quality numbers, and contracts.
- [`docs/architecture.md`](docs/architecture.md): package layout, the single
  download layer, and data flow.
- [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md): the scoreboard and how to
  reproduce it.
- [`examples/`](examples/): runnable scripts, from basics to homograph demos.

## Related projects

- [phoonnx](https://github.com/TigreGotico/phoonnx): an ONNX TTS engine that
  calls stressonnx before phonemization for Russian voices.
- [bifonia](https://github.com/TigreGotico/bifonia): European Portuguese
  homograph disambiguation by meaning (the "written stress, unwritten vowel
  quality" counterpart to this library).
- [scriptconv](https://github.com/TigreGotico/scriptconv): script detection
  and phoneme-notation conversion.
- [silabificador](https://github.com/TigreGotico/silabificador): Portuguese
  syllabification and stress by rule.

## License

Apache-2.0. Model attributions: RUAccent (Den4ikAI, MIT),
[silero-stress](https://github.com/snakers4/silero-stress) (Copyright (c)
2020-present Silero Team, MIT). The `silero` model files are derived from
silero-stress 1.5. Each model directory (`ru`, `ukr`, `bel`) carries the
upstream licence text as `LICENSE.silero-stress`. The `ru` `meta.json` also
names the source package, version, licence and URL.
The Ukrainian dictionary layer of the `silero` model comes from
ukrainian-word-stress (lang-uk, MIT), whose word forms derive from
"Dictionaries of Ukraine" by ULIF. The Belarusian dictionary layer
(`bel/dictionary.txt.gz`) is derived from
[GrammarDB](https://github.com/Belarus/GrammarDB) RELEASE-202601 (Aleś
Bułojčyk and contributors) and is licensed CC-BY-SA-4.0; the bundle carries
the notice and licence text as `LICENSE.grammardb`, and the other `bel`
files stay MIT. The `kubataba` model is derived from
[Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor)
(kubataba, MIT). StressRNN (Vladislav Klim, Apache-2.0; BiLSTM
from RusStress by Ponomareva and Milintsevich, exception dictionary from
A. A. Zaliznyak's Grammatical Dictionary via odict.ru). The `uk-accentor` model
is derived from ukrainian-accentor-transformer (Copyright (c) 2023 Bohdan
Mykhailenko, MIT); `ukr_accentor/` carries the licence text as
`LICENSE.ukrainian-accentor-transformer` and names the source commit and
Hugging Face revision in `meta.json`.
