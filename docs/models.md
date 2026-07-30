# stressonnx: model reference

> Per-language linguistic background lives in [languages.md](languages.md).

## Choosing a model

| Situation | Recommendation |
|-----------|---------------|
| Russian general-purpose | `model="ruaccent"` (default), or `prefer="best"` |
| Russian in memory-constrained environment | `model="silero"` (about 5 MB vs about 470 MB), or `prefer="smallest"`/`prefer="fast"` |
| Ukrainian / Belarusian | `model="silero"` (default) |
| Turkic / Caucasian / minority languages | `model="simple"` (default) |
| Don't care which model, just want a capability | `prefer="best"` / `"fast"` / `"smallest"` instead of a model id |

Full accuracy numbers, not just relative quality claims, live in
[`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md), reproducible with
`python benchmarks/run.py`. Headline results:

| lang | model | accuracy | homograph accuracy |
|------|-------|----------|--------------------|
| ru | ruaccent | 0.938 | 0.820 |
| ru | silero | 0.914 | 0.381 |
| be | silero | 0.873 | not measured |
| be | simple | 0.433 | not measured |

`ruaccent` beats `silero` on both plain accuracy and, by a wide margin, on
homograph accuracy for Russian. The much larger download buys real
correctness, not just parity. For Belarusian, `silero` (neural, the default)
is substantially more accurate than `model="simple"` (vocabulary lookup).
Use `model="simple"` only where ONNX inference is unavailable. See
`benchmarks/RESULTS.md` for row counts, the annotation-noise ceiling, and
per-model dropped-row counts.

---

## Why lexical stress matters

In Russian and most languages supported here, **stress is not marked in
ordinary writing** but is phonemically contrastive. The same sequence of
letters can be pronounced, and mean, two different things depending on which
syllable carries the primary accent.

For TTS this is directly audible. For ASR and NLP downstream it affects
phoneme alignment, duration modelling, and word-boundary detection. Getting
stress wrong produces unnaturally flat or miscued speech.

The difficulty varies enormously by language:

| Language group | Stress predictability | Why it is hard |
|----------------|----------------------|---------------|
| Russian | Very low | Free (can fall on any syllable), 3 000+ homograph pairs with context-dependent stress |
| Ukrainian | Low-medium | Free stress, fewer homographs than Russian but same phonology |
| Belarusian | Low-medium | Free, akane (unstressed /o/ → [a]) changes surface form |
| Turkic (Kazakh, Tatar, Kyrgyz, and others) | High | Predominantly final-syllable stress with grammatically regular exceptions |
| Caucasian (Georgian, Armenian, Kabardino-Balkarian) | Medium | Fixed position in most forms, loanwords and compound words break the rule |
| Erzya / Moksha | Low-medium | Initial stress default, but many exceptions, vowel reduction in unstressed syllables |
| Yakut (Sah) | Medium | Long vowels attract stress, vowel harmony constrains position |

---

## Tokenization (shared across backends)

Every backend splits text on a shared boundary set before looking words up:
whitespace, ASCII punctuation, typographic punctuation
(quotation marks, ellipsis, dashes, `%№*@[]{}`), and digits. This keeps a word glued to adjacent
punctuation or a digit from falling through as one unrecognisable OOV token.
Instead it splits into the clean word plus the punctuation/digit run
(`«дом»`, a dash-joined pair like `текст-текст`, `дом5`).

Apostrophes are deliberately not boundaries. `'` is part of the `az-Latn` /
`uz-Latn` alphabets and `’` is part of the `be` alphabet, so splitting on
them would break words in those languages.

Hyphenated words split into their hyphen-joined parts for tokenization
purposes, with one exception: **Russian hyphenated enclitic particles**
(`-то`, `-нибудь`, `-либо`, `-таки`, `-ка`) never receive stress, for example
`кто́-то`, `како́й-нибудь`, `пришёл-таки`. This applies identically in both
the `ruaccent` and `silero` pipelines for `ru`.

---

## Monosyllable (single-vowel word) policy

Backends differ in whether they force a mark onto a word with exactly one
vowel:

| Backend | Single-vowel word policy |
|---------|--------------------------|
| `simple` | Always stressed. The one vowel present is marked, even under an OOV `"none"` rule (see `be`/`simple` below). |
| `silero` | Always stressed. A word with exactly one vowel gets it marked regardless of the model's own prediction. |
| `ruaccent` | The dictionary/neural accent path only fires for words with **more than one** vowel. A bare, dictionary-absent monosyllable is often left **unmarked**. Single-vowel entries hardcoded in the accent dictionary (for example `о` → `+о`) are still marked. |

---

## Idempotency (re-stressing already-marked text)

Calling a backend on text that already carries the combining acute
(U+0301) is safe, but the backends handle it differently:

| Backend | Behavior on pre-marked input |
|---------|-------------------------------|
| `simple` | Skips: if U+0301 is already present in a token, that token comes back unchanged. |
| `silero` | Skips: a word already containing the stress token short-circuits before any prediction. |
| `ruaccent` | Strips and re-derives from scratch. Marks are not detected as "already done", yo-homograph and omograph resolution always re-run. |

If you need to guarantee a no-op on already-stressed text, prefer `simple`
or `silero` for that language, or check for U+0301 yourself before calling.

---

## `"ruaccent"`: homograph-aware Russian

**Default for `ru`.**

### Why Russian needs a dedicated model

Russian has **about 3,000 homograph pairs** in everyday vocabulary: identical
written forms with different stress and sometimes different meaning.
Well-known examples:

| Written | Pronunciation A | Pronunciation B |
|---------|----------------|----------------|
| замок | за́мок (castle) | замо́к (lock) |
| мука | му́ка (torment) | мука́ (flour) |
| белок | бе́лок (squirrel, gen.) | бело́к (protein / egg white) |
| коса | ко́са (scythe / braid) | коса́ (sandbank / spit) |
| пили | пи́ли (they drank) | пили́ (they sawed) |
| стрелки | стре́лки (arrows, pl.) | стрелки́ (shooters, gen.) |

Predicting which reading is correct needs **sentence context**, not just the
word itself. A word-level model always assigns the same stress to `замок`
regardless of whether it means a castle or a lock.

### Architecture

Four ONNX models run in a fixed pipeline. All are derived from
[RUAccent](https://github.com/Den4ikAI/ruaccent) (Den4ikAI, Apache-2.0).
The original PyTorch checkpoints were converted to ONNX for torch-free
runtime use.

```
Input sentence
      │
      ▼
nn_stress_usage (BERT token classifier, ~111 MB)
  - per-token: STRESS / NO_STRESS
  - skips words known to be unstressed (particles, prepositions, and so on)
      │
      ├──[е-containing tokens]──→ nn_yo_homograph (DistilBERT, ~14 MB)
      │                            - decides whether е should become ё
      │
      ├──[omograph-dictionary tokens]──→ nn_omograph (RoBERTa NLI turbo3.1, ~359 MB)
      │                                    - picks the correct stressed variant
      │                                      from the homograph dictionary
      │
      └──[remaining STRESS tokens]──→ accent dictionary lookup
                                        → nn_accent (RoFormer char-level, ~0.8 MB)
                                          for words not in the dictionary
```

`RuAccentStressor` normalizes input aggressively before processing. It
strips any character outside its allow-list (Cyrillic, ASCII letters/digits,
whitespace, and a fixed set of punctuation), so symbols like `…` are
**dropped**, not preserved, and internal whitespace runs are not guaranteed
to be preserved byte-for-byte. This backend is **not layout-preserving**. Do
not rely on it to round-trip arbitrary punctuation or whitespace.

### Yo (ё) handling

`nn_yo_homograph` decides, per occurrence of `е`, whether to rewrite it as
`ё` (both are the same letter for stress purposes. `ё` is always stressed).
This resolves plain е→ё **restoration**, recovering the true pronunciation
of a word conventionally written with `е`, but it does **not** resolve
**yo-homographs**: pairs where the surface form is ambiguous between two
different words depending on ё (`все` "everyone" vs `всё` "everything").
Both spellings stay as `все` untouched, because the upstream model needed to
disambiguate that specific pair has not been exported to ONNX. Every other
known е→ё correction still applies.

### Runtime details

- Tokenizers: loaded via the `tokenizers` library (HuggingFace fast tokenizer
  JSON format). No `transformers` or `torch` required at runtime.
- `onnxruntime` sessions are created lazily on the first call.
- Total download: about 470 MB. Model files sit in the standard Hugging
  Face cache (respecting `HF_HOME` and `HF_HUB_OFFLINE`) unless an explicit
  `cache_dir=` is passed, in which case the layout is
  `cache_dir/ru_ruaccent/<file>`.

### Limitations

- **Large download.** About 470 MB is unsuitable for edge devices.
- **Russian only.** The pipeline is monolingual by design.
- **Context window.** Works best on short-to-medium sentences. Very long
  inputs (more than 512 subword tokens) are truncated internally by the
  BERT tokenizer.
- **Proper nouns and neologisms** not in the accent dictionary fall through
  to `nn_accent`, which is character-level and may produce suboptimal
  results for foreign-origin words.
- **Poetry / non-standard stress** (for expressive effect) is not handled.
- **Monosyllables** are frequently left unmarked (see the monosyllable table
  above). The dictionary/model path only runs for words with 2+ vowels.
- **Not layout-preserving.** Aggressive input normalization drops symbols
  like `…` and does not guarantee whitespace is round-tripped exactly.
- **Idempotency:** re-running on already-marked text strips and re-derives
  rather than skipping (see the idempotency table above).

---

## `"silero"`: neural ONNX

**Default for `uk`, `be`. Available for `ru`.**

### Architecture

A fasttext-style n-gram embedding-bag plus MLP classification heads.
Derived from [silero_stress](https://github.com/snakers4/silero-models) (MIT).

```
Input word
      │
      ▼
Tokenise → character n-grams (1 to 3 chars)
      │
      ▼
N-gram → embedding row lookup → mean-pool → pooled vector  (NumPy)
      │
      ▼
accentor.onnx  (MLP stress_clf head, + yo_clf head for ru)
      │
      ▼
argmax over character positions → stress index (+ yo index for ru)
      │
      ▼
Insert U+0301 at that position (+ е→ё restoration for ru)
```

Exceptions and skip-lists are consulted before the ONNX call:
`exceptions.txt.gz` (word → explicit stress index, plus an explicit yo index
for `ru`) and `skip_stress_words.txt.gz` / `skip_yo_words.txt.gz` (words that
should not receive a stress mark or a ё rewrite respectively, for example
monosyllables and clitics).

The original PyTorch model uses `nn.EmbeddingBag` with mode `"mean"`.
Since `onnxruntime` does not support `EmbeddingBag` directly, the export
splits the computation:
- The embedding matrix is saved as `embedding.npy` and the n-gram pool is
  computed in NumPy.
- Only the MLP head(s) (linear + activation + linear) are exported as
  `accentor.onnx`.

### е→ё handling for Russian

The Russian silero pipeline **does restore е→ё**. It runs a second MLP head
(`yo_logits`) alongside the stress head and rewrites `е` to `ё` when the
predicted yo-position coincides with the predicted stress position (`ё` is
always stressed in Russian orthography, so it rewrites only where that
constraint holds). What it does **not** do is resolve **yo-homographs**:
words that are ambiguous purely because of the е/ё distinction, such as
`все` ("everyone") vs `всё` ("everything"). Both stay written as `все`.
Neither `silero` nor `ruaccent` (see above) disambiguates this specific
pair. The upstream homograph-resolution model that would be needed has not
been exported.

### Why Ukrainian and Belarusian use this model

Both languages have **free stress** similar to Russian. The silero neural
model covers a large vocabulary and generalises reasonably to OOV words via
character n-gram features.

Ukrainian specifics:
- Vowels і, и, е, є, а, о, у, ю, я are all potentially stressed.
- No е/ё distinction to disambiguate.
- Fewer systematic homograph pairs than Russian.

Belarusian specifics:
- **Akane**: unstressed /o/ merges phonetically with /a/. Stress is needed
  to pronounce words correctly.
- Stress is not marked in standard orthography.
- Fewer neural training resources than Russian, so the silero neural model
  is the best available quality without a full BERT pipeline. See the
  benchmark table above: silero (0.873 accuracy) is dramatically better
  here than the `simple` vocabulary fallback (0.433).

### Why silero is also available for Russian

The silero Russian model exists and is accurate for unambiguous words. It is
a valid alternative when:
- The about-470 MB ruaccent download is impractical.
- The application does not need homograph disambiguation, for example
  spelling pronunciation or read-aloud of unambiguous text.
- Fast cold-start is required (silero ru downloads about 5 MB).

It assigns the same stress to both readings of `замок`. For TTS of
mixed-context text this is audible. See `benchmarks/RESULTS.md`: on
Russian, `ruaccent` scores 0.820 homograph accuracy against `silero`'s 0.381.

### Limitations

- **No homograph disambiguation** for Russian. One-best prediction per word.
- **No yo-homograph disambiguation.** е→ё restoration runs, but
  `все`/`всё`-style ambiguity is left as-is (see above).
- **Word-level context only.** The n-gram features are character-level. The
  model has no access to surrounding words.
- **Vocabulary coverage** is finite. Neologisms and foreign words rely on
  n-gram generalisation, which degrades for non-Cyrillic stems.
- **The Ukrainian and Belarusian models are verified to match** the
  original silero outputs exactly (see `export/verify_e2e.py`). The Russian
  model is verified numerically at export (max |diff| < 1e-3).

---

## `"simple"`: vocabulary + rules

**Default for the 26 rule/vocabulary languages** (Turkic, Caucasian, Uralic, and the Wiktionary/dictionary-backed Slavic and Baltic set).

### Architecture

A word-level dictionary with a per-language OOV positional fallback.
No ONNX inference, no neural computation at runtime.

```
Input sentence
      │
      ▼
Tokenise: split on whitespace + punctuation, preserve hyphens
      │
per token:
      ├──[in vocab]──→ stressed vowel ordinal from dictionary → insert U+0301
      │
      └──[OOV]──→ if exactly 1 vowel, always stress it. Otherwise apply the
                    per-language positional rule:
                    "last"   → rightmost vowel
                    "first"  → leftmost vowel
                    "kat"    → ≤3 vowels → first, else penultimate
                    "none"   → unstressed (no mark inserted)
```

Each entry in `vocab.gz` is `word<space>vowel_ordinal`: "stress the Nth
vowel of the word", counted over the per-script vowel superset in
`stressonnx/_common.py` (`SCRIPT_VOWELS`) rather than a raw character index,
so the vocabulary survives casing and digraph differences that would shift
a character offset. `meta.json` records `"vocab_format":
"vowel_ordinal_v2"`. See `export/ADDING_A_LANGUAGE.md` for the full format
and `export/convert_vocabs_to_ordinals.py` for the converter from the
character-index encoding.

**Wiktionary / dictionary languages.** Several languages are
vocabulary-first additions with no upstream neural model: Bulgarian (`bg`),
Macedonian (`mk`), Slovene (`sl`), Latvian (`lv`), and the dictionary path of
Russian and Ukrainian (`model="simple"` on `ru` / `uk`). Their vocabularies
come from stress-marked English Wiktionary headwords (via kaikki.org, CC
BY-SA), except the Russian dictionary path, built from the RUAccent
pronunciation dictionary (Apache-2.0). Free-stress languages (bg, sl, and
the `ru`/`uk` dictionary paths) use the `none` rule: dictionary lookup only,
never a positional guess. `mk` follows the fixed antepenultimate rule
(Friedman 2001) with an exceptions-only vocabulary, and `lv` uses fixed
initial stress (Nau 1998). Rebuild with `export/build_wiktionary_vocab.py`.

### Per-language OOV rules

Every rule is a named function in `stressonnx/backends/simple.py`
(`OOV_RULES`) whose docstring carries its linguistic source, and every rule
is scored against the language's own curated vocabulary in
[benchmarks/RESULTS.md](../benchmarks/RESULTS.md) (reproduce with
`python benchmarks/oov_rules_eval.py`). Highlights:

| Language(s) | Rule | Source | Accuracy |
|---|---|---|---|
| kk ky tt ba az uz kjh sah udm xal | final vowel | Turkic/Permic final-stress default (Kirchner 1998, Poppe 1964, Krueger 1962, Winkler 2001) | 0.82 to 1.00 (sah is an open item, see scoreboard) |
| cv | last **full** vowel: reduced ӑ/ӗ never stressed, all-reduced words stress the first syllable | Clark 1998, Krueger 1961, Dobrovolsky 1999 (ICPhS) | 0.94 |
| ka | antepenultimate vowel, initial for shorter words | Akhvlediani 1949, Gudava 1969, Aronson 1990 (Georgian stress is weak and contested, Borise 2020 argues fixed initial) | 1.00 |
| hy | last non-schwa vowel (ը never stressed) | Chakmakjian 2024 (Speech Prosody) | 0.78 |
| tg | final vowel, except unstressed word-final izafet -и (stressed final /i/ is written ӣ) | Perry 2005, *A Tajik Persian Reference Grammar* | 0.74 |
| myv mdf | first vowel, Moksha retracts off an initial high vowel to a following а/я syllable | Hamari & Ajanki 2022, *Oxford Guide to the Uralic Languages* §23.2.3 | 0.54 to 0.76 (see per-language rows) |
| kbd | final vowel, penult when the word ends in the schwa letter э | Jaimoukha, consistent with Colarusso 1992 (low confidence) | 0.40 |
| be (`simple`) | no mark: Belarusian stress is lexical, not positional | East Slavic accentology literature | vocabulary only |

Single-vowel OOV words are always stressed on their sole vowel regardless of
rule (upstream `SimpleAccentor` semantics), including for the `be`
dictionary path.

### Limitations

- **Dictionary coverage** determines quality. Words absent from the
  vocabulary fall back to a positional heuristic, which is accurate for
  most agglutinative languages but unreliable for irregular forms, foreign
  loanwords, and proper nouns.
- **No context-sensitivity.** Each word is stressed independently. No
  cross-word influence.
- **Case-insensitive lookup** is used for vocabulary matching. Capitalised
  words (sentence-initial, proper nouns) are lowercased before lookup.
- For **Georgian** and **Armenian**, the OOV heuristic is noticeably weaker
  than for the Turkic languages, so dictionary coverage matters more.
- The vocabulary files are derived from
  [silero_stress](https://github.com/snakers4/silero-models) (MIT) and
  reflect the training data used there. Low-frequency or dialectal forms may
  be missing.
- The original 20 `simple` vocabularies (the `be` dictionary path included)
  are exports of the upstream `silero_stress` `SimpleAccentor` data. There
  is currently no independent gold-standard accuracy measurement for these
  languages (see the "Languages without independent gold" note in
  `benchmarks/RESULTS.md`). Correctness is locked to the upstream reference
  via `tests/test_export_parity.py`, not benchmarked against external data.

---

## Per-language linguistic motivation (`simple` model)

**Azerbaijani (`az-Cyrl`, `az-Latn`)**
An agglutinative Turkic language with predominantly **last-syllable
stress**. Stress shifts predictably with suffixation: each added suffix
carries the potential stress position one syllable further right. OOV rule
`"last"` is correct for the vast majority of native forms. Loanwords are
the main source of error.

**Uzbek (`uz-Cyrl`, `uz-Latn`)**
The same Turkic last-syllable default as Azerbaijani. Uzbek has additional
complexity from vowel harmony collapsing in the southern dialects, but
stress position is not affected. The last-syllable rule holds strongly.

**Bashkir (`ba`)**
Turkic, closely related to Tatar. Last-syllable stress default with
grammar-conditioned exceptions for certain suffixes (negative verb forms,
interrogative particles). The dictionary covers these exceptions. OOV falls
back to the last vowel.

**Chuvash (`cv`)**
Turkic (Oghuric branch), distinctive from mainstream Turkic. Stress
**migrates**: it falls on the last non-reduced syllable. The last-vowel OOV
rule approximates this for standard forms. Morphological reduction is not
modelled.

**Tatar (`tt`)**
Kipchak Turkic. Last-syllable stress default. Some enclitics and particles
are **unstressed**. The dictionary captures the most frequent forms.

**Kyrgyz (`ky`)**
Kipchak Turkic, close to Kazakh. Last-syllable stress. Vowel harmony is
preserved in standard orthography. The OOV last-vowel rule is reliable.

**Kazakh (`kk`)**
Kipchak Turkic. Last-syllable stress default. Russian loanwords retain
their original stress (these are in the dictionary). Stress does not
change with case or conjugation suffixes, so the OOV rule works well.

**Khakas (`kjh`)**
Siberian Turkic, Sayan branch, with agglutinative suffixing. Last-syllable
stress is the default. The OOV last-vowel rule is correct for native
vocabulary.

**Yakut/Sakha (`sah`)**
Siberian Turkic. Stress interacts with vowel length: **long vowels attract
stress** and diphthongs are always stressed. Standard orthography marks
long vowels, so the dictionary covers most cases. The OOV last-vowel rule
is a rough approximation.

**Tajik (`tg`)**
An Iranian language, not Turkic. Stress in Tajik is **penultimate** in most
native words, and last-syllable in loanwords and with certain suffixes. The
last-vowel OOV rule is a simplification but covers loanword-heavy usage.

**Kabardino-Balkarian (`kbd`)**
Northwest Caucasian (Kabardian) plus Turkic (Balkarian): two different
languages sharing an orthographic standard. Kabardian has **first-syllable**
or **root-syllable** default stress. Balkarian follows the Kipchak
last-syllable pattern. The shared dictionary handles the most common forms.
OOV last-vowel is a compromise.

**Armenian (`hy`)**
Indo-European, isolated branch. Eastern Armenian has **penultimate stress**
as the default rule. Western Armenian is last-syllable. The library follows
Eastern Armenian (the more widely standardised form). The OOV last-vowel
rule approximates this poorly for multi-syllable native words, so dictionary
coverage is the main quality lever.

**Georgian (`ka`)**
South Caucasian (Kartvelian), unrelated to Indo-European. Stress in Georgian
is **initial** for words with 3 or fewer vowels and **penultimate** for
longer words. The `"kat"` OOV rule implements this heuristic. Georgian
orthography is fully phonemic and stress is historically recessive, so
dictionary coverage is high for standard vocabulary.

**Erzya (`myv`) and Moksha (`mdf`)**
Mordvinic (Uralic) languages of the Volga region. Both have **initial
stress** as the dominant pattern, with exceptions for certain grammatical
suffixes. The first-vowel OOV rule is reliable for native vocabulary.

**Udmurt (`udm`)**
Permic (Uralic). Stress is consistently on the **last syllable**. The OOV
last-vowel rule is highly accurate for this language.

**Kalmyk (`xal`)**
Mongolic language (Oirat branch). Stress is on the **last syllable** in
most words. Vowel harmony and vowel reduction interact with stress
assignment. The OOV last-vowel rule holds for standard forms.

**Belarusian, `simple` path (`be`, `model="simple"`)**
The same language as the default `silero` path but using vocabulary-only
lookup without neural inference, a lower-accuracy fallback (0.433 vs.
`silero`'s 0.873, see the benchmark table above) meant for offline
environments where ONNX inference is unavailable, or as an explicit
fallback target. Its OOV rule is `"none"`, but that rule only applies to
**multi-vowel** OOV words: a **single-vowel** OOV word is always stressed
on that vowel (the "always stress monosyllables" rule takes precedence
over the per-language OOV rule, matching upstream `SimpleAccentor`
behavior). A multi-vowel word absent from the vocabulary is left with no
mark at all.

---

## Architecture comparison

| | ruaccent | silero | simple |
|---|---|---|---|
| **Languages** | ru | uk, be, ru | 26 languages |
| **Context** | Sentence (BERT) | Word (n-gram) | Word (dict) |
| **Homograph resolution** | Yes (BERT + RoBERTa) | No | No |
| **е→ё restoration (ru)** | Yes | Yes | N/A |
| **Yo-homograph (все/всё) disambiguation** | No | No | N/A |
| **ONNX inference** | 4 models | 1 model | None |
| **Download size** | ~470 MB | ~5-10 MB | ~1-5 MB |
| **Cold-start latency** | High (4 sessions) | Low | Near-zero |
| **Per-sentence latency** | ~50-200 ms | ~2-10 ms | <1 ms |
| **Runtime deps** | onnxruntime, tokenizers | onnxruntime, numpy | numpy only |
| **OOV handling** | nn_accent (char-level) | n-gram generalisation | positional rule |
| **Sentence-level input** | Required | Works word-by-word | Works word-by-word |
| **Max input length** | ~512 subword tokens | Unlimited (per-word) | Unlimited |
| **Idempotent on pre-marked text** | No (re-derives) | Yes (skips) | Yes (skips) |
| **License** | Apache-2.0 | MIT | MIT |

`prefer="best"` picks `ruaccent` where available. `prefer="fast"` and
`prefer="smallest"` pick `silero` then `simple` for languages that have
them.

An additional research-only Russian model, a character-level seq2seq
Transformer (`kubataba`), is exported by `export/export_kubataba.py` but is
not wired into the runtime registry.

See [`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) for measured
accuracy numbers instead of qualitative claims.

---
[← Languages](languages.md) · [Home](../README.md) · [Architecture →](architecture.md)
