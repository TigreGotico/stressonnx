# stressonnx — model reference

> Per-language linguistic background lives in [languages.md](languages.md).

## Choosing a model

| Situation | Recommendation |
|-----------|---------------|
| Russian general-purpose | `model="ruaccent"` (default) |
| Russian in memory-constrained environment | `model="silero"` (~5 MB vs ~470 MB) |
| Russian research / seq2seq comparison | `model="kubataba"` |
| Ukrainian / Belarusian | `model="silero"` (default) |
| Turkic / Caucasian / minority languages | `model="simple"` (default) |

Full accuracy numbers (not just relative quality claims) live in
[`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md), reproducible with
`python benchmarks/run.py`.  Headline results:

| lang | model | accuracy | homograph accuracy |
|------|-------|----------|--------------------|
| ru | ruaccent | 0.908 | 0.742 |
| ru | silero | 0.886 | 0.377 |
| bel | silero | 0.859 | — |
| bel_simple | simple | 0.417 | — |

`ruaccent` beats `silero` on both plain accuracy and (by a wide margin)
homograph accuracy for Russian — the much larger download buys real
correctness, not just parity.  For Belarusian, `silero` (neural) is
substantially more accurate than `bel_simple` (vocabulary lookup); use
`bel_simple` only where ONNX inference is unavailable.  See
`benchmarks/RESULTS.md` for row counts, the annotation-noise ceiling, and
per-model dropped-row counts.

---

## Why lexical stress matters

In Russian and most languages supported here, **stress is not marked in
ordinary writing** but is phonemically contrastive: the same sequence of
letters can be pronounced — and mean — two different things depending on which
syllable carries the primary accent.

For TTS this is directly audible.  For ASR and NLP downstream it affects
phoneme alignment, duration modelling, and word-boundary detection.  Getting
stress wrong produces unnaturally flat or miscued speech.

The difficulty varies enormously by language:

| Language group | Stress predictability | Why it is hard |
|----------------|----------------------|---------------|
| Russian | Very low | Free (can fall on any syllable); 3 000+ homograph pairs with context-dependent stress |
| Ukrainian | Low–medium | Free stress; fewer homographs than Russian but same phonology |
| Belarusian | Low–medium | Free; akane (unstressed /o/ → [a]) changes surface form |
| Turkic (Kazakh, Tatar, Kyrgyz, …) | High | Predominantly final-syllable stress with grammatically regular exceptions |
| Caucasian (Georgian, Armenian, Kabardino-Balkarian) | Medium | Fixed position in most forms; loanwords and compound words break the rule |
| Erzya / Moksha | Low–medium | Initial stress default, but many exceptions; vowel reduction in unstressed syllables |
| Yakut (Sah) | Medium | Long vowels attract stress; vowel harmony constrains position |

---

## Tokenization (shared across backends)

Every backend splits text on a shared boundary set before looking words up:
whitespace, ASCII punctuation, typographic punctuation
(`«» „“ ”…—–%№*@[]{}`), and digits.  This keeps a word glued to adjacent
punctuation or a digit from falling through as one unrecognisable OOV token —
it is split into the clean word plus the punctuation/digit run instead
(`«дом»`, `текст—текст`, `дом5`).

Apostrophes are deliberately **not** boundaries: `'` is part of the
`aze_lat` / `uzb_lat` alphabets and `’` is part of the `bel` alphabet, so
splitting on them would break words in those languages.

Hyphenated words are split into their hyphen-joined parts for tokenization
purposes, with one exception: **Russian hyphenated enclitic particles**
(`-то`, `-нибудь`, `-либо`, `-таки`, `-ка`) never receive stress — e.g.
`кто́-то`, `како́й-нибудь`, `пришёл-таки`.  This applies identically in both
the `ruaccent` and `silero` pipelines for `ru`.

---

## Monosyllable (single-vowel word) policy

Backends differ in whether they force a mark onto a word with exactly one
vowel:

| Backend | Single-vowel word policy |
|---------|--------------------------|
| `simple` | Always stressed — the one vowel present is marked, even under an OOV `"none"` rule (see `bel_simple` below). |
| `silero` | Always stressed — a word with exactly one vowel gets it marked regardless of the model's own prediction. |
| `ruaccent` | The dictionary/neural accent path only fires for words with **more than one** vowel; a bare, dictionary-absent monosyllable is often left **unmarked**.  Single-vowel entries hardcoded in the accent dictionary (e.g. `о` → `+о`) are still marked. |
| `kubataba` | Model-dependent — the seq2seq decoder has no explicit monosyllable rule; behavior follows whatever the training data taught it. |

---

## Idempotency (re-stressing already-marked text)

Calling a backend on text that already carries the combining acute
(U+0301) is safe, but the four backends handle it differently:

| Backend | Behavior on pre-marked input |
|---------|-------------------------------|
| `simple` | Skips: if U+0301 is already present in a token, that token is returned unchanged. |
| `silero` | Skips: a word already containing the stress token short-circuits before any prediction. |
| `ruaccent` | Strips and re-derives from scratch — marks are not detected as "already done"; yo-homograph and omograph resolution always re-run. |
| `kubataba` | Strips and re-derives: the whole sentence is re-encoded and re-decoded character-by-character, so any existing marks are just more input characters to the seq2seq model. |

If you need to guarantee a no-op on already-stressed text, prefer `simple`
or `silero` for that language, or check for U+0301 yourself before calling.

---

## `"ruaccent"` — homograph-aware Russian

**Default for `ru`.**

### Why Russian needs a dedicated model

Russian has **~3 000 homograph pairs** in everyday vocabulary — identical
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

Predicting which reading is correct requires **sentence context**, not just
the word itself.  A word-level model will always assign the same stress to
`замок` regardless of whether it means a castle or a lock.

### Architecture

Four ONNX models run in a fixed pipeline.  All are derived from
[RUAccent](https://github.com/Den4ikAI/ruaccent) (Den4ikAI, Apache-2.0).
The original PyTorch checkpoints were converted to ONNX for torch-free
runtime use.

```
Input sentence
      │
      ▼
nn_stress_usage (BERT token classifier, ~111 MB)
  — per-token: STRESS / NO_STRESS
  — skips words known to be unstressed (particles, prepositions, …)
      │
      ├──[е-containing tokens]──→ nn_yo_homograph (DistilBERT, ~14 MB)
      │                            — decides whether е should become ё
      │
      ├──[omograph-dictionary tokens]──→ nn_omograph (RoBERTa NLI turbo3.1, ~359 MB)
      │                                    — picks the correct stressed variant
      │                                      from the homograph dictionary
      │
      └──[remaining STRESS tokens]──→ accent dictionary lookup
                                        → nn_accent (RoFormer char-level, ~0.8 MB)
                                          for words not in the dictionary
```

`RuAccentStressor` normalizes input aggressively before processing — it
strips any character outside its allow-list (Cyrillic, ASCII letters/digits,
whitespace, and a fixed set of punctuation), so symbols like `…` are
**dropped**, not preserved, and internal whitespace runs are not guaranteed
to be preserved byte-for-byte.  This backend is **not layout-preserving**;
do not rely on it to round-trip arbitrary punctuation or whitespace.

### Yo (ё) handling

`nn_yo_homograph` decides, per occurrence of `е`, whether it should be
rewritten as `ё` (both are the same letter for stress purposes — `ё` is
always stressed).  This resolves plain е→ё **restoration** (recovering the
true pronunciation of a word that is conventionally written with `е`), but
it does **not** resolve **yo-homographs** — pairs where the surface form is
ambiguous between two different words depending on ё (`все` "everyone" vs
`всё` "everything").  Both spellings are left as `все` untouched, because
the upstream model needed to disambiguate that specific pair has not been
exported to ONNX; every other known е→ё correction is still applied.

### Runtime details

- Tokenizers: loaded via the `tokenizers` library (HuggingFace fast tokenizer
  JSON format).  No `transformers` or `torch` required at runtime.
- `onnxruntime` sessions are created lazily on the first call.
- Total download: ~470 MB.  Model files are stored in the standard Hugging
  Face cache (respecting `HF_HOME` and `HF_HUB_OFFLINE`) unless an explicit
  `cache_dir=` is passed, in which case the layout is
  `cache_dir/ru_ruaccent/<file>`.

### Limitations

- **Large download.** ~470 MB is unsuitable for edge devices.
- **Russian only.**  The pipeline is monolingual by design.
- **Context window.** Works best on short-to-medium sentences.  Very long
  inputs (>512 subword tokens) are truncated internally by the BERT
  tokenizer.
- **Proper nouns and neologisms** not in the accent dictionary fall through
  to `nn_accent`, which is character-level and may produce suboptimal results
  for foreign-origin words.
- **Poetry / non-standard stress** (for expressive effect) is not handled.
- **Monosyllables** are frequently left unmarked (see the monosyllable table
  above) — the dictionary/model path only runs for words with 2+ vowels.
- **Not layout-preserving.**  Aggressive input normalization drops symbols
  like `…` and does not guarantee whitespace is round-tripped exactly.
- **Idempotency:** re-running on already-marked text strips and re-derives
  rather than skipping (see the idempotency table above).

---

## `"silero"` — neural ONNX

**Default for `ukr`, `bel`.  Available for `ru`.**

### Architecture

Fasttext-style n-gram embedding-bag + MLP classification heads.
Derived from [silero_stress](https://github.com/snakers4/silero-models) (MIT).

```
Input word
      │
      ▼
Tokenise → character n-grams (1–3 chars)
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
should not receive a stress mark or a ё rewrite respectively — e.g.
monosyllables, clitics).

The original PyTorch model uses `nn.EmbeddingBag` with mode `"mean"`.
Since `onnxruntime` does not support `EmbeddingBag` directly, the export
splits the computation:
- The embedding matrix is saved as `embedding.npy` and the n-gram pool is
  computed in NumPy.
- Only the MLP head(s) (linear + activation + linear) are exported as
  `accentor.onnx`.

### е→ё handling for Russian

The Russian silero pipeline **does restore е→ё**: it runs a second MLP head
(`yo_logits`) alongside the stress head and rewrites `е` to `ё` when the
predicted yo-position coincides with the predicted stress position (`ё` is
always stressed in Russian orthography, so it only rewrites where that
constraint holds).  What it does **not** do is resolve **yo-homographs** —
words that are ambiguous purely because of the е/ё distinction, such as
`все` ("everyone") vs `всё` ("everything").  Both stay written as `все`;
neither `silero` nor `ruaccent` (see above) disambiguates this specific
pair — the upstream homograph-resolution model that would be needed has not
been exported.

### Why Ukrainian and Belarusian use this model

Both languages have **free stress** similar to Russian.  The silero neural
model covers a large vocabulary and provides reasonable generalisation to OOV
words via character n-gram features.

Ukrainian specifics:
- Vowels і, и, е, є, а, о, у, ю, я are all potentially stressed.
- No е/ё distinction to disambiguate.
- Fewer systematic homograph pairs than Russian.

Belarusian specifics:
- **Akane**: unstressed /o/ merges phonetically with /a/; stress is needed to
  pronounce words correctly.
- Stress is not marked in standard orthography.
- Fewer neural training resources than Russian → silero neural model is the
  best available quality without a full BERT pipeline; see the benchmark
  table above — silero (0.859 accuracy) is dramatically better here than the
  `bel_simple` vocabulary fallback (0.417).

### Why silero is also available for Russian

The silero Russian model exists and is accurate for unambiguous words.  It is
a valid alternative when:
- The ~470 MB ruaccent download is impractical.
- The application does not need homograph disambiguation (e.g., spelling
  pronunciation or read-aloud of unambiguous text).
- Fast cold-start is required (silero ru downloads ~5 MB).

It will assign the same stress to both readings of `замок`; for TTS of
mixed-context text this is audible.  See `benchmarks/RESULTS.md`: on
Russian, `ruaccent` scores 0.742 homograph accuracy against `silero`'s 0.377.

### Limitations

- **No homograph disambiguation** for Russian.  One-best prediction per word.
- **No yo-homograph disambiguation.**  е→ё restoration runs, but
  `все`/`всё`-style ambiguity is left as-is (see above).
- **Word-level context only.**  The n-gram features are character-level; the
  model has no access to surrounding words.
- **Vocabulary coverage** is finite.  Neologisms and foreign words rely on
  n-gram generalisation, which degrades for non-Cyrillic stems.
- The **Ukrainian and Belarusian models are verified to match the original**
  silero outputs exactly (see `export/verify_e2e.py`).  The Russian model
  is verified numerically at export (max |diff| < 1e-3).

---

## `"kubataba"` — seq2seq Transformer

**Alternative for `ru`.**

### Architecture

Character-level encoder-decoder Transformer trained on Russian literary
texts with explicit stress marks.  Derived from
[kubataba/Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor)
(MIT).

```
Input sentence (character sequence, max 256 chars)
      │
      ▼
CharacterEmbedding  (char → d_model=256)
      │
      ▼
encoder.onnx  (4-layer Transformer encoder)
      → memory: float32[1, 256, 256]
      │
      ▼
Greedy decode loop (Python):
  tgt = [BOS]
  while not EOS and len < max_len:
    decoder_step.onnx(memory, tgt) → logits[1, vocab_size]
    next_token = argmax(logits[-1])
    tgt.append(next_token)
      │
      ▼
Decode token sequence → output text with U+0301 stress marks
```

The decoder implements all four decoder layers manually using
`F.scaled_dot_product_attention` to avoid ONNX export incompatibilities with
PyTorch's `nn.MultiheadAttention` (which internally creates data-dependent
causal masks incompatible with the TorchScript and dynamo exporters).

### Why a seq2seq model for Russian

A seq2seq model can, in principle, capture **cross-word context** without
requiring a separate disambiguation pipeline: the encoder reads the whole
sentence, and the decoder produces each stressed character attending to the
full context.  This makes it architecturally interesting as a single-model
alternative to the ruaccent multi-stage pipeline.

### Limitations

- **Sentence-level input required.**  The model is trained on sentence pairs,
  not isolated words.  Single words fed in isolation produce degraded output.
- **Slower inference** than silero or simple: autoregressive decode is
  sequential; O(output_len) ONNX session calls per sentence.
- **No explicit homograph dictionary.**  Disambiguation depends entirely on
  what the model learned during training.  Performance on rare homograph pairs
  is unpredictable.
- **Maximum input length: 256 characters.**  Sentences longer than this are
  truncated at the encoder.
- **Research quality.**  This model is provided as an alternative, not as the
  recommended production choice.  Use `"ruaccent"` for production Russian TTS.
- **Numerical precision:** float32 ONNX; verified max |diff| = 2.16×10⁻⁴
  vs the original PyTorch model (argmax identical on all tested sentences).
- **Monosyllable and idempotency behavior is model-dependent** — there is no
  explicit rule for either case; see the tables above.

---

## `"simple"` — vocabulary + rules

**Default for the 26 rule/vocabulary languages** (Turkic, Caucasian, Uralic, and the Wiktionary/dictionary-backed Slavic and Baltic set).

### Architecture

Word-level dictionary with a per-language OOV positional fallback.
No ONNX inference; no neural computation at runtime.

```
Input sentence
      │
      ▼
Tokenise: split on whitespace + punctuation, preserve hyphens
      │
per token:
      ├──[in vocab]──→ stress_char_idx from dictionary → insert U+0301
      │
      └──[OOV]──→ if exactly 1 vowel, always stress it; otherwise apply the
                    per-language positional rule:
                    "last"   → rightmost vowel
                    "first"  → leftmost vowel
                    "kat"    → ≤3 vowels → first, else penultimate
                    "none"   → unstressed (no mark inserted)
```


**Wiktionary / dictionary languages.**  Six languages are vocabulary-first
additions with no upstream neural model: Bulgarian (`bul`), Macedonian
(`mkd`), Slovene (`slv`), Latvian (`lav`), and the `ru_simple` /
`ukr_simple` aliases.  Their vocabularies come from stress-marked English
Wiktionary headwords (via kaikki.org, CC BY-SA) except `ru_simple`, built
from the RUAccent pronunciation dictionary (Apache-2.0).  Free-stress
languages (bul, slv, ru_simple, ukr_simple) use the `none` rule — dictionary
lookup only, never a positional guess; `mkd` follows the fixed
antepenultimate rule (Friedman 2001) with an exceptions-only vocabulary, and
`lav` fixed initial stress (Nau 1998).  Rebuild with
`export/build_wiktionary_vocab.py`.

### Per-language OOV rules

Every rule is a named function in `stressonnx/backends/simple.py`
(`OOV_RULES`) whose docstring carries its linguistic source, and every rule
is scored against the language's own curated vocabulary in
[benchmarks/RESULTS.md](../benchmarks/RESULTS.md) (reproduce with
`python benchmarks/oov_rules_eval.py`).  Highlights:

| Language(s) | Rule | Source | Accuracy |
|---|---|---|---|
| kaz kir tat bak aze uzb kjh sah udm xal | final vowel | Turkic/Permic final-stress default (Kirchner 1998, Poppe 1964, Krueger 1962, Winkler 2001) | 0.82–1.00 (sah is an open item, see scoreboard) |
| chv | last **full** vowel — reduced ӑ/ӗ never stressed; all-reduced words stress the first syllable | Clark 1998, Krueger 1961, Dobrovolsky 1999 (ICPhS) | 0.94 |
| kat | antepenultimate vowel, initial for shorter words | Akhvlediani 1949, Gudava 1969, Aronson 1990 (Georgian stress is weak and contested — Borise 2020 argues fixed initial) | 1.00 |
| hye | last non-schwa vowel (ը never stressed) | Chakmakjian 2024 (Speech Prosody) | 0.78 |
| tgk | final vowel, except unstressed word-final izafet -и (stressed final /i/ is written ӣ) | Perry 2005, *A Tajik Persian Reference Grammar* | 0.74 |
| erz mdf | first vowel; Moksha retracts off an initial high vowel to a following а/я syllable | Hamari & Ajanki 2022, *Oxford Guide to the Uralic Languages* §23.2.3 | 0.54 / 0.76 |
| kbd | final vowel, penult when the word ends in the schwa letter э | Jaimoukha; consistent with Colarusso 1992 (low confidence) | 0.40 |
| bel_simple | no mark — Belarusian stress is lexical, not positional | East Slavic accentology literature | vocabulary only |

Single-vowel OOV words are always stressed on their sole vowel regardless of
rule (upstream `SimpleAccentor` semantics) — including for `bel_simple`.

### Limitations

- **Large download.** ~470 MB is unsuitable for edge devices.
- **Russian only.**  The pipeline is monolingual by design.
- **Context window.** Works best on short-to-medium sentences.  Very long
  inputs (>512 subword tokens) are truncated internally by the BERT
  tokenizer.
- **Proper nouns and neologisms** not in the accent dictionary fall through
  to `nn_accent`, which is character-level and may produce suboptimal results
  for foreign-origin words.
- **Poetry / non-standard stress** (for expressive effect) is not handled.
- **Monosyllables** are frequently left unmarked (see the monosyllable table
  above) — the dictionary/model path only runs for words with 2+ vowels.
- **Not layout-preserving.**  Aggressive input normalization drops symbols
  like `…` and does not guarantee whitespace is round-tripped exactly.
- **Idempotency:** re-running on already-marked text strips and re-derives
  rather than skipping (see the idempotency table above).

---

## `"silero"` — neural ONNX

**Default for `ukr`, `bel`.  Available for `ru`.**

### Architecture

Fasttext-style n-gram embedding-bag + MLP classification heads.
Derived from [silero_stress](https://github.com/snakers4/silero-models) (MIT).

```
Input word
      │
      ▼
Tokenise → character n-grams (1–3 chars)
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
should not receive a stress mark or a ё rewrite respectively — e.g.
monosyllables, clitics).

The original PyTorch model uses `nn.EmbeddingBag` with mode `"mean"`.
Since `onnxruntime` does not support `EmbeddingBag` directly, the export
splits the computation:
- The embedding matrix is saved as `embedding.npy` and the n-gram pool is
  computed in NumPy.
- Only the MLP head(s) (linear + activation + linear) are exported as
  `accentor.onnx`.

### е→ё handling for Russian

The Russian silero pipeline **does restore е→ё**: it runs a second MLP head
(`yo_logits`) alongside the stress head and rewrites `е` to `ё` when the
predicted yo-position coincides with the predicted stress position (`ё` is
always stressed in Russian orthography, so it only rewrites where that
constraint holds).  What it does **not** do is resolve **yo-homographs** —
words that are ambiguous purely because of the е/ё distinction, such as
`все` ("everyone") vs `всё` ("everything").  Both stay written as `все`;
neither `silero` nor `ruaccent` (see above) disambiguates this specific
pair — the upstream homograph-resolution model that would be needed has not
been exported.

### Why Ukrainian and Belarusian use this model

Both languages have **free stress** similar to Russian.  The silero neural
model covers a large vocabulary and provides reasonable generalisation to OOV
words via character n-gram features.

Ukrainian specifics:
- Vowels і, и, е, є, а, о, у, ю, я are all potentially stressed.
- No е/ё distinction to disambiguate.
- Fewer systematic homograph pairs than Russian.

Belarusian specifics:
- **Akane**: unstressed /o/ merges phonetically with /a/; stress is needed to
  pronounce words correctly.
- Stress is not marked in standard orthography.
- Fewer neural training resources than Russian → silero neural model is the
  best available quality without a full BERT pipeline; see the benchmark
  table above — silero (0.859 accuracy) is dramatically better here than the
  `bel_simple` vocabulary fallback (0.417).

### Why silero is also available for Russian

The silero Russian model exists and is accurate for unambiguous words.  It is
a valid alternative when:
- The ~470 MB ruaccent download is impractical.
- The application does not need homograph disambiguation (e.g., spelling
  pronunciation or read-aloud of unambiguous text).
- Fast cold-start is required (silero ru downloads ~5 MB).

It will assign the same stress to both readings of `замок`; for TTS of
mixed-context text this is audible.  See `benchmarks/RESULTS.md`: on
Russian, `ruaccent` scores 0.742 homograph accuracy against `silero`'s 0.377.

### Limitations

- **No homograph disambiguation** for Russian.  One-best prediction per word.
- **No yo-homograph disambiguation.**  е→ё restoration runs, but
  `все`/`всё`-style ambiguity is left as-is (see above).
- **Word-level context only.**  The n-gram features are character-level; the
  model has no access to surrounding words.
- **Vocabulary coverage** is finite.  Neologisms and foreign words rely on
  n-gram generalisation, which degrades for non-Cyrillic stems.
- The **Ukrainian and Belarusian models are verified to match the original**
  silero outputs exactly (see `export/verify_e2e.py`).  The Russian model
  is verified numerically at export (max |diff| < 1e-3).

---

## `"kubataba"` — seq2seq Transformer

**Alternative for `ru`.**

### Architecture

Character-level encoder-decoder Transformer trained on Russian literary
texts with explicit stress marks.  Derived from
[kubataba/Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor)
(MIT).

```
Input sentence (character sequence, max 256 chars)
      │
      ▼
CharacterEmbedding  (char → d_model=256)
      │
      ▼
encoder.onnx  (4-layer Transformer encoder)
      → memory: float32[1, 256, 256]
      │
      ▼
Greedy decode loop (Python):
  tgt = [BOS]
  while not EOS and len < max_len:
    decoder_step.onnx(memory, tgt) → logits[1, vocab_size]
    next_token = argmax(logits[-1])
    tgt.append(next_token)
      │
      ▼
Decode token sequence → output text with U+0301 stress marks
```

The decoder implements all four decoder layers manually using
`F.scaled_dot_product_attention` to avoid ONNX export incompatibilities with
PyTorch's `nn.MultiheadAttention` (which internally creates data-dependent
causal masks incompatible with the TorchScript and dynamo exporters).

### Why a seq2seq model for Russian

A seq2seq model can, in principle, capture **cross-word context** without
requiring a separate disambiguation pipeline: the encoder reads the whole
sentence, and the decoder produces each stressed character attending to the
full context.  This makes it architecturally interesting as a single-model
alternative to the ruaccent multi-stage pipeline.

### Limitations

- **Sentence-level input required.**  The model is trained on sentence pairs,
  not isolated words.  Single words fed in isolation produce degraded output.
- **Slower inference** than silero or simple: autoregressive decode is
  sequential; O(output_len) ONNX session calls per sentence.
- **No explicit homograph dictionary.**  Disambiguation depends entirely on
  what the model learned during training.  Performance on rare homograph pairs
  is unpredictable.
- **Maximum input length: 256 characters.**  Sentences longer than this are
  truncated at the encoder.
- **Research quality.**  This model is provided as an alternative, not as the
  recommended production choice.  Use `"ruaccent"` for production Russian TTS.
- **Numerical precision:** float32 ONNX; verified max |diff| = 2.16×10⁻⁴
  vs the original PyTorch model (argmax identical on all tested sentences).
- **Monosyllable and idempotency behavior is model-dependent** — there is no
  explicit rule for either case; see the tables above.

---

## `"simple"` — vocabulary + rules

**Default for the 26 rule/vocabulary languages** (Turkic, Caucasian, Uralic, and the Wiktionary/dictionary-backed Slavic and Baltic set).

### Architecture

Word-level dictionary with a per-language OOV positional fallback.
No ONNX inference; no neural computation at runtime.

```
Input sentence
      │
      ▼
Tokenise: split on whitespace + punctuation, preserve hyphens
      │
per token:
      ├──[in vocab]──→ stress_char_idx from dictionary → insert U+0301
      │
      └──[OOV]──→ if exactly 1 vowel, always stress it; otherwise apply the
                    per-language positional rule:
                    "last"   → rightmost vowel
                    "first"  → leftmost vowel
                    "kat"    → ≤3 vowels → first, else penultimate
                    "none"   → unstressed (no mark inserted)
```

### Per-language linguistic motivation

**Azerbaijani (`aze_cyr`, `aze_lat`)**
Agglutinative Turkic language with predominantly **last-syllable stress**.
Stress shifts predictably with suffixation: each added suffix carries the
potential stress position one syllable further right.  OOV rule `"last"` is
correct for the vast majority of native forms; loanwords are the main source
of error.

**Uzbek (`uzb_cyr`, `uzb_lat`)**
Same Turkic last-syllable default as Azerbaijani.  Uzbek has additional
complexity from vowel harmony collapsing in the southern dialects, but stress
position is not affected — the last syllable rule holds strongly.

**Bashkir (`bak`)**
Turkic, closely related to Tatar.  Last-syllable stress default with
grammar-conditioned exceptions for certain suffixes (negative verb forms,
interrogative particles).  The dictionary covers these exceptions; OOV falls
back to last vowel.

**Chuvash (`chv`)**
Turkic (Oghuric branch, distinctive from mainstream Turkic).  Stress
**migrates**: it falls on the last non-reduced syllable.  The last-vowel OOV
rule approximates this for standard forms; morphological reduction is not
modelled.

**Tatar (`tat`)**
Kipchak Turkic.  Last-syllable stress default; some enclitics and particles
are **unstressed**.  The dictionary captures the most frequent forms.

**Kyrgyz (`kir`)**
Kipchak Turkic (close to Kazakh).  Last-syllable stress; vowel harmony is
preserved in standard orthography.  OOV last-vowel rule is reliable.

**Kazakh (`kaz`)**
Kipchak Turkic.  Last-syllable stress default; Russian loanwords retain their
original stress (these are in the dictionary).  Stress does not change with
case or conjugation suffixes → OOV rule works well.

**Khakas (`kjh`)**
Siberian Turkic, Sayan branch.  Last-syllable stress; agglutinative suffixing.
OOV last-vowel rule is correct for native vocabulary.

**Yakut/Sakha (`sah`)**
Siberian Turkic.  Stress interacts with vowel length: **long vowels attract
stress** and diphthongs are always stressed.  Standard orthography marks long
vowels, so the dictionary covers most cases.  OOV last-vowel rule is a rough
approximation.

**Tajik (`tgk`)**
Iranian language (not Turkic).  Stress in Tajik is **penultimate** in most
native words; last-syllable in loanwords and with certain suffixes.  The
last-vowel OOV rule is a simplification but covers loanword-heavy usage.

**Kabardino-Balkarian (`kbd`)**
Northwest Caucasian (Kabardian) + Turkic (Balkarian) — two different
languages sharing an orthographic standard.  Kabardian has **first-syllable**
or **root-syllable** default stress; Balkarian follows the Kipchak last-syllable
pattern.  The shared dictionary handles the most common forms; OOV last-vowel
is a compromise.

**Armenian (`hye`)**
Indo-European, isolated branch.  Eastern Armenian has **penultimate stress**
as the default rule; Western Armenian is last-syllable.  The library follows
Eastern Armenian (the more widely standardised form).  OOV last-vowel rule
approximates this poorly for multi-syllable native words — the dictionary
coverage is the main quality lever.

**Georgian (`kat`)**
South Caucasian (Kartvelian), unrelated to Indo-European.  Stress in Georgian
is **initial** for words with ≤3 vowels and **penultimate** for longer words.
The `"kat"` OOV rule implements this heuristic.  Georgian orthography is
fully phonemic and stress is historically recessive, so dictionary coverage
is high for standard vocabulary.

**Erzya (`erz`) and Moksha (`mdf`)**
Mordvinic (Uralic) languages of the Volga region.  Both have **initial
stress** as the dominant pattern, with exceptions for certain grammatical
suffixes.  The first-vowel OOV rule is reliable for native vocabulary.

**Udmurt (`udm`)**
Permic (Uralic).  Stress is consistently on the **last syllable**.
OOV last-vowel rule is highly accurate for this language.

**Kalmyk (`xal`)**
Mongolic language (Oirat branch).  Stress is on the **last syllable** in
most words; vowel harmony and vowel reduction interact with stress assignment.
OOV last-vowel rule holds for standard forms.

**Belarusian simple (`bel_simple`)**
Same language as `bel` but using vocabulary-only lookup without neural
inference — a lower-accuracy fallback (0.417 vs. `silero`'s 0.859, see the
benchmark table above), meant for offline environments where ONNX inference
is unavailable, or as an explicit fallback target.  Its OOV rule is `"none"`,
but that rule only applies to **multi-vowel** OOV words: a **single-vowel**
OOV word is always stressed on that vowel (the "always stress monosyllables"
rule takes precedence over the per-language OOV rule, matching upstream
`SimpleAccentor` behavior).  A multi-vowel word absent from the vocabulary
is left with no mark at all.

### Limitations

- **Dictionary coverage** determines quality.  Words absent from the
  vocabulary fall back to a positional heuristic, which is accurate for most
  agglutinative languages but unreliable for irregular forms, foreign
  loanwords, and proper nouns.
- **No context-sensitivity.**  Each word is stressed independently; no
  cross-word influence.
- **Case-insensitive lookup** is used for vocabulary matching; capitalised
  words (sentence-initial, proper nouns) are lowercased before lookup.
- For **Georgian** and **Armenian**, the OOV heuristic is noticeably weaker
  than for the Turkic languages; dictionary coverage is therefore more
  important.
- The vocabulary files are derived from
  [silero_stress](https://github.com/snakers4/silero-models) (MIT) and
  reflect the training data used there.  Low-frequency or dialectal forms may
  be missing.
- The original 20 `simple` vocabularies (`bel_simple` included) are exports of the
  upstream `silero_stress` `SimpleAccentor` data — there is currently no
  independent gold-standard accuracy measurement for these languages (see
  the "Languages without independent gold" note in
  `benchmarks/RESULTS.md`); correctness is locked to the upstream reference
  via `tests/test_export_parity.py`, not benchmarked against external data.

---

## Architecture comparison

| | ruaccent | silero | kubataba | simple |
|---|---|---|---|---|
| **Languages** | ru | ukr, bel, ru | ru | 26 languages |
| **Context** | Sentence (BERT) | Word (n-gram) | Sentence (Transformer) | Word (dict) |
| **Homograph resolution** | Yes (BERT + RoBERTa) | No | Partial (learned) | No |
| **е→ё restoration (ru)** | Yes | Yes | Model-dependent | N/A |
| **Yo-homograph (все/всё) disambiguation** | No | No | No | N/A |
| **ONNX inference** | 4 models | 1 model | 2 models | None |
| **Download size** | ~470 MB | ~5–10 MB | ~30 MB | ~1–5 MB |
| **Cold-start latency** | High (4 sessions) | Low | Medium (2 sessions) | Near-zero |
| **Per-sentence latency** | ~50–200 ms | ~2–10 ms | ~10–50 ms (autoregressive) | <1 ms |
| **Runtime deps** | onnxruntime, tokenizers | onnxruntime, numpy | onnxruntime, numpy | numpy only |
| **OOV handling** | nn_accent (char-level) | n-gram generalisation | seq2seq generalisation | positional rule |
| **Sentence-level input** | Required | Works word-by-word | Required | Works word-by-word |
| **Max input length** | ~512 subword tokens | Unlimited (per-word) | 256 characters | Unlimited |
| **Idempotent on pre-marked text** | No (re-derives) | Yes (skips) | No (re-derives) | Yes (skips) |
| **License** | Apache-2.0 | MIT | MIT | MIT |

See [`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) for measured
accuracy numbers instead of qualitative claims.
