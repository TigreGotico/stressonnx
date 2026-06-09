# stressonnx — model reference

## Choosing a model

| Situation | Recommendation |
|-----------|---------------|
| Russian general-purpose | `model="ruaccent"` (default) |
| Russian in memory-constrained environment | `model="silero"` (~5 MB vs ~470 MB) |
| Russian research / seq2seq comparison | `model="kubataba"` |
| Ukrainian / Belarusian | `model="silero"` (default) |
| Turkic / Caucasian / minority languages | `model="simple"` (default) |

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
  — per-token: STRESS / NO_STRESS / YO
  — skips words known to be unstressed (particles, prepositions, …)
      │
      ├──[yo tokens]──→ nn_yo_homograph (DistilBERT, ~14 MB)
      │                  — decides whether е should become ё
      │
      ├──[omograph tokens]──→ nn_omograph (RoBERTa NLI turbo2, ~343 MB)
      │                        — picks the correct stressed variant
      │                          from the homograph dictionary
      │
      └──[remaining tokens]──→ accent dictionary lookup
                                  → nn_accent (RoFormer char-level, ~0.8 MB)
                                    for words not in the dictionary
```

### Runtime details

- Tokenizers: loaded via the `tokenizers` library (HuggingFace fast tokenizer
  JSON format).  No `transformers` or `torch` required at runtime.
- `onnxruntime` sessions are created lazily on the first call.
- Total download: ~470 MB (cached after first use under
  `~/.local/share/stressonnx/ru_ruaccent/`).

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
accentor.onnx  (MLP stress_clf head)
      │
      ▼
argmax over character positions → stress index
      │
      ▼
Insert U+0301 at that position
```

Exceptions and skip-lists are consulted before the ONNX call:
`exceptions.txt.gz` (word → explicit stress index) and
`skip_stress_words.txt.gz` (words that should not be stressed, e.g.
monosyllables, clitics).

The original PyTorch model uses `nn.EmbeddingBag` with mode `"mean"`.
Since `onnxruntime` does not support `EmbeddingBag` directly, the export
splits the computation:
- The embedding matrix is saved as `embedding.npy` and the n-gram pool is
  computed in NumPy.
- Only the MLP head (linear + activation + linear) is exported as
  `accentor.onnx`.

### Why Ukrainian and Belarusian use this model

Both languages have **free stress** similar to Russian.  The silero neural
model covers a large vocabulary and provides reasonable generalisation to OOV
words via character n-gram features.

Ukrainian specifics:
- Vowels і, и, е, є, а, о, у, ю, я are all potentially stressed.
- No yo-homograph problem (Ukrainian uses і, not е/ё distinction).
- Fewer systematic homograph pairs than Russian.

Belarusian specifics:
- **Akane**: unstressed /o/ merges phonetically with /a/; stress is needed to
  pronounce words correctly.
- Stress is not marked in standard orthography.
- Fewer neural training resources than Russian → silero neural model is the
  best available quality without a full BERT pipeline.

### Why silero is also available for Russian

The silero Russian model exists and is accurate for unambiguous words.  It is
a valid alternative when:
- The ~470 MB ruaccent download is impractical.
- The application does not need homograph disambiguation (e.g., spelling
  pronunciation or read-aloud of unambiguous text).
- Fast cold-start is required (silero ru downloads ~5 MB).

It will assign the same stress to both readings of `замок`; for TTS of
mixed-context text this is audible.

### Limitations

- **No homograph disambiguation** for Russian.  One-best prediction per word.
- **Word-level context only.**  The n-gram features are character-level; the
  model has no access to surrounding words.
- **Vocabulary coverage** is finite.  Neologisms and foreign words rely on
  n-gram generalisation, which degrades for non-Cyrillic stems.
- The **Ukrainian and Belarusian models were verified to match the original**
  silero outputs exactly (see `export/verify_e2e.py`).  The Russian model
  was verified numerically at export (max |diff| < 1e-3).

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

---

## `"simple"` — vocabulary + rules

**Default for 20 Turkic, Caucasian, and minority Slavic languages.**

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
      └──[OOV]──→ positional rule:
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
inference.  OOV words receive no stress mark.  Use for offline environments
where ONNX is unavailable, or as a fallback.

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

---

## Architecture comparison

| | ruaccent | silero | kubataba | simple |
|---|---|---|---|---|
| **Languages** | ru | ukr, bel, ru | ru | 20 languages |
| **Context** | Sentence (BERT) | Word (n-gram) | Sentence (Transformer) | Word (dict) |
| **Homograph resolution** | Yes (BERT + RoBERTa) | No | Partial (learned) | No |
| **ONNX inference** | 4 models | 1 model | 2 models | None |
| **Download size** | ~470 MB | ~5–10 MB | ~30 MB | ~1–5 MB |
| **Cold-start latency** | High (4 sessions) | Low | Medium (2 sessions) | Near-zero |
| **Per-sentence latency** | ~50–200 ms | ~2–10 ms | ~10–50 ms (autoregressive) | <1 ms |
| **Runtime deps** | onnxruntime, tokenizers | onnxruntime, numpy | onnxruntime, numpy | numpy only |
| **OOV handling** | nn_accent (char-level) | n-gram generalisation | seq2seq generalisation | positional rule |
| **Sentence-level input** | Required | Works word-by-word | Required | Works word-by-word |
| **Max input length** | ~512 subword tokens | Unlimited (per-word) | 256 characters | Unlimited |
| **License** | Apache-2.0 | MIT | MIT | MIT |
