# stressonnx: model reference

> Per-language linguistic background lives in [languages.md](languages.md).

## Choosing a model

| Situation | Recommendation |
|-----------|---------------|
| Russian general-purpose | `model="ruaccent-silero"` (default), or `prefer="best"`: the `ruaccent` pipeline, then the `silero` mark on the words it leaves unmarked; the most accurate id on both gold sets |
| Russian, function words left unmarked | `model="ruaccent"` |
| Russian, comparing homograph models | `model="ruaccent-big-poetry"` (the RUAccent `big_poetry` homograph model, less accurate on prose) |
| Russian in memory-constrained environment | `model="silero"` (about 50 MB download and 170 MB resident, vs about 470 MB download), or `prefer="fast"`. `prefer="smallest"` picks `simple`, the smallest download (10 MB), which holds about 510 MB once loaded |
| Russian, less memory with homograph resolution | `model="ruaccent-silero-int8"`: about 145 MB download and 400 MB resident, a small loss on homographs |
| Ukrainian / Belarusian | `model="silero"` (default) |
| Turkic / Caucasian / minority languages | `model="simple"` (default) |
| Don't care which model, just want a capability | `prefer="best"` / `"fast"` / `"smallest"` instead of a model id |
| Russian, the RusStress/StressRNN reading for comparison or parity with a system built on it | `model="stressrnn"` (never selected by default or by `prefer=`) |
| Russian, a second opinion from a sentence-level seq2seq model | `model="kubataba"` (opt-in only; below both `ruaccent` and `silero` in accuracy and far slower) |
| Ukrainian, the ukrainian-accentor-transformer reading for comparison or parity with a system built on it | `model="uk-accentor"` (never selected by default or by `prefer=`) |

Full accuracy numbers, not just relative quality claims, live in
[`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md), reproducible with
`python benchmarks/run.py`. Headline results:

| lang | model | accuracy | homograph accuracy |
|------|-------|----------|--------------------|
| ru | ruaccent | 0.947 | 0.833 |
| ru | ruaccent-int8 | 0.946 | 0.825 |
| ru | ruaccent-big-poetry | 0.942 | 0.753 |
| ru | ruaccent-silero | 0.951 | 0.833 |
| ru | ruaccent-silero-int8 | 0.951 | 0.825 |
| ru | silero | 0.944 | 0.753 |
| ru | stressrnn | 0.892 | 0.767 |
| ru | kubataba | 0.888 | 0.816 |
| ru | simple | 0.871 | 0.000 |
| uk | silero | 0.843 | not measured |
| uk | uk-accentor | 0.795 | not measured |
| uk | simple | 0.534 | not measured |
| be | silero | 0.924 | not measured |
| be | simple | 0.867 | not measured |

The rows are the russtress gold set (11124 targets; homograph accuracy on
the 724 targets in the RUAccent omograph roster). On Russian, `ruaccent`
and `silero` differ by 0.003 in plain accuracy (exact McNemar p = 0.047);
`ruaccent` is more accurate on the homograph targets. On the Wiktionary
example sentences (137221 targets) `silero` scores 0.984 against
`ruaccent`'s 0.976 (p = 1e-75), and `ruaccent` keeps the higher homograph
accuracy (0.926 against 0.908). Its larger download gives homograph
resolution from a wider context. The default, `ruaccent-silero`, has the
highest score on both gold sets (0.951 and 0.990); its int8 variant ties
with it (McNemar p = 0.307 and 0.786). `stressrnn` is below `ruaccent` and
`silero` (McNemar p = 7e-106 against `ruaccent`, p = 6e-90 against
`silero`), and the russtress file is the
corpus the StressRNN BiLSTM was trained on, so its number is optimistic.
For Belarusian, `silero` (neural, the default) is substantially more
accurate than `model="simple"` (vocabulary lookup). For Ukrainian,
`uk-accentor` scores 0.795 (95% CI 0.788 to 0.802) against `silero`'s 0.843
(0.837 to 0.850) on the same 12 252 russtress targets, McNemar p = 7e-63,
with `benchmarks/run.py` scoring (a hit marks exactly the gold vowel and no
other; `uk-accentor` writes two marks on some words). The russtress rows
carry labels from the same lineage as its training data (see the
`uk-accentor` section), so its number is optimistic. Use `model="simple"`
only where ONNX inference is unavailable. See `benchmarks/RESULTS.md` for
row counts, the Wiktionary and silver sets, confidence intervals and the
McNemar tests.

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
`кто́-то`, `како́й-нибудь`, `пришёл-таки`. This applies identically in the
`ruaccent`, `silero` and `simple` pipelines for `ru`.

---

## Monosyllable (single-vowel word) policy

Backends differ in whether they force a mark onto a word with exactly one
vowel:

| Backend | Single-vowel word policy |
|---------|--------------------------|
| `simple` | Always stressed. The one vowel present is marked, even under an OOV `"none"` rule (see `be`/`simple` below). |
| `silero` | Always stressed. A word with exactly one vowel gets it marked regardless of the model's own prediction. |
| `ruaccent` | The dictionary/neural accent path only fires for words with **more than one** vowel. A bare, dictionary-absent monosyllable is often left **unmarked**. Single-vowel entries hardcoded in the accent dictionary (for example `о` → `+о`) are still marked. |
| `ruaccent-silero` | As `ruaccent`: the fill takes the `silero` mark only on a token with two or more vowels, so `на`, `он` and `дом` stay as `ruaccent` left them. A hyphenated token counts its vowels as one and keeps one mark (`что́-то`, `и́з-за`, `чуть-чу́ть`). |
| `stressrnn` | Always stressed, prepositions and particles included (`на́`, `а́`), and so is every hyphen part (`кто́-то́`). This is upstream behaviour. |
| `kubataba` | Never marked. The upstream model treats a single-vowel word as inherently stressed and emits no mark for it. |
| `uk-accentor` | Whatever the model writes: most monosyllables stay unmarked (`як`, `що`, `село`), some get a mark. This is upstream behaviour. |

---

## Idempotency (re-stressing already-marked text)

Calling a backend on text that already carries the combining acute
(U+0301) is safe, but the backends handle it differently:

| Backend | Behavior on pre-marked input |
|---------|-------------------------------|
| `simple` | Skips: if U+0301 is already present in a token, that token comes back unchanged. |
| `silero` | Skips: a word already containing the stress token short-circuits before any prediction. |
| `ruaccent` | Strips and re-derives from scratch. Marks are not detected as "already done", yo-homograph and omograph resolution always re-run. A word written with ё is always labelled as stressed, so it gets the same marks whether the input writes е or ё. A second run gives back the output of the first unless a restored ё changes what the stress-usage model reads for a neighbouring word. |
| `kubataba` | Skips: the model reads the text without its marks, and a word that already carried one keeps it and gets no new mark. |
| `stressrnn` | Skips: a word that already carries U+0301 is left as it is. |
| `uk-accentor` | Strips the marks for the model, then keeps every pre-marked word as it was and drops the model's marks on it. Upstream re-marks every word. |

If you need to guarantee a no-op on already-stressed text, prefer `simple`
or `silero` for that language, or check for U+0301 yourself before calling.

---

## `"ruaccent"`: homograph-aware Russian

The base of `ruaccent-silero`, the default for `ru`.

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
[RUAccent](https://github.com/Den4ikAI/ruaccent) (Den4ikAI, MIT).
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

The models read a normalized copy of the input, as upstream RUAccent does.
The copy keeps only the Russian alphabet, ASCII letters and digits,
whitespace, and a fixed set of punctuation. The output is the full input:
every other character (`…`, `№`, `і`, `ў`, `é`, Greek, CJK, emoji) and
every run of whitespace comes back unchanged. Only the stress marks and
the е→ё restorations are added. A U+0301 on a vowel or on р is a stress
mark and is derived again; a U+0301 on another letter (Montenegrin `з́`)
is part of that letter, so it comes back and its word gets no mark.

The check for a changed word works on runs of letters and digits. A hyphen,
an apostrophe or a space ends a run, so each part of a compound is checked
on its own. A run that loses a letter, digit or combining mark in the copy
gets no mark, because the models saw a different word. For example, `Київ`
reaches the models as `Кив`. Two runs that the copy joins into one, as in
`молоко😀корова` or `молоко` U+200B `корова`, get no mark either. A soft
hyphen, ZWJ, ZWNJ or WORD JOINER inside a word does not change the word,
and a letter dropped behind one of them is still at the word's edge:
`молоко` U+00AD `і` gets no mark.

### Yo (ё) handling

`nn_yo_homograph` decides, per occurrence of `е`, whether to rewrite it as
`ё` (both are the same letter for stress purposes).
This resolves plain е→ё **restoration**, recovering the true pronunciation
of a word conventionally written with `е`. It also decides
**yo-homographs**, pairs where the surface form is ambiguous between two
different words depending on ё (`все` "everyone" vs `всё` "everything"),
from the sentence context: `все люди пришли` stays `все`, and `он все
знает` becomes `он всё зна́ет`. Either reading of this one-vowel word gets
no acute from `ruaccent`. The decision is a model prediction and can be wrong
(`мы все устали` becomes `мы всё уста́ли`).

The models read the sentence as written, as upstream RUAccent does. The
stress-usage and yo labels are grouped per word of the pipeline's own word
list: a punctuation run (`[…]`, `...`, `»,`) is one word to the pipeline
and one token per character to the BERT pre-tokenizer, and upstream
RUAccent 1.5.8.3 reads a shifted label on every word after such a run. A word
that holds ё is then labelled as stressed whatever the stress-usage model
said, because such a word always carries a stress. The mark position comes
from the accent dictionary or, for a word absent from it, from the character
model. The stress usually falls on the ё, and compounds place it elsewhere
(`трёхле́тний`).
A word gets the same stress whether the writer typed ё or е: `желтый` and
`жёлтый` both give `жё́лтый`. A ё that the input already has stays, and it
is not reported as a restoration. Words without ё keep the marks they get
in upstream RUAccent. The stress-usage model reads a restored ё on the next
run, so a second call can move a mark on a neighbouring word in rare
sentences (`Вот оно, самое важное место` restores `самоё́`, and the second
run marks `оно́`).

### Runtime details

- Tokenizers: loaded via the `tokenizers` library (HuggingFace fast tokenizer
  JSON format). No `transformers` or `torch` required at runtime.
- `onnxruntime` sessions are created lazily on the first call.
- Total download: about 500 MB (about 145 MB for the `-int8` ids). Model files sit in the standard Hugging
  Face cache (respecting `HF_HOME` and `HF_HUB_OFFLINE`) unless an explicit
  `cache_dir=` is passed, in which case the layout is
  `cache_dir/ru_ruaccent/<file>`.

### Precision

`ruaccent-int8` and `ruaccent-silero-int8` load `model.int8.onnx` for the
stress-usage and omograph models: the fp32 graph with its MatMul weights
and embeddings dynamically quantized by onnxruntime (`quantize_dynamic`,
QInt8). The accent and yo-homograph models are small and stay fp32.
`RuAccentStressor(precision="int8")` does the same. The default ids keep
the fp32 weights, because the quantization costs homograph accuracy.

Measured on a laptop with two onnxruntime threads per session:

| | fp32 | int8 |
|---|---|---|
| ruaccent download | 501 MB | 144 MB |
| ruaccent RSS after load | 652 MB | 313 MB |
| ruaccent-silero RSS after load | 766 MB | 398 MB |
| ruaccent cold load | 4.2 s | 2.5 s |
| ruaccent ms/row, russtress | 7.5 | 5.0 |
| ruaccent ms/row, Wiktionary sentences | 84 | 23 |
| ruaccent-silero ms/row, russtress | 10.8 | 12.3 |
| ruaccent-silero ms/row, Wiktionary sentences | 32.7 | 16.2 |
| ruaccent accuracy, russtress (11124 targets) | 0.9471 | 0.9463 |
| ruaccent homograph accuracy, russtress (724) | 0.833 | 0.825 |
| ruaccent accuracy, Wiktionary (137221 targets) | 0.9762 | 0.9760 |
| ruaccent-silero accuracy, russtress | 0.9513 | 0.9507 |
| ruaccent-silero accuracy, Wiktionary | 0.9900 | 0.9900 |
| ruaccent-silero accuracy, ruslan-stressed (SILVER, 176255) | 0.9780 | 0.9781 |

The loss is below 0.1 points on every set and falls on the homographs,
the part the omograph model decides. A homograph whose margin is already
small can flip: `Дверной замок надёжен.` after 80 filler sentences gets
`за́мок` from int8 and `замо́к` from fp32. Choose int8 when download size or
memory matter more than that margin. int8 is faster on long sentences, but
`ruaccent-silero-int8` is slower than fp32 on the short russtress windows.

### `ruaccent-silero`: the ruaccent pipeline with silero marks on its unmarked words

`model="ruaccent-silero"` runs the `ruaccent` pipeline and then, for every
whitespace token of letters and hyphens with two or more vowels that came
out of it with no mark, takes the mark `silero` puts on that token. A token
with a mark on any part keeps it alone (`по-мо́ему` does not become
`по́-мо́ему`), and a token the models did not see whole (a dropped
character in it) stays unmarked. The `ruaccent` marks never move: on
russtress and Wiktionary no target that `ruaccent` gets right is lost.

`silero` marks each hyphen part as a word of its own. A hyphenated token
keeps one mark: on its last marked part, because a Russian hyphenated
word carries its primary stress on the last part and at most a secondary
stress before it (Wiktionary: `чу̀ть-чу́ть`, `во̀т-во́т`, `Нью̀-Йо́рк`,
`ко̀е-что́`, `нау̀чно-техни́ческий`), or on its first part for the compound
prepositions `из-за`, `из-под`, `по-за`, `по-над`, which are proclitics
with a dictionary stress on the first part (Wiktionary: `и́з-за`,
`и́з-под`). The secondary stress is not written: `научно-технический`
comes out `научно-техни́ческий`, with one acute, which is what the gold
sets score and what `ruaccent` writes for the words it marks itself. On
the 326 gold targets where `silero` marks two parts of a token `ruaccent`
left empty, this rule matches the gold vowel 271 times; the last mark alone
164, the first mark alone 155, `silero`'s higher-confidence part 150. The
rest are russtress rows that write `из-за́`. `по-мо́ему` and `по-тво́ему`
come from a two-entry table in the `ruaccent` pipeline (en.wiktionary
headwords; the datives `моему́` and `твоему́` keep the dictionary stress),
so the fill never sees them; `по-сво́ему`, `по-на́шему` and `по-ва́шему`
come out right without an entry.

The words `ruaccent` leaves unmarked are common. Its stress-usage model
labels function words NO_STRESS in most contexts (`или`, `это`, `его`,
`было`, `только`, `через`, `перед`, `если`, `между`, `она`), and the
one-vowel head of `что-то`, `как-то`, `кто-то` is absent from the accent
dictionary and below the character model's two-vowel floor. Upstream
RUAccent 1.5.8.3 leaves the same words unmarked. On the Wiktionary
sentences 2.6% of the gold targets come back empty, on ruslan-stressed
3.0%, on prose-accents 10.5%.

Scores on the two gold sets, against the `ruaccent` rows of the same run
(exact McNemar on the targets both models scored; the run logs and
rows are in `benchmarks/RESULTS.md`):

| set | targets | ruaccent | silero | ruaccent-silero | vs ruaccent (only right / ruaccent only right / p) | vs silero |
|-----|---------|----------|--------|-----------------|------|------|
| russtress | 11124 | 0.947 | 0.944 | 0.951 | 46 / 0 / 3e-14 | 179 / 97 / 9e-07 |
| wiktionary | 137221 | 0.976 | 0.984 | 0.990 | 1893 / 0 / < 1e-300 | 1399 / 615 / 6e-70 |

On russtress the homograph accuracy is that of `ruaccent` (0.833),
because the fill never touches a word `ruaccent` marked. On Wiktionary it
rises from 0.926 to 0.933: the fill marks homograph targets that `ruaccent`
left unmarked.
The cost is the `silero` bundle on top of `ruaccent` (48 MB more download,
about 30 MB more peak memory: 763 MB against 730 MB on 300 russtress rows)
and one `silero` pass per call (11.4 against 9.5 ms per russtress row in
the scoreboard run). `ruaccent` itself is unchanged: `model="ruaccent"`
gives the same output as before.

### The big_poetry omograph model

`model="ruaccent-big-poetry"` runs the same pipeline with the RUAccent
`big_poetry` homograph model in place of `turbo3.1`. It is a ruBERT-base
NLI classifier (BERT, 12 layers, about 714 MB). It reads the same input as
`turbo3.1`: the sentence with the homograph between `<w>` and `</w>`,
paired with one stressed candidate. As a BERT model it also reads the
segment ids of the pair. The files are verbatim copies of
`nn/nn_omograph/big_poetry` in `ruaccent/accentuator` and sit under
`ru_ruaccent/nn_omograph_big_poetry/`. Only the selected homograph model
is downloaded and loaded. The other three models and the dictionaries are
shared with `ruaccent`.

On the Russian evaluation sets, `big_poetry` is less accurate than
`turbo3.1` (`benchmarks/RESULTS.md`, models `ruaccent` and
`ruaccent-big-poetry`). The table scores the vowel of the target word;
the last column is the exact McNemar test between the two.

| set | targets | turbo3.1 | big_poetry | McNemar p |
|-----|---------|----------|------------|-----------|
| russtress ru, all words | 11,124 | 0.947 | 0.942 | 1e-05 |
| russtress ru, homographs | 724 | 0.833 | 0.753 | (subset of the row above) |
| HomographResolutionEval, homograph word (silver) | 1,727 | 0.888 | 0.852 | 1e-04 |
| prose-accents (silver) | 16,839 | 0.944 | 0.944 | 0.742 |

It loads about 450 MB more memory (1.2 GB against 0.7 GB peak RSS). Use it
to compare homograph models, or for text where it measures better. The
default stays `turbo3.1`.

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
- **Letters outside the Russian alphabet.** A run of letters that contains
  `і`, `ў`, `é` or another letter that the models do not read stays
  unmarked. The other parts of a hyphen or apostrophe compound still get a
  mark from the Russian models, and that mark can be wrong for the
  language: `віцэ-каралева` gives `віцэ-кара́лева` and `подвір’ями` gives
  `подвір’ями́`.
- **Idempotency:** re-running on already-marked text strips and re-derives
  rather than skipping (see the idempotency table above).

---

## `"silero"`: neural ONNX

**Default for `uk`, `be`. Available for `ru`.**

### Architecture

A fasttext-style n-gram embedding-bag plus MLP classification heads.
Derived from [silero_stress](https://github.com/snakers4/silero-stress)
(Copyright (c) 2020-present Silero Team, MIT). The ru bundle carries the
upstream licence as `LICENSE.silero-stress`.

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

The exceptions dictionary is consulted before the ONNX call:
`exceptions.txt.gz` (word → explicit stress index, plus an explicit yo index
for `ru`).

For `ru` a homograph resolver (the silero_stress homosolver) runs first. Each
word of its dictionary (`homodict.json.gz`, about 2,200 words with two
stressed variants each) gets the variant that its context selects. A phrase
regex (`homophrases.json.gz`) decides when the context contains a known
phrase. Otherwise a three-layer BERT (`homosolver.onnx`, 312 hidden units)
reads up to 150 characters of cleaned context on each side of the word,
written as `[HOMO] word [/HOMO]`, and a classifier over the marker and word
vectors picks one of the two variants. The accentor then keeps the resolved
stress and ё for those words.

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
constraint holds). **Yo-homographs**, words that are ambiguous purely
because of the е/ё distinction such as `все` ("everyone") vs `всё`
("everything"), go through the homograph resolver described above, which
picks `все́` or `всё́` from the context.

### Why Ukrainian and Belarusian use this model

Both languages have **free stress** similar to Russian. The silero neural
model covers a large vocabulary and generalises reasonably to OOV words via
character n-gram features.

Ukrainian specifics:
- Vowels і, и, е, є, а, о, у, ю, я are all potentially stressed.
- No е/ё distinction to disambiguate.
- Fewer systematic homograph pairs than Russian.

### Ukrainian dictionary layer

The Ukrainian silero bundle carries a dictionary layer built from
[ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress)
2.1.0 (lang-uk, MIT). That project ships about 2.9 million Ukrainian word
forms with their stress, which it derives from
["Dictionaries of Ukraine"](https://lcorp.ulif.org.ua/dictua/) (Словники
України) by ULIF. The layer lives in the bundle's
`exceptions.txt.gz`, which the runtime reads before it runs the neural
model. No torch, Stanza or marisa-trie is needed at runtime.

The layer holds only the forms that have exactly one stress in the
dictionary and that the neural model stresses differently. The model
already stresses the other 2.6 million unambiguous forms the same way as
the dictionary, so they are left out. This keeps the file at about 350 KB
and about 96,000 entries. The full set of unambiguous forms is about 9 MB
compressed and costs about 470 MB of memory when loaded.

The neural model still decides these cases:

- Heteronyms (за́мок/замо́к, бло́хи/блохи́). Upstream resolves them with
  Stanza part-of-speech tags, which this library does not run.
- Forms that the Wiktionary vocabulary of the `simple` model stresses on
  another vowel. For `його` and `кому` the dictionary has only the noun
  forms йо́го (vocative of йо́га) and ко́му (accusative of ко́ма), and lacks
  the pronouns його́ and кому́.
- Forms with variant stress (по́ми́лка), which the dictionary marks twice.
- Forms whose case or apostrophe variants disagree on the stress.
- Hyphenated compounds and multiword entries, which the tokenizer splits.
- Words outside the dictionary.

The layer is keyed like the runtime tokens: lowercase, without the
apostrophe. `м'яч`, `м’яч` and `мʼяч` therefore all find the same entry.
The stored stress is an index into the key, and the runtime maps it onto the
token as written. A quote, a soft hyphen or a missing apostrophe therefore
keeps the mark on the same letter.

With the layer, `silero` scores 0.843 on the russtress Ukrainian gold and
0.905 on the Wiktionary example sentences (`benchmarks/RESULTS.md`).
`export/build_uk_word_stress_vocab.py` rebuilds the layer from the pinned
upstream release. A rebuilt `accentor.onnx` needs a rebuilt layer too,
because the selection depends on the model's own output.

Belarusian specifics:
- **Akane**: unstressed /o/ merges phonetically with /a/. Stress is needed
  to pronounce words correctly.
- Stress is not marked in standard orthography.
- Fewer neural training resources than Russian, so the silero neural model
  is the best available quality without a full BERT pipeline. See the
  benchmark table above: silero (0.924 accuracy) is better here than the
  `simple` vocabulary fallback (0.867), and `simple` leaves every
  homograph unmarked.

### Belarusian dictionary layer

The Belarusian silero bundle carries a dictionary layer built from
[GrammarDB](https://github.com/Belarus/GrammarDB) RELEASE-202601, the
grammar database of the Belarusian language (Aleś Bułojčyk and
contributors, CC-BY-SA-4.0). The release lists about four million
inflected forms, about two million distinct spellings, with `+` after the
stressed vowel. Keyed like the runtime tokens, 1,896,721 keys have one
stress and 25,029 have several readings. The layer lives in the bundle's `dictionary.txt.gz`, a
separate file with its own notice `LICENSE.grammardb`, because the
database is CC-BY-SA-4.0 and the silero files are MIT. The runtime reads
it before the upstream exceptions table and the neural model.

The layer holds only the spellings that have exactly one stress in the
database and that the silero pipeline (exceptions table plus model)
stresses differently. The pipeline already agrees with the database on
the other 1.89 million spellings, so they are left out. This keeps the
file at about 11 KB and 2,493 entries. 184 of them override an entry of
the upstream silero exceptions table, which stays in the bundle unchanged.
2,318 keys are lowercase and 175 capitalized. 1,925 keys are not a plain
spelling of the database's word list (`spellchecker/slovy-2008.txt`):
they are apostrophe spellings (аб'ект) with the apostrophe dropped, which
the model stresses wrongly in that shape. 568 keys are plain spellings
(эканамі́ста, хлушу́, Інтэрфа́ксу).

The neural model still decides these cases:

- Homographs (му́ка/мука́, па́ра/пара́, ру́кі/рукі́) and spellings with
  variant stress, which the database lists with more than one stress.
- Spellings stressed differently inside a hyphenated compound of the
  database (ра́мбурсны against акцэ́птна-рамбу́рсны).
- Spellings that en-Wiktionary (the kaikki dump) stresses on another
  vowel, in either case of the spelling.
- Forms of paradigms the database marks `todo` (data not final); such a
  form also keeps the same spelling out of the layer.
- Hyphenated compounds and multiword entries as a whole, which the
  tokenizer splits; each part is a token of its own.
- Forms the database lists without a stress mark.
- Words outside the database.
- A capitalized word inside a sentence whose capitalized spelling the
  database lacks, and a capitalized word at the start of a sentence whose
  lowercase spelling it lacks. The layer keys a form the database writes
  capitalized (Інтэрфа́ксу) by that spelling. Inside a sentence a
  capitalized token takes only such a key, so `сталі́н` (genitive plural of
  сталі́на) does not reach `Сталін`, and the model keeps `Ста́лін`. At the
  start of a sentence, where a common word is capitalized too, it takes
  only the lowercase key, so the name `Я́на` does not reach the pronoun
  `Яна́`. The runtime takes a sentence start after any `.`, `!`, `?` or
  `…`, so also after an initial, an abbreviation, an ellipsis or a decimal
  number, and a name alone is a sentence start. An all-caps token takes
  either key. A name and a common word of one spelling that disagree give
  neither key, and so do a lowercase key and its capitalized Wiktionary
  spelling (сталі́н, Ста́лін).

Each entry stores the ordinal of the stressed vowel, not a character
offset, and the runtime maps it onto the token as written. A quote,
apostrophe or soft hyphen in the token therefore does not move the mark,
which is the defect the Ukrainian layer has with its character offsets.
The layer is keyed like the runtime tokens, with every character outside
the silero alphabet removed. That alphabet keeps `’` but not the ASCII
apostrophe, so the layer carries both keys: `з'ява`, `з’ява` and `зʼява`
all find their entry. A word in single typographic quotes (`‘слова’`)
keeps the closing `’` in its key, finds no entry and stays with the model,
as on a bundle without the layer.

On the five Belarusian evaluation sets of `benchmarks/run.py` (ser9, two
onnxruntime threads, the harness's bootstrap over rows and exact McNemar on
the targets both scored; the difference column is a paired bootstrap over
targets, 2,000 resamples):

| Set (tier) | Targets | silero | with layer | Difference [95% CI] | Layer only right / model only right | McNemar |
|---|---|---|---|---|---|---|
| russtress (GOLD) | 2,902 | 0.908 | 0.923 | +0.015 [+0.010, +0.019] | 44 / 1 | p = 3e-12 |
| wiktionary (GOLD) | 12,390 | 0.950 | 0.957 | +0.007 [+0.006, +0.009] | 92 / 3 | p = 7e-24 |
| zascienak (GOLD) | 22,965 | 0.949 | 0.960 | +0.011 [+0.010, +0.012] | 252 / 0 | p = 3e-76 |
| homographs-10x10 (GOLD) | 1,496 | 0.499 | 0.499 | 0 | 0 / 0 | n/a |
| homographs-commonvoice (SILVER) | 2,877 | 0.696 | 0.696 | 0 | 0 / 0 | n/a |

The four targets the layer loses are database readings the gold
contradicts: існуе́ (russtress існу́е), навіна́ (Wiktionary наві́на, twice)
and ву́чняў (Wiktionary вучня́ў). The homograph sets do not move because
every homograph has several readings in the database and stays with the
model. The zascienak and the homograph sets come from the BelVoice
project, whose author also maintains GrammarDB, and its card says the
stress of ordinary words is set through the grammar database; the
zascienak row is therefore not independent of the layer's source, which
its 252 against 0 shows. The russtress rows (Toloka-annotated UD treebank
text) and the Wiktionary rows are independent of GrammarDB. On 75,174
inflected forms of the English Wiktionary (kaikki, no context), which is
also the cross-check source, the layer gives 0.957 against 0.954 without
it, +0.0034 [+0.0030, +0.0039], 259 forms only right with the layer
against 0 only right without it.

The case rule was measured on the same sets against two alternatives: a
capitalized token that always takes the lowercase key when it has no
capitalized one, and one that never does. The rule gives 15 Wiktionary and
28 zascienak targets more than "never" and 3 Wiktionary targets more than
"always", and loses none against either.

`export/build_be_grammardb_vocab.py` rebuilds the layer from the pinned
release archive and the pinned kaikki dump, whose archived copy the script
names. A rebuilt `accentor.onnx` or `exceptions.txt.gz` needs a
rebuilt layer too, because the selection depends on the pipeline's own
output.

### Belarusian homograph rules

GrammarDB lists 20,539 spellings with two or more stresses (за́мак/зама́к,
ба́цькі/бацькі́, ву́чыцца/вучы́цца). The dictionary layer leaves them to
the model, which picks one reading of each pair whatever the context: on
the balanced `homographs-10x10` set it gets 75 of the 150 readings
entirely wrong. The bundle's `homographs.json.gz` (65 KB, pure Python at
runtime, `stressonnx/backends/_homograph_rules.py`) resolves such a
spelling from the two words before it, inside the same clause: a rule
never reads across `.,;:!?`, quotation marks, brackets or a dash. For
each reading the file carries
the ordinal of the stressed vowel and the grammatical tags of the
database forms that have it (part of speech, case, number, finite or
infinitive). A rule picks a reading when exactly one reading satisfies
it. The rules are tried in the order of the table, and the first answer
wins. A spelling with no rule answer takes its most frequent reading in
two stressed corpora, and a spelling with neither stays with the model.
The runtime reads the layer after the dictionary layer and before the
exceptions table. A token is looked up in lowercase, and a capitalized
token inside a sentence marks a name the layer does not know (Я́на
against the pronoun яна́), so it stays with the model, as in the
dictionary layer. A capitalized token at the start of a sentence and an
all-caps token take the layer.

| Rule | Fires when | Picks | Source |
|---|---|---|---|
| `gen_prep` | the word before is без, для, да, ад, каля, ля, пасля, акрамя, сярод, супраць, дзеля, замест or апрача | the one genitive reading | English Wiktionary, `{{+obj\|be\|gen}}` on each preposition entry |
| `num234` | the word before is два, дзве, тры, чатыры, абодва or абедзве and every reading is declined | the one nominative plural noun | [be.wikipedia.org/wiki/Лічэбнік](https://be.wikipedia.org/wiki/Лічэбнік): after два, дзве, тры, чатыры "назоўнікі ў назоўным склоне маюць форму множнага ліку" |
| `num5` | the word before is пяць … сто and every reading is declined | the one genitive plural noun | the same page: "ужываюцца назоўнікі ў форме роднага склону множнага ліку" |
| `loc` | the word before is пры; or у, ў, ва, на, аб, па | the one locative reading, or the one locative or accusative reading | English Wiktionary, `{{+obj}}` on each entry: пры `loc` only; на and аб `acc` and `loc`; у `acc`, `loc` and a genitive sense; ва no template, gloss "+ genitive, accusative or locative"; па `acc`, `loc` and `dat`. ў is the post-vocalic spelling of у. The rule answers only when exactly one reading is locative or accusative, so a genitive or dative reading is never picked |
| `acc_prep` | the word before is праз or пад | the one accusative reading | English Wiktionary, `{{+obj\|be\|acc}}` |
| `agree` | the word before is a pronoun-adjective or numeral form of GrammarDB, and a preposition before that restricts its case | the one reading that agrees in case and number | [be.wikipedia.org/wiki/Прыметнік](https://be.wikipedia.org/wiki/Прыметнік) (agreement); the case and number set of each modifier form from the GrammarDB `SA` and `M` paradigm tags (`tags.svg`) |
| `agree_ending` | as `agree`, for a word of four or more letters with an adjectival ending (-ага, -аму, -ай, -ую, -ыя, -ых, …) that is not in the modifier table | the one reading that agrees | MINED: the endings and their case and number sets were read off the adjective paradigms, no citation. Right on 37 of the 43 prior-corpus contexts it fires on |
| `subject_pronoun` | the readings are one infinitive and one finite verb form, and the word before is я, ты, ён, яна, яно, мы, вы, яны, хто (finite) or a modal such as можа, трэба, хачу, буду (infinitive) | that reading | MINED: no citation. Right on 14 of 14 prior-corpus contexts |
| prior | no rule answers | the most frequent reading of the spelling in the prior corpora, none on a tie | MINED: counts on 3,169 homograph targets of zascienak_Malinauka.txt (2,312, CC-BY-SA-4.0) and the Belarusian usage examples of English Wiktionary (857, CC-BY-SA). 733 spellings get a prior, 15 tie |

The prior corpora are two of the harness's general stress sets and not
the two homograph sets. The build pins both by sha256, and they share no
sentence with `homographs-10x10` or `homographs-commonvoice`. The
russtress rows are not a prior source, so that set is independent of the
layer. Only the
reading counts leave the corpora. A tie gives no prior, so a rare reading
is chosen only on a rule signal and the common reading is the default.

Precision of each rule on the homograph targets it decides, and the
coverage of the layer by stage (`benchmarks/homograph_readings.py` on the
harness rows):

| Stage | 10x10 (GOLD): targets, right | commonvoice (SILVER): targets, right |
|---|---|---|
| `gen_prep` | 3, 3 | 17, 17 |
| `loc` | 8, 8 | 63, 63 |
| `agree` | 28, 27 | 17, 16 |
| `agree_ending` (mined) | 84, 74 | 53, 44 |
| `subject_pronoun` (mined) | 11, 11 | 125, 122 |
| `num234`, `acc_prep`, `num5` | 0 | 0 |
| prior | 1,178, 597 | 2,371, 1,960 |
| no answer (tie or no context) | 105 | 177 |
| not in the layer | 79 | 54 |

On the five Belarusian sets of `benchmarks/run.py` (ser9, two onnxruntime
threads, paired bootstrap over rows, exact McNemar on the targets both
models scored). "Rules only" is the same file with the prior removed, run
to separate the two sources of the gain:

| Set (tier) | Targets | silero | with layer | silero only right / layer only right, McNemar | rules only | silero only / rules only right, McNemar |
|---|---|---|---|---|---|---|
| homographs-10x10 (GOLD) | 1,496 | 0.499 [0.475, 0.526] | 0.543 [0.518, 0.570] | 135 / 202, p = 3e-4 | 0.549 [0.525, 0.576] | 1 / 76, p = 1e-21 |
| homographs-commonvoice (SILVER) | 2,877 | 0.696 [0.679, 0.711] | 0.829 [0.815, 0.843] | 179 / 561, p = 1e-46 | 0.754 [0.738, 0.768] | 9 / 176, p = 3e-41 |
| russtress (GOLD, independent) | 2,902 | 0.923 [0.914, 0.932] | 0.924 [0.914, 0.932] | 5 / 6, p = 1 | 0.924 | 0 / 1 |
| wiktionary (GOLD, prior source) | 12,390 | 0.957 [0.953, 0.962] | 0.968 [0.964, 0.972] | 16 / 145, p = 4e-27 | 0.958 [0.955, 0.962] | 4 / 15, p = 0.019 |
| zascienak (GOLD, prior source) | 22,965 | 0.960 [0.957, 0.963] | 0.976 [0.974, 0.978] | 24 / 392, p = 7e-87 | 0.962 [0.959, 0.964] | 7 / 40, p = 1e-6 |

Readings entirely wrong on 10x10: 75 of 150 for silero, 58 with the layer,
58 with the rules only. On commonvoice: 34 of 82, 26 and 25. The rules are
the gain on the balanced gold set: on the 1,178 targets the prior decides
there it is right on 597 where silero alone is right on 605, so the prior
neither helps nor hurts that set, and its wins and losses come in tens per
word and cancel. The prior is the gain on the silver set and on the two
general sets that are its own sources. Those two rows measure agreement
with the counts, not an independent gain, and the rules-only column is
the independent part of each. On the silver set 103 of the layer's 561
wins are one spelling, складзе (скла́дзе in "у складзе", the locative of
склад, against the verb складзе́), which the prior decides where the
sentence writes ў and the `loc` rule where it writes у. On russtress,
which is independent of the layer, the layer wins 6 targets and loses 5.
No gold set loses: the layer's losses on the general sets are 5, 16 and
24 targets against 6, 145 and 392 wins. Per-reading tables for every
model are in the evidence of the build. The layer costs 19 to 47 µs per
homograph token on ser9, under 0.08 ms per row on every set, within the
run-to-run noise of the model's own 0.2 to 5 ms per row.

`export/build_be_homographs.py` rebuilds the file from the pinned release
archive, the pinned kaikki dump and the pinned zascienak file, checks
every sha256, prints the forms in and the keys out, and writes the same
bytes on every run. A spelling whose readings all carry one tag set gives
no rule a choice and is kept only with a prior (12,223 dropped), so the
file holds 8,363 keys.

### Why silero is also available for Russian

The silero Russian model is accurate for unambiguous words and resolves
homographs from a short context window. It is a valid alternative when:
- The about-470 MB ruaccent download is impractical.
- Fast cold-start is required (silero ru downloads about 50 MB).

### Limitations

- **Homograph resolution covers the homosolver dictionary only** (ru).
  Other words get one-best prediction per word.
- **Word-level context only** outside homographs. The n-gram features are
  character-level and have no access to surrounding words.
- **Long input costs time per homograph.** Each homograph that no phrase
  rule settles is one BERT row. A 60,000-character text with 1,416 such
  homographs takes about 14 s on two CPU threads, with about 300 MB peak
  memory; the rows run in chunks of 32.
- **Vocabulary coverage** is finite. Neologisms and foreign words rely on
  n-gram generalisation, which degrades for non-Cyrillic stems.
- **The Ukrainian and Belarusian models are verified to match** the
  original silero outputs exactly (see `export/verify_e2e.py`), except on
  the Ukrainian dictionary layer, where the dictionary wins by design. The
  Russian model is verified numerically at export: the largest difference
  from the upstream logits is below 3e-5 of the largest logit for the stress
  and ё heads. For the homosolver it is below 1e-4 of the largest logit (or
  of 1, when all logits are smaller), and every logit keeps its sign.

---

## `"stressrnn"`: character BiLSTM

**Available for `ru`. Never a default, never chosen by `prefer=`.**

A port of [StressRNN](https://github.com/dbklim/StressRNN) (Vladislav
Klim, Apache-2.0), itself a maintained fork of
[RusStress](https://github.com/MashaPo/russtress) (Ponomareva et al. 2017,
*Automated Word Stress Detection in Russian*). The ONNX graph is the
upstream file as shipped: a bidirectional LSTM (64 units each way) over a
one-hot encoding of the word's letters, 40 positions by 36 symbols, with a
softmax over the 40 positions. Before the network, a word is looked up in
an exception dictionary of 91 851 words (92 276 lines) built from A. A. Zaliznyak's
Grammatical Dictionary, and a word with one vowel is stressed on it.

```
Input text
      │
      ├── tokens: runs of letters, digits, spaces, punctuation (every character kept)
      │
      └── words: lowercase, clause marks `_` at .,:?!…- and newlines,
                 each 2+-vowel word prefixed with the last 3 letters of the previous word
                        │
          ┌─────────────┼──────────────────┐
     1 vowel        in exception dict      else: one-hot [40 × 36] → model.onnx (BiLSTM)
     → that vowel   → dictionary index     → argmax position, kept when p ≥ 0.75
                        │
                        ▼
          stressed words transferred back onto the tokens in order → U+0301
```

The only context the network sees is the three-letter ending of the previous
word, so homographs get one reading (`за́мок` always). There is no е→ё
restoration. The threshold is upstream's default: a word the network is
less than 75 % sure about stays unmarked; `StressRnnStressor(accuracy_threshold=0.0)`
marks every word. The scoreboard measures the default threshold.

Differences from upstream `put_stress`, each a case upstream mishandles:
only U+0301 counts as an existing stress mark (upstream also reads `+` and
`'`), any whitespace separates words (a tab makes upstream leave the rest
of the text unmarked), a prediction that lands in the padding before the
word is read on the vowel it wraps to (upstream prints the mark before the
first letter), and a word longer than 40 characters with its context is
left unmarked (upstream indexes outside the input tensor). The upstream
option `lemmatize_words` is not ported: it is off by default, it only
changes the dictionary lookup, and it is the only use of pymorphy2 in
upstream. The network never sees morphological features, so stressonnx
needs no morphology dependency for this model.

### Limitations

- **Monosyllables, prepositions and particles are stressed**, as are both
  parts of every hyphenated word: `кто́-то́`, `на́ го́ре`. For TTS this is
  audible; `ruaccent` and `silero` mask these.
- **No homograph or ё resolution.** One reading per spelling.
- **Russian alphabet only.** A letter outside `а-яё` splits the word; the
  parts are stressed as separate words.
- **Below `silero`** on the scoreboard (0.892 vs 0.944 on russtress) and
  those rows are the corpus the BiLSTM was trained on, so the gap is a
  lower bound.

---

## `"uk-accentor"`: seq2seq Transformer

**Available for `uk`. Never a default, never chosen by `prefer=`.**

A port of [ukrainian-accentor-transformer](https://github.com/Theodotus1243/ukrainian-accentor-transformer)
(Bohdan Mykhailenko, MIT). Upstream trained a 4+4-layer Transformer
(d 256, 8 heads, ffn 1024) with a shared SentencePiece vocabulary of 4095
pieces on the lang-uk News corpus, stressed with
[ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress),
and ships it only as a CTranslate2 binary. `export/export_uk_accentor.py`
reads that binary, rebuilds the network with CTranslate2's arithmetic and
exports an encoder graph and a decoder graph; the SentencePiece model
becomes a `tokenizer.json`. The model translates a sentence into the same
sentence with U+0301 after each stressed vowel.

```
Input text
      │
      ├── U+0301 stripped, SentencePiece pieces (tokenizer.json)
      │
      ├── cut after every . , ! ? piece; chunks of 30+ pieces halved at a word start
      │
      ├── per chunk: encoder.onnx → memory; decoder.onnx step by step,
      │              beam 2, repetition penalty 1.2, length penalty 1 (numpy)
      │
      ├── hypotheses joined and decoded to text
      │
      └── difflib alignment with the stripped input: a U+0301 the model
          inserted marks the input letter before it; every other
          difference is dropped → marks rendered onto the input
```

The model sees the whole chunk, so it can use context, and it marks what it
was trained on: some words get two marks (`за́мо́к`, `ро́кі́в`), some stay
unmarked, most monosyllables stay unmarked. The model also writes a mark
after a consonant on some words (`хр́е́ще́ною́`); the port drops those,
because a stressed index is the offset of a vowel.

Parity with upstream: the output is identical except for marks after a
consonant, which the port drops: 996 of the 1009 sentences in upstream's
own test file and 12230 of the 12253 russtress windows are identical
(upstream run with its own code on the same NFC text; 13 sentences and 21
windows differ by a dropped consonant mark). The other differences are
the two rules below.

Differences from upstream `Accentor.__call__`: the port works on the NFC
form of the text and maps offsets back, where upstream aligns its output
with the raw text and loses the marks of a word spelled with a combining
mark (`и` + U+0306 for `й`); and a word that already carries a stress mark
keeps it, where upstream strips every mark and writes its own. The port
never changes a letter: the output is the input with marks inserted at
offsets, so a letter the model rewrote is dropped, as upstream does with
its diff. Upstream decodes up to 256 pieces per chunk whatever the input;
the port stops at 2 × (input pieces) + 20. The bound cuts short only the
inputs the model never ends with an end token (`и`, `100`), and their
output is the same.

### Limitations

- **Training labels come from ukrainian-word-stress**, the dictionary that
  also feeds the `uk` dictionary layer proposed for the `silero` model.
  The russtress Ukrainian rows are the only gold available and share that
  lineage, so the scoreboard number is a ceiling, not an independent
  measurement. The ukrainian-word-stress labels are MIT; the lang-uk News
  corpus is a row of the UberText section of
  [lang.org.ua](https://lang.org.ua/en/corpora/), whose page states that
  the texts are distributed on condition of Fair Use for statistical and
  scientific analysis, and the row carries no licence field. `meta.json`
  records both as `training_data_license`.
- **Slow.** Beam search runs the decoder graph once per output piece and
  beam, and reads the whole prefix at every step. Median 123 ms per
  russtress window of three words and 710 ms per sentence of upstream's
  test file (median 98 characters), against silero's 0.2 ms per window on
  the same box (24 cores under load, 2 threads).
- **Double marks and unmarked words.** One reading per chunk, as the model
  wrote it; no post-processing picks a single vowel.
- **Ukrainian alphabet only.** A letter outside the SentencePiece vocabulary
  becomes `<unk>` and the word around it is usually left unmarked.

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
pronunciation dictionary (MIT). Free-stress languages (bg, sl, and
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
  [silero_stress](https://github.com/snakers4/silero-stress) (MIT) and
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
lookup without neural inference, a lower-accuracy fallback (0.867 vs.
`silero`'s 0.924, see the benchmark table above) meant for offline
environments where ONNX inference is unavailable, or as an explicit
fallback target. Its OOV rule is `"none"`, but that rule only applies to
**multi-vowel** OOV words: a **single-vowel** OOV word is always stressed
on that vowel (the "always stress monosyllables" rule takes precedence
over the per-language OOV rule, matching upstream `SimpleAccentor`
behavior). A multi-vowel word absent from the vocabulary is left with no
mark at all.

---

## Architecture comparison

| | ruaccent | silero | simple | kubataba |
|---|---|---|---|---|
| **Languages** | ru | uk, be, ru | 26 languages | ru (opt-in) |
| **Context** | Sentence (BERT) | Word (n-gram) | Word (dict) | Sentence (char seq2seq) |
| **Homograph resolution** | Yes (BERT + RoBERTa) | ru only (BERT) | No | No |
| **е→ё restoration (ru)** | Yes | Yes | N/A | No |
| **Yo-homograph (все/всё) disambiguation** | Yes | ru only | N/A | No |
| **ONNX inference** | 4 models | 1 model (2 for ru) | None | 2 graphs, autoregressive |
| **Download size** | ~470 MB | ~10 MB (ru ~50 MB) | ~1-5 MB (ru 10 MB) | ~31 MB |
| **Cold-start latency** | High (4 sessions) | Low | Near-zero (ru about 15 s, 3.2M entries) | Low |
| **Per-sentence latency** | ~50-200 ms | ~2-10 ms | <1 ms | ~150 ms per 13-character window |
| **Runtime deps** | onnxruntime, tokenizers | onnxruntime, numpy | numpy only | onnxruntime, numpy |
| **OOV handling** | nn_accent (char-level) | n-gram generalisation | positional rule | char-level generation |
| **Sentence-level input** | Required | Works word-by-word | Works word-by-word | Required |
| **Max input length** | ~512 subword tokens | Unlimited (per-word) | Unlimited | Unlimited (200-character windows) |
| **Idempotent on pre-marked text** | No (re-derives) | Yes (skips) | Yes (skips) | Yes (skips) |
| **License** | MIT | MIT | MIT | MIT |

`prefer="best"` picks `ruaccent-silero` for `ru`. Among the models a
language has, `prefer="fast"` tries `silero`, then `simple`, then `ruaccent`,
and `prefer="smallest"` tries `simple`, then `silero`, then `ruaccent`. No strategy picks `kubataba` or `stressrnn` (Russian, 0.6 MB,
onnxruntime + numpy, Apache-2.0): each is reached only by its `model=` id. `uk-accentor` (Ukrainian, 45 MB, onnxruntime + numpy + tokenizers,
MIT) is reached only by `model="uk-accentor"` in the same way.

## `"kubataba"`: character-level seq2seq Transformer (Russian, opt-in)

A port of [Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor)
(kubataba, MIT; PyPI `ruaccent-predictor`): an encoder-decoder Transformer
(4+4 layers, d_model 256, 12.5M parameters) over a 224-character vocabulary.
It reads a sentence as characters and generates a copy with a mark after
each stressed vowel.

### Training data

The upstream model was trained on 224k sentence pairs cut from
[nevmenandr/accentual-syllabic-verse-in-russian-prose](https://huggingface.co/datasets/nevmenandr/accentual-syllabic-verse-in-russian-prose),
462 texts of 19th-century Russian fiction prose. The stress marks of that
corpus were produced by the `ru-accent-poet` package, not by annotators, so
the training labels are silver. The corpus and the russtress benchmark rows
(UD-treebank windows with Toloka annotation) are different sources; verbatim
overlap, measured with accents stripped and lower-cased, is 1.9% of the
benchmark windows of 20 or more characters (5 of 269), 4.2% at 16 or more
and 10.0% at 12 or more, mostly short generic phrases.

### Accuracy

`benchmarks/RESULTS.md` scores `kubataba` on russtress, HomographResolutionEval
and prose-accents. The two large sets (Wiktionary, ruslan-stressed) do not
run it, because its cost grows with the square of the sentence length and
those sets would take about 10 and 14 hours; their tables say so. On the
11124 russtress targets `kubataba` scores 0.888, `silero` 0.944 and
`ruaccent` 0.947 (McNemar p = 7e-91 and p = 3e-105). On prose-accents it
scores 0.968, above every other model, but that set is cut from the corpus
it was trained on, so the number is agreement with its own training labels.
On HomographResolutionEval it scores 0.583.

### Runtime

Two ONNX graphs exported by `export/export_kubataba.py`: the encoder and one
decoder step. Greedy decoding runs in numpy with one decoder call per output
character and no key-value cache, so cost grows with the square of the
window length. One process with two onnxruntime threads on a 24-core host
at load about 25 takes about 150 ms per russtress row (13 characters on
average) and 180 MB peak RSS. Download: about 31 MB.

The generated copy is aligned back onto the input and only a mark on a
vowel the input has at that place is kept; every input character comes
back. Text over 200 characters is cut into windows on any whitespace. The
ONNX port generates the same marked string as the upstream torch model on
all 1011 inputs checked (1000 russtress windows plus the test sentences).

### Limitations

- No homograph resolution and no е→ё restoration; ё and single-vowel words
  are never marked (upstream behaviour).
- The vocabulary holds the Cyrillic and ASCII letters, the digits and basic
  punctuation. A character outside it (`%`, emoji, most other symbols) is
  replaced or dropped by the model; the backend keeps the input character
  and loses only the marks the model could not place around it.
- On a short window the model can append invented words. Their marks are
  dropped in alignment: a run of one matching character is ignored.
- When the model drops one of two adjacent identical vowels (`нии` to
  `ни'`), the mark can land on either twin.
- Accuracy is below `ruaccent` and `silero`, so it is never selected by
  default, by `prefer=` or by `fallback=True`.

See [`../benchmarks/RESULTS.md`](../benchmarks/RESULTS.md) for measured
accuracy numbers instead of qualitative claims.

---
[← Languages](languages.md) · [Home](../README.md) · [Architecture →](architecture.md)
