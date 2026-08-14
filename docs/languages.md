# Language guide: why each language needs stress marking

This page explains, language by language, why a text-to-speech system needs
stress information before it can pronounce text, what kind of stress system
the language has, and exactly what stressonnx does about it. It has three
layers: the **Why** paragraphs assume no linguistics background, the **What
stressonnx does** paragraphs are for developers, and the citations and
typological notes are for linguists who want to check our homework.

Quality numbers quoted below come from the committed, reproducible
[benchmark scoreboard](../benchmarks/RESULTS.md). Rule sources are also in
the code itself: every out-of-vocabulary rule is a named function with its
citation in `stressonnx/backends/simple.py`.

## A two-minute primer (skip if you know what lexical stress is)

Every word in the languages below has one syllable pronounced more
prominently than the others: its **lexical stress** (louder, longer, or
higher-pitched depending on the language). Three facts make it matter for
speech synthesis:

1. **Stress can distinguish words.** Russian за́мок "castle" vs замо́к
   "lock" are different words that happen to be spelled the same
   (**homographs**). Read the wrong one aloud and you said the wrong word.
2. **Stress can change the vowels themselves.** In Russian or Bulgarian,
   an unstressed о is not pronounced [o]. It *reduces* toward [a] or [ə].
   A TTS system that guesses stress wrong does not just get the rhythm
   wrong. Every vowel in the word comes out wrong.
3. **Most orthographies do not write it.** Spanish writes `á` where the
   rules demand it. Russian, Ukrainian, Bulgarian and most languages here
   write nothing. The reader, human or machine, is expected to know.

stressonnx makes the implicit explicit: it inserts the combining acute
accent (U+0301) after the stressed vowel, producing text any downstream
grapheme-to-phoneme system can pronounce correctly.

How hard that is depends entirely on the language's **stress system**:

| System | Meaning | Consequence for TTS |
|---|---|---|
| **Free / lexical** | stress is a property of each word, not predictable from its shape (Russian, Bulgarian, and others) | you need a dictionary, a model, or both |
| **Fixed** | stress always falls on the same position (Latvian: first syllable; Macedonian: antepenultimate) | a one-line rule covers almost everything |
| **Conditioned** | position is predictable from the word's phonology (Chuvash: last non-reduced vowel) | a slightly smarter rule |
| **Written** | the orthography already encodes stress (Spanish, Greek) | you do not need this library, see the README's [bifonia](https://github.com/TigreGotico/bifonia) note for the interesting exception |

---

## East Slavic

### Russian (`ru`)

**Why.** The hardest case in this library, on all three axes. Stress is
free and *mobile*: it moves within a word's inflections (рука́ "hand",
ру́ку "hand, accusative"). Unstressed vowels reduce heavily (акание/иканье:
unstressed о → [ɐ]/[ə], е/я → [ɪ]), so stress placement decides the entire
vowel skeleton of the word: молоко́ is [məlɐˈko], not [moloko]. Russian is
also rich in stress homographs, за́мок/замо́к, му́ка/мука́ "torment/flour",
бе́лки/белки́ "squirrels/proteins", that only sentence context resolves. An
extra twist is **ё**: the letter is always stressed, but writers usually
type it as е, so restoring ё and placing stress are the same problem.

**What stressonnx does.** Three models plus a dictionary path:

- `ruaccent` (default): four ONNX models derived from
  [RUAccent](https://github.com/Den4ikAI/ruaccent): a stress-usage
  classifier, an NLI homograph resolver scored against sentence context, a
  ё-restoration classifier, and a character-level accent model for unknown
  words. Measured: **0.938** word accuracy, **0.820** on the homograph
  subset.
- `silero`: a fast embedding-bag MLP exported from
  [silero_stress](https://github.com/snakers4/silero-stress). It restores
  unambiguous ё but deliberately leaves ё-homographs (все/всё) untouched.
  **0.914** (homographs 0.381, since it cannot use context).
- `simple` (dictionary path, `model="simple"`): 108,972 unambiguous entries
  from the RUAccent pronunciation dictionary, with no guessing on unknown
  words (rule `none`), because Russian stress placement genuinely cannot be
  predicted from word shape. The East Slavic accentological literature is
  unanimous that surface stress is unpredictable without paradigm/accent-class
  information.

**For linguists.** The hyphenated enclitic particles -то, -нибудь, -либо,
-таки, -ка are masked from stress assignment (Русская грамматика, АН СССР
1980). Monosyllable treatment differs by backend (documented in
[models.md](models.md)). `ruaccent` re-derives stress from scratch on
pre-marked input rather than trusting it.

### Ukrainian (`uk`)

**Why.** Free, mobile stress like Russian, and productive homograph pairs
(за́мок/замо́к exists here too; обі́д "lunch" vs о́бід "rim"). Ukrainian
vowels reduce much less than Russian ones, so a stress error costs you the
prosody and the homographs rather than the whole vowel skeleton, still
enough to make synthesis sound wrong or say the wrong word.

**What stressonnx does.** `silero` neural model (default), measured
**0.785** on crowdsourced UD-treebank gold, the weakest neural model in the
library. The export is faithful (parity-locked to upstream); the ceiling is
the model itself. `model="simple"` adds a 49,809-word dictionary path built
from stress-marked English-Wiktionary headwords, rule `none` (free stress,
so no positional guess is defensible).

### Belarusian (`be`)

**Why.** Free stress, with an interesting orthographic property: Belarusian
*writes* its vowel reduction (акание is spelled out, вада́ not вода́), so
text already tells you how unstressed vowels sound. What it does not tell
you is *which* syllable is stressed, and since the spelling of a word
changes with stress position, TTS still needs the stress to read prosody
and homographs correctly.

**What stressonnx does.** `silero` neural model (default), **0.873**.
`model="simple"` is the vocabulary-only fallback (**0.433**, the measured
cost of dropping the neural model, published so the trade-off is explicit).
Rule `none`: the accentological literature treats Belarusian surface stress
as lexically governed, so an unknown multi-vowel word is left unmarked
rather than guessed.

---

## South Slavic

### Bulgarian (`bg`)

**Why.** Free, mobile stress plus Russian-grade vowel reduction: unstressed
а and ъ merge, unstressed о raises toward [u], unstressed е toward [i]
(Scatton 1984, *A Reference Grammar of Modern Bulgarian*). Stress is also
contrastive: въ́лна "wool" vs вълна́ "wave", па́ра "steam" vs пара́ "coin".
Bulgarian TTS without a stress source mispronounces most polysyllabic words.

**What stressonnx does.** A 46,914-word vocabulary extracted from
stress-marked English-Wiktionary headwords (kaikki.org extraction; CC BY-SA,
attribution shipped with the data). Rule `none`: dictionary hit or no
mark, never a positional guess. There is no neural model yet. The
vocabulary covers the standard lexicon and inflection tables Wiktionary
marks.

### Macedonian (`mk`)

**Why.** The mirror image of Bulgarian: stress is **fixed** on the
antepenultimate syllable (third from the end) in words of three or more
syllables, initial in shorter words (Friedman 2001, "Macedonian"), and
vowels do not reduce. TTS mostly needs the rule for natural rhythm, plus a
list of the exceptions, which are mainly recent loanwords (клише́,
резиме́) and certain adverbs.

**What stressonnx does.** The `antepenult` rule implements the default. A
1,667-word vocabulary carries exactly the exceptions: English Wiktionary
marks stress on Macedonian words only when it deviates from the rule, so
the extraction is an exceptions list by construction. (This is why the
scoreboard's rule-vs-own-vocabulary number for mk is low: the vocabulary
contains only the words the rule is not supposed to handle.)

### Slovene (`sl`)

**Why.** Free stress, and more: standard descriptions give Slovene a
**pitch-accent** system where the stressed syllable also carries a tonal
contrast, and the stressed mid vowels distinguish open/close quality
(é [eː] vs ê [ɛ]), none of it written in ordinary text (Herrity 2000,
*Slovene: A Comprehensive Grammar*). Dictionary headwords mark all of this
with tonal diacritics; running text marks nothing.

**What stressonnx does.** A 4,421-word vocabulary from Wiktionary's
tonally-marked headwords: any tonal mark identifies the stressed vowel, and
stressonnx records the position. The tonal and quality distinctions
themselves are out of scope: this library marks *where* stress is, not
which tone it carries. Rule `none` for unknown words.

---

## Baltic

### Latvian (`lv`)

**Why.** Stress is **fixed on the first syllable** with a small set of
exceptions, mostly borrowings and a few native adverbs (Nau 1998, *Latvian*,
Lincom). Like Slovene, Latvian also has syllable tones that dictionaries
mark and text does not, again out of scope; position is what TTS pipelines
consume. The practical value here is prosodic naturalness and the exception
list.

**What stressonnx does.** Rule `first`, plus a 107-word exceptions
vocabulary from Wiktionary (the entries Wiktionary bothers to mark are, as
with Macedonian, the ones that break the rule).

---

## Turkic

A family-wide generalization first, because it drives every entry below:
Turkic word stress is **final-syllable by default and moves rightward with
suffixation** (Johanson 1998; Kirchner 1998): китап → китапла́р →
китапларыбы́з. The systematic exceptions are also family-wide: negation
suffixes, question particles, copular person endings, and many enclitics are
**unstressable**, pushing stress onto the syllable before them. Imperatives
and some interrogatives stress the first syllable in several languages.
Vowel quality is stable under stress shift, so the cost of an error is
rhythm and the occasional exception word, much lower stakes than Russian,
which is why a rule plus an exception dictionary is a defensible design.

### Kazakh (`kk`), Kyrgyz (`ky`), Tatar (`tt`), Bashkir (`ba`), Azerbaijani (`az-Cyrl`/`az-Latn`), Uzbek (`uz-Cyrl`/`uz-Latn`)

**What stressonnx does.** Curated vocabularies exported from silero_stress
(covering common words including the suffix-class exceptions) plus the
`last` rule for unknown words. Measured rule-vs-vocabulary accuracy: ky
0.993, kk 0.987, tt 0.896, ba 0.852, az 0.816, uz 1.000. The spread
reflects how many exception forms each vocabulary happens to contain.
The closed unstressable-suffix lists (Kazakh -ма/-ме negation, -шА, -ДАй,
copulas, and others; Kirchner 1998 via Washington 2006) were measured
against the vocabularies: blind surface-string retraction loses more than
it gains (алма́ "apple" would wrongly become а́лма). Correct application
needs morphological segmentation, so they are documented rather than
enabled. Tatar is the exception: four suffix variants (-ча/-чә adverbial,
-сең 2sg, -ме interrogative; Comrie 1997b) pass a 0.7-or-higher
vocabulary-evidence bar and are enabled in the `tt` rule (0.896 → 0.904).

**For linguists.** Azerbaijani imperative/negation initial stress is
attested in the JIPA Illustration (Ghaffarvand Mokari & Werner 2017). Uzbek
may share Uyghur's syllable-weight sensitivity (Comrie 1997c), unconfirmed,
flagged rather than modeled.

### Chuvash (`cv`)

**Why.** The famous case of **phonologically conditioned** stress: Chuvash
has two reduced vowels, ӑ and ӗ, that can never carry stress. Stress falls
on the **last full vowel**. If every vowel in the word is reduced, it falls
on the first syllable (Clark 1998; Krueger 1961; instrumentally confirmed by
Dobrovolsky 1999, ICPhS). A naive "stress the last vowel" rule is wrong for
every word ending in a reduced-vowel suffix, a very large share of running
text.

**What stressonnx does.** The dedicated `cv` rule implements the
full-vowel scan, measured **0.94** against the 22,050-word vocabulary,
versus 0.60 for the naive final-stress rule. This is the single largest
rule improvement in the library.

### Yakut/Sakha (`sah`)

**Why.** Yakut writes its long vowels and diphthongs as vowel digraphs
(аа, ыы, уо, иэ, and others) and they **attract stress**: бии́р, буо́лан,
эрээ́ри, overriding the plain final-stress default the older literature
leads with (Krueger 1962 notes the default plus exceptions; the
weight-sensitivity is overwhelming in the 85,746-word vocabulary). A TTS
system using naive final stress mis-stresses most polysyllabic Yakut words.

**What stressonnx does.** The `sah` rule: stress the last long
vowel/diphthong (marking its second element, matching the vocabulary
convention), else the final vowel, **0.994** against the vocabulary,
versus 0.178 for naive final stress.

### Khakas (`kjh`)

**Honest entry.** No primary academic description of Khakas stress could
be located at all (neither StressTyp nor WALS carries an entry; Baskakov
1975 and Anderson's grammar were not accessible). The Turkic-default `last`
rule is an explicitly labeled extrapolation, not a sourced rule.

---

## Iranian

### Tajik (`tg`)

**Why.** Persian-type final stress with one high-value exception that is
**syntactic**: the izafet enclitic -и (linking nouns to modifiers: китоби
ман "my book") is unstressed, and Tajik orthography distinguishes it from
stressed word-final ӣ precisely because readers need to know (Perry 2005,
*A Tajik Persian Reference Grammar* §1.6). Getting izafet stress wrong
makes noun phrases sound broken.

**What stressonnx does.** The `tg` rule: final vowel, except word-final
-и → penult. 0.454 → **0.736** against the vocabulary. Verbal morphology
(stress-attracting negation на-?) is flagged unconfirmed in the sources and
not modeled.

---

## Uralic

### Erzya (`myv`) and Moksha (`mdf`)

**Why.** Instrumental studies (Lehiste et al. 2003 for Erzya; Aasmäe et
al. 2013 for Moksha, both via Hamari & Ajanki 2022, *The Oxford Guide to the
Uralic Languages* §23.2.3) find first-syllable stress as the dominant
*tendency* in both languages, but genuinely free variation exists, partly
governed by sentence rhythm. Moksha also has a sonority effect: stress moves
off an initial high vowel (и/у) to a following syllable with /a/.

**What stressonnx does.** Rule `first` for Erzya (0.538, near the honest
ceiling for a positional rule in a free-variation system). The `mdf` rule
adds the documented high-vowel retraction (0.716 → 0.761).

### Udmurt (`udm`)

**Why.** Fixed **final** stress (Winkler 2001) with morphosyntactic
exceptions: negated verbs and imperatives stress the initial syllable
(Edygarova 2015). The exceptions need morphology to detect and are not yet
modeled. The default alone measures 0.975.

---

## Caucasus

### Armenian, Eastern (`hy`)

**Why.** Stress falls on the **last non-schwa syllable**: the vowel ը
(schwa) can never be stressed, so final-schwa words, including everything
carrying the definite article -ը, retract stress leftward (Chakmakjian
2024, Speech Prosody: "stress occurs within the last non-schwa syllable").
A naive final rule mis-stresses every definite noun.

**What stressonnx does.** The `hy` rule (final non-schwa scan), 0.743 →
**0.777**, plus the 8,537-word vocabulary.

### Georgian (`ka`)

**Why, and a caveat.** Georgian word stress is **weak and contested**:
the descriptive literature disagrees three ways (initial: Tschenkeli 1958,
Tevdoradze 1978, and most recently Borise 2020 with instrumental support;
antepenultimate: Akhvlediani 1949, Gudava 1969; penultimate: Zhghenti 1958).
There are no stress minimal pairs, and native speakers report no stable
intuitions. For TTS this means stress placement is low-stakes, but a
consistent choice still sounds better than noise.

**What stressonnx does.** The `antepenult` rule (antepenultimate, initial
for short words), chosen because the curated 12,473-word vocabulary follows
that tradition on **100%** of its entries, making rule and dictionary
mutually consistent. The contested status is documented rather than hidden.

### Kabardian (`kbd`)

**Why.** Northwest Caucasian, with dense consonant clusters and a
two-vowel vertical system. Descriptions (Colarusso 1992; the Jaimoukha
grammar compilation) give final stress with retraction to the penult in
words ending in the schwa letter э. Source quality is the weakest in this
library (paraphrase-level only), and the measured rule accuracy (0.404)
reflects that. Treat kbd as functional but low-confidence.

---

## Mongolic

### Kalmyk (`xal`)

**Why.** Final-syllable stress on the last full vowel is the working
consensus (Bläsing's *Kalmuck* chapter), with the literature itself warning
that Kalmyk stress "has barely been studied." The `last` rule measures
1.000 against the vocabulary, the cleanest fit in the library, but the
thinness of the underlying scholarship is worth knowing.

---

## What "quality" means here, and its limits

- Neural-model numbers (ru/uk/be) are word-level accuracy against
  independent crowdsourced gold with a documented annotation-noise ceiling.
  See the scoreboard's note before quoting absolutes.
- Rule numbers are rule-vs-own-vocabulary accuracy on multi-vowel words.
  They measure how well the *rule* would serve out-of-vocabulary words, and
  are meaningless for exceptions-only vocabularies (mk, lv, documented).
- Languages without any independent gold are locked to their upstream
  reference implementation by CI parity tests instead.

If you can contribute a stressed lexicon or an evaluation set for any of
these languages, university pronunciation dictionaries are ideal, see
[`export/ADDING_A_LANGUAGE.md`](../export/ADDING_A_LANGUAGE.md).

---
[Home](../README.md) · [Models →](models.md)
