<!-- run.py:begin -->
# Benchmark scoreboard

Run date: 2026-11-05.  Reproduce: `python benchmarks/run.py --date <date>`;
languages can run as separate processes (`--lang`), see the script docstring.

Each table scores the stressonnx models on one evaluation set.  A *target*
is one word whose stress the set marks; it is a hit when the model marks
exactly that vowel and nothing else in the word.  Words where ё carries
the stress and the model leaves ё unmarked count as marked on ё.  A target
the model leaves unmarked is a miss, so a low score can mean abstention.
`95% CI` is a paired bootstrap over rows (1000 resamples, seed 0).
`homograph acc` scores the targets in the RUAccent omograph roster (ru).
`dropped` counts targets lost because the model changed the letters of
the row, so its output no longer aligns with the input.  `ms/row` is warm
latency per model call with 2 onnxruntime thread(s) per session.

**Label tiers.**  GOLD: labels written by people.  SILVER: labels made
by a machine (an LLM, a dictionary tool, a context model) or by an
undocumented method.  Silver rows show agreement with that labeller, not
correctness, and no claim that one model is best rests on them.

The training data of silero and ruaccent is unpublished, so overlap with
these sets cannot be established.

## ru

Leader on each GOLD set, against the runner-up:

- russtress: ruaccent 0.947 vs silero 0.918, McNemar p = 3e-50.
- wiktionary: ruaccent 0.964 vs silero 0.948, McNemar p = 3e-115.

### ru / russtress — GOLD

Source: https://github.com/MashaPo/russtress (none stated).  Scored: middle word of each window.
Rows in: 11124; rows scored: 11124; targets: 11124.
Labels agree with the RUAccent accent dictionary, unambiguous entries on 88.9% of 1749 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | GOLD | 11124 | 0 | 0.947 | 0.943–0.951 | 0.833 (724) | 11.2 | 1.1 |
| silero | GOLD | 11124 | 0 | 0.918 | 0.913–0.923 | 0.381 (724) | 0.2 | 0.3 |
| simple | GOLD | 11124 | 0 | 0.142 | 0.135–0.149 | 0.227 (724) | 0.0 | 0.1 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs silero: 11124 common targets, ruaccent only right 406, silero only right 87, p = 3e-50.
- ruaccent vs simple: 11124 common targets, ruaccent only right 8995, simple only right 37, p < 1e-300.
- silero vs simple: 11124 common targets, silero only right 8734, simple only right 95, p < 1e-300.

### ru / wiktionary — GOLD

Source: https://kaikki.org/dictionary/Russian/ (CC-BY-SA).  Scored: marked polysyllabic words.
Rows in: 26494; rows scored: 24419; targets: 137221.
Dropped while loading: Latin letters: transliteration or gloss (rows): 378; acute not on a vowel (targets): 14; example of an affix entry (rows): 1583; monosyllabic token (targets): 828; no usable target (rows): 114; token carries 2+ acutes (targets): 532.
Labels agree with the RUAccent accent dictionary, unambiguous entries on 92.0% of 25918 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | GOLD | 137221 | 0 | 0.964 | 0.962–0.966 | 0.926 (9410) | 112.5 | 1.6 |
| silero | GOLD | 137221 | 0 | 0.948 | 0.946–0.949 | 0.402 (9410) | 1.0 | 0.3 |
| simple | GOLD | 137221 | 0 | 0.180 | 0.177–0.182 | 0.264 (9410) | 0.2 | 0.1 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs silero: 137221 common targets, ruaccent only right 6155, silero only right 3878, p = 3e-115.
- ruaccent vs simple: 137221 common targets, ruaccent only right 108841, simple only right 1202, p < 1e-300.
- silero vs simple: 137221 common targets, silero only right 106792, simple only right 1430, p < 1e-300.

### ru / homograph-eval — SILVER

Source: https://huggingface.co/datasets/inkoziev/HomographResolutionEval (CC-BY-4.0).  Scored: the homograph only.
Rows in: 1741; rows scored: 1727; targets: 1727.
Dropped while loading: homograph field has no single acute (rows): 5; homograph not found in the sentence (rows): 7; homograph occurs 2+ times (rows): 2.
Labels agree with the RUAccent accent dictionary, unambiguous entries on 67.0% of 896 targets the dictionary covers.
Every target is a homograph, so a one-stress dictionary disagrees by design; this check does not measure label noise.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | SILVER | 1727 | 0 | 0.888 | 0.872–0.902 | 0.896 (1660) | 106.4 | 1.4 |
| silero | SILVER | 1727 | 0 | 0.223 | 0.203–0.241 | 0.197 (1660) | 0.4 | 0.2 |
| simple | SILVER | 1727 | 0 | 0.347 | 0.324–0.369 | 0.351 (1660) | 0.1 | 0.1 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs silero: 1727 common targets, ruaccent only right 1201, silero only right 52, p = 7e-285.
- ruaccent vs simple: 1727 common targets, ruaccent only right 981, simple only right 47, p = 4e-228.
- silero vs simple: 1727 common targets, silero only right 270, simple only right 485, p = 4e-15.

### ru / prose-accents — SILVER

Source: https://huggingface.co/datasets/nevmenandr/accentual-syllabic-verse-in-russian-prose (MIT).  Scored: marked polysyllabic words (ru-accent-poet labels, fixed sample).
Rows in: 2211; rows scored: 2200; targets: 16839.
Sample: every 46th file of the sorted list, the first 200 sentences of 4 to 40 tokens per file.
Dropped while loading: no usable target (rows): 11.
Labels agree with the RUAccent accent dictionary, unambiguous entries on 90.9% of 3953 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | SILVER | 16839 | 0 | 0.863 | 0.854–0.871 | 0.829 (1298) | 98.4 | 2.4 |
| silero | SILVER | 16839 | 0 | 0.923 | 0.918–0.926 | 0.361 (1298) | 1.0 | 0.3 |
| simple | SILVER | 16839 | 0 | 0.214 | 0.208–0.221 | 0.272 (1298) | 0.2 | 0.1 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs silero: 16839 common targets, ruaccent only right 778, silero only right 1786, p = 2e-90.
- ruaccent vs simple: 16839 common targets, ruaccent only right 11411, simple only right 496, p < 1e-300.
- silero vs simple: 16839 common targets, silero only right 12124, simple only right 201, p < 1e-300.

### ru / ruslan-stressed — SILVER

Source: https://huggingface.co/datasets/stilletto/ruslan-stressed (CC-BY-4.0).  Scored: marked polysyllabic words (LLM labels).
Rows in: 22200; rows scored: 22185; targets: 176255.
ё is left unmarked by the set's convention, so words stressed on ё are not targets.
Dropped while loading: monosyllabic token (targets): 63147; no usable target (rows): 15; token carries 2+ acutes (targets): 639.
Labels agree with the RUAccent accent dictionary, unambiguous entries on 90.8% of 29767 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | SILVER | 176255 | 0 | 0.948 | 0.947–0.950 | 0.928 (10255) | 122.7 | 1.8 |
| silero | SILVER | 176255 | 0 | 0.945 | 0.944–0.946 | 0.436 (10255) | 1.1 | 0.2 |
| simple | SILVER | 176255 | 0 | 0.152 | 0.150–0.153 | 0.252 (10255) | 0.2 | 0.1 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs silero: 176255 common targets, ruaccent only right 6696, silero only right 6096, p = 1e-07.
- ruaccent vs simple: 176255 common targets, ruaccent only right 141521, simple only right 1082, p < 1e-300.
- silero vs simple: 176255 common targets, silero only right 141375, simple only right 1536, p < 1e-300.

## uk

Leader on each GOLD set, against the runner-up:

- russtress: silero 0.807 vs simple 0.534, McNemar p < 1e-300.
- wiktionary: silero 0.865 vs simple 0.726, McNemar p = 1e-154.

### uk / russtress — GOLD

Source: https://github.com/MashaPo/russtress (none stated).  Scored: middle word of each window.
Rows in: 12253; rows scored: 12252; targets: 12252.
Dropped while loading: gold position not on a vowel of a token (rows): 1.
Labels agree with the stressonnx simple-uk vocabulary (English Wiktionary headwords) on 88.1% of 7431 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 12252 | 0 | 0.807 | 0.800–0.814 | — | 0.2 | 0.2 |
| simple | GOLD | 12252 | 0 | 0.534 | 0.526–0.543 | — | 0.0 | 0.0 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 12252 common targets, silero only right 4054, simple only right 715, p < 1e-300.

### uk / wiktionary — GOLD

Source: https://kaikki.org/dictionary/Ukrainian/ (CC-BY-SA).  Scored: marked polysyllabic words.
Rows in: 5610; rows scored: 4374; targets: 10801.
The simple-uk vocabulary is built from the same Wiktionary, so the simple score on this set is partly self-referential.
Dropped while loading: Latin letters: transliteration or gloss (rows): 176; acute not on a vowel (targets): 4; example of an affix entry (rows): 1051; monosyllabic token (targets): 26; no usable target (rows): 9; token carries 2+ acutes (targets): 106.
Labels agree with the stressonnx simple-uk vocabulary (English Wiktionary headwords) on 96.2% of 8145 targets the dictionary covers (same source as the set, so not independent).

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 10801 | 0 | 0.865 | 0.858–0.871 | — | 0.2 | 0.1 |
| simple | GOLD | 10801 | 0 | 0.726 | 0.716–0.735 | — | 0.1 | 0.0 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 10801 common targets, silero only right 2411, simple only right 911, p = 1e-154.

### uk / uk-tts-stressed — SILVER

Source: https://huggingface.co/datasets/dmytro-m/uk-tts-stressed-serhiy (and -mykyta, -oleksa, -tetiana) (MIT).  Scored: marked polysyllabic words (dictionary + context-model labels).
Rows in: 1024; rows scored: 1023; targets: 6322.
Test and validation splits of four speakers.  The card does not name the context model that made the labels.
Dropped while loading: no usable target (rows): 1; token carries 2+ acutes (targets): 11.
Labels agree with the stressonnx simple-uk vocabulary (English Wiktionary headwords) on 95.7% of 3878 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | SILVER | 6322 | 0 | 0.841 | 0.833–0.851 | — | 0.5 | 0.2 |
| simple | SILVER | 6322 | 0 | 0.587 | 0.574–0.600 | — | 0.1 | 0.0 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 6322 common targets, silero only right 2106, simple only right 499, p = 1e-233.

## be

Leader on each GOLD set, against the runner-up:

- homographs-10x10: silero 0.499 vs simple 0.425, McNemar p = 4e-25.
- russtress: silero 0.908 vs simple 0.435, McNemar p < 1e-300.
- wiktionary: silero 0.950 vs simple 0.387, McNemar p < 1e-300.
- zascienak: silero 0.949 vs simple 0.510, McNemar p < 1e-300.

### be / homographs-10x10 — GOLD

Source: https://huggingface.co/datasets/alex73/benchmarks-stress-bel (CC-BY-SA-4.0).  Scored: the homograph only (expert-written, balanced).
Rows in: 1500; rows scored: 1495; targets: 1496.
Dropped while loading: homograph only inside a hyphenated compound (rows): 5.
Labels agree with the stressonnx simple-be vocabulary (exported from silero_stress) on 49.8% of 1256 targets the dictionary covers.
Every target is a homograph, so a one-stress dictionary disagrees by design; this check does not measure label noise.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 1496 | 0 | 0.499 | 0.475–0.526 | — | 0.6 | 0.1 |
| simple | GOLD | 1496 | 0 | 0.425 | 0.400–0.451 | — | 0.2 | 0.0 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 1496 common targets, silero only right 120, simple only right 10, p = 4e-25.

### be / russtress — GOLD

Source: https://github.com/MashaPo/russtress (none stated).  Scored: middle word of each window.
Rows in: 2907; rows scored: 2902; targets: 2902.
Dropped while loading: gold position not on a vowel of a token (rows): 5.
Labels agree with the stressonnx simple-be vocabulary (exported from silero_stress) on 90.7% of 1399 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 2902 | 0 | 0.908 | 0.898–0.917 | — | 0.2 | 0.1 |
| simple | GOLD | 2902 | 0 | 0.435 | 0.416–0.452 | — | 0.0 | 0.0 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 2902 common targets, silero only right 1389, simple only right 14, p < 1e-300.

### be / wiktionary — GOLD

Source: https://kaikki.org/dictionary/Belarusian/ (CC-BY-SA).  Scored: marked polysyllabic words.
Rows in: 4717; rows scored: 4618; targets: 12390.
Dropped while loading: Latin letters: transliteration or gloss (rows): 19; acute not on a vowel (targets): 8; example of an affix entry (rows): 74; monosyllabic token (targets): 102; no usable target (rows): 6; token carries 2+ acutes (targets): 28.
Labels agree with the stressonnx simple-be vocabulary (exported from silero_stress) on 95.0% of 5105 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 12390 | 0 | 0.950 | 0.946–0.955 | — | 0.2 | 0.1 |
| simple | GOLD | 12390 | 0 | 0.387 | 0.378–0.395 | — | 0.1 | 0.0 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 12390 common targets, silero only right 7000, simple only right 21, p < 1e-300.

### be / zascienak — GOLD

Source: https://huggingface.co/datasets/alex73/benchmarks-stress-bel (CC-BY-SA-4.0).  Scored: marked polysyllabic words (fully hand-marked novel).
Rows in: 1538; rows scored: 1523; targets: 22965.
Dropped while loading: no usable target (rows): 15; token carries 2+ acutes (targets): 51.
Labels agree with the stressonnx simple-be vocabulary (exported from silero_stress) on 95.9% of 12141 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 22965 | 0 | 0.949 | 0.946–0.952 | — | 1.1 | 0.1 |
| simple | GOLD | 22965 | 0 | 0.510 | 0.503–0.517 | — | 0.3 | 0.0 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 22965 common targets, silero only right 10165, simple only right 73, p < 1e-300.

### be / homographs-commonvoice — SILVER

Source: https://huggingface.co/datasets/alex73/benchmarks-stress-bel (CC-BY-SA-4.0).  Scored: the marked homographs (labelling method undocumented).
Rows in: 2930; rows scored: 2850; targets: 2877.
Dropped while loading: commented out upstream (rows): 80.
Labels agree with the stressonnx simple-be vocabulary (exported from silero_stress) on 78.7% of 2063 targets the dictionary covers.
Every target is a homograph, so a one-stress dictionary disagrees by design; this check does not measure label noise.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | SILVER | 2877 | 0 | 0.696 | 0.679–0.711 | — | 0.4 | 0.1 |
| simple | SILVER | 2877 | 0 | 0.565 | 0.546–0.581 | — | 0.1 | 0.0 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 2877 common targets, silero only right 381, simple only right 4, p = 2e-107.

## Languages without independent gold

The 20 `simple`-model languages ship curated vocabularies exported from
silero_stress; no independent stress-annotated corpus is publicly
available for them, so no accuracy row is reported (the parity test in
`tests/test_export_parity.py` locks them to the upstream reference).

<!-- run.py:end -->

## OOV positional rules (simple model)

Accuracy of each language's out-of-vocabulary rule measured against its own
curated vocabulary (multi-vowel words only; `python
benchmarks/oov_rules_eval.py`).  Rules are grounded in the descriptive
literature cited in `SimpleStressor._accentuate_oov`.

| lang | rule | words | accuracy | note |
|------|------|-------|----------|------|
| ka | antepenultimate | 12118 | 1.000 | matches the Akhvlediani/Gudava tradition the vocab follows; Georgian stress is weak and contested (Borise 2020 argues initial) |
| xal | last | 5918 | 1.000 | |
| uz-Cyrl / uz-Latn | last | 4791 / 4517 | 1.000 | |
| ky | last | 11247 | 0.993 | |
| kk | last | 6659 | 0.987 | unstressable-suffix classes (negation -ма etc., Kirchner 1998) not yet modeled |
| udm | last | 13251 | 0.975 | negated/imperative verbs (initial stress) not detectable without morphology |
| cv | last full vowel (ӑ/ӗ never stressed) | 22050 | 0.940 | Clark 1998 / Krueger 1961 / Dobrovolsky 1999 |
| tt | final, unstressed suffixes -ча/-чә/-сең/-ме retract (Comrie 1997b) | 11043 | 0.904 | only vocabulary-validated suffix variants enabled |
| ba | last | 9456 | 0.852 | |
| az-Cyrl / az-Latn | last | 10895 / 10925 | 0.816 | |
| hy | last non-schwa (ը) | 19794 | 0.777 | Chakmakjian 2024 |
| mdf | first, retract off initial high vowel to а/я | 4821 | 0.761 | Hamari & Ajanki 2022 |
| tg | final, word-final izafet -и unstressed | 8530 | 0.736 | Perry 2005 |
| myv | first | 4931 | 0.538 | stress is a tendency, not a rule (Oxford Guide to the Uralic Languages) |
| kbd | final, unless word-final э → penult | 5648 | 0.404 | paraphrase-level sources only; low confidence |
| kjh | last | 12032 | 0.280 | no primary source on Khakas stress could be located; Turkic-default kept |
| sah | last heavy nucleus (long vowel/diphthong), else final | 85746 | 0.994 | Yakut long vowels/diphthongs attract stress (Krueger 1962 + weight sensitivity); the old naive final rule scored 0.178 |
| be (`simple`) | none | 21968 | — | Belarusian stress is lexical, not positional — no rule is defensible; vocabulary only |

Open items: suffix-aware refinements for kk/ky/az — the closed
unstressable-suffix lists (Kirchner 1998) were measured against those
vocabularies and blind surface-string retraction loses more than it gains
(vocabulary words ending in the same strings are ordinary final-stressed
nouns); applying them correctly needs morphological segmentation.  Tatar is
the exception: four suffix variants passed the ≥0.7 vocabulary-evidence bar
and are enabled.

## Wiktionary / dictionary languages (vocabulary-first)

| lang | vocab | source | rule | note |
|------|-------|--------|------|------|
| bg | 46,914 | English Wiktionary (kaikki.org, CC BY-SA) | none | Bulgarian stress is free/lexical (Scatton 1984) — dictionary lookup only, no positional guessing |
| uk / simple | 49,809 | English Wiktionary (kaikki.org, CC BY-SA) | none | rule-free dictionary fallback below the silero neural model |
| ru / simple | 108,972 | RUAccent pronunciation dictionary (Apache-2.0), unambiguous entries | none | rule-free dictionary fallback below ruaccent/silero |
| sl | 4,421 | English Wiktionary tonal-mark headwords (kaikki.org, CC BY-SA) | none | Slovene stress is free/lexical (Herrity 2000) |
| mk | 1,667 | English Wiktionary (kaikki.org, CC BY-SA) | antepenultimate (Friedman 2001) | the vocabulary is deliberately *exceptions-only* — Wiktionary marks stress on Macedonian words precisely when they violate the antepenultimate default, so the 0.29 rule-vs-own-vocab score measures the rule on the words it is not meant for; regular words follow the rule |
| lv | 107 | English Wiktionary (kaikki.org, CC BY-SA) | first (Nau 1998) | exceptions-only vocabulary, same caveat as mk |

The ``none``-rule languages score 0.000 on the rule-vs-vocab metric by
construction (the rule never guesses); their quality is vocabulary coverage.

## Independent Wiktionary-IPA spot-check (simple pipeline)

Full vocabulary+rule pipeline scored against words whose English-Wiktionary
IPA carries a primary-stress mark and aligns syllable-for-vowel with the
orthography (`python benchmarks/wiktionary_ipa_gold.py`).  Small,
lemma-skewed sets — a sanity check independent of the shipped vocabularies,
not a benchmark.  "OOV" rows isolate the positional rule on words the
vocabulary does not contain.

| lang | words | accuracy | OOV words | OOV accuracy |
|------|-------|----------|-----------|--------------|
| hy | 10486 | 0.982 | 64 | 1.000 |
| udm | 537 | 0.998 | 335 | 0.997 |
| cv | 40 | 0.950 | 7 | 0.857 |
| ba | 1740 | 0.937 | 929 | 0.959 |
| tt | 34 | 0.941 | 7 | 0.857 |
| tg | 439 | 0.936 | 343 | 0.933 |
| kk | 1224 | 0.826 | 922 | 0.782 |
| ka | 7 | 0.714 | 4 | 0.750 |

Notably, the Tajik izafet rule and the Armenian schwa rule hold up better on
independent lemmas (0.93 / 0.98) than the vocabulary self-scores suggested.
