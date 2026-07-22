# Benchmark scoreboard

Run date: 2026-07-22.  Reproduce: `python benchmarks/run.py --date <date>`.

Gold: stress-annotated UD-treebank windows from
[russtress](https://github.com/MashaPo/russtress) (Ponomareva et al. 2017,
Yandex.Toloka annotation; fetched at run time — see provenance note in
`benchmarks/run.py`).  Accuracy = predicted stressed character equals the
gold annotation for the window's target word.  `homograph acc` scores the
subset of target words present in the RUAccent omograph dictionary.

| lang | model | rows | accuracy | homograph rows | homograph acc | cold load (s) | ms/row (warm) |
|------|-------|------|----------|----------------|---------------|---------------|---------------|
| ru | ruaccent | 11116 | 0.938 | 724 | 0.820 | 1.8 | 20.8 |
| ru | silero | 11124 | 0.914 | 724 | 0.381 | 0.2 | 0.2 |
| uk | silero | 12252 | 0.785 | — | — | 0.2 | 0.2 |
| be | silero | 2902 | 0.873 | — | — | 0.1 | 0.2 |
| be | simple | 2902 | 0.433 | — | — | 0.0 | 0.0 |

**Annotation-noise ceiling:** on Russian targets covered by the RUAccent
pronunciation dictionary with a single unambiguous stress, the gold
annotation agrees with the dictionary only 88.5% of the time (1769 words checked) —
crowdsourced noise plus genuine variant stress (го́ду/году́).  Absolute
accuracies are bounded by that ceiling; *relative* comparison between
models on identical rows remains meaningful.

Rows dropped per evaluation (no vowel in target / model rewrote the window structurally): ru/ruaccent: 8, ru/silero: 0, uk/silero: 1, be/silero: 5, be/simple: 5.

## Languages without independent gold

The 20 `simple`-model languages ship curated vocabularies exported from
silero_stress; no independent stress-annotated corpus is publicly
available for them, so no accuracy row is reported (the parity test in
`tests/test_export_parity.py` locks them to the upstream reference).
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
