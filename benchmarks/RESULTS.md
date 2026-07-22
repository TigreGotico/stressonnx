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
| ru | ruaccent | 11116 | 0.908 | 724 | 0.742 | 3.5 | 8.7 |
| ru | silero | 11124 | 0.886 | 724 | 0.377 | 1.3 | 0.1 |
| ru | kubataba (first 1000 rows) | 910 | 0.884 | 103 | 0.932 | 0.5 | 61.6 |
| ukr | silero | 12252 | 0.767 | — | — | 1.1 | 0.1 |
| bel | silero | 2902 | 0.859 | — | — | 1.3 | 0.1 |
| bel_simple | simple | 2902 | 0.417 | — | — | 0.3 | 0.0 |

**Annotation-noise ceiling:** on Russian targets covered by the RUAccent
pronunciation dictionary with a single unambiguous stress, the gold
annotation agrees with the dictionary only 88.5% of the time (1769 words checked) —
crowdsourced noise plus genuine variant stress (го́ду/году́).  Absolute
accuracies are bounded by that ceiling; *relative* comparison between
models on identical rows remains meaningful.

Rows dropped per evaluation (no vowel in target / model rewrote the window structurally): ru/ruaccent: 8, ru/silero: 0, ru/kubataba: 90, ukr/silero: 1, bel/silero: 5, bel_simple/simple: 5.

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
| kat | antepenultimate | 12118 | 1.000 | matches the Akhvlediani/Gudava tradition the vocab follows; Georgian stress is weak and contested (Borise 2020 argues initial) |
| xal | last | 5918 | 1.000 | |
| uzb_cyr / uzb_lat | last | 4791 / 4517 | 1.000 | |
| kir | last | 11247 | 0.993 | |
| kaz | last | 6659 | 0.987 | unstressable-suffix classes (negation -ма etc., Kirchner 1998) not yet modeled |
| udm | last | 13251 | 0.975 | negated/imperative verbs (initial stress) not detectable without morphology |
| chv | last full vowel (ӑ/ӗ never stressed) | 22050 | 0.940 | Clark 1998 / Krueger 1961 / Dobrovolsky 1999 |
| tat | last | 11043 | 0.896 | |
| bak | last | 9456 | 0.852 | |
| aze_cyr / aze_lat | last | 10895 / 10925 | 0.816 | |
| hye | last non-schwa (ը) | 19794 | 0.777 | Chakmakjian 2024 |
| mdf | first, retract off initial high vowel to а/я | 4821 | 0.761 | Hamari & Ajanki 2022 |
| tgk | final, word-final izafet -и unstressed | 8530 | 0.736 | Perry 2005 |
| erz | first | 4931 | 0.538 | stress is a tendency, not a rule (Oxford Guide to the Uralic Languages) |
| kbd | final, unless word-final э → penult | 5648 | 0.404 | paraphrase-level sources only; low confidence |
| kjh | last | 12032 | 0.280 | no primary source on Khakas stress could be located; Turkic-default kept |
| sah | last | 85746 | 0.178 | Krueger 1962 says final; the vocabulary systematically disagrees — open item (possibly long-vowel digraph convention) |
| bel_simple | none | 21968 | — | Belarusian stress is lexical, not positional — no rule is defensible; vocabulary only |

Open items: the sah vocabulary anomaly; suffix-aware refinements for
kaz/kir/tat/aze (closed unstressable-suffix lists exist in the literature but
need morphological segmentation to apply).
