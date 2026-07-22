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
