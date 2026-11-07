<!-- run.py:begin -->
# Benchmark scoreboard

Run date: 2026-11-07.  Reproduce: `python benchmarks/run.py --date <date>`;
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

The training data of silero, ruaccent and ruaccent-big-poetry is
unpublished, so overlap with these sets cannot be established.  The set
notes name the overlap of stressrnn and kubataba, whose training corpora
are published.

## ru

Leader on each GOLD set, against the runner-up:

- russtress: ruaccent-silero 0.951 vs ruaccent-silero-int8 0.951, McNemar p = 0.307.
- wiktionary: ruaccent-silero 0.990 vs ruaccent-silero-int8 0.990, McNemar p = 0.786.

### ru / russtress — GOLD

Source: https://github.com/MashaPo/russtress (none stated).  Scored: middle word of each window.
Rows in: 11124; rows scored: 11124; targets: 11124.
This file is the training corpus of StressRNN, so the stressrnn score is optimistic: it is not a held-out measurement.
Labels agree with the RUAccent accent dictionary, unambiguous entries on 88.9% of 1749 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | GOLD | 11124 | 0 | 0.947 | 0.943–0.951 | 0.833 (724) | 7.9 | 1.7 |
| ruaccent-big-poetry | GOLD | 11124 | 0 | 0.942 | 0.938–0.946 | 0.753 (724) | 11.1 | 2.4 |
| stressrnn | GOLD | 11124 | 0 | 0.892 | 0.886–0.897 | 0.767 (724) | 0.2 | 0.1 |
| kubataba | GOLD | 11124 | 0 | 0.888 | 0.882–0.893 | 0.816 (724) | 105.8 | 0.1 |
| ruaccent-int8 | GOLD | 11124 | 0 | 0.946 | 0.942–0.950 | 0.825 (724) | 5.2 | 1.2 |
| silero | GOLD | 11124 | 0 | 0.944 | 0.940–0.948 | 0.753 (724) | 0.3 | 0.2 |
| ruaccent-silero | GOLD | 11124 | 0 | 0.951 | 0.947–0.955 | 0.833 (724) | 10.7 | 1.9 |
| ruaccent-silero-int8 | GOLD | 11124 | 0 | 0.951 | 0.947–0.955 | 0.825 (724) | 7.4 | 1.4 |
| simple | GOLD | 11124 | 0 | 0.871 | 0.865–0.877 | 0.000 (724) | 0.0 | 9.6 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs ruaccent-big-poetry: 11124 common targets, ruaccent only right 114, ruaccent-big-poetry only right 56, p = 1e-05.
- ruaccent vs stressrnn: 11124 common targets, ruaccent only right 749, stressrnn only right 131, p = 7e-106.
- ruaccent vs kubataba: 11124 common targets, ruaccent only right 830, kubataba only right 169, p = 3e-105.
- ruaccent vs ruaccent-int8: 11124 common targets, ruaccent only right 18, ruaccent-int8 only right 9, p = 0.122.
- ruaccent vs silero: 11124 common targets, ruaccent only right 173, silero only right 137, p = 0.047.
- ruaccent vs ruaccent-silero: 11124 common targets, ruaccent only right 0, ruaccent-silero only right 46, p = 3e-14.
- ruaccent vs ruaccent-silero-int8: 11124 common targets, ruaccent only right 15, ruaccent-silero-int8 only right 55, p = 2e-06.
- ruaccent vs simple: 11124 common targets, ruaccent only right 895, simple only right 49, p = 4e-202.
- ruaccent-big-poetry vs stressrnn: 11124 common targets, ruaccent-big-poetry only right 748, stressrnn only right 188, p = 1e-79.
- ruaccent-big-poetry vs kubataba: 11124 common targets, ruaccent-big-poetry only right 834, kubataba only right 231, p = 2e-80.
- ruaccent-big-poetry vs ruaccent-int8: 11124 common targets, ruaccent-big-poetry only right 67, ruaccent-int8 only right 116, p = 4e-04.
- ruaccent-big-poetry vs silero: 11124 common targets, ruaccent-big-poetry only right 138, silero only right 160, p = 0.224.
- ruaccent-big-poetry vs ruaccent-silero: 11124 common targets, ruaccent-big-poetry only right 56, ruaccent-silero only right 160, p = 9e-13.
- ruaccent-big-poetry vs ruaccent-silero-int8: 11124 common targets, ruaccent-big-poetry only right 64, ruaccent-silero-int8 only right 162, p = 6e-11.
- ruaccent-big-poetry vs simple: 11124 common targets, ruaccent-big-poetry only right 837, simple only right 49, p = 5e-186.
- stressrnn vs kubataba: 11124 common targets, stressrnn only right 497, kubataba only right 454, p = 0.173.
- stressrnn vs ruaccent-int8: 11124 common targets, stressrnn only right 138, ruaccent-int8 only right 747, p = 8e-102.
- stressrnn vs silero: 11124 common targets, stressrnn only right 161, silero only right 743, p = 6e-90.
- stressrnn vs ruaccent-silero: 11124 common targets, stressrnn only right 106, ruaccent-silero only right 770, p = 4e-125.
- stressrnn vs ruaccent-silero-int8: 11124 common targets, stressrnn only right 110, ruaccent-silero-int8 only right 768, p = 4e-122.
- stressrnn vs simple: 11124 common targets, stressrnn only right 715, simple only right 487, p = 5e-11.
- kubataba vs ruaccent-int8: 11124 common targets, kubataba only right 178, ruaccent-int8 only right 830, p = 4e-101.
- kubataba vs silero: 11124 common targets, kubataba only right 199, silero only right 824, p = 7e-91.
- kubataba vs ruaccent-silero: 11124 common targets, kubataba only right 126, ruaccent-silero only right 833, p = 2e-128.
- kubataba vs ruaccent-silero-int8: 11124 common targets, kubataba only right 132, ruaccent-silero-int8 only right 833, p = 5e-125.
- kubataba vs simple: 11124 common targets, kubataba only right 816, simple only right 631, p = 1e-06.
- ruaccent-int8 vs silero: 11124 common targets, ruaccent-int8 only right 171, silero only right 144, p = 0.143.
- ruaccent-int8 vs ruaccent-silero: 11124 common targets, ruaccent-int8 only right 9, ruaccent-silero only right 64, p = 2e-11.
- ruaccent-int8 vs ruaccent-silero-int8: 11124 common targets, ruaccent-int8 only right 0, ruaccent-silero-int8 only right 49, p = 4e-15.
- ruaccent-int8 vs simple: 11124 common targets, ruaccent-int8 only right 889, simple only right 52, p = 1e-197.
- silero vs ruaccent-silero: 11124 common targets, silero only right 97, ruaccent-silero only right 179, p = 9e-07.
- silero vs ruaccent-silero-int8: 11124 common targets, silero only right 101, ruaccent-silero-int8 only right 177, p = 6e-06.
- silero vs simple: 11124 common targets, silero only right 832, simple only right 22, p = 4e-214.
- ruaccent-silero vs ruaccent-silero-int8: 11124 common targets, ruaccent-silero only right 15, ruaccent-silero-int8 only right 9, p = 0.307.
- ruaccent-silero vs simple: 11124 common targets, ruaccent-silero only right 906, simple only right 14, p = 7e-247.
- ruaccent-silero-int8 vs simple: 11124 common targets, ruaccent-silero-int8 only right 900, simple only right 14, p = 4e-245.

### ru / wiktionary — GOLD

Source: https://kaikki.org/dictionary/Russian/ (CC-BY-SA).  Scored: marked polysyllabic words.
Rows in: 26494; rows scored: 24419; targets: 137221.
Dropped while loading: Latin letters: transliteration or gloss (rows): 378; acute not on a vowel (targets): 14; example of an affix entry (rows): 1583; monosyllabic token (targets): 828; no usable target (rows): 114; token carries 2+ acutes (targets): 532.
Not run: kubataba, about 1.4 s per row on this set (60 sampled rows), about 10 hours for the set; not run
Labels agree with the RUAccent accent dictionary, unambiguous entries on 92.0% of 25918 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | GOLD | 137221 | 0 | 0.976 | 0.975–0.977 | 0.926 (9410) | 38.6 | 1.5 |
| ruaccent-int8 | GOLD | 137221 | 0 | 0.976 | 0.975–0.977 | 0.925 (9410) | 21.5 | 1.0 |
| ruaccent-big-poetry | GOLD | 137221 | 0 | 0.974 | 0.973–0.975 | 0.893 (9410) | 53.2 | 2.0 |
| stressrnn | GOLD | 137221 | 0 | 0.914 | 0.912–0.915 | 0.739 (9410) | 0.4 | 0.1 |
| silero | GOLD | 137221 | 0 | 0.984 | 0.984–0.985 | 0.908 (9410) | 1.0 | 0.2 |
| ruaccent-silero | GOLD | 137221 | 0 | 0.990 | 0.989–0.990 | 0.933 (9410) | 51.5 | 1.6 |
| ruaccent-silero-int8 | GOLD | 137221 | 0 | 0.990 | 0.989–0.990 | 0.933 (9410) | 33.8 | 1.4 |
| simple | GOLD | 137221 | 0 | 0.915 | 0.913–0.916 | 0.007 (9410) | 0.1 | 8.5 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs ruaccent-int8: 137221 common targets, ruaccent only right 146, ruaccent-int8 only right 120, p = 0.125.
- ruaccent vs ruaccent-big-poetry: 137221 common targets, ruaccent only right 630, ruaccent-big-poetry only right 320, p = 5e-24.
- ruaccent vs stressrnn: 137221 common targets, ruaccent only right 10393, stressrnn only right 1832, p < 1e-300.
- ruaccent vs silero: 137221 common targets, ruaccent only right 1284, silero only right 2393, p = 1e-75.
- ruaccent vs ruaccent-silero: 137221 common targets, ruaccent only right 0, ruaccent-silero only right 1893, p < 1e-300.
- ruaccent vs ruaccent-silero-int8: 137221 common targets, ruaccent only right 63, ruaccent-silero-int8 only right 1952, p < 1e-300.
- ruaccent vs simple: 137221 common targets, ruaccent only right 10271, simple only right 1836, p < 1e-300.
- ruaccent-int8 vs ruaccent-big-poetry: 137221 common targets, ruaccent-int8 only right 686, ruaccent-big-poetry only right 402, p = 6e-18.
- ruaccent-int8 vs stressrnn: 137221 common targets, ruaccent-int8 only right 10384, stressrnn only right 1849, p < 1e-300.
- ruaccent-int8 vs silero: 137221 common targets, ruaccent-int8 only right 1280, silero only right 2415, p = 7e-79.
- ruaccent-int8 vs ruaccent-silero: 137221 common targets, ruaccent-int8 only right 59, ruaccent-silero only right 1978, p < 1e-300.
- ruaccent-int8 vs ruaccent-silero-int8: 137221 common targets, ruaccent-int8 only right 0, ruaccent-silero-int8 only right 1915, p < 1e-300.
- ruaccent-int8 vs simple: 137221 common targets, ruaccent-int8 only right 10266, simple only right 1857, p < 1e-300.
- ruaccent-big-poetry vs stressrnn: 137221 common targets, ruaccent-big-poetry only right 10270, stressrnn only right 2019, p < 1e-300.
- ruaccent-big-poetry vs silero: 137221 common targets, ruaccent-big-poetry only right 1290, silero only right 2709, p = 9e-114.
- ruaccent-big-poetry vs ruaccent-silero: 137221 common targets, ruaccent-big-poetry only right 320, ruaccent-silero only right 2523, p < 1e-300.
- ruaccent-big-poetry vs ruaccent-silero-int8: 137221 common targets, ruaccent-big-poetry only right 319, ruaccent-silero-int8 only right 2518, p < 1e-300.
- ruaccent-big-poetry vs simple: 137221 common targets, ruaccent-big-poetry only right 9961, simple only right 1836, p < 1e-300.
- stressrnn vs silero: 137221 common targets, stressrnn only right 817, silero only right 10487, p < 1e-300.
- stressrnn vs ruaccent-silero: 137221 common targets, stressrnn only right 575, ruaccent-silero only right 11029, p < 1e-300.
- stressrnn vs ruaccent-silero-int8: 137221 common targets, stressrnn only right 570, ruaccent-silero-int8 only right 11020, p < 1e-300.
- stressrnn vs simple: 137221 common targets, stressrnn only right 7921, simple only right 8047, p = 0.323.
- silero vs ruaccent-silero: 137221 common targets, silero only right 615, ruaccent-silero only right 1399, p = 6e-70.
- silero vs ruaccent-silero-int8: 137221 common targets, silero only right 615, ruaccent-silero-int8 only right 1395, p = 2e-69.
- silero vs simple: 137221 common targets, silero only right 9776, simple only right 232, p < 1e-300.
- ruaccent-silero vs ruaccent-silero-int8: 137221 common targets, ruaccent-silero only right 63, ruaccent-silero-int8 only right 59, p = 0.786.
- ruaccent-silero vs simple: 137221 common targets, ruaccent-silero only right 10420, simple only right 92, p < 1e-300.
- ruaccent-silero-int8 vs simple: 137221 common targets, ruaccent-silero-int8 only right 10416, simple only right 92, p < 1e-300.

### ru / homograph-eval — SILVER

Source: https://huggingface.co/datasets/inkoziev/HomographResolutionEval (CC-BY-4.0).  Scored: the homograph only.
Rows in: 1741; rows scored: 1727; targets: 1727.
Dropped while loading: homograph field has no single acute (rows): 5; homograph not found in the sentence (rows): 7; homograph occurs 2+ times (rows): 2.
Labels agree with the RUAccent accent dictionary, unambiguous entries on 67.0% of 896 targets the dictionary covers.
Every target is a homograph, so a one-stress dictionary disagrees by design; this check does not measure label noise.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | SILVER | 1727 | 0 | 0.888 | 0.872–0.902 | 0.896 (1660) | 87.4 | 1.9 |
| ruaccent-int8 | SILVER | 1727 | 0 | 0.892 | 0.875–0.907 | 0.900 (1660) | 43.6 | 1.5 |
| ruaccent-big-poetry | SILVER | 1727 | 0 | 0.852 | 0.834–0.867 | 0.858 (1660) | 128.5 | 3.3 |
| stressrnn | SILVER | 1727 | 0 | 0.521 | 0.499–0.543 | 0.509 (1660) | 0.4 | 0.1 |
| kubataba | SILVER | 1727 | 0 | 0.583 | 0.561–0.606 | 0.575 (1660) | 200.6 | 0.1 |
| silero | SILVER | 1727 | 0 | 0.816 | 0.797–0.834 | 0.814 (1660) | 1.8 | 0.2 |
| ruaccent-silero | SILVER | 1727 | 0 | 0.888 | 0.872–0.902 | 0.896 (1660) | 93.9 | 1.8 |
| ruaccent-silero-int8 | SILVER | 1727 | 0 | 0.892 | 0.875–0.907 | 0.900 (1660) | 47.4 | 1.4 |
| simple | SILVER | 1727 | 0 | 0.027 | 0.019–0.034 | 0.000 (1660) | 0.1 | 8.6 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs ruaccent-int8: 1727 common targets, ruaccent only right 24, ruaccent-int8 only right 30, p = 0.497.
- ruaccent vs ruaccent-big-poetry: 1727 common targets, ruaccent only right 164, ruaccent-big-poetry only right 101, p = 1e-04.
- ruaccent vs stressrnn: 1727 common targets, ruaccent only right 729, stressrnn only right 94, p = 2e-122.
- ruaccent vs kubataba: 1727 common targets, ruaccent only right 635, kubataba only right 108, p = 1e-91.
- ruaccent vs silero: 1727 common targets, ruaccent only right 230, silero only right 105, p = 7e-12.
- ruaccent vs ruaccent-silero: 1727 common targets, ruaccent only right 0, ruaccent-silero only right 0, p = 1.000.
- ruaccent vs ruaccent-silero-int8: 1727 common targets, ruaccent only right 24, ruaccent-silero-int8 only right 30, p = 0.497.
- ruaccent vs simple: 1727 common targets, ruaccent only right 1488, simple only right 0, p < 1e-300.
- ruaccent-int8 vs ruaccent-big-poetry: 1727 common targets, ruaccent-int8 only right 161, ruaccent-big-poetry only right 92, p = 2e-05.
- ruaccent-int8 vs stressrnn: 1727 common targets, ruaccent-int8 only right 730, stressrnn only right 89, p = 5e-126.
- ruaccent-int8 vs kubataba: 1727 common targets, ruaccent-int8 only right 632, kubataba only right 99, p = 7e-96.
- ruaccent-int8 vs silero: 1727 common targets, ruaccent-int8 only right 233, silero only right 102, p = 6e-13.
- ruaccent-int8 vs ruaccent-silero: 1727 common targets, ruaccent-int8 only right 30, ruaccent-silero only right 24, p = 0.497.
- ruaccent-int8 vs ruaccent-silero-int8: 1727 common targets, ruaccent-int8 only right 0, ruaccent-silero-int8 only right 0, p = 1.000.
- ruaccent-int8 vs simple: 1727 common targets, ruaccent-int8 only right 1494, simple only right 0, p < 1e-300.
- ruaccent-big-poetry vs stressrnn: 1727 common targets, ruaccent-big-poetry only right 694, stressrnn only right 122, p = 7e-98.
- ruaccent-big-poetry vs kubataba: 1727 common targets, ruaccent-big-poetry only right 587, kubataba only right 123, p = 3e-73.
- ruaccent-big-poetry vs silero: 1727 common targets, ruaccent-big-poetry only right 230, silero only right 168, p = 0.002.
- ruaccent-big-poetry vs ruaccent-silero: 1727 common targets, ruaccent-big-poetry only right 101, ruaccent-silero only right 164, p = 1e-04.
- ruaccent-big-poetry vs ruaccent-silero-int8: 1727 common targets, ruaccent-big-poetry only right 92, ruaccent-silero-int8 only right 161, p = 2e-05.
- ruaccent-big-poetry vs simple: 1727 common targets, ruaccent-big-poetry only right 1425, simple only right 0, p < 1e-300.
- stressrnn vs kubataba: 1727 common targets, stressrnn only right 135, kubataba only right 243, p = 3e-08.
- stressrnn vs silero: 1727 common targets, stressrnn only right 179, silero only right 689, p = 3e-71.
- stressrnn vs ruaccent-silero: 1727 common targets, stressrnn only right 94, ruaccent-silero only right 729, p = 2e-122.
- stressrnn vs ruaccent-silero-int8: 1727 common targets, stressrnn only right 89, ruaccent-silero-int8 only right 730, p = 5e-126.
- stressrnn vs simple: 1727 common targets, stressrnn only right 861, simple only right 8, p = 4e-243.
- kubataba vs silero: 1727 common targets, kubataba only right 186, silero only right 588, p = 2e-49.
- kubataba vs ruaccent-silero: 1727 common targets, kubataba only right 108, ruaccent-silero only right 635, p = 1e-91.
- kubataba vs ruaccent-silero-int8: 1727 common targets, kubataba only right 99, ruaccent-silero-int8 only right 632, p = 7e-96.
- kubataba vs simple: 1727 common targets, kubataba only right 970, simple only right 9, p = 9e-274.
- silero vs ruaccent-silero: 1727 common targets, silero only right 105, ruaccent-silero only right 230, p = 7e-12.
- silero vs ruaccent-silero-int8: 1727 common targets, silero only right 102, ruaccent-silero-int8 only right 233, p = 6e-13.
- silero vs simple: 1727 common targets, silero only right 1363, simple only right 0, p < 1e-300.
- ruaccent-silero vs ruaccent-silero-int8: 1727 common targets, ruaccent-silero only right 24, ruaccent-silero-int8 only right 30, p = 0.497.
- ruaccent-silero vs simple: 1727 common targets, ruaccent-silero only right 1488, simple only right 0, p < 1e-300.
- ruaccent-silero-int8 vs simple: 1727 common targets, ruaccent-silero-int8 only right 1494, simple only right 0, p < 1e-300.

### ru / prose-accents — SILVER

Source: https://huggingface.co/datasets/nevmenandr/accentual-syllabic-verse-in-russian-prose (MIT).  Scored: marked polysyllabic words (ru-accent-poet labels, fixed sample).
Rows in: 2211; rows scored: 2200; targets: 16839.
Sample: every 46th file of the sorted list, the first 200 sentences of 4 to 40 tokens per file.  kubataba was trained on sentence pairs cut from this corpus, so its score here is agreement with its own training labels.
Dropped while loading: no usable target (rows): 11.
Labels agree with the RUAccent accent dictionary, unambiguous entries on 90.9% of 3953 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | SILVER | 16839 | 0 | 0.944 | 0.940–0.947 | 0.829 (1298) | 54.6 | 1.5 |
| ruaccent-int8 | SILVER | 16839 | 0 | 0.944 | 0.940–0.947 | 0.832 (1298) | 27.8 | 1.0 |
| ruaccent-big-poetry | SILVER | 16839 | 0 | 0.944 | 0.940–0.947 | 0.825 (1298) | 73.4 | 2.1 |
| stressrnn | SILVER | 16839 | 0 | 0.921 | 0.916–0.926 | 0.836 (1298) | 0.5 | 0.1 |
| kubataba | SILVER | 16839 | 0 | 0.968 | 0.965–0.971 | 0.946 (1298) | 664.6 | 0.1 |
| silero | SILVER | 16839 | 0 | 0.959 | 0.956–0.962 | 0.816 (1298) | 1.7 | 0.2 |
| ruaccent-silero | SILVER | 16839 | 0 | 0.963 | 0.960–0.966 | 0.831 (1298) | 72.9 | 1.6 |
| ruaccent-silero-int8 | SILVER | 16839 | 0 | 0.963 | 0.961–0.966 | 0.834 (1298) | 46.6 | 1.2 |
| simple | SILVER | 16839 | 0 | 0.885 | 0.880–0.889 | 0.002 (1298) | 0.1 | 8.8 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs ruaccent-int8: 16839 common targets, ruaccent only right 19, ruaccent-int8 only right 22, p = 0.755.
- ruaccent vs ruaccent-big-poetry: 16839 common targets, ruaccent only right 76, ruaccent-big-poetry only right 71, p = 0.742.
- ruaccent vs stressrnn: 16839 common targets, ruaccent only right 908, stressrnn only right 523, p = 2e-24.
- ruaccent vs kubataba: 16839 common targets, ruaccent only right 284, kubataba only right 683, p = 1e-38.
- ruaccent vs silero: 16839 common targets, ruaccent only right 191, silero only right 449, p = 7e-25.
- ruaccent vs ruaccent-silero: 16839 common targets, ruaccent only right 0, ruaccent-silero only right 327, p = 7e-99.
- ruaccent vs ruaccent-silero-int8: 16839 common targets, ruaccent only right 7, ruaccent-silero-int8 only right 338, p = 3e-90.
- ruaccent vs simple: 16839 common targets, ruaccent only right 1331, simple only right 334, p = 1e-140.
- ruaccent-int8 vs ruaccent-big-poetry: 16839 common targets, ruaccent-int8 only right 86, ruaccent-big-poetry only right 78, p = 0.585.
- ruaccent-int8 vs stressrnn: 16839 common targets, ruaccent-int8 only right 908, stressrnn only right 520, p = 7e-25.
- ruaccent-int8 vs kubataba: 16839 common targets, ruaccent-int8 only right 282, kubataba only right 678, p = 3e-38.
- ruaccent-int8 vs silero: 16839 common targets, ruaccent-int8 only right 190, silero only right 445, p = 2e-24.
- ruaccent-int8 vs ruaccent-silero: 16839 common targets, ruaccent-int8 only right 11, ruaccent-silero only right 335, p = 3e-84.
- ruaccent-int8 vs ruaccent-silero-int8: 16839 common targets, ruaccent-int8 only right 0, ruaccent-silero-int8 only right 328, p = 4e-99.
- ruaccent-int8 vs simple: 16839 common targets, ruaccent-int8 only right 1335, simple only right 335, p = 4e-141.
- ruaccent-big-poetry vs stressrnn: 16839 common targets, ruaccent-big-poetry only right 906, stressrnn only right 526, p = 7e-24.
- ruaccent-big-poetry vs kubataba: 16839 common targets, ruaccent-big-poetry only right 290, kubataba only right 694, p = 7e-39.
- ruaccent-big-poetry vs silero: 16839 common targets, ruaccent-big-poetry only right 204, silero only right 467, p = 1e-24.
- ruaccent-big-poetry vs ruaccent-silero: 16839 common targets, ruaccent-big-poetry only right 71, ruaccent-silero only right 403, p = 2e-57.
- ruaccent-big-poetry vs ruaccent-silero-int8: 16839 common targets, ruaccent-big-poetry only right 66, ruaccent-silero-int8 only right 402, p = 8e-60.
- ruaccent-big-poetry vs simple: 16839 common targets, ruaccent-big-poetry only right 1326, simple only right 334, p = 1e-139.
- stressrnn vs kubataba: 16839 common targets, stressrnn only right 214, kubataba only right 998, p = 3e-121.
- stressrnn vs silero: 16839 common targets, stressrnn only right 306, silero only right 949, p = 7e-77.
- stressrnn vs ruaccent-silero: 16839 common targets, stressrnn only right 278, ruaccent-silero only right 990, p = 6e-94.
- stressrnn vs ruaccent-silero-int8: 16839 common targets, stressrnn only right 274, ruaccent-silero-int8 only right 990, p = 2e-95.
- stressrnn vs simple: 16839 common targets, stressrnn only right 1408, simple only right 796, p = 3e-39.
- kubataba vs silero: 16839 common targets, kubataba only right 442, silero only right 301, p = 3e-07.
- kubataba vs ruaccent-silero: 16839 common targets, kubataba only right 357, ruaccent-silero only right 285, p = 0.005.
- kubataba vs ruaccent-silero-int8: 16839 common targets, kubataba only right 352, ruaccent-silero-int8 only right 284, p = 0.008.
- kubataba vs simple: 16839 common targets, kubataba only right 1628, simple only right 232, p = 7e-258.
- silero vs ruaccent-silero: 16839 common targets, silero only right 131, ruaccent-silero only right 200, p = 2e-04.
- silero vs ruaccent-silero-int8: 16839 common targets, silero only right 126, ruaccent-silero-int8 only right 199, p = 6e-05.
- silero vs simple: 16839 common targets, silero only right 1288, simple only right 33, p < 1e-300.
- ruaccent-silero vs ruaccent-silero-int8: 16839 common targets, ruaccent-silero only right 7, ruaccent-silero-int8 only right 11, p = 0.481.
- ruaccent-silero vs simple: 16839 common targets, ruaccent-silero only right 1344, simple only right 20, p < 1e-300.
- ruaccent-silero-int8 vs simple: 16839 common targets, ruaccent-silero-int8 only right 1348, simple only right 20, p < 1e-300.

### ru / ruslan-stressed — SILVER

Source: https://huggingface.co/datasets/stilletto/ruslan-stressed (CC-BY-4.0).  Scored: marked polysyllabic words (LLM labels).
Rows in: 22200; rows scored: 22185; targets: 176255.
ё is left unmarked by the set's convention, so words stressed on ё are not targets.
Dropped while loading: monosyllabic token (targets): 63147; no usable target (rows): 15; token carries 2+ acutes (targets): 639.
Not run: kubataba, about 2.2 s per row on this set (60 sampled rows), about 14 hours for the set; not run
Labels agree with the RUAccent accent dictionary, unambiguous entries on 90.8% of 29767 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| ruaccent | SILVER | 176255 | 0 | 0.965 | 0.965–0.966 | 0.928 (10255) | 43.6 | 1.4 |
| ruaccent-int8 | SILVER | 176255 | 0 | 0.965 | 0.964–0.966 | 0.929 (10255) | 24.2 | 1.0 |
| ruaccent-big-poetry | SILVER | 176255 | 0 | 0.964 | 0.963–0.965 | 0.908 (10255) | 59.2 | 2.3 |
| stressrnn | SILVER | 176255 | 0 | 0.917 | 0.916–0.919 | 0.807 (10255) | 0.6 | 0.1 |
| silero | SILVER | 176255 | 0 | 0.974 | 0.973–0.975 | 0.914 (10255) | 1.2 | 0.2 |
| ruaccent-silero | SILVER | 176255 | 0 | 0.978 | 0.977–0.979 | 0.930 (10255) | 62.4 | 1.7 |
| ruaccent-silero-int8 | SILVER | 176255 | 0 | 0.978 | 0.977–0.979 | 0.930 (10255) | 39.9 | 1.4 |
| simple | SILVER | 176255 | 0 | 0.902 | 0.900–0.903 | 0.000 (10255) | 0.1 | 9.4 |

McNemar (exact, two-sided) on targets both models scored:

- ruaccent vs ruaccent-int8: 176255 common targets, ruaccent only right 161, ruaccent-int8 only right 117, p = 0.010.
- ruaccent vs ruaccent-big-poetry: 176255 common targets, ruaccent only right 528, ruaccent-big-poetry only right 327, p = 6e-12.
- ruaccent vs stressrnn: 176255 common targets, ruaccent only right 11244, stressrnn only right 2743, p < 1e-300.
- ruaccent vs silero: 176255 common targets, ruaccent only right 1756, silero only right 3245, p = 9e-100.
- ruaccent vs ruaccent-silero: 176255 common targets, ruaccent only right 0, ruaccent-silero only right 2207, p < 1e-300.
- ruaccent vs ruaccent-silero-int8: 176255 common targets, ruaccent only right 52, ruaccent-silero-int8 only right 2266, p < 1e-300.
- ruaccent vs simple: 176255 common targets, ruaccent only right 13295, simple only right 2026, p < 1e-300.
- ruaccent-int8 vs ruaccent-big-poetry: 176255 common targets, ruaccent-int8 only right 579, ruaccent-big-poetry only right 422, p = 8e-07.
- ruaccent-int8 vs stressrnn: 176255 common targets, ruaccent-int8 only right 11235, stressrnn only right 2778, p < 1e-300.
- ruaccent-int8 vs silero: 176255 common targets, ruaccent-int8 only right 1749, silero only right 3282, p = 4e-105.
- ruaccent-int8 vs ruaccent-silero: 176255 common targets, ruaccent-int8 only right 59, ruaccent-silero only right 2310, p < 1e-300.
- ruaccent-int8 vs ruaccent-silero-int8: 176255 common targets, ruaccent-int8 only right 0, ruaccent-silero-int8 only right 2258, p < 1e-300.
- ruaccent-int8 vs simple: 176255 common targets, ruaccent-int8 only right 13296, simple only right 2071, p < 1e-300.
- ruaccent-big-poetry vs stressrnn: 176255 common targets, ruaccent-big-poetry only right 11191, stressrnn only right 2891, p < 1e-300.
- ruaccent-big-poetry vs silero: 176255 common targets, ruaccent-big-poetry only right 1753, silero only right 3443, p = 1e-123.
- ruaccent-big-poetry vs ruaccent-silero: 176255 common targets, ruaccent-big-poetry only right 327, ruaccent-silero only right 2735, p < 1e-300.
- ruaccent-big-poetry vs ruaccent-silero-int8: 176255 common targets, ruaccent-big-poetry only right 313, ruaccent-silero-int8 only right 2728, p < 1e-300.
- ruaccent-big-poetry vs simple: 176255 common targets, ruaccent-big-poetry only right 13094, simple only right 2026, p < 1e-300.
- stressrnn vs silero: 176255 common targets, stressrnn only right 1618, silero only right 11608, p < 1e-300.
- stressrnn vs ruaccent-silero: 176255 common targets, stressrnn only right 1372, ruaccent-silero only right 12080, p < 1e-300.
- stressrnn vs ruaccent-silero-int8: 176255 common targets, stressrnn only right 1360, ruaccent-silero-int8 only right 12075, p < 1e-300.
- stressrnn vs simple: 176255 common targets, stressrnn only right 11347, simple only right 8579, p = 8e-86.
- silero vs ruaccent-silero: 176255 common targets, silero only right 1151, ruaccent-silero only right 1869, p = 3e-39.
- silero vs ruaccent-silero-int8: 176255 common targets, silero only right 1138, ruaccent-silero-int8 only right 1863, p = 3e-40.
- silero vs simple: 176255 common targets, silero only right 13309, simple only right 551, p < 1e-300.
- ruaccent-silero vs ruaccent-silero-int8: 176255 common targets, ruaccent-silero only right 52, ruaccent-silero-int8 only right 59, p = 0.569.
- ruaccent-silero vs simple: 176255 common targets, ruaccent-silero only right 13563, simple only right 87, p < 1e-300.
- ruaccent-silero-int8 vs simple: 176255 common targets, ruaccent-silero-int8 only right 13570, simple only right 87, p < 1e-300.

## uk

Leader on each GOLD set, against the runner-up:

- russtress: silero 0.843 vs uk-accentor 0.795, McNemar p = 7e-63.
- wiktionary: silero 0.905 vs uk-accentor 0.860, McNemar p = 1e-42.

### uk / russtress — GOLD

Source: https://github.com/MashaPo/russtress (none stated).  Scored: middle word of each window.
Rows in: 12253; rows scored: 12252; targets: 12252.
Dropped while loading: gold position not on a vowel of a token (rows): 1.
Labels agree with the stressonnx simple-uk vocabulary (English Wiktionary headwords) on 88.1% of 7431 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 12252 | 0 | 0.843 | 0.837–0.850 | — | 0.2 | 0.2 |
| simple | GOLD | 12252 | 0 | 0.534 | 0.526–0.543 | — | 0.0 | 0.1 |
| uk-accentor | GOLD | 12252 | 0 | 0.795 | 0.788–0.802 | — | 24.0 | 0.2 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 12252 common targets, silero only right 4139, simple only right 353, p < 1e-300.
- silero vs uk-accentor: 12252 common targets, silero only right 943, uk-accentor only right 351, p = 7e-63.
- simple vs uk-accentor: 12252 common targets, simple only right 610, uk-accentor only right 3804, p < 1e-300.

### uk / wiktionary — GOLD

Source: https://kaikki.org/dictionary/Ukrainian/ (CC-BY-SA).  Scored: marked polysyllabic words.
Rows in: 5610; rows scored: 4374; targets: 10801.
The simple-uk vocabulary is built from the same Wiktionary, so the simple score on this set is partly self-referential.
Dropped while loading: Latin letters: transliteration or gloss (rows): 176; acute not on a vowel (targets): 4; example of an affix entry (rows): 1051; monosyllabic token (targets): 26; no usable target (rows): 9; token carries 2+ acutes (targets): 106.
Labels agree with the stressonnx simple-uk vocabulary (English Wiktionary headwords) on 96.2% of 8145 targets the dictionary covers (same source as the set, so not independent).

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 10801 | 0 | 0.905 | 0.899–0.910 | — | 0.2 | 0.2 |
| simple | GOLD | 10801 | 0 | 0.726 | 0.716–0.735 | — | 0.0 | 0.1 |
| uk-accentor | GOLD | 10801 | 0 | 0.860 | 0.854–0.868 | — | 37.7 | 0.2 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 10801 common targets, silero only right 2531, simple only right 600, p = 3e-280.
- silero vs uk-accentor: 10801 common targets, silero only right 858, uk-accentor only right 381, p = 1e-42.
- simple vs uk-accentor: 10801 common targets, simple only right 920, uk-accentor only right 2374, p = 3e-146.

### uk / uk-tts-stressed — SILVER

Source: https://huggingface.co/datasets/dmytro-m/uk-tts-stressed-serhiy (and -mykyta, -oleksa, -tetiana) (MIT).  Scored: marked polysyllabic words (dictionary + context-model labels).
Rows in: 1024; rows scored: 1023; targets: 6322.
Test and validation splits of four speakers.  The card does not name the context model that made the labels.
Dropped while loading: no usable target (rows): 1; token carries 2+ acutes (targets): 11.
Labels agree with the stressonnx simple-uk vocabulary (English Wiktionary headwords) on 95.7% of 3878 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | SILVER | 6322 | 0 | 0.904 | 0.898–0.912 | — | 0.5 | 0.3 |
| simple | SILVER | 6322 | 0 | 0.587 | 0.574–0.600 | — | 0.1 | 0.1 |
| uk-accentor | SILVER | 6322 | 0 | 0.874 | 0.865–0.882 | — | 102.1 | 0.2 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 6322 common targets, silero only right 2193, simple only right 187, p < 1e-300.
- silero vs uk-accentor: 6322 common targets, silero only right 491, uk-accentor only right 297, p = 5e-12.
- simple vs uk-accentor: 6322 common targets, simple only right 221, uk-accentor only right 2033, p < 1e-300.

## be

Leader on each GOLD set, against the runner-up:

- homographs-10x10: silero 0.543 vs simple 0.000, McNemar p = 4e-245.
- russtress: silero 0.924 vs simple 0.867, McNemar p = 2e-43.
- wiktionary: silero 0.969 vs simple 0.881, McNemar p = 5e-307.
- zascienak: silero 0.977 vs simple 0.853, McNemar p < 1e-300.

### be / homographs-10x10 — GOLD

Source: https://huggingface.co/datasets/alex73/benchmarks-stress-bel (CC-BY-SA-4.0).  Scored: the homograph only (expert-written, balanced).
Rows in: 1500; rows scored: 1495; targets: 1496.
Dropped while loading: homograph only inside a hyphenated compound (rows): 5.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 1496 | 0 | 0.543 | 0.518–0.570 | — | 0.6 | 0.2 |
| simple | GOLD | 1496 | 0 | 0.000 | 0.000–0.000 | — | 0.1 | 5.5 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 1496 common targets, silero only right 813, simple only right 0, p = 4e-245.

### be / russtress — GOLD

Source: https://github.com/MashaPo/russtress (none stated).  Scored: middle word of each window.
Rows in: 2907; rows scored: 2902; targets: 2902.
Dropped while loading: gold position not on a vowel of a token (rows): 5.
Labels agree with the stressonnx simple-be vocabulary (GrammarDB, plus English Wiktionary entries for spellings outside it) on 90.5% of 2643 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 2902 | 0 | 0.924 | 0.914–0.933 | — | 0.2 | 0.2 |
| simple | GOLD | 2902 | 0 | 0.867 | 0.854–0.878 | — | 0.0 | 5.5 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 2902 common targets, silero only right 172, simple only right 6, p = 2e-43.

### be / wiktionary — GOLD

Source: https://kaikki.org/dictionary/Belarusian/ (CC-BY-SA).  Scored: marked polysyllabic words.
Rows in: 4717; rows scored: 4618; targets: 12390.
Dropped while loading: Latin letters: transliteration or gloss (rows): 19; acute not on a vowel (targets): 8; example of an affix entry (rows): 74; monosyllabic token (targets): 102; no usable target (rows): 6; token carries 2+ acutes (targets): 28.
Labels agree with the stressonnx simple-be vocabulary (GrammarDB, plus English Wiktionary entries for spellings outside it) on 96.1% of 11067 targets the dictionary covers (same source as the set, so not independent).

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 12390 | 0 | 0.969 | 0.965–0.973 | — | 0.3 | 0.2 |
| simple | GOLD | 12390 | 0 | 0.881 | 0.875–0.888 | — | 0.0 | 5.5 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 12390 common targets, silero only right 1099, simple only right 12, p = 5e-307.

### be / zascienak — GOLD

Source: https://huggingface.co/datasets/alex73/benchmarks-stress-bel (CC-BY-SA-4.0).  Scored: marked polysyllabic words (fully hand-marked novel).
Rows in: 1538; rows scored: 1523; targets: 22965.
Dropped while loading: no usable target (rows): 15; token carries 2+ acutes (targets): 51.
Labels agree with the stressonnx simple-be vocabulary (GrammarDB, plus English Wiktionary entries for spellings outside it) on 98.3% of 19428 targets the dictionary covers.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | GOLD | 22965 | 0 | 0.977 | 0.975–0.979 | — | 1.0 | 0.2 |
| simple | GOLD | 22965 | 0 | 0.853 | 0.848–0.857 | — | 0.2 | 5.5 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 22965 common targets, silero only right 2875, simple only right 16, p < 1e-300.

### be / homographs-commonvoice — SILVER

Source: https://huggingface.co/datasets/alex73/benchmarks-stress-bel (CC-BY-SA-4.0).  Scored: the marked homographs (labelling method undocumented).
Rows in: 2930; rows scored: 2850; targets: 2877.
Dropped while loading: commented out upstream (rows): 80.
Labels agree with the stressonnx simple-be vocabulary (GrammarDB, plus English Wiktionary entries for spellings outside it) on 100.0% of 2 targets the dictionary covers.
Every target is a homograph, so a one-stress dictionary disagrees by design; this check does not measure label noise.

| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |
|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|
| silero | SILVER | 2877 | 0 | 0.829 | 0.815–0.843 | — | 0.4 | 0.2 |
| simple | SILVER | 2877 | 0 | 0.000 | 0.000–0.000 | — | 0.1 | 5.7 |

McNemar (exact, two-sided) on targets both models scored:

- silero vs simple: 2877 common targets, silero only right 2385, simple only right 0, p < 1e-300.

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
literature cited in the rule functions of `stressonnx/backends/simple.py`.

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
| kjh | last | 12018 | 0.280 | no primary source on Khakas stress could be located; Turkic-default kept |
| sah | last heavy nucleus (long vowel/diphthong), else final | 85746 | 0.994 | Yakut long vowels/diphthongs attract stress (Krueger 1962 + weight sensitivity); the old naive final rule scored 0.178 |
| be (`simple`) | none | 1891586 | — | Belarusian stress is lexical, not positional — no rule is defensible; vocabulary only |

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
| ru / simple | 3,167,126 | RUAccent pronunciation dictionary (MIT), forms with one reading | none | rule-free dictionary fallback below ruaccent/silero |
| be / simple | 1,895,356 | GrammarDB RELEASE-202601 (CC-BY-SA-4.0), spellings with one stress, plus English Wiktionary for spellings outside it | none | rule-free dictionary fallback below silero |
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
