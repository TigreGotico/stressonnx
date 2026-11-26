# Supported languages and models

## Why word stress?

**Word stress** (lexical stress) is which syllable of a word is pronounced
prominently: English *REcord* (noun) vs *reCORD* (verb). A text-to-speech
system must know it before it can pick the right sounds. In Russian,
unstressed vowels *reduce* (о sounds like а), so a wrong stress changes
every vowel in the word, not just the melody.

Whether you need this library depends on how your language writes stress:

| Orthography type | Examples | What you need |
|---|---|---|
| Stress is free/mobile and **not written** | Russian, Ukrainian, Belarusian, most languages here | **A stress model. this library.** Nothing in the spelling of `замок` tells you which syllable to stress. Only context does. |
| Stress is **predictable by rule** | Kyrgyz and Tatar (final syllable), Armenian, Georgian | A positional rule covers most words. That is the `simple` backend: a curated exception vocabulary plus a per-language default rule. |
| Stress is **already written** | Spanish, Greek, Portuguese | You do not need stressonnx. The orthography (accent rules) already encodes it. You may still need *sense* disambiguation, see below. |

**The Portuguese case (why "stress is written" is not the end of the story):**
Portuguese spelling pins down the stressed syllable, yet pairs like *sede*
(thirst /ˈsedɨ/ vs headquarters /ˈsɛdɨ/) share spelling and stress position
while differing in vowel **quality**, resolvable only from meaning. Our
sibling library [bifonia](https://github.com/TigreGotico/bifonia) solves that
with the same "disambiguate before G2P" idea used here: its
`add_extra_diacritics(text)` rewrites the homograph with an explicit
open/closed-vowel diacritic (*séde*/*sêde*) so any downstream phonemizer gets
it right. Rule of thumb: unwritten stress needs stressonnx. Written stress
with sense-dependent pronunciation needs a bifonia-style diacritic restorer.

---

## Supported languages

| Model id | Languages | What it is | Measured quality |
|----------|-----------|------------|------------------|
| `ruaccent` | `ru` | Homograph-aware 4-model ONNX pipeline (derived from [RUAccent](https://github.com/Den4ikAI/ruaccent), MIT) | 0.947 word accuracy, **0.833 on homographs** |
| `ruaccent-int8`, `ruaccent-silero-int8` | `ru` (opt-in) | The same pipelines with int8 weights in the two large models: a 144 MB download in place of 501 MB, 313 MB RSS in place of 652 MB, 2 to 4 times faster on homograph-heavy text. See [docs/models.md](models.md#precision) | 0.946 word accuracy, 0.825 on homographs (a homograph in a long window can flip) |
| `ruaccent-big-poetry` | `ru` | The `ruaccent` pipeline with the RUAccent `big_poetry` homograph model in place of `turbo3.1` | 0.942 word accuracy, 0.753 on homographs. Below `ruaccent`, see [docs/models.md](models.md#the-big_poetry-omograph-model) |
| `ruaccent-silero` | `ru` (default) | The `ruaccent` pipeline, then the `silero` mark on every word of two or more vowels that `ruaccent` leaves unmarked (`и́ли`, `э́то`, `бы́ло`, `что́-то`) | 0.951 word accuracy, 0.833 on homographs. 0.990 on the Wiktionary sentences, where `ruaccent` has 0.976 and `silero` 0.984. See [docs/models.md](models.md#ruaccent-silero-the-ruaccent-pipeline-with-silero-marks-on-its-unmarked-words) |
| `silero` | `uk`, `be` (defaults), `ru` | Neural ONNX pipeline exported from [silero_stress](https://github.com/snakers4/silero-stress) (MIT). The `ru` variant also restores е→ё and resolves homographs. The `uk` variant looks words up in a dictionary layer from [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) (MIT) first, and the `be` variant in one from [GrammarDB](https://github.com/Belarus/GrammarDB) (CC-BY-SA-4.0), then resolves the GrammarDB homographs with cited grammar rules and a frequency prior | ru 0.944 (0.753 on homographs) / uk 0.843 / be 0.924² |
| `stressrnn` | `ru` (selectable) | Character BiLSTM with a Zaliznyak exception dictionary, ported from [StressRNN](https://github.com/dbklim/StressRNN) (Apache-2.0. A fork of RusStress). No sentence context, no е→ё. 0.6 MB | ru 0.892, scored on the model's own training corpus |
| `uk-accentor` | `uk` (selectable) | Seq2seq Transformer ported from [ukrainian-accentor-transformer](https://github.com/Theodotus1243/ukrainian-accentor-transformer) (MIT). Rewrites the sentence with marks, the port keeps every input letter. 45 MB | uk 0.795 (silero 0.843) |
| `simple` | 26 languages¹ | Curated vocabulary + per-language positional rule. No neural inference | hy 0.935 (46 words from a JIPA Illustration), bg 0.764 (292 lemmas from the BAS dictionary; a lemma outside the vocabulary stays unmarked). The other languages have no human-labelled set |
| `kubataba` | `ru` (opt-in) | Character-level seq2seq Transformer exported from [Russian-Stress-Accent-Predictor](https://github.com/kubataba/Russian-Stress-Accent-Predictor) (MIT). Sentence-level, no homograph resolution, no ё restoration | 0.888 word accuracy (0.816 on homographs) where `silero` gets 0.944 and `ruaccent` 0.947. See docs/models.md |

¹ `ru uk be bg mk sl lv hy ka kk ky tt ba cv sah kjh tg udm mdf myv kbd xal
az-Latn az-Cyrl uz-Latn uz-Cyrl`: Russian, Ukrainian, Belarusian (all
dictionary-path), Bulgarian, Macedonian, Slovene, Latvian, Armenian,
Georgian, Kazakh, Kyrgyz, Tatar, Bashkir, Chuvash, Yakut, Khakas, Tajik,
Udmurt, Moksha, Erzya, Kabardian, Kalmyk, Azerbaijani (both scripts), Uzbek
(both scripts). Every language, including ru/uk/be, has a torch-free,
ONNX-free rule/vocabulary path (`model="simple"`), so `fallback=True` always
bottoms out in a model that needs nothing but a vocabulary file.

² Word accuracy on the russtress UD-treebank gold set, scored by
`benchmarks/run.py`: the scorer compares letters in NFC and counts a target
as a hit when the model marks exactly the gold vowel. uk uses the bundle
with the dictionary layer. be uses the bundle with the dictionary layer
and the homograph rules. Their effect is measured in
[docs/models.md](models.md#belarusian-dictionary-layer).

Numbers come from the committed, reproducible
[benchmark scoreboard](../benchmarks/RESULTS.md), which also scores the
Wiktionary example sentences, the Belarusian hand-marked sets and the
silver sets. Read its label-tier note before quoting absolutes. Defaults per
language: `ru → ruaccent-silero`, `uk/be → silero`, everything else → `simple`.
`kubataba` is reached only with `model="kubataba"`: on russtress it is
below `ruaccent` and `silero` (McNemar p = 7e-91 against `silero`, p = 2e-104
against `ruaccent`) and took about 0.4 s per row in the scoreboard run (two
onnxruntime threads on a shared host), so no default, `prefer=` strategy or
fallback picks it. The scoreboard does not run it on the two large sets
(Wiktionary, ruslan-stressed) and says so in their tables. Its port is
byte-identical to the upstream torch model on 1011 inputs.

Models are hosted on
[TigreGotico/stressonnx-models](https://huggingface.co/TigreGotico/stressonnx-models)
and download automatically on first use (see *Offline & failure behavior*).

---

[Home](index.md) · [Usage →](usage.md)
