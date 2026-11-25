# Add a language

Two paths, both documented step-by-step for newcomers in
[`export/ADDING_A_LANGUAGE.md`](../export/ADDING_A_LANGUAGE.md):

1. **You have a stressed wordlist.** Ship a `simple` language: package the
   vocabulary, declare alphabet/vowels/OOV rule, upload to the HF repo,
   register the language tag. No training, no torch.
2. **You have (or train) a neural accentor.** Export it to ONNX with the
   scripts in `export/` and add a backend entry.

All 20 upstream silero_stress vocabularies are already shipped, and
`export/build_wiktionary_vocab.py` turns any language whose Wiktionary
headwords carry stress marks into a `simple` language. That is how
Bulgarian, Macedonian, Slovene, Latvian and Ukrainian were built. Russian
came from the RUAccent pronunciation dictionary.

---

[Index](index.md)
