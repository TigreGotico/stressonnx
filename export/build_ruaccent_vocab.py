"""Build the ``ru_simple`` vocabulary from the RUAccent pronunciation dictionary.

RUAccent (Den4ikAI, https://github.com/Den4ikAI/ruaccent, MIT) publishes its
dictionary at https://huggingface.co/ruaccent/accentuator under
``dictionary/``.  ``accents.json.gz`` maps every form to its ``+``-marked
spelling (``январ+я``); ``omographs.json.gz`` lists the forms with more than
one reading.  ``accents_nn.json.gz`` is the subset the neural accentor
consults and holds about 3 percent of the forms.

The vocabulary keeps every form with exactly one reading: one ``+`` in
``accents.json.gz`` and no entry in ``omographs.json.gz``.  The stored value
is the vowel ordinal over the Cyrillic superset (vowel_ordinal_v2).

Usage::

    python export/build_ruaccent_vocab.py --accents accents.json.gz --omographs omographs.json.gz --out out/ru_simple
"""
import argparse
import collections
import gzip
import json
import os

from stressonnx._common import SCRIPT_VOWELS
from stressonnx.registry import LANGUAGES


def build(accents: dict, omographs: dict):
    alpha = set(LANGUAGES["ru"]["alpha"])
    superset = SCRIPT_VOWELS["cyrillic"]
    vocab = {}
    dropped: collections.Counter = collections.Counter()
    for word, marked in accents.items():
        if marked.count("+") != 1:
            dropped["no single +"] += 1
            continue
        if word in omographs:
            dropped["homograph"] += 1
            continue
        if marked.replace("+", "") != word:
            dropped["form mismatch"] += 1
            continue
        if any(c not in alpha for c in word):
            dropped["outside alphabet"] += 1
            continue
        idx = marked.index("+")
        if word[idx] not in superset:
            dropped["+ not before a vowel"] += 1
            continue
        vocab[word] = sum(1 for c in word[:idx] if c in superset)
    return vocab, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--accents", required=True)
    ap.add_argument("--omographs", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with gzip.open(args.accents, "rt", encoding="utf-8") as fh:
        accents = json.load(fh)
    with gzip.open(args.omographs, "rt", encoding="utf-8") as fh:
        omographs = json.load(fh)
    vocab, dropped = build(accents, omographs)

    os.makedirs(args.out, exist_ok=True)
    with gzip.open(os.path.join(args.out, "vocab.gz"), "wt", encoding="utf-8") as fh:
        for word, ordinal in sorted(vocab.items()):
            fh.write(f"{word} {ordinal}\n")
    spec = LANGUAGES["ru"]
    meta = {
        "family": "simple_accentor",
        "lang": "ru_simple",
        "alpha": spec["alpha"],
        "vowels": spec["vowels"],
        "oov_rule": spec["rule"],
        "n_vocab": len(vocab),
        "source": "RUAccent pronunciation dictionary (Den4ikAI/ruaccent, "
                  "https://huggingface.co/ruaccent/accentuator dictionary/accents.json.gz), "
                  "forms with one reading",
        "license": "MIT",
        "vocab_format": "vowel_ordinal_v2",
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(
            "# ru_simple stress vocabulary\n\n"
            "Word → stressed-vowel-ordinal pairs from the RUAccent pronunciation "
            "dictionary (Den4ikAI/ruaccent, MIT; `dictionary/accents.json.gz` of "
            "https://huggingface.co/ruaccent/accentuator), forms with one reading "
            "only.  Built by `export/build_ruaccent_vocab.py` in TigreGotico/stressonnx.  "
            "The licence notice is in `LICENSE.ruaccent`.\n"
        )
    print(f"ru_simple: {len(accents)} entries in, {len(vocab)} out, dropped {dict(dropped)}")


if __name__ == "__main__":
    main()
