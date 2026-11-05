"""Rewrite a simple-accentor vocabulary in the form the runtime looks it up.

The runtime (:meth:`SimpleStressor._usable`) brings every key to NFC,
lowercases it, strips characters outside the language alphabet (the
Belarusian apostrophe, the combining dot of a lowercased Azerbaijani İ)
and drops entries whose ordinal lands on a letter outside the language's
vowels (the Belarusian entries that point at ў).  This script applies the
same pass to a vocabulary file, so the published data no longer depends on
that pass, and prints the counts in and out.

Usage::

    python export/clean_simple_vocab.py --lang be --vocab bel/vocab.gz --meta bel/meta.json --out out/bel
"""
import argparse
import gzip
import json
import os

from stressonnx.backends.simple import SimpleStressor, _load_vocab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = _load_vocab(args.vocab)
    vocab = SimpleStressor(args.lang)._usable(raw)
    os.makedirs(args.out, exist_ok=True)
    with gzip.open(os.path.join(args.out, "vocab.gz"), "wt", encoding="utf-8") as fh:
        for word, ordinal in sorted(vocab.items()):
            fh.write(f"{word} {ordinal}\n")
    with open(args.meta, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["n_vocab"] = len(vocab)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"{args.lang}: {len(raw)} entries in, {len(vocab)} out")


if __name__ == "__main__":
    main()
