"""Rewrite a simple-accentor vocabulary in the form the runtime looks it up.

The runtime (:meth:`SimpleStressor._usable`) brings every key to NFC,
lowercases it, strips characters outside the language alphabet (the
Belarusian apostrophe, the combining dot of a lowercased Azerbaijani İ)
and drops entries whose ordinal lands on a letter outside the language's
vowels.  This script applies the same pass to a vocabulary file, so the
published data no longer depends on that pass, and prints the counts in
and out.

An ordinal that lands on a non-vowel is repaired before the pass: it moves
to the next vowel of the language, or to the previous one when none
follows.  The Belarusian silero_stress file indexes ў itself in 421
entries; ў does not form syllables, and the stress sits on the vowel after
it in 64 of the 67 such words that the gold sets hold (3 with no vowel
after ў sit on the vowel before).

Usage::

    python export/clean_simple_vocab.py --lang be --vocab bel/vocab.gz --meta bel/meta.json --out out/bel
"""
import argparse
import gzip
import json
import os
import unicodedata

from stressonnx._common import lower_preserving_length
from stressonnx.backends.simple import SimpleStressor, _load_vocab


def repair(stressor: SimpleStressor, word: str, ordinal: int) -> int:
    """*ordinal* moved to the next vowel of the language, else the previous
    one, when it lands on a letter that is not one; unchanged otherwise."""
    lower = lower_preserving_length(unicodedata.normalize("NFC", word), stressor._turkic)
    superset = [i for i, c in enumerate(lower) if c in stressor._ordinal_vowels]
    if ordinal >= len(superset) or lower[superset[ordinal]] in stressor._vowels:
        return ordinal
    after = [n for n in range(ordinal + 1, len(superset)) if lower[superset[n]] in stressor._vowels]
    before = [n for n in range(ordinal - 1, -1, -1) if lower[superset[n]] in stressor._vowels]
    return after[0] if after else before[0] if before else ordinal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--meta", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw = _load_vocab(args.vocab)
    stressor = SimpleStressor(args.lang)
    repaired = {word: repair(stressor, word, ordinal) for word, ordinal in raw.items()}
    moved = sum(1 for word in raw if repaired[word] != raw[word])
    vocab = stressor._usable(repaired)
    os.makedirs(args.out, exist_ok=True)
    with gzip.open(os.path.join(args.out, "vocab.gz"), "wt", encoding="utf-8") as fh:
        for word, ordinal in sorted(vocab.items()):
            fh.write(f"{word} {ordinal}\n")
    with open(args.meta, encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["n_vocab"] = len(vocab)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=2)
    print(f"{args.lang}: {len(raw)} entries in, {len(vocab)} out, {moved} ordinals moved to a vowel")


if __name__ == "__main__":
    main()
