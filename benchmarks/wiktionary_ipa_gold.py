"""Independent spot-check: simple-language accuracy vs Wiktionary IPA stress.

The rule-vs-own-vocabulary numbers in RESULTS.md are self-referential; this
script builds a small *independent* gold set per language from English
Wiktionary IPA transcriptions (kaikki.org dumps): entries whose IPA carries
a primary-stress mark (ˈ) and whose syllable count matches the word's
orthographic vowel count, so the stressed syllable ordinal maps directly to
a vowel ordinal.  It then scores the full simple pipeline (vocabulary +
rule) against it.

The sets are small (tens to hundreds of words) and skewed toward lemmas —
read them as a sanity check, not a benchmark.  IPA gold rows that overlap
the shipped vocabulary mostly test the vocabulary; ``--oov-only`` restricts
scoring to words the vocabulary does not contain, isolating the rule.

Usage::

    python benchmarks/wiktionary_ipa_gold.py --lang chv --jsonl Chuvash.jsonl [--oov-only]
"""
import argparse
import json
import re

from stressonnx import STRESS_TOKEN
from stressonnx.backends.simple import SimpleStressor

_IPA_JUNK = re.compile(r"[/\[\]().ˌ‿|‖]")


def ipa_gold(jsonl_path: str, stressor: SimpleStressor):
    """Yield (word, stressed_vowel_ordinal) pairs with aligned syllable counts."""
    vowels = stressor._vowels
    for line in open(jsonl_path, encoding="utf-8"):
        e = json.loads(line)
        word = (e.get("word") or "").lower()
        if not word or " " in word or "-" in word:
            continue
        word_vowels = [c for c in word if c in vowels]
        if len(word_vowels) < 2:
            continue
        for snd in e.get("sounds", []):
            ipa = snd.get("ipa") or ""
            if "ˈ" not in ipa or ipa.count("ˈ") != 1:
                continue
            body = _IPA_JUNK.sub("", ipa)
            # syllable nuclei ≈ IPA vowel letters; require an exact count
            # match with the orthographic vowels so ordinals align
            nuclei = re.findall(r"[aeiouyæɑɒɔəɘɵɤɛɜɞɪʏʊøœɶʉɯɨeoiũ̯ː]+", body)
            nuclei = [n for n in re.split(r"[^aeiouyæɑɒɔəɘɵɤɛɜɞɪʏʊøœɶʉɯɨ]+",
                                          body.replace("ː", "")) if n]
            if len(nuclei) != len(word_vowels):
                continue
            pre = body.split("ˈ")[0].replace("ː", "")
            pre_nuclei = [n for n in re.split(r"[^aeiouyæɑɒɔəɘɵɤɛɜɞɪʏʊøœɶʉɯɨ]+", pre) if n]
            yield word, len(pre_nuclei)
            break


def evaluate(lang: str, jsonl_path: str, oov_only: bool = False):
    s = SimpleStressor(lang)
    s._ensure_loaded()
    vowels = s._vowels
    n = correct = 0
    for word, gold_ord in ipa_gold(jsonl_path, s):
        if oov_only and word in s._vocab:
            continue
        out = s(word)
        mark = out.find(STRESS_TOKEN)
        if mark == -1:
            got = None
        else:
            got = sum(1 for c in out[: mark - 1] if c in vowels)
        n += 1
        correct += got == gold_ord
    return n, correct / n if n else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--oov-only", action="store_true")
    args = ap.parse_args()
    n, acc = evaluate(args.lang, args.jsonl, args.oov_only)
    scope = "OOV-only" if args.oov_only else "all"
    print(f"{args.lang} ({scope}): n={n} accuracy={acc:.3f}")


if __name__ == "__main__":
    main()
