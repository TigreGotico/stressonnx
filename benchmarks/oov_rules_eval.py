"""Measure OOV positional-rule quality against each language's vocabulary.

For every word in a ``simple``-language vocabulary (word → stressed char
index), ask: *if this word were out-of-vocabulary, would the positional rule
stress the right vowel?*  The vocabulary is curated upstream data, so this
scores the **rule**, not the vocabulary — large-n evidence for choosing
between candidate rules per language.

Usage::

    python benchmarks/oov_rules_eval.py            # score the shipped rules
    python benchmarks/oov_rules_eval.py --compare  # shipped vs candidate rules
"""
import argparse
from collections import Counter

from stressonnx import SIMPLE_LANGS
from stressonnx.backends.simple import SimpleStressor


def make_rule_fn(stressor):
    """Vowel-ordinal predictor delegating to the REAL shipped rule logic."""
    vowels = stressor._ordinal_vowels

    def predict(word):
        idx = stressor._rule_offset(word)
        if idx is None:
            return None
        return sum(1 for c in word[:idx] if c in vowels)
    return predict


def score_rule(vocab: dict, vowels, rule, restrict_multi=True):
    """Accuracy of *rule* (callable word→vowel-ordinal) over the vocabulary.

    ``restrict_multi``: score only words with ≥2 vowels — monosyllables are
    always stressed on their sole vowel by every rule, so including them
    inflates every score equally.
    """
    n = correct = 0
    for word, ordinal in vocab.items():
        n_vowels = sum(1 for c in word if c in vowels)
        if n_vowels < (2 if restrict_multi else 1):
            continue
        pred = rule(word)
        n += 1
        correct += pred == ordinal
    return correct / n if n else float("nan"), n


def load(lang: str):
    s = SimpleStressor(lang)
    s._ensure_loaded()
    return s._vocab, s._vowels, s._oov_rule


def load_with_fn(lang: str):
    s = SimpleStressor(lang)
    s._ensure_loaded()
    return s._vocab, s._ordinal_vowels, s._oov_rule, make_rule_fn(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="*", default=sorted(SIMPLE_LANGS))
    args = ap.parse_args()

    print(f"{'lang':<12} {'rule':<8} {'multi-vowel words':>17} {'rule accuracy':>14}")
    for lang in args.langs:
        vocab, vowels, rule, fn = load_with_fn(lang)
        acc, n = score_rule(vocab, vowels, fn)
        print(f"{lang:<12} {rule:<8} {n:>17} {acc:>14.3f}")


if __name__ == "__main__":
    main()
