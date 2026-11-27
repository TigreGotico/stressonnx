"""alex73/benchmarks-stress-bel: Belarusian stress sets (CC-BY-SA-4.0).

Source: https://huggingface.co/datasets/alex73/benchmarks-stress-bel, built for
the BelVoice speech framework.  ``+`` after a vowel marks stress.

* ``zascienak_Malinauka.txt`` (GOLD): the first 15 chapters of Arkadz
  Charnyshevich's novel with fully manual stress marks on every word.  Every
  marked polysyllabic token is a target.
* ``10x10.tsv`` (GOLD): sentences written by experts, ten per stress variant
  of each homograph; the stressed homograph is the first column.  Every
  occurrence of the homograph in the sentence is a target.
* ``common_voice.txt`` (SILVER): Common Voice sentences with the frequent
  homographs marked.  The card does not say how the marks were made, so the
  set is not counted as human-annotated.
"""
import unicodedata
from collections import Counter

from gold._cache import cached
from scoring import STRESS, Row, Target, clean_word, marked_ordinals, marked_sentence_row, strip_stress

REVISION = "94bb5a86120feb531107132eef7466c399420c18"
BASE = f"https://huggingface.co/datasets/alex73/benchmarks-stress-bel/resolve/{REVISION}/datasets/"


def _plus_to_acute(text: str) -> str:
    return unicodedata.normalize("NFC", text).replace("+", STRESS)


def _lines(name: str) -> list:
    path = cached(BASE + name, f"belarusian_homographs/{name}")
    return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines()]


def load_sentences(name: str):
    """Rows from a file of ``+``-marked lines (zascienak, common_voice).

    Lines starting with ``#`` are commented out upstream and skipped, as the
    upstream benchmark script does.
    """
    lines = [ln for ln in _lines(name) if ln]
    rows = []
    drops = Counter()
    commented = sum(1 for ln in lines if ln.startswith("#"))
    if commented:
        drops["commented out upstream (rows)"] = commented
    lines = [ln for ln in lines if not ln.startswith("#")]
    for n, line in enumerate(lines):
        row = marked_sentence_row(f"{name}:{n}", _plus_to_acute(line), drops)
        if row is not None:
            rows.append(row)
    return rows, len(lines) + commented, drops


def load_10x10():
    lines = [ln for ln in _lines("10x10.tsv") if "\t" in ln]
    rows = []
    drops = Counter()
    for n, line in enumerate(lines):
        stressed, sentence = line.split("\t", 1)
        marks = marked_ordinals(_plus_to_acute(stressed))
        if len(marks) != 1 or marks[0] < 0:
            drops["stressed form has no single acute (rows)"] += 1
            continue
        sentence = strip_stress(sentence)
        word = clean_word(stressed)
        tokens = sentence.split()
        hits = [i for i, tok in enumerate(tokens) if clean_word(tok) == word]
        if not hits:
            parts = [clean_word(part) for tok in tokens for part in tok.split("-")]
            reason = ("homograph only inside a hyphenated compound (rows)" if word in parts
                      else "homograph not found in the sentence (rows)")
            drops[reason] += 1
            continue
        rows.append(Row(f"10x10:{n}", sentence, tuple(Target(i, marks[0]) for i in hits)))
    return rows, len(lines), drops
