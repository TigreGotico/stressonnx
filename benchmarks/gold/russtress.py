"""russtress context windows for ru, uk and be (GOLD).

Source: https://github.com/MashaPo/russtress, ``data/data_{ru,uk,be}.csv``.
Samples from Universal Dependencies treebanks, stress-annotated through
Yandex.Toloka crowdsourcing (Ponomareva et al. 2017, "Automated Word Stress
Detection in Russian").  Each row is a short window whose middle word is the
target; ``position`` is the character index of the stressed vowel in the
window.  The files carry no explicit license.
"""
import csv
from collections import Counter

from gold._cache import cached
from scoring import Row, Target, is_vowel

#: Pinned commit of the upstream repository.
REVISION = "8c46eac0750b1cd6eabcd5c380091d36d9aec7c4"
URL = f"https://raw.githubusercontent.com/MashaPo/russtress/{REVISION}/data/data_{{code}}.csv"


def gold_target(window: str, pos: int):
    """(token index, vowel ordinal) of the stressed vowel at *pos*, or None.

    The token containing *pos* is the target.  Scoring by vowel ordinal
    inside the token, not by absolute character index, keeps the comparison
    valid for models that substitute е→ё elsewhere in the window.
    """
    if not (0 <= pos < len(window)):
        return None
    start = window.rfind(" ", 0, pos) + 1
    end = window.find(" ", pos)
    if end == -1:
        end = len(window)
    token = window[start:end]
    offset = pos - start
    if not (0 <= offset < len(token)) or not is_vowel(token[offset]):
        return None
    ordinal = sum(1 for c in token[:offset] if is_vowel(c))
    return len(window[:start].split()), ordinal


def load(code: str):
    path = cached(URL.format(code=code), f"russtress/data_{code}.csv")
    rows = []
    drops = Counter()
    with open(path, encoding="utf-8", newline="") as fh:
        records = list(csv.DictReader(fh))
    for n, rec in enumerate(records):
        window = rec["string"]
        target = gold_target(window, int(rec["position"]))
        if target is None:
            drops["gold position not on a vowel of a token (rows)"] += 1
            continue
        rows.append(Row(f"{code}:{n}", window, (Target(*target),)))
    return rows, len(records), drops
