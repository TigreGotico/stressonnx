"""Word-stress scoring: rows, per-target hits, paired bootstrap, exact McNemar.

A *row* is one model input (a context window or a sentence).  A *target* is
one whitespace token of that row whose stressed vowel the gold set knows,
stored as the vowel ordinal inside the token.  A stress function maps text to
text with U+0301 after each stressed vowel; any callable works, so a
competitor plugs in the same way as a stressonnx model.

Ordinals are counted on the NFC form of a token.  NFD would split й and ў
into a vowel letter plus a breve, and count a vowel that is not there.
"""
import math
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

STRESS = "\u0301"
GRAVE = "\u0300"
#: Union of the Russian, Ukrainian and Belarusian vowel letters (ў is a glide).
VOWELS = frozenset("аеёиоуыэюя" "іїє")

DROP_LETTERS = "model changed the letters of the row (targets)"
DROP_CALL_FAILED = "stress function raised (targets)"


def is_vowel(c: str) -> bool:
    return c.lower() in VOWELS


def strip_stress(text: str) -> str:
    """*text* without acute or grave marks, in NFC."""
    nfd = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", nfd.replace(STRESS, "").replace(GRAVE, ""))


def marked_ordinals(token: str) -> list:
    """Vowel ordinal of every acute in *token*; -1 for an acute after a non-vowel."""
    token = unicodedata.normalize("NFC", token)
    marks = []
    seen = 0
    prev_vowel = None
    for c in token:
        if c == STRESS:
            marks.append(prev_vowel if prev_vowel is not None else -1)
            prev_vowel = None
            continue
        if is_vowel(c):
            prev_vowel = seen
            seen += 1
        elif not unicodedata.combining(c):
            prev_vowel = None
    return marks


def _letter_key(c: str) -> str:
    return "е" if c in "ёЁ" else c.lower()


def align(text: str, output: str):
    """*output* cut into one piece per whitespace token of *text*, or None.

    Pieces are cut by letters, not by whitespace, so a model that only
    changes spacing or punctuation stays aligned.  Each piece holds the
    token's letters and the marks that follow them.  The output must keep
    the input's letters in order (е↔ё substitutions allowed); otherwise the
    alignment is lost and None is returned.
    """
    tokens = text.split()
    want = [_letter_key(c) for c in unicodedata.normalize("NFC", text) if c.isalpha()]
    sizes = [sum(1 for c in unicodedata.normalize("NFC", t) if c.isalpha()) for t in tokens]
    owner = [i for i, n in enumerate(sizes) for _ in range(n)]
    pieces = [[] for _ in tokens]
    got = 0
    current = 0
    for c in unicodedata.normalize("NFC", output):
        if c.isalpha():
            if got >= len(want) or _letter_key(c) != want[got]:
                return None
            current = owner[got]
            got += 1
        if tokens:
            pieces[current].append(c)
    if got != len(want):
        return None
    return ["".join(p) for p in pieces]


def vowel_count(token: str) -> int:
    return sum(1 for c in unicodedata.normalize("NFC", token) if is_vowel(c))


def predicted_ordinals(token: str) -> frozenset:
    """Stressed-vowel ordinals a model output token claims.

    An unmarked token with exactly one ё claims the ё: ё carries the stress
    by orthographic convention, and some models leave it unmarked.
    """
    marks = marked_ordinals(token)
    if marks:
        return frozenset(marks)
    token = unicodedata.normalize("NFC", token).lower()
    if token.count("ё") == 1:
        return frozenset({sum(1 for c in token[:token.index("ё")] if is_vowel(c))})
    return frozenset()


def clean_word(token: str) -> str:
    """Lowercase letters of *token* only, stress marks removed."""
    return "".join(c for c in strip_stress(token).lower() if c.isalpha())


@dataclass(frozen=True)
class Target:
    token_index: int
    ordinal: int


@dataclass(frozen=True)
class Row:
    id: str
    text: str
    targets: tuple


def marked_sentence_row(row_id: str, marked: str, drops: Counter):
    """Row from a sentence with acute marks; every marked polysyllabic token is a target.

    Tokens with two or more acutes are ambiguous and are dropped, counted in
    *drops*.  Returns None when no target remains.
    """
    tokens = marked.split()
    targets = []
    for i, tok in enumerate(tokens):
        marks = marked_ordinals(tok)
        if not marks:
            continue
        if vowel_count(tok) < 2:
            drops["monosyllabic token (targets)"] += 1
            continue
        if len(marks) > 1:
            drops["token carries 2+ acutes (targets)"] += 1
            continue
        if marks[0] < 0:
            drops["acute not on a vowel (targets)"] += 1
            continue
        targets.append(Target(i, marks[0]))
    if not targets:
        drops["no usable target (rows)"] += 1
        return None
    return Row(row_id, strip_stress(marked), tuple(targets))


@dataclass
class Scored:
    """Per-row outcome of one stress function on one set."""
    rows: list
    elapsed_s: float
    calls: int

    def hits(self) -> dict:
        """(row id, target index) → hit, for every scored target."""
        return {
            (r["id"], t["i"]): t["hit"]
            for r in self.rows for t in r["targets"] if t["hit"] is not None
        }

    def summary(self) -> dict:
        n = correct = homo_n = homo_correct = 0
        drops = Counter()
        for r in self.rows:
            if r["dropped"]:
                drops[r["dropped"]] += len(r["targets"])
                continue
            for t in r["targets"]:
                n += 1
                correct += t["hit"]
                if t["homograph"]:
                    homo_n += 1
                    homo_correct += t["hit"]
        return {
            "n": n,
            "correct": correct,
            "accuracy": correct / n if n else float("nan"),
            "homo_n": homo_n,
            "homo_acc": homo_correct / homo_n if homo_n else None,
            "dropped": dict(drops),
            "ms_per_row": 1000 * self.elapsed_s / max(1, self.calls),
        }


def score(stress_fn: Callable, rows: Iterable, homograph_words=None) -> Scored:
    """Run *stress_fn* on every row and score each target.

    A target is a hit when the model output token carries exactly the gold
    vowel and no other.  When the output changes the row's letters, the
    alignment is lost and the row's targets are dropped, never scored.
    """
    out_rows = []
    elapsed = 0.0
    calls = 0
    for row in rows:
        tokens = row.text.split()
        record = {"id": row.id, "output": None, "dropped": None, "targets": []}
        pieces = None
        t0 = time.perf_counter()
        try:
            out = stress_fn(row.text)
        except Exception as exc:  # a competitor may fail on one input; count it
            out = None
            record["error"] = f"{type(exc).__name__}: {exc}"
        elapsed += time.perf_counter() - t0
        calls += 1
        record["output"] = out
        if out is None:
            record["dropped"] = DROP_CALL_FAILED
        elif (pieces := align(row.text, out)) is None:
            record["dropped"] = DROP_LETTERS
        for k, tgt in enumerate(row.targets):
            word = clean_word(tokens[tgt.token_index])
            entry = {
                "i": k,
                "token": tgt.token_index,
                "word": word,
                "gold": tgt.ordinal,
                "pred": None,
                "hit": None,
                "homograph": bool(homograph_words) and word in homograph_words,
            }
            if record["dropped"] is None:
                pred = predicted_ordinals(pieces[tgt.token_index])
                entry["pred"] = sorted(pred)
                entry["hit"] = pred == {tgt.ordinal}
            record["targets"].append(entry)
        out_rows.append(record)
    return Scored(out_rows, elapsed, calls)


def bootstrap_ci(scored: dict, n_resamples: int = 1000, seed: int = 0) -> dict:
    """Paired percentile 95% CI of accuracy for each model in *scored*.

    Rows are the resampling unit (a sentence's targets move together), and
    every model sees the same resampled rows.  *scored* maps a model name to
    its :class:`Scored`; all models must have scored the same row list.
    """
    names = list(scored)
    n_rows = len(next(iter(scored.values())).rows)
    hits = {}
    totals = {}
    for name in names:
        h = np.zeros(n_rows)
        t = np.zeros(n_rows)
        for i, r in enumerate(scored[name].rows):
            if r["dropped"]:
                continue
            t[i] = len(r["targets"])
            h[i] = sum(x["hit"] for x in r["targets"])
        hits[name], totals[name] = h, t
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(n_resamples) for name in names}
    for b in range(n_resamples):
        idx = rng.integers(0, n_rows, n_rows)
        for name in names:
            denom = totals[name][idx].sum()
            samples[name][b] = hits[name][idx].sum() / denom if denom else np.nan
    return {
        name: (float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5)))
        for name, s in samples.items()
    }


def mcnemar_exact(a: dict, b: dict) -> dict:
    """Exact two-sided McNemar test on the targets both models scored.

    *a* and *b* map a target key to a hit flag.  ``b_only``/``a_only`` count
    the discordant pairs; ``p`` is the two-sided binomial p-value.
    """
    common = a.keys() & b.keys()
    a_only = sum(1 for k in common if a[k] and not b[k])
    b_only = sum(1 for k in common if b[k] and not a[k])
    n = a_only + b_only
    k = min(a_only, b_only)
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"common": len(common), "a_only": a_only, "b_only": b_only, "p": p}
