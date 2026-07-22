"""Benchmark harness: word-level stress accuracy + latency per (lang, model).

Usage (from a checkout, with the ``test`` extra installed)::

    python benchmarks/run.py [--out benchmarks/RESULTS.md] [--date YYYY-MM-DD]

Gold data
---------
Stress-annotated context windows from the **russtress** project
(https://github.com/MashaPo/russtress, ``data/data_{ru,uk,be}.csv``):
samples from Universal Dependencies treebanks, stress-annotated via
Yandex.Toloka crowdsourcing (Ponomareva et al. 2017, "Automated Word Stress
Detection in Russian").  Each row is a short character window whose *middle
word* is the annotation target; ``position`` is the character index of the
stressed vowel inside the window.  The files carry no explicit license, so
they are fetched at run time and never committed to this repository.

Scoring
-------
A prediction is correct when the model's combining-acute mark on the middle
word sits on the same character the gold row marks.  Rows whose middle word
carries no vowel, or where the gold position does not point inside the middle
word, are dropped (counted, reported).  The Russian homograph subset is the
rows whose middle word appears in the RUAccent omograph dictionary.

The 20 simple-accentor-only languages have no independent gold available —
they are reported with vocabulary size only (explicitly, not silently).
"""
import argparse
import csv
import io
import time
import unicodedata
import urllib.request

from stressonnx import STRESS_TOKEN, make_stressor

RUSSTRESS_RAW = "https://raw.githubusercontent.com/MashaPo/russtress/master/data/data_{code}.csv"

#: (stressonnx lang, russtress file code, model ids to evaluate)
TARGETS = [
    ("ru", "ru", ["ruaccent", "silero", "kubataba"]),
    ("ukr", "uk", ["silero"]),
    ("bel", "be", ["silero"]),
    ("bel_simple", "be", ["simple"]),
]

# kubataba decodes autoregressively (~1 s/row) — cap its rows, loudly.
KUBATABA_CAP = 1000


def fetch_gold(code: str) -> list:
    with urllib.request.urlopen(RUSSTRESS_RAW.format(code=code)) as resp:
        text = resp.read().decode("utf-8")
    rows = []
    for row in csv.DictReader(io.StringIO(text)):
        window, pos = row["string"], int(row["position"])
        rows.append((window, pos))
    return rows


def gold_target(window: str, pos: int):
    """Decode a gold row → (token_index, vowel_ordinal) of the stressed vowel.

    ``pos`` is a character index into the whitespace-tokenized window; the
    token containing it is the annotation target.  Scoring by vowel ordinal
    inside the token (not absolute char index) makes the comparison robust
    to backends that strip punctuation or substitute е→ё elsewhere in the
    window, as long as the token count is preserved.
    """
    if not (0 <= pos < len(window)):
        return None
    start = window.rfind(" ", 0, pos) + 1
    end = window.find(" ", pos)
    if end == -1:
        end = len(window)
    token = window[start:end]
    offset = pos - start
    if not (0 <= offset < len(token)):
        return None
    vowel_ordinal = sum(1 for c in token[:offset] if _is_vowelish(c))
    if not _is_vowelish(token[offset]):
        return None  # gold points at a non-vowel char — unusable row
    token_index = len(window[:start].split())
    return token_index, vowel_ordinal


# Union of Russian, Ukrainian and Belarusian vowel letters (ў is a glide)
_VOWELISH = set("аеёиоуыэюя" + "іїє")


def _is_vowelish(c: str) -> bool:
    return c.lower() in _VOWELISH


def predicted_vowel_ordinal(stressed_window: str, token_index: int):
    """Stressed-vowel ordinal of token #token_index in the model output.

    Returns (ordinal | None, token_count) — None when the token is unmarked.
    е→ё substitution keeps the vowel count stable, so ordinals align.
    """
    stressed_window = unicodedata.normalize("NFD", stressed_window)
    tokens = stressed_window.split()
    if token_index >= len(tokens):
        return None, len(tokens)
    token = tokens[token_index]
    mark = token.find(STRESS_TOKEN)
    if mark == -1:
        return None, len(tokens)
    ordinal = sum(1 for c in token[:mark - 1] if _is_vowelish(c))
    return ordinal, len(tokens)


def evaluate(lang: str, model: str, gold: list, homograph_words=None):
    stressor = make_stressor(model=model, lang=lang)
    t0 = time.perf_counter()
    stressor("тест" if lang != "ukr" else "тест")
    cold_s = time.perf_counter() - t0

    rows = gold[:KUBATABA_CAP] if model == "kubataba" else gold
    capped = len(gold) - len(rows)

    total = correct = dropped = 0
    homo_total = homo_correct = 0
    t0 = time.perf_counter()
    for window, pos in rows:
        target = gold_target(window, pos)
        if target is None:
            dropped += 1
            continue
        token_index, gold_ordinal = target
        tokens = window.split()
        word = tokens[token_index]
        clean = "".join(c for c in word.lower() if c.isalpha())
        out = stressor(window)
        got_ordinal, out_token_count = predicted_vowel_ordinal(out, token_index)
        if out_token_count != len(tokens):
            # backend rewrote the window structurally (dropped a token) —
            # token alignment is lost, row unusable
            dropped += 1
            continue
        total += 1
        hit = got_ordinal == gold_ordinal
        correct += hit
        if homograph_words and clean in homograph_words:
            homo_total += 1
            homo_correct += hit
    elapsed = time.perf_counter() - t0

    return {
        "lang": lang,
        "model": model,
        "n": total,
        "accuracy": correct / total if total else float("nan"),
        "homo_n": homo_total,
        "homo_acc": homo_correct / homo_total if homo_total else None,
        "dropped": dropped,
        "capped": capped,
        "cold_load_s": cold_s,
        "ms_per_row": 1000 * elapsed / max(1, total + dropped),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/RESULTS.md")
    ap.add_argument("--date", required=True, help="ISO date stamped into the report")
    args = ap.parse_args()

    # homograph roster for the ru subset + dictionary for the noise estimate
    from stressonnx.backends.ruaccent import RuAccentStressor
    ra = RuAccentStressor()
    ra._ensure_loaded()
    homographs = set(ra._omographs)

    def gold_noise_estimate(gold):
        """Share of dictionary-covered unambiguous targets where the gold
        annotation agrees with the RUAccent pronunciation dictionary — an
        upper-bound sanity check on annotation quality."""
        n = agree = 0
        for w, p in gold:
            t = gold_target(w, p)
            if not t:
                continue
            ti, go = t
            word = w.split()[ti].lower()
            plus = ra._accents.get(word)
            if not plus or plus.count("+") != 1:
                continue
            i = plus.index("+")
            n += 1
            agree += sum(1 for c in plus[:i] if _is_vowelish(c)) == go
        return agree / n if n else float("nan"), n

    gold_cache = {}
    results = []
    noise = {}
    for lang, code, models in TARGETS:
        if code not in gold_cache:
            gold_cache[code] = fetch_gold(code)
            print(f"gold {code}: {len(gold_cache[code])} rows")
            if code == "ru":
                noise[code] = gold_noise_estimate(gold_cache[code])
        for model in models:
            print(f"evaluating {lang}/{model} …", flush=True)
            res = evaluate(lang, model, gold_cache[code],
                           homograph_words=homographs if lang == "ru" else None)
            print("  ", res)
            results.append(res)

    lines = [
        "# Benchmark scoreboard",
        "",
        f"Run date: {args.date}.  Reproduce: `python benchmarks/run.py --date <date>`.",
        "",
        "Gold: stress-annotated UD-treebank windows from",
        "[russtress](https://github.com/MashaPo/russtress) (Ponomareva et al. 2017,",
        "Yandex.Toloka annotation; fetched at run time — see provenance note in",
        "`benchmarks/run.py`).  Accuracy = predicted stressed character equals the",
        "gold annotation for the window's target word.  `homograph acc` scores the",
        "subset of target words present in the RUAccent omograph dictionary.",
        "",
        "| lang | model | rows | accuracy | homograph rows | homograph acc | cold load (s) | ms/row (warm) |",
        "|------|-------|------|----------|----------------|---------------|---------------|---------------|",
    ]
    for r in results:
        homo = f"{r['homo_acc']:.3f}" if r["homo_acc"] is not None else "—"
        homo_n = r["homo_n"] or "—"
        note = f" (first {KUBATABA_CAP} rows)" if r["capped"] else ""
        lines.append(
            f"| {r['lang']} | {r['model']}{note} | {r['n']} | {r['accuracy']:.3f} "
            f"| {homo_n} | {homo} | {r['cold_load_s']:.1f} | {r['ms_per_row']:.1f} |"
        )
    lines += [
        "",
        "**Annotation-noise ceiling:** on Russian targets covered by the RUAccent",
        "pronunciation dictionary with a single unambiguous stress, the gold",
        "annotation agrees with the dictionary only "
        f"{noise['ru'][0]:.1%} of the time ({noise['ru'][1]} words checked) —",
        "crowdsourced noise plus genuine variant stress (го́ду/году́).  Absolute",
        "accuracies are bounded by that ceiling; *relative* comparison between",
        "models on identical rows remains meaningful.",
        "",
        "Rows dropped per evaluation (no vowel in target / model rewrote the window"
        " structurally): " + ", ".join(f"{r['lang']}/{r['model']}: {r['dropped']}" for r in results) + ".",
        "",
        "## Languages without independent gold",
        "",
        "The 20 `simple`-model languages ship curated vocabularies exported from",
        "silero_stress; no independent stress-annotated corpus is publicly",
        "available for them, so no accuracy row is reported (the parity test in",
        "`tests/test_export_parity.py` locks them to the upstream reference).",
    ]
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
