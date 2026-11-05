"""Benchmark harness: word-stress accuracy, confidence and latency per (lang, set, model).

Usage (from a checkout, with the ``test`` extra installed)::

    python benchmarks/run.py --date YYYY-MM-DD [--lang ru uk be]
        [--targets ruaccent silero simple] [--sets russtress wiktionary ...]
        [--external NAME=COMMAND ...] [--raw-dir DIR] [--out benchmarks/RESULTS.md]
        [--threads 2] [--no-report | --report-only]

Each (lang, set) run writes ``<raw-dir>/<lang>__<set>.rows.json.gz`` (every
model output and target verdict) and ``<lang>__<set>.summary.json``.  The
report is rebuilt from every summary in the raw directory, so languages can
run in separate processes.  ``--external`` scores a command that speaks the JSON-lines
protocol in ``benchmarks/external.py``, for a model in another environment.

Evaluation sets live in ``benchmarks/gold/``, each with its provenance and
label tier.  Data is downloaded at run time to ``~/.cache/stressonnx-bench/``
and never committed.

Scoring (``benchmarks/scoring.py``): a target is one gold-marked word; it is
a hit when the model marks exactly the gold vowel of that word and no other.
Accuracy has a paired row-bootstrap 95% CI (1000 resamples, seed 0), and
every pair of models gets an exact McNemar test on the targets both scored.
"""
import argparse
import gzip
import json
import time
from functools import cache
from itertools import combinations
from pathlib import Path

from stressonnx import make_stressor

from external import JsonlCommand
from gold import GOLD, SETS, SILVER, sets_for, wiktionary
from gold._cache import CACHE
from scoring import bootstrap_ci, clean_word, is_vowel, mcnemar_exact, score

DEFAULT_TARGETS = {
    "ru": ["ruaccent", "silero", "simple"],
    "uk": ["silero", "simple"],
    "be": ["silero", "simple"],
}
BEGIN = "<!-- run.py:begin -->"
END = "<!-- run.py:end -->"
#: First section of RESULTS.md that other scripts produce.
FOREIGN_SECTION = "## OOV positional rules"


@cache
def ruaccent_dictionaries():
    """RUAccent omograph roster and unambiguous accent dictionary (word → ordinal)."""
    from stressonnx.backends.ruaccent import RuAccentStressor
    ra = RuAccentStressor()
    ra._ensure_loaded()
    accents = {}
    for word, plus in ra._accents.items():
        if plus.count("+") == 1:
            i = plus.index("+")
            accents[word] = sum(1 for c in plus[:i] if is_vowel(c))
    return set(ra._omographs), accents


@cache
def simple_vocabulary(lang: str) -> dict:
    from stressonnx.backends.simple import SimpleStressor
    s = SimpleStressor(lang)
    s._ensure_loaded()
    return s._vocab


#: lang → (dictionary description, loader, set names the dictionary is NOT independent of)
DICTIONARIES = {
    "ru": ("RUAccent accent dictionary, unambiguous entries", lambda: ruaccent_dictionaries()[1], ()),
    "uk": ("stressonnx simple-uk vocabulary (English Wiktionary headwords)",
           lambda: simple_vocabulary("uk"), ("wiktionary",)),
    "be": ("stressonnx simple-be vocabulary (exported from silero_stress)",
           lambda: simple_vocabulary("be"), ()),
}


def dictionary_agreement(rows, dictionary: dict):
    """Share of targets found in *dictionary* whose gold ordinal agrees with it."""
    n = agree = 0
    for row in rows:
        tokens = row.text.split()
        for t in row.targets:
            ordinal = dictionary.get(clean_word(tokens[t.token_index]))
            if ordinal is None:
                continue
            n += 1
            agree += ordinal == t.ordinal
    return {"agree": agree / n if n else None, "checked": n}


def cap_onnx_threads(n: int) -> None:
    """Build every onnxruntime session with *n* intra-op threads.

    Fixed threads make latency comparable between runs and keep parallel
    benchmark processes from oversubscribing a shared machine.
    """
    import onnxruntime as ort
    base = ort.InferenceSession

    def session(path, sess_options=None, **kwargs):
        opts = sess_options or ort.SessionOptions()
        opts.intra_op_num_threads = n
        opts.inter_op_num_threads = 1
        return base(path, sess_options=opts, **kwargs)

    ort.InferenceSession = session


def stressonnx_fn(model: str, lang: str):
    """(stress function, cold load seconds, closer) for a stressonnx model."""
    t0 = time.perf_counter()
    stressor = make_stressor(model=model, lang=lang)
    stressor("тест")
    return stressor, time.perf_counter() - t0, None


def external_fn(command: str, lang: str):
    t0 = time.perf_counter()
    proc = JsonlCommand(command, lang)
    proc("тест")
    return proc, time.perf_counter() - t0, proc.close


def run_set(evs, models: dict, homographs, dictionary, raw_dir: Path, threads: int):
    rows, n_in, load_drops = evs.load()
    print(f"[{evs.lang}/{evs.name}] {evs.tier}: {n_in} in, {len(rows)} rows, "
          f"{sum(len(r.targets) for r in rows)} targets, load drops {dict(load_drops)}", flush=True)
    if evs.name == "wiktionary" and len(rows) < wiktionary.MIN_SENTENCES:
        print(f"  skipped: fewer than {wiktionary.MIN_SENTENCES} usable sentences")
        return None
    scored = {}
    summaries = {}
    for name, factory in models.items():
        fn, cold_s, close = factory()
        result = score(fn, rows, homographs)
        if close:
            close()
        scored[name] = result
        summaries[name] = {**result.summary(), "cold_load_s": cold_s}
        print(f"  {name}: {summaries[name]}", flush=True)
    for name, ci in bootstrap_ci(scored).items():
        summaries[name]["ci95"] = ci
    hits = {name: s.hits() for name, s in scored.items()}
    mcnemar = [{"a": a, "b": b, **mcnemar_exact(hits[a], hits[b])}
               for a, b in combinations(scored, 2)]
    desc, dict_loader, not_independent = dictionary
    check = {"dictionary": desc, "independent": evs.name not in not_independent,
             **dictionary_agreement(rows, dict_loader())}
    result = {
        "lang": evs.lang, "set": evs.name, "tier": evs.tier, "license": evs.license,
        "source": evs.source, "scored": evs.scored, "homographs_only": evs.homographs_only,
        "n_in": n_in, "rows": len(rows), "targets": sum(len(r.targets) for r in rows),
        "load_drops": dict(load_drops), "dictionary_check": check,
        "models": summaries, "mcnemar": mcnemar, "threads": threads,
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{evs.lang}__{evs.name}"
    with gzip.open(raw_dir / f"{stem}.rows.json.gz", "wt", encoding="utf-8") as fh:
        json.dump({name: s.rows for name, s in scored.items()}, fh, ensure_ascii=False)
    (raw_dir / f"{stem}.summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


def _fmt_p(p: float) -> str:
    if p == 0:
        return "p < 1e-300"
    return f"p = {p:.3f}" if p >= 0.001 else f"p = {p:.0e}"


def _leader(r: dict) -> str:
    ranked = sorted(r["models"], key=lambda m: r["models"][m]["accuracy"], reverse=True)
    first, second = ranked[:2]
    test = next(t for t in r["mcnemar"] if {t["a"], t["b"]} == {first, second})
    return (f"{first} {r['models'][first]['accuracy']:.3f} vs {second} "
            f"{r['models'][second]['accuracy']:.3f}, McNemar {_fmt_p(test['p'])}.")


def render(results: list, date: str) -> list:
    notes = {(s.lang, s.name): s.note for s in SETS}
    threads = "/".join(sorted({str(r["threads"]) for r in results}))
    lines = [
        BEGIN,
        "# Benchmark scoreboard",
        "",
        f"Run date: {date}.  Reproduce: `python benchmarks/run.py --date <date>`;",
        "languages can run as separate processes (`--lang`), see the script docstring.",
        "",
        "Each table scores the stressonnx models on one evaluation set.  A *target*",
        "is one word whose stress the set marks; it is a hit when the model marks",
        "exactly that vowel and nothing else in the word.  Words where ё carries",
        "the stress and the model leaves ё unmarked count as marked on ё.  A target",
        "the model leaves unmarked is a miss, so a low score can mean abstention.",
        "`95% CI` is a paired bootstrap over rows (1000 resamples, seed 0).",
        "`homograph acc` scores the targets in the RUAccent omograph roster (ru).",
        "`dropped` counts targets lost because the model changed the letters of",
        "the row, so its output no longer aligns with the input.  `ms/row` is warm",
        f"latency per model call with {threads} onnxruntime thread(s) per session.",
        "",
        f"**Label tiers.**  {GOLD}: labels written by people.  {SILVER}: labels made",
        "by a machine (an LLM, a dictionary tool, a context model) or by an",
        "undocumented method.  Silver rows show agreement with that labeller, not",
        "correctness, and no claim that one model is best rests on them.",
        "",
        "The training data of silero and ruaccent is unpublished, so overlap with",
        "these sets cannot be established.",
        "",
    ]
    for lang in ("ru", "uk", "be"):
        lang_results = [r for r in results if r["lang"] == lang]
        if not lang_results:
            continue
        lang_results.sort(key=lambda r: (r["tier"] != GOLD, r["set"]))
        lines += [f"## {lang}", "", "Leader on each GOLD set, against the runner-up:", ""]
        for r in lang_results:
            if r["tier"] == GOLD and len(r["models"]) > 1:
                lines.append(f"- {r['set']}: {_leader(r)}")
        lines.append("")
        for r in lang_results:
            lines += [
                f"### {lang} / {r['set']} — {r['tier']}",
                "",
                f"Source: {r['source']} ({r['license']}).  Scored: {r['scored']}.",
                f"Rows in: {r['n_in']}; rows scored: {r['rows']}; targets: {r['targets']}.",
            ]
            if notes.get((lang, r["set"])):
                lines.append(notes[(lang, r["set"])])
            if r["load_drops"]:
                drops = "; ".join(f"{k}: {v}" for k, v in sorted(r["load_drops"].items()))
                lines.append(f"Dropped while loading: {drops}.")
            check = r["dictionary_check"]
            if check["checked"]:
                note = "" if check["independent"] else " (same source as the set, so not independent)"
                lines.append(
                    f"Labels agree with the {check['dictionary']} on {check['agree']:.1%} of "
                    f"{check['checked']} targets the dictionary covers{note}.")
                if r["homographs_only"]:
                    lines.append("Every target is a homograph, so a one-stress dictionary "
                                 "disagrees by design; this check does not measure label noise.")
            lines += [
                "",
                "| model | tier | targets | dropped | accuracy | 95% CI | homograph acc (n) | ms/row | cold load (s) |",
                "|-------|------|---------|---------|----------|--------|-------------------|--------|---------------|",
            ]
            for name, m in r["models"].items():
                homo = f"{m['homo_acc']:.3f} ({m['homo_n']})" if m["homo_acc"] is not None else "—"
                lo, hi = m["ci95"]
                dropped = sum(m["dropped"].values())
                lines.append(
                    f"| {name} | {r['tier']} | {m['n']} | {dropped} | {m['accuracy']:.3f} "
                    f"| {lo:.3f}–{hi:.3f} | {homo} | {m['ms_per_row']:.1f} | {m['cold_load_s']:.1f} |")
            if r["mcnemar"]:
                lines += ["", "McNemar (exact, two-sided) on targets both models scored:", ""]
                for t in r["mcnemar"]:
                    lines.append(
                        f"- {t['a']} vs {t['b']}: {t['common']} common targets, "
                        f"{t['a']} only right {t['a_only']}, {t['b']} only right {t['b_only']}, "
                        f"{_fmt_p(t['p'])}.")
            lines.append("")
    lines += [
        "## Languages without independent gold",
        "",
        "The 20 `simple`-model languages ship curated vocabularies exported from",
        "silero_stress; no independent stress-annotated corpus is publicly",
        "available for them, so no accuracy row is reported (the parity test in",
        "`tests/test_export_parity.py` locks them to the upstream reference).",
        "",
        END,
    ]
    return lines


def write_report(out: Path, lines: list) -> None:
    """Replace the run.py block of *out*, keeping sections other scripts produce."""
    block = "\n".join(lines) + "\n"
    old = out.read_text(encoding="utf-8") if out.exists() else ""
    if BEGIN in old and END in old:
        head, rest = old.split(BEGIN, 1)
        tail = rest.split(END, 1)[1].lstrip("\n")
    elif FOREIGN_SECTION in old:
        head, tail = "", old[old.index(FOREIGN_SECTION):]
    else:
        head, tail = "", ""
    out.write_text(head + block + ("\n" + tail if tail else ""), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="ISO date stamped into the report")
    ap.add_argument("--lang", nargs="+", default=list(DEFAULT_TARGETS))
    ap.add_argument("--targets", nargs="+", help="stressonnx model ids (default: all per language)")
    ap.add_argument("--sets", nargs="+", help="evaluation set names (default: all for the language)")
    ap.add_argument("--external", action="append", default=[], metavar="NAME=COMMAND",
                    help="score a JSON-lines command (see benchmarks/external.py)")
    ap.add_argument("--raw-dir", type=Path, default=CACHE / "results")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/RESULTS.md"))
    ap.add_argument("--threads", type=int, default=2, help="onnxruntime intra-op threads per session")
    ap.add_argument("--no-report", action="store_true", help="write raw results only")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild the report from the raw directory without running models")
    args = ap.parse_args()

    cap_onnx_threads(args.threads)
    externals = dict(spec.split("=", 1) for spec in args.external)
    for lang in [] if args.report_only else args.lang:
        homographs = ruaccent_dictionaries()[0] if lang == "ru" else None
        models = {}
        for model in args.targets or DEFAULT_TARGETS[lang]:
            if model not in DEFAULT_TARGETS[lang]:
                ap.error(f"{model!r} is not a stressonnx model for {lang}: {DEFAULT_TARGETS[lang]}")
            models[model] = lambda m=model, l=lang: stressonnx_fn(m, l)
        for name, command in externals.items():
            models[name] = lambda c=command, l=lang: external_fn(c, l)
        for evs in sets_for(lang, args.sets):
            run_set(evs, models, homographs, DICTIONARIES[lang], args.raw_dir, args.threads)

    if not args.no_report:
        results = [json.loads(p.read_text(encoding="utf-8"))
                   for p in sorted(args.raw_dir.glob("*.summary.json"))]
        write_report(args.out, render(results, args.date))
        print(f"wrote {args.out} from {len(results)} result files")


if __name__ == "__main__":
    main()
