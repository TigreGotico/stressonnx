"""Per-reading accuracy and rule precision on the homograph sets of a harness raw directory.

    python benchmarks/homograph_readings.py --raw-dir DIR --lang be --layer <homographs.json.gz>
        [--sets homographs-10x10 homographs-commonvoice] [--out readings.json] [--md readings.md]

For every ``<lang>__<set>.rows.json.gz`` of a homograph-only set the script
reports, per model, the accuracy on every reading (a spelling and its gold
vowel ordinal), the number of readings a model gets entirely wrong, and,
with ``--layer``, the stage the homograph layer reaches on each target
(``rule:<name>``, ``prior``, ``none``, ``not in layer``) with the rule's
precision on the targets it touched.  An overall score hides a model that
always picks one reading of a pair; the per-reading table shows it.
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

from gold import sets_for
from gold._cache import CACHE
from scoring import clean_word
from stressonnx.backends._homograph_rules import MINED, HomographRules


def load_rows(raw_dir: Path, lang: str, name: str) -> dict:
    with gzip.open(raw_dir / f"{lang}__{name}.rows.json.gz", "rt", encoding="utf-8") as fh:
        return json.load(fh)


def per_reading(rows_by_model: dict) -> dict:
    """``reading → model → [hits, targets]`` over the targets every model scored."""
    table = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    scored = None
    for model, rows in rows_by_model.items():
        keys = {(r["id"], t["i"]) for r in rows if not r["dropped"] for t in r["targets"]}
        scored = keys if scored is None else scored & keys
    for model, rows in rows_by_model.items():
        for r in rows:
            if r["dropped"]:
                continue
            for t in r["targets"]:
                if (r["id"], t["i"]) in scored:
                    cell = table[(t["word"], t["gold"])][model]
                    cell[0] += t["hit"]
                    cell[1] += 1
    return table


def stages(layer: HomographRules, evs, rows: list) -> dict:
    """``(row id, target index) → (stage, hit)`` of the layer on the set's own text."""
    texts = {r.id: r for r in evs.load()[0]}
    out = {}
    for r in rows:
        row = texts[r["id"]]
        words = [clean_word(t) for t in row.text.split()]
        for t in r["targets"]:
            if words[t["token"]] not in layer:
                out[(r["id"], t["i"])] = ("not in layer", None)
                continue
            ordinal, stage = layer.resolve(words, t["token"])
            out[(r["id"], t["i"])] = (stage or "none", None if ordinal is None else ordinal == t["gold"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, default=CACHE / "results")
    ap.add_argument("--lang", default="be")
    ap.add_argument("--sets", nargs="+")
    ap.add_argument("--layer", help="homographs.json.gz of the bundle, for the stage coverage")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--md", type=Path)
    args = ap.parse_args()
    layer = HomographRules.load(args.layer) if args.layer else None
    report = {}
    lines = []
    for evs in sets_for(args.lang, args.sets):
        if not evs.homographs_only:
            continue
        rows_by_model = load_rows(args.raw_dir, evs.lang, evs.name)
        table = per_reading(rows_by_model)
        models = list(rows_by_model)
        wrong = {m: sum(1 for cells in table.values() if cells[m][0] == 0) for m in models}
        entry = {"readings": len(table), "entirely_wrong": wrong,
                 "per_reading": {f"{w}/{g}": dict(cells) for (w, g), cells in sorted(table.items())}}
        lines += [f"### {evs.lang} / {evs.name} — {evs.tier}: {len(table)} readings", "",
                  "| model | readings entirely wrong |", "|---|---|"]
        lines += [f"| {m} | {wrong[m]} of {len(table)} |" for m in models]
        if layer:
            by_target = stages(layer, evs, rows_by_model[models[0]])
            coverage = Counter(stage for stage, _ in by_target.values())
            precision = defaultdict(lambda: [0, 0])
            for stage, hit in by_target.values():
                if hit is not None:
                    precision[stage][0] += hit
                    precision[stage][1] += 1
            entry["coverage"] = dict(coverage)
            entry["precision"] = {s: {"right": r, "touched": n} for s, (r, n) in precision.items()}
            lines += ["", "| stage | targets | right | source |", "|---|---|---|---|"]
            for stage, n in sorted(coverage.items(), key=lambda kv: -kv[1]):
                right = f"{precision[stage][0]}/{precision[stage][1]}" if stage in precision else "—"
                source = ("mined" if stage[5:] in MINED else "cited") if stage.startswith("rule:") else ""
                lines.append(f"| {stage} | {n} | {right} | {source} |")
        lines += ["", "| reading | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
        for (w, g), cells in sorted(table.items()):
            lines.append(f"| {w}/{g} | " + " | ".join(f"{cells[m][0]}/{cells[m][1]}" for m in models) + " |")
        lines.append("")
        report[f"{evs.lang}__{evs.name}"] = entry
        print(f"{evs.lang}/{evs.name}: {len(table)} readings; entirely wrong {wrong}; "
              f"coverage {entry.get('coverage')}", flush=True)
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.md:
        args.md.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
