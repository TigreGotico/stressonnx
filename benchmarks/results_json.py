"""Write ``benchmarks/results.json``: the GOLD scores of every raw summary.

    python benchmarks/results_json.py --raw-dir DIR [--out benchmarks/results.json]

SILVER sets are left out: they measure agreement with a machine labeller, not
accuracy.  One record per (language, set, model), sorted.
"""
import argparse
import json
from pathlib import Path


def records(raw_dir: Path) -> list:
    out = []
    for path in sorted(raw_dir.glob("*.summary.json")):
        s = json.loads(path.read_text(encoding="utf-8"))
        if s["tier"] != "GOLD":
            continue
        for model, r in s["models"].items():
            out.append({
                "lang": s["lang"], "set": s["set"], "model": model,
                "source": s["source"], "license": s["license"],
                "homographs_only": s["homographs_only"],
                "targets": r["n"], "correct": r["correct"],
                "accuracy": round(r["accuracy"], 4),
                "ci95": [round(x, 4) for x in r["ci95"]],
            })
    return sorted(out, key=lambda r: (r["lang"], r["set"], r["model"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/results.json"))
    args = ap.parse_args()
    rows = records(args.raw_dir)
    args.out.write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{len(list(args.raw_dir.glob('*.summary.json')))} summaries in, {len(rows)} GOLD records out")


if __name__ == "__main__":
    main()
