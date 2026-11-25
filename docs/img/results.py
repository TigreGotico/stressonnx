"""Read the GOLD tables of benchmarks/RESULTS.md as {(lang, set): {model: accuracy}}."""
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "benchmarks" / "RESULTS.md"


def gold_accuracy():
    tables, key = {}, None
    for line in RESULTS.read_text(encoding="utf-8").splitlines():
        m = re.match(r"### (\S+) / (\S+) — (GOLD|SILVER)", line)
        if m:
            key = (m[1], m[2]) if m[3] == "GOLD" else None
            continue
        if key and line.startswith("| ") and not line.startswith("| model"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) > 4 and re.fullmatch(r"[0-9.]+", cells[4]):
                tables.setdefault(key, {})[cells[0]] = float(cells[4])
        elif line.startswith("## "):
            key = None
    return tables
