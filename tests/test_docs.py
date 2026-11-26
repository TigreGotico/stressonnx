"""The docs match the code: figure data, measured numbers and the printed values of examples."""
import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = sorted([ROOT / "README.md", *(ROOT / "docs").glob("*.md")])
#: `expr  # 'value'`: an example line whose comment is the repr of its result
EXAMPLE = re.compile(r"^(?P<expr>\S.*?)\s+#\s+(?P<want>(['\"\[(]).*?\3|\[.*?\]|\(.*?\)|-?\d+|None|True|False)(\s|$)")


@pytest.mark.parametrize("script", [p for p in sorted((ROOT / "docs" / "img").glob("plot_*.py")) if p.name != "plot_style.py"], ids=lambda p: p.name)
def test_figure_data_matches_the_result_file(script, tmp_path):
    stem = re.search(r'parent / "([^"]+)"', script.read_text(encoding="utf-8"))[1]
    committed = script.with_name(f"{stem}.json")
    before = committed.read_bytes()
    try:
        subprocess.run([sys.executable, str(script)], check=True,
                       env={**os.environ, "DOCS_FIGURES_DATA_ONLY": "1"})
        assert committed.read_bytes() == before, f"rerun plot_{stem}: {committed.name} is stale"
    finally:
        committed.write_bytes(before)


#: A three-decimal score; a footnote mark such as ² may follow it.
SCORE = r"(?<![\d.])0\.\d{3}(?!\d)"
#: Each score on a docs line, in the order it appears, as (lang, set, model, column) of RESULTS.md.
SCORES = {
    ("index.md", "Default-model word accuracy"): [
        ("ru", "russtress", "ruaccent-silero", "acc"), ("ru", "wiktionary", "ruaccent-silero", "acc"),
        ("be", "russtress", "silero", "acc"), ("be", "zascienak", "silero", "acc"),
        ("uk", "russtress", "silero", "acc"), ("uk", "wiktionary", "silero", "acc"),
        ("hy", "jipa", "simple", "acc"), ("bg", "rbe", "simple", "acc"),
        ("be", "homographs-10x10", "silero", "acc")],
    ("supported-languages.md", "| `ruaccent` |"): [
        ("ru", "russtress", "ruaccent", "acc"), ("ru", "russtress", "ruaccent", "homo")],
    ("supported-languages.md", "| `ruaccent-int8`"): [
        ("ru", "russtress", "ruaccent-int8", "acc"), ("ru", "russtress", "ruaccent-int8", "homo")],
    ("supported-languages.md", "| `ruaccent-big-poetry` |"): [
        ("ru", "russtress", "ruaccent-big-poetry", "acc"), ("ru", "russtress", "ruaccent-big-poetry", "homo")],
    ("supported-languages.md", "| `ruaccent-silero` |"): [
        ("ru", "russtress", "ruaccent-silero", "acc"), ("ru", "russtress", "ruaccent-silero", "homo"),
        ("ru", "wiktionary", "ruaccent-silero", "acc"), ("ru", "wiktionary", "ruaccent", "acc"),
        ("ru", "wiktionary", "silero", "acc")],
    ("supported-languages.md", "| `silero` |"): [
        ("ru", "russtress", "silero", "acc"), ("ru", "russtress", "silero", "homo"),
        ("uk", "russtress", "silero", "acc"), ("be", "russtress", "silero", "acc")],
    ("supported-languages.md", "| `stressrnn` |"): [("ru", "russtress", "stressrnn", "acc")],
    ("supported-languages.md", "| `uk-accentor` |"): [
        ("uk", "russtress", "uk-accentor", "acc"), ("uk", "russtress", "silero", "acc")],
    ("supported-languages.md", "| `simple` |"): [("hy", "jipa", "simple", "acc"), ("bg", "rbe", "simple", "acc")],
    ("supported-languages.md", "| `kubataba` |"): [
        ("ru", "russtress", "kubataba", "acc"), ("ru", "russtress", "kubataba", "homo"),
        ("ru", "russtress", "silero", "acc"), ("ru", "russtress", "ruaccent", "acc")],
}


def _results_cells():
    """(lang, set, model, "acc" | "homo") → the value as RESULTS.md prints it."""
    cells, key = {}, None
    for line in (ROOT / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8").splitlines():
        m = re.match(r"### (\S+) / (\S+) — ", line)
        if m:
            key = (m[1], m[2])
        elif line.startswith("## "):
            key = None
        elif key and line.startswith("| ") and not line.startswith(("| model", "|---")):
            c = [x.strip() for x in line.strip("|").split("|")]
            cells[(*key, c[0], "acc")] = c[4]
            cells[(*key, c[0], "homo")] = c[6].split()[0]
    return cells


def test_each_docs_score_is_its_results_cell():
    cells = _results_cells()
    for (page, marker), keys in SCORES.items():
        (line,) = [x for x in (ROOT / "docs" / page).read_text(encoding="utf-8").splitlines() if marker in x]
        got = re.findall(SCORE, line)
        assert got == [cells[k] for k in keys], f"docs/{page} {marker!r}: {got} against RESULTS.md"


def test_every_docs_score_is_mapped():
    for page in ("index.md", "supported-languages.md"):
        lines = (ROOT / "docs" / page).read_text(encoding="utf-8").splitlines()
        mapped = {x for x in lines for p, marker in SCORES if p == page and marker in x}
        unmapped = [x[:60] for x in lines if re.search(SCORE, x) and x not in mapped]
        assert not unmapped, f"docs/{page}: add these lines to SCORES: {unmapped}"


def _example(line):
    """(expr, value) when *line* is an example whose comment is a Python literal, else None."""
    m = EXAMPLE.match(line)
    if not m:
        return None
    try:
        return m["expr"], ast.literal_eval(m["want"])
    except (ValueError, SyntaxError):
        return None


def _example_pages():
    """Pages with example lines; each page's blocks share one namespace, in order."""
    for page in DOCS:
        blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.DOTALL)
        blocks = [b for b in blocks if any(_example(line) for line in b.splitlines())]
        if blocks:
            yield page.relative_to(ROOT).as_posix(), blocks


@pytest.mark.parametrize("page, blocks", list(_example_pages()), ids=lambda v: v if isinstance(v, str) else "")
def test_example_prints_its_comment(page, blocks):
    def check(got, want, expr):
        assert got == want, f"{page}: {expr} gives {got!r}, the comment says {want!r}"

    namespace = {"_check": check}
    for block in blocks:
        lines = []
        for line in block.splitlines():
            example = _example(line)
            if example:
                expr, want = example
                line = f"_check({expr}, {want!r}, {expr!r})"
            lines.append(line)
        exec(compile("\n".join(lines), page, "exec"), namespace)  # noqa: S102 - runs the docs examples
