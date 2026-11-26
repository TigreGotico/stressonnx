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


def test_measured_numbers_come_from_results():
    # Each three-decimal score on these pages must occur in RESULTS.md. A changed score that
    # matches another cell of RESULTS.md passes, so this check catches only invented numbers.
    results = (ROOT / "benchmarks" / "RESULTS.md").read_text(encoding="utf-8")
    for page in ("index.md", "supported-languages.md"):
        text = (ROOT / "docs" / page).read_text(encoding="utf-8")
        missing = sorted({n for n in re.findall(r"\b0\.\d{3}\b", text) if n not in results})
        assert not missing, f"docs/{page}: {missing} not in benchmarks/RESULTS.md"


def _example_pages():
    """Pages with example lines; each page's blocks share one namespace, in order."""
    for page in DOCS:
        blocks = re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.DOTALL)
        blocks = [b for b in blocks if any(EXAMPLE.match(line) for line in b.splitlines())]
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
            m = EXAMPLE.match(line)
            if m:
                line = f"_check({m['expr']}, {ast.literal_eval(m['want'])!r}, {m['expr']!r})"
            lines.append(line)
        exec(compile("\n".join(lines), page, "exec"), namespace)  # noqa: S102 - runs the docs examples
