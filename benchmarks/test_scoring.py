"""Tests for the benchmark scorer.  Run: ``pytest benchmarks/test_scoring.py``."""
import sys
from collections import Counter

import pytest

from external import JsonlCommand
from gold.russtress import gold_target
from run import BEGIN, END, write_report
from scoring import (
    DROP_LETTERS, Row, Target, align, bootstrap_ci, marked_ordinals, marked_sentence_row,
    mcnemar_exact, score,
)


def replay(outputs: dict):
    return lambda text: outputs[text]


def test_glide_letters_do_not_count_as_vowels():
    # й and ў decompose under NFD into и/у plus a breve; they are not vowels
    assert marked_ordinals("райо́н") == [1]
    assert marked_ordinals("аўто́бус") == [1]
    assert marked_ordinals("йо́д") == [0]
    assert marked_ordinals("її́") == [1]


def test_correct_prediction_on_glide_words_is_a_hit():
    drops = Counter()
    row = marked_sentence_row("r", "Наш райо́н і аўто́бус.", drops)
    assert row.text == "Наш район і аўтобус."
    assert row.targets == (Target(1, 1), Target(3, 1))
    result = score(replay({row.text: "Наш райо́н і аўто́бус."}), [row])
    assert result.summary()["accuracy"] == 1.0
    assert result.summary()["n"] == 2


def test_extra_mark_or_mark_on_consonant_is_a_miss():
    row = Row("r", "по-моему аўтобус", (Target(0, 1), Target(1, 1)))
    over_marked = score(replay({row.text: "по́-моему́ аў́тобус"}), [row]).summary()
    assert over_marked["correct"] == 0 and over_marked["n"] == 2
    exact = score(replay({row.text: "по-мо́ему аўто́бус"}), [row]).summary()
    assert exact["correct"] == 2


def test_unmarked_yo_claims_the_stress():
    row = Row("r", "зеленый", (Target(0, 1),))
    assert score(replay({row.text: "зелёный"}), [row]).summary()["correct"] == 1
    assert score(replay({row.text: "зе́лёный"}), [row]).summary()["correct"] == 0


def test_spacing_change_stays_aligned():
    # a model that moves a space or a quote still marks the same letters
    row = Row("r", 'синоним слова "топовая".', (Target(1, 0), Target(2, 0)))
    summary = score(replay({row.text: 'сино́ним сло́ва"то́повая".'}), [row]).summary()
    assert (summary["n"], summary["correct"]) == (2, 2)
    assert align("мой район", "мойрайо́н") == ["мой", "райо́н"]
    assert align("все ежи", "всё ежи́") == ["всё ", "ежи́"]


def test_letter_change_drops_instead_of_scoring():
    row = Row("r", "мой район", (Target(1, 1),))
    for out in ("мо райо́н", "мой райо́ны", "мой"):
        summary = score(replay({row.text: out}), [row]).summary()
        assert summary["n"] == 0
        assert summary["dropped"] == {DROP_LETTERS: 1}


def test_failing_stress_function_drops_its_row():
    def broken(text):
        raise ValueError("boom")
    row = Row("r", "мой район", (Target(1, 1),))
    result = score(broken, [row])
    assert result.summary()["n"] == 0
    assert result.rows[0]["error"] == "ValueError: boom"


def test_homograph_flag_uses_the_clean_word():
    row = Row("r", "«Замок» стоит", (Target(0, 0),))
    summary = score(replay({row.text: "«За́мок» стои́т"}), [row], {"замок"}).summary()
    assert (summary["homo_n"], summary["homo_acc"]) == (1, 1.0)


def test_marked_sentence_row_drop_reasons():
    drops = Counter()
    row = marked_sentence_row("r", "Я́ люблю́ ка́к-нибу́дь кошек", drops)
    assert row.targets == (Target(1, 1),)
    assert drops == Counter({"monosyllabic token (targets)": 1,
                             "token carries 2+ acutes (targets)": 1})
    assert marked_sentence_row("r", "без ударений", drops) is None
    assert drops["no usable target (rows)"] == 1


def test_russtress_gold_target():
    window = "мой район хороший"
    assert gold_target(window, window.index("он")) == (1, 1)
    assert gold_target(window, window.index("р")) is None
    assert gold_target(window, len(window)) is None


@pytest.mark.parametrize("a_only, b_only, p", [
    (10, 0, 2 / 2 ** 10),
    (3, 1, 2 * (1 + 4) / 2 ** 4),
    (5, 5, 1.0),
    (0, 0, 1.0),
])
def test_mcnemar_exact(a_only, b_only, p):
    a, b = {}, {}
    for i in range(a_only):
        a[i], b[i] = True, False
    for i in range(a_only, a_only + b_only):
        a[i], b[i] = False, True
    for i in range(100, 107):  # concordant pairs never change p
        a[i] = b[i] = i % 2 == 0
    result = mcnemar_exact(a, b)
    assert (result["a_only"], result["b_only"], result["common"]) == (a_only, b_only, a_only + b_only + 7)
    assert result["p"] == pytest.approx(p)


def test_bootstrap_is_paired_and_seeded():
    rows = [Row(str(i), f"слово{i} район", (Target(1, 1),)) for i in range(50)]
    right = {r.text: r.text.replace("район", "райо́н") for r in rows}
    half = {r.text: (right[r.text] if int(r.id) % 2 else r.text) for r in rows}
    scored = {"all": score(replay(right), rows), "half": score(replay(half), rows),
              "same": score(replay(half), rows)}
    ci = bootstrap_ci(scored)
    assert ci["all"] == (1.0, 1.0)
    assert ci["half"] == ci["same"]
    assert ci["half"][0] < 0.5 < ci["half"][1]
    assert bootstrap_ci(scored) == ci


ECHO = (
    "import json, sys\n"
    "for line in sys.stdin:\n"
    "    req = json.loads(line)\n"
    "    if req['text'] == 'boom':\n"
    "        print(json.dumps({'error': 'no'}), flush=True)\n"
    "    else:\n"
    "        print(json.dumps({'text': req['text'].replace('район', 'райо́н')}), flush=True)\n"
)


def test_external_jsonl_command(tmp_path):
    script = tmp_path / "echo.py"
    script.write_text(ECHO, encoding="utf-8")
    proc = JsonlCommand(f"{sys.executable} {script}", "ru")
    rows = [Row("a", "мой район", (Target(1, 1),)), Row("b", "boom", ())]
    result = score(proc, rows)
    proc.close()
    assert result.summary()["correct"] == 1
    assert result.rows[1]["error"] == "RuntimeError: no"


def test_write_report_keeps_foreign_sections(tmp_path):
    out = tmp_path / "RESULTS.md"
    out.write_text("# old board\n\nold table\n## OOV positional rules (simple model)\n\nkept\n",
                   encoding="utf-8")
    write_report(out, [BEGIN, "# new board", END])
    first = out.read_text(encoding="utf-8")
    assert first == f"{BEGIN}\n# new board\n{END}\n\n## OOV positional rules (simple model)\n\nkept\n"
    write_report(out, [BEGIN, "# newer board", END])
    assert out.read_text(encoding="utf-8") == first.replace("# new board", "# newer board")
