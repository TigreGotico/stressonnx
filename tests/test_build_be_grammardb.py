"""Selection rules of export/build_be_grammardb_vocab.py on a small GrammarDB-shaped archive."""
import importlib.util
import os
import zipfile
from pathlib import Path

import pytest

from stressonnx.backends.silero import _SileroStressor

CACHE_DIR = os.environ.get("STRESSONNX_TEST_CACHE_DIR")
SCRIPT = Path(__file__).resolve().parents[1] / "export" / "build_be_grammardb_vocab.py"


@pytest.fixture(scope="module")
def build():
    spec = importlib.util.spec_from_file_location("build_be_grammardb_vocab", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def stressor():
    s = _SileroStressor("be", cache_dir=CACHE_DIR)
    s._ensure_loaded()
    return s


def paradigm(lemma, forms, todo=None):
    attrs = f' todo="{todo}"' if todo else ""
    body = "".join(f'<Form tag="X">{f}</Form>' for f in forms)
    return f'<Paradigm pdgId="1" lemma="{lemma}" tag="N"{attrs}><Variant id="a">{body}</Variant></Paradigm>'


def archive(tmp_path, build, paradigms):
    path = tmp_path / "db.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for member in build.XML_MEMBERS:
            content = "".join(paradigms) if member == "N1.xml" else ""
            zf.writestr(member, f'<?xml version="1.0" encoding="UTF-8"?><Wordlist>{content}</Wordlist>')
    return str(path)


def test_todo_paradigm_gives_no_entry_and_vetoes_its_spelling(tmp_path, build, stressor):
    zip_path = archive(tmp_path, build, [
        paradigm("во+семдзясят", ["васьмі+дзесяці"], todo="націск о/ё рукамі"),
        paradigm("хлушу+", ["хлушу+"]),
        paradigm("Я+на", ["Я+на"]),
        paradigm("ё+н", ["яна+"], todo="націск о/ё рукамі"),
    ])
    entries, counts = build.collect(zip_path, stressor)
    assert counts["todo"] == 2
    assert "васьмідзесяці" not in entries
    assert entries["хлушу"] == (1, "хлушу")
    assert "Яна" not in entries and "яна" not in entries


def test_hyphen_part_is_a_reading_of_its_token(tmp_path, build, stressor):
    zip_path = archive(tmp_path, build, [
        paradigm("ра+мбурсны", ["ра+мбурсны"]),
        paradigm("акцэ+птна-рамбу+рсны", ["акцэ+птна-рамбу+рсны"]),
        paradigm("галы+м-го+лы", ["галы+м-го+лы"]),
    ])
    entries, counts = build.collect(zip_path, stressor)
    assert counts["not_one_token"] == 2
    assert "рамбурсны" not in entries
    assert counts["conflicting_keys"] == 1
    assert entries["акцэптна"] == (1, "акцэптна")
    assert entries["галым"] == (1, "галым")
    assert entries["голы"] == (0, "голы")


def test_capitalized_form_keeps_its_capital(tmp_path, build, stressor):
    zip_path = archive(tmp_path, build, [
        paradigm("А+лен", ["А+лена"]),
        paradigm("сталі+на", ["сталі+н"]),
        paradigm("Я+на", ["Я+на"]),
        paradigm("яна+", ["яна+"]),
        paradigm("Іо+сіф", ["Іо+сіф"]),
        paradigm("іо+сіф", ["іо+сіф"]),
    ])
    entries, counts = build.collect(zip_path, stressor)
    assert entries["Алена"] == (0, "алена")
    assert "алена" not in entries
    assert entries["сталін"] == (1, "сталін")
    assert "Сталін" not in entries
    # a name and a common word of one spelling that disagree: neither key
    assert "Яна" not in entries and "яна" not in entries
    assert counts["conflicting_keys"] == 2
    # the same spelling with the same stress: both keys
    assert entries["Іосіф"] == (1, "іосіф")
    assert entries["іосіф"] == (1, "іосіф")


def test_apostrophe_form_gets_both_keys(tmp_path, build, stressor):
    zip_path = archive(tmp_path, build, [paradigm("з'я+ва", ["з'я+ва"])])
    entries, _ = build.collect(zip_path, stressor)
    assert entries["зява"] == (0, "з'ява")
    assert entries["з’ява"] == (0, "з’ява")


def test_wiktionary_disagreement_drops_the_key(build):
    entries = {"выявы": (1, "выявы"), "хлушу": (1, "хлушу"), "рыкай": (0, "рыкай"),
               "сталін": (1, "сталін"), "Іосіф": (1, "іосіф")}
    readings = {"выявы": {0}, "хлушу": {1}, "рыкай": {0, 1}, "Сталін": {0}, "іосіф": {1}}
    assert build.wiktionary_disagrees(entries, readings) == {
        "выявы": [0], "рыкай": [0, 1], "сталін": [0]}
