"""Selection rules of export/build_be_homographs.py on a small GrammarDB-shaped archive.

The forms and tags follow GrammarDB RELEASE-202601 (``tags.svg``): a noun
form tag is case and number, a pronoun-adjective form tag gender, case and
number, a verb form tag ``0`` for the infinitive and tense plus person for
a finite form.
"""
import gzip
import importlib.util
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "export" / "build_be_homographs.py"


@pytest.fixture(scope="module")
def build():
    sys.path.insert(0, str(ROOT / "benchmarks"))
    spec = importlib.util.spec_from_file_location("build_be_homographs", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def paradigm(lemma, tag, forms, todo=None):
    attrs = f' todo="{todo}"' if todo else ""
    body = "".join(f'<Form tag="{ftag}">{f}</Form>' for f, ftag in forms)
    return f'<Paradigm pdgId="1" lemma="{lemma}" tag="{tag}"{attrs}><Variant id="a">{body}</Variant></Paradigm>'


def archive(tmp_path, build, members):
    path = tmp_path / "db.zip"
    with zipfile.ZipFile(path, "w") as zf:
        for member in build.XML_MEMBERS:
            content = "".join(members.get(member, []))
            zf.writestr(member, f'<?xml version="1.0" encoding="UTF-8"?><Wordlist>{content}</Wordlist>')
    return str(path)


@pytest.mark.parametrize("marked,ordinal", [
    ("ба+цькі", 0), ("бацькі+", 1), ("Я+на", 0), ("яна+", 1), ("стала", None), ("з'я+ва", 0),
])
def test_ordinal_counts_vowels_before_the_mark(build, marked, ordinal):
    assert build.ordinal_of(marked) == ordinal


@pytest.mark.parametrize("ptag,ftag,tags", [
    ("NCIIN2", "GS", {"pos:N", "case:G", "num:S"}),
    ("NCIIN2", "NP", {"pos:N", "case:N", "num:P"}),
    ("SAS0", "MGS", {"pos:S", "case:G", "num:S"}),
    ("ADP", "FNS", {"pos:A", "case:N", "num:S"}),
    ("VTMN1", "0", {"pos:V", "inf"}),
    ("VTMN1", "R3S", {"pos:V", "fin"}),
    ("VTMN1", "I2S", {"pos:V", "imp"}),
    ("VTMN1", "PG", {"pos:V"}),
])
def test_tags_follow_the_grammardb_tag_set(build, ptag, ftag, tags):
    assert build.tags_of(ptag, ftag) == tags


def test_collect_keeps_spellings_with_two_stresses_and_their_tags(tmp_path, build):
    zip_path = archive(tmp_path, build, {
        "N1.xml": [paradigm("ба+цька", "NCAIN1", [("ба+цькі", "GS"), ("бацькі+", "NP"), ("бацькі+", "AP")]),
                   paradigm("хлушу+", "NCIIN2", [("хлушу+", "NS")]),
                   paradigm("Я+на", "NPAPNM1", [("Я+на", "GS")]),
                   paradigm("ко+са", "NCIIN2", [("каса+", "NS"), ("косы+ травы+", "NS")])],
        "S.xml": [paradigm("ё+н", "SNP3", [("яна+", "FNS")], todo="націск о/ё рукамі"),
                  paradigm("мо+й", "SAS0", [("майго+", "MGS"), ("майго+", "MAS"), ("мая+", "FNS")])],
        "V.xml": [paradigm("вучы+цца", "VTMN1", [("вучы+цца", "0"), ("ву+чыцца", "R3S")])],
    })
    multi, agree, apostrophe, counts = build.collect(zip_path)
    assert multi["бацькі"] == {0: {"pos:N", "case:G", "num:S"}, 1: {"pos:N", "case:N", "num:P", "case:A"}}
    assert multi["вучыцца"] == {0: {"pos:V", "fin"}, 1: {"pos:V", "inf"}}
    # a paradigm marked todo still names a reading: the pronoun against the name
    assert multi["яна"] == {0: {"pos:N", "case:G", "num:S"}, 1: {"pos:S", "case:N", "num:S"}}
    assert "хлушу" not in multi and "каса" not in multi
    assert agree == {"майго": ["AS", "GS"], "мая": ["NS"]}
    assert apostrophe == {}
    assert counts["forms"] == 13 and counts["not_one_token"] == 1 and counts["todo"] == 1
    assert counts["multi_stress_keys"] == 3


def test_apostrophe_spellings_get_both_runtime_keys(tmp_path, build):
    zip_path = archive(tmp_path, build, {
        "V.xml": [paradigm("з'яві+цца", "VTMN1", [("з'яві+цца", "0"), ("з'я+віцца", "R3S")])],
    })
    multi, _, apostrophe, counts = build.collect(zip_path)
    assert set(multi) == {"зявіцца"}
    assert apostrophe == {"зявіцца": "з’явіцца"}
    words = {k: {"readings": {str(o): sorted(t) for o, t in v.items()}} for k, v in multi.items()}
    kept = build.select(words, apostrophe, Counter())
    assert set(kept) == {"зявіцца", "з’явіцца"}
    assert kept["зявіцца"] is kept["з’явіцца"]


def test_prior_is_the_majority_and_a_tie_gives_none(build):
    assert build.prior_of(Counter({1: 239, 0: 13})) == 1
    assert build.prior_of(Counter({0: 2, 1: 2})) is None
    assert build.prior_of(Counter({0: 1})) == 0


def test_select_drops_one_tag_set_without_prior(build):
    counts = Counter()
    words = {
        "мука": {"readings": {"0": ["case:N", "num:S", "pos:N"], "1": ["case:N", "num:S", "pos:N"]}, "prior": 0},
        "пара": {"readings": {"0": ["case:N", "num:S", "pos:N"], "1": ["case:N", "num:S", "pos:N"]}},
        "бацькі": {"readings": {"0": ["case:G", "num:S", "pos:N"], "1": ["case:N", "num:P", "pos:N"]}},
    }
    kept = build.select(words, {}, counts)
    assert set(kept) == {"мука", "бацькі"}
    assert counts["dropped_no_rule_no_prior"] == 1 and counts["homographs"] == 2


def test_layer_bytes_are_reproducible(tmp_path, build):
    blob = {"lang": "be", "sets": {"b": ["y", "x"], "a": []}, "words": {"б": {"readings": {"1": ["pos:N"]}}, "а": {}}}
    build.write_layer(str(tmp_path / "one.json.gz"), blob)
    build.write_layer(str(tmp_path / "two.json.gz"), {"words": blob["words"], "sets": blob["sets"], "lang": "be"})
    one, two = (tmp_path / "one.json.gz").read_bytes(), (tmp_path / "two.json.gz").read_bytes()
    assert one == two
    assert json.loads(gzip.decompress(one)) == blob


def test_pinned_input_refused_on_hash_mismatch(tmp_path, build):
    path = tmp_path / "zascienak_Malinauka.txt"
    path.write_text("Яна+ вучы+цца.\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="not the pinned zascienak file"):
        build._hashed(str(path), build.PRIOR_SOURCES["zascienak"]["sha256"], "the pinned zascienak file")


def test_prior_sources_exclude_the_homograph_test_sets(build):
    urls = {src["url"] for src in build.PRIOR_SOURCES.values()}
    assert not any(name in url for url in urls for name in ("10x10", "common_voice"))
    assert set(build.PRIOR_SOURCES) == {"zascienak", "wiktionary"}
    assert all(src["license"].startswith("CC-BY-SA") and src["tier"] == "GOLD"
               for src in build.PRIOR_SOURCES.values())
