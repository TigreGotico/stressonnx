"""Build the Belarusian homograph layer of the ``silero`` model.

GrammarDB (https://github.com/Belarus/GrammarDB, CC-BY-SA-4.0) lists
about 21,000 spellings with two or more stress readings (за́мак/зама́к).
The dictionary layer (``build_be_grammardb_vocab.py``) leaves them out.
This script writes them to ``homographs.json.gz`` with, for each reading,
the ordinal of its stressed vowel and the grammatical tags of the forms
that carry it (part of speech, case, number, finite or infinitive), plus
the closed word classes the runtime rules read (prepositions by the case
they govern, numerals, subject pronouns, modals, the case and number set
of every pronoun-adjective and numeral form) and a frequency prior.

The prior is the most frequent reading of a spelling in two stressed
corpora of the benchmark harness, neither of which is a homograph test set:
``zascienak_Malinauka.txt`` of alex73/benchmarks-stress-bel (CC-BY-SA-4.0)
and the Belarusian usage examples of English Wiktionary through the
kaikki.org dump (CC-BY-SA).  Only reading counts leave the corpora; no
sentence is shipped.
A spelling whose two most frequent readings tie gets no prior.

A spelling whose readings all carry the same tags cannot be told apart by
any rule; it is kept only when it has a prior.  Every input is pinned by
sha256, and the output is byte-identical between runs.

Usage::

    python export/build_be_homographs.py --zip RELEASE-202601.zip \\
        --kaikki kaikki.org-dictionary-Belarusian.jsonl[.gz] \\
        --zascienak zascienak_Malinauka.txt --out out/bel

Then upload ``out/bel/homographs.json.gz``, ``out/bel/meta.json`` and
``out/bel/LICENSE.grammardb`` to ``TigreGotico/stressonnx-models/bel/`` and
bump ``HF_REPO_REVISION``.
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import unicodedata
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
from build_be_grammardb_vocab import (  # noqa: E402
    LICENSE_URL, RELEASE, RELEASE_COMMIT, RELEASE_ZIP_SHA256, KAIKKI_SHA256, KAIKKI_URL,
    NOTICE, UPSTREAM, UPSTREAM_URL, XML_MEMBERS, fetch_license, sha256,
)
from gold.belarusian_homographs import BASE as ALEX73_BASE  # noqa: E402
from scoring import STRESS, clean_word, marked_sentence_row, strip_stress  # noqa: E402
from stressonnx.backends._homograph_rules import MINED, RULES, HomographRules  # noqa: E402
from stressonnx.download import _download_files  # noqa: E402
from stressonnx.registry import _MAIN_FILES  # noqa: E402

VOWELS = frozenset("аеёіоуыэюя")

#: Prior corpora: name, pinned URL, licence, label tier, sha256 of the file.
PRIOR_SOURCES = {
    "zascienak": {
        "name": "alex73/benchmarks-stress-bel, zascienak_Malinauka.txt",
        "url": ALEX73_BASE + "zascienak_Malinauka.txt",
        "license": "CC-BY-SA-4.0",
        "tier": "GOLD",
        "sha256": "bf9dcc189639dc8e8bd7b145dc7a587ed00311271944152f5edc484e33352b08",
    },
    "wiktionary": {
        "name": "English Wiktionary Belarusian usage examples (kaikki.org dump)",
        "url": KAIKKI_URL,
        "license": "CC-BY-SA",
        "tier": "GOLD",
        "sha256": KAIKKI_SHA256,
    },
}

# Closed word classes the rules read.  Each preposition set holds the
# prepositions whose English Wiktionary entry carries the ``{{+obj|be|<case>}}``
# template for exactly that case (https://en.wiktionary.org/wiki/<word>#Belarusian).
# The numerals follow https://be.wikipedia.org/wiki/Лічэбнік.
SETS = {
    "gen_preps": "без для да пасля акрамя сярод супраць дзеля замест апрача".split(),
    "loc_preps": ["пры"],
    "loc_acc_preps": "у ва на аб па".split(),
    "acc_preps": "праз пад".split(),
    "num234": "два дзве тры чатыры абодва абедзве".split(),
    "num5": "пяць шэсць сем восем дзевяць дзесяць адзінаццаць дваццаць трыццаць сорак пяцьдзясят сто".split(),
    "subject_pronouns": "я ты ён яна яно мы вы яны хто".split(),
    "modals": ("хачу хочаш хоча хочам хочаце хочуць трэба можна нельга магу можаш можа можам можаце "
               "могуць павінен павінна павінны буду будзеш будзе будзем будзеце будуць пачаў пачала "
               "пачалі стаў стала сталі").split(),
}

#: Adjectival endings and the case and number pairs they can mark (MINED:
#: read off the adjective paradigms, no citation; counted in meta.json).
ADJ_ENDINGS = {
    "ага": "GS AS", "яга": "GS AS", "ога": "GS AS", "аму": "DS", "яму": "DS", "ому": "DS",
    "ым": "IS DP", "ім": "IS DP", "ай": "GS DS IS LS NS", "яй": "GS DS IS LS", "ой": "GS DS IS LS",
    "ая": "NS", "яя": "NS", "ую": "AS", "юю": "AS", "ыя": "NP AP", "ія": "NP AP",
    "ых": "GP AP LP", "іх": "GP AP LP", "ымі": "IP", "імі": "IP", "ае": "NS AS", "яе": "NS AS", "ое": "NS AS",
}

def ordinal_of(marked: str) -> int | None:
    """Vowel ordinal of the ``+``-marked vowel (GrammarDB writes ``+`` after it)."""
    if "+" not in marked:
        return None
    before = marked[:marked.index("+")].lower()
    n = sum(1 for c in before if c in VOWELS)
    return n - 1 if n else None


def tags_of(ptag: str, ftag: str) -> set:
    """Normalised tags of one form from its paradigm tag and form tag.

    GrammarDB (``tags.svg``): a noun form tag is case and number; an
    adjective, pronoun or participle form tag is gender, case and number;
    a verb form tag is ``0`` for the infinitive, a tense letter (R, F, P)
    followed by person or gender for a finite form, ``I`` for an imperative.
    """
    pos = ptag[:1] or "?"
    tags = {"pos:" + pos}
    if pos == "N" and len(ftag) == 2:
        tags.update(("case:" + ftag[0], "num:" + ftag[1]))
    elif pos in "APS" and len(ftag) == 3:
        tags.update(("case:" + ftag[1], "num:" + ftag[2]))
    elif pos == "V":
        if ftag == "0":
            tags.add("inf")
        elif ftag[:1] in ("R", "F", "P") and ftag != "PG":
            tags.add("fin")
        elif ftag[:1] == "I":
            tags.add("imp")
    return tags


def grammardb_forms(zip_path: str):
    """``(form text, paradigm tag, form tag, todo)`` for every form of the archive."""
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = [m for m in XML_MEMBERS if m not in names]
        if missing:
            raise SystemExit(f"{zip_path} lacks {missing}")
        for member in XML_MEMBERS:
            with zf.open(member) as fh:
                ptag = todo = None
                for event, el in ET.iterparse(fh, events=("start", "end")):
                    if event == "start":
                        if el.tag == "Paradigm":
                            ptag, todo = el.get("tag"), el.get("todo")
                        elif el.tag == "Variant" and el.get("tag"):
                            ptag = el.get("tag")
                        continue
                    if el.tag == "Form":
                        yield (el.text or ""), ptag or "", el.get("tag") or "", todo
                        el.clear()
                    elif el.tag == "Paradigm":
                        el.clear()


def _key(form: str) -> str:
    """The runtime key of *form*: lowercase, no stress mark, no apostrophe."""
    plain = unicodedata.normalize("NFC", form.replace("+", "")).lower()
    return plain.replace("'", "").replace("’", "").replace("ʼ", "")


def collect(zip_path: str):
    """Readings with tags per multi-stress spelling, the modifier table, counts."""
    counts = Counter()
    readings = defaultdict(lambda: defaultdict(set))
    stresses = defaultdict(set)
    agree = defaultdict(set)
    apostrophe = {}
    for form, ptag, ftag, todo in grammardb_forms(zip_path):
        counts["forms"] += 1
        if ptag[:2] == "SA" and len(ftag) == 3:
            agree[_key(form)].add(ftag[1] + ftag[2])
        elif ptag[:1] == "M" and len(ftag) >= 2 and ftag[-2] in "NGDAIL" and ftag[-1] in "SP":
            agree[_key(form)].add(ftag[-2] + ftag[-1])
        if " " in form or "-" in form:
            counts["not_one_token"] += 1
            continue
        if todo:
            # a paradigm whose stress is not final still names its readings;
            # the pronoun paradigm of ён (яна́ against the name Я́на) is one
            counts["todo"] += 1
        ordinal = ordinal_of(form)
        if ordinal is None:
            counts["unstressed"] += 1
            continue
        key = _key(form)
        stresses[key].add(ordinal)
        readings[key][ordinal] |= tags_of(ptag, ftag)
        if "'" in form:
            apostrophe.setdefault(key, form.replace("+", "").lower().replace("'", "’"))
    counts["keys"] = len(stresses)
    multi = {k: dict(readings[k]) for k, s in stresses.items() if len(s) > 1}
    counts["multi_stress_keys"] = len(multi)
    return multi, {k: sorted(v) for k, v in agree.items()}, apostrophe, counts


def _hashed(path: str, expected: str, what: str):
    digest = hashlib.sha256()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise SystemExit(f"{path} is not {what} (sha256 {digest.hexdigest()})")


def zascienak_rows(path: str):
    lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    drops = Counter()
    for n, line in enumerate(ln for ln in lines if ln and not ln.startswith("#")):
        row = marked_sentence_row(f"zascienak:{n}", unicodedata.normalize("NFC", line).replace("+", STRESS), drops)
        if row is not None:
            yield row


def wiktionary_rows(path: str):
    opener = gzip.open if path.endswith(".gz") else open
    seen = set()
    drops = Counter()
    with opener(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if entry.get("lang_code") != "be":
                continue
            word = entry.get("word", "")
            if word.startswith("-") or word.endswith("-"):
                continue
            for sense in entry.get("senses", []):
                for ex in sense.get("examples", []):
                    text = (ex.get("text") or "").strip()
                    if STRESS not in text or text in seen or any(c.isascii() and c.isalpha() for c in text):
                        continue
                    seen.add(text)
                    row = marked_sentence_row(f"wikt-be:{len(seen)}", text, drops)
                    if row is not None:
                        yield row


def contexts(rows, homographs: dict):
    """``(words, index, gold ordinal)`` for every target that is a homograph."""
    for row in rows:
        words = [clean_word(t) for t in strip_stress(row.text).split()]
        for target in row.targets:
            word = words[target.token_index]
            if word in homographs:
                yield words, target.token_index, target.ordinal


def prior_of(counter: Counter) -> int | None:
    """The most frequent reading, or None when the two most frequent tie."""
    ranked = counter.most_common(2)
    if len(ranked) == 1 or ranked[0][1] > ranked[1][1]:
        return ranked[0][0]
    return None


def select(words: dict, apostrophe: dict, counts: Counter) -> dict:
    """The entries the runtime can use, keyed as the runtime keys a token.

    A spelling whose readings all carry one tag set gives no rule a choice,
    so it stays only with a prior.  The silero alphabet keeps ’ and drops
    the ASCII apostrophe, so з'ява and зʼява look up зява and з’ява looks
    up з’ява: a spelling with an apostrophe gets both keys.
    """
    kept = {}
    for key, entry in words.items():
        separable = len({tuple(t) for t in entry["readings"].values()}) > 1
        if not separable and "prior" not in entry:
            counts["dropped_no_rule_no_prior"] += 1
            continue
        kept[key] = entry
        if key in apostrophe:
            kept[apostrophe[key]] = entry
    counts["homographs"] = len(kept)
    return kept


def write_layer(path: str, blob: dict) -> None:
    """*blob* as sorted compact JSON in a gzip file without a name or mtime: the bytes are reproducible."""
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as gz:
        gz.write(json.dumps(blob, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help=f"path to {RELEASE}.zip")
    ap.add_argument("--kaikki", required=True, help="the pinned kaikki Belarusian dump (.jsonl or .jsonl.gz)")
    ap.add_argument("--zascienak", required=True, help="zascienak_Malinauka.txt of alex73/benchmarks-stress-bel")
    ap.add_argument("--out", required=True, help="output directory for the bel bundle files")
    ap.add_argument("--cache-dir", default=None, help="stressonnx model cache holding the current bel bundle")
    args = ap.parse_args()

    if sha256(args.zip) != RELEASE_ZIP_SHA256:
        raise SystemExit(f"{args.zip} is not {RELEASE}.zip (sha256 differs)")
    _hashed(args.kaikki, KAIKKI_SHA256, "the pinned kaikki dump")
    _hashed(args.zascienak, PRIOR_SOURCES["zascienak"]["sha256"], "the pinned zascienak file")

    homographs, agree, apostrophe, counts = collect(args.zip)
    print(f"GrammarDB: {counts['forms']} forms in, {counts['multi_stress_keys']} multi-stress spellings", flush=True)

    sets = dict(SETS)
    sets["agree"] = agree
    sets["adj_endings"] = {k: v.split() for k, v in ADJ_ENDINGS.items()}
    words = {k: {"readings": {str(o): sorted(t) for o, t in sorted(v.items())}} for k, v in homographs.items()}
    rules = HomographRules({"sets": sets, "words": words})

    prior_counts = defaultdict(Counter)
    source_contexts = Counter()
    rule_fired, rule_right = Counter(), Counter()
    sources = (("zascienak", zascienak_rows(args.zascienak)),
               ("wiktionary", wiktionary_rows(args.kaikki)))
    for name, rows in sources:
        for tokens, idx, gold in contexts(rows, homographs):
            if gold not in homographs[tokens[idx]]:
                counts["contexts_outside_readings"] += 1
                continue
            source_contexts[name] += 1
            prior_counts[tokens[idx]][gold] += 1
            ordinal, stage = rules.resolve(tokens, idx)
            if stage and stage.startswith("rule:"):
                rule_fired[stage[5:]] += 1
                rule_right[stage[5:]] += ordinal == gold
        print(f"{name}: {source_contexts[name]} homograph contexts", flush=True)

    for key, c in prior_counts.items():
        prior = prior_of(c)
        if prior is None:
            counts["prior_ties"] += 1
        else:
            words[key]["prior"] = prior
        words[key]["prior_counts"] = {str(o): n for o, n in sorted(c.items())}
    counts["spellings_with_prior"] = sum(1 for w in words.values() if "prior" in w)

    kept = select(words, apostrophe, counts)
    print(f"homographs.json.gz: {len(words)} spellings in, {len(kept)} keys out "
          f"({counts['dropped_no_rule_no_prior']} dropped: one tag set and no prior)", flush=True)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "homographs.json.gz")
    write_layer(path, {"lang": "be", "sets": sets, "words": kept})
    with open(os.path.join(args.out, "LICENSE.grammardb"), "w", encoding="utf-8") as fh:
        fh.write(NOTICE + fetch_license())

    data = _download_files("bel", _MAIN_FILES, args.cache_dir, model_id="silero")
    with open(data["meta.json"], encoding="utf-8") as fh:
        meta = json.load(fh)
    meta["n_homographs"] = len(kept)
    meta["homograph_source"] = {
        "name": UPSTREAM,
        "release": RELEASE,
        "commit": RELEASE_COMMIT,
        "url": f"{UPSTREAM_URL}/releases/tag/{RELEASE}",
        "archive_sha256": RELEASE_ZIP_SHA256,
        "license": "CC-BY-SA-4.0",
        "license_file": "LICENSE.grammardb",
        "copyright": "GrammarDB by Aleś Bułojčyk and contributors",
        "selection": "spellings with two or more stresses in the database, paradigms marked "
                     "todo included, each reading with the paradigm and form tags of its forms; a "
                     "spelling whose readings share one tag set is kept only with a prior",
        "prior": {
            "rule": "the most frequent reading of the spelling in the sources; none on a tie",
            "sources": [{**src, "contexts": source_contexts[name]} for name, src in PRIOR_SOURCES.items()],
            "spellings": counts["spellings_with_prior"],
            "ties": counts["prior_ties"],
        },
        "rules": {name: {"source": "mined" if name in MINED else "cited",
                         "fired": rule_fired[name], "right": rule_right[name]}
                  for name, _ in RULES},
        "counts": dict(counts),
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(json.dumps(meta["homograph_source"]["rules"], indent=1))
    print(json.dumps(dict(counts), indent=2))
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
