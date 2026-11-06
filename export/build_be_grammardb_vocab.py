"""Build the Belarusian dictionary layer of the ``silero`` model.

GrammarDB (https://github.com/Belarus/GrammarDB, CC-BY-SA-4.0) is the
grammar database of the Belarusian language.  Its release archive holds
about four million inflected forms in XML, with ``+`` after the stressed
vowel.  Most spellings have one stress; homographs (за+мак/зама+к) and
variant stresses appear as several marked forms of one spelling.

This script takes every spelling with exactly one stress and writes it to
the ``dictionary.txt.gz`` file of the ``bel`` silero bundle.  The runtime
looks a word up in that file before the exceptions table and the neural
model, so dictionary forms win and the model handles everything else:
homographs, forms with variant stress, and words outside the database.

Forms that the current pipeline (exceptions table plus model) already
stresses like the database are left out.  Their output does not change, so
the file holds only the forms where the database corrects the pipeline.
That keeps the download and the memory cost small.  A rebuilt
``accentor.onnx`` or ``exceptions.txt.gz`` requires a rebuild of this file.

A paradigm the database marks ``todo`` (data not final) gives no entry,
and its forms still veto a reading of the same spelling elsewhere.  Every
part of a hyphenated or multiword form is a reading of its own token, so
``акцэ+птна-рамбу+рсны`` conflicts with ``ра+мбурсны``.  A spelling that
en-Wiktionary (the kaikki dump, pinned by hash) stresses on another vowel
stays out: two sources that disagree do not make a form unambiguous.

The layer is keyed like the runtime keys a token: with every character
outside the silero alphabet removed, lowercased, except that a form the
database writes capitalized (a name) keeps its capital, so the runtime does
not stamp a common-noun reading on a name.  The alphabet keeps ’ but not
the ASCII apostrophe that GrammarDB writes, so ``з'ява`` and ``зʼява`` look
up ``зява`` and ``з’ява`` looks up ``з’ява``; both keys are written.  The
stored value is the ordinal of the stressed vowel in the word, which the
runtime maps onto the token as written, so a quote or apostrophe in the
token does not shift the mark.

The database is CC-BY-SA-4.0, and the silero bundle is MIT.  The layer is
therefore a separate file with its own notice, ``LICENSE.grammardb``, and
``exceptions.txt.gz`` stays the upstream silero table.

Usage::

    pip install "stressonnx[test]"
    python export/build_be_grammardb_vocab.py --zip RELEASE-202601.zip \\
        --kaikki kaikki.org-dictionary-Belarusian.jsonl --out out/bel

Then upload ``out/bel/dictionary.txt.gz``, ``out/bel/meta.json`` and
``out/bel/LICENSE.grammardb`` to ``TigreGotico/stressonnx-models/bel/`` and
bump ``HF_REPO_REVISION``.  ``out/bel/wiktionary_disagrees.json`` lists the
keys the cross-check dropped and is not part of the bundle.
"""
import argparse
import bisect
import gzip
import hashlib
import json
import os
import unicodedata
import urllib.request
import zipfile
import xml.etree.ElementTree as ET

from stressonnx.backends.silero import _SileroStressor
from stressonnx.download import _download_files
from stressonnx.registry import _MAIN_FILES

UPSTREAM = "GrammarDB"
UPSTREAM_URL = "https://github.com/Belarus/GrammarDB"
RELEASE = "RELEASE-202601"
RELEASE_COMMIT = "a3d1c9eb18387d234b3e4bbb7dc21ec3f50a0b29"
RELEASE_ZIP_SHA256 = "393756f3cb5c94ab86cdbd64b95fa9d096775f559ae65de6375ebb6d5571df00"
LICENSE_URL = f"https://raw.githubusercontent.com/Belarus/GrammarDB/{RELEASE_COMMIT}/LICENSE.txt"
LICENSE_SHA256 = "3b2890eacd851373001c4a14623458e3adaf1b1967939aa9c38a318e28d61c00"
#: kaikki.org wiktextract dump of en-Wiktionary Belarusian entries, the
#: cross-check source; the site keeps no versions, so the file is pinned by
#: hash, and the pinned copy is archived gzipped as
#: knowledge/wiki/audits/stressonnx-bench/artifacts/grammardb/kaikki.org-dictionary-Belarusian.jsonl.gz
#: in the TigreGotico AgentWorkspaces tree (sha256 of the unpacked file below)
KAIKKI_URL = "https://kaikki.org/dictionary/Belarusian/kaikki.org-dictionary-Belarusian.jsonl"
KAIKKI_SHA256 = "7331d5ea2c55baa5c8b7e84e7005cff17f48b93a8b26d9b35d20db079d3300cf"
ACUTE = "\u0301"
NOTICE = f"""The file dictionary.txt.gz of this bundle is derived from GrammarDB,
the grammar database of the Belarusian language, {RELEASE}
({UPSTREAM_URL}/releases/tag/{RELEASE}, commit {RELEASE_COMMIT}),
by Aleś Bułojčyk and contributors, and is licensed under the Creative
Commons Attribution-ShareAlike 4.0 International licence (CC-BY-SA-4.0).
It holds a selection of the database's spellings with one stress, those
the silero pipeline of this bundle stresses differently, each with the
ordinal of its stressed vowel.  The other files of this bundle are derived
from silero_stress and stay under the MIT licence.  The full licence text
follows.

"""

#: Batch size for running the silero model over candidate forms.
BATCH = 4096

#: Archive members that hold the word list; the rest are caches and the
#: spellchecker export.
XML_MEMBERS = ("A1.xml", "A2.xml", "C.xml", "E.xml", "I.xml", "M.xml", "N1.xml",
               "N2.xml", "N3.xml", "NP.xml", "P.xml", "R.xml", "S.xml", "V.xml",
               "W.xml", "Y.xml", "Z.xml")


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def forms(zip_path: str):
    """Every ``<Form>`` text of the archive with its paradigm's ``todo`` note.

    GrammarDB marks a paradigm whose data is not final with a ``todo``
    attribute (``націск о/ё рукамі``: stress on о/ё still to be set by hand).
    """
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = [m for m in XML_MEMBERS if m not in names]
        if missing:
            raise SystemExit(f"{zip_path} lacks {missing}")
        for member in XML_MEMBERS:
            with zf.open(member) as fh:
                todo = None
                for event, el in ET.iterparse(fh, events=("start", "end")):
                    if event == "start":
                        if el.tag == "Paradigm":
                            todo = el.get("todo")
                    elif el.tag == "Form":
                        yield el.text or "", todo
                    elif el.tag == "Paradigm":
                        el.clear()


def cased_key(clean: str, raw: str) -> str:
    """Runtime key of a token, keeping whether the database writes it capitalized."""
    return clean[:1].upper() + clean[1:] if raw[:1].isupper() else clean


def marked_parts(marked: str, raw: list) -> list:
    """Split a ``+``-marked form into the marked text of each runtime token."""
    parts, pos = [], 0
    for token in raw:
        seg = ""
        taken = 0
        while taken < len(token):
            c = marked[pos]
            pos += 1
            seg += c
            if c != "+":
                taken += 1
        if pos < len(marked) and marked[pos] == "+":
            seg += "+"
            pos += 1
        parts.append(seg)
    return parts


def collect(zip_path: str, stressor: _SileroStressor):
    """Map each runtime key to its single stressed-vowel ordinal, plus counts.

    A key is the token as the runtime cleans it, capitalized when the
    database writes the form capitalized.  A key enters the result only when
    every database reading that maps onto it has the same single stress, and
    a capitalized key and its lowercase spelling must agree with each other.
    Each part of a hyphenated or multiword form is a reading of its own
    token, so a compound can conflict with a plain form.
    """
    vowels = set(stressor._vowels)
    options: dict = {}
    counts = dict.fromkeys(
        ("forms", "single_stress", "multi_stress", "unstressed", "todo", "not_one_token"), 0)

    def record(key, ordinal, form):
        options.setdefault(key, {}).setdefault(ordinal, form)

    for marked, todo in forms(zip_path):
        counts["forms"] += 1
        n = marked.count("+")
        if n == 1:
            counts["single_stress"] += 1
        elif n:
            counts["multi_stress"] += 1
        else:
            counts["unstressed"] += 1
        if todo:
            counts["todo"] += 1
        form = marked.replace("+", "")
        if len(form.lower()) != len(form):
            counts["not_one_token"] += 1
            continue
        raw, clean, _ = stressor._tokenize(form)
        if len(raw) != 1:
            counts["not_one_token"] += 1
        for token, key, seg in zip(raw, clean, marked_parts(marked, raw)):
            if not key:
                continue
            marks = seg.count("+")
            if todo:
                # data not final: the key gets no entry, and the form still
                # vetoes a reading of the same spelling elsewhere
                ordinal = None
            elif marks == 1:
                # the database puts + after the stressed vowel
                ordinal = sum(1 for c in seg.lower()[:seg.index("+")] if c in vowels) - 1
            elif marks == 0 and len(raw) > 1:
                # a compound that marks only one of its parts
                continue
            else:
                ordinal = None
            lower = token.rstrip("-").lower()
            record(cased_key(key, token), ordinal, lower)
            typo = stressor._tokenize(lower.replace("'", "\u2019")).clean[0]
            if typo != key:
                record(cased_key(typo, token), ordinal, lower.replace("'", "\u2019"))
    entries = {}
    conflicts = 0
    for key, by_ordinal in options.items():
        if len(by_ordinal) == 1 and None not in by_ordinal:
            entries[key] = next(iter(by_ordinal.items()))
        elif len(by_ordinal) > 1:
            conflicts += 1
    # a name and a common word of one spelling (Я+на, яна+) that disagree:
    # neither key, because a sentence-initial token cannot tell them apart
    for key in [k for k in options if k[:1].isupper()]:
        lower = key[:1].lower() + key[1:]
        if lower in options and set(options[lower]) != set(options[key]):
            conflicts += entries.pop(key, None) is not None
            conflicts += entries.pop(lower, None) is not None
    counts["keys"] = len(options)
    counts["unambiguous_keys"] = len(entries)
    counts["conflicting_keys"] = conflicts
    return entries, counts


def wiktionary_readings(path: str) -> dict:
    """Cased spelling → stressed-vowel ordinals in the kaikki dump of en-Wiktionary.

    Headwords and inflected forms with exactly one acute count; hyphenated
    and multiword forms do not.
    """
    vowels = set("\u0430\u043e\u0443\u0456\u044d\u044b\u044f\u0435\u0451\u044e")
    readings: dict = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            entry = json.loads(line)
            if entry.get("lang_code") != "be":
                continue
            spellings = [entry.get("word") or ""]
            for form in entry.get("forms") or []:
                if {"romanization", "table-tags", "inflection-template"} & set(form.get("tags") or []):
                    continue
                spellings.append(form.get("form") or "")
            for text in spellings:
                text = unicodedata.normalize("NFC", text)
                if text.count(ACUTE) != 1 or " " in text or "-" in text:
                    continue
                i = text.index(ACUTE)
                if i < 1 or text[i - 1].lower() not in vowels:
                    continue
                plain = text.replace(ACUTE, "")
                readings.setdefault(plain, set()).add(sum(1 for c in text[:i - 1] if c.lower() in vowels))
    return readings


def wiktionary_disagrees(entries: dict, readings: dict) -> dict:
    """Keys whose spelling en-Wiktionary stresses on another vowel: key → its ordinals.

    The other case of the spelling counts too: a name Wiktionary stresses
    otherwise (Ста́лін) drops the common-noun key (сталі́н), because the
    runtime gives a name at the start of a sentence the lowercase key.
    """
    out = {}
    for key, (ordinal, _) in entries.items():
        other = key[:1].swapcase() + key[1:]
        found = readings.get(key, set()) | readings.get(other, set())
        if found and found != {ordinal}:
            out[key] = sorted(found)
    return out


def pipeline_ordinals(stressor: _SileroStressor, words: list) -> list:
    """Vowel ordinal the pipeline marks in each word, or None when it marks none."""
    vowels = set(stressor._vowels)
    out = []
    for b in range(0, len(words), BATCH):
        batch = words[b:b + BATCH]
        text = " ".join(batch)
        starts, pos = [], 0
        for w in batch:
            starts.append(pos)
            pos += len(w) + 1
        marks, _ = stressor._composed_mark_offsets(text)
        by_word = {}
        for m in marks:
            i = bisect.bisect_right(starts, m) - 1
            by_word.setdefault(i, sum(1 for c in batch[i][:m - starts[i]] if c in vowels))
        out.extend(by_word.get(i) for i in range(len(batch)))
    return out


def fetch_license() -> str:
    with urllib.request.urlopen(LICENSE_URL) as resp:
        data = resp.read()
    if hashlib.sha256(data).hexdigest() != LICENSE_SHA256:
        raise SystemExit(f"{LICENSE_URL}: sha256 differs from the pinned licence text")
    return data.decode("utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help=f"path to {RELEASE}.zip")
    ap.add_argument("--kaikki", required=True, help="path to the kaikki Belarusian dump (see KAIKKI_URL)")
    ap.add_argument("--out", required=True, help="output directory for the bel bundle files")
    ap.add_argument("--cache-dir", default=None,
                    help="stressonnx model cache holding the current bel bundle")
    args = ap.parse_args()

    for path, digest, what in ((args.zip, RELEASE_ZIP_SHA256, f"{RELEASE}.zip"),
                               (args.kaikki, KAIKKI_SHA256, "the pinned kaikki dump")):
        if sha256(path) != digest:
            raise SystemExit(f"{path} is not {what} (sha256 differs)")
    data = _download_files("bel", _MAIN_FILES, args.cache_dir, model_id="silero")
    with open(data["meta.json"], encoding="utf-8") as fh:
        meta = json.load(fh)
    stressor = _SileroStressor("be", cache_dir=args.cache_dir)
    stressor._ensure_loaded()
    # drop the entries of an earlier run so the pipeline is measured alone
    stressor._dictionary = {}

    entries, counts = collect(args.zip, stressor)
    contradicted = wiktionary_disagrees(entries, wiktionary_readings(args.kaikki))
    counts["wiktionary_disagrees"] = len(contradicted)
    keys = sorted(set(entries) - set(contradicted))
    # the form as the database spells it (ASCII apostrophe) and as the silero
    # alphabet spells it (’): both must come out right
    ascii_forms = [entries[k][1] for k in keys]
    typo_forms = [f.replace("'", "’") for f in ascii_forms]
    predicted = zip(pipeline_ordinals(stressor, ascii_forms), pipeline_ordinals(stressor, typo_forms))
    layer = {k: entries[k][0] for k, (a, b) in zip(keys, predicted)
             if a != entries[k][0] or b != entries[k][0]}
    counts["pipeline_agrees"] = len(keys) - len(layer)
    counts["exceptions_table_disagrees"] = sum(
        1 for k, o in layer.items() if k in stressor._exceptions
        and sum(1 for c in k[:stressor._exceptions[k][0]] if c in stressor._vowels) != o)
    counts["dictionary"] = len(layer)

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "dictionary.txt.gz")
    # mtime=0 keeps the gzip bytes reproducible
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for key in sorted(layer):
            gz.write(f"{key}\t{layer[key]}\n".encode())
    with open(os.path.join(args.out, "LICENSE.grammardb"), "w", encoding="utf-8") as fh:
        fh.write(NOTICE + fetch_license())

    # the keys the cross-check dropped, with both readings, for the record
    with open(os.path.join(args.out, "wiktionary_disagrees.json"), "w", encoding="utf-8") as fh:
        json.dump({k: {"form": entries[k][1], "grammardb": entries[k][0], "wiktionary": v}
                   for k, v in sorted(contradicted.items())}, fh, ensure_ascii=False, indent=1)
    meta["n_dictionary"] = len(layer)
    meta["dictionary_source"] = {
        "name": UPSTREAM,
        "release": RELEASE,
        "commit": RELEASE_COMMIT,
        "url": f"{UPSTREAM_URL}/releases/tag/{RELEASE}",
        "archive_sha256": RELEASE_ZIP_SHA256,
        "license": "CC-BY-SA-4.0",
        "license_file": "LICENSE.grammardb",
        "copyright": "GrammarDB by Aleś Bułojčyk and contributors",
        "selection": "spellings with one stress in the database, outside paradigms "
                     "marked todo, that en-Wiktionary (kaikki) does not stress on another "
                     "vowel and that the silero pipeline of this bundle (exceptions.txt.gz "
                     "and accentor.onnx) stresses differently; a key is capitalized when "
                     "the database writes the form capitalized",
        "cross_check": {"name": "en-Wiktionary via kaikki.org", "url": KAIKKI_URL,
                        "sha256": KAIKKI_SHA256, "dropped": len(contradicted)},
        "accentor_onnx_sha256": sha256(data["accentor.onnx"]),
        "exceptions_sha256": sha256(data["exceptions.txt.gz"]),
        "counts": counts,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(json.dumps(counts, indent=2))
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")


if __name__ == "__main__":
    main()
