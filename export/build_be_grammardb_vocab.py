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

The layer is keyed like the runtime keys a token: lowercased, with every
character outside the silero alphabet removed.  The alphabet keeps ’ but not
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
    python export/build_be_grammardb_vocab.py --zip RELEASE-202601.zip --out out/bel

Then upload ``out/bel/dictionary.txt.gz``, ``out/bel/meta.json`` and
``out/bel/LICENSE.grammardb`` to ``TigreGotico/stressonnx-models/bel/`` and
bump ``HF_REPO_REVISION``.
"""
import argparse
import bisect
import gzip
import hashlib
import json
import os
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
NOTICE = f"""The file dictionary.txt.gz of this bundle is derived from GrammarDB,
the grammar database of the Belarusian language, {RELEASE}
({UPSTREAM_URL}/releases/tag/{RELEASE}, commit {RELEASE_COMMIT}),
by Aleś Bulojčyk and contributors, and is licensed under the Creative
Commons Attribution-ShareAlike 4.0 International licence (CC-BY-SA-4.0).
It holds, for every spelling with one stress in the database, the ordinal
of the stressed vowel.  The other files of this bundle are derived from
silero_stress and stay under the MIT licence.  The full licence text
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
    """Every ``<Form>`` text of the release archive, in archive order."""
    if sha256(zip_path) != RELEASE_ZIP_SHA256:
        raise SystemExit(f"{zip_path} is not {RELEASE}.zip (sha256 differs)")
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
        missing = [m for m in XML_MEMBERS if m not in names]
        if missing:
            raise SystemExit(f"{zip_path} lacks {missing}")
        for member in XML_MEMBERS:
            with zf.open(member) as fh:
                for _, el in ET.iterparse(fh):
                    if el.tag == "Form":
                        yield el.text or ""
                    elif el.tag == "Paradigm":
                        el.clear()


def collect(zip_path: str, stressor: _SileroStressor):
    """Map each runtime key to its single stressed-vowel ordinal, plus counts.

    A key enters the result only when every database form that maps onto it
    has the same single stress.
    """
    vowels = set(stressor._vowels)
    options: dict = {}
    counts = dict.fromkeys(
        ("forms", "single_stress", "multi_stress", "unstressed", "not_one_token"), 0)
    for marked in forms(zip_path):
        counts["forms"] += 1
        n = marked.count("+")
        if n == 1:
            counts["single_stress"] += 1
        elif n:
            counts["multi_stress"] += 1
        else:
            counts["unstressed"] += 1
        form = marked.replace("+", "")
        lower = form.lower()
        raw, clean, _ = stressor._tokenize(lower)
        if len(lower) != len(form) or len(raw) != 1 or not clean[0]:
            # multiword entries and hyphenated compounds: the runtime never
            # looks them up whole
            counts["not_one_token"] += 1
            continue
        if n == 1:
            # the database puts + after the stressed vowel
            ordinal = sum(1 for c in marked.lower()[:marked.index("+")] if c in vowels) - 1
        else:
            ordinal = None
        keys = {clean[0], stressor._tokenize(lower.replace("'", "’")).clean[0]}
        for key in keys:
            options.setdefault(key, {}).setdefault(ordinal, lower)
    entries = {}
    conflicts = 0
    for key, by_ordinal in options.items():
        if len(by_ordinal) == 1 and None not in by_ordinal:
            entries[key] = next(iter(by_ordinal.items()))
        elif len(by_ordinal) > 1:
            conflicts += 1
    counts["keys"] = len(options)
    counts["unambiguous_keys"] = len(entries)
    counts["conflicting_keys"] = conflicts
    return entries, counts


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
    ap.add_argument("--out", required=True, help="output directory for the bel bundle files")
    ap.add_argument("--cache-dir", default=None,
                    help="stressonnx model cache holding the current bel bundle")
    args = ap.parse_args()

    data = _download_files("bel", _MAIN_FILES, args.cache_dir, model_id="silero")
    with open(data["meta.json"], encoding="utf-8") as fh:
        meta = json.load(fh)
    stressor = _SileroStressor("be", cache_dir=args.cache_dir)
    stressor._ensure_loaded()
    # drop the entries of an earlier run so the pipeline is measured alone
    stressor._dictionary = {}

    entries, counts = collect(args.zip, stressor)
    keys = sorted(entries)
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

    meta["n_dictionary"] = len(layer)
    meta["dictionary_source"] = {
        "name": UPSTREAM,
        "release": RELEASE,
        "commit": RELEASE_COMMIT,
        "url": f"{UPSTREAM_URL}/releases/tag/{RELEASE}",
        "archive_sha256": RELEASE_ZIP_SHA256,
        "license": "CC-BY-SA-4.0",
        "license_file": "LICENSE.grammardb",
        "copyright": "GrammarDB by Aleś Bulojčyk and contributors",
        "selection": "spellings with one stress in the database that the silero "
                     "pipeline of this bundle (exceptions.txt.gz and accentor.onnx) "
                     "stresses differently",
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
