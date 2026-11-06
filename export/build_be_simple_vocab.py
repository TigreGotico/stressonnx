"""Build the Belarusian ``simple`` vocabulary (``bel/vocab.gz``) from GrammarDB.

GrammarDB (https://github.com/Belarus/GrammarDB, CC-BY-SA-4.0) holds about
four million inflected Belarusian forms with ``+`` after the stressed vowel.
``build_be_grammardb_vocab.py`` reads the archive for the silero dictionary
layer; this script reuses its parser and its selection rules (one stress,
every hyphen part a reading of its own token, a name and a common word of
one spelling must agree, en-Wiktionary must not stress the spelling on
another vowel) and writes every selected spelling, not only the ones the
silero pipeline gets wrong.  Homographs stay out: the ``simple`` model has
no context to choose a reading.

Two rules differ from the silero layer.  A capitalized spelling whose
readings are all proper nouns (Але, the dative of the name Ала; the name
Яна) does not veto the common word of that spelling (але, яна): the
project defaults a homograph to its common reading, and a name is the rare
one.  A paradigm the database marks ``todo`` counts like any other here.  The ``simple`` model has no neural fallback, and the
todo paradigms hold the pronouns (яго, яна, яе) and the past tense of быць
(было, была): the 12,669 spellings they add agree with the gold sets at
0.90 (russtress, n=101), 0.97 (en-Wiktionary, n=479) and 0.999 (Zascienak,
n=1,707), the same rate as the other entries.

The vocabulary is keyed like the ``simple`` runtime keys a token: every
character outside the alphabet removed, lowercased.  The stored value is
the ordinal of the stressed vowel over the Cyrillic superset
(``vowel_ordinal_v2``), so ``ў`` counts as a vowel for the ordinal although
it never carries stress.

The vocabulary on the Hub before this script came from en-Wiktionary.  Its
entries are kept only for spellings the database does not have at all;
where the database has the spelling, the database decides, including when
the database lists it with several stresses.

``bel/meta.json`` is shared with the silero bundle.  The script rewrites the
current file with ``n_vocab`` and a ``vocab_source`` block and changes no
other key.

Usage::

    pip install "stressonnx[test]"
    python export/build_be_simple_vocab.py --zip RELEASE-202601.zip \\
        --kaikki kaikki.org-dictionary-Belarusian.jsonl \\
        --wiktionary-vocab vocab.gz --out out/bel

``--wiktionary-vocab`` is the ``bel/vocab.gz`` of the current Hub revision.
Then upload ``out/bel/vocab.gz``, ``out/bel/meta.json``, ``out/bel/README.md``
and ``out/bel/LICENSE.grammardb`` to ``TigreGotico/stressonnx-models/bel/``
and bump ``HF_REPO_REVISION``.
"""
import argparse
import collections
import gzip
import json
import os

from build_be_grammardb_vocab import (
    KAIKKI_SHA256, KAIKKI_URL, RELEASE, RELEASE_COMMIT, RELEASE_ZIP_SHA256, UPSTREAM,
    UPSTREAM_URL, collect, fetch_license, forms, NOTICE, sha256, wiktionary_disagrees,
    wiktionary_readings,
)
from stressonnx._common import SCRIPT_VOWELS
from stressonnx.backends.silero import _SileroStressor
from stressonnx.backends.simple import SimpleStressor, _load_vocab
from stressonnx.download import _download_files
from stressonnx.registry import LANGUAGES

SUPERSET = SCRIPT_VOWELS["cyrillic"]
APOSTROPHES = "'\u2019\u02bc"
README = f"""# bel stress vocabulary

Word → stressed-vowel-ordinal pairs (`vowel_ordinal_v2`, ordinals over the
Cyrillic vowel superset) for the `simple` model of TigreGotico/stressonnx.
Every spelling of GrammarDB {RELEASE} ({UPSTREAM_URL}, CC-BY-SA-4.0) with
exactly one stress, paradigms marked `todo` included, that en-Wiktionary
does not stress on another vowel.  Spellings the database does not have keep
their entry from the English Wiktionary (CC-BY-SA).  Homographs are left
out.  Built by `export/build_be_simple_vocab.py`.  The licence notice is in
`LICENSE.grammardb`.  `meta.json` is shared with the silero bundle of this
directory.
"""


def superset_ordinal(form: str, ordinal: int, vowels: set) -> int:
    """Ordinal over the Cyrillic superset of the *ordinal*-th language vowel of *form*."""
    seen = -1
    for i, c in enumerate(form):
        if c in vowels:
            seen += 1
            if seen == ordinal:
                return sum(1 for d in form[:i] if d in SUPERSET)
    raise ValueError(f"{form!r} has no vowel {ordinal}")


def simple_entries(entries: dict, simple: SimpleStressor, vowels: set):
    """*entries* keyed the way the ``simple`` runtime looks a token up, and the drop counts.

    The silero parser keys a name with its capital and an apostrophe form
    under both spellings; the ``simple`` key is lowercase with every
    character outside its alphabet removed, so those twins fold into one.
    When a name folds onto a common word that disagrees (В’етнаму onto
    ве+тнаму) the common word keeps the key; two common words that disagree
    cannot occur and raise.
    """
    vocab: dict = {}
    names: dict = {}
    dropped = collections.Counter()
    for key, (ordinal, form) in entries.items():
        form = form.lower()
        # a form with a letter outside the alphabet (Russian и, Latin) is
        # keyed without that letter, so its ordinal would not fit the key
        if set(form) - simple._alpha - set(APOSTROPHES):
            dropped["outside alphabet"] += 1
            continue
        target = names if key[:1].isupper() else vocab
        lower = simple._key(key)
        value = superset_ordinal(form, ordinal, vowels)
        if target.setdefault(lower, value) != value:
            raise ValueError(f"{lower}: {target[lower]} and {value}")
    for lower, value in names.items():
        if vocab.setdefault(lower, value) != value:
            dropped["name under a common word"] += 1
    return vocab, dropped


def spellings(zip_path: str, stressor: _SileroStressor, simple: SimpleStressor) -> set:
    """Every ``simple`` key the database has a form for, with any stress."""
    keys = set()
    for marked, _, _ in forms(zip_path):
        form = marked.replace("+", "")
        if len(form.lower()) != len(form):
            continue
        keys.update(simple._key(k) for k in stressor._tokenize(form).clean if k)
    return keys


def merge_wiktionary(vocab: dict, wiktionary: dict, known: set) -> collections.Counter:
    """Add the Wiktionary entries for spellings outside *known* to *vocab*; return counts."""
    counts: collections.Counter = collections.Counter()
    for word, ordinal in wiktionary.items():
        if word in vocab:
            counts["in vocabulary, same" if vocab[word] == ordinal else "in vocabulary, differs"] += 1
        elif word in known:
            counts["spelling in GrammarDB without an entry"] += 1
        else:
            vocab[word] = ordinal
            counts["kept"] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True, help=f"path to {RELEASE}.zip")
    ap.add_argument("--kaikki", required=True, help="path to the kaikki Belarusian dump (see KAIKKI_URL)")
    ap.add_argument("--wiktionary-vocab", required=True, help="bel/vocab.gz of the current Hub revision")
    ap.add_argument("--out", required=True, help="output directory for the bel files")
    ap.add_argument("--cache-dir", default=None, help="stressonnx model cache holding the current bel bundle")
    args = ap.parse_args()

    for path, digest, what in ((args.zip, RELEASE_ZIP_SHA256, f"{RELEASE}.zip"),
                               (args.kaikki, KAIKKI_SHA256, "the pinned kaikki dump")):
        if sha256(path) != digest:
            raise SystemExit(f"{path} is not {what} (sha256 differs)")
    meta_path = _download_files("bel", ["meta.json"], args.cache_dir, model_id="silero")["meta.json"]
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    stressor = _SileroStressor("be", cache_dir=args.cache_dir)
    stressor._ensure_loaded()
    vowels = set(stressor._vowels)
    simple = SimpleStressor("be")
    # the silero bundle and the simple model must agree on what a vowel is;
    # the simple alphabet lacks the apostrophe, which simple._key removes
    if vowels != set(simple._vowels) or set(meta["alpha"]) - {"\u2019"} != simple._alpha:
        raise SystemExit("silero bel meta and languages/be.json disagree on the alphabet or vowels")

    entries, counts = collect(args.zip, stressor, todo_vetoes=False, common_over_name=True)
    contradicted = wiktionary_disagrees(entries, wiktionary_readings(args.kaikki))
    counts["wiktionary_disagrees"] = len(contradicted)
    vocab, dropped = simple_entries({k: v for k, v in entries.items() if k not in contradicted}, simple, vowels)
    counts["simple_key_drops"] = dict(dropped)
    counts["grammardb_entries"] = len(vocab)
    known = spellings(args.zip, stressor, simple)
    counts["grammardb_spellings"] = len(known)
    wiktionary = _load_vocab(args.wiktionary_vocab)
    counts["wiktionary_in"] = len(wiktionary)
    # the previous file still holds a key the runtime drops at load
    usable_wiktionary = simple._usable(wiktionary)
    merged = merge_wiktionary(vocab, usable_wiktionary, known)
    merged["unusable"] = len(wiktionary) - len(usable_wiktionary)
    counts["wiktionary"] = dict(merged)
    counts["vocab"] = len(vocab)
    usable = simple._usable(vocab)
    if usable != vocab:
        odd = sorted(set(vocab.items()) ^ set(usable.items()))[:20]
        raise SystemExit(f"the runtime would drop or re-key an entry; the file must not depend on that pass: {odd}")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "vocab.gz")
    # mtime=0 keeps the gzip bytes reproducible
    with open(path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for word in sorted(vocab):
            gz.write(f"{word} {vocab[word]}\n".encode())
    meta["n_vocab"] = len(vocab)
    meta["vocab_source"] = {
        "name": UPSTREAM,
        "release": RELEASE,
        "commit": RELEASE_COMMIT,
        "url": f"{UPSTREAM_URL}/releases/tag/{RELEASE}",
        "archive_sha256": RELEASE_ZIP_SHA256,
        "license": "CC-BY-SA-4.0",
        "license_file": "LICENSE.grammardb",
        "copyright": "GrammarDB by Aleś Bułojčyk and contributors",
        "selection": "every spelling with one stress in the database, paradigms marked "
                     "todo included, a common word kept when only a proper noun of the same "
                     "spelling disagrees, that en-Wiktionary (kaikki) does not stress on another "
                     "vowel, keyed as the simple runtime keys a token; plus the en-Wiktionary entries of the previous "
                     "vocabulary for spellings the database does not have",
        "cross_check": {"name": "en-Wiktionary via kaikki.org", "url": KAIKKI_URL,
                        "sha256": KAIKKI_SHA256, "dropped": len(contradicted)},
        "wiktionary_vocab_sha256": sha256(args.wiktionary_vocab),
        "counts": counts,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    with open(os.path.join(args.out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(README)
    with open(os.path.join(args.out, "LICENSE.grammardb"), "w", encoding="utf-8") as fh:
        fh.write(NOTICE + fetch_license())
    print(json.dumps(counts, indent=2, ensure_ascii=False))
    print(f"wrote {path} ({os.path.getsize(path)} bytes, {len(vocab)} entries)")


if __name__ == "__main__":
    main()
