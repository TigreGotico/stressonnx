"""Build the Ukrainian dictionary layer of the ``silero`` model.

ukrainian-word-stress (lang-uk, MIT, https://github.com/lang-uk/ukrainian-word-stress)
ships a marisa-trie of about 2.9 million Ukrainian word forms with their
stress.  Upstream derives the forms from "Dictionaries of Ukraine"
(Словники України) by ULIF, https://lcorp.ulif.org.ua/dictua/.  Most forms have exactly one stress; heteronyms (за́мок/замо́к) carry
several readings that upstream resolves with Stanza POS tags.

This script takes every form with exactly one possible stress and writes it
to the ``exceptions.txt.gz`` file of the ``ukr`` silero bundle.  The runtime
looks a word up in that file before it runs the neural model, so dictionary
forms win and the model handles everything else: heteronyms, forms with
variant stress (по́ми́лка), and words outside the dictionary.

A form also stays out when the Wiktionary vocabulary of the uk ``simple``
model stresses it on another vowel: two sources that disagree do not make
a form unambiguous.  The dictionary lists, for example, only йо́го for
``його``, and the cross-check leaves the pronoun його́ to the model.

Forms that the neural model already stresses like the dictionary are left
out.  Their output does not change, so the file holds only the forms where
the dictionary overrides the model.  That keeps the download and the memory cost
small.  A rebuilt ``accentor.onnx`` requires a rebuild of this file too.

Exceptions are keyed like the runtime keys a token: lowercased, with every
character outside the silero alphabet removed.  The silero alphabet has no
apostrophe, so ``м'ячем``, ``м’ячем`` and ``мʼячем`` all look up ``мячем``.
The stored stress is a character index into the token as written (with its
apostrophe), because that is how the runtime applies it.

Usage::

    pip install "stressonnx[export]"
    python export/build_uk_word_stress_vocab.py --out out/ukr

Then upload ``out/ukr/exceptions.txt.gz`` and ``out/ukr/meta.json`` to
``TigreGotico/stressonnx-models/ukr/`` and bump ``HF_REPO_REVISION``.
"""
import argparse
import bisect
import gzip
import hashlib
import json
import os
from importlib import metadata, resources

import marisa_trie

from stressonnx._common import SCRIPT_VOWELS
from stressonnx.backends.silero import _SileroStressor
from stressonnx.backends.simple import _load_vocab
from stressonnx.download import _download_files
from stressonnx.registry import _MAIN_FILES, _SIMPLE_FILES, hf_dir

UPSTREAM = "ukrainian-word-stress"
UPSTREAM_VERSION = "2.1.0"

#: Value layout of the upstream trie (ukrainian_word_stress/tags.py): a value
#: without a record separator is the list of accent positions; otherwise it
#: is a list of ``accents POS_SEP tags REC_SEP`` records, one per reading.
POS_SEP = b"\xfe"
REC_SEP = b"\xff"

#: Batch size for running the silero model over candidate forms.
BATCH = 4096


def read_trie() -> marisa_trie.BytesTrie:
    version = metadata.version(UPSTREAM)
    if version != UPSTREAM_VERSION:
        raise SystemExit(f"{UPSTREAM}=={UPSTREAM_VERSION} required, found {version}")
    trie = marisa_trie.BytesTrie()
    trie.load(str(resources.files("ukrainian_word_stress").joinpath("data/stress.trie")))
    return trie


def stress_options(value: bytes) -> set:
    """Distinct accent-position tuples one trie value allows.

    An accent position is the index of the character *after* the stressed
    vowel (upstream inserts its mark there).
    """
    if REC_SEP not in value:
        return {tuple(value)}
    return {tuple(rec.partition(POS_SEP)[0]) for rec in value.split(REC_SEP) if rec}


def collect(trie: marisa_trie.BytesTrie, stressor: _SileroStressor):
    """Map each runtime key to its single stressed-vowel index, plus counts.

    A key enters the result only when every trie form that maps onto it has
    the same single stress.
    """
    vowels = set(stressor._vowels)
    options: dict = {}
    counts = dict.fromkeys(
        ("forms", "single_stress", "multi_stress", "unstressed", "heteronym",
         "not_one_token", "bad_position"), 0)
    for form in trie.keys():
        counts["forms"] += 1
        opts = stress_options(trie[form][0])
        if len(opts) > 1:
            counts["heteronym"] += 1
        elif len(next(iter(opts))) == 1:
            counts["single_stress"] += 1
        elif next(iter(opts)):
            counts["multi_stress"] += 1
        else:
            counts["unstressed"] += 1
        lower = form.lower()
        raw, clean, _ = stressor._tokenize(lower)
        if len(lower) != len(form) or len(raw) != 1 or not clean[0]:
            # multiword entries and hyphenated compounds: the runtime never
            # looks them up whole
            counts["not_one_token"] += 1
            continue
        key = clean[0]
        indices = set()
        for opt in opts:
            if len(opt) == 1 and 0 < opt[0] <= len(lower) and lower[opt[0] - 1] in vowels:
                indices.add(opt[0] - 1)
            else:
                if len(opt) == 1:
                    counts["bad_position"] += 1
                indices.add(None)
        options.setdefault(key, {})
        for i in indices:
            options[key].setdefault(i, lower)
    entries = {}
    conflicts = 0
    for key, by_index in options.items():
        if len(by_index) == 1 and None not in by_index:
            entries[key] = next(iter(by_index.items()))
        elif len(by_index) > 1:
            conflicts += 1
    counts["keys"] = len(options)
    counts["unambiguous_keys"] = len(entries)
    counts["conflicting_keys"] = conflicts
    return entries, counts


def wiktionary_disagrees(entries: dict, vocab: dict) -> set:
    """Keys whose form the Wiktionary vocabulary stresses on another vowel.

    The vocabulary stores vowel ordinals and spells the apostrophe ’.
    """
    vowels = SCRIPT_VOWELS["cyrillic"]
    disagree = set()
    for key, (index, form) in entries.items():
        ordinal = vocab.get(form.replace("'", "’"))
        if ordinal is not None and ordinal != sum(1 for c in form[:index] if c in vowels):
            disagree.add(key)
    return disagree


def silero_marks(stressor: _SileroStressor, words: list) -> list:
    """Character index silero marks in each word, or None when it marks none."""
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
            by_word.setdefault(i, m - starts[i])
        out.extend(by_word.get(i) for i in range(len(batch)))
    return out


def sha256(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory for the ukr bundle files")
    ap.add_argument("--cache-dir", default=None,
                    help="stressonnx model cache holding the current ukr bundle")
    args = ap.parse_args()

    data = _download_files("ukr", _MAIN_FILES, args.cache_dir, model_id="silero")
    with open(data["meta.json"], encoding="utf-8") as fh:
        meta = json.load(fh)
    stressor = _SileroStressor("uk", cache_dir=args.cache_dir)
    stressor._ensure_loaded()
    if stressor._exceptions and "exceptions_source" not in meta:
        raise SystemExit("the ukr bundle has exceptions from another source; "
                         "this script would overwrite them")
    # drop the entries of an earlier run so the neural model is measured alone
    stressor._exceptions = {}

    entries, counts = collect(read_trie(), stressor)
    simple = _download_files(hf_dir("uk", "simple"), _SIMPLE_FILES, args.cache_dir,
                             model_id="simple")
    contradicted = wiktionary_disagrees(entries, _load_vocab(simple["vocab.gz"]))
    counts["wiktionary_disagrees"] = len(contradicted)
    keys = sorted(set(entries) - contradicted)
    predicted = silero_marks(stressor, [entries[k][1] for k in keys])
    corrections = {k: entries[k][0] for k, p in zip(keys, predicted) if p != entries[k][0]}
    counts["silero_agrees"] = len(keys) - len(corrections)
    counts["exceptions"] = len(corrections)

    os.makedirs(args.out, exist_ok=True)
    exc_path = os.path.join(args.out, "exceptions.txt.gz")
    # mtime=0 keeps the gzip bytes reproducible
    with open(exc_path, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as gz:
        for key in sorted(corrections):
            gz.write(f"{key}\t{corrections[key]}\t-1\n".encode())

    meta["n_exceptions"] = len(corrections)
    meta["exceptions_source"] = {
        "name": UPSTREAM,
        "version": UPSTREAM_VERSION,
        "url": "https://github.com/lang-uk/ukrainian-word-stress",
        "license": "MIT",
        "copyright": "Copyright (c) 2022 lang-uk",
        "derived_from": "\"Dictionaries of Ukraine\" (Словники України) by ULIF, "
                        "https://lcorp.ulif.org.ua/dictua/, as stated by upstream",
        "selection": "forms with one unambiguous stress that the ukr_simple "
                     "Wiktionary vocabulary does not contradict and that the "
                     "silero accentor.onnx of this bundle stresses differently",
        "accentor_onnx_sha256": sha256(data["accentor.onnx"]),
        "counts": counts,
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)
        fh.write("\n")
    print(json.dumps(counts, indent=2))
    print(f"wrote {exc_path} ({os.path.getsize(exc_path)} bytes)")


if __name__ == "__main__":
    main()
