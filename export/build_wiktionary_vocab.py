"""Build a simple-accentor vocabulary from a kaikki.org Wiktionary dump.

kaikki.org (Tatu Ylonen's wiktextract) publishes machine-readable English
Wiktionary extractions per language.  For many languages the canonical
headword form carries stress diacritics that the plain orthography omits
(``вода`` → ``вода́``); this script recovers ``word → stressed-char-index``
pairs from them and writes the ``vocab.gz`` + ``meta.json`` pair that
``SimpleStressor`` consumes.

Wiktionary content is CC BY-SA; the generated vocabulary files must carry
attribution (a README.md is written next to the vocab for upload to the HF
model repo).

Usage::

    python export/build_wiktionary_vocab.py --lang bul --jsonl Bulgarian.jsonl --out out/bul
"""
import argparse
import gzip
import json
import os
import unicodedata

#: Combining marks that Wiktionary uses to indicate the stressed vowel.
#: Acute/grave cover the Slavic Cyrillic conventions; the tonal marks
#: (circumflex, double grave, inverted breve) appear on Slovene headwords,
#: where a tonally marked vowel is the stressed one.
STRESS_MARKS = {"́", "̀", "̂", "̏", "̑"}

#: Per-language alphabet + vowel inventories and the OOV rule to record.
LANG_META = {
    "bul": dict(
        alpha="абвгдежзийклмнопрстуфхцчшщъьюя",
        vowels="аеиоуъюя",  # ъ is a full vowel in Bulgarian
        oov_rule="none",
    ),
    "mkd": dict(
        alpha="абвгдѓежзѕијклљмнњопрстќуфхцчџш",
        vowels="аеиоу",
        oov_rule="antepenult",
    ),
    "slv": dict(
        alpha="abcčdefghijklmnoprsštuvzž",
        vowels="aeiou",
        oov_rule="none",
    ),
    "lav": dict(
        alpha="aābcčdeēfgģhiījkķlļmnņoōprsštuūvzž",
        vowels="aāeēiīoōuū",
        oov_rule="first",
    ),
    "ukr_simple": dict(
        alpha="абвгґдеєжзиіїйклмнопрстуфхцчшщьюя’",
        vowels="аеєиіїоуюя",
        oov_rule="none",
        lang_code="uk",
    ),
}


def stressed_index(plain: str, marked: str, vowels: str):
    """Index in *plain* of the vowel the *marked* form stresses, or None.

    Walks the NFD-decomposed marked form against the plain word; the first
    stress mark encountered names the preceding character.  Returns None
    when the forms do not align, the mark does not sit on a vowel of the
    language, or more than one stress mark is present (ambiguous).
    """
    decomposed = unicodedata.normalize("NFD", marked)
    idx = None
    pi = 0
    for ch in decomposed:
        if ch in STRESS_MARKS:
            if pi == 0 or pi > len(plain):
                return None
            if idx is not None:
                return None  # second mark → ambiguous
            idx = pi - 1
            continue
        if unicodedata.combining(ch):
            continue  # diacritic that is part of the orthography (č, ī …)
        if pi >= len(plain):
            return None
        # compare base characters so precomposed letters (ī, č) still match
        plain_base = unicodedata.normalize("NFD", plain[pi])[0]
        if unicodedata.normalize("NFD", ch)[0].lower() != plain_base.lower():
            return None
        pi += 1
    if pi != len(plain) or idx is None:
        return None
    if plain[idx].lower() not in vowels:
        return None
    return idx


def candidate_forms(entry: dict):
    """Stress-marked spellings of an entry: canonical forms first, then word."""
    for f in entry.get("forms", []):
        if "canonical" in (f.get("tags") or []):
            yield f.get("form", "")
    yield entry.get("word", "")


def build(lang: str, jsonl_path: str):
    meta = LANG_META[lang]
    code = meta.get("lang_code")
    vowels = meta["vowels"]
    alpha = set(meta["alpha"])
    vocab = {}
    total = 0
    for line in open(jsonl_path, encoding="utf-8"):
        entry = json.loads(line)
        if code and entry.get("lang_code") not in (code,):
            continue
        word = entry.get("word", "")
        total += 1
        if not word or " " in word or "-" in word:
            continue
        plain = word.lower()
        if any(c not in alpha for c in plain):
            continue
        for marked in candidate_forms(entry):
            if not marked or marked == word:
                continue
            idx = stressed_index(plain, marked.lower(), vowels)
            if idx is not None:
                # store the vowel ORDINAL over the script superset (v2
                # vocabulary format), not the character index
                from stressonnx._common import SCRIPT_VOWELS
                sup = SCRIPT_VOWELS[
                    "latin" if plain[0].isascii() or lang in ("slv", "lav") else "cyrillic"
                ]
                ordinal = sum(1 for c in plain[:idx] if c in sup)
                vocab.setdefault(plain, ordinal)
                break
    return vocab, total


def write(lang: str, vocab: dict, out_dir: str):
    meta = LANG_META[lang]
    os.makedirs(out_dir, exist_ok=True)
    with gzip.open(os.path.join(out_dir, "vocab.gz"), "wt", encoding="utf-8") as fh:
        for word, idx in sorted(vocab.items()):
            fh.write(f"{word} {idx}\n")
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(
            {
                "family": "simple_accentor",
                "lang": lang,
                "alpha": meta["alpha"],
                "vowels": meta["vowels"],
                "oov_rule": meta["oov_rule"],
                "n_vocab": len(vocab),
                "vocab_format": "vowel_ordinal_v2",
                "source": "English Wiktionary via kaikki.org (wiktextract)",
                "license": "CC BY-SA 4.0 (Wiktionary contributors)",
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(
            f"# {lang} stress vocabulary\n\n"
            "Word → stressed-character-index pairs extracted from stress-marked\n"
            "headwords of the English Wiktionary, via the machine-readable\n"
            "[kaikki.org](https://kaikki.org) extraction (wiktextract, Tatu\n"
            "Ylonen).  Content license: CC BY-SA 4.0, © Wiktionary\n"
            "contributors.  Built by `export/build_wiktionary_vocab.py` in\n"
            "TigreGotico/stressonnx.\n"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, choices=sorted(LANG_META))
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    vocab, total = build(args.lang, args.jsonl)
    write(args.lang, vocab, args.out)
    print(f"{args.lang}: {len(vocab)} vocab entries from {total} Wiktionary entries")


if __name__ == "__main__":
    main()
