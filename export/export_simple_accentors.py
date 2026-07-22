"""Export silero_stress SimpleAccentor vocabulary data for all supported languages.

SimpleAccentor is purely vocabulary + rule-based (no neural net).  This script
vendors the per-language vocab gz files and writes a meta.json describing the
alphabet, vowels and OOV fallback rule for each language.

No torch required.  Run once per language (or all at once).

Usage::

    python export/export_simple_accentors.py --lang bak
    python export/export_simple_accentors.py --all
    python export/export_simple_accentors.py --all --out_base /tmp/out
"""
import argparse
import gzip
import json
import os
import shutil

# ---------------------------------------------------------------------------
# Per-language alphabet + vowel metadata (mirrors silero_stress/simple_accentor.py)
# ---------------------------------------------------------------------------
SIMPLE_LANGS_META = {
    "aze_cyr": {
        "alpha": "абвгғдеәжзиыјкҝлмноөпрстуүфхһчҹш'",
        "vowels": "аеәиыоөуү",
        "oov_rule": "last",
    },
    "aze_lat": {
        "alpha": "abcçdeəfgğhxıijkqlmnoöprsştuüvyzw'",
        "vowels": "aeəıioöuü",
        "oov_rule": "last",
    },
    "uzb_cyr": {
        "alpha": "абдеэфгҳижклмнопқрстувхйзўғшчнгъ",
        "vowels": "аеиоуў",
        "oov_rule": "last",
    },
    "uzb_lat": {
        "alpha": "abcdefghijklmnopqrstuvxyzw'",
        "vowels": "aeiou",
        "oov_rule": "last",
    },
    "bak": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяёғҙҡңҫүһәө",
        "vowels": "аеиоуыэюяёүәө",
        "oov_rule": "last",
    },
    "bel": {
        "alpha": "абвгдежзйклмнопрстуфхцчшыьэюяёіў'",
        "vowels": "аоуіэыяеёю",
        "oov_rule": "none",   # bel: OOV → no stress (stress_idx = None)
    },
    "chv": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяёҫӑӗӳ",
        "vowels": "аеиоуыэюяёӑӗӳ",
        "oov_rule": "last",
    },
    "erz": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяё",
        "vowels": "аеиоуыэюяё",
        "oov_rule": "first",
    },
    "hye": {
        "alpha": "աբգդեզէըթժիլխծկհձղճմյնշոչպջռսվտրցւփքօֆև",
        "vowels": "աեէըիուօ",
        "oov_rule": "last",
    },
    "kat": {
        "alpha": "აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
        "vowels": "აეიოუ",
        "oov_rule": "kat",    # special: ≤3 vowels → first, else penultimate
    },
    "kaz": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщыьэюяіғқңүұһәө",
        "vowels": "аеиоуыэюяіүұәө",
        "oov_rule": "last",
    },
    "kbd": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяёӏ",
        "vowels": "аеиоуыэюяё",
        "oov_rule": "last",
    },
    "kir": {
        "alpha": "абвгдежзийклмнопрстуфхцчшыьэюяёңүө",
        "vowels": "аеиоуыэюяёүө",
        "oov_rule": "last",
    },
    "kjh": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяёіғңҷӧӱ",
        "vowels": "аеиоуыэюяёіӧӱ",
        "oov_rule": "last",
    },
    "mdf": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяё",
        "vowels": "аеиоуыэюяё",
        "oov_rule": "first",
    },
    "sah": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяёҕҥүһө",
        "vowels": "аеиоуыэюяёүө",
        "oov_rule": "last",
    },
    "tat": {
        "alpha": "абвгдежзийклмнопрстуфхцчшъыьэюяҗңүһәө",
        "vowels": "аеиоуыэюяүәө",
        "oov_rule": "last",
    },
    "tgk": {
        "alpha": "абвгдежзийклмнопрстуфхчшъэюяёғқҳҷӣӯ",
        "vowels": "аеиоуэюяёӣӯ",
        "oov_rule": "last",
    },
    "udm": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяёӝӟӥӧӵ",
        "vowels": "аеиоуыэюяёӥӧ",
        "oov_rule": "last",
    },
    "xal": {
        "alpha": "абвгдежзийклмнопрстуфхцчшщъыьэюяҗңүһәө",
        "vowels": "аеиоуыэюяүәө",
        "oov_rule": "last",
    },
}

SILERO_VOCAB_DIR = None  # auto-detect from importlib

# ---------------------------------------------------------------------------


def _find_vocab_dir():
    # Use the installed package path directly — importlib.resources returns a
    # MultiplexedPath that cannot be used as a plain filesystem path.
    import silero_stress
    pkg_dir = os.path.dirname(os.path.abspath(silero_stress.__file__))
    return os.path.join(pkg_dir, "data", "vocabularies")


def _count_vocab(path):
    with gzip.open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def export_lang(lang, out_dir):
    meta_info = SIMPLE_LANGS_META[lang]
    os.makedirs(out_dir, exist_ok=True)

    vocab_dir = _find_vocab_dir()
    src = os.path.join(vocab_dir, f"vocab-{lang}.gz")
    dst = os.path.join(out_dir, f"vocab.gz")
    shutil.copy2(src, dst)
    n_vocab = _count_vocab(dst)
    print(f"[{lang}] copied vocab ({n_vocab} entries) → {dst}")

    # Build re_cond the same way SimpleAccentor does
    alpha = meta_info["alpha"]
    alpha_set = "".join(sorted(set(alpha + alpha.upper())))

    meta = {
        "family": "simple_accentor",
        "lang": lang,
        "alpha": meta_info["alpha"],
        "vowels": meta_info["vowels"],
        "oov_rule": meta_info["oov_rule"],
        "n_vocab": n_vocab,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"[{lang}] wrote meta.json")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lang", choices=list(SIMPLE_LANGS_META.keys()))
    group.add_argument("--all", action="store_true")
    parser.add_argument("--out_base", default=None,
                        help="Base output dir (default: parent of this script)")
    args = parser.parse_args()

    HERE = os.path.dirname(os.path.abspath(__file__))
    out_base = args.out_base or os.path.dirname(HERE)

    langs = list(SIMPLE_LANGS_META.keys()) if args.all else [args.lang]
    for lang in langs:
        export_lang(lang, os.path.join(out_base, lang))
    print("Done.")
