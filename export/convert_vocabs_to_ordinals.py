"""Convert vocabularies from character-index to vowel-ordinal format.

The historical format stored ``word → stressed character index``; character
positions are orthography-fragile (case folding, digraphs, combining marks).
The v2 format stores ``word → stressed vowel ordinal`` — "the Nth vowel of
the word" — counted over the per-script vowel superset in
``stressonnx/_common.py``, which survives every one of those transformations.

Run from a checkout (network + HF write access required)::

    python export/convert_vocabs_to_ordinals.py            # validate only
    python export/convert_vocabs_to_ordinals.py --upload   # convert + upload

After uploading, bump ``HF_REPO_REVISION`` in ``stressonnx/registry.py``.
"""
import argparse
import gzip
import json
import os
import tempfile

from stressonnx._common import SCRIPT_VOWELS, lower_preserving_length
from stressonnx.download import _download_files
from stressonnx.registry import LANGUAGES


def convert(lang: str, spec: dict):
    subdir = spec["hf"]["simple"]
    data = _download_files(subdir, ["vocab.gz", "meta.json"], model_id="simple")
    meta = json.load(open(data["meta.json"], encoding="utf-8"))
    vowels = SCRIPT_VOWELS[spec["script"]]

    entries = {}
    dropped = []
    for line in gzip.open(data["vocab.gz"], "rt", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            word, idx_s = line.rsplit(None, 1)
            idx = int(idx_s)
        except ValueError:
            dropped.append(line)
            continue
        lower = lower_preserving_length(word)
        if not (0 <= idx < len(lower)) or lower[idx] not in vowels:
            dropped.append(line)
            continue
        ordinal = sum(1 for c in lower[:idx] if c in vowels)
        entries[word] = ordinal
    meta = dict(meta, vocab_format="vowel_ordinal_v2", n_vocab=len(entries))
    return subdir, entries, meta, dropped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--langs", nargs="*")
    args = ap.parse_args()

    to_upload = []
    for lang, spec in sorted(LANGUAGES.items()):
        if args.langs and lang not in args.langs:
            continue
        subdir, entries, meta, dropped = convert(lang, spec)
        pct = 100 * len(dropped) / max(1, len(entries) + len(dropped))
        print(f"{lang:8} ({subdir:10}): {len(entries):6} entries, "
              f"{len(dropped)} dropped ({pct:.2f}%)"
              + (f"  e.g. {dropped[:3]}" if dropped else ""))
        to_upload.append((subdir, entries, meta))

    if not args.upload:
        print("\n(validation only — rerun with --upload to publish)")
        return

    from huggingface_hub import HfApi
    api = HfApi()
    for subdir, entries, meta in to_upload:
        with tempfile.TemporaryDirectory() as tmp:
            with gzip.open(os.path.join(tmp, "vocab.gz"), "wt", encoding="utf-8") as fh:
                for word, ordinal in sorted(entries.items()):
                    fh.write(f"{word} {ordinal}\n")
            with open(os.path.join(tmp, "meta.json"), "w", encoding="utf-8") as fh:
                json.dump(meta, fh, ensure_ascii=False, indent=2)
            api.upload_folder(
                repo_id="TigreGotico/stressonnx-models",
                folder_path=tmp,
                path_in_repo=subdir,
                commit_message=f"feat: {subdir} vocabulary in vowel-ordinal format",
            )
            print("uploaded", subdir)
    print("\nNow bump HF_REPO_REVISION in stressonnx/registry.py.")


if __name__ == "__main__":
    main()
