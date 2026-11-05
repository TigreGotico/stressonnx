"""19th-century Russian prose with dictionary-tool stress marks (SILVER, MIT).

Source: https://huggingface.co/datasets/nevmenandr/accentual-syllabic-verse-in-russian-prose
(Orekhov), 462 novels from the "Corpus of 19th-century narrative prose"
(Sobchuk, Lekarevich 2020), marked by the ``ru-accent-poet`` package.  An
apostrophe after a vowel marks stress; ``!`` after ё inside a word marks ё.

The full corpus is 330 MB, so a fixed sample is scored: every
:data:`FILE_STEP`-th file of the sorted file list, and in each file the first
:data:`PER_FILE` sentences of 4 to 40 tokens with a usable target.
"""
import json
import re
from collections import Counter

from gold._cache import cached
from scoring import STRESS, marked_sentence_row

REPO = "nevmenandr/accentual-syllabic-verse-in-russian-prose"
REVISION = "7c12708c16a1453274f0c49bd85b5ecb905b7a4a"
TREE = f"https://huggingface.co/api/datasets/{REPO}/tree/{REVISION}/accented_texts"
FILE = f"https://huggingface.co/datasets/{REPO}/resolve/{REVISION}/accented_texts/{{name}}"
FILE_STEP = 46
PER_FILE = 200

_VOWEL_APOSTROPHE = re.compile(r"(?<=[аеёиоуыэюяАЕЁИОУЫЭЮЯ])'")
_YO_BANG = re.compile(r"(?<=[ёЁ])!(?=\w)")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _to_acute(text: str) -> str:
    return _YO_BANG.sub(STRESS, _VOWEL_APOSTROPHE.sub(STRESS, text))


def sample_files() -> list:
    listing = json.loads(cached(TREE, "prose_accents/tree.json").read_text(encoding="utf-8"))
    names = sorted(e["path"].rsplit("/", 1)[1] for e in listing if e["type"] == "file")
    return names[::FILE_STEP]


def load():
    rows = []
    drops = Counter()
    n_in = 0
    for name in sample_files():
        text = cached(FILE.format(name=name), f"prose_accents/{name}").read_text(encoding="utf-8-sig")
        kept = 0
        for paragraph in text.splitlines():
            for sentence in _SENTENCE_END.split(paragraph.strip()):
                if kept >= PER_FILE:
                    break
                if not 4 <= len(sentence.split()) <= 40:
                    continue
                n_in += 1
                row = marked_sentence_row(f"prose:{name}:{n_in}", _to_acute(sentence), drops)
                if row is not None:
                    rows.append(row)
                    kept += 1
    return rows, n_in, drops
