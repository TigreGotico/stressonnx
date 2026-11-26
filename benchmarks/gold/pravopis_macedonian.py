"""Macedonian words with the stress that the Pravopis marks (GOLD).

Source: Институт за македонски јазик „Крсте Мисирков“ (2017), Правопис на
македонскиот јазик, 2nd ed., Skopje.  The Pravopis marks the stress with an
apostrophe before the stressed vowel (§25, п'ортр'ет), in its stress rules and
in the word lists where a word does not take antepenult stress.
``data/mk_pravopis.json`` holds only the word, the 0-based ordinal of the
stressed vowel and the printed page; no text is copied.  ``build`` makes the
file again from the PDF with ``pdftotext``.

A word is left out when it has two marks, when the mark is on the syllabic р,
when it has one vowel letter, or when two places give it different stresses.
"""
import json
import re
import subprocess
import unicodedata
from collections import Counter
from pathlib import Path

from scoring import Row, Target

from gold._cache import cached

URL = ("http://web.archive.org/web/20260818065020/"
       "https://www.pravopis.mk/sites/default/files/Pravopis-2017.PDF")
DATA = Path(__file__).with_name("data") / "mk_pravopis.json"
VOWELS = frozenset("аеиоу")
LETTER = "а-шѓќљњџѕј"
#: A token next to a hyphen is part of a compound such as ког'о-годе.
MARKED = re.compile(rf"(?<![{LETTER}'\-‑])[{LETTER}']*'[{LETTER}']+(?![{LETTER}'\-‑])")
#: The text layer prints д as the Latin script g in many places.
GLYPHS = str.maketrans({"ɡ": "д"})


def stress_ordinal(marked: str):
    """(plain word, vowel ordinal) or (None, drop reason)."""
    if marked.count("'") != 1:
        return None, "more than one mark (words)"
    head, tail = marked.split("'")
    if tail[0] not in VOWELS:
        return None, "mark on the syllabic р or a consonant (words)"
    word = head + tail
    if sum(c in VOWELS for c in word) < 2:
        return None, "one vowel letter (words)"
    return word, sum(c in VOWELS for c in head)


def build():
    pdf = cached(URL, "Pravopis-2017.PDF")
    text = subprocess.run(["pdftotext", "-layout", str(pdf), "-"], check=True,
                          capture_output=True, text=True).stdout
    found, drops, n_in = {}, Counter(), 0
    for pdf_page, page in enumerate(text.split("\f"), start=1):
        page = unicodedata.normalize("NFC", page.translate(GLYPHS)).lower()
        page = re.sub("\u00ad\\s*", "", page)
        for marked in MARKED.findall(page):
            n_in += 1
            word, ordinal = stress_ordinal(marked)
            if word is None:
                drops[ordinal] += 1
            else:
                found.setdefault(word, {}).setdefault(ordinal, pdf_page - 10)
    words = []
    for word, ordinals in sorted(found.items()):
        if len(ordinals) > 1:
            drops["word with two stresses (words)"] += 1
        else:
            (ordinal, page), = ordinals.items()
            words.append({"word": word, "stressed_vowel_index": ordinal, "page": page})
    DATA.write_text(json.dumps({"n_in": n_in, "drops": drops, "words": words},
                               ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    print(f"{n_in} marked tokens in, {len(words)} words out, drops {dict(drops)}")


def load():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    rows = [Row(f"pravopis-mk:{w['word']}", w["word"], (Target(0, w["stressed_vowel_index"]),))
            for w in data["words"]]
    return rows, data["n_in"], Counter(data["drops"])


if __name__ == "__main__":
    build()
