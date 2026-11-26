"""Bulgarian lemmas with the stress of the BAS dictionary (GOLD).

Source: Речник на българския език, Институт за български език (БАН),
https://ibl.bas.bg/rbe/.  The dictionary marks the stressed vowel of each
headword with a grave accent.  ``data/bg_rbe.json`` holds only the lemma, the
0-based ordinal of that vowel and the entry URL; no entry text is copied.
Lemmas with two stresses in the dictionary are left out.  Each lemma is one
row, scored with ъ as a vowel.
"""
import json
from collections import Counter
from pathlib import Path

from scoring import Row, Target

DATA = Path(__file__).with_name("data") / "bg_rbe.json"


def load():
    words = json.loads(DATA.read_text(encoding="utf-8"))["words"]
    rows = [Row(f"rbe-bg:{n}", w["word"], (Target(0, w["stressed_vowel_index"]),))
            for n, w in enumerate(words)]
    return rows, len(words), Counter()
