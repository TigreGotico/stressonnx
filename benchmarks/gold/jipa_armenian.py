"""Armenian citation forms from a JIPA Illustration of the IPA (GOLD, CC-BY-4.0).

Source: Seyfarth, Dolatian, Guekguezian et al. 2024, "Armenian (Yerevan
Eastern Armenian and Beirut Western Armenian)", JIPA 54(1),
doi:10.1017/S0025100323000130.  The authors transcribe each word in IPA with
the stress mark ˈ.  ``data/hy_jipa.json`` holds the polysyllabic Yerevan
forms in reformed spelling, with U+0301 after the stressed vowel letter and
the PDF page of each word.  Beirut forms are left out: for some words the
two dialects stress different syllables.  Each word is one row.
"""
import json
from collections import Counter
from pathlib import Path

from scoring import marked_sentence_row

DATA = Path(__file__).with_name("data") / "hy_jipa.json"


def load():
    words = json.loads(DATA.read_text(encoding="utf-8"))["words"]
    drops = Counter()
    rows = [row for n, w in enumerate(words)
            if (row := marked_sentence_row(f"jipa-hy:{n}", w["word"], drops)) is not None]
    return rows, len(words), drops
