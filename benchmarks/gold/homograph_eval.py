"""inkoziev/HomographResolutionEval (GOLD, CC-BY-4.0).

Source: https://huggingface.co/datasets/inkoziev/HomographResolutionEval.
Original Russian sentences, each with one homograph and its stressed form.
The card guarantees only the homograph's stress, so the homograph token is
the only target.  The model input is ``accentuated_context`` with its marks
removed, so that its tokens align with the marked text.  A homograph that
occurs more than once in a sentence has one stated variant for several
occurrences, so such rows are dropped.
"""
import json
from collections import Counter

from gold._cache import cached
from scoring import Row, Target, clean_word, marked_ordinals, strip_stress

REVISION = "583f2b49f44dfd899d7df2b7d1798865016005d4"
URL = ("https://huggingface.co/datasets/inkoziev/HomographResolutionEval/"
       f"resolve/{REVISION}/HomographResolutionEval.json")


def load():
    path = cached(URL, "homograph_eval/HomographResolutionEval.json")
    records = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    drops = Counter()
    for rec in records:
        homograph = rec["homograph"]
        marks = marked_ordinals(homograph)
        if len(marks) != 1 or marks[0] < 0:
            drops["homograph field has no single acute (rows)"] += 1
            continue
        word = clean_word(homograph)
        tokens = rec["accentuated_context"].split()
        hits = [i for i, tok in enumerate(tokens) if clean_word(tok) == word]
        if not hits:
            drops["homograph not found in the sentence (rows)"] += 1
            continue
        if len(hits) > 1:
            drops["homograph occurs 2+ times (rows)"] += 1
            continue
        in_text = marked_ordinals(tokens[hits[0]])
        if in_text and in_text != marks:
            drops["sentence marks the homograph differently (rows)"] += 1
            continue
        rows.append(Row(f"hre:{rec['id']}", strip_stress(rec["accentuated_context"]),
                        (Target(hits[0], marks[0]),)))
    return rows, len(records), drops
