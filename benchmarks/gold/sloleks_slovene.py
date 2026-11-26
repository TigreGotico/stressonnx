"""Slovene lemmas with the stress of the Sloleks 3.0 lexicon (GOLD).

Source: Čibej et al. (2022), Morphological lexicon Sloleks 3.0, CLARIN.SI,
http://hdl.handle.net/11356/1745, CC BY-SA 4.0.  The README says that the
accentuated forms of the 100,000 entries with status MANUAL "were manually
corrected", and that the forms of the other entries are fully automatic.
Only MANUAL entries are read, and only their accentuation forms: the IPA and
SAMPA forms are automatic in every entry.

Each row is the lemma form of one single-word entry.  A lemma is left out
when its accentuation form has no mark or more than one, when the mark is on
the syllabic r (ŕ), when it has one vowel letter, or when two entries give
the same lemma different stresses.  The extracted rows are cached as JSON
next to the zip.
"""
import json
import unicodedata
import zipfile
from collections import Counter
from xml.etree.ElementTree import iterparse

from scoring import Row, Target

from gold._cache import CACHE, cached

URL = "https://www.clarin.si/repository/xmlui/bitstream/11356/1745/1/Sloleks.3.0.zip"
VOWELS = frozenset("aeiou")
#: Acute, grave and circumflex: the three accent marks of Sloleks forms.
MARKS = {"́", "̀", "̂"}


def stress_ordinal(accented: str):
    """(plain word, vowel ordinal of the one mark) or (None, drop reason)."""
    plain, ordinal, seen, marks = [], None, 0, 0
    for c in unicodedata.normalize("NFD", accented.lower()):
        if c in MARKS:
            marks += 1
            if not plain or plain[-1] not in VOWELS:
                return None, "mark on the syllabic r or a consonant (rows)"
            ordinal = seen - 1
        else:
            plain.append(c)
            seen += c in VOWELS
    if marks != 1:
        return None, "no mark or more than one mark (rows)"
    if seen < 2:
        return None, "one vowel letter (rows)"
    return unicodedata.normalize("NFC", "".join(plain)), ordinal


def lemmas():
    """{lemma: set of ordinals} and drop counts over the MANUAL entries."""
    out, drops, n_in = {}, Counter(), 0
    with zipfile.ZipFile(cached(URL, "Sloleks.3.0.zip")) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".xml")):
            with zf.open(name) as fh:
                for _, el in iterparse(fh):
                    if el.tag != "entry":
                        continue
                    n_in += 1
                    head = el.find("head")
                    unit = head.find("lexicalUnit")
                    if head.findtext("status") != "MANUAL":
                        drops["entry status AUTOMATIC (rows)"] += 1
                    elif unit.get("type") != "single":
                        drops["multi-word entry (rows)"] += 1
                    else:
                        lemma = head.findtext("headword/lemma")
                        accented = [
                            a.text for wf in el.iterfind("body/wordFormList/wordForm")
                            if wf.findtext("formRepresentations/orthographyList/orthography/form") == lemma
                            for a in wf.iterfind(
                                "formRepresentations/accentuationList[@type='dynamic']/accentuation/form")
                        ]
                        if not accented:
                            drops["no lemma form with an accentuation (rows)"] += 1
                        for form in set(accented):
                            word, ordinal = stress_ordinal(form)
                            if word is None:
                                drops[ordinal] += 1
                            elif word != lemma.lower():
                                drops["accentuation letters differ from the lemma (rows)"] += 1
                            else:
                                out.setdefault(word, set()).add(ordinal)
                    el.clear()
    return out, n_in, drops


def load():
    path = CACHE / "sloleks_3.0_manual_lemmas.json"
    if not path.exists():
        found, n_in, drops = lemmas()
        words = {}
        for word, ordinals in sorted(found.items()):
            if len(ordinals) == 1:
                words[word] = ordinals.pop()
            else:
                drops["lemma with two stresses (rows)"] += 1
        path.write_text(json.dumps({"n_in": n_in, "drops": drops, "words": words},
                                   ensure_ascii=False), encoding="utf-8")
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [Row(f"sloleks-sl:{word}", word, (Target(0, ordinal),))
            for word, ordinal in data["words"].items()]
    return rows, data["n_in"], Counter(data["drops"])
