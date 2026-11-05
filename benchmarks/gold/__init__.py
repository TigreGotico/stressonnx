"""Evaluation sets: loaders and provenance, never the data itself.

Every set carries a label tier.  GOLD labels were written by people.  SILVER
labels were made by a machine (an LLM, a dictionary tool, a context model) or
by a method the source does not document.  Silver sets may score models, but
a claim that one model is best rests on gold sets only.

A loader returns ``(rows, n_in, drops)``: the scorable rows, the number of
source items read, and a Counter of drop reasons whose labels state their
unit (rows or targets).
"""
from dataclasses import dataclass
from typing import Callable

from gold import belarusian_homographs, homograph_eval, prose_accents, russtress, silver_tts, wiktionary

GOLD = "GOLD"
SILVER = "SILVER"


@dataclass(frozen=True)
class EvalSet:
    name: str
    lang: str
    tier: str
    license: str
    source: str
    scored: str
    load: Callable
    homographs_only: bool = False
    note: str = ""


SETS = [
    EvalSet("russtress", "ru", GOLD, "none stated",
            "https://github.com/MashaPo/russtress", "middle word of each window",
            lambda: russtress.load("ru")),
    EvalSet("homograph-eval", "ru", SILVER, "CC-BY-4.0",
            "https://huggingface.co/datasets/inkoziev/HomographResolutionEval",
            "the homograph only", homograph_eval.load, homographs_only=True),
    EvalSet("wiktionary", "ru", GOLD, "CC-BY-SA",
            "https://kaikki.org/dictionary/Russian/", "marked polysyllabic words",
            lambda: wiktionary.load("ru")),
    EvalSet("ruslan-stressed", "ru", SILVER, "CC-BY-4.0",
            "https://huggingface.co/datasets/stilletto/ruslan-stressed",
            "marked polysyllabic words (LLM labels)", silver_tts.load_ruslan,
            note="ё is left unmarked by the set's convention, so words stressed on ё are not targets."),
    EvalSet("prose-accents", "ru", SILVER, "MIT",
            "https://huggingface.co/datasets/nevmenandr/accentual-syllabic-verse-in-russian-prose",
            "marked polysyllabic words (ru-accent-poet labels, fixed sample)",
            prose_accents.load,
            note=f"Sample: every {prose_accents.FILE_STEP}th file of the sorted list, the first "
                 f"{prose_accents.PER_FILE} sentences of 4 to 40 tokens per file."),
    EvalSet("russtress", "uk", GOLD, "none stated",
            "https://github.com/MashaPo/russtress", "middle word of each window",
            lambda: russtress.load("uk")),
    EvalSet("wiktionary", "uk", GOLD, "CC-BY-SA",
            "https://kaikki.org/dictionary/Ukrainian/", "marked polysyllabic words",
            lambda: wiktionary.load("uk"),
            note="The simple-uk vocabulary is built from the same Wiktionary, so the simple "
                 "score on this set is partly self-referential."),
    EvalSet("uk-tts-stressed", "uk", SILVER, "MIT",
            "https://huggingface.co/datasets/dmytro-m/uk-tts-stressed-serhiy (and -mykyta, -oleksa, -tetiana)",
            "marked polysyllabic words (dictionary + context-model labels)",
            silver_tts.load_uk_tts,
            note="Test and validation splits of four speakers.  The card does not name the "
                 "context model that made the labels."),
    EvalSet("russtress", "be", GOLD, "none stated",
            "https://github.com/MashaPo/russtress", "middle word of each window",
            lambda: russtress.load("be")),
    EvalSet("wiktionary", "be", GOLD, "CC-BY-SA",
            "https://kaikki.org/dictionary/Belarusian/", "marked polysyllabic words",
            lambda: wiktionary.load("be")),
    EvalSet("zascienak", "be", GOLD, "CC-BY-SA-4.0",
            "https://huggingface.co/datasets/alex73/benchmarks-stress-bel",
            "marked polysyllabic words (fully hand-marked novel)",
            lambda: belarusian_homographs.load_sentences("zascienak_Malinauka.txt")),
    EvalSet("homographs-10x10", "be", GOLD, "CC-BY-SA-4.0",
            "https://huggingface.co/datasets/alex73/benchmarks-stress-bel",
            "the homograph only (expert-written, balanced)", belarusian_homographs.load_10x10,
            homographs_only=True),
    EvalSet("homographs-commonvoice", "be", SILVER, "CC-BY-SA-4.0",
            "https://huggingface.co/datasets/alex73/benchmarks-stress-bel",
            "the marked homographs (labelling method undocumented)",
            lambda: belarusian_homographs.load_sentences("common_voice.txt"), homographs_only=True),
]


def sets_for(lang: str, names=None) -> list:
    return [s for s in SETS if s.lang == lang and (not names or s.name in names)]
