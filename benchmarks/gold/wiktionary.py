"""English Wiktionary usage examples with stress marks (GOLD, CC-BY-SA).

Source: kaikki.org wiktextract dumps (Tatu Ylonen), one JSON entry per line:
``https://kaikki.org/dictionary/<Language>/kaikki.org-dictionary-<Language>.jsonl``.
Wiktionary editors write the example sentences of Russian, Ukrainian and
Belarusian entries with acute stress marks.  Every polysyllabic token with
exactly one acute is a target; unmarked words are not scored.  Examples of
affix entries and examples with Latin letters (transliterations and glosses
such as ``вдова́ (vdová, "widow")``) are dropped.  The dump is
streamed once and only the distinct example texts are cached.

The stressonnx ``simple`` vocabulary for uk is built from the same
Wiktionary, so ``uk``/``simple`` on this set is partly self-referential.
"""
import json
import re
from collections import Counter

from gold._cache import CACHE, stream_lines
from scoring import STRESS, marked_sentence_row

URL = "https://kaikki.org/dictionary/{name}/kaikki.org-dictionary-{name}.jsonl"
NAMES = {"ru": "Russian", "uk": "Ukrainian", "be": "Belarusian"}
#: A language with fewer usable sentences is not reported.
MIN_SENTENCES = 500

_LATIN = re.compile(r"[A-Za-z]")


def _examples(lang: str) -> list:
    path = CACHE / "wiktionary" / f"{lang}_examples.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    name = NAMES[lang]
    texts = {}
    for line in stream_lines(URL.format(name=name)):
        entry = json.loads(line)
        if entry.get("lang_code") != lang:
            continue
        for sense in entry.get("senses", []):
            for ex in sense.get("examples", []):
                text = (ex.get("text") or "").strip()
                if STRESS in text and text not in texts:
                    texts[text] = entry.get("word", "")
    examples = [{"text": t, "headword": w} for t, w in texts.items()]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(json.dumps(examples, ensure_ascii=False), encoding="utf-8")
    tmp.rename(path)
    return examples


def load(lang: str):
    examples = _examples(lang)
    rows = []
    drops = Counter()
    for n, ex in enumerate(examples):
        if ex["headword"].startswith("-") or ex["headword"].endswith("-"):
            drops["example of an affix entry (rows)"] += 1
            continue
        if _LATIN.search(ex["text"]):
            drops["Latin letters: transliteration or gloss (rows)"] += 1
            continue
        row = marked_sentence_row(f"wikt-{lang}:{n}", ex["text"], drops)
        if row is not None:
            rows.append(row)
    return rows, len(examples), drops
