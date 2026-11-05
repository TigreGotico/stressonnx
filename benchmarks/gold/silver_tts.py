"""TTS transcripts with machine-made stress marks (SILVER).

* ``stilletto/ruslan-stressed`` (ru, CC-BY-4.0): the RUSLAN corpus
  transcripts, stressed by an LLM (Claude Opus) and checked by automatic
  vowel-position and text-integrity validation.  ``metadata.csv`` holds
  ``filename|text``.  ё is left unmarked by the set's convention, so ё-words
  are not targets.
* ``dmytro-m/uk-tts-stressed-<speaker>`` (uk, MIT): four speakers' transcripts
  stressed from a Ukrainian dictionary and a context model, then checked
  against the recordings; a human reviewed each disagreement.  The labels
  stay machine-made, and the card does not name the context model, so this
  set may favour whichever model produced them.  Only the ``test`` and
  ``validation`` splits are scored.
"""
import json
from collections import Counter

from gold._cache import cached
from scoring import marked_sentence_row

RUSLAN_URL = ("https://huggingface.co/datasets/stilletto/ruslan-stressed/resolve/"
              "ba6686afc84ec0518271ddddc7dca2ceca4d176d/metadata.csv")
UK_TTS_URL = "https://huggingface.co/datasets/dmytro-m/uk-tts-stressed-{speaker}/resolve/{rev}/{split}.jsonl"
#: speaker → pinned dataset revision
UK_TTS_SPEAKERS = {
    "serhiy": "ab992a6a6b3ab096404cf812d76646e0567e8632",
    "mykyta": "9e708f38b22ef930f7fb628c5768ce8c7d58fdfa",
    "oleksa": "4e54247eeeb4a0f56254fa2dfc05a17ae5b6f7e7",
    "tetiana": "aee96b6e1f7263a36d49be3c491f5e681fa7e324",
}
UK_TTS_SPLITS = ("test", "validation")


def load_ruslan():
    path = cached(RUSLAN_URL, "ruslan_stressed/metadata.csv")
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    rows = []
    drops = Counter()
    for line in lines:
        name, text = line.split("|", 1)
        row = marked_sentence_row(f"ruslan:{name}", text, drops)
        if row is not None:
            rows.append(row)
    return rows, len(lines), drops


def load_uk_tts():
    rows = []
    drops = Counter()
    n_in = 0
    for speaker, rev in UK_TTS_SPEAKERS.items():
        for split in UK_TTS_SPLITS:
            path = cached(UK_TTS_URL.format(speaker=speaker, rev=rev, split=split),
                          f"uk_tts_stressed/{speaker}_{split}.jsonl")
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                n_in += 1
                row = marked_sentence_row(f"uk-tts:{rec['id']}", rec["text"], drops)
                if row is not None:
                    rows.append(row)
    return rows, n_in, drops
