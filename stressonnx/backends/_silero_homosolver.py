"""Russian homograph resolver of silero_stress, ported to onnxruntime.

Each word of the homograph dictionary gets one of its two stressed variants.
A phrase regex picks the variant when the context matches a known phrase;
otherwise a BERT classifier reads the word in a character window of cleaned
context, marked ``[HOMO] word [/HOMO]``.  Every homograph of a text is padded
with ``[PAD]`` to the longest row of the text and run without an attention
mask, exactly as upstream does — padding changes the logits, so the padded
width is part of the result.
"""
import gzip
import json
import re
import unicodedata

import numpy as np
import onnxruntime as ort

_RE_HOMO_CANDIDATE = re.compile(r"(?=.*[а-яё])[а-яё+]+", re.IGNORECASE)
_RE_REMOVE_EXTRA = re.compile(r"[^a-zA-Zа-яА-ЯёЁ0-9\s.!?,\-]")
_RE_SPACES = re.compile(r"\s+")
_RE_DOUBLE_DASH = re.compile(r"-{2,}")
_RE_REPEAT_PUNCT = re.compile(r"([.!?])\1+")
_RE_REPEAT_COMMA = re.compile(r",{2,}")
_RE_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,!?])")
_RE_PUNCT_ADD_SPACE = re.compile(r"([.,!?])(?=\S)")

HOMO_START, HOMO_END = "[HOMO]", "[/HOMO]"

_RE_WORD_CHAR = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9]")

#: Raw characters cleaned on each side of a homograph on the first try
_CONTEXT_SPAN = 1024

#: Rows per ONNX run.  Every chunk is padded to the width of the whole batch,
#: so a row's logits do not depend on the chunking.
_CHUNK_ROWS = 32


def _clean_context(text: str, is_start: bool) -> str:
    if not text:
        return ""
    text = _RE_REMOVE_EXTRA.sub("", text)
    text = _RE_SPACES.sub(" ", text)
    text = _RE_DOUBLE_DASH.sub(" - ", text)
    text = _RE_REPEAT_PUNCT.sub(r"\1", text)
    text = _RE_REPEAT_COMMA.sub(",", text)
    text = _RE_SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _RE_REPEAT_PUNCT.sub(r"\1", text)
    text = _RE_REPEAT_COMMA.sub(",", text)
    text = _RE_PUNCT_ADD_SPACE.sub(r"\1 ", text)
    text = _RE_SPACES.sub(" ", text).strip()
    if is_start:
        text = text.lstrip(" .,!?-")
        if text:
            text = text[0].upper() + text[1:].lower()
    else:
        if text:
            text = text[0] + text[1:].lower()
        if text and text[-1] not in ".!?":
            text += "."
    return text


def _context_before(text: str, start: int, size: int) -> str:
    """``_clean_context(text[:start], is_start=True)[-size:]``, cleaning a tail only.

    Every cleaning rule acts on runs of whitespace and punctuation, and the
    case rule changes only the first character.  So a cleaned tail of the
    prefix agrees with the cleaned whole prefix from the first character
    after its first letter or digit.  The tail grows until that part holds
    *size* characters.
    """
    span = _CONTEXT_SPAN
    while True:
        cut = max(0, start - span)
        cleaned = _clean_context(text[cut:start], is_start=True)
        if cut == 0:
            return cleaned[-size:]
        first = _RE_WORD_CHAR.search(cleaned)
        if first and len(cleaned) - first.start() > size:
            return cleaned[-size:]
        span *= 2


def _context_after(text: str, end: int, size: int) -> str:
    """``_clean_context(text[end:], is_start=False)[:size]``, cleaning a head only.

    The cleaned head agrees with the cleaned whole suffix up to its last
    letter or digit; the head grows until that part holds *size*
    characters.
    """
    span = _CONTEXT_SPAN
    while True:
        stop = end + span
        cleaned = _clean_context(text[end:stop], is_start=False)
        if stop >= len(text):
            return cleaned[:size]
        last = None
        for last in _RE_WORD_CHAR.finditer(cleaned):
            pass
        if last and last.start() >= size:
            return cleaned[:size]
        span *= 2


def _is_punctuation(char: str) -> bool:
    cp = ord(char)
    if 33 <= cp <= 47 or 58 <= cp <= 64 or 91 <= cp <= 96 or 123 <= cp <= 126:
        return True
    return unicodedata.category(char).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
            or 0x20000 <= cp <= 0x2A6DF or 0x2A700 <= cp <= 0x2B73F
            or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
            or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F)


class WordPiece:
    """Cased BERT tokenizer: basic split, then greedy longest-match WordPiece."""

    def __init__(self, vocab: list, never_split=(HOMO_START, HOMO_END)) -> None:
        self.ids = {token: i for i, token in enumerate(vocab)}
        self.never_split = set(never_split)
        self.unk_id = self.ids["[UNK]"]
        self.cls_id = self.ids["[CLS]"]
        self.sep_id = self.ids["[SEP]"]

    def _basic(self, text: str) -> list:
        chars = []
        for c in text:
            cp = ord(c)
            if cp == 0 or cp == 0xFFFD:
                continue
            if c in " \t\n\r" or unicodedata.category(c) == "Zs":
                chars.append(" ")
            elif unicodedata.category(c).startswith("C"):
                continue
            elif _is_cjk(cp):
                chars.append(f" {c} ")
            else:
                chars.append(c)
        tokens = []
        for token in "".join(chars).split():
            if token in self.never_split:
                tokens.append(token)
                continue
            piece = []
            for c in token:
                if _is_punctuation(c):
                    if piece:
                        tokens.append("".join(piece))
                        piece = []
                    tokens.append(c)
                else:
                    piece.append(c)
            if piece:
                tokens.append("".join(piece))
        return tokens

    def _wordpiece(self, token: str) -> list:
        if len(token) > 100:
            return [self.unk_id]
        out, start = [], 0
        while start < len(token):
            end = len(token)
            while start < end:
                sub = token[start:end] if start == 0 else "##" + token[start:end]
                if sub in self.ids:
                    out.append(self.ids[sub])
                    break
                end -= 1
            else:
                return [self.unk_id]
            start = end
        return out

    def encode(self, text: str) -> list:
        ids = [self.cls_id]
        for token in self._basic(text):
            ids.extend(self._wordpiece(token))
        ids.append(self.sep_id)
        return ids


def _variant_from_group(group: str) -> str | None:
    """Phrase-regex group name ``зАмок`` → variant ``з+амок``."""
    for i, c in enumerate(group):
        if c.isupper():
            return group[:i] + "+" + group[i:].lower()
    return None


class HomoSolver:
    """Pick the stressed variant of each dictionary homograph in a text."""

    def __init__(self, files: dict, config: dict) -> None:
        with gzip.open(files["homosolver_vocab.txt.gz"], "rt", encoding="utf-8") as fh:
            vocab = fh.read().split("\n")[:-1]
        with gzip.open(files["homodict.json.gz"], "rt", encoding="utf-8") as fh:
            self._variants = {w: sorted(v) for w, v in json.load(fh).items()}
        with gzip.open(files["homophrases.json.gz"], "rt", encoding="utf-8") as fh:
            self._phrase_sources = json.load(fh)
        self._phrases: dict = {}
        self._tok = WordPiece(vocab)
        self._pad_id = config["pad_id"]
        self._start_id = config["homo_start_id"]
        self._end_id = config["homo_end_id"]
        self._half_window = config["window"] // 2
        self._sess = ort.InferenceSession(
            files["homosolver.onnx"], providers=["CPUExecutionProvider"]
        )

    def _phrase_pattern(self, word: str):
        if word not in self._phrases:
            self._phrases[word] = re.compile(self._phrase_sources[word], re.IGNORECASE)
        return self._phrases[word]

    def resolve(self, text: str) -> list:
        """``(start, variant)`` per resolved homograph of *text*.

        *variant* is lowercase, as long as the word, and carries one ``+``
        before the stressed vowel (``вс+ё``).  A word that sits against a
        combining mark is part of a pre-stressed word and is left alone.
        """
        found, batch = [], []
        for match in _RE_HOMO_CANDIDATE.finditer(text):
            start, end = match.span()
            word = match.group().lower()
            if word not in self._variants and word not in self._phrase_sources:
                continue
            if (end < len(text) and unicodedata.combining(text[end])) or (
                start > 0 and unicodedata.combining(text[start - 1])
            ):
                continue
            before = _context_before(text, start, self._half_window)
            after = _context_after(text, end, self._half_window)
            marked = f"{before} {HOMO_START} {word} {HOMO_END} {after}".strip()
            variant = None
            if word in self._phrase_sources:
                hit = self._phrase_pattern(word).search(marked)
                if hit:
                    group = next(g for g, v in hit.groupdict().items() if v is not None)
                    variant = _variant_from_group(group)
            if variant is None and word not in self._variants:
                continue
            found.append((start, word, variant))
            if variant is None:
                batch.append(self._tok.encode(marked))

        if batch:
            picks = iter(self._classify(batch))
            found = [(s, w, v if v is not None else self._variants[w][next(picks)])
                     for s, w, v in found]
        return [(s, v) for s, _, v in found]

    def _classify(self, rows: list) -> list:
        width = max(len(r) for r in rows)
        picks = []
        for first in range(0, len(rows), _CHUNK_ROWS):
            chunk = rows[first:first + _CHUNK_ROWS]
            ids = np.full((len(chunk), width), self._pad_id, dtype=np.int64)
            for i, r in enumerate(chunk):
                ids[i, : len(r)] = r
            starts = np.array([r.index(self._start_id) for r in chunk], dtype=np.int64)
            ends = np.array([r.index(self._end_id) for r in chunk], dtype=np.int64)
            (logits,) = self._sess.run(
                None, {"input_ids": ids, "homo_start": starts, "homo_end": ends}
            )
            probs = 1.0 / (1.0 + np.exp(-logits[:, 0].astype(np.float32)))
            picks.extend(int(p) for p in np.round(probs))
        return picks
