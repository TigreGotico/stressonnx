"""Stress-mark notation: token, insertion, and format conversions."""
import unicodedata

from stressonnx.download import LOG
from stressonnx.registry import StressNotation

STRESS_TOKEN = "́"  # combining acute accent — placed AFTER the stressed vowel


def _insert_stress(raw_word: str, idx: int, vowels: str | None = None) -> str:
    """Insert :data:`STRESS_TOKEN` after ``raw_word[idx]``, defensively.

    Dictionaries store character indices; an out-of-range entry degrades to
    "no mark" instead of raising or wrapping around.  When *vowels* is given
    the indexed character must be one of them; pass *None* for curated
    vocabularies that legitimately contain loanwords with vowels outside the
    language's core set (e.g. aze_cyr ``дюнья́``).
    """
    if 0 <= idx < len(raw_word) and (vowels is None or raw_word[idx].lower() in vowels):
        return raw_word[: idx + 1] + STRESS_TOKEN + raw_word[idx + 1:]
    LOG.debug(
        "stress index %d not usable in %r — left unstressed", idx, raw_word
    )
    return raw_word


def _decompose_acute(text: str) -> str:
    """Split precomposed acute-accented characters into base + U+0301.

    Latin stress output (e.g. ``aze_lat``) may reach a consumer NFC-composed
    (``"á"`` instead of ``"a" + U+0301``); only characters whose canonical
    decomposition ends in U+0301 are expanded — everything else (``ё``,
    ``ö``, ``й`` …) is left untouched, so this is NOT a general NFD pass.
    """
    out = []
    for ch in text:
        decomp = unicodedata.normalize("NFD", ch)
        if len(decomp) > 1 and decomp[-1] == STRESS_TOKEN:
            out.append(decomp)
        else:
            out.append(ch)
    return "".join(out)


def to_plus_notation(text: str) -> str:
    """Convert combining-acute stress notation to ``+``-before-vowel.

    ``"приве́т"`` → ``"прив+ет"``

    Useful for models that were trained on ``+``-marked text.  Operates on
    any script — it moves every U+0301 (combining acute, including
    NFC-precomposed forms like ``á``) from after its base character to a
    ``+`` before it.
    """
    text = _decompose_acute(text)
    result = []
    start = 0
    # one letter at a time: a base character and the combining marks after
    # it, so the "+" lands before the base and not before another mark
    for end in range(1, len(text) + 1):
        if end < len(text) and unicodedata.combining(text[end]):
            continue
        letter = text[start:end]
        if STRESS_TOKEN in letter[1:]:
            letter = "+" + letter[0] + letter[1:].replace(STRESS_TOKEN, "")
        result.append(letter)
        start = end
    return "".join(result)


def composed(text: str) -> tuple[str, list[int]]:
    """*text* in NFC, and for each NFC character the offset in *text* of
    the character it came from.

    Each base character is composed with the combining marks that follow
    it, so a vowel in the NFC form maps back to its base character in
    *text*.
    """
    out, origin = [], []
    start = 0
    for end in range(1, len(text) + 1):
        if end < len(text) and unicodedata.combining(text[end]):
            continue
        chunk = unicodedata.normalize("NFC", text[start:end])
        out.append(chunk)
        origin += [start + min(k, end - start - 1) for k in range(len(chunk))]
        start = end
    return "".join(out), origin


def in_form_of(original: str, result: str) -> str:
    """*result* in NFD when *original* is decomposed, else unchanged.

    Backends work on composed text; a caller that sends NFD text gets NFD
    text back.
    """
    if unicodedata.is_normalized("NFC", original):
        return result
    if unicodedata.is_normalized("NFD", original):
        return unicodedata.normalize("NFD", result)
    return result


def render_marks(text: str, mark_offsets, yo_offsets=()) -> str:
    """Render stress marks (and е→ё substitutions) onto *text* by offset.

    Offsets index *text* itself; the combining acute is inserted after each
    marked character and the combining marks that follow it, back to front
    so earlier offsets stay valid.  This is the single place span-native
    backends turn positions into a string.
    """
    chars = list(text)
    for o in yo_offsets:
        if 0 <= o < len(chars) and chars[o] in "еЕ":
            chars[o] = "ё" if chars[o] == "е" else "Ё"
    for o in sorted(mark_offsets, reverse=True):
        if 0 <= o < len(chars):
            end = o + 1
            while end < len(chars) and unicodedata.combining(chars[end]):
                end += 1
            chars.insert(end, STRESS_TOKEN)
    return "".join(chars)


def _apply_notation(text: str, notation: str | StressNotation) -> str:
    """Convert *text* from combining-acute to the requested *notation*.

    Raises :class:`ValueError` for unknown notation values — the same
    contract :class:`stressonnx.Stressor` enforces, so a typo like
    ``notation="pluss"`` fails loudly instead of silently returning
    diacritic output.
    """
    if notation in (StressNotation.PLUS, "plus"):
        return to_plus_notation(text)
    if notation in (StressNotation.DIACRITIC, "diacritic"):
        return text
    raise ValueError(
        f"Unknown notation {notation!r}; expected 'diacritic' or 'plus'."
    )


def _plus_to_diacritic(word: str) -> str:
    """Convert a single word from ``+``-before-vowel to combining-acute notation.

    ``"з+амок"`` → ``"за́мок"``
    """
    result = []
    i = 0
    while i < len(word):
        ch = word[i]
        if ch == "+" and i + 1 < len(word):
            # Insert the vowel then the combining acute
            result.append(word[i + 1])
            result.append(STRESS_TOKEN)
            i += 2
        else:
            result.append(ch)
            i += 1
    return "".join(result)


_RU_VOWELS_SET = set("аоуыэиеяёюАОУЫЭИЕЯЁЮ")
