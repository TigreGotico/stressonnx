"""Stress-mark notation: token, insertion, and format conversions."""
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


def _apply_notation(text: str, notation: str | StressNotation) -> str:
    """Convert *text* from combining-acute to the requested *notation*.

    A no-op when *notation* is ``"diacritic"`` / :attr:`StressNotation.DIACRITIC`.
    """
    from stressonnx import to_plus_notation  # avoid circular at module level
    if notation == StressNotation.PLUS or notation == "plus":
        return to_plus_notation(text)
    return text


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


def _apostrophe_to_diacritic(text: str) -> str:
    """Convert kubataba's apostrophe-after-stressed-vowel output to combining acute.

    The model outputs an apostrophe (') immediately after a stressed vowel.
    We convert that to combining acute (U+0301) placed after the vowel,
    which is the standard stressonnx diacritic notation.
    """
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'" and out and out[-1] in _RU_VOWELS_SET:
            out.append(STRESS_TOKEN)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


_RU_VOWELS_SET = set("аоуыэиеяёюАОУЫЭИЕЯЁЮ")
