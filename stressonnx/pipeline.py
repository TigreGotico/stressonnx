"""StressPipeline: an explicit, self-contained stressing engine.

The module-level convenience API (:func:`stressonnx.stress`,
:func:`stressonnx.warm_up`) delegates to one shared default pipeline; servers
that want scoped caches, their own failure policy, or disposal semantics
construct their own ``StressPipeline``.

Beyond the marked-string API, :meth:`StressPipeline.analyze` returns
**structured results**: per-word spans with offsets into the *original,
untouched* input — no parsing marks back out of a mutated string.
"""
import difflib
import logging
import threading
import time
from dataclasses import dataclass

from stressonnx.errors import ModelDownloadError, ModelLoadError
from stressonnx.langs import resolve_lang
from stressonnx.notation import STRESS_TOKEN, _apply_notation, render_marks
from stressonnx.registry import ALL_LANGS, DEFAULT_MODEL, LANGUAGES, MODEL_REGISTRY
from stressonnx.stressor import make_stressor

LOG = logging.getLogger("stressonnx")

#: Model priority for ``fallback=True`` and ``prefer="best"``: quality first.
FALLBACK_PRIORITY = ("ruaccent", "silero", "simple")

#: ``prefer=`` strategies.  ``"best"`` orders by each language's measured
#: accuracy (the ``accuracy`` field of its data file) where numbers exist,
#: falling back to the quality-first default; ``"fast"`` and ``"smallest"``
#: order by latency and footprint.
_PREFER_ORDERS = {
    "fast": ("silero", "simple", "ruaccent"),
    "smallest": ("simple", "silero", "ruaccent"),
}


def _best_order(lang: str):
    measured = LANGUAGES.get(lang, {}).get("accuracy")
    if measured:
        return tuple(sorted(
            FALLBACK_PRIORITY,
            key=lambda m: measured.get(m, -1.0),
            reverse=True,
        ))
    return FALLBACK_PRIORITY


@dataclass(frozen=True)
class StressedWord:
    """One word of the input, with its stress located by offsets.

    ``start``/``end`` slice the **original input** (``text ==
    original[start:end]``); ``stressed_index`` is the offset *within the
    word* of the stressed vowel, or ``None`` when the word carries no mark.
    ``yo_restored`` is True when the backend rewrote е→ё inside this word.
    """

    text: str
    start: int
    end: int
    stressed_index: int | None
    yo_restored: bool = False

    @property
    def stressed_char(self) -> str | None:
        return None if self.stressed_index is None else self.text[self.stressed_index]


@dataclass(frozen=True)
class StressResult:
    """Structured output of :meth:`StressPipeline.analyze`.

    ``original`` is the input exactly as given; ``marked`` is the diacritic
    rendering; ``words`` covers every whitespace-delimited token of the
    original, marked or not.
    """

    original: str
    marked: str
    words: tuple

    @property
    def text(self) -> str:
        return self.marked


def _align_marks(original: str, marked: str):
    """Map each U+0301 in *marked* to a character offset in *original*.

    Backends only insert marks and substitute е↔ё — except ruaccent, which
    may also normalize (drop symbols, collapse whitespace).  A sequence
    alignment on yo-neutralized copies absorbs all of that; marks whose
    base character cannot be aligned are dropped rather than guessed.
    Returns ``(mark_offsets, yo_offsets)`` into *original*.
    """
    stripped = []          # marked, minus stress marks
    mark_positions = []    # indices into `stripped` of each mark's base char
    for ch in marked:
        if ch == STRESS_TOKEN:
            if stripped:
                mark_positions.append(len(stripped) - 1)
        else:
            stripped.append(ch)
    stripped = "".join(stripped)

    def neutral(s: str) -> str:
        return s.replace("ё", "е").replace("Ё", "Е")

    to_original = {}
    matcher = difflib.SequenceMatcher(None, neutral(stripped), neutral(original), autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                to_original[i1 + k] = j1 + k

    marks = sorted({to_original[p] for p in mark_positions if p in to_original})
    yo = sorted(
        to_original[i]
        for i in range(len(stripped))
        if i in to_original
        and stripped[i] in "ёЁ"
        and original[to_original[i]] in "еЕ"
    )
    return marks, yo


class StressPipeline:
    """A stressing engine with its own backend cache and failure policy."""

    def __init__(self, failure_cooldown_s: float = 30.0) -> None:
        self._singletons: dict = {}
        self._failures: dict = {}
        self._cooldown = failure_cooldown_s
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _resolve(self, lang: str, model: str | None, prefer: str | None):
        lang = resolve_lang(lang, ALL_LANGS)
        if model is None and prefer is not None:
            if prefer == "best":
                order = _best_order(lang)
            elif prefer in _PREFER_ORDERS:
                order = _PREFER_ORDERS[prefer]
            else:
                raise ValueError(
                    f"Unknown prefer={prefer!r}; expected one of "
                    f"{sorted(_PREFER_ORDERS) + ['best']}."
                )
            model = next(
                (m for m in order if lang in MODEL_REGISTRY[m].langs), None
            )
        resolved = model if model is not None else DEFAULT_MODEL.get(lang)
        return lang, model, resolved

    def _chain(self, lang: str) -> list:
        return [
            (m, lang) for m in FALLBACK_PRIORITY if lang in MODEL_REGISTRY[m].langs
        ]

    def _backend(self, lang: str, model: str | None, resolved: str | None):
        key = (lang, resolved)
        with self._lock:
            if key not in self._singletons:
                self._singletons[key] = make_stressor(model=model, lang=lang)
            return key, self._singletons[key]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def _execute(self, text: str, lang: str, model: str | None,
                 fallback: bool, prefer: str | None, runner):
        """Run *runner(backend, text)* against the selected model, walking
        the fallback chain on download/load failures and honoring the
        failure cooldown."""
        lang, model, resolved = self._resolve(lang, model, prefer)

        attempts = [(model, lang)]
        if fallback:
            attempts += [p for p in self._chain(lang) if p[0] != model]
            if model is None and len(attempts) > 1:
                attempts = attempts[1:]

        last_error = None
        now = time.monotonic()
        for try_model, try_lang in attempts:
            try_resolved = try_model if try_model is not None else DEFAULT_MODEL.get(try_lang)
            key = (try_lang, try_resolved)

            recent = self._failures.get(key)
            if recent is not None and now - recent[0] < self._cooldown:
                last_error = recent[1]
                if not fallback:
                    raise last_error
                continue

            key, backend = self._backend(try_lang, try_model, try_resolved)
            try:
                result = runner(backend, text)
            except (ModelDownloadError, ModelLoadError) as exc:
                self._failures[key] = (time.monotonic(), exc)
                if not fallback:
                    raise
                last_error = exc
                LOG.warning(
                    "Model %r unavailable for lang %r (%s) — falling back.",
                    try_model, try_lang, exc,
                )
                continue
            self._failures.pop(key, None)
            return result
        raise last_error

    def stress(self, text: str, lang: str = "ru", model: str | None = None,
               notation: str = "diacritic", fallback: bool = False,
               prefer: str | None = None) -> str:
        """Insert stress marks into *text* (see :func:`stressonnx.stress`)."""
        result = self._execute(text, lang, model, fallback, prefer,
                               lambda backend, t: backend(t))
        return _apply_notation(result, notation)

    def stress_batch(self, texts, lang: str = "ru", model: str | None = None,
                     notation: str = "diacritic", fallback: bool = False,
                     prefer: str | None = None) -> list:
        """Stress a sequence of texts with one backend resolution."""
        return [
            self.stress(t, lang, model=model, notation=notation,
                        fallback=fallback, prefer=prefer)
            for t in texts
        ]

    def analyze(self, text: str, lang: str = "ru", model: str | None = None,
                fallback: bool = False, prefer: str | None = None) -> StressResult:
        """Structured stressing: spans with offsets into the original input.

        Words are the whitespace-delimited tokens of *text*; each carries
        the offset of its stressed vowel (or ``None``) and whether е→ё was
        restored inside it.  Offsets refer to the input exactly as the
        caller passed it.
        """
        def runner(backend, t):
            if hasattr(backend, "mark_offsets"):
                marks, yo = backend.mark_offsets(t)
                return marks, yo, render_marks(t, marks, yo)
            marked = backend(t)
            marks, yo = _align_marks(t, marked)
            return marks, yo, marked

        mark_offsets, yo_offsets, marked = self._execute(
            text, lang, model, fallback, prefer, runner
        )
        mark_set, yo_set = set(mark_offsets), set(yo_offsets)

        words = []
        pos = 0
        for token in text.split():
            start = text.index(token, pos)
            end = start + len(token)
            pos = end
            stressed = next((o - start for o in mark_offsets if start <= o < end), None)
            words.append(StressedWord(
                text=token, start=start, end=end,
                stressed_index=stressed,
                yo_restored=any(start <= o < end for o in yo_set),
            ))
        return StressResult(original=text, marked=marked, words=tuple(words))

    def warm_up(self, lang: str, model: str | None = None) -> None:
        """Download and load the model for *lang* ahead of the first request."""
        lang, model, resolved = self._resolve(lang, model, None)
        _key, backend = self._backend(lang, model, resolved)
        ensure = getattr(backend, "_ensure_loaded", None)
        if ensure is not None:
            ensure()


#: The pipeline behind the module-level convenience API.
DEFAULT_PIPELINE = StressPipeline()
