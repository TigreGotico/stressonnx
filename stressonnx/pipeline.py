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
import unicodedata
from dataclasses import dataclass

from stressonnx.errors import ModelDownloadError, ModelLoadError
from stressonnx.langs import resolve_lang
from stressonnx.notation import STRESS_TOKEN, _apply_notation, in_form_of, is_stress_base, render_marks
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


#: Ceiling of the retry delay after repeated :class:`ModelLoadError`s, in
#: seconds.  Each consecutive load failure of a (lang, model) pair doubles
#: the delay, starting from the pipeline's ``failure_cooldown_s``.
MAX_LOAD_RETRY_S = 600.0


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
class _Failure:
    """A remembered (lang, model) failure and when to try the pair again."""

    error: Exception
    traceback: object
    retry_at: float
    delay: float


@dataclass(frozen=True)
class StressedWord:
    """One word of the input, with its stress located by offsets.

    ``start``/``end`` slice the **original input** (``text ==
    original[start:end]``).  ``stressed_indices`` holds the offset *within
    the word* of every stressed vowel, in order; a hyphenated compound such
    as ``се́веро-за́падный`` carries one mark per part.  ``stressed_index``
    is the first of them, or ``None`` when the word carries no mark; a value
    that contradicts ``stressed_indices`` raises :class:`ValueError`.  A word
    unpickled from a version without ``stressed_indices`` gets its single
    ``stressed_index`` there.
    ``yo_restored`` is True when the backend rewrote е→ё inside this word.
    """

    text: str
    start: int
    end: int
    stressed_index: int | None
    yo_restored: bool = False
    stressed_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.stressed_indices:
            if self.stressed_index is not None:
                object.__setattr__(self, "stressed_indices", (self.stressed_index,))
        elif self.stressed_index != self.stressed_indices[0]:
            raise ValueError(
                f"stressed_index {self.stressed_index!r} must be the first of "
                f"stressed_indices {self.stressed_indices!r}"
            )

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self.__post_init__()

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
    A U+0301 after a consonant is part of that letter (``ќ``, ``ś`` in
    NFD) and aligns as an ordinary character.
    Returns ``(mark_offsets, yo_offsets)`` into *original*.
    """
    stripped = []          # marked, minus stress marks
    mark_positions = []    # indices into `stripped` of each mark's base char
    base, base_pos = "", -1
    for ch in marked:
        if ch == STRESS_TOKEN and is_stress_base(base):
            mark_positions.append(base_pos)
        else:
            if not unicodedata.combining(ch):
                base, base_pos = ch, len(stripped)
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
        the fallback chain on download/load failures.

        A failed (lang, model) pair is remembered and raised again, with the
        traceback of the original failure, until its retry time.  A download
        failure is retried after the cooldown.  A load failure is retried
        after a delay that doubles with each consecutive load failure, up to
        :data:`MAX_LOAD_RETRY_S`, because a reload rebuilds the model and a
        corrupt file stays corrupt until someone repairs it.  Success clears
        the record.  A record is ignored once the backend reports loaded,
        so a failure stored after a concurrent :meth:`warm_up` cannot
        shadow the loaded model.
        """
        lang, model, resolved = self._resolve(lang, model, prefer)

        attempts = [(model, lang)]
        if fallback:
            attempts += [p for p in self._chain(lang) if p[0] != model]
            if model is None and len(attempts) > 1:
                attempts = attempts[1:]

        last_error = last_tb = None
        now = time.monotonic()
        for try_model, try_lang in attempts:
            try_resolved = try_model if try_model is not None else DEFAULT_MODEL.get(try_lang)
            key = (try_lang, try_resolved)

            failure = self._failures.get(key)
            if failure is not None and getattr(self._singletons.get(key), "_loaded", False):
                self._failures.pop(key, None)
                failure = None
            if failure is not None and now < failure.retry_at:
                last_error, last_tb = failure.error, failure.traceback
                if not fallback:
                    raise last_error.with_traceback(last_tb)
                continue

            key, backend = self._backend(try_lang, try_model, try_resolved)
            try:
                result = runner(backend, text)
            except (ModelDownloadError, ModelLoadError) as exc:
                delay = self._cooldown
                if isinstance(exc, ModelLoadError) and failure is not None:
                    delay = max(delay, min(failure.delay * 2, MAX_LOAD_RETRY_S))
                self._failures[key] = _Failure(
                    exc, exc.__traceback__, time.monotonic() + delay, delay
                )
                if not fallback:
                    raise
                last_error, last_tb = exc, exc.__traceback__
                LOG.warning(
                    "Model %r unavailable for lang %r (%s) — falling back.",
                    try_model, try_lang, exc,
                )
                continue
            self._failures.pop(key, None)
            return result
        raise last_error.with_traceback(last_tb)

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
        the offsets of its stressed vowels and whether е→ё was restored
        inside it.  Offsets refer to the input exactly as the
        caller passed it.
        """
        def runner(backend, t):
            if hasattr(backend, "mark_offsets"):
                marks, yo = backend.mark_offsets(t)
                return marks, yo, in_form_of(t, render_marks(t, marks, yo))
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
            stressed = tuple(sorted(o - start for o in mark_offsets if start <= o < end))
            words.append(StressedWord(
                text=token, start=start, end=end,
                stressed_index=stressed[0] if stressed else None,
                yo_restored=any(start <= o < end for o in yo_set),
                stressed_indices=stressed,
            ))
        return StressResult(original=text, marked=marked, words=tuple(words))

    def warm_up(self, lang: str, model: str | None = None) -> None:
        """Download and load the model for *lang* ahead of the first request.

        A successful load also clears a remembered failure for the pair, so
        a service recovers after an operator repairs a corrupt cache.
        """
        lang, model, resolved = self._resolve(lang, model, None)
        key, backend = self._backend(lang, model, resolved)
        ensure = getattr(backend, "_ensure_loaded", None)
        if ensure is not None:
            ensure()
        self._failures.pop(key, None)


#: The pipeline behind the module-level convenience API.
DEFAULT_PIPELINE = StressPipeline()
