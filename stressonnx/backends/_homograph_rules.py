"""Grammar rules and a frequency prior for the homographs of a silero bundle.

A homograph is a spelling with two or more stress readings in the bundle's
dictionary source (``be``: GrammarDB lists за́мак and зама́к).  The
dictionary layer skips such spellings, and the neural model guesses.  This
layer picks a reading from the words before the token: first a grammar rule
over the grammatical tags of each reading (every rule names its source in
its docstring; a rule marked MINED has no citation and carries its counts
in the bundle's ``meta.json``), then the most frequent reading in the
corpora the bundle's ``meta.json`` names.  A homograph with no rule signal
and no prior stays with the model.

Runtime is pure Python over one gzip JSON file (``homographs.json.gz``):

* ``sets``: the closed word classes the rules read (prepositions by the
  case they govern, numerals, subject pronouns, modals, the modifier
  agreement table, adjectival endings);
* ``words``: ``spelling → {"readings": {ordinal: [tag, ...]}, "prior": ordinal}``
  with the stressed-vowel ordinal of each reading and its tags
  (``pos:N``, ``case:G``, ``num:P``, ``inf``, ``fin``, ``imp``).
"""
import gzip
import json


def _unique(readings: dict, want) -> int | None:
    """The one ordinal whose tags satisfy *want* while no other does."""
    hits = [o for o, tags in readings.items() if want(tags)]
    return hits[0] if len(hits) == 1 else None


def _all_cased(readings: dict) -> bool:
    """Every reading is a declined form, so a case rule can choose between them."""
    return all(any(t.startswith("case:") for t in tags) for tags in readings.values())


def _agreement(readings: dict, combos: set) -> int | None:
    """The one reading whose case and number are in *combos* (``{"GS", "NP", ...}``)."""
    def want(tags):
        cases = {t[5:] for t in tags if t.startswith("case:")}
        nums = {t[4:] for t in tags if t.startswith("num:")}
        return any(c[0] in cases and c[1] in nums for c in combos)
    return _unique(readings, want)


class _Context:
    """The two words before the token (``""`` at the start of the text)."""
    __slots__ = ("l1", "l2", "sets")

    def __init__(self, words: list, idx: int, sets: dict):
        self.sets = sets
        self.l1 = words[idx - 1] if idx >= 1 else ""
        self.l2 = words[idx - 2] if idx >= 2 else ""

    def governed_case(self, word: str) -> str | None:
        """The one case *word* governs as a preposition, or None."""
        if word in self.sets["gen_preps"]:
            return "G"
        if word in self.sets["acc_preps"]:
            return "A"
        if word in self.sets["loc_preps"]:
            return "L"
        return None


def rule_gen_prep(ctx, readings):
    """Genitive after a preposition that governs the genitive only.

    Source: English Wiktionary, the ``{{+obj|be|gen}}`` template on each
    preposition entry (без, для, да, пасля, акрамя, сярод, супраць, дзеля,
    замест, апрача: https://en.wiktionary.org/wiki/<word>#Belarusian)."""
    if ctx.l1 in ctx.sets["gen_preps"]:
        return _unique(readings, lambda t: "case:G" in t)
    return None


def rule_num234(ctx, readings):
    """Nominative plural noun after два, дзве, тры, чатыры, абодва, абедзве.

    Source: https://be.wikipedia.org/wiki/Лічэбнік: after два, дзве, тры,
    чатыры "назоўнікі ў назоўным склоне маюць форму множнага ліку"."""
    if ctx.l1 in ctx.sets["num234"] and _all_cased(readings):
        return _unique(readings, lambda t: {"case:N", "num:P", "pos:N"} <= t)
    return None


def rule_num5(ctx, readings):
    """Genitive plural noun after пяць and higher.

    Source: https://be.wikipedia.org/wiki/Лічэбнік: after пяць and higher
    "ужываюцца назоўнікі ў форме роднага склону множнага ліку"."""
    if ctx.l1 in ctx.sets["num5"] and _all_cased(readings):
        return _unique(readings, lambda t: {"case:G", "num:P", "pos:N"} <= t)
    return None


def rule_loc(ctx, readings):
    """Locative after пры; the locative or accusative reading after у, ва,
    на, аб, па when exactly one reading has either case.

    Source: English Wiktionary ``{{+obj}}`` on each preposition entry: пры
    governs the locative only; у, ва, на, аб, па govern the locative or the
    accusative (https://en.wiktionary.org/wiki/<word>#Belarusian)."""
    if ctx.l1 in ctx.sets["loc_preps"]:
        return _unique(readings, lambda t: "case:L" in t)
    if ctx.l1 in ctx.sets["loc_acc_preps"]:
        return _unique(readings, lambda t: "case:L" in t or "case:A" in t)
    return None


def rule_acc_prep(ctx, readings):
    """Accusative after праз or пад.

    Source: English Wiktionary ``{{+obj|be|acc}}`` on the entries of праз
    and пад (https://en.wiktionary.org/wiki/<word>#Belarusian)."""
    if ctx.l1 in ctx.sets["acc_preps"]:
        return _unique(readings, lambda t: "case:A" in t)
    return None


def _modifier_before(ctx, readings) -> bool:
    return (_all_cased(readings) and ctx.l1 not in ctx.sets["num234"]
            and ctx.l1 not in ctx.sets["num5"])


def _restrict(ctx, combos: set) -> set:
    """*combos* narrowed to the case a preposition before the modifier governs."""
    case = ctx.governed_case(ctx.l2)
    return {c for c in combos if c[0] == case} if case else combos


def rule_agree(ctx, readings):
    """A noun agrees in case and number with the pronoun-adjective or
    numeral before it; a preposition before that modifier restricts the
    case.

    Sources: agreement, https://be.wikipedia.org/wiki/Прыметнік (the
    adjective agrees with the noun in gender, number and case); the case
    and number set of each modifier form, GrammarDB paradigm tags of
    pronoun-adjectives (``SA``) and numerals (``M``),
    https://github.com/Belarus/GrammarDB (``tags.svg``)."""
    if not _modifier_before(ctx, readings) or ctx.l1 not in ctx.sets["agree"]:
        return None
    return _agreement(readings, _restrict(ctx, ctx.sets["agree"][ctx.l1]))


def rule_agree_ending(ctx, readings):
    """As ``rule_agree``, for a word before the noun that ends in an
    adjectival ending (``sets.adj_endings``) and is not in the modifier table.

    MINED: no citation.  The endings and their case and number sets were
    read off the adjective paradigms, and the rule's counts on the prior
    corpora are in the bundle's ``meta.json`` (``mined``)."""
    if not _modifier_before(ctx, readings) or ctx.l1 in ctx.sets["agree"]:
        return None
    for ending, combos in ctx.sets["adj_endings"].items():
        if ctx.l1.endswith(ending) and len(ctx.l1) > len(ending) + 2:
            return _agreement(readings, _restrict(ctx, combos))
    return None


def rule_subject_pronoun(ctx, readings):
    """The finite reading after a subject pronoun, the infinitive after a
    modal, where the readings are one infinitive and one finite verb form.

    MINED: no citation; counts in the bundle's ``meta.json`` (``mined``)."""
    inf = _unique(readings, lambda t: "inf" in t)
    fin = _unique(readings, lambda t: "fin" in t)
    if inf is None or fin is None or inf == fin:
        return None
    if ctx.l1 in ctx.sets["subject_pronouns"]:
        return fin
    if ctx.l1 in ctx.sets["modals"]:
        return inf
    return None


#: Rules in the order they are tried; the first answer wins.
RULES = (
    ("gen_prep", rule_gen_prep),
    ("num234", rule_num234),
    ("num5", rule_num5),
    ("loc", rule_loc),
    ("acc_prep", rule_acc_prep),
    ("agree", rule_agree),
    ("agree_ending", rule_agree_ending),
    ("subject_pronoun", rule_subject_pronoun),
)

#: Rules with no citation, reported as mined wherever their counts appear.
MINED = frozenset({"agree_ending", "subject_pronoun"})


class HomographRules:
    """The loaded layer of one bundle: ``resolve`` answers per word."""

    def __init__(self, blob: dict):
        sets = blob["sets"]
        self.sets = {k: frozenset(v) for k, v in sets.items() if isinstance(v, list)}
        self.sets["agree"] = {k: frozenset(v) for k, v in sets["agree"].items()}
        self.sets["adj_endings"] = {k: frozenset(v) for k, v in sets["adj_endings"].items()}
        self.words = {
            w: {"readings": {int(o): frozenset(t) for o, t in e["readings"].items()},
                "prior": e.get("prior")}
            for w, e in blob["words"].items()
        }

    @classmethod
    def load(cls, path: str) -> "HomographRules":
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            return cls(json.load(fh))

    def __contains__(self, word: str) -> bool:
        return word in self.words

    def resolve(self, words: list, idx: int):
        """(stressed-vowel ordinal, stage) for ``words[idx]``, or ``(None, None)``.

        *words* are the cleaned word tokens of the whole text in order, with
        separators and punctuation left out.  The stage is ``rule:<name>``
        or ``prior``.
        """
        entry = self.words.get(words[idx])
        if entry is None:
            return None, None
        ctx = _Context(words, idx, self.sets)
        for name, rule in RULES:
            ordinal = rule(ctx, entry["readings"])
            if ordinal is not None:
                return ordinal, f"rule:{name}"
        if entry["prior"] is not None:
            return entry["prior"], "prior"
        return None, None
