"""The homograph rules of stressonnx.backends._homograph_rules on a small layer.

Each expected reading comes from the rule's cited source, not from the
code: English Wiktionary ``{{+obj|be|<case>}}`` on the preposition entries
(без, дзеля: genitive; пры: locative; у, на: locative or accusative; праз,
пад: accusative), https://be.wikipedia.org/wiki/Лічэбнік (nominative plural
after два, тры, чатыры; genitive plural after пяць and higher), and
https://be.wikipedia.org/wiki/Прыметнік (a modifier agrees with its noun in
case and number).  The readings are GrammarDB RELEASE-202601 forms:
ба́цькі (genitive singular of ба́цька) and бацькі́ (nominative plural),
скла́дзе (locative singular of скла́д) and складзе́ (locative singular of
склада́), два́ры (locative singular of дво́р) and двары́ (nominative plural),
гу́льні (nominative plural of гу́льня) and гульні́ (genitive, dative
and locative singular), ву́чыцца (third person singular) and вучы́цца
(infinitive).
"""
import pytest

from stressonnx.backends._homograph_rules import MINED, RULES, HomographRules

LAYER = {
    "sets": {
        "gen_preps": ["без", "дзеля"],
        "loc_preps": ["пры"],
        "loc_acc_preps": ["у", "на"],
        "acc_preps": ["праз", "пад"],
        "num234": ["два", "тры", "чатыры"],
        "num5": ["пяць", "дзесяць"],
        "subject_pronouns": ["яна", "ён"],
        "modals": ["можа", "трэба"],
        # GrammarDB SA paradigm forms: майго is genitive or accusative
        # singular, маёй genitive, dative, instrumental or locative singular
        "agree": {"майго": ["GS", "AS"], "маёй": ["GS", "DS", "IS", "LS"]},
        "adj_endings": {"ай": ["GS", "DS", "IS", "LS", "NS"], "ага": ["GS", "AS"]},
    },
    "words": {
        "бацькі": {"readings": {"0": ["pos:N", "case:G", "num:S"], "1": ["pos:N", "case:N", "num:P"]},
                   "prior": 1},
        "складзе": {"readings": {"0": ["pos:N", "case:L", "num:S"], "1": ["pos:N", "case:L", "num:S"]}},
        "двары": {"readings": {"0": ["pos:N", "case:L", "num:S"], "1": ["pos:N", "case:N", "num:P"]}},
        "гульні": {"readings": {"0": ["pos:N", "case:N", "num:P", "case:A"],
                                "1": ["pos:N", "case:G", "num:S", "case:D", "case:L"]}, "prior": 0},
        "вучыцца": {"readings": {"0": ["pos:V", "fin"], "1": ["pos:V", "inf"]}},
        "гады": {"readings": {"0": ["pos:N", "case:N", "num:P"], "1": ["pos:N", "case:G", "num:S"]}},
        "гадоў": {"readings": {"0": ["pos:N", "case:G", "num:P"], "1": ["pos:N", "case:A", "num:S"]}},
        "стала": {"readings": {"0": ["pos:V", "fin"], "1": ["pos:N", "case:G", "num:S"]}},
        "рукі": {"readings": {"0": ["pos:N", "case:N", "num:P", "case:A"], "1": ["pos:N", "case:G", "num:S"]}},
        "мука": {"readings": {"0": ["pos:N", "case:N", "num:S"], "1": ["pos:N", "case:N", "num:S"]}, "prior": 0},
        "пара": {"readings": {"0": ["pos:N", "case:N", "num:S"], "1": ["pos:N", "case:N", "num:S"]}},
    },
}


@pytest.fixture(scope="module")
def layer():
    return HomographRules(LAYER)


def resolve(layer, text):
    """(ordinal, stage) of the last word of *text*."""
    words = text.split()
    return layer.resolve(words, len(words) - 1)


@pytest.mark.parametrize("text,expected", [
    ("без бацькі", (0, "rule:gen_prep")),      # без + genitive: ба́цькі
    ("дзеля бацькі", (0, "rule:gen_prep")),
    ("пры стала", (None, None)),                # стала́ is genitive, пры governs the locative
    ("праз рукі", (0, "rule:acc_prep")),        # праз + accusative: ру́кі
    ("пад стала", (None, None)),                # no accusative reading
    ("у рукі", (0, "rule:loc")),                # у + accusative: ру́кі
    ("на рукі", (0, "rule:loc")),
])
def test_preposition_picks_the_governed_case(layer, text, expected):
    assert resolve(layer, text) == expected


def test_locative_preposition_needs_one_qualifying_reading(layer):
    # both readings of складзе are locative, so у cannot choose and the
    # word has no prior; пры reads the one reading with the locative only
    assert resolve(layer, "у складзе") == (None, None)
    assert resolve(layer, "пры складзе") == (None, None)
    assert resolve(layer, "пры двары") == (0, "rule:loc")
    assert resolve(layer, "у двары") == (0, "rule:loc")


@pytest.mark.parametrize("text,expected", [
    ("тры гады", (0, "rule:num234")),   # nominative plural after тры
    ("два гады", (0, "rule:num234")),
    ("пяць гадоў", (0, "rule:num5")),   # genitive plural after пяць
])
def test_numeral_picks_the_number_and_case(layer, text, expected):
    assert resolve(layer, text) == expected


def test_numeral_rule_needs_declined_readings(layer):
    # стала: one reading is a verb, so the numeral rules stay silent and the prior is absent
    assert resolve(layer, "тры стала") == (None, None)


@pytest.mark.parametrize("text,expected", [
    ("майго бацькі", (0, "rule:agree")),       # майго: GS or AS; only ба́цькі is GS
    ("без майго бацькі", (0, "rule:agree")),
    ("маёй гульні", (1, "rule:agree")),        # маёй: G D I L singular; гульні́ is G D L singular
])
def test_modifier_agreement_from_the_cited_table(layer, text, expected):
    assert resolve(layer, text) == expected


def test_preposition_before_the_modifier_restricts_the_case(layer):
    # праз + accusative: майго is GS or AS, and ба́цькі is only GS, so no reading agrees
    assert resolve(layer, "праз майго бацькі") == (1, "prior")


def test_adjectival_ending_rule_is_mined_and_marked(layer):
    assert "agree_ending" in MINED and "subject_pronoun" in MINED
    assert resolve(layer, "складанай гульні") == (1, "rule:agree_ending")
    # a word in the modifier table never reaches the ending rule
    assert resolve(layer, "маёй гульні") == (1, "rule:agree")
    # a short word with the ending is not read as an adjective
    assert resolve(layer, "ай гульні") == (0, "prior")


@pytest.mark.parametrize("text,expected", [
    ("яна вучыцца", (0, "rule:subject_pronoun")),
    ("можа вучыцца", (1, "rule:subject_pronoun")),
    ("добра вучыцца", (None, None)),
])
def test_subject_pronoun_rule(layer, text, expected):
    assert resolve(layer, text) == expected


def test_rules_come_before_the_prior_and_the_prior_before_nothing(layer):
    assert [name for name, _ in RULES].index("gen_prep") < [name for name, _ in RULES].index("agree_ending")
    assert resolve(layer, "без бацькі") == (0, "rule:gen_prep")
    assert resolve(layer, "бацькі") == (1, "prior")
    assert resolve(layer, "складзе") == (None, None)


def test_identical_tag_sets_leave_only_the_prior(layer):
    assert resolve(layer, "без мука") == (0, "prior")
    assert resolve(layer, "без пара") == (None, None)


def test_unknown_word_and_text_start(layer):
    assert layer.resolve(["дом"], 0) == (None, None)
    assert "дом" not in layer and "бацькі" in layer
    assert layer.resolve(["бацькі", "без"], 0) == (1, "prior")
