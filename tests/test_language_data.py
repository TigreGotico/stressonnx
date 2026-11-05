"""Language-data defects of the ``simple`` model.

Expected values come from the sources named next to each case, never from
the library's own output.
"""
import gzip
import unicodedata

import pytest

from stressonnx import stress
from stressonnx._common import lower_preserving_length
from stressonnx.backends.simple import SimpleStressor
from stressonnx.registry import HF_REPO_REVISION, LANGUAGES

ACUTE = "́"

#: Vocabulary files that the runtime repairs at load until they are
#: re-uploaded; the cases marked with it pass once HF_REPO_REVISION moves.
WAITS_ON_UPLOAD = pytest.mark.xfail(
    HF_REPO_REVISION == "13ab09f55a725395f857695ac218b5ae1f5b22e4",
    reason="waits on the re-uploaded vocabularies", strict=True,
)


def simple(text, lang):
    return stress(text, lang, model="simple")


def vocab_words(lang):
    s = SimpleStressor(lang)
    s._ensure_loaded()
    return s._vocab


# --- Uzbek Cyrillic -------------------------------------------------------

def test_uz_cyrl_alphabet_is_the_35_letter_set():
    # Wikipedia, "Uzbek alphabet": the Cyrillic alphabet has 35 letters.
    letters = "абвгдеёжзийклмнопрстуфхцчшъьэюяўқғҳ"
    assert len(letters) == 35
    assert set(LANGUAGES["uz-Cyrl"]["alpha"]) == set(letters)
    # Ё Ю Я correspond to Latin yo yu ya; э is a vowel letter
    assert set("ёюяэ") <= set(LANGUAGES["uz-Cyrl"]["vowels"])


@pytest.mark.parametrize("word,expected", [
    # Boeschoten 1998: final stress.  дунё (dunyo) has the vowels у, ё.
    ("дунё", "дунё́"),
    # акция: а, и, я → the last vowel is я
    ("акция", "акция́"),
    # ҳаёт (hayot): а, ё → ё
    ("ҳаёт", "ҳаё́т"),
])
def test_uz_cyrl_rule_sees_every_vowel(word, expected):
    assert simple(word, "uz-Cyrl") == expected


# --- Azerbaijani Latin ----------------------------------------------------

def test_turkic_lowercase_round_trip():
    assert lower_preserving_length("İIıi", turkic=True) == "iııi"
    assert lower_preserving_length("İIıi") == "iiıi"


def test_az_dotted_capital_i_reaches_the_vocabulary():
    # audit finding 11: the vocabulary key is i̇brahimov with ordinal 2,
    # so the second i is stressed: İbrahímov
    assert simple("İbrahimov", "az-Latn") == "İbrahi\u0301mov"
    assert "̇" not in "".join(vocab_words("az-Latn"))


# --- Belarusian -----------------------------------------------------------

def test_be_never_stresses_the_non_syllabic_u():
    # Wikipedia, "Short U": ў "does not form syllables"
    words = vocab_words("be")
    assert "аўтобус" not in words
    out = simple(" ".join(w for w in words if "ў" in w), "be")
    assert "ў" + ACUTE not in out
    assert simple("аўтобус", "be") == "аўтобус"


@WAITS_ON_UPLOAD
def test_be_repaired_entries_stress_the_vowel_after_u():
    # Wiktionary be: аўто́бус, даўно́, заўсёды with ё; the shipped index
    # pointed at ў and the staged bundle moves it to the next vowel
    assert simple("аўтобус даўно", "be") == "аўто́бус даўно́"


def test_be_apostrophe_forms_match():
    # Belarusian orthography writes the apostrophe; сям’я́ is stressed on я
    assert simple("сям’я", "be") == "сям’я́"
    assert simple("сям'я", "be") == "сям'я́"
    assert simple("з’ява", "be") == "з’я́ва"


def test_be_vocabulary_keys_are_nfc_without_apostrophe():
    words = vocab_words("be")
    assert all(unicodedata.is_normalized("NFC", w) for w in words)
    assert not any("’" in w for w in words)


# --- Macedonian -----------------------------------------------------------

@pytest.mark.parametrize("word,expected", [
    # Wikipedia, "Macedonian language": р "acts as a vowel when found between
    # two consonants (e.g. црква)"; a disyllabic word is stressed on the
    # second-to-last syllable: цр́-ква
    ("црква", "цр́ква"),
    ("прв", "пр́в"),
    ("срце", "ср́це"),
    # three nuclei р, и, а → antepenultimate
    ("Србија", "Ср́бија"),
    # р next to a vowel is a consonant: ра́бота keeps its antepenultimate rule
    ("работа", "ра́бота"),
    # Wiktionary writes word-initial syllabic р with an apostrophe: ’рж
    # "rye", IPA [r̩ʃ], derived ’ржан; the only or first nucleus is р
    ("’рж", "’р́ж"),
    ("’ржан", "’р́жан"),
    ("'рж", "'р́ж"),
])
def test_mk_syllabic_r_is_a_stress_base(word, expected):
    assert simple(word, "mk") == expected


@WAITS_ON_UPLOAD
def test_mk_vocabulary_holds_words_with_gje_and_kje():
    # Wiktionary marks these headwords; the acute of ќ is not a stress mark
    words = vocab_words("mk")
    assert sum(1 for w in words if set(w) & set("ѓќ")) >= 300


def test_wiktionary_builder_ignores_the_acute_of_kje():
    from export.build_wiktionary_vocab import stressed_index
    assert stressed_index("куќа", "ку́ќа", "аеиоу") == 1
    assert stressed_index("веќе", "ве́ќе", "аеиоу") == 1
    assert stressed_index("вода", "во́да", "аеиоу") == 1


# --- Russian --------------------------------------------------------------

def test_ru_yo_marks_the_stressed_syllable():
    # Wikipedia, "Yo (Cyrillic)": ё "is almost always stressed"; the other
    # Russian backends and the README write the acute on it (надё́жен)
    assert simple("Ещё всё зелёный", "ru") == "Ещё́ всё́ зелё́ный"
    assert simple("жыццё", "be") == "жыццё́"
    # a word with two ё gives the rule nothing to pick: unmarked
    assert simple("ёлкаёлка", "ru") == "ёлкаёлка"


@pytest.mark.xfail(strict=True, reason="waits on a ru_simple vocabulary built from the full RUAccent dictionary")
def test_ru_vocabulary_holds_everyday_words():
    # RUAccent accents.json.gz: январ+я, +это, м+ама
    assert simple("января это мама", "ru") == "января́ э́то ма́ма"
