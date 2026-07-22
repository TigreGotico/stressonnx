"""Adversarial linguistic edge cases across all backends.

Every assertion is an exact natural-language input → exact output check.
Expected values were verified against the running models and, where stress
placement is concerned, against standard dictionaries (Зализняк А. А.,
«Грамматический словарь русского языка»; СУМ for Ukrainian).
"""
import pytest

from stressonnx import (
    DEFAULT_MODEL,
    RUACCENT_LANGS,
    MAIN_LANGS,
    SIMPLE_LANGS,
    STRESS_TOKEN,
    stress,
    to_plus_notation,
)
from stressonnx.notation import _insert_stress


# ---------------------------------------------------------------------------
# Tokenizer robustness: typographic punctuation and digits are boundaries
# ---------------------------------------------------------------------------

def test_guillemets_do_not_block_lookup():
    assert stress("«замок» — надёжный", "ru") == "«замо́к» — надёжный"


def test_ellipsis_and_em_dash_kaz():
    assert stress("Алматы… Астана—Шымкент", "kaz") == "Алматы́… Астана́—Шымке́нт"


def test_guillemets_kaz():
    assert stress("«Сәлем» Қазақстан!", "kaz") == "«Сәле́м» Қазақста́н!"


def test_digits_are_boundaries():
    assert stress("дом 5, квартира 2-я", "ru") == "дом 5, кварти́ра 2-я"


# ---------------------------------------------------------------------------
# Russian hyphenated enclitic particles never carry stress
# (Русская грамматика, АН СССР 1980: частицы -то, -либо, -нибудь, -таки, -ка)
# ---------------------------------------------------------------------------

def test_ruaccent_clitics_unstressed():
    assert stress("что-либо взял-таки, скажи-ка", "ru") == "что-либо взял-таки, скажи́-ка"


def test_ruaccent_standalone_libo_is_stressed():
    # ли́бо as a standalone conjunction DOES carry stress
    assert stress("либо ты, либо я", "ru") == "ли́бо ты, ли́бо я"


def test_silero_clitic_to():
    assert stress("кто-то пришёл", "ru", model="silero") == "кто́-то пришё́л"


# ---------------------------------------------------------------------------
# ё restoration (silero ru): е→ё where ё must carry the stress
# ---------------------------------------------------------------------------

def test_silero_ru_yo_restoration():
    assert stress("зеленый лес шумит", "ru", model="silero") == "зелё́ный ле́с шуми́т"
    assert stress("мой котенок пьет молоко", "ru", model="silero") == "мо́й котё́нок пьё́т молоко́"
    assert stress("самолет летит высоко", "ru", model="silero") == "самолё́т лети́т высоко́"


def test_silero_ru_existing_yo_is_stressed():
    # ё is inherently stressed in Russian orthography
    assert stress("ёжик в тумане", "ru", model="silero") == "ё́жик в тума́не"


def test_silero_ru_ambiguous_yo_homograph_skipped():
    # все/всё needs the (unexported) homograph model — must NOT guess
    result = stress("все еще идет дождь", "ru", model="silero")
    assert result.startswith("все́")  # not всё́


# ---------------------------------------------------------------------------
# Idempotency / re-derivation contracts per backend
# ---------------------------------------------------------------------------

def test_simple_skips_already_stressed():
    once = stress("Мен қазақша сөйлеймін", "kaz")
    assert stress(once, "kaz") == once


def test_silero_skips_already_stressed():
    once = stress("Добрий вечір", "ukr")
    assert stress(once, "ukr") == once


def test_ruaccent_rederives_stress():
    # RuAccent strips marks and re-derives — wrong input marks get corrected
    assert stress("Москва́ большая", "ru") == stress("Мо́сква большая", "ru")


def test_kubataba_rederives_stress():
    once = stress("красивый закат над рекой", "ru", model="kubataba")
    assert once == "краси́вый зака́т над реко́й"
    assert stress(once, "ru", model="kubataba") == once


def test_no_double_marks_any_backend():
    for lang, model in [("ru", None), ("ru", "silero"), ("ru", "kubataba"), ("kaz", None)]:
        text = "красивый город" if lang == "ru" else "Қазақстан"
        once = stress(text, lang, model=model)
        twice = stress(once, lang, model=model)
        assert STRESS_TOKEN * 2 not in twice


# ---------------------------------------------------------------------------
# Degenerate inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lang,model", [
    ("ru", "silero"), ("ukr", None), ("bel", None), ("kaz", None), ("kat", None),
])
def test_empty_and_punct_only(lang, model):
    assert stress("", lang, model=model) == ""
    assert stress("!!!", lang, model=model) == "!!!"
    assert stress("   ", lang, model=model) == "   "


def test_ruaccent_normalizes_aggressively():
    # documented contract: RUAccent normalizes input (drops unknown symbols,
    # collapses whitespace) before stressing — it is not layout-preserving
    assert stress("Привет… мир!", "ru") == "Приве́т мир!"


# ---------------------------------------------------------------------------
# Case preservation
# ---------------------------------------------------------------------------

def test_uppercase_input_keeps_case():
    assert stress("ПРИВЕТ", "ru") == "ПРИВЕ́Т"


def test_capitalized_word_mark_position():
    assert stress("Салам", "aze_cyr") == "Сала́м"


# ---------------------------------------------------------------------------
# Real-sentence spot checks (dictionary-verified stress)
# ---------------------------------------------------------------------------

def test_ru_homographs_in_context():
    assert stress("я купил муку в магазине", "ru") == "я купи́л муку́ в магази́не"
    assert stress("эта мука невыносима", "ru") == "эта му́ка невыноси́ма"
    assert stress("белок яйца полезен", "ru") == "бело́к яйца́ поле́зен"
    assert stress("белка прыгает по деревьям", "ru") == "бе́лка пры́гает по дере́вьям"


def test_bel_sentence():
    assert stress("Я люблю чытаць кнігі", "bel") == "Я́ люблю́ чыта́ць кні́гі"


def test_simple_language_sentences():
    assert stress("Мин татарча сөйләшәм", "tat") == "Ми́н тата́рча сөйләшә́м"
    assert stress("Салом дунё", "tgk") == "Сало́м дунё́"
    assert stress("გამარჯობა მეგობრებო", "kat") == "გამარჯო́ბა მეგო́ბრებო"


def test_wrong_script_input_left_untouched():
    # Georgian model + Cyrillic input: nothing matches the alphabet → no-op.
    # Callers guard with lang_to_script()/input_scripts (see README).
    assert stress("Тбилиси и Батуми", "kat") == "Тбилиси и Батуми"


# ---------------------------------------------------------------------------
# Notation conversion
# ---------------------------------------------------------------------------

def test_to_plus_notation_handles_precomposed_acute():
    # NFC-composed á must be recognized as a + U+0301
    assert to_plus_notation("dünyá") == "düny+a"
    assert to_plus_notation("приве́т") == "прив+ет"


def test_to_plus_notation_leaves_other_diacritics():
    # ё (diaeresis), ö, й (breve) must never be decomposed
    assert to_plus_notation("ёжик öl йод") == "ёжик öl йод"


def test_notation_round_trip():
    from stressonnx.notation import _plus_to_diacritic
    for s in ["приве́т", "Сәле́м Қазақста́н", "düny+a"]:
        plus = to_plus_notation(s) if STRESS_TOKEN in s else s
        assert to_plus_notation(_plus_to_diacritic(plus)) == plus


# ---------------------------------------------------------------------------
# Registry invariants
# ---------------------------------------------------------------------------

def test_full_default_model_mapping():
    """The complete DEFAULT_MODEL table, frozen: priority ruaccent > silero > simple."""
    expected = {"ru": "ruaccent", "ukr": "silero", "bel": "silero"}
    expected.update({lang: "simple" for lang in SIMPLE_LANGS})
    assert DEFAULT_MODEL == expected


def test_lang_sets_are_consistent():
    assert RUACCENT_LANGS == {"ru"}
    assert MAIN_LANGS == {"ru", "ukr", "bel"}
    assert "bel" not in SIMPLE_LANGS and "bel_simple" in SIMPLE_LANGS
    assert len(SIMPLE_LANGS) == 20


# ---------------------------------------------------------------------------
# Defensive insertion guard
# ---------------------------------------------------------------------------

def test_insert_stress_bounds_and_vowel_guard():
    assert _insert_stress("дом", 1, "аоу") == "до́м"
    assert _insert_stress("дом", 0, "аоу") == "дом"      # consonant → no mark
    assert _insert_stress("дом", 99, None) == "дом"      # out of range → no mark
    assert _insert_stress("дом", -5, None) == "дом"
    assert _insert_stress("дюнья", 4, None) == "дюнья́"   # loanword vowel, no set
