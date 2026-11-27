"""Belarusian hyphen parts that never carry stress are masked out of prediction."""
import re

from stressonnx._common import _RE_RU_COND, tokenize

_BE = re.compile("[^ЁІЎАБВГДЕЖЗЙКЛМНОПРСТУФХЦЧШЫЬЭЮЯабвгдежзйклмнопрстуфхцчшыьэюяёіў]")


def test_be_prefix_and_particle_are_masked():
    t = tokenize("Па-беларуску абы-хто такі-та усё-такі", _BE, lang="be")
    assert t.raw == ["Па-", "беларуску", " ", "абы-", "хто", " ", "такі-", "та", " ", "усё-", "такі"]
    assert t.predict == [False, True, False, False, True, False, True, False, False, True, False]


def test_be_particle_after_a_host_word_only():
    # GrammarDB slovy-2008-stress.txt: такі́-та, то́-та, та́к-такі, даво́лі-такі
    # unstressed; та-та́, сякі́-такі́, про́ста-такі́ stressed
    assert tokenize("то-та так-такі даволі-такі", _BE, lang="be").predict == [
        True, False, False, True, False, False, True, False]
    assert tokenize("та-та сякі-такі проста-такі", _BE, lang="be").predict == [
        True, True, False, True, True, False, True, True]


def test_be_host_word_matches_with_a_stress_mark():
    assert tokenize("такі́-та", _BE, lang="be").predict == [True, False]


def test_be_nebudz_stays_predicted():
    # -не́будзь carries its own stress (en-Wiktionary: які́-не́будзь, [ˈnʲebut͡sʲ])
    t = tokenize("хто-небудзь", _BE, lang="be")
    assert t.predict == [True, True]


def test_ru_simple_leaves_clitics_unstressed():
    from stressonnx import stress
    assert stress("кто-то скажи-ка", "ru", model="simple") == "кто́-то скажи́-ка"


def test_ru_mask_unchanged_and_other_languages_unmasked():
    assert tokenize("кто-то как-нибудь", _RE_RU_COND, lang="ru").predict == [True, False, False, True, False]
    assert tokenize("па-беларуску", _BE).predict == [True, True]
    assert tokenize("па-беларуску", _BE, lang="uk").predict == [True, True]
