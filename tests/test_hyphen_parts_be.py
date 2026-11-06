"""Belarusian hyphen parts that never carry stress are masked out of prediction."""
import re

from stressonnx._common import _RE_RU_COND, tokenize

_BE = re.compile("[^ЁІЎАБВГДЕЖЗЙКЛМНОПРСТУФХЦЧШЫЬЭЮЯабвгдежзйклмнопрстуфхцчшыьэюяёіў]")


def test_be_prefix_and_particle_are_masked():
    t = tokenize("Па-беларуску абы-хто сяк-та усё-такі", _BE, lang="be")
    assert t.raw == ["Па-", "беларуску", " ", "абы-", "хто", " ", "сяк-", "та", " ", "усё-", "такі"]
    assert t.predict == [False, True, False, False, True, False, True, False, False, True, False]


def test_be_nebudz_stays_predicted():
    # -не́будзь carries its own stress (en-Wiktionary: які́-не́будзь, [ˈnʲebut͡sʲ])
    t = tokenize("хто-небудзь", _BE, lang="be")
    assert t.predict == [True, True]


def test_ru_mask_unchanged_and_other_languages_unmasked():
    assert tokenize("кто-то как-нибудь", _RE_RU_COND, lang="ru").predict == [True, False, False, True, False]
    assert tokenize("па-беларуску", _BE).predict == [True, True]
    assert tokenize("па-беларуску", _BE, lang="uk").predict == [True, True]
