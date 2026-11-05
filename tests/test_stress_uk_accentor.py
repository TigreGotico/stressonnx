"""uk-accentor: the ukrainian-accentor-transformer seq2seq model as a selectable
Ukrainian model.

The expected strings are the output of upstream ukrainian-accentor-transformer
0.1.0 at commit 1af138199c61d996094575fb0efe34a0b2e85adc
(``Accentor()(text)`` with ctranslate2 4.8.5 and sentencepiece 0.2.3 on the
model at Hugging Face revision 5289b00e07f0aa8678825cca59684b1fb0ec8c3c).
They are parity values, not gold stress: upstream marks more than one vowel
in some words (``за́мо́к``, ``та́ко́ж``) and leaves some words unmarked.

``STRESSONNX_TEST_CACHE_DIR`` points the stressor at a local model tree
(``<dir>/ukr_accentor/<file>``) so the bundle can be tested before it is
published; unset, the bundle comes from the pinned Hugging Face revision.
"""
import os
import unicodedata

import pytest

from stressonnx import DEFAULT_MODEL, MODEL_REGISTRY, UkAccentorStressor, make_stressor
from stressonnx.backends.uk_accentor import _split_at_punctuation, _split_long
from stressonnx.pipeline import FALLBACK_PRIORITY, _PREFER_ORDERS, StressPipeline
from stressonnx.registry import LANGUAGES

CACHE_DIR = os.environ.get("STRESSONNX_TEST_CACHE_DIR")

CASES = [
    ("Привіт хлопче, як справи.", "Приві́т хло́пче, як спра́ви."),
    ("Кам'янець-Подільський - місто в Хмельницькій області України, центр "
     "Кам'янець-Подільської міської об'єднаної територіальної громади і "
     "Кам'янець-Подільського району.",
     "Кам'яне́ць-Поді́льський - мі́сто в Хмельни́цькій о́бласті Украї́ни, центр "
     "Кам'яне́ць-Поді́льської місько́ї об'є́днаної територіа́льної грома́ди і "
     "Кам'яне́ць-Поді́льського райо́ну."),
    ("Адже за 25 років кошти, які спрямовувалися на чорнобильські програми, не дали жодного результату.",
     "А́дже за 25 ро́кі́в ко́шти, які́ спрямо́вувалися на чорно́бильські про́гра́ми́, не дали́ жо́дного результа́ту."),
    ("старовинний замок", "старови́нний за́мо́к"),
    ("# зради на", "# зра́ди на"),
    ("% домогосподарств мус", "% домогосподарств мус"),
    ("село", "село"),
    ("xyz 123", "xyz 123"),
    ("", ""),
    ("   ", "   "),
    # 60 pieces without punctuation: cut in halves until every chunk is under 30
    ("Веселка також райдуга атмосферне оптичне явище що являє собою одну дві чи декілька "
     "спектральних дуг або кіл якщо дивитися з повітря що спостерігаються на тлі хмари якщо "
     "вона розташована проти Сонця Червоний колір спектру ми бачимо з зовнішнього боку "
     "первинної веселки а фіолетовий із внутрішнього",
     "Весе́лка та́ко́ж ра́йдуга атмосфе́рне опти́чне я́вище що явля́є собо́ю одну́ дві чи де́кілька "
     "спектра́льних дуг або́ кіл якщо́ диви́тися з пові́тря що спостеріга́ються на тлі хма́ри якщо́ "
     "вона́ розташо́вана про́ти Со́нця Черво́ний ко́лір спе́ктру ми ба́чимо з зо́внішнього бо́ку "
     "перви́нної весе́лки а фіоле́товий із вну́трішнього"),
]
CASE_IDS = ["readme-test", "readme", "chunks", "two-marks", "hash", "unmarked", "mono",
            "latin", "empty", "spaces", "long"]


@pytest.fixture(scope="module")
def accentor():
    return make_stressor(model="uk-accentor", lang="uk", cache_dir=CACHE_DIR)


@pytest.mark.parametrize("text,expected", CASES, ids=CASE_IDS)
def test_matches_upstream(accentor, text, expected):
    assert accentor(text) == expected


def test_registry_entry():
    entry = MODEL_REGISTRY["uk-accentor"]
    assert entry.langs == frozenset({"uk"})
    assert entry.family == "uk_accentor"
    assert entry.hf_subdir == "ukr_accentor"


def test_never_selected_by_default_or_prefer():
    assert DEFAULT_MODEL["uk"] == "silero"
    assert "uk-accentor" not in FALLBACK_PRIORITY
    for order in _PREFER_ORDERS.values():
        assert "uk-accentor" not in order


def test_make_stressor_returns_uk_accentor():
    assert isinstance(make_stressor(model="uk-accentor", lang="uk"), UkAccentorStressor)
    assert isinstance(make_stressor(model="uk-accentor", lang="uk-UA"), UkAccentorStressor)


def test_make_stressor_rejects_other_language():
    with pytest.raises(ValueError, match="does not support language"):
        make_stressor(model="uk-accentor", lang="ru")


def test_punctuation_digits_and_latin_pass_through(accentor):
    # upstream: 'Приві́т, світ!  «Так»… 42 — ok; ї/Ї'
    assert accentor("Привіт, світ!  «Так»… 42 — ok; ї/Ї") == "Приві́т, світ!  «Так»… 42 — ok; ї/Ї"


def test_tab_and_double_space_survive(accentor):
    # upstream: 'Приві́т,\tсвіт! два  пробі́ли'
    assert accentor("Привіт,\tсвіт! два  пробіли") == "Приві́т,\tсвіт! два  пробі́ли"


def test_keeps_every_input_character(accentor):
    text = "Привіт,\tсвіт!  «Так»… 42 — ok; ї/Ї\n+ъ' Кам'янець-Подільський"
    out = accentor(text)
    assert out.replace("́", "") == text
    assert out.startswith("Приві́т,")


def test_premarked_word_is_left_alone(accentor):
    # upstream re-marks every word and gives 'за́мо́к стої́ть на го́рі́'
    assert accentor("за́мок стоїть на горі") == "за́мок стої́ть на го́рі́"


def test_decomposed_input_comes_back_decomposed(accentor):
    nfd = unicodedata.normalize("NFD", "Привіт хлопче, як справи.")
    out = accentor(nfd)
    assert unicodedata.is_normalized("NFD", out)
    assert unicodedata.normalize("NFC", out) == "Приві́т хло́пче, як спра́ви."


def test_decomposed_letters_are_still_marked(accentor):
    # upstream aligns its output with the raw text, so a word spelled with
    # и + U+0306 loses its mark there; the port works on the NFC form
    text, expected = CASES[1]
    assert "̆" in unicodedata.normalize("NFD", text)
    out = accentor(unicodedata.normalize("NFD", text))
    assert unicodedata.is_normalized("NFD", out)
    assert unicodedata.normalize("NFC", out) == expected


def _offsets_of_marks(expected: str) -> list:
    """Offsets, in the text without marks, of the letters the upstream
    output marks."""
    offsets, i = [], 0
    for ch in expected:
        if ch == "\u0301":
            offsets.append(i - 1)
        else:
            i += 1
    return offsets


def test_mark_offsets_point_at_vowels(accentor):
    text = "Привіт хлопче, як справи."
    marks, yo = accentor.mark_offsets(text)
    assert yo == []
    assert marks == [4, 9, 21]
    assert [text[o] for o in marks] == ["і", "о", "а"]
    assert marks == _offsets_of_marks("Приві́т хло́пче, як спра́ви.")


def test_marks_after_a_consonant_are_dropped(accentor):
    # the model writes U+0301 after the р of хрещеною and after the ю;
    # a stressed index is an offset of a vowel, so the р mark goes
    text = "ути хрещеною мат"
    vowels = set(LANGUAGES["uk"]["vowels"])
    pipe = StressPipeline()
    pipe._singletons[("uk", "uk-accentor")] = accentor
    words = pipe.analyze(text, lang="uk", model="uk-accentor").words
    letters = {w.text: [w.text[i] for i in w.stressed_indices] for w in words}
    assert letters == {"ути": ["и"], "хрещеною": ["е", "е", "ю"], "мат": []}
    for word, marked in letters.items():
        assert all(ch in vowels for ch in marked), (word, marked)
    assert words[1].stressed_index == 2


def test_decoder_steps_are_bounded_by_the_source_length(accentor, monkeypatch):
    # the model never writes an end token for this input, so the beam
    # search runs to its bound: 2 pieces + 20 steps, not upstream's 256
    text = "и"
    pieces = accentor._encode(text)
    steps = []
    run = accentor._decoder.run
    monkeypatch.setattr(accentor._decoder, "run", lambda *a, **k: steps.append(1) or run(*a, **k))
    assert accentor(text) == "и"
    assert len(steps) == 2 * len(pieces) + 20


def test_mark_offsets_on_decomposed_text_index_base_letters(accentor):
    text, expected = CASES[1]
    nfd = unicodedata.normalize("NFD", text)
    assert len(nfd) > len(text)
    marks, _ = accentor.mark_offsets(nfd)
    nfc_marks = _offsets_of_marks(expected)
    assert [text[o] for o in nfc_marks] == [
        "е", "і", "і", "и", "о", "ї", "е", "і", "о", "є", "а", "а", "е", "і", "о"]
    # in NFD an offset lands on the base letter (ї is і + U+0308)
    assert [nfd[o] for o in marks] == [unicodedata.normalize("NFD", text[o])[0] for o in nfc_marks]
    # an offset moves right by one for every combining mark before it
    assert marks == [
        o + sum(1 for ch in unicodedata.normalize("NFD", text[:o]) if unicodedata.combining(ch))
        for o in nfc_marks]


def test_split_at_punctuation_keeps_every_piece():
    pieces = ["▁При", "віт", ",", "▁сві", "т", "!", "▁так"]
    chunks = _split_at_punctuation(pieces)
    assert chunks == [["▁При", "віт", ","], ["▁сві", "т", "!"], ["▁так"]]
    assert _split_at_punctuation(["▁так", "."]) == [["▁так", "."]]
    assert _split_at_punctuation([]) == []


def test_split_long_halves_at_word_start():
    pieces = [f"▁w{i}" if i % 2 == 0 else f"x{i}" for i in range(31)]
    chunks = _split_long([pieces])
    assert [p for c in chunks for p in c] == pieces
    assert all(len(c) < 30 for c in chunks)
    # the cut goes to the word start nearest the middle, left side first
    assert [len(c) for c in chunks] == [14, 17]
    assert chunks[1][0] == "▁w14"
