"""Letters of a language's alphabet that its spec lacked, and a letter its vocabulary spells another way."""
import json
from importlib.resources import files

from stressonnx import stress


def test_kk_yo_is_a_vowel_of_the_alphabet():
    # Kazakh Cyrillic has 42 letters, Ё and Ъ among them, for Russian loanwords; ё is a vowel
    spec = json.loads((files("stressonnx") / "languages" / "kk.json").read_text(encoding="utf-8"))
    assert "ё" in spec["alpha"] and "ъ" in spec["alpha"]
    assert "ё" in spec["vowels"]


def test_kjh_letter_che_with_descender_finds_the_vocabulary_entry():
    # the kjh vocabulary writes Ӌ as Ҷ; both spellings get its reading, and the input letter stays
    assert stress("аалыҷаам", "kjh") == "аа́лыҷаам"
    assert stress("аалыӌаам Аалыӌаам", "kjh") == "аа́лыӌаам Аа́лыӌаам"
    assert stress("ААЛЫӋААМ", "kjh") == "АА́ЛЫӋААМ"
