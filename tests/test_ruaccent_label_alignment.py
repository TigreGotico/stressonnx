"""The stress-usage and yo labels of ``ruaccent`` line up with its word list.

The pipeline reads a punctuation run (``[…]``, ``...``, ``»,``, ``(же)``) as
one word, while the BERT pre-tokenizer cuts it into one word per
character.  Every word after such a run used to read the label of an
earlier word.  Expected marks come from the Wiktionary gold rows of the
scoring campaign (``wikt-ru:<n>`` row ids), never from the model output.
"""
import pytest

from stressonnx import make_stressor
from stressonnx.backends.ruaccent import _ruaccent_split_by_words

from tests.test_ruaccent_silero_fill import ordinals_by_token


@pytest.fixture(scope="module")
def plain(model_cache_dir):
    return make_stressor(model="ruaccent", cache_dir=model_cache_dir)


# (gold sentence, token, gold vowel ordinal): the token follows a
# punctuation run and ``ruaccent`` left it unmarked while the labels were
# shifted
GOLD = [
    ("Она звучит так (же), как в фильмах.", "фильмах", 0),                                  # wikt-ru:339
    ("Они расположены так (же), как на схеме.", "схеме", 0),                                # wikt-ru:340
    ("[…] красиво изгибаясь под тяжестью хлеба, она побежала через двор к гумну, шмыгнула через плетень и, окунувшись в облако золотистой половы, скрылась за арбами.", "красиво", 1),  # wikt-ru:484
    ("[…] красиво изгибаясь под тяжестью хлеба, она побежала через двор к гумну, шмыгнула через плетень и, окунувшись в облако золотистой половы, скрылась за арбами.", "тяжестью", 0),
    ("[…] красиво изгибаясь под тяжестью хлеба, она побежала через двор к гумну, шмыгнула через плетень и, окунувшись в облако золотистой половы, скрылась за арбами.", "она", 1),
    ("[…] красиво изгибаясь под тяжестью хлеба, она побежала через двор к гумну, шмыгнула через плетень и, окунувшись в облако золотистой половы, скрылась за арбами.", "облако", 0),
    ("Я пью... Я пью...\nКачу на новой доске\nПо пожелтевшей Москве", "новой", 0),           # wikt-ru:748
    ("Я пью... Я пью...\nКачу на новой доске\nПо пожелтевшей Москве", "доске", 1),
    ("Я пью... Я пью...\nКачу на новой доске\nПо пожелтевшей Москве", "пожелтевшей", 2),
    ("Оливковые деревья поздно вступают в плодоношение (через 10—12 лет), но зато очень долговечны (до 100 лет и более). Произрастают они в Крыму, Азербайджане, на Черноморском побережье Кавказа и в Туркмении.", "зато", 1),  # wikt-ru:1910
    ("Оливковые деревья поздно вступают в плодоношение (через 10—12 лет), но зато очень долговечны (до 100 лет и более). Произрастают они в Крыму, Азербайджане, на Черноморском побережье Кавказа и в Туркмении.", "Произрастают", 3),
    ("Оливковые деревья поздно вступают в плодоношение (через 10—12 лет), но зато очень долговечны (до 100 лет и более). Произрастают они в Крыму, Азербайджане, на Черноморском побережье Кавказа и в Туркмении.", "Туркмении", 1),
    ("Так много игрушек... Я хочу вон ту.", "хочу", 1),                                      # wikt-ru:2547
]


@pytest.mark.parametrize("sentence,token,gold", GOLD)
def test_word_after_a_punctuation_run_reads_its_own_label(plain, sentence, token, gold):
    assert ordinals_by_token(plain(sentence))[token] == [gold]


def test_one_label_per_word(plain):
    plain._ensure_loaded()
    sentence = "Она хотела хлеба […] но она не нашла его?!.. «Нет», — сказала она."
    words, _ = _ruaccent_split_by_words(sentence)
    labels = plain._predict_word_labels(
        sentence, plain._stress_usage_sess, plain._stress_usage_tok, plain._stress_id2label,
    )
    assert len(labels) == len(words) == 17


def test_punctuation_run_does_not_change_the_marks(plain):
    # the same words, with a comma in place of the run: the marks after the
    # run are those the words get with aligned labels
    with_run = plain("Она хотела хлеба […] но она не нашла его.")
    with_comma = plain("Она хотела хлеба, но она не нашла его.")
    assert with_run.split()[4:] == with_comma.split()[3:]
