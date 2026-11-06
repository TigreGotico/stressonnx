"""``ruaccent-silero``: the ruaccent pipeline with silero marks on the words it leaves unmarked.

The stress-usage model of ruaccent gates common function words (``или``,
``это``, ``было``) and the one-vowel head of ``что-то`` gets no mark from the
dictionary or the character model.  Expected marks come from the gold rows
of the scoring campaign: the Wiktionary example sentences (``wikt-ru:<n>``
row ids, with the number of rows that carry the same vowel), never from the
output of either model.
"""
import pytest

from stressonnx import MODEL_REGISTRY, STRESS_TOKEN, make_stressor, stress
from stressonnx.backends.ruaccent import RuAccentStressor

VOWELS = "аеёиоуыэюя"


def marked_ordinals(token: str) -> list:
    """Vowel ordinals that carry an acute in *token*."""
    out, seen = [], -1
    for ch in token:
        if ch == STRESS_TOKEN:
            out.append(seen)
        elif ch.lower() in VOWELS:
            seen += 1
    return out


def ordinals_by_token(text: str) -> dict:
    """Marked vowel ordinals per whitespace token, keyed by the token without marks or punctuation."""
    return {t.replace(STRESS_TOKEN, "").strip("—–,.!?;:«»…"): marked_ordinals(t) for t in text.split()}


# (Wiktionary example sentence, token, gold vowel ordinal): the row id is the
# one of the scoring campaign; ``ruaccent`` leaves the token unmarked in
# this sentence and Wiktionary marks the vowel given
GOLD = [
    ("Сегодня тепло, а вчера было холодно.", "было", 0),                       # wikt-ru:97
    ("Это будет трагедией, но, похоже, всё к этому и идёт.", "Это", 0),       # wikt-ru:143
    ("Господь его знает!", "его", 1),                                          # wikt-ru:1371
    ("Мне как-то не по себе.", "как-то", 0),                                  # wikt-ru:1072
    ("Я перешёл через улицу и вошёл вслед за стариком в кондитерскую.", "через", 0),  # wikt-ru:2122
    ("— Вы смелый зверь! — ответил Гудвин. — Вам недостаёт только веры в себя.", "только", 0),  # wikt-ru:5097
    ("— Она тебе не нравится? — Да нет, она неплохая, но что-то с ней странное происходит.", "что-то", 0),  # wikt-ru:433
    ("Сделав один или два поворота, герой наш очутился наконец перед самым домом, который показался теперь ещё печальнее.", "перед", 0),  # wikt-ru:382
    ("Между прочим, сека — самая блефовая игра из тех, что я знаю. Три карты в руке, а весь характер как на ладони.", "Между", 0),  # wikt-ru:1989
    ("Разговоры за столом велись приземлённые. В одном углу кто-то выгодно продал машину. В другом спорили о рецептах блюд из баклажанов.", "кто-то", 0),  # wikt-ru:2374
    ("– Вы себя кем считаете – русским или украинцем?", "или", 0),            # wikt-ru:254, first line
]


@pytest.fixture(scope="module")
def filled(model_cache_dir):
    return make_stressor(model="ruaccent-silero", cache_dir=model_cache_dir)


@pytest.fixture(scope="module")
def plain(model_cache_dir):
    return make_stressor(model="ruaccent", cache_dir=model_cache_dir)


@pytest.mark.parametrize("sentence,token,gold", GOLD)
def test_fill_marks_the_gold_vowel(filled, plain, sentence, token, gold):
    assert ordinals_by_token(plain(sentence))[token] == [], "ruaccent now marks this word; move it"
    assert ordinals_by_token(filled(sentence))[token] == [gold]


# (sentence from a gold row, token, gold vowel ordinal): silero marks both
# hyphen parts; the token keeps one mark.  Wiktionary writes чу̀ть-чу́ть,
# во̀т-во́т and Нью̀-Йо́рк (primary stress on the last part) and и́з-за,
# и́з-под (a compound preposition, stress on the first part).
HYPHEN_GOLD = [
    ("Не из-за меня же весь этот шум.", "из-за", 0),                       # ruslan:001781_RUSLAN; 109 of 118 gold rows
    ("Из-под асфальта доносился грохот сабвея.", "Из-под", 0),              # ruslan:005772_RUSLAN
    ("Еще один благополучно переправлен в Нью-Йорк.", "Нью-Йорк", 1),      # ruslan:000330_RUSLAN; 33 of 33 gold rows
    ("Как нарочно, это случилось в ту самую пору, когда страсть к законодательству приняла в нашем отечестве размеры чуть-чуть не опасные; канцелярии кипели уставами, как никогда не кипели сказочные реки млеком и мёдом, и каждый устав весил отнюдь не менее фунта.", "чуть-чуть", 1),  # wikt-ru:2640; 18 of 18
    ("Вдруг, как это часто бывает, звук гона мгновенно приблизился, как будто вот-вот перед ними самими были лающие рты собак и улюлюканье Данилы.", "вот-вот", 1),  # wikt-ru:21416; 3 of 3
]


@pytest.mark.parametrize("sentence,token,gold", HYPHEN_GOLD)
def test_hyphen_token_keeps_one_mark(filled, plain, sentence, token, gold):
    assert ordinals_by_token(plain(sentence))[token] == [], "ruaccent now marks this word; move it"
    assert ordinals_by_token(filled(sentence))[token] == [gold]


@pytest.mark.xfail(strict=True, reason="silero and the ruaccent character model both read моему́; Wiktionary and the gold rows write по-мо́ему")
def test_po_moemu_gold(filled):
    # ruslan:009867_RUSLAN, 014472, 021325: gold По-мо́ему (vowel 1 of 3)
    out = ordinals_by_token(filled("– Читал я твою юмореску в «Авроре». По-моему, говно… Мы просидели в ресторане часов до одиннадцати."))
    assert out["По-моему"] == [1]


def test_po_moemu_one_mark(filled):
    # the same rows: the token gets one mark, never По́-моему́
    out = ordinals_by_token(filled("– Читал я твою юмореску в «Авроре». По-моему, говно… Мы просидели в ресторане часов до одиннадцати."))
    assert len(out["По-моему"]) == 1


def test_ruaccent_marks_are_kept(filled, plain):
    # the omograph resolver and the dictionary decide these; the fill adds
    # nothing to a word that already carries a mark
    text = "старинный замок стоит на горе"
    plain_out, filled_out = ordinals_by_token(plain(text)), ordinals_by_token(filled(text))
    for token in ("старинный", "замок", "стоит", "горе"):
        assert plain_out[token] == filled_out[token] != []


def test_one_vowel_words_stay_as_ruaccent_left_them(filled):
    # на, он and дом have one vowel: the gold sets do not score them and
    # the fill leaves them alone, so a preposition is not marked as stressed
    out = ordinals_by_token(filled("он сел на пол в дом"))
    assert out["на"] == [] and out["он"] == [] and out["дом"] == []


def test_registry_entry():
    entry = MODEL_REGISTRY["ruaccent-silero"]
    assert entry.family == "ruaccent"
    assert entry.fill == "silero"
    assert entry.variant == MODEL_REGISTRY["ruaccent"].variant
    assert MODEL_REGISTRY["ruaccent"].fill is None
    s = make_stressor(model="ruaccent-silero")
    assert isinstance(s, RuAccentStressor)


def test_module_api_reaches_the_model(model_cache_dir):
    assert stress("или это", "ru", model="ruaccent-silero") == "и́ли э́то"
    assert stress("или это", "ru", model="ruaccent") == "или э́то"


def test_unknown_fill_is_rejected():
    with pytest.raises(ValueError):
        RuAccentStressor(fill="dictionary")
