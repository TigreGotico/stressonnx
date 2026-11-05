"""ruaccent-big-poetry: the ruaccent pipeline with the RUAccent big_poetry omograph model.

The expected sentences are the output of ruaccent 1.5.8.3,
``RUAccent().load(omograph_model_size=..., use_dictionary=True,
tiny_mode=False).process_all(text)`` with each ``+`` turned into U+0301
after its vowel.  They are parity values, not gold stress: each sentence is
one where upstream big_poetry and turbo3.1 choose different readings.  The
expected token ids are the output of the transformers ``BertTokenizerFast``
that upstream loads for big_poetry.

``STRESSONNX_TEST_CACHE_DIR`` points the stressors at a local model tree
(``<dir>/ru_ruaccent/<file>``) so a bundle can be tested before it is
published; unset, the bundle comes from the pinned Hugging Face revision.
"""
import os

import pytest

import stressonnx
from stressonnx import MODEL_REGISTRY, RuAccentStressor, make_stressor
from stressonnx.backends import ruaccent

CACHE_DIR = os.environ.get("STRESSONNX_TEST_CACHE_DIR")

# (input, upstream big_poetry, upstream turbo3.1)
CASES = [
    ("девушка заплела длинную косу.",
     "де́вушка заплела́ дли́нную косу́.", "де́вушка заплела́ дли́нную ко́су."),
    ("крестьянин точил острую косу.",
     "крестья́нин точи́л о́струю косу́.", "крестья́нин точи́л о́струю ко́су."),
    ("бабушка купила вкусный ирис.",
     "ба́бушка купи́ла вку́сный ири́с.", "ба́бушка купи́ла вку́сный и́рис."),
    ("атлас дорог лежал на полке.",
     "атла́с доро́г лежа́л на по́лке.", "а́тлас до́рог лежа́л на по́лке."),
    ("двери закрыли на замок.",
     "две́ри закры́ли на замо́к.", "двери́ закры́ли на замо́к."),
    ("стрелки часов показывали полдень.",
     "стрелки́ часо́в пока́зывали по́лдень.", "стре́лки часо́в пока́зывали по́лдень."),
]
CASE_IDS = ["kosu-braid", "kosu-scythe", "iris", "atlas", "dveri", "strelki"]


@pytest.fixture(scope="module")
def big_poetry():
    return make_stressor(model="ruaccent-big-poetry", lang="ru", cache_dir=CACHE_DIR)


@pytest.fixture(scope="module")
def turbo():
    return make_stressor(model="ruaccent", lang="ru", cache_dir=CACHE_DIR)


@pytest.mark.parametrize("text,expected,_", CASES, ids=CASE_IDS)
def test_big_poetry_matches_upstream(big_poetry, text, expected, _):
    assert big_poetry(text) == expected


@pytest.mark.parametrize("text,_,expected", CASES, ids=CASE_IDS)
def test_default_model_keeps_turbo(turbo, text, _, expected):
    assert turbo(text) == expected


def test_big_poetry_pair_has_segment_ids(big_poetry):
    big_poetry._ensure_loaded()
    tok = big_poetry._omograph_tok
    text = "двери закрыли на  <w>замок</w>."
    ids, attention_mask = tok.encode(text, pair="зам+ок")
    assert ids == [101, 4932, 23178, 660, 120138, 10556, 120139, 126, 102, 3785, 250, 1157, 102]
    assert attention_mask == [1] * 13
    assert tok.type_ids(text, "зам+ок") == [0] * 9 + [1] * 4


def test_only_the_selected_omograph_model_is_fetched(monkeypatch):
    requested = []

    def fake_download(hf_subdir, filenames, cache_dir=None, model_id=None):
        requested.extend(filenames)
        raise stressonnx.ModelDownloadError(model_id, f"{hf_subdir}/{filenames[0]}", OSError("offline"))

    monkeypatch.setattr(ruaccent, "_download_files", fake_download)
    with pytest.raises(stressonnx.ModelDownloadError):
        make_stressor(model="ruaccent-big-poetry", lang="ru")("замок")
    omograph_files = [f for f in requested if f.startswith("nn_omograph")]
    assert omograph_files == ["nn_omograph_big_poetry/model.onnx",
                              "nn_omograph_big_poetry/tokenizer.json"]


def test_registry_selects_the_omograph_model():
    assert MODEL_REGISTRY["ruaccent"].variant == "turbo3.1"
    assert MODEL_REGISTRY["ruaccent-big-poetry"].variant == "big_poetry"
    assert MODEL_REGISTRY["ruaccent-big-poetry"].family == "ruaccent"
    assert stressonnx.DEFAULT_MODEL["ru"] == "ruaccent"


def test_unknown_omograph_model_is_a_typed_error():
    with pytest.raises(stressonnx.UnknownModelError) as exc_info:
        RuAccentStressor(omograph="medium_poetry")
    assert exc_info.value.model == "medium_poetry"
    assert exc_info.value.supported == ["big_poetry", "turbo3.1"]
    assert isinstance(exc_info.value, ValueError)


def test_unknown_model_id_is_a_typed_error():
    with pytest.raises(stressonnx.UnknownModelError) as exc_info:
        make_stressor(model="ruaccent-medium-poetry", lang="ru")
    assert exc_info.value.model == "ruaccent-medium-poetry"
    assert "ruaccent-big-poetry" in exc_info.value.supported
