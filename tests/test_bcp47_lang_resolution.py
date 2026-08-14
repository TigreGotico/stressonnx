"""Regression tests: BCP-47 regional/case/separator variants must resolve
to the canonical language tag everywhere a lang lookup happens.

No network access and no ONNX session is created: backend constructors are
lazy (the model files are only fetched on first call), so instantiating a
stressor and checking its class/``lang`` attribute is enough to exercise the
full resolution path without downloading anything.
"""
import pytest

from stressonnx.errors import UnsupportedLanguageError
from stressonnx.langs import resolve_lang
from stressonnx.registry import ALL_LANGS, DEFAULT_MODEL, LANG_SCRIPT, lang_to_script
from stressonnx.stressor import make_stressor


# ---------------------------------------------------------------------------
# resolve_lang: the shared normalizer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag,expected", [
    ("ru-RU", "ru"),
    ("uk-UA", "uk"),
    ("kk-KZ", "kk"),
    ("ka-GE", "ka"),
    ("lv-LV", "lv"),
    ("tt", "tt"),
    ("az-Cyrl", "az-Cyrl"),
    ("az-cyrl", "az-Cyrl"),
    ("uz-Latn-UZ", "uz-Latn"),
    ("uz_Latn_UZ", "uz-Latn"),
    ("RU-ru", "ru"),
    ("ru_RU", "ru"),
])
def test_resolve_lang_regional_and_case_variants(tag, expected):
    assert resolve_lang(tag, ALL_LANGS) == expected


def test_resolve_lang_bare_script_ambiguous_tag_unresolved():
    """'az' alone must not silently guess a script."""
    with pytest.raises(UnsupportedLanguageError):
        resolve_lang("az", ALL_LANGS)


def test_resolve_lang_unsupported_tag_raises_typed_error():
    with pytest.raises(UnsupportedLanguageError) as exc_info:
        resolve_lang("xx-XX", ALL_LANGS)
    assert exc_info.value.lang == "xx-XX"
    assert "xx-XX" in str(exc_info.value)


def test_resolve_lang_accepts_dict_supported():
    # LANG_SCRIPT is a dict keyed by canonical tag, not a bare set.
    assert resolve_lang("ru-RU", LANG_SCRIPT) == "ru"


# ---------------------------------------------------------------------------
# make_stressor: the reported production failure
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag,canonical", [
    ("ru-RU", "ru"),
    ("uk-UA", "uk"),
    ("kk-KZ", "kk"),
    ("ka-GE", "ka"),
    ("lv-LV", "lv"),
    ("tt", "tt"),
])
def test_make_stressor_accepts_bcp47_regional_tags(tag, canonical):
    s = make_stressor(lang=tag)
    assert s.lang == canonical if hasattr(s, "lang") else True


def test_make_stressor_az_cyrl_exact_still_works():
    from stressonnx.backends.simple import SimpleStressor
    s = make_stressor(model="simple", lang="az-Cyrl")
    assert isinstance(s, SimpleStressor)
    assert s.lang == "az-Cyrl"


def test_make_stressor_uz_latn_regional_resolves_to_script_variant():
    from stressonnx.backends.simple import SimpleStressor
    s = make_stressor(model="simple", lang="uz-Latn-UZ")
    assert isinstance(s, SimpleStressor)
    assert s.lang == "uz-Latn"


def test_make_stressor_bare_az_raises_unsupported():
    with pytest.raises(UnsupportedLanguageError):
        make_stressor(lang="az")


def test_make_stressor_unsupported_region_raises_unsupported():
    with pytest.raises(UnsupportedLanguageError):
        make_stressor(lang="xx-XX")


# ---------------------------------------------------------------------------
# lang_to_script: same normalization must apply
# ---------------------------------------------------------------------------

def test_lang_to_script_regional_tag():
    from stressonnx.registry import Script
    assert lang_to_script("ru-RU") == Script.CYRILLIC
    assert lang_to_script("ka-GE") == Script.GEORGIAN


def test_lang_to_script_unsupported_raises():
    with pytest.raises(UnsupportedLanguageError):
        lang_to_script("xx-XX")


# ---------------------------------------------------------------------------
# Backend constructors normalize independently too
# ---------------------------------------------------------------------------

def test_silero_backend_accepts_regional_tag():
    from stressonnx.backends.silero import _SileroStressor
    b = _SileroStressor(lang="uk-UA")
    assert b.lang == "uk"


def test_simple_backend_accepts_underscore_separator():
    from stressonnx.backends.simple import SimpleStressor
    b = SimpleStressor(lang="kk_KZ")
    assert b.lang == "kk"


# ---------------------------------------------------------------------------
# DEFAULT_MODEL sanity: canonical tags are the source of truth
# ---------------------------------------------------------------------------

def test_default_model_keys_are_canonical_tags_only():
    assert "ru-RU" not in DEFAULT_MODEL
    assert "ru" in DEFAULT_MODEL


# ---------------------------------------------------------------------------
# F1 — non-str input keeps the typed-error contract (not AttributeError)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", [None, 5, 3.14, ["ru"]])
def test_resolve_lang_non_str_raises_typed_error(bad):
    with pytest.raises(UnsupportedLanguageError):
        resolve_lang(bad, ALL_LANGS)


def test_lang_to_script_none_raises_typed_error():
    with pytest.raises(UnsupportedLanguageError):
        lang_to_script(None)


def test_lang_to_script_int_raises_typed_error():
    with pytest.raises(UnsupportedLanguageError):
        lang_to_script(5)


def test_simple_stressor_lang_none_raises_typed_error():
    from stressonnx.backends.simple import SimpleStressor
    with pytest.raises(UnsupportedLanguageError):
        SimpleStressor(lang=None)


def test_lang_to_script_none_is_also_plain_valueerror():
    # examples/typed_errors_fallback.py relies on catching UnsupportedLanguageError
    # as a plain ValueError too.
    with pytest.raises(ValueError):
        lang_to_script(None)


# ---------------------------------------------------------------------------
# F2 — a present-but-wrong script subtag must not fall through to the bare
# language (which may use a different, wrong, script).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tag", ["ru-Latn", "kk-Latn", "uk-Latn", "uz-Arab"])
def test_resolve_lang_wrong_script_subtag_raises_not_bare_fallback(tag):
    with pytest.raises(UnsupportedLanguageError):
        resolve_lang(tag, ALL_LANGS)


def test_resolve_lang_correct_script_subtag_with_region_still_resolves():
    assert resolve_lang("az-Cyrl-AZ", ALL_LANGS) == "az-Cyrl"


def test_resolve_lang_region_not_script_still_resolves_to_bare():
    # "RU" is a 2-letter region subtag, not a 4-letter script subtag, so the
    # bare-language fallback still applies here.
    assert resolve_lang("ru-RU", ALL_LANGS) == "ru"


def test_make_stressor_wrong_script_subtag_raises():
    with pytest.raises(UnsupportedLanguageError):
        make_stressor(lang="ru-Latn")


# ---------------------------------------------------------------------------
# F3 — the pipeline call site: _resolve, _chain, and stress() must all see
# the normalized tag, not just make_stressor.
# ---------------------------------------------------------------------------

def test_pipeline_resolve_normalizes_regional_tag():
    from stressonnx.pipeline import StressPipeline
    p = StressPipeline()
    lang, model, resolved = p._resolve("ru-RU", None, None)
    assert lang == "ru"
    assert model is None
    assert resolved == DEFAULT_MODEL["ru"]


def test_pipeline_resolve_prefer_fast_routes_to_silero_for_regional_tag():
    from stressonnx.pipeline import StressPipeline
    from stressonnx.registry import MODEL_REGISTRY
    p = StressPipeline()
    lang, model, resolved = p._resolve("ru-RU", None, "fast")
    assert lang == "ru"
    assert resolved == "silero"
    assert "ru" in MODEL_REGISTRY["silero"].langs


def test_pipeline_chain_nonempty_after_resolving_regional_tag():
    from stressonnx.pipeline import StressPipeline
    p = StressPipeline()
    lang, _model, _resolved = p._resolve("ru-RU", None, None)
    chain = p._chain(lang)
    assert chain
    assert all(entry_lang == "ru" for _model_id, entry_lang in chain)


def test_pipeline_stress_routes_regional_tag_to_normalized_backend(monkeypatch):
    """stress() with a regional tag must reach the backend keyed by the
    canonical tag — verified with a stub backend so no model is downloaded.
    """
    import stressonnx.pipeline as pipeline_mod

    calls = []

    def fake_make_stressor(model=None, lang=None, cache_dir=None):
        calls.append((model, lang))
        return lambda text: text.upper()

    monkeypatch.setattr(pipeline_mod, "make_stressor", fake_make_stressor)

    p = pipeline_mod.StressPipeline()
    result = p.stress("hello", lang="ru-RU")

    assert result == "HELLO"
    assert calls, "make_stressor was never called"
    _model, called_lang = calls[0]
    assert called_lang == "ru"
