"""Tests for the typed data model: ModelEntry, StressNotation, StressorBackend."""
import pytest


# ---------------------------------------------------------------------------
# ModelEntry
# ---------------------------------------------------------------------------

def test_model_entry_importable():
    from stressonnx import ModelEntry
    assert ModelEntry is not None


def test_model_entry_fields():
    from stressonnx import ModelEntry, Script
    e = ModelEntry(
        langs=frozenset({"ru"}),
        family="test",
        hf_subdir="test_sub",
        description="A test entry",
        input_scripts=frozenset({Script.CYRILLIC}),
    )
    assert e.langs == frozenset({"ru"})
    assert e.family == "test"
    assert e.hf_subdir == "test_sub"
    assert e.description == "A test entry"
    assert e.input_scripts == frozenset({Script.CYRILLIC})


def test_model_entry_frozen():
    from stressonnx import ModelEntry, Script
    e = ModelEntry(
        langs=frozenset({"ru"}), family="test", hf_subdir=None, description="",
        input_scripts=frozenset({Script.CYRILLIC}),
    )
    with pytest.raises(Exception):   # FrozenInstanceError
        e.family = "changed"


def test_model_entry_langs_is_frozenset():
    from stressonnx import MODEL_REGISTRY
    for entry in MODEL_REGISTRY.values():
        assert isinstance(entry.langs, frozenset)


def test_model_entry_family_is_str():
    from stressonnx import MODEL_REGISTRY
    for entry in MODEL_REGISTRY.values():
        assert isinstance(entry.family, str)


def test_model_entry_description_nonempty():
    from stressonnx import MODEL_REGISTRY
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.description, f"{model_id} entry has empty description"


def test_model_entry_registry_all_have_hf_subdir_or_none():
    from stressonnx import MODEL_REGISTRY
    for model_id, entry in MODEL_REGISTRY.items():
        assert entry.hf_subdir is None or isinstance(entry.hf_subdir, str)


def test_model_entry_equality():
    from stressonnx import ModelEntry, Script
    kw = dict(langs=frozenset({"ru"}), family="f", hf_subdir=None, description="d",
              input_scripts=frozenset({Script.CYRILLIC}))
    assert ModelEntry(**kw) == ModelEntry(**kw)


# ---------------------------------------------------------------------------
# StressorBackend Protocol
# ---------------------------------------------------------------------------

def test_stressor_backend_importable():
    from stressonnx import StressorBackend
    assert StressorBackend is not None


def test_make_stressor_returns_backend():
    from stressonnx import make_stressor, StressorBackend
    s = make_stressor(model="simple", lang="kk")
    assert isinstance(s, StressorBackend)


def test_custom_callable_satisfies_protocol():
    from stressonnx import StressorBackend

    class MyStressor:
        def __call__(self, text: str) -> str:
            return text

    assert isinstance(MyStressor(), StressorBackend)


def test_non_callable_does_not_satisfy_protocol():
    from stressonnx import StressorBackend

    class NotAStressor:
        pass

    assert not isinstance(NotAStressor(), StressorBackend)


def test_factory_backends_satisfy_protocol():
    from stressonnx import make_stressor, StressorBackend
    assert isinstance(make_stressor(lang="kk"), StressorBackend)


def test_ruaccent_stressor_satisfies_protocol():
    from stressonnx import RuAccentStressor, StressorBackend
    assert isinstance(RuAccentStressor, type)
    # Protocol check on the instance (lazy — not loaded yet)
    # We can check the class satisfies __call__ structurally
    assert hasattr(RuAccentStressor, "__call__")


# ---------------------------------------------------------------------------
# Public __all__ completeness
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Script enum and lang_to_script
# ---------------------------------------------------------------------------

def test_script_enum_values():
    from stressonnx import Script
    assert Script.CYRILLIC == "cyrillic"
    assert Script.LATIN == "latin"
    assert Script.ARMENIAN == "armenian"
    assert Script.GEORGIAN == "georgian"


def test_script_is_str():
    from stressonnx import Script
    assert isinstance(Script.CYRILLIC, str)


def test_lang_to_script_cyrillic():
    from stressonnx import lang_to_script, Script
    for lang in ("ru", "uk", "be", "kk", "tt"):
        assert lang_to_script(lang) == Script.CYRILLIC, lang


def test_lang_to_script_latin():
    from stressonnx import lang_to_script, Script
    assert lang_to_script("az-Latn") == Script.LATIN
    assert lang_to_script("uz-Latn") == Script.LATIN


def test_lang_to_script_armenian():
    from stressonnx import lang_to_script, Script
    assert lang_to_script("hy") == Script.ARMENIAN


def test_lang_to_script_georgian():
    from stressonnx import lang_to_script, Script
    assert lang_to_script("ka") == Script.GEORGIAN


def test_lang_to_script_unknown():
    from stressonnx import lang_to_script, UnsupportedLanguageError
    # ValueError catch must keep working (UnsupportedLanguageError subclasses it)
    with pytest.raises(ValueError, match="Unsupported language"):
        lang_to_script("xx")
    with pytest.raises(UnsupportedLanguageError):
        lang_to_script("xx")


def test_lang_script_covers_all_langs():
    from stressonnx import LANG_SCRIPT, ALL_LANGS
    missing = ALL_LANGS - set(LANG_SCRIPT)
    assert not missing, f"LANG_SCRIPT missing entries for: {missing}"


def test_model_entry_input_scripts_field():
    from stressonnx import MODEL_REGISTRY, Script
    for name, entry in MODEL_REGISTRY.items():
        assert isinstance(entry.input_scripts, frozenset), name
        for s in entry.input_scripts:
            assert isinstance(s, Script), f"{name}: {s!r} is not a Script"


def test_default_model_script_consistency():
    """Every language's script is accepted by its default model."""
    from stressonnx import ALL_LANGS, DEFAULT_MODEL, MODEL_REGISTRY, lang_to_script
    for lang in ALL_LANGS:
        script = lang_to_script(lang)
        entry = MODEL_REGISTRY[DEFAULT_MODEL[lang]]
        assert script in entry.input_scripts, (
            f"{lang}: script {script.value} not in "
            f"{DEFAULT_MODEL[lang]}.input_scripts"
        )


def test_lang_script_importable():
    from stressonnx import LANG_SCRIPT, Script
    assert isinstance(LANG_SCRIPT, dict)
    assert all(isinstance(v, Script) for v in LANG_SCRIPT.values())


# ---------------------------------------------------------------------------
# Public __all__ completeness
# ---------------------------------------------------------------------------

def test_all_exports_present():
    import stressonnx
    for name in stressonnx.__all__:
        assert hasattr(stressonnx, name), f"{name!r} in __all__ but not importable"


def test_typed_api_in_all():
    import stressonnx
    for name in ("ModelEntry", "StressNotation", "StressorBackend",
                 "Script", "lang_to_script", "LANG_SCRIPT"):
        assert name in stressonnx.__all__, f"{name} missing from __all__"
