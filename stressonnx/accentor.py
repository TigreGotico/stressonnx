"""Deprecated module kept for backward compatibility — import from stressonnx instead."""
import warnings

from stressonnx.errors import (  # noqa: F401
    StressonnxError,
    UnsupportedLanguageError,
    ModelDownloadError,
)
from stressonnx.registry import (  # noqa: F401
    Script,
    StressNotation,
    StressorBackend,
    ModelEntry,
    MODEL_REGISTRY,
    LANG_SCRIPT,
    lang_to_script,
    DEFAULT_MODEL,
    RUACCENT_LANGS,
    MAIN_LANGS,
    SIMPLE_LANGS,
    ALL_LANGS,
    HF_REPO_ID,
    _KUBATABA_FILES,
    _MAIN_FILES,
    _SIMPLE_FILES,
    _OOV_RULES,
)
from stressonnx.download import (  # noqa: F401
    LOG,
    _LEGACY_CACHE,
    _legacy_cache_notified,
    _download_files,
)
from stressonnx._common import (  # noqa: F401
    _softmax,
    _RU_VOWELS,
    _RE_SPLIT,
    _RE_RU_COND,
    _UNSTRESSED_HYPHEN_CLITICS,
)
from stressonnx.notation import (  # noqa: F401
    STRESS_TOKEN,
    _insert_stress,
    _apply_notation,
    _plus_to_diacritic,
    _apostrophe_to_diacritic,
    _RU_VOWELS_SET,
)
from stressonnx.stressor import Stressor, make_stressor  # noqa: F401
from stressonnx.backends.silero import (  # noqa: F401
    _SileroStressor,
    _load_ngram_dict,
    _load_exceptions,
    _load_set,
)
from stressonnx.backends.simple import SimpleStressor, _load_vocab  # noqa: F401
from stressonnx.backends.kubataba import _KubatabaStressor  # noqa: F401
from stressonnx.backends.ruaccent import (  # noqa: F401
    RuAccentStressor,
    _RUACCENT_FILES,
    _RE_RUACCENT_NORM,
    _PUNC_CHARS,
    _RU_SPLIT_RE,
    _ruaccent_norm,
    _delete_spaces_before_punc,
    _fix_capital,
    _ruaccent_split_by_words,
    _ruaccent_split_by_sentences,
)

warnings.warn(
    "stressonnx.accentor is deprecated; import from 'stressonnx' or its submodules",
    DeprecationWarning,
    stacklevel=2,
)
