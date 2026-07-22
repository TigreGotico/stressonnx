# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StressonnxError(Exception):
    """Base class for all stressonnx errors."""


class UnsupportedLanguageError(StressonnxError, ValueError):
    """Raised when a language tag is not supported (by the library or a model).

    Subclasses :class:`ValueError` so callers written against the untyped API
    keep working.
    """

    def __init__(self, lang: str, supported) -> None:
        self.lang = lang
        self.supported = sorted(supported)
        super().__init__(
            f"Unsupported language {lang!r}.  Supported: {self.supported}."
        )


class ModelDownloadError(StressonnxError):
    """Raised when a model file cannot be fetched from the Hugging Face Hub.

    Wraps the underlying ``huggingface_hub`` exception (available as
    ``__cause__``) and names the exact repo path that failed so the fix is
    actionable: upload the missing file or pass a different ``model=``.
    """

    def __init__(self, model_id: str, hf_path: str, cause: Exception,
                 repo_id: str = "the model hub") -> None:
        self.model_id = model_id
        self.hf_path = hf_path
        super().__init__(
            f"Could not fetch {hf_path!r} from {repo_id!r} for model "
            f"{model_id!r} ({cause}).  Check network/HF_HUB_OFFLINE, upload "
            f"the missing file, or select another model via model=."
        )


class ModelLoadError(StressonnxError):
    """Raised when downloaded model files exist but cannot be loaded.

    Covers corrupt or truncated artefacts that fail at parse/session-creation
    time (bad gzip/JSON, unreadable ONNX graph …).  Participates in the
    ``stress(..., fallback=True)`` chain exactly like
    :class:`ModelDownloadError`, so a broken cache degrades instead of
    hard-failing.  The underlying exception is available as ``__cause__``.
    """

    def __init__(self, model_id: str, cause: Exception) -> None:
        self.model_id = model_id
        super().__init__(
            f"Model files for {model_id!r} are present but could not be "
            f"loaded ({cause}).  The cache may be corrupt — delete the "
            f"affected files and retry, or select another model via model=."
        )
