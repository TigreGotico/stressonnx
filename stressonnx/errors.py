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
    actionable: upload the missing file to ``TigreGotico/stressonnx-models``
    or pass a different ``model=``.
    """

    def __init__(self, model_id: str, hf_path: str, cause: Exception) -> None:
        from stressonnx.registry import HF_REPO_ID  # avoid circular import

        self.model_id = model_id
        self.hf_path = hf_path
        super().__init__(
            f"Could not fetch {hf_path!r} from {HF_REPO_ID!r} for model "
            f"{model_id!r} ({cause}).  Check network/HF_HUB_OFFLINE, upload "
            f"the missing file, or select another model via model=."
        )
