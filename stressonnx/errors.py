# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class StressonnxError(Exception):
    """Base class for all stressonnx errors."""


class UnsupportedLanguageError(StressonnxError, ValueError):
    """Raised when a language tag is not supported (by the library or a model).

    An unsupported tag is an invalid argument value, so this is also a
    :class:`ValueError`.  ``model`` names the model that rejected the tag,
    or is ``None`` when no stressonnx model serves it.
    """

    def __init__(self, lang: str, supported, model: str | None = None) -> None:
        self.lang = lang
        self.supported = sorted(supported)
        self.model = model
        prefix = (f"Unsupported language {lang!r}" if model is None
                  else f"Model {model!r} does not support language {lang!r}")
        super().__init__(f"{prefix}.  Supported: {self.supported}.")


class UnknownModelError(StressonnxError, ValueError):
    """Raised when a model id, or a model variant of a backend, is unknown.

    An unknown name is an invalid argument value, so this is also a
    :class:`ValueError`.  ``model`` holds the rejected name and
    ``supported`` the names that exist.
    """

    def __init__(self, model: str, supported) -> None:
        self.model = model
        self.supported = sorted(supported)
        super().__init__(f"Unknown model {model!r}.  Available models: {self.supported}.")


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
            f"loaded ({cause}).  The files may be corrupt, or an installed "
            f"dependency may be incompatible with them.  Check the cause, "
            f"delete corrupt files and retry, or select another model via "
            f"model=."
        )
