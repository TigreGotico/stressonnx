"""Single download layer: resolve model files from the HuggingFace Hub cache."""
import logging
import os

from huggingface_hub import hf_hub_download

from stressonnx.errors import ModelDownloadError
from stressonnx.registry import HF_REPO_ID

LOG = logging.getLogger("stressonnx")

# ---------------------------------------------------------------------------
# HF download helpers
# ---------------------------------------------------------------------------

_LEGACY_CACHE = os.path.join(os.path.expanduser("~"), ".local", "share", "stressonnx")
_legacy_cache_notified = False


def _download_files(hf_subdir: str, filenames: list, cache_dir: str | None = None) -> dict:
    """Resolve model files for one HF subdir; return ``{relative_name: local_path}``.

    This is the single download layer — every backend obtains its files
    through the returned dict and never constructs model paths itself.

    With ``cache_dir=None`` (the default) files live in the standard Hugging
    Face cache: ``hf_hub_download`` handles reuse, ``HF_HOME`` relocation and
    ``HF_HUB_OFFLINE`` semantics, and models are shared with every other HF
    consumer on the machine.

    With an explicit ``cache_dir`` the invariant is
    ``local(f) = cache_dir / hf_subdir / f`` — exactly the tree layout of the
    HF repo, never ``cache_dir/hf_subdir/hf_subdir/f``.  Existing files are
    returned without touching the network.

    Raises :class:`ModelDownloadError` (chaining the hub exception) when a
    file cannot be fetched.
    """
    global _legacy_cache_notified
    if cache_dir is None and not _legacy_cache_notified and os.path.isdir(_LEGACY_CACHE):
        LOG.info(
            "Models now live in the standard Hugging Face cache; the old "
            "tree at %s is no longer used and can be deleted.", _LEGACY_CACHE
        )
        _legacy_cache_notified = True

    paths = {}
    for fname in filenames:
        hf_path = f"{hf_subdir}/{fname}"
        if cache_dir is not None:
            local = os.path.join(cache_dir, hf_subdir, fname)
            if os.path.exists(local):
                paths[fname] = local
                continue
        try:
            paths[fname] = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_path,
                local_dir=cache_dir,
            )
        except Exception as exc:
            raise ModelDownloadError(hf_subdir, hf_path, exc) from exc
    return paths
