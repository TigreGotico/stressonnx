"""Contract tests for the single download layer (``_download_files``).

The invariants proven here are what keeps every backend working from a cold
cache:

* files are resolved through the returned dict — never by hand-joined paths;
* an explicit ``cache_dir`` uses the layout ``cache_dir/<hf_subdir>/<file>``
  and never nests the subdir twice;
* warm loads perform zero network calls;
* hub failures surface as :class:`ModelDownloadError` naming the repo path.
"""
import os

import pytest

import stressonnx.download as download
from stressonnx import ModelDownloadError
from stressonnx.backends.ruaccent import RuAccentStressor
from stressonnx.download import _download_files


def test_explicit_cache_dir_layout_no_double_nesting(tmp_path):
    """Files land at cache_dir/<subdir>/<file> — the HF repo tree, once."""
    paths = _download_files("kaz", ["meta.json"], cache_dir=str(tmp_path))
    expected = tmp_path / "kaz" / "meta.json"
    assert paths["meta.json"] == str(expected)
    assert expected.is_file()
    assert not (tmp_path / "kaz" / "kaz").exists()


def test_explicit_cache_dir_offline_reuse(tmp_path, monkeypatch):
    """A second call with the same cache_dir never touches the network."""
    first = _download_files("kaz", ["meta.json"], cache_dir=str(tmp_path))

    def _no_network(*args, **kwargs):
        raise AssertionError("hf_hub_download called on a warm cache")

    monkeypatch.setattr(download, "hf_hub_download", _no_network)
    second = _download_files("kaz", ["meta.json"], cache_dir=str(tmp_path))
    assert second == first


def test_default_uses_standard_hf_cache():
    """Without cache_dir, paths come from the shared HF cache (HF_HOME)."""
    paths = _download_files("kaz", ["meta.json"])
    path = paths["meta.json"]
    assert os.path.isfile(path)
    hf_home = os.environ.get("HF_HOME", os.path.join(os.path.expanduser("~"), ".cache", "huggingface"))
    assert os.path.realpath(path).startswith(os.path.realpath(hf_home))
    legacy = os.path.join(os.path.expanduser("~"), ".local", "share", "stressonnx")
    assert not path.startswith(legacy)


def test_download_failure_raises_typed_error(monkeypatch, tmp_path):
    def _boom(*args, **kwargs):
        raise OSError("simulated network failure")

    monkeypatch.setattr(download, "hf_hub_download", _boom)
    with pytest.raises(ModelDownloadError) as excinfo:
        _download_files("kaz", ["meta.json"], cache_dir=str(tmp_path / "cold"))
    err = excinfo.value
    assert err.model_id == "kaz"
    assert err.hf_path == "kaz/meta.json"
    assert "kaz/meta.json" in str(err)
    assert isinstance(err.__cause__, OSError)


def test_ruaccent_consumes_dict_no_hand_joined_paths():
    """RuAccentStressor must obtain every model file from the download dict.

    Guards against the regression where ``_ensure_loaded`` ignored the
    returned dict and joined paths by hand (which broke every fresh install
    via double-nested downloads).
    """
    import inspect

    src = inspect.getsource(RuAccentStressor._load)
    assert "os.path.join" not in src
    assert 'data["' in src


def test_ruaccent_warm_load_is_offline(monkeypatch):
    """After a first load, a fresh instance loads from cache with no network."""
    RuAccentStressor()._ensure_loaded()  # warm the cache

    real = download.hf_hub_download

    def _local_only(*args, **kwargs):
        kwargs["local_files_only"] = True  # any network need → LocalEntryNotFoundError
        return real(*args, **kwargs)

    monkeypatch.setattr(download, "hf_hub_download", _local_only)
    stressor = RuAccentStressor()
    stressor._ensure_loaded()  # must succeed from the shared HF cache alone
    assert stressor._loaded
