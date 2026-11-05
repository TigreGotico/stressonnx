import os

import pytest

import stressonnx


@pytest.fixture(autouse=True)
def _clear_failure_cooldown():
    """Reset the negative-failure cache around every test.

    ``stress()`` remembers failed (lang, model) pairs for a cooldown window
    so real consumers don't hammer a failing download; tests that simulate
    outages must not poison the tests that follow them.
    """
    stressonnx._RECENT_FAILURES.clear()
    yield
    stressonnx._RECENT_FAILURES.clear()


@pytest.fixture(scope="session")
def model_cache_dir():
    """Model directory for tests that build a stressor directly.

    ``STRESSONNX_TEST_CACHE_DIR`` points them at a local tree laid out like
    the model repository (``<dir>/<hf_subdir>/<file>``), which tests model
    files before they are published; unset, models come from the pinned
    Hugging Face revision.
    """
    return os.environ.get("STRESSONNX_TEST_CACHE_DIR")
