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
