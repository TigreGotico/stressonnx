import pytest

import stressonnx


@pytest.mark.parametrize(("word", "want"), [
    ("արու", "արու\u0301"), ("հասու", "հասու\u0301"), ("աղյուսե", "աղյու\u0301սե"), ("լեզու", "լեզու\u0301"),
])
def test_u_digraph_mark_follows_yiwn(word, want):
    assert stressonnx.stress(word, lang="hy") == want
