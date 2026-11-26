import pytest

import stressonnx

ACUTE = "́"


@pytest.mark.parametrize("word", ["արու", "հասու", "աղյուսե"])
def test_u_digraph_mark_follows_yiwn(word):
    out = stressonnx.stress(word, lang="hy")
    assert "ո" + ACUTE not in out
