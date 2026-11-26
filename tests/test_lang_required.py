"""``lang`` has no default: text in an unnamed language is an error, not Russian."""
import pytest

import stressonnx
from stressonnx import StressPipeline


@pytest.mark.parametrize("call", [
    lambda: stressonnx.stress("молоко"),
    lambda: stressonnx.stress_batch(["молоко"]),
    lambda: stressonnx.analyze("молоко"),
    lambda: StressPipeline().stress("молоко"),
    lambda: StressPipeline().stress_batch(["молоко"]),
    lambda: StressPipeline().analyze("молоко"),
])
def test_a_call_without_lang_is_a_type_error(call):
    with pytest.raises(TypeError, match="lang"):
        call()
