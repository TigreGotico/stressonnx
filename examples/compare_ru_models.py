"""Compare all three Russian stress backends on the same sentences.

Prints a table so differences in output (especially on homographs) are
immediately visible.
"""
from stressonnx import stress

SENTENCES = [
    "старинный замок стоит на горе",
    "дверной замок надёжен",
    "мука для хлеба",
    "мука была невыносима",
    "белок яйца очень полезен",
    "в лесу было много белок",
    "красивый город Москва",
    "молоко убежало из кастрюли",
    "Россия великая страна",
    "привет мир",
]

MODELS = ["ruaccent", "silero", "simple"]

col = 40
header = f"{'input':<{col}}" + "".join(f"{m:<{col}}" for m in MODELS)
print(header)
print("-" * len(header))

for sent in SENTENCES:
    outputs = {}
    for m in MODELS:
        try:
            outputs[m] = stress(sent, "ru", model=m)
        except Exception as exc:
            outputs[m] = f"ERROR: {exc}"
    row = f"{sent:<{col}}" + "".join(f"{outputs[m]:<{col}}" for m in MODELS)
    print(row)
