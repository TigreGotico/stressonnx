"""Structured stressing with analyze() — offsets into the original input.

``stress()`` returns a marked string; ``analyze()`` additionally returns a
``StressResult`` with one ``StressedWord`` per word, each carrying the
stressed-vowel offset and start/end span **into the original, untouched
input** — no need to re-parse the marked string to find a word.  This is the
recommended integration surface for TTS front-ends that reason about
individual words (e.g. aligning stress marks with phoneme spans).
"""
from stressonnx import analyze

# --- basic offsets -------------------------------------------------------

result = analyze("замок стоит на горе", "ru")

print("marked text:", result.text)          # same string stress() would return
print("original:   ", result.original)      # exactly as passed in
print()

for w in result.words:
    print(
        f"  {w.text!r:12} start={w.start:<3} end={w.end:<3} "
        f"stressed_index={w.stressed_index!r:<6} "
        f"stressed_char={w.stressed_char!r:<6} yo_restored={w.yo_restored}"
    )

# Every word's span slices the original input directly.
for w in result.words:
    assert result.original[w.start:w.end] == w.text

print()

# --- yo (ё) restoration --------------------------------------------------

# "зеленый" is conventionally typed with е even though it is pronounced with
# ё; ruaccent restores it and analyze() reports the restoration per word.
result = analyze("зеленый лес", "ru")
print("marked:", result.text)
for w in result.words:
    print(f"  {w.text!r:12} yo_restored={w.yo_restored}")

print()

# --- a homograph sentence, word by word ----------------------------------

result = analyze("старинный замок стоит на горе", "ru")
for w in result.words:
    mark = w.stressed_char or "(none)"
    print(f"  {w.text:<12} stressed vowel: {mark}")
