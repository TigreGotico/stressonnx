"""stressonnx basic usage — getting started.

Models are downloaded automatically on first run.
"""
from stressonnx import stress, to_plus_notation, Stressor

# --- module-level function ---------------------------------------------------

# Russian (default model: ruaccent, homograph-aware)
print(stress("привет мир", "ru"))
# → приве́т мир

# Ukrainian — neural ONNX (silero)
print(stress("Привіт світ", "uk"))
# → Приві́т сві́т

# Belarusian — neural ONNX (silero)
print(stress("Прывітанне свет", "be"))
# → Прывіта́нне све́т

# Kazakh — vocabulary + rules
print(stress("Сәлем Қазақстан", "kk"))
# → Сәле́м Қазақста́н

# --- prefer=: capability instead of a model id --------------------------------

print(stress("красивый город", "ru", prefer="best"))    # ruaccent
print(stress("красивый город", "ru", prefer="fast"))    # silero
print(stress("красивый город", "ru", prefer="smallest"))  # simple

# --- notation ----------------------------------------------------------------

diacritic = stress("привет", "ru")          # приве́т  (combining acute U+0301)
plus_form = stress("привет", "ru", notation="plus")   # прив+ет

print(f"diacritic: {diacritic!r}")
print(f"plus:      {plus_form!r}")
print(f"convert:   {to_plus_notation(diacritic)!r}")   # same as plus_form

# --- Stressor class ----------------------------------------------------------

s = Stressor(lang="ru")
print(s.model)       # ruaccent
print(s.lang)        # ru
print(s.notation)    # diacritic

print(s("белок яйца полезен"))   # бело́к яйца́ поле́зен

# Stressor persists its model; reuse it across many calls
sentences = [
    "замок стоит на горе",
    "дверной замок надёжен",
    "острая коса лежала на лугу",
    "коса русалки длинная",
]
for sent in sentences:
    print(s(sent))
