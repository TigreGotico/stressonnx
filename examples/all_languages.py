"""Enumerate every supported language and print a stressed sample.

Useful for a quick smoke-test of all models after updating or re-exporting.
Models are downloaded automatically on first use.
"""
from stressonnx import ALL_LANGS, DEFAULT_MODEL, MODEL_REGISTRY, stress

# One representative word or phrase per language, keyed by canonical tag.
SAMPLES = {
    "ru":       "старинный замок стоит на горе",
    "uk":       "Привіт світ",
    "be":       "Прывітанне свет",
    "az-Cyrl":  "Бакы шəhəри",
    "az-Latn":  "Bakı şəhəri",
    "uz-Cyrl":  "Тошкент шаҳри",
    "uz-Latn":  "Toshkent shahri",
    "ba":       "Өфө ҡалаһы",
    "bg": "водата е студена",
    "mk": "Добро утро Македонија",
    "sl": "voda je mrzla",
    "lv": "Labdien, mani draugi",
    "cv":       "Шупашкар хули",
    "myv":      "Саранск ош",
    "hy":       "Բարեւ Երեւան",
    "ka":       "გამარჯობა თბილისი",
    "kk":       "Сәлем Қазақстан",
    "kbd":      "Налшык къалэ",
    "ky":       "Бишкек шаары",
    "kjh":      "Абакан хакас",
    "mdf":      "Саранск ош",
    "sah":      "Дьокуускай куорат",
    "tt":       "Казан шәһәре",
    "tg":       "Душанбе шаҳр",
    "udm":      "Ижевск кар",
    "xal":      "Элиста балhсн",
}

header = f"{'lang':<12}{'model':<12}{'input':<40}{'output'}"
print(header)
print("-" * len(header))

for lang in sorted(ALL_LANGS):
    sample = SAMPLES.get(lang, lang)
    model_id = DEFAULT_MODEL[lang]
    try:
        result = stress(sample, lang)
        print(f"{lang:<12}{model_id:<12}{sample:<40}{result}")
    except Exception as exc:
        print(f"{lang:<12}{model_id:<12}{sample:<40}ERROR: {exc}")
