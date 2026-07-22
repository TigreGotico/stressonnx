"""Enumerate every supported language and print a stressed sample.

Useful for a quick smoke-test of all models after updating or re-exporting.
Models are downloaded automatically on first use.
"""
from stressonnx import ALL_LANGS, DEFAULT_MODEL, MODEL_REGISTRY, stress

# One representative word or phrase per language.
SAMPLES = {
    "ru":       "старинный замок стоит на горе",
    "ukr":      "Привіт світ",
    "bel":      "Прывітанне свет",
    "aze_cyr":  "Бакы шəhəri",
    "aze_lat":  "Bakı şəhəri",
    "uzb_cyr":  "Тошкент шаҳри",
    "uzb_lat":  "Toshkent shahri",
    "bak":      "Өфө ҡалаһы",
    "bel_simple": "свет і цемра",
    "bul": "водата е студена",
    "mkd": "Добро утро Македонија",
    "slv": "voda je mrzla",
    "lav": "Labdien, mani draugi",
    "ru_simple": "вода холодная",
    "ukr_simple": "вода холодна",
    "chv":      "Шупашкар хули",
    "erz":      "Саранск ош",
    "hye":      "Բարեւ Երեւան",
    "kat":      "გამარჯობა თბილისი",
    "kaz":      "Сәлем Қазақстан",
    "kbd":      "Налшык къалэ",
    "kir":      "Бишкек шаары",
    "kjh":      "Абакан хакас",
    "mdf":      "Саранск ош",
    "sah":      "Дьокуускай куорат",
    "tat":      "Казан шәһәре",
    "tgk":      "Душанбе шаҳр",
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
