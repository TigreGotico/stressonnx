"""End-to-end verification: stressonnx output vs silero reference.

Usage::

    python stressonnx/export/verify_e2e.py --lang ru
    python stressonnx/export/verify_e2e.py --lang ukr
    python stressonnx/export/verify_e2e.py --lang kaz   # simple_accentor
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from stressonnx import stress
from stressonnx.accentor import MAIN_LANGS, SIMPLE_LANGS

# ---------------------------------------------------------------------------
SENTENCES = {
    "ru": [
        "Привет мир",
        "Молоко убежало из кастрюли.",
        "Котёнок играет с клубком ниток.",
        "Кому-то нравится зима, а кому-то лето.",
        "В лесу родилась ёлочка, в лесу она росла.",
        "Учёные открыли новую планету.",
        "Дети пошли в школу первого сентября.",
        "Он работает программистом в большой компании.",
        "Солнце светит ярко над городом.",
        "Мы поехали на дачу в выходные.",
    ],
    "ukr": [
        "Привіт світ",
        "Молоко втекло з каструлі.",
        "Кошеня грається з клубком ниток.",
        "Вода тече в морі.",
        "Голова болить після роботи.",
        "Замок стоїть на горі.",
        "Сонце світить яскраво над містом.",
        "Ми поїхали на дачу у вихідні.",
        "Діти пішли до школи першого вересня.",
        "Він працює програмістом у великій компанії.",
    ],
    "bel": [
        "Прывітанне свет",
        "Малако ўцякло з каструлі.",
        "Кацяня гуляе з клубком ніцяў.",
        "Вада цячэ ў моры.",
        "Галава баліць пасля працы.",
        "Замак стаіць на гары.",
        "Сонца свеціць ярка над горадам.",
        "Мы паехалі на дачу ў выходныя.",
        "Дзеці пайшлі ў школу першага верасня.",
        "Ён працуе праграмістам у вялікай кампаніі.",
    ],
    "kaz": [
        "Сәлем дүние",
        "Мектеп оқушысы",
        "Қазақстан халқы",
        "Алматы қаласы",
        "Ана тілі",
        "Ел мен жер",
        "Жаңа жыл",
        "Мектеп кітапханасы",
        "Су ішу",
        "Жер бетінде",
    ],
    "tat": [
        "Сәлам дөнья",
        "Казан каласы",
        "Татар теле",
        "Мәктәп укучысы",
        "Яңа ел",
        "Ил белән халык",
        "Су эчү",
        "Тау итәге",
        "Кояш чыга",
        "Урман аша",
    ],
    "kir": [
        "Саламаттыкпы",
        "Бишкек шаары",
        "Кыргыз тили",
        "Мектеп окуучусу",
        "Жаңы жыл",
        "Эл менен жер",
        "Суу ичүү",
        "Тоо этеги",
        "Күн чыгат",
        "Токой аша",
    ],
    "aze_lat": [
        "Salam dünya",
        "Bakı şəhəri",
        "Azərbaycan dili",
        "Məktəb şagirdi",
        "Yeni il",
        "El və torpaq",
        "Su içmək",
        "Dağ ətəyi",
        "Günəş doğur",
        "Meşə içi",
    ],
    "hye": [
        "Բարեւ աշխարհ",
        "Երեւան քաղաք",
        "Հայոց լեզու",
        "Դպրոցի աշակերտ",
        "Նոր տարի",
        "Ժողովուրդ",
        "Ջուր խմել",
        "Լեռ ստորոտ",
        "Արեւ ծագում",
        "Անտառ ճամփա",
    ],
    "kat": [
        "გამარჯობა სამყარო",
        "თბილისი ქალაქი",
        "ქართული ენა",
        "სკოლის მოსწავლე",
        "ახალი წელი",
        "ხალხი",
        "წყლის დალევა",
        "მთის ძირი",
        "მზე ამოდის",
        "ტყის გზა",
    ],
}

# Default sentences for langs not in the table above.
_DEFAULT_SENTS = [
    "Сәлем дүние",
    "Мектеп оқушысы",
    "Жаңа жыл мерекесі",
    "Елімізді сүйеміз",
    "Тауда серуендедік",
    "Күн жарқын жылы",
    "Балалар ойнады",
    "Мектеп кітапханасы",
    "Су ішу пайдалы",
    "Ел бірлігі",
]


def get_reference(lang, sentence):
    if lang in MAIN_LANGS:
        from silero_stress import load_accentor
        acc = load_accentor(lang)
        return acc(sentence)
    else:
        from silero_stress.simple_accentor import SimpleAccentor
        # bel_simple → bel for SimpleAccentor
        slang = lang.removesuffix("_simple") if lang.endswith("_simple") else lang
        acc = SimpleAccentor(lang=slang)
        return acc(sentence)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", required=True)
    args = parser.parse_args()
    lang = args.lang

    sents = SENTENCES.get(lang, _DEFAULT_SENTS)
    n = len(sents)
    ok = 0
    mismatches = []

    for s in sents:
        ref = get_reference(lang, s)
        got = stress(s, lang)
        if ref == got:
            ok += 1
        else:
            mismatches.append((s, ref, got))

    print(f"[{lang}] EXACT MATCH: {ok}/{n} = {100.0*ok/n:.1f}%")
    if mismatches:
        print("\n--- MISMATCHES ---")
        for s, r, o in mismatches:
            print(f"  IN : {s}")
            print(f"  REF: {r}")
            print(f"  OUT: {o}")
