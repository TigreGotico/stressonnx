"""Russian homograph disambiguation — why ruaccent is the default for Russian.

Russian has many word pairs (and triples) that look identical in spelling but
carry different stress depending on grammatical meaning or context:

    замок  →  за́мок (castle)  vs  замо́к (lock)
    мука   →  му́ка (torment)  vs  мука́ (flour)
    белок  →  бе́лок (squirrel genitive)  vs  бело́к (protein / egg white)
    коса   →  коса́ (braid / scythe)  vs  ко́са (sandbank/spit)

All three Russian backends handle these differently:
- ruaccent: sentence-context BERT pipeline → correct disambiguation
- silero:   word-level MLP → no disambiguation, one-best prediction
- kubataba: sentence-level seq2seq Transformer → partial disambiguation
"""
from stressonnx import stress

HOMOGRAPH_PAIRS = [
    # (meaning_a_sentence, expected_a, meaning_b_sentence, expected_b, note)
    (
        "старинный замок стоит на горе",
        "стари́нный за́мок сто́ит на горе́",
        "дверной замок надёжен",
        "дверно́й замо́к надёжен",
        "замок: castle vs lock",
    ),
    (
        "мука для хлеба нужна",
        "мука́ для хле́ба нужна́",
        "мука была невыносима",
        "му́ка была́ невыноси́ма",
        "мука: flour vs torment",
    ),
    (
        "белок яйца полезен",
        "бело́к яйца́ поле́зен",
        "белок было много в лесу",
        None,   # model-dependent
        "белок: protein vs squirrel (genitive)",
    ),
    (
        # Dictionary stress: коса́ 'scythe' and 'braid' (Зализняк).  A ✗ here
        # is a genuine model miss, kept for honesty — see benchmarks/RESULTS.md
        # for the measured homograph accuracy.
        "острая коса лежала на лугу",
        "о́страя коса́ лежа́ла на лугу́",
        "коса русалки длинная",
        "коса́ руса́лки дли́нная",
        "коса: scythe / braid (both коса́)",
    ),
]

print("=== ruaccent (default, homograph-aware) ===\n")
for sent_a, exp_a, sent_b, exp_b, note in HOMOGRAPH_PAIRS:
    out_a = stress(sent_a, "ru", model="ruaccent")
    out_b = stress(sent_b, "ru", model="ruaccent")
    ok_a = "✓" if exp_a is None or out_a == exp_a else "✗"
    ok_b = "✓" if exp_b is None or out_b == exp_b else "✗"
    print(f"  [{note}]")
    print(f"  {ok_a}  {out_a}")
    print(f"  {ok_b}  {out_b}")
    print()

print("=== silero (neural, word-level, no homograph resolution) ===\n")
for sent_a, _, sent_b, _, note in HOMOGRAPH_PAIRS:
    out_a = stress(sent_a, "ru", model="silero")
    out_b = stress(sent_b, "ru", model="silero")
    print(f"  [{note}]")
    print(f"     {out_a}")
    print(f"     {out_b}")
    print()

print("=== kubataba (seq2seq Transformer, sentence-level) ===\n")
for sent_a, _, sent_b, _, note in HOMOGRAPH_PAIRS:
    out_a = stress(sent_a, "ru", model="kubataba")
    out_b = stress(sent_b, "ru", model="kubataba")
    print(f"  [{note}]")
    print(f"     {out_a}")
    print(f"     {out_b}")
    print()
