"""Parity tests against the reference silero_stress library (requires the
``export`` extra: torch + silero_stress).  Run by the dedicated CI job, not
the default matrix."""
import csv
import io
import urllib.request

import numpy as np
import pytest

from stressonnx import make_stressor, stress, to_plus_notation
from stressonnx._common import _UNSTRESSED_HYPHEN_CLITICS

# russtress (Ponomareva et al. 2017) carries no license: fetched, never committed
RUSSTRESS_RU = "https://raw.githubusercontent.com/MashaPo/russtress/master/data/data_ru.csv"

#: float32 has 2**-23 ≈ 1.2e-7 relative precision, and the deepest dot
#: product of the accentor heads sums 128 terms.  onnxruntime and torch pick
#: different GEMM kernels per CPU (AVX2, AVX-512), so the summation order
#: differs; 2 * 128 * 2**-23 ≈ 3e-5 of a row's largest |logit| bounds that
#: reordering error.  Measured on Zen 5: at most 2.2e-7 of it.  Logits reach
#: 7e5, so a fixed absolute tolerance does not work.
HEAD_TOLERANCE = 3e-5

#: The homosolver is deeper (three BERT layers, sums of up to 600 terms,
#: softmax, LayerNorm, GELU); 1e-4 of the largest logit, against a measured
#: 4e-7.  The decision is the logit's sign, checked where |logit| exceeds
#: the tolerance.
HOMOSOLVER_TOLERANCE = 1e-4


def plus_to_acute(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "+" and i + 1 < len(s):
            out.append(s[i + 1] + "\u0301")
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


@pytest.fixture(scope="module")
def reference_ru():
    from silero_stress import load_accentor
    return load_accentor("ru")


@pytest.fixture(scope="module")
def russtress_windows(reference_ru):
    """(all windows with a homograph, every 10th window without one).

    Windows with a hyphen clitic other than -то are left out: stressonnx
    keeps -нибудь/-либо/-таки/-ка unstressed where upstream stresses them
    (see ``_UNSTRESSED_HYPHEN_CLITICS``).
    """
    with urllib.request.urlopen(RUSSTRESS_RU) as resp:
        rows = [r["string"] for r in csv.DictReader(io.StringIO(resp.read().decode("utf-8")))]
    homo = reference_ru.homosolver
    clitics = tuple("-" + c for c in _UNSTRESSED_HYPHEN_CLITICS - {"то"})
    with_homograph, without = [], []
    for window in rows:
        if any(c in window for c in clitics):
            continue
        tagged = homo._find_and_tag_homos(window)
        (with_homograph if any(t[4] for t in tagged) else without).append(window)
    return with_homograph, without[::10]


def test_stress_simple_matches_silero():
    """Exact match against SimpleAccentor for a sample of langs (plus notation)."""
    from stressonnx import stress, to_plus_notation
    from silero_stress.simple_accentor import SimpleAccentor

    # (stressonnx tag, silero_stress code).  Georgian is deliberately
    # absent: for OOV words stressonnx applies the antepenultimate rule
    # (Akhvlediani/Gudava/Aronson), which the upstream vocabulary itself
    # follows on 100% of its entries, while upstream's own OOV fallback
    # contradicts its vocabulary on 55% of them — see tests/test_oov_rules.py
    # and benchmarks/oov_rules_eval.py.
    sample = {
        ("kk", "kaz"): "Сәлем Қазақстан",
        ("tt", "tat"): "Сәлам Казан",
        ("hy", "hye"): "Բարեւ Երեւան",
        ("az-Latn", "aze_lat"): "Salam Bakı",
        ("uz-Latn", "uzb_lat"): "Salom Toshkent",
        ("sah", "sah"): "Дорообо Дьокуускай",
    }
    for (lang, ref_code), sentence in sample.items():
        ref = SimpleAccentor(lang=ref_code)(sentence)  # silero emits + notation
        got = to_plus_notation(stress(sentence, lang))  # convert to + for comparison
        assert ref == got, f"[{lang}] ref={ref!r} got={got!r}"


def test_silero_ru_yo_parity():
    """The ported yo-capable ru decode must match the reference accentor."""
    from silero_stress import load_accentor

    ref = load_accentor("ru")

    sentences = [
        "зеленый лес шумит",
        "теплый прием",
        "мой котенок пьет молоко",
        "самолет летит высоко",
        "красивый город",
        "ёжик в тумане",
    ]
    for s in sentences:
        assert stress(s, "ru", model="silero") == plus_to_acute(ref(s)), s


def test_silero_ru_homosolver_parity_russtress(russtress_windows, reference_ru, model_cache_dir):
    """The ported ru pipeline (homosolver + accentor) matches upstream exactly."""
    with_homograph, without = russtress_windows
    assert len(with_homograph) >= 200 and len(without) >= 200
    ours = make_stressor(model="silero", lang="ru", cache_dir=model_cache_dir)
    windows = with_homograph + without
    differ = [(w, ours(w), plus_to_acute(reference_ru(w))) for w in windows]
    differ = [d for d in differ if d[1] != d[2]]
    assert differ == [], f"{len(differ)} of {len(windows)} windows differ"


def test_silero_ru_homosolver_inputs_and_logits(russtress_windows, reference_ru, model_cache_dir):
    """WordPiece ids match upstream; ONNX logits match the scripted model."""
    import torch

    from stressonnx.backends._silero_homosolver import HomoSolver

    homo = reference_ru.homosolver
    ours = make_stressor(model="silero", lang="ru", cache_dir=model_cache_dir)
    ours._ensure_loaded()
    solver = ours._homosolver
    assert isinstance(solver, HomoSolver)
    checked = 0
    for window in russtress_windows[0]:
        rows = []
        for start, end, word, word_lower, raw_mark, _ in homo._find_and_tag_homos(window):
            if raw_mark is None or word_lower not in homo.homodict:
                continue
            ids = homo.tokenizer(raw_mark)
            assert solver._tok.encode(raw_mark) == ids, raw_mark
            rows.append(ids)
        if not rows:
            continue
        width = max(len(r) for r in rows)
        ids = np.array([r + [homo.tokenizer.pad_token_id] * (width - len(r)) for r in rows])
        starts = np.array([r.index(homo.tokenizer.homo_start_id) for r in rows])
        ends = np.array([r.index(homo.tokenizer.homo_end_id) for r in rows])
        (got,) = solver._sess.run(None, {"input_ids": ids, "homo_start": starts, "homo_end": ends})
        with torch.no_grad():
            want = homo.model(torch.tensor(ids), torch.tensor(starts), torch.tensor(ends)).numpy()
        tol = HOMOSOLVER_TOLERANCE * max(1.0, float(np.abs(want).max()))
        np.testing.assert_allclose(got, want, rtol=0, atol=tol)
        clear = np.abs(want) > tol
        assert ((got > 0) == (want > 0))[clear].all()
        checked += len(rows)
    assert checked >= 200


@pytest.mark.parametrize("lang,code", [("ru", "ru"), ("uk", "ukr"), ("be", "bel")])
def test_silero_accentor_matches_upstream(lang, code, model_cache_dir):
    """Embedding table, n-grams, exceptions and ONNX heads equal upstream's."""
    import torch
    from silero_stress import load_accentor

    ref = load_accentor(code).accentor
    ours = make_stressor(model="silero", lang=lang, cache_dir=model_cache_dir)
    ours._ensure_loaded()
    np.testing.assert_array_equal(ours._W, ref.model.embedding.weight.detach().numpy())
    assert ours._ngram_dict == dict(ref.model.embedding.ngram_dict)
    assert ours._exceptions == {w: tuple(v) for w, v in ref.exceptions.items()}
    # pseudo-words glued from two real 4-grams: pooled vectors in the range
    # the heads see in use
    grams = sorted(g for g in ours._ngram_dict if len(g) == 4 and g.isalpha())
    words = [a + b for a, b in zip(grams[::7], grams[3::7])][:2000]
    pooled = ours._pool(words)
    got = ours._sess.run(None, {"pooled": pooled})
    with torch.no_grad():
        want = [ref.model.stress_clf(torch.tensor(pooled)).numpy()]
        if code == "ru":
            want.append(ref.model.yo_clf(torch.tensor(pooled)).numpy())
    assert len(got) == len(want)
    for g, w in zip(got, want):
        tol = HEAD_TOLERANCE * np.abs(w).max(axis=1, keepdims=True)
        assert (np.abs(g - w) <= tol).all()
        # the decision: argmax agrees wherever the top two logits are apart
        top2 = np.sort(w, axis=1)[:, -2:]
        clear = top2[:, 1] - top2[:, 0] > 2 * tol[:, 0]
        assert clear.mean() > 0.9
        assert (g.argmax(1) == w.argmax(1))[clear].all()
