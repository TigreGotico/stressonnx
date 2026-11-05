"""Smoke tests for silero Russian stress backend."""
import pytest


def test_silero_ru_stress_basic():
    from stressonnx import stress
    result = stress("молоко", "ru", model="silero")
    assert "́" in result


def test_silero_ru_stress_sentence():
    from stressonnx import stress
    result = stress("красивый город", "ru", model="silero")
    assert "́" in result
    assert isinstance(result, str)


def test_silero_ru_plus_notation():
    from stressonnx import stress
    result = stress("привет", "ru", model="silero", notation="plus")
    assert "+" in result
    assert "́" not in result


def test_silero_ru_not_default():
    """ruaccent must remain the default for ru."""
    from stressonnx import DEFAULT_MODEL
    assert DEFAULT_MODEL["ru"] == "ruaccent"


def test_silero_ru_in_registry():
    from stressonnx import MODEL_REGISTRY
    assert "ru" in MODEL_REGISTRY["silero"].langs


def test_silero_ru_make_stressor():
    from stressonnx import make_stressor, _SileroStressor
    s = make_stressor(model="silero", lang="ru")
    assert isinstance(s, _SileroStressor)
    assert callable(s)


@pytest.fixture(scope="module")
def silero_ru(model_cache_dir):
    from stressonnx import make_stressor
    return make_stressor(model="silero", lang="ru", cache_dir=model_cache_dir)


# The reading each context calls for: за́мок "castle", замо́к "lock",
# всё ещё "still", все "everyone", му́ка "torment".
@pytest.mark.parametrize("text,reading", [
    ("старинный замок стоит на горе", "за́мок"),
    ("он открыл замок ключом", "замо́к"),
    ("все еще идет дождь", "всё́ ещё́"),
    ("все пришли домой", "все́ пришли́"),
    ("это была настоящая мука", "му́ка"),
])
def test_silero_ru_homograph_resolved_from_context(silero_ru, text, reading):
    assert reading in silero_ru(text)


def test_silero_ru_homograph_keeps_case(silero_ru):
    assert silero_ru("ВСЕ ЕЩЕ ИДЕТ ДОЖДЬ").startswith("ВСЁ́ ЕЩЁ́")


def _stale_ru_bundle(tmp_path, model_cache_dir, skip_stress, skip_yo):
    """A ru bundle as published before the homosolver: no resolver files,
    skip lists instead."""
    import gzip
    import json
    import shutil

    from stressonnx.download import _download_files
    from stressonnx.registry import _MAIN_FILES

    current = _download_files("ru", _MAIN_FILES, model_cache_dir)
    bundle = tmp_path / "ru"
    bundle.mkdir()
    for name in _MAIN_FILES:
        if name != "meta.json":
            shutil.copy(current[name], bundle / name)
    with open(current["meta.json"]) as fh:
        meta = json.load(fh)
    meta.pop("homosolver", None)
    (bundle / "meta.json").write_text(json.dumps(meta))
    for name, words in (("skip_stress_words.txt.gz", skip_stress),
                        ("skip_yo_words.txt.gz", skip_yo)):
        with gzip.open(bundle / name, "wt", encoding="utf-8") as fh:
            fh.write("\n".join(words))
    return str(tmp_path)


def test_silero_ru_bundle_without_homosolver_warns_and_skips(tmp_path, caplog, model_cache_dir):
    import logging

    from stressonnx import make_stressor

    cache = _stale_ru_bundle(tmp_path, model_cache_dir, ["замок", "все"], ["все"])
    with caplog.at_level(logging.WARNING, logger="stressonnx"):
        stale = make_stressor(model="silero", lang="ru", cache_dir=cache)
        out = stale("старинный замок стоит на горе, все пришли")
        make_stressor(model="silero", lang="ru", cache_dir=cache)("замок")
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "homosolver.onnx" in warnings[0] and f"delete {tmp_path}" in warnings[0]
    # skip-list words stay unstressed (все has one vowel and is still stressed)
    assert " замок " in out
    assert "все́" in out and "всё" not in out
    assert "стари́нный" in out


def test_silero_ru_bundle_with_homosolver_does_not_warn(caplog, model_cache_dir):
    import logging

    from stressonnx import make_stressor

    with caplog.at_level(logging.WARNING, logger="stressonnx"):
        out = make_stressor(model="silero", lang="ru", cache_dir=model_cache_dir)(
            "старинный замок стоит на горе")
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "за́мок" in out


def test_silero_ru_homosolver_chunks_match_one_batch(silero_ru):
    """The homosolver has no attention mask, so padding changes the logits.
    Every chunk must be padded to the width of the widest row of the whole
    batch, which here sits in the last chunk."""
    import numpy as np

    silero_ru._ensure_loaded()
    solver = silero_ru._homosolver
    rows = []
    classify = solver._classify
    solver._classify = lambda r: (rows.extend(r), classify(r))[1]
    try:
        for text in ["Он открыл замок.", "Где мука?", "Это мука."] * 25:
            solver.resolve(text)
        short_width = max(len(r) for r in rows)
        solver.resolve(
            "В старинном городе, где улицы узкие, а дома высокие и каменные, "
            "путники долго шли по дороге к горе и наконец увидели замок, "
            "который стоял над рекой, окружённый лесом, полями и озёрами.")
    finally:
        del solver._classify
    width = max(len(r) for r in rows)
    assert len(rows) > 64 and width > 4 * short_width
    assert max(len(r) for r in rows[:-32]) == short_width

    ids = np.full((len(rows), width), solver._pad_id, dtype=np.int64)
    for i, r in enumerate(rows):
        ids[i, : len(r)] = r
    (whole,) = solver._sess.run(None, {
        "input_ids": ids,
        "homo_start": np.array([r.index(solver._start_id) for r in rows]),
        "homo_end": np.array([r.index(solver._end_id) for r in rows])})

    session = solver._sess
    chunks = []

    class _Recorder:
        def run(self, names, feed):
            out = session.run(names, feed)
            chunks.append(out[0])
            return out

    solver._sess = _Recorder()
    try:
        picks = solver._classify(rows)
    finally:
        solver._sess = session
    chunked = np.concatenate(chunks)
    assert len(chunks) > 2
    np.testing.assert_allclose(chunked, whole, rtol=0, atol=1e-4 * np.abs(whole).max())
    assert picks == [int(v > 0) for v in whole[:, 0]]
