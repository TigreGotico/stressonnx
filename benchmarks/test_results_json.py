import json

from results_json import records


def summary(tier, models):
    return {"lang": "xx", "set": "s", "tier": tier, "source": "u", "license": "l",
            "homographs_only": False, "models": models}


def test_only_gold_sets_become_records(tmp_path):
    result = {"n": 10, "correct": 7, "accuracy": 0.7, "ci95": [0.4, 0.9]}
    (tmp_path / "a__gold.summary.json").write_text(json.dumps(summary("GOLD", {"m": result})))
    (tmp_path / "a__silver.summary.json").write_text(json.dumps(summary("SILVER", {"m": result})))
    assert records(tmp_path) == [{
        "lang": "xx", "set": "s", "model": "m", "source": "u", "license": "l",
        "homographs_only": False, "targets": 10, "correct": 7, "accuracy": 0.7, "ci95": [0.4, 0.9],
    }]
