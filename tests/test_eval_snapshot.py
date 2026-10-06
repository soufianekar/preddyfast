import json

import pytest

from utils import eval_snapshot as es


def test_precipitation_is_never_verified_against_a_t0():
    refs = es.usable_references("precipitation_24h")
    assert set(refs) == {"stations", "imerg"}


def test_temperature_uses_stations_and_both_t0_references_not_imerg():
    refs = es.usable_references("temperature_2m")
    assert "stations" in refs and "control_t0" in refs and "own_t0" in refs and "imerg" not in refs


def test_brier_offered_only_where_thresholds_exist():
    assert any(o["value"].startswith("brier_") for o in es.score_options("precipitation_24h"))
    assert not any(o["value"].startswith("brier_") for o in es.score_options("temperature_2m"))


@pytest.fixture
def _dir(tmp_path, monkeypatch):
    monkeypatch.setattr(es, "PRECOMPUTE_DIR", str(tmp_path))
    return tmp_path


def test_index_and_figures_read_back(_dir):
    assert es.load_index() == {"stations": {}}
    assert es.load_figure("X", "temperature_2m", "stations", "crps") is None
    rel = es.figure_path("X", "temperature_2m", "stations", "crps")
    es._write(rel, json.dumps({"data": [], "layout": {"title": {"text": "t"}}}))
    es._write("eval/index.json", json.dumps({"stations": {"X": {}}}))
    assert es.load_figure("X", "temperature_2m", "stations", "crps")["layout"]["title"]["text"] == "t"
    assert es.load_index()["stations"] == {"X": {}}
