"""Deterministic models on the latest-run meteogram (Open-Meteo forecast API)."""

import numpy as np
import pandas as pd

from utils import openmeteo_api as om
from utils.settings import (
    DETERMINISTIC_MODELS,
    ENSEMBLE_MODELS,
    GRID_STORES,
    SYSTEM_STYLES,
)

# Checked against the live API on 2026-10-04 (the docs' longer names are rejected).
VERIFIED_API_IDS = {"ecmwf_ifs", "ecmwf_aifs025_single", "gfs_seamless", "icon_seamless",
                    "meteofrance_seamless", "ukmo_seamless", "gem_seamless", "jma_seamless"}


def test_keys_never_collide_with_ensemble_systems():
    assert not set(DETERMINISTIC_MODELS) & set(ENSEMBLE_MODELS)


def test_api_ids_are_the_verified_ones():
    assert {m["api_model"] for m in DETERMINISTIC_MODELS.values()} == VERIFIED_API_IDS


def test_archived_twins_share_the_store_key_and_colour():
    for key, meta in DETERMINISTIC_MODELS.items():
        if meta.get("archive_store"):
            assert meta["archive_store"] == key and GRID_STORES[key]["kind"] == "deterministic"
        assert key in SYSTEM_STYLES


class _Var:
    def __init__(self, values):
        self._v = np.asarray(values, dtype=np.float32)

    def ValuesAsNumpy(self):
        return self._v

    def EnsembleMember(self):
        return 0


class _Block:
    def __init__(self, start, n, step_s, values):
        self.start, self.n, self.step, self.values = start, n, step_s, values

    def Time(self):
        return self.start

    def TimeEnd(self):
        return self.start + self.n * self.step

    def Interval(self):
        return self.step

    def Variables(self, i):
        return _Var(self.values)

    def VariablesLength(self):
        return 1


class _Response:
    def __init__(self, block):
        self.block = block

    def Daily(self):
        return self.block

    def Hourly(self):
        return self.block


START = int(pd.Timestamp("2026-10-04", tz="UTC").timestamp())


def test_one_request_for_all_models_in_order(monkeypatch):
    asked = []

    def weather_api(url, params):
        asked.append((url, params["models"]))
        return [_Response(_Block(START, 3, 86400, [10.0 + i, 11.0 + i, 12.0 + i])) for i in range(len(params["models"]))]

    monkeypatch.setattr(om._client, "weather_api", weather_api)
    out = om.fetch_deterministic(48.86, 2.35, "temperature_2m_max", ("ecmwf_ifs_hres", "jma"), 3)
    assert asked == [(om.FORECAST_URL, ["ecmwf_ifs", "jma_seamless"])]
    assert list(out["jma"]["value"]) == [11.0, 12.0, 13.0] and set(out["jma"]["member"]) == {0}
    assert {"date", "value", "member", "doy", "variable"} <= set(out["ecmwf_ifs_hres"].columns)


def test_a_model_returning_nothing_is_left_out_not_drawn_flat(monkeypatch):
    def weather_api(url, params):
        return [_Response(_Block(START, 3, 86400, [np.nan] * 3)), _Response(_Block(START, 3, 86400, [1.0, 2.0, 3.0]))]

    monkeypatch.setattr(om._client, "weather_api", weather_api)
    out = om.fetch_deterministic(48.86, 2.35, "temperature_2m_max", ("ecmwf_ifs_hres", "gfs"), 3)
    assert set(out) == {"gfs"}


def test_accumulated_windows_are_disjoint_totals(monkeypatch):
    # Hourly totals end at their time stamp: a UTC day is 01:00..00:00 next day,
    # so 2 complete days need 49 values from 00:00; the 00:00 value of day 1
    # belongs to the day before and is not counted.
    hourly = [5.0] + [1.0] * 48

    monkeypatch.setattr(om._client, "weather_api", lambda url, params: [_Response(_Block(START, 49, 3600, hourly))])
    out = om.fetch_deterministic_accumulated(48.86, 2.35, "precipitation_sum", ("gfs",), 2, 24)
    assert list(out["gfs"]["value"]) == [24.0, 24.0]
    assert list(out["gfs"]["date"]) == [pd.Timestamp("2026-10-05"), pd.Timestamp("2026-10-06")]


def test_an_incomplete_last_day_is_dropped_not_reported_short(monkeypatch):
    monkeypatch.setattr(om._client, "weather_api", lambda url, params: [_Response(_Block(START, 48, 3600, [1.0] * 48))])
    out = om.fetch_deterministic_accumulated(48.86, 2.35, "precipitation_sum", ("gfs",), 2, 24)
    assert list(out["gfs"]["value"]) == [24.0]


def test_validation_accepts_deterministic_keys():
    from utils.climatology import validate_ensemble_selection

    alerts, days, valid = validate_ensemble_selection(["ecmwf_ifs_hres", "meteofrance"], 48.86, 2.35, 7)
    assert valid == ["ecmwf_ifs_hres", "meteofrance"]
    # Each model keeps its own reach: the short one does not shorten the other.
    assert days == {"ecmwf_ifs_hres": 7, "meteofrance": DETERMINISTIC_MODELS["meteofrance"]["max_forecast_days"]}
    assert any("Météo-France" in a["message"] for a in alerts)


def test_meteogram_draws_a_deterministic_model_as_one_dashed_line():
    from utils.figures import make_meteogram_fig

    times = pd.date_range("2026-10-04", periods=3, freq="D")
    det = pd.DataFrame({"date": times, "member": 0, "value": [1.0, 2.0, 3.0]})
    fig = make_meteogram_fig({"ecmwf_ifs_hres": det}, {"ecmwf_ifs_hres": "ECMWF IFS HRES 9 km"}, {"label": "T", "unit": "°C"}, "Paris")
    assert len(fig.data) == 1 and fig.data[0].line.dash == "dash" and "deterministic" in fig.data[0].name
