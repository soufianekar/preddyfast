"""Daily max/min from forecasts: the model's own window extremes where it has them."""

import numpy as np
import pandas as pd

from utils.climatology import daily_extreme

INIT = pd.Timestamp("2026-10-01 00:00")


def _series(step, hours, values, member=0):
    leads = np.arange(step, hours + step, step, dtype=float) if values is not None else None
    return pd.DataFrame({"date": INIT + pd.to_timedelta(leads, unit="h"), "lead_hours": leads,
                         "member": member, "value": values})


def test_window_maxima_build_a_utc_day_from_windows_ending_03_to_24():
    # 3-hourly window maxima over two days; the 00 UTC stamp of day 2 closes day 1.
    vals = [10, 12, 15, 20, 22, 18, 14, 11] + [9, 10, 13, 17, 19, 16, 12, 30.0]
    df = _series(3, 48, np.array(vals, dtype=float))
    out = daily_extreme(df, "max", step_type="max")
    assert out["date"].dt.strftime("%m-%d").tolist() == ["10-01", "10-02"]
    assert out["value"].tolist() == [22.0, 30.0]  # 30 is stamped 03 Oct 00:00, i.e. (21, 24] of 02 Oct


def test_instantaneous_samples_give_the_max_of_samples_within_the_day():
    vals = [10, 12, 15, 20, 22, 18, 14, 11] + [9, 10, 13, 17, 19, 16, 12, 30.0]
    df = pd.concat([pd.DataFrame({"date": [INIT], "lead_hours": [0.0], "member": [0], "value": [8.0]}),
                    _series(3, 48, np.array(vals, dtype=float))], ignore_index=True)
    out = daily_extreme(df, "max", step_type="instant")
    # Day 1 = samples at 00..21 UTC of 01 Oct (8, 10, ..., 14); 11 at 24:00 belongs to 02 Oct.
    assert out["value"].tolist()[0] == 22.0 and out["date"].dt.strftime("%m-%d").tolist()[0] == "10-01"


def test_incomplete_days_are_dropped_never_reported_from_part_of_the_day():
    df = _series(6, 30, np.array([5, 9, 7, 6, 4.0]))  # one full day + 6 h of the next
    out = daily_extreme(df, "min", step_type="min")
    assert out["date"].dt.strftime("%m-%d").tolist() == ["10-01"] and out["value"].tolist() == [5.0]


def test_each_member_is_its_own_series():
    a = _series(6, 24, np.array([1, 2, 3, 4.0]), member=0)
    b = _series(6, 24, np.array([5, 6, 7, 8.0]), member=1)
    out = daily_extreme(pd.concat([a, b]), "max", step_type="max").sort_values("member")
    assert out["value"].tolist() == [4.0, 8.0]


def test_widening_steps_still_need_the_whole_day():
    # 3-hourly to T+12 then 6-hourly: windows (0,3],(3,6],(6,9],(9,12],(12,18],(18,24].
    leads = np.array([3, 6, 9, 12, 18, 24.0])
    df = pd.DataFrame({"date": INIT + pd.to_timedelta(leads, unit="h"), "lead_hours": leads, "member": 0,
                       "value": [1, 2, 3, 4, 9, 5.0]})
    assert daily_extreme(df, "max", step_type="max")["value"].tolist() == [9.0]


def test_samples_that_stop_early_in_the_day_are_incomplete():
    leads = np.array([0, 6, 12, 18, 24, 30.0])  # day 2 has only 00 and 06 UTC
    df = pd.DataFrame({"date": INIT + pd.to_timedelta(leads, unit="h"), "lead_hours": leads, "member": 0,
                       "value": [1, 2, 3, 4, 50, 60.0]})
    out = daily_extreme(df, "max", step_type="instant")
    assert out["date"].dt.strftime("%m-%d").tolist() == ["10-01"]


def test_past_meteogram_daily_max_uses_window_maxima_where_the_store_has_them(monkeypatch):
    from utils import earthmover_api as em

    read = []

    def fake(lat, lon, variable, init_time, accum, store):
        read.append((store, variable))
        leads = np.arange(6, 49, 6, dtype=float)
        df = pd.DataFrame({"valid_time": INIT + pd.to_timedelta(leads, unit="h"), "lead_hours": leads,
                           "member": 0, "value": [10, 20, 15, 12, 11, 25, 14, 9.0]})
        return df, {"init_time": INIT}

    monkeypatch.setattr(em, "fetch_point_ensemble", fake)
    monkeypatch.setattr(em, "store_steps", lambda s: "6 h")
    monkeypatch.setattr(em, "store_variables", lambda s: ["temperature_2m", "maximum_temperature_2m"] if s == "gefs" else ["temperature_2m"])
    df, var_meta, meta = em.archive_meteogram(48.85, 2.35, "temperature_2m_max", "2026-10-01T00:00", 2, None, store="gefs")
    assert read[-1] == ("gefs", "maximum_temperature_2m") and meta["daily_source"] == "window"
    assert df["value"].tolist() == [20.0, 25.0] and "daily" in var_meta["label"].lower()
    df, _, meta = em.archive_meteogram(48.85, 2.35, "temperature_2m_max", "2026-10-01T00:00", 2, None, store="ifs_ens")
    assert read[-1] == ("ifs_ens", "temperature_2m") and meta["daily_source"] == "samples"


def test_climate_subtitle_says_how_daily_max_is_made():
    from utils.figures import _subtitle
    from utils.settings import VARIABLES

    text = _subtitle("ERA5-Seamless", "1991–2020", VARIABLES["temperature_2m_max"])
    assert "hourly analysed 2 m temperature" in text
    assert "hourly" not in _subtitle("ERA5-Seamless", "1991–2020", VARIABLES["precipitation_sum"])


def test_a_lone_midnight_sample_is_not_a_day():
    leads = np.array([0, 6, 12, 18, 24.0])  # day 2 has only its 00 UTC sample
    df = pd.DataFrame({"date": INIT + pd.to_timedelta(leads, unit="h"), "lead_hours": leads, "member": 0,
                       "value": [1, 2, 3, 4, 13.0]})
    assert daily_extreme(df, "max", step_type="instant")["date"].dt.strftime("%m-%d").tolist() == ["10-01"]
