import numpy as np
import pandas as pd
import pytest

from utils.verification import fair_crps, match_obs, obs_window_total, scores_by_lead

INIT = pd.Timestamp("2026-09-20 00:00")


def _forecast(leads=(24, 48), members=(1.0, 3.0)):
    rows = []
    for lead in leads:
        for m, v in enumerate(members):
            rows.append({"init_time": INIT, "valid_time": INIT + pd.Timedelta(hours=lead),
                         "lead_hours": float(lead), "member": m, "value": v})
    return pd.DataFrame(rows)


def test_crps_of_one_member_is_absolute_error():
    assert fair_crps(np.array([3.0]), 1.0) == pytest.approx(2.0)


def test_crps_hand_value():
    # mean|x-y| = 0.5; sum_ij |xi-xj| = 2, / (2*2*1) = 0.5  ->  0
    assert fair_crps(np.array([0.0, 1.0]), 0.0) == pytest.approx(0.0)


def test_match_within_tolerance_and_not_beyond():
    fc = _forecast()
    obs = pd.DataFrame({"valid_time": pd.to_datetime(["2026-09-21 00:10", "2026-09-22 00:40"], utc=True),
                        "value": [2.0, 9.0]})
    m = match_obs(fc, obs, tolerance="15min")
    assert m.loc[m["lead_hours"] == 24, "obs"].eq(2.0).all()
    assert m.loc[m["lead_hours"] == 48, "obs"].isna().all()


def test_lead_without_obs_has_no_score_never_zero():
    fc = _forecast()
    obs = pd.DataFrame({"valid_time": pd.to_datetime(["2026-09-21 00:00"], utc=True), "value": [2.0]})
    scores = scores_by_lead(match_obs(fc, obs)).set_index("lead_hours")
    assert scores.loc[24.0, "n"] == 1
    assert 48.0 not in scores.index or (scores.loc[48.0, "n"] == 0 and np.isnan(scores.loc[48.0, "rmse"]))


def test_bias_is_forecast_minus_obs():
    fc = _forecast(leads=(24,), members=(4.0, 6.0))
    obs = pd.DataFrame({"valid_time": pd.to_datetime(["2026-09-21 00:00"], utc=True), "value": [2.0]})
    s = scores_by_lead(match_obs(fc, obs)).iloc[0]
    assert s["bias"] == pytest.approx(3.0) and s["rmse"] == pytest.approx(3.0)
    assert s["spread"] == pytest.approx(np.sqrt(2.0))


def test_window_with_a_missing_hour_is_missing():
    hours = pd.date_range("2026-09-20 01:00", "2026-09-21 00:00", freq="h", tz="UTC")
    obs = pd.DataFrame({"valid_time": hours, "value": 1.0})
    end = pd.Timestamp("2026-09-21 00:00", tz="UTC")
    assert obs_window_total(obs, [end], 24).iloc[0] == pytest.approx(24.0)
    assert np.isnan(obs_window_total(obs.drop(index=5), [end], 24).iloc[0])


def test_window_uses_last_report_in_each_hour():
    # Running totals within an hour (METAR, doc §IX): the last one is the hour's total.
    times = pd.to_datetime(["2026-09-21 00:20", "2026-09-21 00:40", "2026-09-21 00:55"], utc=True)
    obs = pd.DataFrame({"valid_time": times, "value": [0.2, 0.5, 0.8]})
    end = pd.Timestamp("2026-09-21 01:00", tz="UTC")
    assert obs_window_total(obs, [end], 1).iloc[0] == pytest.approx(0.8)


def test_match_across_time_resolutions():
    # Real data: forecast times are datetime64[ns] (xarray), parsed obs times
    # datetime64[us] (pandas). merge_asof refuses mixed resolutions.
    fc = _forecast()
    fc["valid_time"] = fc["valid_time"].astype("datetime64[ns]")
    obs = pd.DataFrame({"valid_time": pd.to_datetime(["2026-09-21 00:00"], utc=True).as_unit("us"), "value": [2.0]})
    m = match_obs(fc, obs)
    assert m.loc[m["lead_hours"] == 24, "obs"].eq(2.0).all()
