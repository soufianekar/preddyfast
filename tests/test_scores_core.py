"""Per-run scores, their aggregation by lead, the WMO set, and common-lead sampling."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils.verification import (
    aggregate_scores,
    common_leads,
    fair_crps,
    per_run_scores,
    sample_leads,
    scores_by_lead,
)

INIT = pd.Timestamp("2026-09-20 00:00")


def _matched(values_by_run: dict, obs_by_run: dict, lead=24.0):
    rows = []
    for init, values in values_by_run.items():
        for m, v in enumerate(values):
            rows.append({"init_time": init, "valid_time": init + pd.Timedelta(hours=lead), "lead_hours": lead,
                         "member": m, "value": v, "obs": obs_by_run[init]})
    return pd.DataFrame(rows)


def test_per_run_row_carries_every_additive_part():
    m = _matched({INIT: [4.0, 6.0]}, {INIT: 2.0})
    row = per_run_scores(m).iloc[0]
    assert row["n_members"] == 2 and row["err"] == pytest.approx(3.0)
    assert row["abs_err"] == pytest.approx(3.0) and row["sq_err"] == pytest.approx(9.0)
    assert row["var"] == pytest.approx(2.0) and row["obs"] == pytest.approx(2.0)


def test_aggregate_equals_the_old_scores_by_lead():
    runs = {INIT: [1.0, 3.0, 4.0], INIT + pd.Timedelta("1D"): [0.0, 2.0, 2.5]}
    m = _matched(runs, {INIT: 2.0, INIT + pd.Timedelta("1D"): 1.0})
    new = aggregate_scores(per_run_scores(m)).set_index("lead_hours")
    old = scores_by_lead(m).set_index("lead_hours")
    for col in ("n", "bias", "mae", "rmse", "crps", "spread", "spread_skill"):
        assert new.loc[24.0, col] == pytest.approx(old.loc[24.0, col])


def test_sd_of_error_is_the_wmo_standard_deviation_of_errors():
    runs = {INIT: [3.0], INIT + pd.Timedelta("1D"): [5.0]}
    agg = aggregate_scores(per_run_scores(_matched(runs, {INIT: 2.0, INIT + pd.Timedelta("1D"): 2.0}))).iloc[0]
    # errors 1 and 3: ME 2, SD sqrt(2) (n-1), RMSE sqrt(5) = sqrt(ME^2 + SD^2 (n-1)/n)
    assert agg["bias"] == pytest.approx(2.0) and agg["sd_error"] == pytest.approx(np.sqrt(2.0))
    assert agg["rmse"] == pytest.approx(np.sqrt(5.0))


def test_brier_is_squared_probability_error_at_or_above_threshold():
    runs = {INIT: [0.0, 2.0, 6.0], INIT + pd.Timedelta("1D"): [0.0, 1.0, 6.0]}
    m = _matched(runs, {INIT: 0.0, INIT + pd.Timedelta("1D"): 5.0})
    rows = per_run_scores(m, thresholds=(1.0,)).set_index("init_time")
    assert rows.loc[INIT, "brier_1"] == pytest.approx((2 / 3) ** 2)  # 2 of 3 members >= 1, none observed
    assert rows.loc[INIT + pd.Timedelta("1D"), "brier_1"] == pytest.approx((1 / 3) ** 2)  # 1.0 counts as >= 1
    agg = aggregate_scores(rows.reset_index(), thresholds=(1.0,)).iloc[0]
    assert agg["brier_1"] == pytest.approx(((2 / 3) ** 2 + (1 / 3) ** 2) / 2)


def test_fair_crps_matches_the_scores_library():
    from scores.probability import crps_for_ensemble

    rng = np.random.default_rng(1)
    members, y = rng.normal(size=51), 0.3
    ref = crps_for_ensemble(xr.DataArray(members, dims="member"), xr.DataArray(y), "member", method="fair")
    assert fair_crps(members, y) == pytest.approx(float(ref))


def test_brier_matches_the_scores_library():
    from scores.probability import brier_score

    rng = np.random.default_rng(2)
    values = rng.gamma(1.0, 3.0, size=(20, 11))
    obs = rng.gamma(1.0, 3.0, size=20)
    inits = [INIT + pd.Timedelta(days=i) for i in range(20)]
    m = _matched({t: v for t, v in zip(inits, values)}, {t: o for t, o in zip(inits, obs)})
    ours = aggregate_scores(per_run_scores(m, thresholds=(5.0,)), thresholds=(5.0,)).iloc[0]["brier_5"]
    p = xr.DataArray((values >= 5.0).mean(axis=1), dims="run")
    ref = brier_score(p, xr.DataArray((obs >= 5.0).astype(float), dims="run"))
    assert ours == pytest.approx(float(ref))


def test_common_leads_on_a_step_every_model_has():
    a = pd.DataFrame({"lead_hours": [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 48.0]})
    b = pd.DataFrame({"lead_hours": [12.0, 24.0, 36.0, 48.0, 60.0]})
    assert common_leads({"a": a, "b": b}, 12) == [12.0, 24.0, 36.0, 48.0]
    assert common_leads({"a": a, "b": b}, 24) == [24.0, 48.0]
    assert common_leads({"a": a}, 6) == [6.0, 12.0, 18.0, 24.0, 30.0, 36.0, 48.0]
    assert common_leads({"a": a, "b": b.iloc[:0]}, 12) == [12.0, 24.0, 36.0, 48.0]  # an empty model does not veto


def test_sample_leads_keeps_only_the_common_ones():
    df = pd.DataFrame({"lead_hours": [6.0, 12.0, 24.0], "n": [1, 2, 3]})
    assert sample_leads(df, [12.0, 24.0])["n"].tolist() == [2, 3]
