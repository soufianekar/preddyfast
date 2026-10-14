"""scorecard.aggregate and scorecard.relative_to on hand-built pair rows."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scores.categorical import BinaryContingencyManager

from utils import scorecard

NEWEST = pd.Timestamp("2026-10-05")


def _rows(store="ifs_ens", variable="temperature_2m", inits=(NEWEST,), stations=("A",),
          leads=(24.0,), **values):
    """One row per (init, station, lead); `values` give columns (scalar or per-row)."""
    idx = pd.MultiIndex.from_product(
        [list(inits), list(stations), list(leads)], names=["init", "station", "lead_hours"]
    ).to_frame(index=False)
    n = len(idx)
    out = idx.assign(
        store=store,
        variable=variable,
        valid_time=idx["init"] + pd.to_timedelta(idx["lead_hours"], unit="h"),
        obs=0.0, fc_mean=0.0, err=0.0, ae=0.0, se=0.0, crps=0.0,
        spread2=np.nan, wet_fc=np.nan, wet_obs=np.nan, brier=np.nan,
    )
    for key, val in values.items():
        out[key] = np.broadcast_to(np.asarray(val, dtype=float), (n,)).copy()
    return out


def _err_rows(err, **kw):
    err = np.asarray(err, dtype=float)
    return _rows(err=err, ae=np.abs(err), se=err**2, **kw)


def _one(agg, **where):
    sel = agg
    for key, val in where.items():
        sel = sel[sel[key] == val]
    assert len(sel) == 1, sel
    return sel.iloc[0]


def _counts_rows(h, m, f, cn, store="ifs_ens"):
    """Precipitation pairs with known contingency counts, one init, several stations."""
    wet_fc = [1.0] * h + [0.0] * m + [1.0] * f + [0.0] * cn
    wet_obs = [1.0] * h + [1.0] * m + [0.0] * f + [0.0] * cn
    stations = [f"S{i}" for i in range(len(wet_fc))]
    return _rows(store=store, variable="precipitation_24h", stations=stations,
                 wet_fc=wet_fc, wet_obs=wet_obs, brier=0.1)


# --- brief bullets -------------------------------------------------------------


def test_rmse_is_root_of_mean_square_not_mean_of_roots():
    rows = pd.concat([
        _err_rows(1.0, inits=(NEWEST,)),
        _err_rows(3.0, inits=(NEWEST - pd.Timedelta(days=1),)),
    ])
    agg = scorecard.aggregate(rows, 7, min_n=1)
    assert _one(agg, store="ifs_ens", lead_day=1)["rmse"] == pytest.approx(np.sqrt(5.0))
    assert _one(agg, store="ifs_ens", lead_day=1)["n_inits"] == 2


def test_ets_from_known_counts_matches_dynamical_formula():
    h, m, f, cn = 10, 5, 3, 82
    agg = scorecard.aggregate(_counts_rows(h, m, f, cn), 7, min_n=1)
    n = h + m + f + cn
    hr = (h + m) * (h + f) / n
    assert _one(agg, lead_day=1)["ets"] == pytest.approx((h - hr) / (h + m + f - hr))


def test_counts_match_binary_contingency_manager():
    h, m, f, cn = 10, 5, 3, 82
    rows = _counts_rows(h, m, f, cn)
    got = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    bcm = BinaryContingencyManager(
        xr.DataArray(rows["wet_fc"].to_numpy()), xr.DataArray(rows["wet_obs"].to_numpy())
    )
    assert got["ets"] == pytest.approx(float(bcm.equitable_threat_score()))
    assert got["hss"] == pytest.approx(float(bcm.heidke_skill_score()))
    assert got["frequency_bias"] == pytest.approx(float(bcm.frequency_bias()))


def test_bias_corrected_rmse_of_constant_error_is_zero():
    rows = _err_rows(2.0, inits=[NEWEST - pd.Timedelta(days=d) for d in range(5)],
                     stations=("A", "B"))
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert row["rmse"] == pytest.approx(2.0) and row["bias"] == pytest.approx(2.0)
    assert row["rmse_bc"] == pytest.approx(0.0) and row["mae_bc"] == pytest.approx(0.0)


def test_relative_to_lower_rmse_and_higher_ets_are_positive():
    agg = pd.DataFrame({
        "variable": ["precipitation_24h"] * 2,
        "store": ["ifs_ens", "gefs"],
        "lead_day": [1, 1],
        "rmse": [2.0, 1.8],
        "ets": [0.30, 0.33],
    })
    rel = scorecard.relative_to(agg)
    gefs = _one(rel, store="gefs")
    assert gefs["rmse"] == pytest.approx(10.0)
    assert gefs["ets"] == pytest.approx(10.0)
    base = _one(rel, store="ifs_ens")
    assert base["rmse"] == 0.0 and base["ets"] == 0.0


def test_lead_day_bin_under_min_n_is_flagged():
    inits = [NEWEST - pd.Timedelta(days=d) for d in range(5)]  # >= SCORECARD_MIN_INITS
    rows = pd.concat([
        _err_rows(1.0, inits=inits, stations=[f"S{i}" for i in range(9)], leads=(24.0,)),
        _err_rows(1.0, inits=inits, stations=[f"S{i}" for i in range(10)], leads=(48.0,)),
    ])
    agg = scorecard.aggregate(rows, 7)
    assert bool(_one(agg, lead_day=1)["low_n"]) is True
    assert bool(_one(agg, lead_day=2)["low_n"]) is False
    assert _one(agg, lead_day=1)["n"] == 45


# --- task extras ---------------------------------------------------------------


def test_frequency_bias_and_hss_from_known_counts():
    h, m, f, cn = 10, 5, 3, 82
    row = _one(scorecard.aggregate(_counts_rows(h, m, f, cn), 7, min_n=1), lead_day=1)
    assert row["frequency_bias"] == pytest.approx((h + f) / (h + m))
    hss = 2 * (h * cn - f * m) / ((h + m) * (m + cn) + (h + f) * (f + cn))
    assert row["hss"] == pytest.approx(hss)


def test_zero_denominator_categorical_scores_are_nan():
    # No event forecast or observed: ETS, HSS and frequency bias are undefined.
    row = _one(scorecard.aggregate(_counts_rows(0, 0, 0, 20), 7, min_n=1), lead_day=1)
    assert np.isnan(row["ets"]) and np.isnan(row["hss"]) and np.isnan(row["frequency_bias"])
    assert np.isfinite(row["mae"])


def test_spread_skill_is_one_for_a_calibrated_51_member_ensemble():
    m = 51
    err = np.array([1.0, -2.0, 0.5, 3.0])
    rows = _err_rows(err, stations=("A", "B", "C", "D"))
    rows = rows.assign(spread2=rows["se"] * m / (m + 1))
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert row["spread_skill"] == pytest.approx(1.0)


def test_bias_correction_uses_the_whole_window_not_the_period():
    old = [NEWEST - pd.Timedelta(days=d) for d in range(30, 37)]
    recent = [NEWEST - pd.Timedelta(days=d) for d in range(7)]
    rows = pd.concat([_err_rows(4.0, inits=old), _err_rows(0.0, inits=recent)])
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    # Station mean over all rows is +2; the 7-day pairs (err 0) are -2 after correction.
    assert row["n_inits"] == 7
    assert row["rmse"] == pytest.approx(0.0)
    assert row["rmse_bc"] == pytest.approx(2.0) and row["mae_bc"] == pytest.approx(2.0)


def test_common_inits_drop_an_init_held_by_one_model_only():
    two = (NEWEST, NEWEST - pd.Timedelta(days=1))
    rows = pd.concat([
        _err_rows(1.0, store="ifs_ens", inits=two),
        _err_rows(1.0, store="gfs", inits=two[:1]),
    ])
    agg = scorecard.aggregate(rows, 7, min_n=1)
    assert _one(agg, store="ifs_ens")["n_inits"] == 1
    loose = scorecard.aggregate(rows, 7, min_n=1, common_inits=False)
    assert _one(loose, store="ifs_ens")["n_inits"] == 2


def test_relative_to_target_valued_scores():
    agg = pd.DataFrame({
        "variable": ["precipitation_24h"] * 2,
        "store": ["ifs_ens", "gefs"],
        "lead_day": [1, 1],
        "bias": [0.4, -0.2],
        "frequency_bias": [1.4, 0.8],
    })
    gefs = _one(scorecard.relative_to(agg), store="gefs")
    assert gefs["bias"] == pytest.approx(50.0)  # |bias| halved
    assert gefs["frequency_bias"] == pytest.approx(50.0)  # |FB - 1| halved


def test_relative_to_without_baseline_is_nan():
    agg = pd.DataFrame({"variable": ["t", "t"], "store": ["ifs_ens", "gefs"],
                        "lead_day": [1, 2], "rmse": [1.0, 1.0]})
    rel = scorecard.relative_to(agg)
    assert np.isnan(_one(rel, store="gefs")["rmse"])


def test_deterministic_store_has_nan_spread_skill_and_brier():
    temp = _one(scorecard.aggregate(_err_rows(1.0, store="gfs"), 7, min_n=1), lead_day=1)
    assert np.isnan(temp["spread_skill"]) and temp["crps"] == pytest.approx(0.0)
    precip = _counts_rows(2, 1, 1, 6, store="gfs").assign(brier=np.nan)
    row = _one(scorecard.aggregate(precip, 7, min_n=1), lead_day=1)
    assert np.isnan(row["brier"]) and np.isfinite(row["ets"])


def test_lead_day_bins_and_t0_is_excluded():
    rows = _err_rows(1.0, leads=(0.0, 6.0, 24.0, 30.0, 48.0))
    agg = scorecard.aggregate(rows, 7, min_n=1)
    assert sorted(agg["lead_day"]) == [1, 2]
    assert _one(agg, lead_day=1)["n"] == 2 and _one(agg, lead_day=2)["n"] == 2


def test_scores_the_variable_does_not_offer_are_nan():
    rows = _err_rows(1.0, variable="wind_speed_10m")
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert row["rmse"] == pytest.approx(1.0)
    assert np.isnan(row["ets"]) and np.isnan(row["rmse_bc"]) and np.isnan(row["spread_skill"])


def test_variables_are_never_pooled():
    rows = pd.concat([_err_rows(1.0), _err_rows(3.0, variable="wind_speed_10m")])
    agg = scorecard.aggregate(rows, 7, min_n=1)
    assert _one(agg, variable="temperature_2m")["rmse"] == pytest.approx(1.0)
    assert _one(agg, variable="wind_speed_10m")["rmse"] == pytest.approx(3.0)


def test_period_counts_back_from_the_newest_init_or_end():
    inits = [NEWEST - pd.Timedelta(days=d) for d in range(10)]
    rows = _err_rows(1.0, inits=inits)
    assert _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)["n_inits"] == 7
    end = NEWEST - pd.Timedelta(days=5)
    assert _one(scorecard.aggregate(rows, 7, end=end, min_n=1), lead_day=1)["n_inits"] == 5


def test_by_station_and_counts():
    rows = _err_rows([1.0, 2.0, 3.0, 4.0], stations=("A", "B"), leads=(24.0, 48.0))
    agg = scorecard.aggregate(rows, 7, by=("store", "station"), min_n=1)
    assert sorted(agg["station"]) == ["A", "B"] and "lead_day" not in agg
    a = _one(agg, station="A")
    assert a["n"] == 2 and a["n_stations"] == 1 and a["mae"] == pytest.approx(1.5)
    full = scorecard.aggregate(rows, 7, by=("store", "lead_day", "station"), min_n=1)
    assert len(full) == 4


def test_empty_rows_give_an_empty_frame():
    agg = scorecard.aggregate(_rows().iloc[:0], 7)
    assert agg.empty and {"store", "lead_day", "variable", "n", "rmse"} <= set(agg.columns)


def test_score_direction_covers_every_scorecard_score():
    from utils.settings import SCORECARD_VARIABLES

    offered = {s for spec in SCORECARD_VARIABLES.values() for s in spec["scores"]}
    assert offered <= set(scorecard.SCORE_DIRECTION)


def test_relative_to_negative_baseline_keeps_sign():
    # A fair CRPS can sit slightly below zero: a lower model score must still read as better.
    agg = pd.DataFrame({
        "variable": ["temperature_2m"] * 2,
        "store": ["ifs_ens", "gefs"],
        "lead_day": [1, 1],
        "crps": [-0.1, -0.2],
    })
    rel = scorecard.relative_to(agg)
    assert _one(rel, store="gefs")["crps"] == pytest.approx(100.0)
    worse = agg.assign(crps=[-0.1, 0.0])
    assert _one(scorecard.relative_to(worse), store="gefs")["crps"] == pytest.approx(-100.0)


# --- final review: R23 lead step, R24 run count -----------------------------------


def test_hourly_and_six_hourly_stores_give_the_same_lead_set_per_bin():
    """R23: a lead-day bin holds the same valid hours whatever the store's step."""
    hourly = np.arange(1.0, 49.0)
    three = np.arange(3.0, 49.0, 3.0)
    six = np.arange(6.0, 49.0, 6.0)
    rows = pd.concat([
        _err_rows(1.0, store="gfs", leads=hourly),
        _err_rows(1.0, store="ifs_ens", leads=three),
        _err_rows(1.0, store="aifs_ens", leads=six),
    ])
    agg = scorecard.aggregate(rows, 7, min_n=1)
    for day in (1, 2):
        n = agg[agg["lead_day"] == day].set_index("store")["n"]
        assert n.to_dict() == {"aifs_ens": 4, "gfs": 4, "ifs_ens": 4}, day


def test_bias_offset_is_computed_on_the_six_hourly_leads_only():
    """Off-step leads (err 5) never enter the station offset, so rmse_bc is 0."""
    leads = np.arange(1.0, 25.0)
    err = np.where(leads % 6 == 0, 2.0, 5.0)
    rows = _err_rows(np.tile(err, 2), stations=("A", "B"), leads=leads)
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert row["n"] == 8
    assert row["rmse"] == pytest.approx(2.0)
    assert row["rmse_bc"] == pytest.approx(0.0)


def test_cells_on_fewer_than_min_inits_runs_are_flagged():
    """R24: many pairs on few runs are still low_n."""
    stations = [f"S{i}" for i in range(100)]
    few = [NEWEST - pd.Timedelta(days=d) for d in range(4)]
    enough = [NEWEST - pd.Timedelta(days=d) for d in range(5)]
    low = _one(scorecard.aggregate(_err_rows(1.0, inits=few, stations=stations), 7),
               lead_day=1)
    ok = _one(scorecard.aggregate(_err_rows(1.0, inits=enough, stations=stations), 7),
              lead_day=1)
    assert low["n"] == 400 and low["n_inits"] == 4 and bool(low["low_n"]) is True
    assert ok["n_inits"] == 5 and bool(ok["low_n"]) is False


def test_lead_zero_z500_rows_and_control_self_pairs_never_reach_the_aggregates():
    """T+0 Z500 vs IFS control T+0 is the control against itself (err 0): it
    must never enter a bin."""
    leads = (0.0, 24.0)
    rows = pd.concat([
        _err_rows([0.0, 2.0], store="ifs_control", variable="z500",
                  leads=leads),
        _err_rows([1.0, 2.0], store="ifs_ens", variable="z500",
                  leads=leads),
    ])
    agg = scorecard.aggregate(rows, 7, min_n=1)
    assert sorted(agg["lead_day"].unique()) == [1]
    ctrl = _one(agg, store="ifs_control", lead_day=1)
    assert ctrl["n"] == 1 and ctrl["bias"] == pytest.approx(2.0)
    assert ctrl["rmse"] == pytest.approx(2.0)
    assert (agg["n"] == 1).all()  # no lead-0 pair of either store counted


# --- Task 5: new scores, modes, significance, diagnostics ------------------------

INITS10 = [NEWEST - pd.Timedelta(days=d) for d in range(10)]


def test_error_sd_removes_the_bias():
    rows = _err_rows([1.5, 1.5, -0.5, -0.5], stations=("A", "B", "C", "D"))
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert row["bias"] == pytest.approx(0.5)
    assert row["error_sd"] == pytest.approx(1.0)


def test_spread_error_is_one_when_spread_matches_the_error_variance():
    m = 51
    rows = _err_rows([1.5, 1.5, -0.5, -0.5], stations=("A", "B", "C", "D"),
                     spread2=1.0 * m / (m + 1), n_members=float(m))
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert row["spread_error"] == pytest.approx(1.0)


def test_spread_error_uses_the_perturbed_member_count_in_representativeness():
    rows = _err_rows([1.5, 1.5, -0.5, -0.5], stations=("A", "B", "C", "D"),
                     store="gfs", n_members=1.0, n_members_pert=50.0,
                     err_pert=[1.5, 1.5, -0.5, -0.5], crps_pert=0.3,
                     spread2_pert=50.0 / 51.0)
    raw = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    pert = _one(scorecard.aggregate(rows, 7, min_n=1, mode="representativeness"),
                lead_day=1)
    assert np.isnan(raw["spread_error"])
    assert pert["spread_error"] == pytest.approx(1.0)


def test_a_forecast_equal_to_persistence_has_zero_persistence_skill():
    err = np.array([1.0, -2.0, 0.5, 3.0])
    rows = _err_rows(err, stations=("A", "B", "C", "D"), obs=10.0,
                     obs_persist=10.0 + err, crps=np.abs(err))
    # A pair with no persistence observation enters neither side.
    extra = _err_rows(50.0, stations=("E",), obs=10.0, obs_persist=np.nan, crps=50.0)
    row = _one(scorecard.aggregate(pd.concat([rows, extra]), 7, min_n=1), lead_day=1)
    assert row["rmse_ss_persist"] == pytest.approx(0.0)
    assert row["crpss_persist"] == pytest.approx(0.0)


def test_persistence_skill_of_a_perfect_forecast_is_one():
    rows = _err_rows(0.0, stations=("A", "B"), obs=10.0, obs_persist=[12.0, 7.0])
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert row["rmse_ss_persist"] == pytest.approx(1.0)
    assert row["crpss_persist"] == pytest.approx(1.0)


def test_height_corrected_mode_uses_err_hc_for_temperature_only():
    t2m = _err_rows(1.0, err_hc=2.0, crps_hc=0.7)
    hc = _one(scorecard.aggregate(t2m, 7, min_n=1, mode="height_corrected"), lead_day=1)
    assert hc["rmse"] == pytest.approx(2.0) and hc["bias"] == pytest.approx(2.0)
    assert hc["crps"] == pytest.approx(0.7) and bool(hc["mode_applied"]) is True
    raw = _one(scorecard.aggregate(t2m, 7, min_n=1), lead_day=1)
    assert raw["rmse"] == pytest.approx(1.0) and bool(raw["mode_applied"]) is True

    wind = _err_rows(1.0, variable="wind_speed_10m", err_hc=2.0, crps_hc=0.7)
    w = _one(scorecard.aggregate(wind, 7, min_n=1, mode="height_corrected"), lead_day=1)
    assert w["rmse"] == pytest.approx(1.0) and bool(w["mode_applied"]) is False


def test_representativeness_mode_uses_the_perturbed_columns():
    wind = _err_rows(1.0, variable="wind_speed_10m", err_pert=3.0, crps_pert=0.4)
    row = _one(scorecard.aggregate(wind, 7, min_n=1, mode="representativeness"),
               lead_day=1)
    assert row["rmse"] == pytest.approx(3.0) and row["crps"] == pytest.approx(0.4)
    assert bool(row["mode_applied"]) is True
    z = _err_rows(1.0, variable="z500", err_pert=3.0)
    zr = _one(scorecard.aggregate(z, 7, min_n=1, mode="representativeness"), lead_day=1)
    assert zr["rmse"] == pytest.approx(1.0) and bool(zr["mode_applied"]) is False


def test_bias_correction_follows_the_mode_error_column():
    rows = _err_rows(1.0, inits=INITS10[:5], stations=("A", "B"),
                     err_hc=[3.0, 1.0] * 5)
    row = _one(scorecard.aggregate(rows, 7, min_n=1, mode="height_corrected"),
               lead_day=1)
    assert row["rmse_bc"] == pytest.approx(0.0)  # each station's own offset removed


def test_per_threshold_contingency_and_brier():
    h, m, f, cn = 10, 5, 3, 82
    rows = _counts_rows(h, m, f, cn)
    rows = rows.assign(wet_fc_5=rows["wet_fc"], wet_obs_5=rows["wet_obs"], brier_5=0.2)
    row = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert row["pod_5"] == pytest.approx(h / (h + m))
    assert row["far_5"] == pytest.approx(f / (h + f))
    assert row["fbi_5"] == pytest.approx((h + f) / (h + m))
    assert row["ets_5"] == pytest.approx(row["ets"])
    assert row["brier_5"] == pytest.approx(0.2)
    assert np.isnan(row["ets_10"])


def _prob_rows(p, wet, store="ifs_ens", m=50, inits=(NEWEST,), leads=(24.0,), **kw):
    """Precipitation pairs with given member fractions and observed events."""
    p = np.asarray(p, dtype=float)
    stations = [f"S{i}" for i in range(p.size)]
    return _rows(store=store, variable="precipitation_24h", stations=stations,
                 inits=inits, leads=leads, p_1=np.tile(p, len(inits) * len(leads)),
                 wet_obs_1=np.tile(np.asarray(wet, dtype=float), len(inits) * len(leads)),
                 n_members=float(m), n_members_pert=float(m), **kw)


def test_roc_area_of_a_perfect_forecast_is_one_and_of_a_random_one_half():
    wet = np.r_[np.ones(50), np.zeros(50)]
    perfect = _prob_rows(wet, wet)
    row = _one(scorecard.aggregate(perfect, 7, min_n=1), lead_day=1)
    assert row["roc_area_1"] == pytest.approx(1.0)

    rng = np.random.default_rng(1)
    n = 4000
    rand = _prob_rows(rng.integers(0, 51, n) / 50, rng.random(n) < 0.3)
    row = _one(scorecard.aggregate(rand, 7, min_n=1), lead_day=1)
    assert row["roc_area_1"] == pytest.approx(0.5, abs=0.03)


def test_roc_points_include_the_corners_and_area_matches():
    wet = np.r_[np.ones(50), np.zeros(50)]
    pts = scorecard.roc(_prob_rows(wet, wet), 7, 1.0)
    assert {(0.0, 0.0), (1.0, 1.0)} <= set(zip(pts["pofd"], pts["pod"]))
    assert pts["area"].unique() == pytest.approx([1.0])


def test_reliability_of_reliable_probabilities_lies_on_the_diagonal():
    rng = np.random.default_rng(2)
    n = 40000
    p = rng.integers(0, 51, n) / 50
    rows = _prob_rows(p, rng.random(n) < p)
    rel = scorecard.reliability(rows, 7, 1.0)
    rel = rel[(rel["store"] == "ifs_ens") & (rel["lead_group"] == "1-3")]
    assert len(rel) == 10 and rel["count"].sum() == n
    np.testing.assert_allclose(rel["obs_freq"], rel["p_mean"], atol=0.03)
    assert rel["bin_lo"].tolist() == pytest.approx(np.arange(10) / 10)


def test_reliability_of_p_one_falls_in_the_last_bin():
    rows = _prob_rows([1.0, 0.0, 0.1], [1, 0, 0])
    rel = scorecard.reliability(rows, 7, 1.0).set_index("bin")
    assert rel.loc[9, "count"] == 1 and rel.loc[0, "count"] == 1
    assert rel.loc[1, "count"] == 1


def test_deterministic_store_has_reliability_only_when_perturbed():
    rows = _prob_rows([1.0, 0.0], [1, 0], store="gfs", m=1,
                      p_pert_1=[0.9, 0.1], err_pert=0.0).assign(n_members=1.0, n_members_pert=50.0)
    assert scorecard.reliability(rows, 7, 1.0).empty
    assert scorecard.roc(rows, 7, 1.0).empty
    raw = _one(scorecard.aggregate(rows, 7, min_n=1), lead_day=1)
    assert np.isnan(raw["roc_area_1"])
    pert = scorecard.reliability(rows, 7, 1.0, mode="representativeness")
    assert pert["count"].sum() == 2
    assert not scorecard.roc(rows, 7, 1.0, mode="representativeness").empty


@pytest.mark.parametrize("m", [50, 30])
def test_rank_histogram_of_exchangeable_integer_ranks_is_exactly_flat(m):
    ranks = np.tile(np.arange(m + 1, dtype=float), 3)
    stations = [f"S{i}" for i in range(ranks.size)]
    rows = _rows(stations=stations, rank=ranks, n_members=float(m))
    hist = scorecard.rank_histogram(rows, 7, lead_days=(1,))
    assert len(hist) == 10
    np.testing.assert_allclose(hist["count"], ranks.size / 10)
    np.testing.assert_allclose(hist["freq"], 0.1)


def test_rank_histogram_of_random_exchangeable_draws_is_flat():
    rng = np.random.default_rng(3)
    m, n = 9, 20000
    members = rng.normal(size=(n, m))
    obs = rng.normal(size=n)
    ranks = (members < obs[:, None]).sum(axis=1).astype(float)
    rows = _rows(stations=[f"S{i}" for i in range(n)], rank=ranks, n_members=float(m))
    hist = scorecard.rank_histogram(rows, 7, lead_days=(1,))
    np.testing.assert_allclose(hist["freq"], 0.1, atol=0.01)


def test_rank_histogram_uses_rank_pert_in_representativeness():
    rows = _rows(stations=("A",), store="gfs", rank=np.nan, n_members=1.0,
                 rank_pert=50.0, n_members_pert=50.0, err_pert=0.0)
    assert scorecard.rank_histogram(rows, 7, lead_days=(1,)).empty
    pert = scorecard.rank_histogram(rows, 7, lead_days=(1,), mode="representativeness")
    assert pert.set_index("bin").loc[9, "count"] == pytest.approx(1.0)  # 50.5/51 in bin 9


def test_diurnal_groups_by_valid_hour_and_lead_group():
    leads = (6.0, 12.0, 18.0, 24.0, 30.0, 102.0)
    err = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    rows = _err_rows(err, leads=leads)
    d = scorecard.diurnal(rows, 7)
    first = d[d["lead_group"] == "1-3"].set_index("valid_hour")
    assert sorted(first.index) == [0, 6, 12, 18]
    assert first.loc[6, "bias"] == pytest.approx(3.0)  # leads 6 and 30
    assert first.loc[6, "n"] == 2 and first.loc[6, "error_sd"] == pytest.approx(2.0)
    assert d[d["lead_group"] == "4-7"]["valid_hour"].tolist() == [6]


def _paired(store_err, base_err, inits=INITS10, stations=("A", "B", "C")):
    return pd.concat([
        _err_rows(base_err, store="ifs_ens", inits=inits, stations=stations),
        _err_rows(store_err, store="gfs", inits=inits, stations=stations),
    ], ignore_index=True)


def test_significance_of_identical_rows_is_zero_and_not_significant():
    rng = np.random.default_rng(4)
    err = rng.normal(size=len(INITS10) * 3)
    sig = scorecard.significance(_paired(err, err), score="rmse", period_days=14)
    gfs = _one(sig, store="gfs", lead_day=1)
    assert gfs["diff"] == pytest.approx(0.0) and bool(gfs["sig"]) is False
    base = _one(sig, store="ifs_ens", lead_day=1)
    assert base["diff"] == 0.0 and np.isnan(base["sig"])


@pytest.mark.parametrize("score", ["mae", "rmse", "crps"])
def test_significance_of_a_constant_one_kelvin_error_is_significant(score):
    rows = _paired(1.0, 0.0).assign(crps=lambda r: r["ae"])
    sig = scorecard.significance(rows, score=score, period_days=14)
    gfs = _one(sig, store="gfs", lead_day=1)
    assert gfs["diff"] == pytest.approx(1.0)
    assert gfs["lo"] > 0 and bool(gfs["sig"]) is True


def test_significance_on_rmse_scale_reports_the_rmse_difference():
    rows = _paired(2.0, 1.0)
    gfs = _one(scorecard.significance(rows, score="rmse", period_days=14),
               store="gfs", lead_day=1)
    assert gfs["diff"] == pytest.approx(1.0)
    assert gfs["lo"] == pytest.approx(1.0) and gfs["hi"] == pytest.approx(1.0)


def test_significance_with_too_few_inits_is_nan_not_false():
    rows = _paired(1.0, 0.0, inits=INITS10[:3])
    gfs = _one(scorecard.significance(rows, score="mae", period_days=14),
               store="gfs", lead_day=1)
    assert gfs["diff"] == pytest.approx(1.0)
    assert np.isnan(gfs["sig"]) and np.isnan(gfs["lo"]) and np.isnan(gfs["hi"])


def test_significance_is_paired_on_init_station_and_lead():
    rows = _paired(1.0, 0.0)
    # An extra gfs station the baseline lacks never enters the comparison.
    rows = pd.concat([rows, _err_rows(100.0, store="gfs", inits=INITS10,
                                      stations=("Z",))])
    gfs = _one(scorecard.significance(rows, score="mae", period_days=14),
               store="gfs", lead_day=1)
    assert gfs["diff"] == pytest.approx(1.0)


def test_significance_does_not_depend_on_row_order():
    rng = np.random.default_rng(5)
    rows = _paired(rng.normal(size=30) + 0.2, rng.normal(size=30))
    a = scorecard.significance(rows, score="mae", period_days=14)
    b = scorecard.significance(rows.sample(frac=1.0, random_state=0), score="mae",
                               period_days=14)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))
