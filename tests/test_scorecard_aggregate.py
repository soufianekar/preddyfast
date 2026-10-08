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
    return _rows(store=store, variable="precipitation_6h", stations=stations,
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
        "variable": ["precipitation_6h"] * 2,
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
        "variable": ["precipitation_6h"] * 2,
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
