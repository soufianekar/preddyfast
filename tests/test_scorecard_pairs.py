"""scorecard: station cube + observations -> per-pair score rows (no network)."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils import scorecard, verification
from utils.settings import SCORECARD_VARIABLES

INIT = pd.Timestamp("2026-09-01 00:00")


def _cube(values, leads, stations=("S1",), variable="temperature_2m", window=None):
    """`values` (station, member, lead_hours) as a station_cube-shaped DataArray."""
    values = np.asarray(values, dtype=float)
    leads = np.asarray(leads, dtype=float)
    return xr.DataArray(
        values,
        dims=("station", "member", "lead_hours"),
        coords={
            "station": list(stations),
            "member": np.arange(values.shape[1]),
            "lead_hours": leads,
            "valid_time": ("lead_hours", INIT + pd.to_timedelta(leads, unit="h")),
            "lat": ("station", np.full(len(stations), 50.0)),
            "lon": ("station", np.full(len(stations), 10.0)),
        },
        attrs={
            "init": INIT.isoformat(),
            "store": "ifs_ens",
            "variable": variable,
            "window_hours": window,
            "step_type": "instant",
        },
    )


def _obs(times, **columns):
    """A network_obs-shaped frame: tz-aware UTC `valid_time` plus value columns."""
    frame = pd.DataFrame({"valid_time": pd.to_datetime(times).tz_localize("UTC")})
    for name, values in columns.items():
        frame[name] = values
    return frame


def _hours(end, n):
    return pd.date_range(end=end, periods=n, freq="h")


# --- matching -------------------------------------------------------------------


def test_instant_matches_within_tolerance_only():
    cube = _cube([[[1.0, 2.0]]], [6, 12])
    obs = _obs(
        [INIT + pd.Timedelta("6h10min"), INIT + pd.Timedelta("12h20min")],
        temperature_2m=[5.0, 7.0],
    )
    m = scorecard.match_station_obs(cube, {"S1": obs}, "temperature_2m")
    assert m["obs"].sel(station="S1", lead_hours=6).item() == 5.0
    # 20 min off valid_time: outside the +-15 min tolerance.
    assert np.isnan(m["obs"].sel(station="S1", lead_hours=12).item())


def test_matched_dataset_layout():
    cube = _cube([[[1.0, 3.0], [2.0, 5.0]]], [6, 12])
    obs = _obs([INIT + pd.Timedelta("6h")], temperature_2m=[0.0])
    m = scorecard.match_station_obs(cube, {"S1": obs}, "temperature_2m")
    assert m["fc"].dims == ("station", "member", "lead_hours")
    assert m["obs"].dims == ("station", "lead_hours")
    np.testing.assert_allclose(m["fc_mean"].sel(station="S1").values, [1.5, 4.0])
    assert "valid_time" in m.coords or "valid_time" in m


def test_station_without_obs_is_nan():
    cube = _cube(np.zeros((2, 1, 1)), [6], stations=("S1", "S2"))
    obs = _obs([INIT + pd.Timedelta("6h")], temperature_2m=[1.0])
    m = scorecard.match_station_obs(cube, {"S1": obs}, "temperature_2m")
    assert np.isnan(m["obs"].sel(station="S2").item(0))


V24 = "precipitation_24h"
END30 = INIT + pd.Timedelta("30h")  # 06 UTC the next day


def test_precip_uses_reported_24h_total():
    cube = _cube([[[1.0]]], [30], variable=V24, window=24)
    obs = _obs(_hours(END30, 24), precipitation_1h=[1.0] * 24,
               precip_total_24h=[np.nan] * 23 + [2.5])
    m = scorecard.match_station_obs(cube, {"S1": obs}, V24)
    assert m["obs"].item(0) == 2.5


def test_precip_never_falls_back_to_hourly_sums():
    cube = _cube([[[1.0]]], [30], variable=V24, window=24)
    obs = _obs(_hours(END30, 24), precipitation_1h=[0.5] * 24,
               precip_total_24h=[np.nan] * 24)
    m = scorecard.match_station_obs(cube, {"S1": obs}, V24)
    assert np.isnan(m["obs"].item(0))
    assert scorecard.pair_scores(m, V24).empty


def test_window_obs_reported_days_kept_and_hourly_only_day_is_nan():
    spec = SCORECARD_VARIABLES[V24]
    d1, d2, d3 = (pd.Timestamp(f"2026-09-0{d} 06:00") for d in (2, 3, 4))
    frame = pd.concat([
        _obs([d1, d3], precip_total_24h=[3.0, 0.0], precipitation_1h=[np.nan] * 2),
        _obs(_hours(d2, 24), precip_total_24h=[np.nan] * 24,
             precipitation_1h=[9.0 / 24] * 24),
    ], ignore_index=True)
    got = scorecard._window_obs(frame, spec, pd.Series([d1, d2, d3]))
    assert got[0] == 3.0 and got[2] == 0.0
    assert np.isnan(got[1])  # 9 mm of hourly reports is never used


def test_other_window_variables_default_to_hourly_fallback():
    spec = {"window_hours": 6, "obs": "precipitation_1h", "obs_reported": None}
    end = INIT + pd.Timedelta("6h")
    frame = _obs(_hours(end, 6), precipitation_1h=[0.5] * 6)
    got = scorecard._window_obs(frame, spec, pd.Series([end]))
    assert got[0] == pytest.approx(3.0)


def test_precip_keeps_only_leads_ending_at_06_utc():
    leads = [6, 24, 30, 36, 54]
    cube = _cube(np.ones((1, 1, 5)), leads, variable=V24, window=24)
    m = scorecard.match_station_obs(cube, {}, V24)
    assert list(m["lead_hours"].values) == [6, 30, 54]


def test_report_not_ending_at_valid_time_is_not_used():
    cube = _cube([[[1.0]]], [30], variable=V24, window=24)
    obs = _obs([END30 + pd.Timedelta("1h")], precip_total_24h=[3.0],
               precipitation_1h=[np.nan])
    m = scorecard.match_station_obs(cube, {"S1": obs}, V24)
    assert np.isnan(m["obs"].item(0))


def test_pod_and_far_directions():
    from utils.settings import SCORE_DIRECTION
    assert SCORE_DIRECTION["pod"] == "higher" and SCORE_DIRECTION["far"] == "lower"


def test_variable_without_obs_is_refused():
    assert SCORECARD_VARIABLES["z500"]["obs"] is None
    cube = _cube([[[5500.0]]], [24], variable="z500")
    with pytest.raises(ValueError, match="z500"):
        scorecard.match_station_obs(cube, {}, "z500")


def test_leads_beyond_360h_are_dropped():
    leads = [354, 360, 366, 384]
    cube = _cube(np.ones((1, 2, 4)), leads)
    obs = _obs(INIT + pd.to_timedelta(leads, unit="h"), temperature_2m=[0.0] * 4)
    m = scorecard.match_station_obs(cube, {"S1": obs}, "temperature_2m")
    assert list(m["lead_hours"].values) == [354, 360]
    rows = scorecard.pair_scores(m, "temperature_2m")
    assert rows["lead_hours"].max() == 360


# --- scores ---------------------------------------------------------------------


def _matched(fc, obs_values, variable="temperature_2m", leads=None):
    """A matched dataset for one station from fc (member, lead) and obs (lead)."""
    fc = np.asarray(fc, dtype=float)
    if leads is None:
        end_hour = SCORECARD_VARIABLES[variable].get("window_end_hour")
        # 06 UTC ends for a 24 h window variable, else every 6 h.
        leads = [6 + 24 * (i + 1) if end_hour == 6 else 6 * (i + 1)
                 for i in range(fc.shape[1])]
    window = SCORECARD_VARIABLES[variable]["window_hours"]
    cube = _cube(fc[None], leads, variable=variable, window=window)
    col = SCORECARD_VARIABLES[variable]["obs"]
    times = INIT + pd.to_timedelta(leads, unit="h")
    if window:
        reported = SCORECARD_VARIABLES[variable]["obs_reported"]
        obs = _obs(times, **{reported: obs_values, col: [np.nan] * len(leads)})
    else:
        obs = _obs(times, **{col: obs_values})
    return scorecard.match_station_obs(cube, {"S1": obs}, variable)


def test_hand_computed_three_members():
    m = _matched([[1.0], [2.0], [6.0]], [2.0])
    row = scorecard.pair_scores(m, "temperature_2m").iloc[0]
    assert row["fc_mean"] == pytest.approx(3.0)
    assert row["err"] == pytest.approx(1.0)  # forecast minus observed
    assert row["ae"] == pytest.approx(1.0)
    assert row["se"] == pytest.approx(1.0)
    # mean|x-y| = 5/3; sum_ij|xi-xj| = 2*(1+5+4) = 20; fair: 20 / (2*3*2) = 5/3.
    assert row["crps"] == pytest.approx(5 / 3 - 20 / 12)
    assert row["spread2"] == pytest.approx(np.var([1.0, 2.0, 6.0], ddof=1))


def test_err_sign_is_forecast_minus_observed():
    m = _matched([[1.0], [1.0]], [4.0])
    row = scorecard.pair_scores(m, "temperature_2m").iloc[0]
    assert row["err"] == pytest.approx(-3.0)
    assert row["ae"] == pytest.approx(3.0)
    assert row["se"] == pytest.approx(9.0)


def test_fair_crps_matches_verification_on_51_members():
    rng = np.random.default_rng(42)
    fc = rng.normal(10, 3, size=(51, 8))
    obs = rng.normal(10, 3, size=8)
    rows = scorecard.pair_scores(_matched(fc, obs), "temperature_2m")
    expected = [verification.fair_crps(fc[:, i], obs[i]) for i in range(8)]
    np.testing.assert_allclose(rows["crps"].to_numpy(), expected, rtol=0, atol=1e-9)


def test_single_member_crps_is_absolute_error():
    m = _matched([[3.0, -1.0]], [1.0, 2.0])
    rows = scorecard.pair_scores(m, "temperature_2m")
    np.testing.assert_allclose(rows["crps"], [2.0, 3.0])
    np.testing.assert_allclose(rows["crps"], rows["ae"])
    assert rows["spread2"].isna().all()


def test_nan_obs_gives_no_row_not_zero():
    m = _matched([[1.0, 2.0], [3.0, 4.0]], [np.nan, 2.0])
    rows = scorecard.pair_scores(m, "temperature_2m")
    assert list(rows["lead_hours"]) == [12]
    assert not (rows[["err", "ae", "se", "crps"]] == 0).all(axis=1).any()


def test_nan_forecast_pair_is_dropped_and_counted():
    # Member 0 is NaN at lead 12: that pair is not scored.
    m = _matched([[1.0, np.nan], [3.0, 4.0]], [2.0, 2.0])
    rows = scorecard.pair_scores(m, "temperature_2m")
    assert list(rows["lead_hours"]) == [6]
    assert rows.attrs["dropped_nan_fc"] == 1


def test_wet_flags_from_ensemble_mean_inclusive_threshold():
    threshold = SCORECARD_VARIABLES[V24]["threshold"]
    # Leads: mean exactly at threshold (wet), mean just below (dry), mean well above.
    fc = [[threshold, threshold - 0.01, 0.0], [threshold, threshold - 0.01, 3.0]]
    obs = [threshold, 0.0, 0.59]
    rows = scorecard.pair_scores(_matched(fc, obs, V24), V24)
    assert list(rows["wet_fc"]) == [1.0, 0.0, 1.0]
    assert list(rows["wet_obs"]) == [1.0, 0.0, 0.0]


def test_brier_for_ensembles_fair_and_inclusive():
    threshold = SCORECARD_VARIABLES[V24]["threshold"]
    # 1 of 4 members wet (one exactly at the threshold), obs wet.
    fc = [[threshold], [0.0], [0.0], [0.0]]
    rows = scorecard.pair_scores(_matched(fc, [1.0], V24), V24)
    i, m = 1, 4
    expected = (i / m - 1) ** 2 - i * (m - i) / (m**2 * (m - 1))
    assert rows["brier"].iloc[0] == pytest.approx(expected)


def test_brier_nan_for_deterministic_and_non_brier_variables():
    det = scorecard.pair_scores(_matched([[1.0]], [1.0], V24), V24)
    assert det["brier"].isna().all()
    assert det["wet_fc"].iloc[0] == 1.0  # contingency counts still apply
    temp = scorecard.pair_scores(_matched([[1.0], [2.0]], [1.0]), "temperature_2m")
    assert temp["brier"].isna().all()
    assert temp["wet_fc"].isna().all() and temp["wet_obs"].isna().all()


def test_row_columns_and_init():
    rows = scorecard.pair_scores(_matched([[1.0], [2.0]], [1.0]), "temperature_2m")
    assert list(rows.columns[:14]) == [
        "station", "init", "lead_hours", "valid_time", "obs", "fc_mean",
        "err", "ae", "se", "crps", "spread2", "wet_fc", "wet_obs", "brier",
    ]  # the original layout leads; Task 4 columns follow
    assert rows["init"].iloc[0] == INIT
    assert rows["valid_time"].iloc[0] == INIT + pd.Timedelta("6h")
    assert rows["station"].iloc[0] == "S1"


def test_no_valid_pairs_gives_empty_frame_with_columns():
    rows = scorecard.pair_scores(_matched([[1.0], [2.0]], [np.nan]), "temperature_2m")
    assert rows.empty
    assert "crps" in rows.columns


def test_wet_flag_survives_a_float32_cube(monkeypatch):
    # 1.0 is exact in float32, so the ulp case needs a non-representable threshold.
    monkeypatch.setitem(SCORECARD_VARIABLES, V24, {**SCORECARD_VARIABLES[V24], "threshold": 0.6})
    # station_cube returns float32. These members' exact mean is >= 0.6, but a
    # float32 mean lands one ulp below it and would flag the pair dry.
    members = np.array([0.6000002026557922, 0.6000000834465027, 0.599999725818634], dtype=np.float32)
    cube = _cube(members.reshape(1, 3, 1), [30], variable=V24, window=24)
    cube = cube.astype("float32")
    obs = _obs([END30], precipitation_1h=[0.0], precip_total_24h=[0.7])
    m = scorecard.match_station_obs(cube, {"S1": obs}, V24)
    assert m["fc"].dtype == np.float64
    rows = scorecard.pair_scores(m, V24)
    assert list(rows["wet_fc"]) == [1.0]


# --- Task 4: height correction, TM865 perturbation, ranks, persistence ------------

from utils.settings import SCORECARD_DETERMINISTIC_DRAWS  # noqa: E402


def _many(fc, obs, stations, leads, variable="temperature_2m", store="ifs_ens"):
    """fc (station, member, lead), obs (station, lead) -> matched dataset."""
    cube = _cube(fc, leads, stations=stations, variable=variable)
    cube.attrs["store"] = store
    times = INIT + pd.to_timedelta(leads, unit="h")
    col = SCORECARD_VARIABLES[variable]["obs"]
    by = {s: _obs(times, **{col: obs[i]}) for i, s in enumerate(stations)}
    return scorecard.match_station_obs(cube, by, variable)


def test_threshold_key():
    assert scorecard.threshold_key(1.0) == "1"
    assert scorecard.threshold_key(0.5) == "0.5"
    assert scorecard.threshold_key(10) == "10"


def test_new_columns_present():
    rows = scorecard.pair_scores(_matched([[1.0], [2.0]], [1.0]), "temperature_2m")
    for col in ("dz", "err_hc", "crps_hc", "crps_pert", "spread2_pert", "err_pert",
                "rank", "rank_pert", "n_members", "n_members_pert", "obs_persist",
                "p_1", "p_pert_1", "wet_fc_5", "wet_obs_10", "brier_1",
                "brier_pert_10"):
        assert col in rows.columns, col
    assert list(rows.columns) == scorecard.PAIR_COLUMNS


def test_rank_extremes():
    m = _matched([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], [100.0, -100.0])
    rows = scorecard.pair_scores(m, "temperature_2m")
    assert list(rows["rank"]) == [3.0, 0.0]
    assert (rows["n_members"] == 3).all()


def _flat(ranks, m, tol):
    counts = np.bincount(ranks.astype(int), minlength=m + 1)
    expected = len(ranks) / (m + 1)
    assert counts.size == m + 1
    assert np.abs(counts / expected - 1).max() < tol, counts


def test_rank_histogram_flat_for_exchangeable_obs():
    rng = np.random.default_rng(1)
    stations = [f"S{i}" for i in range(200)]
    leads = np.arange(6, 366, 6.0)
    m = 9
    fc = rng.normal(size=(200, m, leads.size))
    obs = rng.normal(size=(200, leads.size))
    rows = scorecard.pair_scores(_many(fc, obs, stations, leads), "temperature_2m")
    assert len(rows) == 200 * leads.size
    _flat(rows["rank"].to_numpy(), m, 0.15)


def test_rank_ties_broken_uniformly():
    # Integer draws: most obs tie with members; random tie-breaking keeps it flat.
    rng = np.random.default_rng(2)
    stations = [f"S{i}" for i in range(200)]
    leads = np.arange(6, 366, 6.0)
    m = 5
    fc = rng.integers(0, 3, size=(200, m, leads.size)).astype(float)
    obs = rng.integers(0, 3, size=(200, leads.size)).astype(float)
    rows = scorecard.pair_scores(_many(fc, obs, stations, leads), "temperature_2m")
    _flat(rows["rank"].to_numpy(), m, 0.15)


def test_crps_hc_is_crps_of_members_plus_lapse_shift():
    fc = np.array([[1.0, 4.0], [2.0, 5.0], [6.0, 3.0]])
    obs = [2.0, 4.5]
    rows = scorecard.pair_scores(_matched(fc, obs), "temperature_2m",
                                 dz={"S1": 100.0})
    expected = [verification.fair_crps(fc[:, i] + 0.65, obs[i]) for i in range(2)]
    np.testing.assert_allclose(rows["crps_hc"], expected, atol=1e-12)
    np.testing.assert_allclose(rows["err_hc"], fc.mean(0) + 0.65 - obs, atol=1e-12)
    assert (rows["dz"] == 100.0).all()


def test_deterministic_store_perturbed_draws():
    m = _matched([[3.0, -1.0]], [1.0, 2.0])
    m.attrs["store"] = "gfs"
    rows = scorecard.pair_scores(m, "temperature_2m", dz={"S1": 20.0})
    assert (rows["n_members"] == 1).all()
    assert SCORECARD_DETERMINISTIC_DRAWS == 50
    assert (rows["n_members_pert"] == 50).all()
    assert np.isfinite(rows["crps_pert"]).all()
    assert np.isfinite(rows["spread2_pert"]).all() and (rows["spread2_pert"] > 0).all()
    assert rows["rank"].isna().all()
    assert rows["rank_pert"].between(0, 50).all()
    wind = scorecard.pair_scores(_matched([[3.0, 5.0]], [1.0, 2.0],
                                          "wind_speed_10m"), "wind_speed_10m")
    assert np.isfinite(wind["crps_pert"]).all()


def test_station_without_dz_has_nan_corrections_only_there():
    rng = np.random.default_rng(3)
    stations = ["S1", "S2"]
    leads = np.array([6.0, 12.0])
    fc = rng.normal(10, 1, size=(2, 4, 2))
    obs = rng.normal(10, 1, size=(2, 2))
    rows = scorecard.pair_scores(_many(fc, obs, stations, leads), "temperature_2m",
                                 dz={"S1": 30.0})
    s1, s2 = rows[rows["station"] == "S1"], rows[rows["station"] == "S2"]
    for col in ("err_hc", "crps_hc", "crps_pert", "err_pert", "spread2_pert",
                "rank_pert", "dz"):
        assert s2[col].isna().all(), col
        assert np.isfinite(s1[col]).all(), col
    assert np.isfinite(s2["crps"]).all()  # raw scores unaffected


def test_corrections_nan_without_any_dz_and_for_other_variables():
    rows = scorecard.pair_scores(_matched([[1.0], [2.0]], [1.0]), "temperature_2m")
    assert rows[["dz", "err_hc", "crps_hc", "crps_pert"]].isna().all().all()
    wind = scorecard.pair_scores(_matched([[1.0], [2.0]], [1.0], "wind_speed_10m"),
                                 "wind_speed_10m", dz={"S1": 30.0})
    assert wind[["dz", "err_hc", "crps_hc"]].isna().all().all()
    assert np.isfinite(wind["crps_pert"]).all()


def test_perturbation_reproducible_and_seeded_per_item():
    m = _matched(np.random.default_rng(4).normal(size=(10, 3)), [0.1, 0.2, 0.3],
                 "wind_speed_10m")
    a = scorecard.pair_scores(m, "wind_speed_10m")
    b = scorecard.pair_scores(m, "wind_speed_10m")
    pd.testing.assert_frame_equal(a, b)
    other = m.copy()
    other.attrs = {**m.attrs, "init": (INIT + pd.Timedelta("1D")).isoformat()}
    c = scorecard.pair_scores(other, "wind_speed_10m")
    assert not np.allclose(a["crps_pert"], c["crps_pert"])


def test_obs_persist_lead_30_and_6():
    leads = [6.0, 24.0, 30.0]
    cube = _cube(np.zeros((1, 2, 3)), leads)
    times = pd.date_range(INIT - pd.Timedelta("3D"), INIT + pd.Timedelta("2D"), freq="h")
    values = (times - INIT) / pd.Timedelta("1h")  # hours relative to init
    m = scorecard.match_station_obs(cube, {"S1": _obs(times, temperature_2m=values)},
                                    "temperature_2m")
    persist = m["obs_persist"].sel(station="S1").values
    # lead 6: valid - 24 h = init - 18 h; lead 24: init; lead 30: valid - 48 h.
    np.testing.assert_allclose(persist, [-18.0, 0.0, -18.0])
    assert (persist <= 0).all()
    rows = scorecard.pair_scores(m, "temperature_2m")
    np.testing.assert_allclose(rows["obs_persist"], [-18.0, 0.0, -18.0])


def test_obs_persist_nan_without_a_report():
    cube = _cube(np.zeros((1, 2, 1)), [6.0])
    obs = _obs([INIT + pd.Timedelta("6h")], temperature_2m=[1.0])
    m = scorecard.match_station_obs(cube, {"S1": obs}, "temperature_2m")
    assert np.isnan(m["obs_persist"].item(0))


def test_precip_obs_persist_is_the_reported_total_ending_then():
    cube = _cube([[[1.0]]], [30], variable=V24, window=24)
    prev = INIT - pd.Timedelta("18h")  # 06 UTC the day before init: valid - 48 h
    obs = _obs([prev, END30], precip_total_24h=[4.0, 2.5],
               precipitation_1h=[np.nan, np.nan])
    m = scorecard.match_station_obs(cube, {"S1": obs}, V24)
    assert m["obs_persist"].item(0) == 4.0


def test_precip_thresholds_columns():
    # 4 members: 0, 2, 6, 12 mm; obs 7 mm.
    fc = [[0.0], [2.0], [6.0], [12.0]]
    rows = scorecard.pair_scores(_matched(fc, [7.0], V24), V24)
    r = rows.iloc[0]
    assert (r["p_1"], r["p_5"], r["p_10"]) == (0.75, 0.5, 0.25)
    assert (r["wet_fc_1"], r["wet_fc_5"], r["wet_fc_10"]) == (1.0, 1.0, 0.0)
    assert (r["wet_obs_1"], r["wet_obs_5"], r["wet_obs_10"]) == (1.0, 1.0, 0.0)
    m = 4
    for t, i, o in ((1, 3, 1), (5, 2, 1), (10, 1, 0)):
        expected = (i / m - o) ** 2 - i * (m - i) / (m**2 * (m - 1))
        assert r[f"brier_{t}"] == pytest.approx(expected)
        assert 0.0 <= r[f"p_pert_{t}"] <= 1.0
        assert np.isfinite(r[f"brier_pert_{t}"])
    # headline columns unchanged, at threshold 1
    assert r["brier"] == pytest.approx(r["brier_1"])
    assert r["wet_fc"] == r["wet_fc_1"]


def test_precip_deterministic_brier_raw_nan_perturbed_finite():
    rows = scorecard.pair_scores(_matched([[6.0]], [7.0], V24), V24)
    r = rows.iloc[0]
    assert np.isnan(r["brier_5"]) and r["p_5"] == 1.0
    assert np.isfinite(r["brier_pert_5"]) and r["n_members_pert"] == 50


def test_threshold_columns_nan_for_temperature():
    rows = scorecard.pair_scores(_matched([[1.0], [2.0]], [1.0]), "temperature_2m")
    assert rows[["p_1", "p_pert_5", "brier_10", "wet_obs_1"]].isna().all().all()



# --- Task 5 fix round 1: rank_hc (Ruling V5) ----------------------------------------


def test_rank_hc_ranks_the_obs_among_the_height_corrected_members():
    # Members 10, 11, 12 and obs 11.5: raw rank 2. With dz = +100 m the members
    # become 10.65, 11.65, 12.65, so the obs falls to rank 1.
    stations = ["S1", "S2"]
    leads = np.array([6.0])
    fc = np.array([[[10.0], [11.0], [12.0]]] * 2)
    obs = np.array([[11.5], [11.5]])
    rows = scorecard.pair_scores(_many(fc, obs, stations, leads), "temperature_2m",
                                 dz={"S1": 100.0})
    s1, s2 = rows[rows["station"] == "S1"].iloc[0], rows[rows["station"] == "S2"].iloc[0]
    assert s1["rank"] == 2.0 and s1["rank_hc"] == 1.0
    assert np.isnan(s2["rank_hc"]) and s2["rank"] == 2.0  # no dz: NaN, never raw


def test_rank_hc_is_nan_for_one_member_and_other_variables():
    rows = scorecard.pair_scores(_matched([[1.0]], [1.0]), "temperature_2m",
                                 dz={"A": 10.0})
    assert rows["rank_hc"].isna().all()
    stations = ["S1"]
    fc = np.array([[[1.0], [2.0], [3.0]]])
    wind = scorecard.pair_scores(
        _many(fc, np.array([[2.5]]), stations, np.array([6.0]),
              variable="wind_speed_10m"), "wind_speed_10m")
    assert wind["rank_hc"].isna().all()


def test_rank_hc_is_reproducible_and_finite_with_dz():
    """Same item, same numbers (rank_hc is drawn last, after rank_pert)."""
    rng = np.random.default_rng(3)
    stations = ["S1", "S2"]
    leads = np.array([6.0, 12.0])
    fc = np.round(rng.normal(10, 1, size=(2, 4, 2)))
    obs = np.round(rng.normal(10, 1, size=(2, 2)))
    m = _many(fc, obs, stations, leads)
    a = scorecard.pair_scores(m, "temperature_2m", dz={"S1": 30.0, "S2": 0.0})
    b = scorecard.pair_scores(m, "temperature_2m", dz={"S1": 30.0, "S2": 0.0})
    pd.testing.assert_frame_equal(a, b)
    assert np.isfinite(a["rank_hc"]).all()
