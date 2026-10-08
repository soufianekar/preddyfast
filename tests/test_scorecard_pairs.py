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


def test_precip_prefers_reported_total():
    cube = _cube([[[1.0]]], [6], variable="precipitation_6h", window=6)
    end = INIT + pd.Timedelta("6h")
    hours = _hours(end, 6)
    obs = _obs(hours, precipitation_1h=[1.0] * 6, precip_total_6h=[np.nan] * 5 + [2.5])
    m = scorecard.match_station_obs(cube, {"S1": obs}, "precipitation_6h")
    assert m["obs"].item(0) == 2.5


def test_precip_falls_back_to_complete_hourly_sum():
    cube = _cube([[[1.0]]], [6], variable="precipitation_6h", window=6)
    end = INIT + pd.Timedelta("6h")
    obs = _obs(_hours(end, 6), precipitation_1h=[0.5, 0.0, 1.0, 0.0, 0.25, 0.25],
               precip_total_6h=[np.nan] * 6)
    m = scorecard.match_station_obs(cube, {"S1": obs}, "precipitation_6h")
    assert m["obs"].item(0) == pytest.approx(2.0)


def test_precip_missing_hour_without_report_gives_no_row():
    cube = _cube([[[1.0]]], [6], variable="precipitation_6h", window=6)
    end = INIT + pd.Timedelta("6h")
    hours = _hours(end, 6).delete(2)  # one hour of the six missing
    obs = _obs(hours, precipitation_1h=[1.0] * 5, precip_total_6h=[np.nan] * 5)
    m = scorecard.match_station_obs(cube, {"S1": obs}, "precipitation_6h")
    assert np.isnan(m["obs"].item(0))  # never a partial sum
    assert scorecard.pair_scores(m, "precipitation_6h").empty


def test_report_not_ending_at_valid_time_is_not_used():
    cube = _cube([[[1.0]]], [6], variable="precipitation_6h", window=6)
    obs = _obs([INIT + pd.Timedelta("7h")], precip_total_6h=[3.0], precipitation_1h=[np.nan])
    m = scorecard.match_station_obs(cube, {"S1": obs}, "precipitation_6h")
    assert np.isnan(m["obs"].item(0))


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
    leads = leads or [6 * (i + 1) for i in range(fc.shape[1])]
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
    threshold = SCORECARD_VARIABLES["precipitation_6h"]["threshold"]
    # Leads: mean exactly at threshold (wet), mean just below (dry), mean well above.
    fc = [[threshold, threshold - 0.01, 0.0], [threshold, threshold - 0.01, 3.0]]
    obs = [threshold, 0.0, 0.59]
    rows = scorecard.pair_scores(_matched(fc, obs, "precipitation_6h"), "precipitation_6h")
    assert list(rows["wet_fc"]) == [1.0, 0.0, 1.0]
    assert list(rows["wet_obs"]) == [1.0, 0.0, 0.0]


def test_brier_for_ensembles_fair_and_inclusive():
    threshold = SCORECARD_VARIABLES["precipitation_6h"]["threshold"]
    # 1 of 4 members wet (one exactly at the threshold), obs wet.
    fc = [[threshold], [0.0], [0.0], [0.0]]
    rows = scorecard.pair_scores(_matched(fc, [1.0], "precipitation_6h"), "precipitation_6h")
    i, m = 1, 4
    expected = (i / m - 1) ** 2 - i * (m - i) / (m**2 * (m - 1))
    assert rows["brier"].iloc[0] == pytest.approx(expected)


def test_brier_nan_for_deterministic_and_non_brier_variables():
    det = scorecard.pair_scores(_matched([[1.0]], [1.0], "precipitation_6h"), "precipitation_6h")
    assert det["brier"].isna().all()
    assert det["wet_fc"].iloc[0] == 1.0  # contingency counts still apply
    temp = scorecard.pair_scores(_matched([[1.0], [2.0]], [1.0]), "temperature_2m")
    assert temp["brier"].isna().all()
    assert temp["wet_fc"].isna().all() and temp["wet_obs"].isna().all()


def test_row_columns_and_init():
    rows = scorecard.pair_scores(_matched([[1.0], [2.0]], [1.0]), "temperature_2m")
    assert list(rows.columns) == [
        "station", "init", "lead_hours", "valid_time", "obs", "fc_mean",
        "err", "ae", "se", "crps", "spread2", "wet_fc", "wet_obs", "brier",
    ]
    assert rows["init"].iloc[0] == INIT
    assert rows["valid_time"].iloc[0] == INIT + pd.Timedelta("6h")
    assert rows["station"].iloc[0] == "S1"


def test_no_valid_pairs_gives_empty_frame_with_columns():
    rows = scorecard.pair_scores(_matched([[1.0], [2.0]], [np.nan]), "temperature_2m")
    assert rows.empty
    assert "crps" in rows.columns


def test_wet_flag_survives_a_float32_cube():
    # station_cube returns float32. These members' exact mean is >= 0.6, but a
    # float32 mean lands one ulp below it and would flag the pair dry.
    members = np.array([0.6000002026557922, 0.6000000834465027, 0.599999725818634], dtype=np.float32)
    cube = _cube(members.reshape(1, 3, 1), [6], variable="precipitation_6h", window=6)
    cube = cube.astype("float32")
    obs = _obs([INIT + pd.Timedelta("6h")], precipitation_1h=[0.0], precip_total_6h=[0.7])
    m = scorecard.match_station_obs(cube, {"S1": obs}, "precipitation_6h")
    assert m["fc"].dtype == np.float64
    rows = scorecard.pair_scores(m, "precipitation_6h")
    assert list(rows["wet_fc"]) == [1.0]
