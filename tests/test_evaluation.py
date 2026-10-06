import numpy as np
import pandas as pd
import pytest

from utils import evaluation as ev

STATION = {"id": "FRM00007156", "name": "PARIS-MONTSOURIS", "lat": 48.8167, "lon": 2.3333, "elev_m": 77.0, "dist_km": 3.9}
INITS = [pd.Timestamp("2026-09-30 00:00"), pd.Timestamp("2026-09-29 00:00")]


def _fc(store, variable, lat, lon, inits, accum_hours):
    rows = []
    for init in inits:
        for lead in (24.0, 48.0, 96.0):
            for m in range(3):
                rows.append({"init_time": init, "valid_time": init + pd.Timedelta(hours=lead),
                             "lead_hours": lead, "member": m, "value": 10.0 + m + (1.0 if store == "aifs_ens" else 0.0)})
    return pd.DataFrame(rows)


def _obs(station_id, year):
    # Hourly obs up to 2026-10-01 12 UTC only: T+96 of both runs is in the future.
    t = pd.date_range("2026-09-28", "2026-10-01 12:00", freq="h", tz="UTC")
    return pd.DataFrame({"valid_time": t, "temperature_2m": 11.0, "precipitation_1h": 0.5})


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(ev, "eval_inits", lambda stores, n, hour=None, before=None: INITS[:n])
    monkeypatch.setattr(ev, "check_read_budget", lambda *a, **k: None)
    monkeypatch.setattr(ev, "point_archive", _fc)
    monkeypatch.setattr(ev, "fetch_station_year", _obs)
    monkeypatch.setattr(ev, "last_obs_time", lambda station_id, column: pd.Timestamp("2026-10-01 12:00"))
    monkeypatch.setattr(ev, "store_variables", lambda store: ["temperature_2m", "precipitation_surface"])
    monkeypatch.setattr(ev, "_run_chunks", lambda store, lat, lon: 1)


def test_each_model_scored_separately(wired):
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "aifs_ens"), 2)
    ifs = out["results"]["ifs_ens"]["scores"].set_index("lead_hours")
    aifs = out["results"]["aifs_ens"]["scores"].set_index("lead_hours")
    assert ifs.loc[24.0, "bias"] == pytest.approx(0.0)
    assert aifs.loc[24.0, "bias"] == pytest.approx(1.0)


def test_future_leads_have_no_score(wired):
    scores = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens",), 2)["results"]["ifs_ens"]["scores"]
    assert 96.0 not in set(scores["lead_hours"])
    assert scores.set_index("lead_hours").loc[24.0, "n"] == 2


def test_precipitation_uses_complete_24h_windows(wired):
    scores = ev.evaluate_point(STATION, "precipitation_24h", ("ifs_ens",), 2)["results"]["ifs_ens"]["scores"]
    # obs: 24 complete hours of 0.5 mm -> 12 mm; forecast mean 11 mm -> bias -1
    assert scores.set_index("lead_hours").loc[24.0, "bias"] == pytest.approx(-1.0)


def test_no_runs_is_a_user_facing_error(wired, monkeypatch):
    monkeypatch.setattr(ev, "eval_inits", lambda stores, n, hour=None, before=None: [])
    with pytest.raises(ev.GridDataError):
        ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens",), 5)


def test_precipitation_obs_returned_as_window_totals(wired):
    out = ev.evaluate_point(STATION, "precipitation_24h", ("ifs_ens",), 2)
    # 24 complete hours of 0.5 mm: the plotted obs must be 12 mm totals, not 0.5 mm hours.
    assert out["obs"]["value"].dropna().eq(12.0).all() and not out["obs"].dropna().empty


def test_cold_reads_are_capped_per_request(wired, monkeypatch):
    three = [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-09-29"), pd.Timestamp("2026-09-28")]
    monkeypatch.setattr(ev, "eval_inits", lambda stores, n, hour=None, before=None: three[:n])
    monkeypatch.setattr(ev, "point_run_cached", lambda *a: False)
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "aifs_ens"), 3, max_cold=4)
    # 2 models x 2 runs = 4 cold reads; the third run waits for the next request.
    assert out["pending"] == 1
    assert out["inits"] == three[:2]
    for r in out["results"].values():
        assert set(r["matched"]["init_time"]) == set(three[:2])


def test_cached_runs_cost_nothing(wired, monkeypatch):
    monkeypatch.setattr(ev, "point_run_cached", lambda *a: True)
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "aifs_ens"), 2, max_cold=0)
    assert out["pending"] == 0 and len(out["inits"]) == 2


def test_runs_are_00utc_and_end_at_the_last_observation(wired, monkeypatch):
    asked = {}

    def fake(stores, n, hour=None, before=None):
        asked.update(n=n, hour=hour, before=before)
        return INITS[:n]

    monkeypatch.setattr(ev, "eval_inits", fake)
    out = ev.evaluate_point(STATION, "temperature_2m", ("aifs_ens",), 30)
    # GHCNh lags by days: newer runs than the last report have nothing to verify.
    assert asked == {"n": 30, "hour": 0, "before": pd.Timestamp("2026-10-01 12:00")}
    assert out["obs_end"] == pd.Timestamp("2026-10-01 12:00")


def test_model_without_the_variable_is_reported_not_read(wired, monkeypatch):
    read = []
    monkeypatch.setattr(ev, "store_variables", lambda s: ["temperature_2m"] if s == "ifs_ens" else ["precipitation_surface"])
    monkeypatch.setattr(ev, "point_archive", lambda store, *a: read.append(store) or _fc(store, *a))
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "gefs"), 2)
    assert read == ["ifs_ens"] and set(out["results"]) == {"ifs_ens"}
    assert any("NOAA GEFS" in e for e in out["errors"])


def test_no_model_with_the_variable_is_a_user_facing_error(wired, monkeypatch):
    monkeypatch.setattr(ev, "store_variables", lambda s: [])
    with pytest.raises(ev.GridDataError):
        ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens",), 2)


def test_station_with_no_reports_of_the_variable(wired, monkeypatch):
    monkeypatch.setattr(ev, "last_obs_time", lambda station_id, column: None)
    with pytest.raises(ev.ObservationError):
        ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens",), 2)


def test_budget_counts_only_the_runs_this_request_reads_cold(wired, monkeypatch):
    asked = []
    monkeypatch.setattr(ev, "check_read_budget", lambda stores, box, n_inits: asked.append(n_inits))
    cached = {pd.Timestamp("2026-09-30 00:00").strftime("%Y-%m-%dT%H:%M")}
    monkeypatch.setattr(ev, "point_run_cached", lambda lat, lon, var, stamp, accum, store: stamp in cached)
    ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens",), 2)
    # 30 cached runs must not count against the budget; only the cold one does.
    assert asked == [1]


def test_cold_cap_counts_chunks_not_runs(wired, monkeypatch):
    three = [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-09-29"), pd.Timestamp("2026-09-28")]
    monkeypatch.setattr(ev, "eval_inits", lambda stores, n, hour=None, before=None: three[:n])
    monkeypatch.setattr(ev, "point_run_cached", lambda *a: False)
    # A GEFS point run spans 3 lead blocks: 8 chunks fit two runs, not eight.
    monkeypatch.setattr(ev, "_run_chunks", lambda store, lat, lon: 3)
    out = ev.evaluate_point(STATION, "temperature_2m", ("gefs",), 3, max_cold=8)
    assert out["inits"] == three[:2] and out["pending"] == 1


def test_control_and_ens_share_one_cold_read(wired, monkeypatch):
    three = [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-09-29"), pd.Timestamp("2026-09-28")]
    monkeypatch.setattr(ev, "eval_inits", lambda stores, n, hour=None, before=None: three[:n])
    monkeypatch.setattr(ev, "point_run_cached", lambda *a: False)
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "ifs_control"), 3, max_cold=2)
    # One chunk per run for the pair, not two: two runs fit a 2-chunk cap.
    assert out["inits"] == three[:2]


# --- references other than stations -----------------------------------------
REF_INITS = [pd.Timestamp("2026-09-28") + pd.Timedelta(days=d) for d in range(6)]  # 28 Sep .. 3 Oct, 00 UTC


def _fc_t0(store, variable, lat, lon, inits, accum_hours):
    rows = []
    for init in inits:
        for lead in (0.0, 24.0, 48.0):
            for m in ((0,) if store == "ifs_control" else range(3)):
                # Lead 0 is the run's own initial state; later leads drift by +2.
                value = 10.0 + m + (0.0 if lead == 0 else 2.0) + (1.0 if store == "aifs_ens" else 0.0)
                rows.append({"init_time": init, "valid_time": init + pd.Timedelta(hours=lead),
                             "lead_hours": lead, "member": m, "value": value})
    return pd.DataFrame(rows)


@pytest.fixture
def gridded(wired, monkeypatch):
    monkeypatch.setattr(ev, "point_archive", _fc_t0)
    monkeypatch.setattr(ev, "reference_inits", lambda store: REF_INITS)
    monkeypatch.setattr(ev, "store_leads", lambda store: np.arange(0.0, 49.0, 6.0))


def test_control_t0_reference_scores_every_model_against_ifs_control(gridded):
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "aifs_ens"), 2, reference="control_t0", lead_step=24)
    ifs = out["results"]["ifs_ens"]["scores"].set_index("lead_hours")
    aifs = out["results"]["aifs_ens"]["scores"].set_index("lead_hours")
    # Control T+0 is 10; IFS ENS mean at T+24 is 13, AIFS ENS 14.
    assert ifs.loc[24.0, "bias"] == pytest.approx(3.0) and aifs.loc[24.0, "bias"] == pytest.approx(4.0)
    assert out["reference"] == "control_t0"


def test_own_t0_reference_uses_each_models_initial_state(gridded):
    out = ev.evaluate_point(STATION, "temperature_2m", ("aifs_ens",), 2, reference="own_t0", lead_step=24)
    # AIFS ENS T+0 mean is 12, its T+24 mean 14.
    assert out["results"]["aifs_ens"]["scores"].set_index("lead_hours").loc[24.0, "bias"] == pytest.approx(2.0)


def test_gridded_reference_is_not_limited_by_the_station_lag(gridded, monkeypatch):
    asked = {}
    monkeypatch.setattr(ev, "eval_inits", lambda stores, n, hour=None, before=None: asked.update(before=before) or INITS[:n])
    ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens",), 2, reference="control_t0")
    assert asked["before"] is None


def test_precipitation_cannot_be_verified_against_t0(gridded):
    with pytest.raises(ev.GridDataError, match="T\\+0"):
        ev.evaluate_point(STATION, "precipitation_24h", ("ifs_ens",), 2, reference="control_t0")


def test_reference_runs_count_against_the_cold_cap(gridded, monkeypatch):
    monkeypatch.setattr(ev, "point_run_cached", lambda *a: False)
    out = ev.evaluate_point(STATION, "temperature_2m", ("aifs_ens",), 2, reference="control_t0", lead_step=24, max_cold=3)
    # Newest run: AIFS read + control runs valid at its T+24 and T+48 = 3 chunks; the next run waits.
    assert out["inits"] == INITS[:1] and out["pending"] == 1


def test_lead_step_keeps_leads_every_model_shares(wired):
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "aifs_ens"), 2, lead_step=48)
    for r in out["results"].values():
        assert set(r["scores"]["lead_hours"]) <= {48.0}


def test_precipitation_carries_brier_scores(wired):
    out = ev.evaluate_point(STATION, "precipitation_24h", ("ifs_ens",), 2)
    assert {"brier_1", "brier_5", "brier_10"} <= set(out["results"]["ifs_ens"]["scores"].columns)
    assert not out["results"]["ifs_ens"]["per_run"].empty


def test_persist_scores_writes_each_models_per_run_rows(wired, monkeypatch):
    saved = []
    monkeypatch.setattr(ev, "save_per_run", lambda *a: saved.append(a))
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "aifs_ens"), 2)
    ev.persist_scores(out, STATION, "temperature_2m")
    assert [(a[0], a[1], a[2], a[3]) for a in saved] == [
        ("stations", "ifs_ens", "temperature_2m", "FRM00007156"),
        ("stations", "aifs_ens", "temperature_2m", "FRM00007156"),
    ]
    assert all(not a[4].empty for a in saved)


def test_own_t0_with_several_models_returns_no_pooled_reference_line(gridded):
    out = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "aifs_ens"), 2, reference="own_t0", lead_step=24)
    assert out["obs"].empty  # two models' initial states are not one series
    one = ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens", "aifs_ens"), 2, reference="control_t0", lead_step=24)
    assert not one["obs"].empty


def test_refresh_scores_runs_every_reference_until_nothing_is_pending(wired, monkeypatch):
    stations = pd.DataFrame([{**STATION, "dist_km": 0.0}]).drop(columns=["dist_km"])
    monkeypatch.setattr(ev, "station_list", lambda: stations)
    seen = []

    def fake_eval(station, variable, stores, n, reference="stations", lead_step=24, max_cold=8, hour=0):
        seen.append((variable, reference, stores, max_cold))
        return {"results": {}, "reference": reference, "pending": 0}

    monkeypatch.setattr(ev, "evaluate_point", fake_eval)
    monkeypatch.setattr(ev, "persist_scores", lambda out, st, var: None)
    n = ev.refresh_scores(("FRM00007156",), ("ifs_ens", "gefs"))
    refs = {(v, r) for v, r, _, _ in seen}
    # Precipitation only against stations; instantaneous fields against all three.
    assert ("precipitation_24h", "stations") in refs and ("precipitation_24h", "control_t0") not in refs
    assert ("temperature_2m", "own_t0") in refs and ("temperature_2m", "control_t0") in refs
    assert all(mc >= 10_000 for *_, mc in seen) and n == len(seen)


# --- precipitation: verify only the windows a station (or IMERG) can supply ---
def _koeln_like(station_id, year):
    # 4 reports a day (no complete hours) but 06–06 UTC totals of 12 mm.
    t = pd.date_range("2026-09-28", "2026-10-01 18:00", freq="6h", tz="UTC")
    frame = pd.DataFrame({"valid_time": t, "temperature_2m": 11.0, "precipitation_1h": 0.5,
                          "precip_total_6h": np.nan, "precip_total_12h": np.nan, "precip_total_24h": np.nan})
    frame.loc[frame["valid_time"].dt.hour == 6, "precip_total_24h"] = 12.0
    return frame


def _fc_6h(store, variable, lat, lon, inits, accum_hours):
    rows = []
    for init in inits:
        for lead in (24.0, 30.0, 48.0, 54.0):
            for m in range(3):
                rows.append({"init_time": init, "valid_time": init + pd.Timedelta(hours=lead),
                             "lead_hours": lead, "member": m, "value": 10.0 + m})
    return pd.DataFrame(rows)


def test_station_reported_daily_totals_align_leads_to_their_end_hour(wired, monkeypatch):
    monkeypatch.setattr(ev, "fetch_station_year", _koeln_like)
    monkeypatch.setattr(ev, "point_archive", _fc_6h)
    out = ev.evaluate_point(STATION, "precipitation_24h", ("ifs_ens",), 2)
    scores = out["results"]["ifs_ens"]["scores"].set_index("lead_hours")
    # 06–06 totals of 12 mm against a forecast mean of 11: T+30 and T+54, bias -1.
    assert set(scores.index) <= {30.0, 54.0} and scores.loc[30.0, "bias"] == pytest.approx(-1.0)
    assert out["precip_source"] == {"source": "reported", "hours": 24, "end_hour": 6, "count": 4}


def test_station_with_no_verifiable_window_points_to_imerg(wired, monkeypatch):
    def gaps(station_id, year):
        f = _koeln_like(station_id, year)
        f["precip_total_24h"] = np.nan
        return f

    monkeypatch.setattr(ev, "fetch_station_year", gaps)
    with pytest.raises(ev.ObservationError, match="IMERG"):
        ev.evaluate_point(STATION, "precipitation_24h", ("ifs_ens",), 2)


def test_imerg_reference_scores_24h_totals(wired, monkeypatch):
    t = pd.date_range("2026-09-25", "2026-10-03", freq="30min")
    monkeypatch.setattr(ev, "imerg_halfhours", lambda lat, lon, start, end: pd.DataFrame({"time": t, "mm": 0.25}))
    monkeypatch.setattr(ev, "point_archive", _fc_6h)
    out = ev.evaluate_point(STATION, "precipitation_24h", ("ifs_ens",), 2, reference="imerg")
    scores = out["results"]["ifs_ens"]["scores"].set_index("lead_hours")
    assert scores.loc[24.0, "bias"] == pytest.approx(-1.0)  # 48 x 0.25 mm = 12 mm observed


def test_imerg_is_for_accumulated_variables_only(wired):
    with pytest.raises(ev.GridDataError, match="precipitation"):
        ev.evaluate_point(STATION, "temperature_2m", ("ifs_ens",), 2, reference="imerg")
