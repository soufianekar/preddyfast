"""scorecard.refresh and the scores_db scorecard files (no network, no reads)."""

import logging

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils import scorecard, scores_db, station_network
from utils.earthmover_api import GridDataError
from utils.observations import ObservationError
from utils.settings import SCORECARD_DAYS, SCORECARD_OBS_GRACE_HOURS

TODAY = pd.Timestamp("2026-10-06")
GRACE = pd.Timedelta(hours=SCORECARD_OBS_GRACE_HOURS)
STORES = ("ifs_ens", "gfs")
VARS = ("temperature_2m",)


def _inits(n=5, store_missing=()):
    return [TODAY - pd.Timedelta(days=d) for d in range(1, n + 1)]


def _network(ids=("A", "B")):
    return pd.DataFrame(
        {
            "id": list(ids),
            "name": list(ids),
            "lat": [50.0 + i for i in range(len(ids))],
            "lon": [10.0] * len(ids),
            "elev_m": [100.0] * len(ids),
            "ok_temperature": True,
            "ok_precip": True,
            "ok_wind": True,
        }
    )


def _cube(store, variable, init, stations, accum, leads=(0.0, 6.0, 12.0)):
    leads = np.asarray(leads, dtype=float)
    n = len(stations)
    return xr.DataArray(
        np.full((n, 2, len(leads)), 10.0, dtype="float32"),
        dims=("station", "member", "lead_hours"),
        coords={
            "station": [str(i) for i in stations["id"]],
            "member": [0, 1],
            "lead_hours": leads,
            "valid_time": ("lead_hours", init + pd.to_timedelta(leads, unit="h")),
        },
        attrs={"init": init.isoformat(), "store": store, "variable": variable,
               "window_hours": accum},
    )


def _obs_frame(end):
    times = pd.date_range(end=end, periods=24 * 100, freq="h", tz="UTC")
    return pd.DataFrame({"valid_time": times, "temperature_2m": 9.0})


class Env:
    """Stubs for everything that would touch a network, with call records."""

    def __init__(self, monkeypatch, tmp_path, obs_end="2026-10-06 00:00"):
        self.calls = []
        self.fail = set()
        self.obs_end = pd.Timestamp(obs_end)
        self.selected = 0
        self.saved = None
        self.net = _network()
        self.inits = {s: _inits(100) for s in STORES}
        self.held = None
        self.obs_calls = 0
        self.leads = (0.0, 6.0, 12.0)
        self.late = {}  # station -> its own last report time
        self.configured = ()
        monkeypatch.setattr(scorecard, "SCORECARD_STATIONS", ())  # general mode
        monkeypatch.setattr(scores_db, "SCORECARD_DIR", str(tmp_path / "scorecard"))
        monkeypatch.setattr(scorecard, "PRECOMPUTE_DIR", str(tmp_path / "precomputed"))
        monkeypatch.setattr(scorecard.earthmover_api, "station_cube", self.station_cube)
        monkeypatch.setattr(scorecard.earthmover_api, "available_inits", lambda s: sorted(self.inits[s]))
        monkeypatch.setattr(scorecard.earthmover_api, "store_variables",
                            lambda s: ["temperature_2m", "precipitation_surface"])
        monkeypatch.setattr(station_network, "load_network", lambda: self.saved)
        monkeypatch.setattr(station_network, "select_network", self.select_network)
        monkeypatch.setattr(station_network, "save_network", self.save_network)
        monkeypatch.setattr(station_network, "load_configured", lambda: self.configured)
        monkeypatch.setattr(station_network, "network_obs", self.network_obs)

    def station_cube(self, store, variable, init, stations, accum, budget_mb=None, window_end_hour=None):
        self.calls.append((store, variable, init, budget_mb))
        if (store, init) in self.fail:
            raise GridDataError("incomplete")
        return _cube(store, variable, init, stations, accum, self.leads)

    def select_network(self, start, end):
        self.selected += 1
        return self.net

    def save_network(self, net, start, end, configured=()):
        self.saved = (net, pd.Timestamp(start), pd.Timestamp(end))
        self.configured = tuple(sorted(configured))

    def network_obs(self, ids, start, end):
        self.obs_calls += 1
        self.obs_span = (pd.Timestamp(start), pd.Timestamp(end))
        return {i: _obs_frame(self.late.get(i, self.obs_end)) for i in ids}


@pytest.fixture
def env(monkeypatch, tmp_path):
    return Env(monkeypatch, tmp_path)


def _files(tmp_path):
    return sorted((tmp_path / "scorecard").glob("*/*/*.parquet"))


def test_first_run_writes_every_init_and_second_run_nothing(env, tmp_path):
    n = scorecard.refresh(STORES, VARS, TODAY)
    assert n == 2 * SCORECARD_DAYS
    assert len(_files(tmp_path)) == 2 * SCORECARD_DAYS
    calls = len(env.calls)
    assert scorecard.refresh(STORES, VARS, TODAY) == 0
    assert len(env.calls) == calls  # skipped without touching station_cube


def test_budget_is_passed_and_one_obs_call(env):
    from utils.settings import SCORECARD_READ_BUDGET_MB

    scorecard.refresh(STORES, VARS, TODAY)
    assert {c[3] for c in env.calls} == {SCORECARD_READ_BUDGET_MB}
    assert env.obs_calls == 1


def test_window_ends_the_day_before_the_last_report(env, tmp_path):
    # median last report 10-06 00:00 (no grace, R22) -> inits 09-22..10-05
    scorecard.refresh(("gfs",), VARS, TODAY)
    days = [p.stem for p in _files(tmp_path)]
    assert days[0] == "20260922" and days[-1] == "20261005"
    assert len(days) == SCORECARD_DAYS == 14


def test_window_with_a_six_day_observation_lag(env, tmp_path, caplog):
    env.obs_end = pd.Timestamp("2026-09-30 14:30")  # the real GHCNh lag on 6 Oct
    with caplog.at_level(logging.INFO):
        scorecard.refresh(("gfs",), VARS, TODAY)
    days = [p.stem for p in _files(tmp_path)]
    # anchored on the day of the median last report, grace not applied (R22)
    assert days[0] == "20260916" and days[-1] == "20260929" and len(days) == SCORECARD_DAYS
    assert "inits 2026-09-16 to 2026-09-29" in caplog.text
    assert "anchor 2026-09-30" in caplog.text


def test_observations_are_read_back_far_enough_for_the_lag(env):
    scorecard.refresh(("gfs",), VARS, TODAY)
    assert env.obs_calls == 1
    assert env.obs_span[0] <= TODAY - pd.Timedelta(days=SCORECARD_DAYS + 30)
    assert env.obs_span[1] == TODAY


def test_missing_inits_are_skipped_not_errors(env, tmp_path, caplog):
    env.inits["gfs"] = [TODAY - pd.Timedelta(days=d) for d in (3, 4)]
    with caplog.at_level(logging.INFO):
        n = scorecard.refresh(("gfs",), VARS, TODAY)
    assert n == 2
    assert "not in the store" in caplog.text


def test_scores_are_stored_with_metadata(env, tmp_path):
    scorecard.refresh(("gfs",), VARS, TODAY)
    init = TODAY - pd.Timedelta(days=3)
    meta = scores_db.scorecard_meta("gfs", "temperature_2m", init)
    assert meta["complete"] is True  # last lead (12 h) is before obs_end
    assert meta["n_pairs"] == 6 and meta["dropped_nan_fc"] == 0
    assert meta["network_hash"] == scores_db.station_hash(env.net)
    assert pd.Timestamp(meta["obs_end"]) == env.obs_end - GRACE
    rows = pd.read_parquet(scores_db.scorecard_path("gfs", "temperature_2m", init))
    assert set(rows["store"]) == {"gfs"} and set(scorecard.PAIR_COLUMNS) <= set(rows.columns)


def test_new_obs_rescore_only_inits_with_unverified_leads(env, tmp_path):
    env.leads = (0.0, 6.0, 12.0, 72.0)
    env.obs_end = pd.Timestamp("2026-10-03 00:00")  # window 09-19..10-02, obs_end 10-01
    scorecard.refresh(("gfs",), VARS, TODAY)
    meta = scores_db.scorecard_meta("gfs", "temperature_2m", pd.Timestamp("2026-09-30"))
    assert meta["complete"] is False
    env.calls.clear()
    env.obs_end = pd.Timestamp("2026-10-06 00:00")  # window 09-22..10-05, obs_end 10-04
    n = scorecard.refresh(("gfs",), VARS, TODAY)
    # 3 new inits (10-03..10-05) plus the 4 whose 72 h lead was unverified at
    # obs_end 10-01 (09-29..10-02); 09-22..09-28 were complete and are left alone.
    rescored = sorted(c[2] for c in env.calls)
    assert n == len(rescored) == 7
    assert rescored[0] == pd.Timestamp("2026-09-29")


def test_no_newer_obs_does_not_rescore_incomplete(env):
    env.leads = (0.0, 6.0, 12.0, 72.0)
    env.obs_end = pd.Timestamp("2026-10-03 00:00")
    scorecard.refresh(("gfs",), VARS, TODAY)
    env.calls.clear()
    assert scorecard.refresh(("gfs",), VARS, TODAY) == 0
    assert env.calls == []


def test_changed_network_rescores_everything(env):
    scorecard.refresh(("gfs",), VARS, TODAY)
    env.calls.clear()
    env.saved = (_network(("A", "B", "C")), TODAY - pd.Timedelta(days=SCORECARD_DAYS), TODAY)
    assert scorecard.refresh(("gfs",), VARS, TODAY) == SCORECARD_DAYS
    assert len(env.calls) == SCORECARD_DAYS


def test_old_inits_are_deleted(env, tmp_path):
    scorecard.refresh(("gfs",), VARS, TODAY)
    old = scores_db.scorecard_path("gfs", "temperature_2m", TODAY - pd.Timedelta(days=95))
    old.write_bytes(b"x")
    scorecard.refresh(("gfs",), VARS, TODAY)
    assert not old.exists() and len(_files(tmp_path)) == SCORECARD_DAYS


def test_rows_of_a_variable_no_longer_configured_are_pruned(env, tmp_path):
    scorecard.refresh(("gfs",), VARS, TODAY)
    stale = scores_db.scorecard_path("gfs", "precipitation_6h", TODAY)
    stale.parent.mkdir(parents=True, exist_ok=True)
    stale.write_bytes(b"x")
    kept = scores_db.scorecard_path("gfs", "temperature_2m", TODAY - pd.Timedelta(days=1))
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert not stale.exists() and not stale.parent.exists()
    assert kept.exists()


def test_network_selected_only_when_missing_old_or_reselect(env):
    scorecard.refresh(("gfs",), VARS, TODAY)
    assert env.selected == 1 and env.saved is not None
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert env.selected == 1  # saved window is current
    scorecard.refresh(("gfs",), VARS, TODAY, reselect=True, max_minutes=0)
    assert env.selected == 2
    env.saved = (env.net, TODAY - pd.Timedelta(days=300), TODAY - pd.Timedelta(days=200))
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert env.selected == 3


def test_variable_the_store_lacks_is_skipped(env, monkeypatch, caplog):
    monkeypatch.setattr(scorecard.earthmover_api, "store_variables", lambda s: ["temperature_2m"])
    with caplog.at_level(logging.INFO):
        n = scorecard.refresh(("gfs",), ("temperature_2m", "precipitation_24h"), TODAY)
    assert n == SCORECARD_DAYS
    assert "does not hold it" in caplog.text


def test_precipitation_reads_its_accumulation(env):
    scorecard.refresh(("gfs",), ("precipitation_24h",), TODAY, max_minutes=None)
    assert {c[1] for c in env.calls} == {"precipitation_surface"}


def test_z500_in_a_store_without_it_is_skipped_and_recorded(env, caplog):
    with caplog.at_level(logging.INFO):
        n = scorecard.refresh(("gfs",), ("z500",), TODAY)
    assert n == 0 and env.calls == []
    assert "does not hold it" in caplog.text
    assert scorecard.load_lacking() == {"z500": ["gfs"]}


def test_grid_error_leaves_the_others_written(env, tmp_path):
    bad = TODAY - pd.Timedelta(days=5)  # 10-01, inside the window
    env.fail.add(("gfs", bad))
    assert scorecard.refresh(("gfs",), VARS, TODAY) == SCORECARD_DAYS - 1
    assert not scores_db.scorecard_path("gfs", "temperature_2m", bad).exists()
    env.fail.clear()  # retried next run
    assert scorecard.refresh(("gfs",), VARS, TODAY) == 1


def test_observation_error_propagates(env, monkeypatch):
    def boom(*a, **k):
        raise ObservationError("down")

    monkeypatch.setattr(station_network, "network_obs", boom)
    with pytest.raises(ObservationError):
        scorecard.refresh(("gfs",), VARS, TODAY)


def test_max_minutes_stops_between_inits_and_resumes(env, monkeypatch, tmp_path):
    ticks = iter(range(0, 10_000))
    monkeypatch.setattr(scorecard, "_clock", lambda: next(ticks) * 30.0)  # 30 s per query
    n = scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=2)
    assert 0 < n < SCORECARD_DAYS
    again = scorecard.refresh(("gfs",), VARS, TODAY)
    assert again == SCORECARD_DAYS - n


def test_a_run_stopped_by_max_minutes_still_writes_the_aggregates(env, monkeypatch):
    """The workflow always passes --max-minutes: a cut-short run must publish what it scored."""
    ticks = iter(range(0, 10_000))
    monkeypatch.setattr(scorecard, "_clock", lambda: next(ticks) * 30.0)
    n = scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=2)
    assert 0 < n < SCORECARD_DAYS
    loaded = scorecard.load_aggregates()
    assert loaded is not None
    agg, meta = loaded
    assert set(agg["store"]) == {"gfs"} and meta["start"] <= meta["end"]


def test_backfill_walks_oldest_first(env):
    scorecard.refresh(("gfs",), VARS, TODAY, backfill=True)
    inits = [c[2] for c in env.calls]
    assert inits == sorted(inits)
    env.calls.clear()
    # a fresh directory: default order is newest first
    scorecard_dir = scores_db.scorecard_root()
    for p in scorecard_dir.rglob("*.parquet"):
        p.unlink()
    scorecard.refresh(("gfs",), VARS, TODAY)
    inits = [c[2] for c in env.calls]
    assert inits == sorted(inits, reverse=True)


def test_load_rows_union_and_range(env):
    scorecard.refresh(STORES, VARS, TODAY)
    start, end = TODAY - pd.Timedelta(days=4), TODAY - pd.Timedelta(days=3)
    rows = scores_db.load_rows(STORES, VARS, start, end)
    assert set(rows["store"]) == set(STORES)
    assert set(rows["init"]) == {start, end}
    assert len(rows) == 2 * 2 * 6
    empty = scores_db.load_rows(("gefs",), VARS, start, end)
    assert empty.empty and "store" in empty.columns


def test_writes_are_atomic(env, tmp_path):
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=None)
    assert not list((tmp_path / "scorecard").rglob("*.part"))
    assert not list((tmp_path / "scorecard").rglob("*.tmp"))


def test_complete_uses_the_last_scored_lead_not_the_cube_end(env):
    # GEFS runs to 840 h; only leads <= 360 h are scored, so an init whose
    # last scored lead has been observed is complete although 840 h is in the future.
    env.leads = (0.0, 6.0, 240.0, 840.0)
    scorecard.refresh(("gfs",), VARS, TODAY)
    init = TODAY - pd.Timedelta(days=14)  # 09-22: +240 h = 10-02, before obs_end 10-04
    meta = scores_db.scorecard_meta("gfs", "temperature_2m", init)
    assert meta["complete"] is True and meta["n_pairs"] == 2 * 3
    env.calls.clear()
    env.obs_end = env.obs_end + pd.Timedelta(days=1)
    scorecard.refresh(("gfs",), VARS, TODAY)
    assert init not in {c[2] for c in env.calls}  # not rescored for nothing


def _stations_ending(ends):
    return {sid: _obs_frame(end) for sid, end in ends.items()}


def test_obs_end_is_the_median_recent_station_minus_grace():
    t = pd.Timestamp("2026-10-06 00:00")
    ends = {"A": t, "B": t, "C": t, "D": t - pd.Timedelta(hours=30),
            "E": t - pd.Timedelta(days=20)}  # E stopped: not among the recent stations
    assert scorecard._obs_end(_stations_ending(ends)) == t - GRACE


def test_one_bad_timestamp_cannot_move_obs_end():
    t = pd.Timestamp("2026-10-05 23:00")
    obs = _stations_ending({s: t for s in "ABCDE"})
    for bad in (pd.Timedelta(hours=12), pd.Timedelta(days=30)):
        frame = obs["A"]
        wrong = frame.iloc[[-1]].assign(valid_time=frame["valid_time"].iloc[-1] + bad)
        skewed = {**obs, "A": pd.concat([frame, wrong], ignore_index=True)}
        got = scorecard._obs_end(skewed, cap=pd.Timestamp("2026-10-07"))
        assert got == t - GRACE, bad
    assert scorecard._obs_end({}) is None


def test_late_station_gets_its_last_leads_scored(env):
    env.net = _network(("A", "B", "C", "D"))
    env.leads = (0.0, 6.0, 12.0, 36.0)
    init = TODAY - pd.Timedelta(days=3)  # last lead 10-04 12 UTC, after obs_end 10-04 00
    env.late = {"D": pd.Timestamp("2026-10-03 08:00")}  # D's reports lag
    scorecard.refresh(("gfs",), VARS, TODAY)
    meta = scores_db.scorecard_meta("gfs", "temperature_2m", init)
    assert meta["complete"] is False and meta["n_pairs"] == 14  # D lacks 12 h and 36 h
    env.late = {}
    env.obs_end = pd.Timestamp("2026-10-07 00:00")
    scorecard.refresh(("gfs",), VARS, TODAY)
    meta = scores_db.scorecard_meta("gfs", "temperature_2m", init)
    assert meta["complete"] is True and meta["n_pairs"] == 16


def test_changed_station_list_triggers_reselection(env):
    env.net = _network(("A", "B"))
    scorecard.SCORECARD_STATIONS = ("A", "B")  # restored by monkeypatch teardown
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert env.selected == 1 and env.configured == ("A", "B")
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert env.selected == 1  # same list, current window: reused
    scorecard.SCORECARD_STATIONS = ("A", "B", "C")
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert env.selected == 2 and env.configured == ("A", "B", "C")


def test_window_anchor_follows_each_refresh_not_the_network_window(env):
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert scorecard.window_anchor(TODAY) == pd.Timestamp("2026-10-06")
    env.obs_end = pd.Timestamp("2026-10-03 12:00")  # moved back; network reused
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert env.selected == 1
    assert scorecard.window_anchor(TODAY) == pd.Timestamp("2026-10-03")


def test_prune_boundary_first_removed_before_kept_from(env, tmp_path):
    scorecard.refresh(("gfs",), VARS, TODAY)  # window 09-22..10-05
    first = scores_db.scorecard_path("gfs", "temperature_2m", pd.Timestamp("2026-09-22"))
    before = scores_db.scorecard_path("gfs", "temperature_2m", pd.Timestamp("2026-09-21"))
    before.write_bytes(b"x")
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert not before.exists() and first.exists()


def test_changed_list_reads_observations_for_the_configured_stations(env, monkeypatch):
    env.saved = (_network(("OLD1", "OLD2")), TODAY - pd.Timedelta(days=SCORECARD_DAYS), TODAY)
    env.configured = ("OLD1", "OLD2")
    scorecard.SCORECARD_STATIONS = ("A", "B")
    asked = []
    real = env.network_obs
    monkeypatch.setattr(
        station_network, "network_obs",
        lambda ids, s, e: (asked.append(list(ids)), real(ids, s, e))[1],
    )
    scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0)
    assert asked == [["A", "B"]]  # no wasted read of the old ids


def test_aggregates_rewritten_when_pruning_removed_files_and_nothing_was_written(env):
    """The window advances, its oldest init is pruned, no new init is scored:
    the published aggregates must drop the pruned init all the same."""
    scorecard.refresh(("gfs",), VARS, TODAY)  # window 09-22..10-05
    assert scorecard.load_aggregates()[1]["start"] == "2026-09-22"
    env.obs_end = pd.Timestamp("2026-10-07 00:00")  # window 09-23..10-06
    assert scorecard.refresh(("gfs",), VARS, TODAY, max_minutes=0) == 0
    assert not scores_db.scorecard_path(
        "gfs", "temperature_2m", pd.Timestamp("2026-09-22")
    ).exists()
    assert scorecard.load_aggregates()[1]["start"] == "2026-09-23"


# --- Task 4: row version and rescoring from the cached cube -------------------------

from utils import earthmover_api  # noqa: E402
from utils.earthmover_api import station_cube as _real_station_cube  # noqa: E402
from utils.settings import SCORECARD_ROW_VERSION  # noqa: E402


def test_row_version_is_written(env):
    scorecard.refresh(("gfs",), VARS, TODAY)
    meta = scores_db.scorecard_meta("gfs", "temperature_2m", TODAY - pd.Timedelta(days=3))
    assert meta["row_version"] == SCORECARD_ROW_VERSION == 3


def test_needs_scoring_on_row_version():
    meta = {"network_hash": "h", "complete": True, "row_version": SCORECARD_ROW_VERSION}
    assert not scorecard._needs_scoring(meta, "h", None)
    assert scorecard._needs_scoring({**meta, "row_version": 1}, "h", None)
    old = {k: v for k, v in meta.items() if k != "row_version"}
    assert scorecard._needs_scoring(old, "h", None)


def test_row_version_bump_rescores_every_init_from_the_disk_cache(
    env, monkeypatch, tmp_path
):
    # The real station_cube, served from its disk cache: any archive open counts.
    monkeypatch.setattr(earthmover_api, "POINT_CACHE_DIR", str(tmp_path / "cache"))
    opened = []

    def no_archive(store):
        opened.append(store)
        raise AssertionError("archive opened")

    monkeypatch.setattr(earthmover_api, "_open_dataset", no_archive)
    monkeypatch.setattr(earthmover_api, "station_cube", _real_station_cube)
    for init in _inits(SCORECARD_DAYS + 2):
        cube = _cube("gfs", "temperature_2m", init, env.net, None, env.leads)
        cube.attrs.update(dropped_outside_domain=0, res_deg=0.25, step_type="instant")
        path = earthmover_api._station_cache_path(
            "gfs", "temperature_2m", init, env.net, None, None
        )
        earthmover_api._save_station_cube(cube, path)

    monkeypatch.setattr(scorecard, "SCORECARD_ROW_VERSION", 1)
    assert scorecard.refresh(("gfs",), VARS, TODAY) == SCORECARD_DAYS
    assert scorecard.refresh(("gfs",), VARS, TODAY) == 0  # same version: skipped
    monkeypatch.setattr(scorecard, "SCORECARD_ROW_VERSION", 2)
    assert scorecard.refresh(("gfs",), VARS, TODAY) == SCORECARD_DAYS  # mismatch
    assert opened == []
    meta = scores_db.scorecard_meta("gfs", "temperature_2m", TODAY - pd.Timedelta(days=3))
    assert meta["row_version"] == 2
    rows = pd.read_parquet(scores_db.scorecard_path("gfs", "temperature_2m",
                                                    TODAY - pd.Timedelta(days=3)))
    assert list(rows.columns[: len(scorecard.PAIR_COLUMNS)]) == scorecard.PAIR_COLUMNS


def test_height_correction_uses_the_network_elevation(env, monkeypatch):
    monkeypatch.setattr(scorecard, "station_orography", lambda: {"A": 200.0})
    scorecard.refresh(("ifs_ens",), VARS, TODAY)
    rows = pd.read_parquet(scores_db.scorecard_path("ifs_ens", "temperature_2m",
                                                    TODAY - pd.Timedelta(days=3)))
    a, b = rows[rows["station"] == "A"], rows[rows["station"] == "B"]
    assert (a["dz"] == 100.0).all()  # z 200 m - elev 100 m
    np.testing.assert_allclose(a["err_hc"], a["err"] + 0.65)
    assert b["dz"].isna().all() and b["crps_hc"].isna().all()


def test_aggregates_read_old_files_without_the_new_columns(env, monkeypatch):
    scorecard.refresh(("gfs",), VARS, TODAY)
    path = scores_db.scorecard_path("gfs", "temperature_2m", TODAY - pd.Timedelta(days=3))
    rows = pd.read_parquet(path)
    rows[["station", "init", "lead_hours", "store", "variable", "err", "ae", "se",
          "crps", "spread2", "wet_fc", "wet_obs", "brier"]].to_parquet(path)
    scorecard.write_aggregates(TODAY)  # a v1 file must not break the aggregation
    assert "crps_pert" in scorecard._AGG_PAIR_COLUMNS
