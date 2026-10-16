"""Z500 verified against IFS control T+0 (R5: 00 UTC valid times only)."""

import importlib
import logging

import dash
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils import scorecard, scores_db, station_network
from utils.settings import SCORECARD_DAYS

TODAY = pd.Timestamp("2026-10-06")
INIT = pd.Timestamp("2026-09-25")
Z500 = "geopotential_height_500hpa"


def _cube(store, init, stations, leads, value, members=1):
    """Station cube whose value at lead L is value(init, L) for every member."""
    leads = np.asarray(leads, dtype=float)
    data = np.array([[value(init, lead) for lead in leads]] * members, dtype="float32")
    return xr.DataArray(
        np.broadcast_to(data, (len(stations), members, len(leads))).copy(),
        dims=("station", "member", "lead_hours"),
        coords={
            "station": list(stations),
            "member": list(range(members)),
            "lead_hours": leads,
            "valid_time": ("lead_hours", init + pd.to_timedelta(leads, unit="h")),
        },
        attrs={"init": init.isoformat(), "store": store, "variable": Z500, "window_hours": None},
    )


def _ref_value(init, lead):
    # T+0 of each init is unique (day of month * 10); any other lead is far off,
    # so a reference taken from another init's lead is visible.
    return init.day * 10.0 if lead == 0 else -9999.0


# --- pure matching -----------------------------------------------------------------


def test_reference_at_valid_time_is_init_v_lead_zero_never_another_lead():
    fc = _cube("ifs_ens", INIT, ["A", "B"], [0, 12, 24, 36, 48, 72], lambda i, h: 5500.0, 3)
    refs = {
        INIT: _cube("ifs_control", INIT, ["A", "B"], [0, 24, 48], _ref_value),
        INIT + pd.Timedelta(hours=24): _cube(
            "ifs_control", INIT + pd.Timedelta(hours=24), ["A", "B"], [0, 24], _ref_value
        ),
        # 72 h: init INIT+72 h is not held -> no pair, although INIT+24 h's
        # +48 h lead would cover that valid time.
        INIT + pd.Timedelta(hours=48): _cube(
            "ifs_control", INIT + pd.Timedelta(hours=48), ["A", "B"], [0, 24], _ref_value
        ),
    }
    matched = scorecard.match_reference(fc, refs, "z500")
    obs = matched["obs"].sel(station="A").to_series()
    assert obs[24.0] == (INIT + pd.Timedelta(hours=24)).day * 10.0
    assert obs[48.0] == (INIT + pd.Timedelta(hours=48)).day * 10.0
    assert np.isnan(obs[72.0])  # no init at V: never INIT+24h's or INIT+48h's lead
    assert (matched["obs"] != -9999.0).all()
    rows = scorecard.pair_scores(matched, "z500")
    assert 72.0 not in set(rows["lead_hours"])


def test_twelve_utc_valid_times_are_dropped():
    fc = _cube("ifs_ens", INIT, ["A"], [0, 6, 12, 18, 24, 36], lambda i, h: 5500.0, 2)
    refs = {INIT + pd.Timedelta(hours=h): _cube(
        "ifs_control", INIT + pd.Timedelta(hours=h), ["A"], [0], _ref_value) for h in (0, 24)}
    matched = scorecard.match_reference(fc, refs, "z500")
    assert list(matched["lead_hours"].values) == [0.0, 24.0]
    valid = pd.to_datetime(matched["valid_time"].values)
    assert (valid.hour == 0).all()


def test_reference_station_missing_gives_no_pair():
    fc = _cube("gefs", INIT, ["A", "B"], [0, 24], lambda i, h: 5500.0, 2)
    refs = {INIT + pd.Timedelta(hours=24): _cube(
        "ifs_control", INIT + pd.Timedelta(hours=24), ["A"], [0], _ref_value)}
    matched = scorecard.match_reference(fc, refs, "z500")
    assert np.isnan(float(matched["obs"].sel(station="B", lead_hours=24.0)))


def test_match_station_obs_still_refuses_z500():
    fc = _cube("ifs_ens", INIT, ["A"], [0, 24], lambda i, h: 5500.0)
    with pytest.raises(ValueError):
        scorecard.match_station_obs(fc, {}, "z500")


# --- refresh -----------------------------------------------------------------------


def _network(ids=("A", "B")):
    return pd.DataFrame({
        "id": list(ids), "name": list(ids), "lat": [50.0, 51.0], "lon": [10.0, 10.0],
        "elev_m": 100.0, "ok_temperature": True, "ok_precip": False, "ok_wind": True,
    })


class Env:
    def __init__(self, monkeypatch, tmp_path):
        self.calls = []
        self.fail = set()
        self.net = _network()
        self.saved = None
        self.obs_end = pd.Timestamp("2026-10-06 00:00")
        inits = [TODAY - pd.Timedelta(days=d) for d in range(1, 60)]
        self.inits = {"ifs_ens": inits, "ifs_control": inits, "gfs": inits}
        monkeypatch.setattr(scorecard, "SCORECARD_STATIONS", ())
        monkeypatch.setattr(scores_db, "SCORECARD_DIR", str(tmp_path / "scorecard"))
        monkeypatch.setattr(scorecard, "PRECOMPUTE_DIR", str(tmp_path / "precomputed"))
        monkeypatch.setattr(scorecard.earthmover_api, "station_cube", self.station_cube)
        monkeypatch.setattr(scorecard.earthmover_api, "available_inits",
                            lambda s: sorted(self.inits[s]))
        monkeypatch.setattr(
            scorecard.earthmover_api, "store_variables",
            lambda s: ["temperature_2m"] if s == "gfs" else ["temperature_2m", Z500],
        )
        monkeypatch.setattr(station_network, "load_network", lambda: self.saved)
        monkeypatch.setattr(station_network, "select_network", lambda s, e: self.net)
        monkeypatch.setattr(station_network, "save_network", self.save_network)
        monkeypatch.setattr(station_network, "load_configured", lambda: ())
        monkeypatch.setattr(
            station_network, "load_rule_hash", lambda: station_network.rule_fingerprint()
        )
        monkeypatch.setattr(station_network, "network_obs", self.network_obs)

    def save_network(self, net, start, end, configured=()):
        self.saved = (net, pd.Timestamp(start), pd.Timestamp(end))

    def network_obs(self, ids, start, end):
        times = pd.date_range(end=self.obs_end, periods=24 * 60, freq="h", tz="UTC")
        return {i: pd.DataFrame({"valid_time": times, "temperature_2m": 9.0}) for i in ids}

    def station_cube(self, store, variable, init, stations, accum, budget_mb=None, window_end_hour=None):
        self.calls.append((store, variable, init))
        if (store, init) in self.fail:
            raise scorecard.GridDataError("incomplete run")
        ids = [str(i) for i in stations["id"]]
        if store == "ifs_control":
            return _cube(store, init, ids, [0, 24, 48], _ref_value)
        return _cube(store, init, ids, [0, 12, 24, 36, 48], lambda i, h: 5500.0, 2)


@pytest.fixture
def env(monkeypatch, tmp_path):
    return Env(monkeypatch, tmp_path)


def test_refresh_scores_z500_from_ifs_control_t0_and_leaves_gfs_out(env, caplog):
    with caplog.at_level(logging.INFO):
        n = scorecard.refresh(("ifs_ens", "gfs"), ("z500",), TODAY)
    assert n == SCORECARD_DAYS  # ifs_ens only
    assert "does not hold it" in caplog.text
    assert not any(c[0] == "gfs" for c in env.calls)
    # window 09-22..10-05; references read for the valid times, 00 UTC inits only
    refs = {c[2] for c in env.calls if c[0] == "ifs_control"}
    assert all(t.hour == 0 for t in refs)
    assert pd.Timestamp("2026-10-07") not in refs  # beyond the archive: not asked
    assert {c[1] for c in env.calls} == {Z500}
    init = pd.Timestamp("2026-09-30")
    rows = pd.read_parquet(scores_db.scorecard_path("ifs_ens", "z500", init))
    assert list(rows.columns[: len(scorecard.PAIR_COLUMNS)]) == scorecard.PAIR_COLUMNS
    assert set(rows["store"]) == {"ifs_ens"}
    assert sorted(set(rows["lead_hours"])) == [0.0, 24.0, 48.0]  # 12/36 h dropped
    at24 = rows[rows["lead_hours"] == 24.0]
    assert (at24["obs"] == 10.0).all()  # 10-01 T+0: day 1 * 10
    assert scorecard.load_lacking() == {"z500": ["gfs"]}


def test_reference_reads_are_shared_and_cached_within_a_run(env):
    scorecard.refresh(("ifs_ens",), ("z500",), TODAY)
    refs = [c[2] for c in env.calls if c[0] == "ifs_control"]
    assert len(refs) == len(set(refs))


def test_z500_incomplete_until_its_references_exist(env):
    # Newest init 10-05: +48 h valid 10-07 has no reference init yet.
    scorecard.refresh(("ifs_ens",), ("z500",), TODAY)
    newest = pd.Timestamp("2026-10-05")
    meta = scores_db.scorecard_meta("ifs_ens", "z500", newest)
    assert meta["complete"] is False
    old = scores_db.scorecard_meta("ifs_ens", "z500", pd.Timestamp("2026-09-22"))
    assert old["complete"] is True
    env.calls.clear()
    assert scorecard.refresh(("ifs_ens",), ("z500",), TODAY) == 0  # nothing new
    env.inits = {k: [*v, pd.Timestamp("2026-10-06"), pd.Timestamp("2026-10-07")]
                 for k, v in env.inits.items()}
    assert scorecard.refresh(("ifs_ens",), ("z500",), TODAY) >= 1
    assert scores_db.scorecard_meta("ifs_ens", "z500", newest)["complete"] is True


def test_failed_reference_read_scores_the_rest_and_retries(env):
    bad = pd.Timestamp("2026-10-01")
    env.fail.add(("ifs_control", bad))
    scorecard.refresh(("ifs_ens",), ("z500",), TODAY)
    init = pd.Timestamp("2026-09-30")  # needs 09-30, 10-01 (failed), 10-02
    rows = pd.read_parquet(scores_db.scorecard_path("ifs_ens", "z500", init))
    assert sorted(set(rows["lead_hours"])) == [0.0, 48.0]
    assert scores_db.scorecard_meta("ifs_ens", "z500", init)["complete"] is False
    assert scores_db.scorecard_meta("ifs_ens", "z500", pd.Timestamp("2026-09-22"))["complete"]
    env.fail.clear()
    scorecard.refresh(("ifs_ens",), ("z500",), TODAY)
    rows = pd.read_parquet(scores_db.scorecard_path("ifs_ens", "z500", init))
    assert sorted(set(rows["lead_hours"])) == [0.0, 24.0, 48.0]
    assert scores_db.scorecard_meta("ifs_ens", "z500", init)["complete"] is True


def test_aggregates_include_z500_and_record_the_lacking_model(env):
    scorecard.refresh(("ifs_ens", "gfs"), ("z500",), TODAY)
    agg, meta = scorecard.load_aggregates()
    assert "z500" in set(agg["variable"])
    assert set(agg.loc[agg["variable"] == "z500", "store"]) == {"ifs_ens"}
    assert meta["lacking"] == {"z500": ["gfs"]}


# --- page --------------------------------------------------------------------------


def test_page_notes_gfs_absent_from_the_z500_card(env):
    scorecard.refresh(("ifs_ens", "gfs"), ("z500",), TODAY)
    dash.Dash(__name__, use_pages=True, pages_folder="", suppress_callback_exceptions=True)
    page = importlib.import_module("pages.evaluation")
    heat, lines, _, alert, _ = page.update_scorecard(
        variable="z500", score="rmse", period=str(SCORECARD_DAYS), display="absolute",
        models=["ifs_ens", "gfs"], lead="1", map_model="ifs_ens", obs_mode="raw",
    )
    assert alert.color == "yellow" and "GFS" in str(alert.children)
    from utils.figures import figure_text

    text = figure_text(heat)
    assert "pseudo-analysis: IFS control T+0, favours IFS and AIFS" in text
    assert "GFS" in text  # the note travels with an exported figure
