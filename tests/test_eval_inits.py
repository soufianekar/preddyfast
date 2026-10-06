import pandas as pd

from utils import earthmover_api as em

CAL = {
    "first": "2026-09-29",
    "last": "2026-10-02",
    "hours": [0, 12],
    "days": {"2026-09-29": [0, 12], "2026-09-30": [0], "2026-10-01": [0, 12], "2026-10-02": [0]},
    "missing": [],
    "unfinished": ["2026-10-02T00:00"],
}


def test_newest_finished_runs_first(monkeypatch):
    monkeypatch.setattr(em, "common_run_calendar", lambda stores: CAL)
    got = em.eval_inits(("ifs_ens", "aifs_ens"), 3)
    assert got == [pd.Timestamp("2026-10-01 12:00"), pd.Timestamp("2026-10-01 00:00"), pd.Timestamp("2026-09-30 00:00")]


def test_hour_filter(monkeypatch):
    monkeypatch.setattr(em, "run_calendar", lambda store: CAL)
    got = em.eval_inits(("ifs_ens",), 5, hour=12)
    assert got == [pd.Timestamp("2026-10-01 12:00"), pd.Timestamp("2026-09-29 12:00")]


def test_point_archive_tags_each_run(monkeypatch):
    def fake(lat, lon, variable, init_time, accum_hours, store):
        t = pd.Timestamp(init_time)
        return pd.DataFrame({"valid_time": [t + pd.Timedelta("24h")], "lead_hours": [24.0], "member": [0], "value": [1.0]}), {}

    monkeypatch.setattr(em, "_point_run", fake)
    inits = [pd.Timestamp("2026-10-01"), pd.Timestamp("2026-09-30")]
    df = em.point_archive("ifs_ens", "temperature_2m", 48.85, 2.35, inits, None)
    assert sorted(df["init_time"].unique()) == sorted(inits)
    assert {"init_time", "valid_time", "lead_hours", "member", "value"} <= set(df.columns)


def test_runs_after_the_last_observation_are_left_out(monkeypatch):
    monkeypatch.setattr(em, "run_calendar", lambda store: CAL)
    got = em.eval_inits(("ifs_ens",), 5, hour=0, before=pd.Timestamp("2026-09-30 14:00"))
    assert got == [pd.Timestamp("2026-09-30 00:00"), pd.Timestamp("2026-09-29 00:00")]


def test_tz_aware_cutoff_compares_as_utc(monkeypatch):
    monkeypatch.setattr(em, "run_calendar", lambda store: CAL)
    got = em.eval_inits(("ifs_ens",), 5, before=pd.Timestamp("2026-09-30 00:00", tz="UTC"))
    assert got[0] == pd.Timestamp("2026-09-30 00:00")
