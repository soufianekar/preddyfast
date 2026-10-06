import pandas as pd

from utils import earthmover_api as em

CAL = {"first": "2026-10-01", "last": "2026-10-02", "hours": [0],
       "days": {"2026-10-01": [0], "2026-10-02": [0]}, "missing": [], "unfinished": ["2026-10-02T00:00"]}


def test_run_in_calendar():
    assert em.run_in_calendar(CAL, "2026-10-01T00:00")
    assert not em.run_in_calendar(CAL, "2026-10-01T06:00")      # hour the stores do not share
    assert not em.run_in_calendar(CAL, "2026-10-02T00:00")      # still being written
    assert not em.run_in_calendar(CAL, "2026-09-01T00:00")      # outside the calendar
    assert em.run_in_calendar(CAL, "latest")


def test_latest_with_several_models_is_one_common_init(monkeypatch):
    monkeypatch.setattr(em, "eval_inits", lambda stores, n, hour=None: [pd.Timestamp("2026-10-01 00:00")])
    assert em.resolve_run(("ifs_ens", "aifs_ens"), "latest") == "2026-10-01T00:00"


def test_latest_with_one_model_stays_latest():
    assert em.resolve_run(("aifs_ens",), "latest") is None


def test_past_run_is_kept():
    assert em.resolve_run(("ifs_ens", "aifs_ens"), "2026-09-30T00:00") == "2026-09-30T00:00"


def test_latest_candidates_skip_runs_still_being_written(monkeypatch):
    inits = [pd.Timestamp(f"2026-10-02 {h:02d}:00") for h in (0, 6, 12)]
    monkeypatch.setattr(em, "available_inits", lambda store: inits)
    monkeypatch.setattr(em, "run_calendar", lambda store: {"unfinished": ["2026-10-02T12:00"]})
    assert em._latest_candidates("gfs") == [inits[1], inits[0]]
