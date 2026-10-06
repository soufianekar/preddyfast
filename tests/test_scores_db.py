"""Persisted per-run scores: one Parquet file per (reference, model, variable, site)."""

import pandas as pd
import pytest

from utils import scores_db as db

INIT = pd.Timestamp("2026-09-20")


def _rows(inits, leads=(24.0, 48.0), err=1.0):
    return pd.DataFrame([{"init_time": t, "lead_hours": h, "valid_time": t + pd.Timedelta(hours=h), "n_members": 51,
                          "obs": 2.0, "err": err, "abs_err": abs(err), "sq_err": err * err, "var": 0.5, "crps": 0.4}
                         for t in inits for h in leads])


@pytest.fixture(autouse=True)
def _dir(monkeypatch, tmp_path):
    monkeypatch.setattr(db, "SCORES_DIR", str(tmp_path))


def test_round_trip():
    db.save_per_run("stations", "ifs_ens", "temperature_2m", "FRM00007156", _rows([INIT]))
    got = db.load_per_run("stations", "ifs_ens", "temperature_2m", "FRM00007156")
    assert len(got) == 2 and set(got["lead_hours"]) == {24.0, 48.0}


def test_rescoring_a_run_replaces_it_never_duplicates():
    db.save_per_run("stations", "ifs_ens", "temperature_2m", "S", _rows([INIT], err=1.0))
    db.save_per_run("stations", "ifs_ens", "temperature_2m", "S", _rows([INIT, INIT + pd.Timedelta("1D")], err=3.0))
    got = db.load_per_run("stations", "ifs_ens", "temperature_2m", "S")
    assert len(got) == 4 and got["err"].eq(3.0).all()


def test_period_filter_on_init_time():
    inits = [INIT + pd.Timedelta(days=d) for d in range(5)]
    db.save_per_run("stations", "ifs_ens", "temperature_2m", "S", _rows(inits))
    got = db.load_per_run("stations", "ifs_ens", "temperature_2m", "S", since=INIT + pd.Timedelta(days=3))
    assert sorted(got["init_time"].unique()) == inits[3:]


def test_missing_file_is_empty_not_an_error():
    assert db.load_per_run("own_t0", "gefs", "temperature_2m", "nowhere").empty


def test_sites_and_references_stay_apart():
    db.save_per_run("stations", "ifs_ens", "temperature_2m", "A", _rows([INIT]))
    db.save_per_run("control_t0", "ifs_ens", "temperature_2m", "A", _rows([INIT], err=9.0))
    assert db.load_per_run("stations", "ifs_ens", "temperature_2m", "A")["err"].eq(1.0).all()
    assert db.load_per_run("stations", "ifs_ens", "temperature_2m", "B").empty


def test_site_id_cannot_escape_the_directory(tmp_path):
    db.save_per_run("stations", "ifs_ens", "temperature_2m", "../../etc", _rows([INIT]))
    assert all(tmp_path in p.parents for p in tmp_path.rglob("*.parquet"))
