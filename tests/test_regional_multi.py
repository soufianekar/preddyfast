import pandas as pd

from utils import earthmover_api as em


def _fake_series(bounds, variable, init_time, accum_hours, region=None, store="ifs_ens"):
    if store == "aifs_ens":
        raise em.GridDataError("ECMWF AIFS ENS does not archive wind gust (10 m).")
    return pd.DataFrame({"value": [1.0]}), {"store": store}


def test_one_failing_store_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(em, "check_read_budget", lambda *a, **k: None)
    monkeypatch.setattr(em, "_ensemble_series", _fake_series)
    results, errors = em.fetch_area_multi(
        "box:45,50,5,10", "wind_gust_10m", None, None, ("ifs_ens", "aifs_ens")
    )
    assert list(results) == ["ifs_ens"]
    assert len(errors) == 1 and "AIFS" in errors[0]


def test_results_keep_store_order(monkeypatch):
    monkeypatch.setattr(em, "check_read_budget", lambda *a, **k: None)
    monkeypatch.setattr(em, "_ensemble_series", lambda *a, store="ifs_ens", **k: (pd.DataFrame(), {"store": store}))
    results, errors = em.fetch_area_multi("box:45,50,5,10", "temperature_2m", None, None, ("aifs_ens", "ifs_ens"))
    assert list(results) == ["aifs_ens", "ifs_ens"] and errors == []
