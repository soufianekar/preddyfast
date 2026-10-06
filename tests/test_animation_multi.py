import numpy as np
import pandas as pd

from utils import map_products as mp
from utils.earthmover_api import GridDataError


def _cube(store):
    return {"cube": np.ones((2, 3, 4, 4), dtype=np.float32), "lead_hours": [24, 48],
            "valid_time": [pd.Timestamp("2026-10-02"), pd.Timestamp("2026-10-03")],
            "lat": np.arange(4.0), "lon": np.arange(4.0), "members": [0, 1, 2],
            "init_time": pd.Timestamp("2026-10-01"), "stride": 1, "bounds": (45.0, 50.0, 5.0, 10.0),
            "region": None, "unit": "m/s", "store": store}


def test_a_store_lacking_the_variable_skips_its_panel(monkeypatch):
    def cube(area_key, variable, init_time, step, accum, store, horizon):
        if store == "aifs_ens":
            raise GridDataError("ECMWF AIFS ENS does not archive wind gust (10 m).")
        return _cube(store)

    monkeypatch.setattr(mp, "check_read_budget", lambda *a, **k: None)
    monkeypatch.setattr(mp, "frame_cube", cube)
    monkeypatch.setattr(mp, "_render_frames", lambda cube, values, keep, *a: [
        {"src": "data:x", "lead_hours": cube["lead_hours"][i], "valid": "v"} for i in keep])
    out = mp.forecast_animation("box:45,50,5,10", "wind_gust_10m", None, 24, 72, None, "mean", 90.0, (10, 90),
                                "auto", "data", "default", ("ifs_ens", "aifs_ens"))
    assert [p["store"] for p in out["panels"]] == ["ifs_ens"]
    assert any("AIFS" in e for e in out["errors"])
    assert all(len(f["srcs"]) == 1 for f in out["frames"])


def test_deterministic_panel_shows_the_field_and_says_so(monkeypatch):
    monkeypatch.setattr(mp, "check_read_budget", lambda *a, **k: None)
    monkeypatch.setattr(mp, "frame_cube", lambda area_key, variable, init_time, step, accum, store, horizon: {
        **_cube(store), "cube": np.ones((2, 1 if store == "gfs" else 3, 4, 4), dtype=np.float32), "members": [0] if store == "gfs" else [0, 1, 2]})
    monkeypatch.setattr(mp, "_render_frames", lambda cube, values, keep, *a: [
        {"src": "data:x", "lead_hours": cube["lead_hours"][i], "valid": "v"} for i in keep])
    out = mp.forecast_animation("box:45,50,5,10", "temperature_2m", None, 24, 72, None, "mean", 90.0, (10, 90),
                                "auto", "data", "default", ("ifs_ens", "gfs"))
    labels = {p["store"]: p["label"] for p in out["panels"]}
    assert "deterministic" in labels["gfs"] and "deterministic" not in labels["ifs_ens"]
    # Spread: the deterministic panel is left out (its field is not a spread).
    spread = mp.forecast_animation("box:45,50,5,10", "temperature_2m", None, 24, 72, None, "range", 90.0, (10, 90),
                                   "auto", "data", "default", ("ifs_ens", "gfs"))
    assert [p["store"] for p in spread["panels"]] == ["ifs_ens"]
    assert any("spread" in e for e in spread["errors"])
