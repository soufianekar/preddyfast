import numpy as np
import pandas as pd
import xarray as xr

from utils import earthmover_api as em
from utils.climatology import compute_member_pct
from utils.figures import make_regional_multi_fig
from utils.settings import GRID_STORES, GRID_VARIABLES


def test_member_axis_added_when_absent():
    ds = xr.Dataset({"t": (("init_time", "lead_time"), np.zeros((1, 3)))})
    out = em._as_ensemble(ds)
    assert out.sizes["ensemble_member"] == 1 and out["ensemble_member"].values.tolist() == [0]


def test_member_axis_kept_when_present():
    ds = xr.Dataset({"t": (("ensemble_member", "lead_time"), np.zeros((5, 3)))})
    assert em._as_ensemble(ds).sizes["ensemble_member"] == 5


def test_chunk_dims_skip_the_added_member_axis():
    dims = ("ensemble_member", "init_time", "lead_time", "latitude", "longitude")
    assert em._chunk_dims(dims, (1, 105, 121, 121), deterministic=True) == {
        "init_time": 1, "lead_time": 105, "latitude": 121, "longitude": 121}


def _series(n_members, offset, store):
    times = pd.date_range("2026-10-01 06:00", periods=4, freq="6h")
    df = pd.DataFrame([{"valid_time": t, "member": m, "value": offset + m + i}
                       for i, t in enumerate(times) for m in range(n_members)])
    meta = {"init_time": pd.Timestamp("2026-10-01"), "n_members": n_members, "n_lat": 21, "n_lon": 21,
            "bbox": (45.0, 50.0, 5.0, 10.0), "resolution_deg": 0.25, "unit": "°C", "accum_hours": None,
            "step_window": None, "store": store}
    return df, compute_member_pct(df), meta


def test_deterministic_draws_one_line_no_bands():
    fig = make_regional_multi_fig(
        {"ifs_ens": _series(3, 0.0, "ifs_ens"), "gfs": _series(1, 100.0, "gfs")},
        GRID_VARIABLES["temperature_2m"],
    )
    gfs = [t for t in fig.data if GRID_STORES["gfs"]["label"] in t.name]
    assert len(gfs) == 1 and "deterministic" in gfs[0].name
    assert not any(t.legendgroup.startswith("gfs-band") or t.legendgroup.startswith("gfs-members") for t in fig.data)
    assert any(t.legendgroup == "ifs_ens-band" for t in fig.data)


def test_open_leaves_the_dataset_alone_and_selection_adds_the_member_axis(monkeypatch):
    ds = xr.Dataset(
        {"temperature_2m": (("init_time", "lead_time", "latitude", "longitude"), np.zeros((1, 2, 3, 3)))},
        coords={"init_time": [pd.Timestamp("2026-10-01")], "lead_time": pd.to_timedelta([0, 6], "h"),
                "latitude": [50.0, 49.75, 49.5], "longitude": [5.0, 5.25, 5.5]},
    )
    da = em._select_box(ds, "temperature_2m", pd.Timestamp("2026-10-01"), (49.5, 50.0, 5.0, 5.5))
    assert da.sizes["ensemble_member"] == 1 and "ensemble_member" not in ds.dims


def test_subtitle_states_each_models_own_extent_and_grid():
    a = _series(3, 0.0, "ifs_ens")
    b = _series(3, 0.0, "icon_eu")
    b[2].update({"resolution_deg": 0.0625, "bbox": (47.0, 50.0, 6.0, 9.0), "n_lat": 49, "n_lon": 49})
    from utils.figures import figure_text
    title = figure_text(make_regional_multi_fig({"ifs_ens": a, "icon_eu": b}, GRID_VARIABLES["temperature_2m"]))
    assert "21×21 cells" in title and "49×49 cells" in title and "0.0625°" in title


def test_meteogram_deterministic_is_one_line_not_an_ensemble_mean():
    from utils.figures import make_meteogram_fig
    times = pd.date_range("2026-10-01 06:00", periods=4, freq="6h")
    det = pd.DataFrame({"date": times, "member": 0, "value": [1.0, 2.0, 3.0, 4.0]})
    fig = make_meteogram_fig({"gfs": det}, {"gfs": "NOAA GFS (deterministic)"}, {"label": "T", "unit": "°C"}, "Paris")
    names = [t.name for t in fig.data]
    assert not any("ensemble mean" in n or "control" in n for n in names), names
    assert any("deterministic" in n for n in names)
