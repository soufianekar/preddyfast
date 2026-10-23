"""Global AIFS Single export: stride, windows, run selection, manifest."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils import earthmover_api as em
from utils import globe
from utils.earthmover_api import GridDataError

NEW = pd.Timestamp("2026-10-23 06:00")
OLD = pd.Timestamp("2026-10-23 00:00")
LEADS = np.arange(0, 241, 6)


def _fake(newest_complete: bool = True) -> xr.Dataset:
    lat = np.linspace(90, -90, 721)
    lon = np.arange(-180, 180, 0.25)
    shape = (2, len(LEADS), lat.size, lon.size)
    t2m = np.full(shape, 15.0, np.float32)
    rate = np.full(shape, 1.0 / 3600.0, np.float32)  # 1 mm/h
    msl = np.full(shape, 101000.0, np.float32)
    msl += (np.linspace(-2000, 2000, lat.size)[:, None]).astype(np.float32)
    if not newest_complete:
        for arr in (t2m, rate, msl):
            arr[1, -10:] = np.nan
    dims = ("init_time", "lead_time", "latitude", "longitude")
    return xr.Dataset(
        {
            "temperature_2m": (dims, t2m, {"step_type": "instant"}),
            "precipitation_surface": (dims, rate, {"step_type": "avg"}),
            "pressure_reduced_to_mean_sea_level": (dims, msl, {"step_type": "instant"}),
        },
        coords={
            "init_time": [OLD, NEW],
            "lead_time": pd.to_timedelta(LEADS, unit="h"),
            "latitude": lat,
            "longitude": lon,
        },
    )


@pytest.fixture
def store(monkeypatch, tmp_path):
    ds = _fake()
    monkeypatch.setattr(globe, "_open_global", lambda s: ds)
    monkeypatch.setattr(globe, "check_read_budget", lambda *a, **k: None)
    monkeypatch.setattr(em, "_latest_candidates", lambda s: [NEW, OLD])
    monkeypatch.setattr(globe, "PRECOMPUTE_DIR", str(tmp_path))
    return ds


def test_read_global_strides_to_half_a_degree_with_both_poles(store):
    out = globe.read_global("t2m", NEW)
    assert out["values"].shape == (41, 361, 720)
    assert out["lat"][0] == 90.0 and out["lat"][-1] == -90.0
    assert out["lon"][0] == -180.0 and out["lon"][-1] == 179.5
    assert out["lead_hours"][0] == 0 and out["lead_hours"][-1] == 240
    assert out["valid_time"][1] == NEW + pd.Timedelta(hours=6)


def test_precipitation_frames_are_six_hour_totals_from_t6(store):
    out = globe.read_global("tp6", NEW)
    assert out["lead_hours"][0] == 6 and out["window_hours"] == 6
    assert np.allclose(out["values"], 6.0)
    assert out["unit"] == "mm"


def test_msl_is_in_hpa(store):
    out = globe.read_global("msl", NEW)
    assert 990 < float(np.nanmean(out["values"])) < 1030 and out["unit"] == "hPa"


def test_a_run_still_being_written_is_skipped_and_the_manifest_kept(
    monkeypatch, tmp_path
):
    ds = _fake(newest_complete=False)
    monkeypatch.setattr(globe, "_open_global", lambda s: ds)
    monkeypatch.setattr(globe, "check_read_budget", lambda *a, **k: None)
    monkeypatch.setattr(em, "_latest_candidates", lambda s: [NEW, OLD])
    monkeypatch.setattr(globe, "PRECOMPUTE_DIR", str(tmp_path))
    old = {"store": "aifs_single", "init_time": "2026-10-23T00:00:00", "layers": {}}
    globe.save_manifest(old)
    failures: list[str] = []
    assert globe.refresh(old, failures=failures) == 0
    assert globe.load_manifest() == old
    assert not failures


def test_refresh_writes_every_frame_and_a_matching_manifest(store, tmp_path):
    n = globe.refresh({})
    m = globe.load_manifest()
    assert m["init_time"] == "2026-10-23T06:00:00"
    assert set(m["layers"]) == {"t2m", "tp6", "msl"}
    files = [f for L in m["layers"].values() for f in L["frames"]]
    assert n == len(files) == 41 + 40 + 41
    assert all((tmp_path / f["path"]).exists() for f in files)
    assert m["layers"]["t2m"]["frames"][0]["path"] == (
        "globe/aifs_single/2026102306/t2m/T000.png"
    )
    assert m["layers"]["msl"]["frames"][0]["path"].endswith("msl/T000.geojson")
    assert m["layers"]["tp6"]["window_hours"] == 6
    assert m["layers"]["msl"]["interval"] == 5.0
    assert m["grid"] == {
        "res_deg": 0.5,
        "native_res_deg": 0.25,
        "lon0": -180.0,
        "lat0": 90.0,
        "nx": 720,
        "ny": 361,
    }


def test_a_current_manifest_reads_nothing(store, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("read")

    monkeypatch.setattr(globe, "read_global", boom)
    current = {"init_time": "2026-10-23T06:00:00"}
    assert globe.refresh(current) == 0


def test_the_budget_is_checked_before_reading(store, monkeypatch, tmp_path):
    def refuse(*a, **k):
        raise GridDataError("over budget")

    monkeypatch.setattr(globe, "check_read_budget", refuse)
    failures: list[str] = []
    assert globe.refresh({}, failures=failures) == 0
    assert failures and "over budget" in failures[0]
    assert not (tmp_path / "globe" / "manifest.json").exists()
    assert not (tmp_path / "globe" / "aifs_single").exists()


def test_the_budget_refuses_before_the_run_probe_reads(store, monkeypatch):
    def refuse(*a, **k):
        raise GridDataError("over budget")

    def never(*a, **k):
        raise AssertionError("opened or probed")

    monkeypatch.setattr(globe, "check_read_budget", refuse)
    monkeypatch.setattr(globe, "_open_global", never)
    monkeypatch.setattr(globe, "_complete_run", never)
    failures: list[str] = []
    assert globe.refresh({}, failures=failures) == 0
    assert failures and "over budget" in failures[0]


@pytest.mark.parametrize("exc", [OSError("net down"), RuntimeError("icechunk")])
def test_a_network_error_in_the_run_probe_is_recorded(store, monkeypatch, exc):
    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(globe, "_complete_run", boom)
    failures: list[str] = []
    assert globe.refresh({}, failures=failures) == 0
    assert failures and str(exc) in failures[0]
