import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils import earthmover_api, scorecard
from utils.settings import SCORECARD_STATIONS

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "build_station_orography.py"
)
_spec = importlib.util.spec_from_file_location("build_station_orography", _SCRIPT)
bso = importlib.util.module_from_spec(_spec)
sys.modules["build_station_orography"] = bso
_spec.loader.exec_module(bso)


def test_area_padded_and_rounded_outward():
    lat, lon = np.array([38.01, 51.48]), np.array([-3.56, 28.81])
    n, w, s, e = bso.request_area(lat, lon)
    assert (n, s) == (52.0, 37.5)
    assert (w, e) == (-4.25, 29.5)
    assert n >= lat.max() + 0.5 and s <= lat.min() - 0.5
    assert w <= lon.min() - 0.5 and e >= lon.max() + 0.5


def test_request_is_the_brief_request():
    r = bso.build_request([1, 2, 3, 4])
    assert r["variable"] == ["geopotential"] and r["time"] == ["00:00"]
    assert (r["year"], r["month"], r["day"]) == (["2020"], ["01"], ["01"])
    assert r["data_format"] == "netcdf" and r["download_format"] == "unarchived"


def test_cells_agree_with_station_cells(monkeypatch):
    stations = pd.DataFrame(
        {"lat": [51.47, 49.01, -0.1, 40.47], "lon": [-0.46, 2.55, 179.9, -3.56]}
    )
    lat = np.arange(90, -90.25, -0.25)
    lon = np.arange(-180, 180, 0.25)
    ds = xr.Dataset(coords={"latitude": lat, "longitude": lon})
    monkeypatch.setattr(earthmover_api, "store_geometry", lambda s: {"res_deg": 0.25})
    kept, rows, cols = earthmover_api._station_cells(ds, "x", stations)
    la, lo = bso.snap_cells(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    assert kept.all()
    assert np.allclose(lat[rows], la) and np.allclose(lon[cols], lo)


def _ds(with_time=True):
    lat = np.array([52.0, 51.75, 51.5, 51.25])
    lon = np.array([-0.75, -0.5, -0.25])
    z = np.arange(12, dtype=float).reshape(4, 3) * bso.G
    dims = ("latitude", "longitude")
    if with_time:
        z = z[None]
        dims = ("valid_time",) + dims
    return xr.Dataset(
        {"z": (dims, z)},
        coords={"latitude": lat, "longitude": lon}
        | ({"valid_time": [np.datetime64("2020-01-01")]} if with_time else {}),
    )


@pytest.mark.parametrize("with_time", [True, False])
def test_extract_cells_divides_by_g_and_drops_time(with_time):
    st = pd.DataFrame(
        {
            "id": ["A", "B"],
            "lat": [51.47, 51.9],
            "lon": [-0.46, -0.7],
            "elev_m": [25.3, np.nan],
        }
    )
    out = bso.extract_cells(_ds(with_time), st, "2026-10-14")
    assert out["retrieved"] == "2026-10-14" and "ERA5" in out["source"]
    assert out["cells"]["A"] == {"lat": 51.5, "lon": -0.5, "z_m": 7.0, "elev_m": 25.3}
    assert out["cells"]["B"]["z_m"] == 0.0 and out["cells"]["B"]["elev_m"] is None


def test_extract_cells_missing_cell_raises():
    st = pd.DataFrame({"id": ["A"], "lat": [10.0], "lon": [10.0], "elev_m": [1.0]})
    with pytest.raises(ValueError):
        bso.extract_cells(_ds(), st, "d")


def _point(monkeypatch, path):
    scorecard._read_orography.cache_clear()
    monkeypatch.setattr(scorecard, "OROGRAPHY_PATH", path)


def test_loader_and_dz(tmp_path, monkeypatch):
    p = tmp_path / "o.json"
    p.write_text(json.dumps({"cells": {"A": {"z_m": 110.0}, "B": {"z_m": 5.0}}}))
    _point(monkeypatch, p)
    assert scorecard.station_orography() == {"A": 110.0, "B": 5.0}
    dz = scorecard.station_dz(["A", "B", "C", "A"], {"A": 100.0, "C": 1.0})
    assert dz[0] == 10.0 and dz[3] == 10.0
    assert np.isnan(dz[1]) and np.isnan(dz[2])  # no elev / no cell
    scorecard._read_orography.cache_clear()


def test_missing_file_is_empty_and_logged_once(tmp_path, monkeypatch, caplog):
    _point(monkeypatch, tmp_path / "none.json")
    with caplog.at_level("WARNING"):
        assert scorecard.station_orography() == {}
        assert scorecard.station_orography() == {}
    assert len([r for r in caplog.records if "orography" in r.message]) == 1
    assert np.isnan(scorecard.station_dz(["A"], {"A": 1.0})).all()
    scorecard._read_orography.cache_clear()


def test_scorecard_stations_unchanged():
    assert len(SCORECARD_STATIONS) == 8
