"""A smaller area is cut from a cached larger field, never re-retrieved."""

import json

import numpy as np
import pytest
import xarray as xr

from utils import cds_api as cds

REQ = {"product_type": ["reanalysis"], "variable": ["2m_temperature"], "date": ["2026-07-01/2026-07-01"],
       "time": ["12:00"], "data_format": "netcdf"}


@pytest.fixture
def cache_dir(monkeypatch, tmp_path):
    monkeypatch.setattr(cds, "CDS_CACHE_DIR", str(tmp_path))
    return tmp_path


def _cached(cache_dir, request, name):
    (cache_dir / f"{name}.nc").write_bytes(b"x")
    (cache_dir / f"{name}.json").write_text(json.dumps({"dataset": "reanalysis-era5-single-levels", "request": request}))


def test_a_global_field_on_disk_serves_any_box(cache_dir):
    _cached(cache_dir, REQ, "global1")
    want = {**REQ, "area": [55.0, -10.0, 40.0, 20.0]}
    assert cds.find_enclosing("reanalysis-era5-single-levels", want) == "global1"


def test_an_enclosing_box_is_reused_and_the_smallest_wins(cache_dir):
    _cached(cache_dir, {**REQ, "area": [70.0, -30.0, 30.0, 45.0]}, "europe")
    _cached(cache_dir, REQ, "global1")
    want = {**REQ, "area": [55.0, -10.0, 40.0, 20.0]}
    assert cds.find_enclosing("reanalysis-era5-single-levels", want) == "europe"


@pytest.mark.parametrize("cached", [
    {**REQ, "area": [50.0, -10.0, 40.0, 20.0]},          # does not reach 55°N
    {**REQ, "grid": [1.0, 1.0]},                          # regridded: would lose resolution
    {**REQ, "date": ["2026-07-02/2026-07-02"]},           # another day
])
def test_anything_else_is_retrieved(cache_dir, cached):
    _cached(cache_dir, cached, "other")
    want = {**REQ, "area": [55.0, -10.0, 40.0, 20.0]}
    assert cds.find_enclosing("reanalysis-era5-single-levels", want) is None


def test_a_request_without_an_area_is_never_served_by_a_box(cache_dir):
    _cached(cache_dir, {**REQ, "area": [70.0, -30.0, 30.0, 45.0]}, "europe")
    assert cds.find_enclosing("reanalysis-era5-single-levels", REQ) is None


def _field():
    lat = np.arange(60.0, 29.0, -1.0)
    lon = np.arange(-180.0, 180.0, 1.0)
    return xr.DataArray(np.zeros((lat.size, lon.size)), dims=("latitude", "longitude"),
                        coords={"latitude": lat, "longitude": lon})


def test_crop_keeps_the_box_including_across_the_dateline():
    da = cds._crop(_field(), "latitude", "longitude", (40.0, 50.0, -10.0, 20.0))
    assert float(da.latitude.min()) == 40.0 and float(da.latitude.max()) == 50.0
    assert float(da.longitude.min()) == -10.0 and float(da.longitude.max()) == 20.0
    wrap = cds._crop(_field(), "latitude", "longitude", (40.0, 50.0, 170.0, -170.0))
    assert set(np.round(wrap.longitude.values)) == {170.0, 171.0, 172.0, 173.0, 174.0, 175.0, 176.0, 177.0, 178.0, 179.0,
                                                    -180.0, -179.0, -178.0, -177.0, -176.0, -175.0, -174.0, -173.0,
                                                    -172.0, -171.0, -170.0}


def test_crop_outside_the_field_is_refused():
    with pytest.raises(cds.CDSDataError):
        cds._crop(_field(), "latitude", "longitude", (-50.0, -40.0, 0.0, 10.0))
