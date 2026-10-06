import numpy as np
import pytest

from utils import earthmover_api as em
from utils.settings import GRID_STORES


def test_icon_eu_like_axis_is_regional_and_fine():
    lat = np.arange(70.5, 29.5 - 1e-9, -0.0625)
    lon = np.arange(-23.5, 62.5 + 1e-9, 0.0625)
    g = em._geometry_from_coords(lat, lon, np.arange(0, 121, 1.0), {"latitude": 219, "longitude": 153}, 1)
    assert g["res_deg"] == pytest.approx(0.0625)
    assert g["global"] is False
    assert g["lat_range"] == pytest.approx((29.5, 70.5)) and g["lon_range"] == pytest.approx((-23.5, 62.5))
    assert g["lat_descending"] is True


def test_coarse_global_axis():
    lat = np.arange(90, -90 - 1e-9, -1.5)
    lon = np.arange(-180, 180, 1.5)
    g = em._geometry_from_coords(lat, lon, np.arange(0, 1105, 6.0), {}, 101)
    assert g["res_deg"] == pytest.approx(1.5) and g["global"] is True and g["n_members"] == 101


def test_snap_to_a_coarse_grid():
    assert em.snap_bbox(45.01, 45.2, 5.03, 5.1, res=1.5) == (45.0, 46.5, 4.5, 6.0)


@pytest.mark.parametrize(
    "box, kwargs, expected",
    [  # outputs of snap_bbox on main before this change
        ((45.01, 45.2, 5.03, 5.1), {}, (45.0, 45.5, 5.0, 5.5)),
        ((44.9, 50.1, 4.9, 10.1), {}, (44.75, 50.25, 4.75, 10.25)),
        ((-90, 90, -180, 180), {}, (-5.0, 5.0, -5.0, 5.0)),
        ((-90, 90, -180, 180), {"max_span": 360}, (-90.0, 90.0, -180.0, 179.75)),
        ((10.0, 10.0, 20.0, 20.0), {}, (10.0, 10.25, 20.0, 20.25)),
    ],
)
def test_snap_default_unchanged(box, kwargs, expected):
    assert em.snap_bbox(*box, **kwargs) == expected


def test_every_store_has_the_new_keys():
    for key, spec in GRID_STORES.items():
        assert spec["kind"] in ("ensemble", "deterministic"), key
        assert spec["family"] and spec["fallback_href"].startswith("s3://"), key
    assert {"gefs", "gfs", "aifs_single", "icon_eu"} <= set(GRID_STORES)


def test_probe_variable_falls_back_for_stores_without_temperature():
    import xarray as xr
    ds = xr.Dataset({"precipitation_surface": ("x", np.zeros(2)), "maximum_temperature_2m": ("x", np.zeros(2))})
    assert em._probe_variable(ds) == "precipitation_surface"
    ds2 = xr.Dataset({"temperature_2m": ("x", np.zeros(2)), "precipitation_surface": ("x", np.zeros(2))})
    assert em._probe_variable(ds2) == "temperature_2m"


def test_extended_range_stores_are_retired():
    assert not any("46d" in key for key in GRID_STORES)
