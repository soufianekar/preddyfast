"""Z500 on every page it fits: meteogram (synoptic hours), area, maps, evaluation, presets."""

import numpy as np
import pytest
from test_deterministic_latest import START, _Block, _Response

from utils import openmeteo_api as om
from utils.settings import (
    EVAL_VARIABLES,
    GRID_VARIABLES,
    PRECOMPUTE_VARIABLES,
    VARIABLE_FAMILIES,
    VARIABLES,
)

Z = "geopotential_height_500hPa"


@pytest.fixture(scope="module", autouse=True)
def _app():
    import dash

    return dash.Dash(__name__, use_pages=True, pages_folder="", suppress_callback_exceptions=True)


def test_meteogram_variable_is_hourly_at_synoptic_hours():
    meta = VARIABLES[Z]
    assert meta["synoptic_hours"] == (0, 12) and meta["climate"] is False and meta["unit"] == "m"


def test_family_links_meteogram_area_and_era5_names():
    fam = VARIABLE_FAMILIES["z500"]
    assert fam["daily"] == (Z,) and fam["grid"] == ("geopotential_height_500hpa",)
    assert fam["era5"] == ("geopotential_500",) and "geopotential_height_500hpa" in GRID_VARIABLES


def test_ensemble_synoptic_keeps_00_and_12_utc_only(monkeypatch):
    values = np.arange(48, dtype=float) + 5500.0
    monkeypatch.setattr(om._client, "weather_api", lambda url, params: [_Response(_Block(START, 48, 3600, values))])
    df = om.fetch_ensemble_synoptic(48.86, 2.35, Z, "ecmwf_ifs025", 2)
    assert list(df["date"].dt.hour) == [0, 12, 0, 12] and list(df["value"]) == [5500.0, 5512.0, 5524.0, 5536.0]


def test_deterministic_synoptic_leaves_out_a_model_without_the_field(monkeypatch):
    def weather_api(url, params):
        return [_Response(_Block(START, 48, 3600, [np.nan] * 48)), _Response(_Block(START, 48, 3600, np.full(48, 5600.0)))]

    monkeypatch.setattr(om._client, "weather_api", weather_api)
    out = om.fetch_deterministic_synoptic(48.86, 2.35, Z, ("ecmwf_ifs_hres", "gfs"), 2)
    assert set(out) == {"gfs"} and list(out["gfs"]["date"].dt.hour) == [0, 12, 0, 12]


def test_evaluation_z500_has_no_station_reference():
    spec = EVAL_VARIABLES["z500"]
    assert spec["grid"] == "geopotential_height_500hpa" and spec["obs"] is None
    from utils.eval_snapshot import usable_references

    refs = usable_references("z500")
    assert "stations" not in refs and "control_t0" in refs


def test_evaluate_point_refuses_stations_for_z500(monkeypatch):
    from utils import evaluation as ev

    monkeypatch.setattr(ev, "store_variables", lambda s: ["geopotential_height_500hpa"])
    with pytest.raises(ev.GridDataError, match="station"):
        ev.evaluate_point({"id": "X", "name": "X", "lat": 48.8, "lon": 2.3}, "z500", ("ifs_ens",), 5)


def test_z500_is_a_preset():
    assert PRECOMPUTE_VARIABLES["geopotential_height_500hpa"] is None
