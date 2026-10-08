"""station_cube: one box read per chunk-tile group of stations, every station's cell cut from it."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from utils import earthmover_api as em

INIT = pd.Timestamp("2026-09-01 00:00")
LEADS_3H = np.arange(0, 49, 3.0)
LEADS_IFS = np.r_[np.arange(0, 145, 3.0), 150.0, 156.0]
GLOBAL = {"global": True, "lat_range": (-90.0, 90.0), "lon_range": (-180.0, 179.75), "res_deg": 0.25}
ICON = {"global": False, "lat_range": (29.5, 70.5), "lon_range": (-23.5, 62.5), "res_deg": 0.0625}


def _grid(res, lat0, lat1, lon0, lon1):
    lat = np.round(np.arange(lat1, lat0 - res / 2, -res), 6)  # descending, as stored
    lon = np.round(np.arange(lon0, lon1 + res / 2, res), 6)
    return lat, lon


def _ds(fields, *, res=0.25, members=3, leads=LEADS_3H, box=(46.0, 56.0, 5.0, 16.0), step_types=None, tile=None):
    """A synthetic store: `fields[name](member, lead, lat, lon)` -> values (broadcast).

    `tile` sets a zarr-style chunk encoding of `tile` x `tile` lat/lon cells;
    without it the store has no chunking and the whole grid is one tile.
    """
    lat, lon = _grid(res, *box)
    lead_td = pd.to_timedelta(leads, unit="h")
    m = np.arange(members)
    dims = ("init_time", "ensemble_member", "lead_time", "latitude", "longitude")
    data_vars = {}
    for name, fn in fields.items():
        arr = fn(m[:, None, None, None], np.asarray(leads)[None, :, None, None],
                 lat[None, None, :, None], lon[None, None, None, :])
        arr = np.broadcast_to(arr, (members, len(leads), len(lat), len(lon))).astype("float32")
        da = xr.DataArray(arr[None], dims=dims)
        if members == 1:
            da = da.isel(ensemble_member=0, drop=True)
        da.attrs["step_type"] = (step_types or {}).get(name, "instant")
        if tile:
            da.encoding["chunks"] = tuple(
                tile if d in ("latitude", "longitude") else da.sizes[d] for d in da.dims
            )
        data_vars[name] = da
    coords = {"init_time": [INIT], "lead_time": lead_td, "latitude": lat, "longitude": lon}
    if members > 1:
        coords["ensemble_member"] = m
    return xr.Dataset(data_vars, coords=coords)


def _cell_value(m, lead, lat, lon):
    return 1000.0 * m + lead + 0.01 * np.round(lat * 16) + 0.0001 * np.round(lon * 16)


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Patch the store, its geometry, the budget estimate and the cache dir; count opens and reads."""
    state = {"stores": {}, "opens": [], "reads": [], "boxes": [], "budget": []}
    monkeypatch.setattr(em, "POINT_CACHE_DIR", str(tmp_path))
    monkeypatch.setattr(em, "store_geometry", lambda s: ICON if s == "icon_eu" else GLOBAL)
    monkeypatch.setattr(em, "read_estimate", lambda stores, bounds, n_inits=1, horizon_hours=None: (1, 1.0))

    def open_(store):
        state["opens"].append(store)
        return state["stores"][store]

    real_select = em._select_box

    def select(ds, variable, init, bounds):
        state["reads"].append(variable)
        da = real_select(ds, variable, init, bounds)
        state["boxes"].append((da.sizes["latitude"], da.sizes["longitude"]))
        return da

    real_budget = em.check_station_read_budget

    def budget(*args, **kwargs):
        state["budget"].append((args, kwargs))
        return real_budget(*args, **kwargs)

    monkeypatch.setattr(em, "_open_dataset", open_)
    monkeypatch.setattr(em, "_select_box", select)
    monkeypatch.setattr(em, "check_station_read_budget", budget)
    return state


def _stations(*rows):
    return pd.DataFrame(rows, columns=["id", "lat", "lon"])


def test_two_stations_in_one_cell_share_values(env):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value})
    st = _stations(("A", 50.01, 10.02), ("B", 49.95, 9.97), ("C", 52.5, 13.4))
    cube = em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    assert cube.dims == ("station", "member", "lead_hours")
    assert list(cube["station"].values) == ["A", "B", "C"]
    np.testing.assert_array_equal(cube.sel(station="A").values, cube.sel(station="B").values)
    assert float(cube["lat"].sel(station="A")) == 50.0 and float(cube["lon"].sel(station="A")) == 10.0
    expected = _cell_value(np.arange(3)[:, None], LEADS_3H[None, :], 52.5, 13.5)
    np.testing.assert_allclose(cube.sel(station="C").values, expected, rtol=1e-6)
    assert env["reads"] == ["temperature_2m"]  # one box read, not one per station


def test_a_station_on_a_cell_edge_snaps_by_the_stores_own_spacing(env):
    env["stores"]["icon_eu"] = _ds({"temperature_2m": _cell_value}, res=0.0625, members=1)
    env["stores"]["aifs_single"] = _ds({"temperature_2m": _cell_value}, members=1)
    st = _stations(("E", 50.125, 10.125), ("F", 50.1, 10.1))
    icon = em.station_cube("icon_eu", "temperature_2m", INIT, st, None)
    glob = em.station_cube("aifs_single", "temperature_2m", INIT, st, None)
    assert icon.attrs["res_deg"] == 0.0625 and glob.attrs["res_deg"] == 0.25
    assert list(icon["lat"].values) == [50.125, 50.125] and list(icon["lon"].values) == [10.125, 10.125]
    # 0.25°: 50.125 is the edge between the 50.0 and 50.25 cells; round-half-even, as the point reads.
    assert list(glob["lat"].values) == [50.0, 50.0] and list(glob["lon"].values) == [10.0, 10.0]
    assert icon.sizes["member"] == 1 and glob.sizes["member"] == 1


def test_wind_speed_is_per_member_never_the_speed_of_the_mean_wind(env):
    sign = lambda m: np.where(m == 0, 1.0, -1.0)  # noqa: E731
    env["stores"]["ifs_ens"] = _ds(
        {"wind_u_10m": lambda m, t, la, lo: 3.0 * sign(m) + 0 * t + 0 * la + 0 * lo,
         "wind_v_10m": lambda m, t, la, lo: 4.0 * sign(m) + 0 * t + 0 * la + 0 * lo},
        members=2,
    )
    cube = em.station_cube("ifs_ens", "wind_speed_10m", INIT, _stations(("A", 50.0, 10.0)), None)
    np.testing.assert_allclose(cube.values, 5.0)  # mean wind of (3,4) and (-3,-4) is 0
    assert sorted(env["reads"]) == ["wind_u_10m", "wind_v_10m"]
    assert len(env["budget"]) == 1 and len(env["budget"][0][0][1]) == 2  # both components budgeted


def test_precipitation_6h_windows_total_the_avg_rate_and_end_at_synoptic_hours(env):
    rate = 1.0 / 3600.0  # kg m-2 s-1 = 1 mm/h
    env["stores"]["ifs_ens"] = _ds(
        {"precipitation_surface": lambda m, t, la, lo: np.where(t == 0, np.nan, rate * (m + 1)) + 0 * la + 0 * lo},
        step_types={"precipitation_surface": "avg"},
        leads=LEADS_IFS,  # 3 h to T+144, 6 h after
    )
    cube = em.station_cube("ifs_ens", "precipitation_surface", INIT, _stations(("A", 50.0, 10.0)), 6)
    expected_leads = np.arange(6.0, 157.0, 6.0)
    np.testing.assert_array_equal(cube["lead_hours"].values, expected_leads)
    assert set(pd.DatetimeIndex(cube["valid_time"].values).hour) <= {0, 6, 12, 18}
    # Constant rate: every 6 h window, before and past the T+144 step change, totals 6 x rate.
    np.testing.assert_allclose(
        cube.sel(station="A").values, 6.0 * (np.arange(3)[:, None] + 1) * np.ones((1, expected_leads.size))
    )
    assert cube.attrs["window_hours"] == 6 and cube.attrs["step_type"] == "avg" and cube.attrs["unit"] == "mm"


def test_valid_time_is_init_plus_lead_and_attrs_carry_provenance(env):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value})
    cube = em.station_cube("ifs_ens", "temperature_2m", INIT, _stations(("A", 50.0, 10.0)), None)
    expected = INIT + pd.to_timedelta(cube["lead_hours"].values, unit="h")
    assert (pd.DatetimeIndex(cube["valid_time"].values) == expected).all()
    assert cube["valid_time"].dims == ("lead_hours",)
    a = cube.attrs
    assert a["init"] == INIT.isoformat() and a["store"] == "ifs_ens" and a["variable"] == "temperature_2m"
    assert a["window_hours"] is None and a["step_type"] == "instant" and a["dropped_outside_domain"] == 0


def test_budget_checked_once_and_a_refusal_raises_before_any_read(env, monkeypatch):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value})
    monkeypatch.setattr(em, "read_estimate", lambda *a, **k: (999, 5000.0))
    with pytest.raises(em.GridDataError, match="MB"):
        em.station_cube("ifs_ens", "temperature_2m", INIT, _stations(("A", 50.0, 10.0)), None, budget_mb=100.0)
    assert env["reads"] == [] and len(env["budget"]) == 1
    assert env["budget"][0][1]["budget_mb"] == 100.0
    # A scorecard budget lets the same read through.
    cube = em.station_cube("ifs_ens", "temperature_2m", INIT, _stations(("A", 50.0, 10.0)), None, budget_mb=1e4)
    assert cube.sizes["station"] == 1 and len(env["budget"]) == 2


def test_check_read_budget_default_unchanged_and_override(monkeypatch):
    monkeypatch.setattr(em, "read_estimate", lambda *a, **k: (10, em.READ_BUDGET_MB + 1))
    with pytest.raises(em.GridDataError, match="interactive limit"):
        em.check_read_budget(("ifs_ens",), (47, 55, 6, 15))
    em.check_read_budget(("ifs_ens",), (47, 55, 6, 15), budget_mb=em.READ_BUDGET_MB * 2)


def test_second_call_is_served_from_disk_without_opening_the_store(env):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value})
    st = _stations(("A", 50.0, 10.0), ("B", 52.5, 13.4))
    first = em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    env["opens"].clear()
    env["stores"].clear()  # any open now would KeyError
    second = em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    assert env["opens"] == [] and len(env["budget"]) == 1
    xr.testing.assert_identical(first, second)
    assert second.attrs["window_hours"] is None
    assert first.attrs == second.attrs
    for key in ("dropped_outside_domain",):
        assert type(first.attrs[key]) is type(second.attrs[key]) is int
    assert type(second.attrs["res_deg"]) is float


def test_cached_window_hours_is_a_plain_int(env):
    env["stores"]["ifs_ens"] = _ds(
        {"precipitation_surface": lambda m, t, la, lo: np.where(t == 0, np.nan, 1e-4) + 0 * m + 0 * la + 0 * lo},
        step_types={"precipitation_surface": "avg"},
    )
    st = _stations(("A", 50.0, 10.0))
    first = em.station_cube("ifs_ens", "precipitation_surface", INIT, st, 6)
    second = em.station_cube("ifs_ens", "precipitation_surface", INIT, st, 6)
    assert type(second.attrs["window_hours"]) is int and first.attrs == second.attrs


def test_a_station_nan_at_every_lead_refuses_the_run_and_writes_nothing(env, tmp_path):
    def field(m, t, la, lo):
        hole = (np.abs(la - 52.5) < 1e-6) & (np.abs(lo - 13.5) < 1e-6)
        return np.where(hole & (t > 0), np.nan, _cell_value(m, t, la, lo))

    env["stores"]["ifs_ens"] = _ds({"temperature_2m": field})
    st = _stations(("A", 50.0, 10.0), ("B", 52.5, 13.4))
    with pytest.raises(em.GridDataError, match=r"16 lead\(s\) non-finite at 1 of 2 station"):
        em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    assert not list(tmp_path.rglob("*.nc")) and not list(tmp_path.rglob("*.part"))


def test_one_interior_nan_lead_at_one_member_refuses_the_run(env, tmp_path):
    def field(m, t, la, lo):
        return np.where((m == 1) & (t == 21), np.nan, _cell_value(m, t, la, lo))

    env["stores"]["ifs_ens"] = _ds({"temperature_2m": field})
    with pytest.raises(em.GridDataError, match="incomplete"):
        em.station_cube("ifs_ens", "temperature_2m", INIT, _stations(("A", 50.0, 10.0)), None)
    assert not list(tmp_path.rglob("*.nc"))


def test_an_instant_field_nan_at_t0_is_refused_but_avg_is_exempt(env):
    env["stores"]["ifs_ens"] = _ds(
        {"temperature_2m": lambda m, t, la, lo: np.where(t == 0, np.nan, _cell_value(m, t, la, lo))}
    )
    with pytest.raises(em.GridDataError, match="incomplete"):
        em.station_cube("ifs_ens", "temperature_2m", INIT, _stations(("A", 50.0, 10.0)), None)


def test_ifs_control_is_cut_from_the_ifs_ens_file(env):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value})
    st = _stations(("A", 50.0, 10.0))
    ens = em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    env["opens"].clear()
    ctrl = em.station_cube("ifs_control", "temperature_2m", INIT, st, None)
    assert env["opens"] == [] and env["reads"] == ["temperature_2m"]
    assert list(ctrl["member"].values) == [0] and ctrl.attrs["store"] == "ifs_control"
    np.testing.assert_array_equal(ctrl.values[:, 0], ens.sel(member=0).values)
    assert ens.attrs["store"] == "ifs_ens"  # the source's cube is not relabelled


def test_icon_eu_drops_and_counts_stations_outside_its_domain(env):
    env["stores"]["icon_eu"] = _ds({"temperature_2m": _cell_value}, res=0.0625, members=1, box=(29.5, 70.5, -23.5, 30.0))
    st = _stations(("IN", 50.0, 10.0), ("NORTH", 78.2, 15.6), ("WEST", 40.0, -30.0))
    cube = em.station_cube("icon_eu", "temperature_2m", INIT, st, None)
    assert list(cube["station"].values) == ["IN"] and cube.attrs["dropped_outside_domain"] == 2


def test_cache_path_depends_on_the_station_list_and_window():
    a = _stations(("A", 50.0, 10.0), ("B", 52.5, 13.4))
    b = _stations(("A", 50.0, 10.0), ("C", 52.5, 13.4))
    pa = em._station_cache_path("ifs_ens", "precipitation_surface", INIT, a, 6)
    assert pa != em._station_cache_path("ifs_ens", "precipitation_surface", INIT, b, 6)
    assert pa != em._station_cache_path("ifs_ens", "precipitation_surface", INIT, a, None)
    assert pa == em._station_cache_path("ifs_ens", "precipitation_surface", INIT, a.copy(), 6)
    assert pa.parent.parts[-3:] == ("stations", "ifs_ens", "precipitation_surface")
    assert pa.name.startswith("2026090100-") and pa.suffix == ".nc"


def test_cache_key_carries_a_format_version(monkeypatch):
    a = _stations(("A", 50.0, 10.0))
    before = em._station_cache_path("ifs_ens", "temperature_2m", INIT, a, None)
    monkeypatch.setattr(em, "_STATION_CUBE_VERSION", em._STATION_CUBE_VERSION + 1)
    assert em._station_cache_path("ifs_ens", "temperature_2m", INIT, a, None) != before


# --- Tile-group reads (R21): only the chunk tiles that hold stations -----------------
# Grid of `_ds`: 0.25°, latitude descending from 56 (row = (56 - lat) / 0.25),
# longitude from 5 (col = (lon - 5) / 0.25). With 8 x 8 tiles:
#   A (50.0, 10.0)  -> row 24, col 20 -> tile (3, 2)
#   A2 (49.5, 10.25) -> row 26, col 21 -> tile (3, 2), same as A
#   C (52.5, 13.5)  -> row 14, col 34 -> tile (1, 4); A..C bbox spans 3 x 3 tiles
#   D (50.0, 12.25) -> row 24, col 29 -> tile (3, 3); A..D bbox is exactly 2 tiles


def test_stations_in_different_tiles_read_one_small_box_each(env):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value}, tile=8)
    st = _stations(("A", 50.0, 10.0), ("C", 52.5, 13.5))
    em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    assert env["reads"] == ["temperature_2m", "temperature_2m"]
    assert env["boxes"] == [(1, 1), (1, 1)]  # each group's own cells, never the A..C box


def test_stations_in_one_tile_are_read_together(env):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value}, tile=8)
    st = _stations(("A", 50.0, 10.0), ("A2", 49.5, 10.25))
    em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    assert env["reads"] == ["temperature_2m"]
    assert env["boxes"] == [(3, 2)]  # the minimal box over the group's cells


def test_a_dense_network_whose_bbox_costs_no_more_keeps_one_read(env):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value}, tile=8)
    st = _stations(("A", 50.0, 10.0), ("D", 50.0, 12.25))  # bbox = 2 tiles = 2 groups: a tie
    em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    assert env["reads"] == ["temperature_2m"] and env["boxes"] == [(1, 10)]


def test_tile_group_values_equal_one_big_bbox_read_in_station_order(env):
    st = _stations(("C", 52.5, 13.5), ("A", 50.0, 10.0), ("D", 50.0, 12.25), ("A2", 49.5, 10.25))
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value}, tile=8)
    grouped = em.station_cube("ifs_ens", "temperature_2m", INIT, st, None)
    assert len(env["reads"]) == 3  # tiles (1,4), (3,2), (3,3)
    env["reads"].clear()
    env["stores"]["aifs_ens"] = _ds({"temperature_2m": _cell_value})  # unchunked: one bbox
    single = em.station_cube("aifs_ens", "temperature_2m", INIT, st, None)
    assert env["reads"] == ["temperature_2m"]
    assert list(grouped["station"].values) == ["C", "A", "D", "A2"]
    np.testing.assert_array_equal(grouped.values, single.values)
    np.testing.assert_array_equal(grouped["lat"].values, single["lat"].values)
    np.testing.assert_array_equal(grouped["lon"].values, single["lon"].values)


def test_budget_sums_the_group_boxes_and_refuses_before_any_read(env, monkeypatch):
    env["stores"]["ifs_ens"] = _ds({"temperature_2m": _cell_value}, tile=8)
    asked = []

    def estimate(stores, bounds, n_inits=1, horizon_hours=None):
        asked.append(bounds)
        return 1, 600.0

    monkeypatch.setattr(em, "read_estimate", estimate)
    st = _stations(("A", 50.0, 10.0), ("C", 52.5, 13.5))
    with pytest.raises(em.GridDataError, match="1,200 MB"):
        em.station_cube("ifs_ens", "temperature_2m", INIT, st, None, budget_mb=1000.0)
    assert env["reads"] == [] and len(asked) == 2
    # Each group box is asked for, never the A..C bounding box.
    for lat_min, lat_max, lon_min, lon_max in asked:
        assert lat_max - lat_min < 0.25 and lon_max - lon_min < 0.25
    em.station_cube("ifs_ens", "temperature_2m", INIT, st, None, budget_mb=1200.0)
    assert len(env["reads"]) == 2


def test_wind_budgets_both_components_of_every_group(env, monkeypatch):
    zero = lambda m, t, la, lo: 1.0 + 0 * m + 0 * t + 0 * la + 0 * lo  # noqa: E731
    env["stores"]["ifs_ens"] = _ds({"wind_u_10m": zero, "wind_v_10m": zero}, tile=8)
    calls = []
    monkeypatch.setattr(em, "read_estimate", lambda *a, **k: calls.append(a) or (1, 300.0))
    st = _stations(("A", 50.0, 10.0), ("C", 52.5, 13.5))
    with pytest.raises(em.GridDataError, match="1,200 MB"):
        em.station_cube("ifs_ens", "wind_speed_10m", INIT, st, None, budget_mb=1000.0)
    assert len(calls) == 4 and env["reads"] == []


def test_station_cube_version_is_2_so_tile_group_files_never_reuse_v1(monkeypatch):
    a = _stations(("A", 50.0, 10.0))
    assert em._STATION_CUBE_VERSION == 2
    now = em._station_cache_path("ifs_ens", "temperature_2m", INIT, a, None)
    monkeypatch.setattr(em, "_STATION_CUBE_VERSION", 1)
    assert em._station_cache_path("ifs_ens", "temperature_2m", INIT, a, None) != now
