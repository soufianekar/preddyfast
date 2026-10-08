"""Scorecard configuration is consistent with the grid and store tables."""

from utils import settings


def test_grid_variables_exist_and_derived_parts_are_store_components():
    for key, entry in settings.SCORECARD_VARIABLES.items():
        for name in entry["grid"]:
            assert name in settings.GRID_VARIABLES, (key, name)
            parts = settings.GRID_VARIABLES[name].get("derived_from", ())
            assert set(parts) <= set(settings.STORE_RAW_COMPONENTS), (key, parts)
    assert settings.GRID_VARIABLES["wind_speed_10m"]["derived_from"]


def test_reported_total_column_is_the_ghcnh_internal_name():
    entry = settings.SCORECARD_VARIABLES["precipitation_6h"]
    assert entry["obs_reported"] == settings.GHCNH_FIELDS["precipitation_6_hour"]
    others = [v for k, v in settings.SCORECARD_VARIABLES.items() if k != "precipitation_6h"]
    assert all(v["obs_reported"] is None for v in others)


def test_stores_are_declared():
    assert settings.SCORECARD_STORES == (
        "ifs_ens", "aifs_ens", "gefs", "ifs_control", "aifs_single", "gfs", "icon_eu",
    )
    for store in settings.SCORECARD_STORES:
        assert store in settings.GRID_STORES


def test_precipitation_threshold_is_rate_times_window():
    entry = settings.SCORECARD_VARIABLES["precipitation_6h"]
    assert entry["threshold"] == 0.1 * 6
    assert entry["window_hours"] == 6


def test_entries_share_keys():
    keys = [set(v) for v in settings.SCORECARD_VARIABLES.values()]
    assert all(k == keys[0] for k in keys)
    assert set(settings.SCORECARD_VARIABLES) == {
        "temperature_2m", "precipitation_6h", "wind_speed_10m", "z500",
    }


def test_z500_uses_control_t0_without_obs():
    z = settings.SCORECARD_VARIABLES["z500"]
    assert z["reference"] == "control_t0"
    assert z["obs"] is None


def test_wind_is_derived_and_observed():
    assert settings.GRID_VARIABLES["wind_speed_10m"]["derived_from"] == (
        "wind_u_10m", "wind_v_10m",
    )
    assert settings.GHCNH_FIELDS["wind_speed"] == "wind_speed_10m"


def test_scalars():
    assert settings.SCORECARD_DAYS == 14
    assert settings.SCORECARD_INIT_HOUR == 0
    assert settings.SCORECARD_MAX_STATIONS == 200
    assert settings.SCORECARD_COUNTRY == "DE"
    assert settings.SCORECARD_PERIODS == (7, 14)
    assert max(settings.SCORECARD_PERIODS) <= settings.SCORECARD_DAYS
    assert settings.SCORECARD_FIXED_PRECIP_COMPLETENESS == 0.5
    assert settings.SCORECARD_STATIONS == (
        "UKI0000EGLL", "FRI0000LFPG", "GMI0000EDDB", "SPMU0098221",
        "ITMU0016242", "AUI0000LOWW", "GRI0000LGAV", "TUI0000LTFM",
    )


def test_scores_package_importable():
    import scores  # noqa: F401


def test_country_selects_only_german_stations(monkeypatch):
    import pandas as pd

    from utils import observations

    fake = pd.DataFrame(
        {
            "id": ["DE1", "DE2", "GM1"],
            "name": ["Berlin", "Munich", "Banjul"],
            "lat": [52.5, 48.1, 13.45],
            "lon": [13.4, 11.6, -16.6],
            "elev_m": [50.0, 520.0, 5.0],
            "country": ["DE", "DE", "GM"],
        }
    )
    monkeypatch.setattr(observations, "station_list", lambda: fake)
    st = observations.station_list()
    chosen = st[st["country"] == settings.SCORECARD_COUNTRY]
    assert set(chosen["id"]) == {"DE1", "DE2"}
    assert chosen["lat"].between(47, 55.5).all()
    assert chosen["lon"].between(5.5, 15.5).all()


def test_derived_wind_speed_is_not_offered_on_the_area_view():
    """The area readers never build a derived variable; only station_cube does."""
    import importlib

    import dash

    dash.Dash(__name__, use_pages=True, pages_folder="", suppress_callback_exceptions=True)
    regional = importlib.import_module("pages.regional")
    assert "wind_speed_10m" not in {o["value"] for o in regional._VARIABLE_OPTIONS}


def test_derived_wind_speed_from_the_url_falls_back_on_the_area_view():
    import importlib

    import dash

    dash.Dash(__name__, use_pages=True, pages_folder="", suppress_callback_exceptions=True)
    regional = importlib.import_module("pages.regional")

    def find(component, cid):
        if getattr(component, "id", None) == cid:
            return component
        children = getattr(component, "children", None)
        for child in children if isinstance(children, (list, tuple)) else [children]:
            if child is not None and hasattr(child, "to_plotly_json"):
                hit = find(child, cid)
                if hit is not None:
                    return hit
        return None

    select = find(regional.layout(variable="wind_speed_10m"), "rg-variable-select")
    assert select is not None and select.value == "temperature_2m"
