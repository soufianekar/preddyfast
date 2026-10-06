import numpy as np

from components.model_select import model_options
from utils import earthmover_api as em
from utils.settings import GRID_STORES


def test_daily_steps_cannot_build_six_hours():
    daily = np.arange(0, 1105, 24.0)
    assert em._buildable(daily, 24) and not em._buildable(daily, 6)


def test_three_then_six_hourly_builds_all_windows():
    lead = np.r_[np.arange(0, 145, 3), np.arange(150, 361, 6)].astype(float)
    assert all(em._buildable(lead, p) for p in (6, 12, 24))


def test_model_options_grouped_by_family_with_members_and_grid():
    groups = {g["group"]: g["items"] for g in model_options()}
    assert set(groups) == {s["family"] for s in GRID_STORES.values()}
    gefs = next(i for i in groups["NOAA"] if i["value"] == "gefs")
    assert gefs["label"] == "NOAA GEFS · 31 members · 0.25°"
    gfs = next(i for i in groups["NOAA"] if i["value"] == "gfs")
    assert gfs["label"] == "NOAA GFS · deterministic · 0.25°"


def test_meteogram_horizons_survive_a_store_that_will_not_open(monkeypatch):
    from pages import meteogram as mg
    from utils.earthmover_api import GridDataError

    def boom(store):
        raise GridDataError("unreachable")

    monkeypatch.setattr(mg, "store_horizon", boom)
    values = [o["value"] for o in mg._past_day_options(["gefs"])]
    assert values and values[-1] == "14"


def test_retheme_rebuilds_the_animation_on_screen_not_the_current_controls():
    import dash

    dash.Dash(__name__, use_pages=True, pages_folder="", suppress_callback_exceptions=True)
    from pages import regional as rg
    built = {"args": ["box:45,50,5,10", "temperature_2m", None, 24, 72, None, "mean", 90.0, [10, 90],
                      "auto", "data", "default", ["ifs_ens", "gfs"], "dark"]}
    args = rg._rethemed_args(built, "light")
    # Built before the Levels control (14 args): theme replaced, levels Auto appended.
    assert args[13] == "light" and args[14] == "auto" and args[0] == "box:45,50,5,10"
    assert args[8] == (10, 90) and args[12] == ("ifs_ens", "gfs")  # hashable for the memo key
