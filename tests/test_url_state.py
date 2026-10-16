"""Pure query-param parsing."""

import pytest

from utils import scorecard
from utils.url_state import query_choice


def test_obs_mode_from_the_query_falls_back_to_raw():
    def obs(params, variable="temperature_2m"):
        return query_choice(params, "obs", scorecard.obs_modes(variable), "raw")

    assert obs({}) == "raw"
    assert obs({"obs": "representativeness"}) == "representativeness"
    assert obs({"obs": "height_corrected"}) == "height_corrected"
    assert obs({"obs": "height_corrected"}, "wind_speed_10m") == "raw"
    assert obs({"obs": "HEIGHT"}) == "raw"


@pytest.mark.parametrize(
    ("key", "good", "default"),
    [("diurnal", "error_sd", "bias"), ("rank", "10", "3"), ("thr", "5", "1"),
     ("group", "8-15", "1-3")],
)
def test_diagnostic_controls_parse_and_fall_back(key, good, default):
    import dash

    dash.Dash(__name__, use_pages=True, pages_folder="", suppress_callback_exceptions=True)
    import importlib

    page = importlib.import_module("pages.evaluation")
    valid = {"diurnal": page._DIURNAL_SCORES, "rank": page._LEADS,
             "thr": page._THRESHOLDS, "group": page._GROUPS}[key]
    assert query_choice({key: good}, key, valid, default) == good
    for bad in ("", "nope", "7", "2-9", "crps"):
        assert query_choice({key: bad}, key, valid, default) == default
    ids = {v: k for k, v in page.URL_KEYS.items()}
    control = _find_id(page.layout(**{key: good}), ids[key])
    assert control.value == good
    assert _find_id(page.layout(**{key: "nope"}), ids[key]).value == default


def _find_id(component, component_id):
    if getattr(component, "id", None) == component_id:
        return component
    children = getattr(component, "children", None)
    for child in children if isinstance(children, (list, tuple)) else [children]:
        if child is not None and not isinstance(child, str):
            found = _find_id(child, component_id)
            if found is not None:
                return found
    return None
