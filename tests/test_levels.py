"""Levels: Default (fixed stepped) or fitted to the drawn area, Auto choosing between them."""

import numpy as np
import pytest

from utils.map_products import _shared_levels, fitted_levels, resolve_levels
from utils.settings import GRID_VARIABLES, MAP_PALETTES

STYLE = MAP_PALETTES[GRID_VARIABLES["temperature_2m"]["palette"]]


def test_fitted_levels_span_the_areas_bulk_with_round_steps():
    rng = np.random.default_rng(0)
    values = rng.normal(15.0, 3.0, 5000)
    levels = fitted_levels(values, 10)
    lo, hi = np.percentile(values, [2, 98])
    assert len(levels) == 11 and levels[0] <= lo and levels[-1] >= hi
    step = np.diff(levels)
    assert np.allclose(step, step[0]) and step[0] in (0.5, 1.0, 2.0, 2.5, 5.0)


def test_a_flat_field_still_gets_levels():
    assert len(fitted_levels(np.full(10, 3.0), 6)) == 7


@pytest.mark.parametrize("choice,cut,expected", [
    ("auto", True, "fitted"), ("auto", False, "default"), ("default", True, "default"), ("fitted", False, "fitted"),
])
def test_auto_fits_only_a_field_cut_from_a_larger_one(choice, cut, expected):
    assert resolve_levels(choice, cut) == expected


def test_fitted_animation_levels_keep_the_palette_and_extend_both_ways():
    a = np.full((2, 3, 3), 10.0)
    b = np.full((2, 3, 3), 20.0)
    shading, _, _ = _shared_levels({"a": a, "b": b}, "mean", "default", STYLE, GRID_VARIABLES["temperature_2m"],
                                   None, "°C", levels="fitted")
    assert shading["extend"] == "both" and shading["colors"] == STYLE["colors"].split("/")
    assert shading["levels"][0] <= 10.0 and shading["levels"][-1] >= 20.0
    default, _, _ = _shared_levels({"a": a}, "mean", "default", STYLE, GRID_VARIABLES["temperature_2m"], None, "°C")
    assert "extend" not in default  # Default keeps its fixed levels


def test_fitted_stepped_colours_render(tmp_path):
    from utils.static_figures import render_field_map

    lon = np.linspace(0, 10, 11)
    lat = np.linspace(40, 50, 11)
    values = np.add.outer(lat, lon)
    colors = STYLE["colors"].split("/")
    png = render_field_map(lon, lat, values, extent=(40, 50, 0, 10), projection="plate_carree",
                           shading={"levels": fitted_levels(values, 8), "colors": colors, "extend": "both"},
                           colorbar_label="°C", title="t", subtitle="s", footer="f")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
