import numpy as np

from utils.map_products import _shared_levels
from utils.settings import GRID_VARIABLES, MAP_PALETTES

VAR = GRID_VARIABLES["temperature_2m"]
STYLE = MAP_PALETTES[VAR["palette"]]


def test_one_level_array_covers_both_models():
    cold = np.full((2, 4, 4), 0.0) + np.linspace(0, 5, 16).reshape(4, 4)
    warm = np.full((2, 4, 4), 20.0) + np.linspace(0, 5, 16).reshape(4, 4)
    shading, _, _ = _shared_levels({"ifs_ens": cold, "aifs_ens": warm}, "mean", "viridis", STYLE, VAR, None, "°C")
    levels = shading["levels"]
    assert levels[0] <= 1.0 and levels[-1] >= 24.0


def test_spread_levels_start_at_zero_and_cover_both():
    a = np.full((1, 3, 3), 1.0)
    b = np.full((1, 3, 3), 8.0)
    shading, _, label = _shared_levels({"a": a, "b": b}, "range", "default", STYLE, VAR, None, "°C")
    assert shading["levels"][0] == 0 and shading["levels"][-1] >= 7.8
    assert "spread" in label
