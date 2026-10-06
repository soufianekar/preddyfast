import io

import numpy as np
from matplotlib.image import imread

from utils.settings import THEMES
from utils.static_figures import render_field_map


def _corner(png: bytes) -> str:
    img = imread(io.BytesIO(png))
    r, g, b = (round(float(v) * 255) for v in img[2, 2, :3])
    return f"#{r:02X}{g:02X}{b:02X}"


def _png(theme: str) -> bytes:
    lon = np.arange(0.0, 10.25, 0.25)
    lat = np.arange(40.0, 50.25, 0.25)
    values = np.add.outer(lat, lon)
    return render_field_map(
        lon, lat, values, extent=(40.0, 50.0, 0.0, 10.0), projection="platecarree",
        shading={"levels": np.linspace(40, 60, 9), "cmap": "viridis", "extend": "both"},
        colorbar_label="°C", title="t", subtitle="s", footer="f", theme=theme,
    )


def test_dark_map_has_the_dark_panel_ground():
    assert _corner(_png("dark")) == THEMES["dark"]["panel"].upper()


def test_light_map_has_the_light_panel_ground():
    assert _corner(_png("light")) == THEMES["light"]["panel"].upper()


import pandas as pd
import pytest

from utils.static_figures import render_index_timeseries


@pytest.mark.parametrize("theme", ["dark", "light"])
def test_index_timeseries_renders_in_both_themes(theme):
    idx = pd.date_range("2000-01-01", periods=36, freq="MS")
    frame = pd.DataFrame({"value": np.sin(np.arange(36) / 4), "monthly": np.sin(np.arange(36) / 4)}, index=idx)
    png = render_index_timeseries(frame, {"unit": "°C", "smooth": 3, "thresholds": (0.5, -0.5),
                                          "threshold_labels": ("warm", "cold")},
                                  title="t", subtitle="s", footer="f", theme=theme)
    assert _corner(png) == THEMES[theme]["panel"].upper()
