"""Scorecard figures from tiny hand-built aggregate tables (no data reads)."""

import numpy as np
import pandas as pd
import pytest
from test_palette_contrast import contrast

from utils import figures
from utils.settings import GRID_STORES, SCORECARD_STORES, SCORECARD_VARIABLES, THEMES

CTX = {
    "n_stations": 120,
    "period_days": 30,
    "init_hour": 0,
    "end": pd.Timestamp("2026-10-05"),
    "baseline": "ifs_ens",
}


def _table(variable="temperature_2m", score="rmse", stores=("ifs_ens", "gefs", "icon_eu"),
           days=range(1, 16), low=((("gefs", 3)),)):
    rows = []
    for k, store in enumerate(stores):
        horizon = 5 if store == "icon_eu" else 15
        for d in days:
            value = np.nan if d > horizon else 1.0 + 0.1 * d + 0.05 * k
            rows.append({
                "variable": variable, "store": store, "lead_day": d, score: value,
                "n": 40 if (store, d) in low else 500, "n_inits": 30,
                "n_stations": 120, "low_n": (store, d) in low,
            })
    return pd.DataFrame(rows)


def _heat(fig):
    (trace,) = [t for t in fig.data if t.type == "heatmap"]
    return trace


def _ctx(**kw):
    return {**CTX, **kw}


def test_scorecard_fig_builds_with_models_as_rows_and_15_days():
    fig = figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX)
    h = _heat(fig)
    assert list(h.x) == list(range(1, 16))
    order = [GRID_STORES[s]["label"] for s in SCORECARD_STORES if s in ("ifs_ens", "gefs", "icon_eu")]
    assert list(h.y) == order
    assert "RMSE" in figures.figure_text(fig)


def test_relative_heatmap_is_symmetric_about_zero_and_positive_is_good():
    fig = figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", True, CTX)
    h = _heat(fig)
    assert h.zmid == 0
    assert h.zmin == -h.zmax and h.zmax > 0
    assert "better than" in h.colorbar.title.text
    assert GRID_STORES["ifs_ens"]["label"] in h.colorbar.title.text
    # gefs rmse is higher than the baseline's: negative (worse)
    row = list(h.y).index(GRID_STORES["gefs"]["label"])
    assert h.z[row][0] < 0
    # baseline row is zero
    assert h.z[list(h.y).index(GRID_STORES["ifs_ens"]["label"] + " (baseline)")][0] == 0


def test_absolute_bias_is_diverging_about_zero():
    fig = figures.make_scorecard_fig(
        _table(score="bias").assign(bias=lambda d: d["bias"] - 1.4), "temperature_2m", "bias", False, CTX
    )
    h = _heat(fig)
    assert h.zmid == 0 and h.zmin == -h.zmax


def test_missing_horizon_is_blank_not_zero():
    h = _heat(figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX))
    row = list(h.y).index(GRID_STORES["icon_eu"]["label"])
    assert h.z[row][5] is None or np.isnan(h.z[row][5])  # day 6
    assert not np.isnan(h.z[row][4])  # day 5


def test_low_n_cell_is_marked_and_flagged_in_hover():
    fig = figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX)
    marks = [t for t in fig.data if t.type == "scatter" and t.name == "n below minimum"]
    assert len(marks) == 1
    assert list(marks[0].x) == [3] and list(marks[0].y) == [GRID_STORES["gefs"]["label"]]
    h = _heat(fig)
    row, col = list(h.y).index(GRID_STORES["gefs"]["label"]), 2
    assert h.customdata[row][col][4] == " (n<50)"
    assert h.customdata[row][col + 1][4] == ""
    assert "%{customdata[4]}" in h.hovertemplate


@pytest.mark.parametrize("theme", list(THEMES))
@pytest.mark.parametrize("store", SCORECARD_STORES)
def test_store_colours_readable_on_both_panels(store, theme):
    assert contrast(figures._store_color(store), THEMES[theme]["panel"]) >= 3.0


def test_z500_reference_sentence_only_for_z500():
    z = _table("z500", "rmse")
    t = _table("temperature_2m", "rmse")
    z_text = figures.figure_text(figures.make_scorecard_fig(z, "z500", "rmse", False, CTX))
    t_text = figures.figure_text(figures.make_scorecard_fig(t, "temperature_2m", "rmse", False, CTX))
    assert "pseudo-analysis: IFS control T+0, favours IFS and AIFS" in z_text
    assert "pseudo-analysis" not in t_text


def test_subtitle_states_the_fixed_caveats():
    text = figures.figure_text(
        figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX)
    )
    for phrase in ("120 stations", "30 days", "00 UTC runs", "±15 min",
                   "nearest grid cell, no height correction", "coastal cells may be partly sea",
                   "common inits"):
        assert phrase in text, phrase
    precip = figures.figure_text(
        figures.make_scorecard_fig(_table("precipitation_6h", "mae"), "precipitation_6h", "mae", False, CTX)
    )
    assert "6 h totals ending 00/06/12/18 UTC" in precip


def test_crps_note_only_with_a_deterministic_model_in_a_crps_figure():
    with_det = figures.figure_text(figures.make_scorecard_fig(
        _table(score="crps"), "temperature_2m", "crps", False, CTX))
    assert "CRPS of a deterministic model equals its MAE" in with_det
    assert "fair crps" in with_det.lower()
    ens_only = figures.figure_text(figures.make_scorecard_fig(
        _table(score="crps", stores=("ifs_ens", "gefs")), "temperature_2m", "crps", False, CTX))
    assert "equals its MAE" not in ens_only
    rmse = figures.figure_text(figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX))
    assert "equals its MAE" not in rmse


def test_spread_skill_lines_skip_deterministic_and_draw_ideal_one():
    df = _table(score="spread_skill", stores=("ifs_ens", "gefs", "icon_eu", "gfs"))
    fig = figures.make_score_lead_fig(df, "temperature_2m", "spread_skill", CTX)
    names = [t.name for t in fig.data if t.type == "scatter"]
    assert GRID_STORES["icon_eu"]["label"] not in names and GRID_STORES["gfs"]["label"] not in names
    assert GRID_STORES["ifs_ens"]["label"] in names
    ideal = [s for s in fig.layout.shapes if s.y0 == 1 and s.y1 == 1]
    assert len(ideal) == 1 and ideal[0].line.dash == "dash"


def test_bias_and_frequency_bias_reference_lines():
    bias = figures.make_score_lead_fig(_table(score="bias"), "temperature_2m", "bias", CTX)
    assert any(s.y0 == 0 for s in bias.layout.shapes)
    fb = figures.make_score_lead_fig(
        _table("precipitation_6h", "frequency_bias"), "precipitation_6h", "frequency_bias", CTX)
    assert any(s.y0 == 1 for s in fb.layout.shapes)
    assert not figures.make_score_lead_fig(
        _table(), "temperature_2m", "rmse", CTX).layout.shapes


def test_lead_fig_hover_carries_n_and_unit():
    fig = figures.make_score_lead_fig(_table(), "temperature_2m", "rmse", CTX)
    t = next(t for t in fig.data if t.type == "scatter")
    assert "n=" in t.hovertemplate and "°C" in t.hovertemplate


def test_station_map_one_dot_per_station_with_hover():
    df = pd.DataFrame({
        "variable": "temperature_2m", "store": "ifs_ens", "station": ["A", "B", "C"],
        "lat": [48.1, 50.9, 53.6], "lon": [11.5, 6.9, 10.0], "name": ["Munich", "Cologne", "Hamburg"],
        "rmse": [1.0, 1.5, 2.0], "n": [100, 90, 80],
    })
    fig = figures.make_station_score_map(df, "temperature_2m", "rmse", CTX)
    dots = [t for t in fig.data if t.name == "stations"]
    assert len(dots) == 1 and len(dots[0].x) == 3
    assert list(dots[0].marker.color) == [1.0, 1.5, 2.0]
    assert "°C" in dots[0].hovertemplate and "n=" in dots[0].hovertemplate
    assert dots[0].marker.colorbar.title.text


def test_every_scorecard_variable_has_a_figure_label():
    for var in SCORECARD_VARIABLES:
        assert figures.make_scorecard_fig(
            _table(var, "rmse"), var, "rmse", False, CTX).data


@pytest.mark.parametrize("kind", ["scorecard", "lead", "map"])
@pytest.mark.parametrize("wrong_variable", [False, True])
def test_empty_tables_give_the_empty_figure(kind, wrong_variable):
    df = _table() if wrong_variable else _table().iloc[0:0]
    df = df.assign(lat=50.0, lon=10.0, station="A")
    var = "wind_speed_10m" if wrong_variable else "temperature_2m"
    fig = {
        "scorecard": lambda: figures.make_scorecard_fig(df, var, "rmse", True, CTX),
        "lead": lambda: figures.make_score_lead_fig(df, var, "rmse", CTX),
        "map": lambda: figures.make_station_score_map(df, var, "rmse", CTX),
    }[kind]()
    assert not fig.data
    assert "No data available" in [a.text for a in fig.layout.annotations]
