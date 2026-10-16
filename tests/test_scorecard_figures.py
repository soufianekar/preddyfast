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
        figures.make_scorecard_fig(_table("precipitation_24h", "mae"), "precipitation_24h", "mae", False, CTX)
    )
    assert "24 h totals 06–06 UTC reported by the station — never hourly sums" in precip


def test_precipitation_without_a_day_1_column_still_draws():
    # 24 h windows end at T+30 at the earliest: lead day 1 has no pairs.
    table = _table("precipitation_24h", "mae", days=range(2, 16))
    fig = figures.make_scorecard_fig(table, "precipitation_24h", "mae", False, CTX)
    h = _heat(fig)
    assert 1 in list(h.x) and 15 in list(h.x)
    assert np.isnan(np.asarray(h.z, dtype=float)[:, 0]).all()
    relative = figures.make_scorecard_fig(table, "precipitation_24h", "mae", True, _ctx(baseline="ifs_ens"))
    assert _heat(relative) is not None


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
        _table("precipitation_24h", "frequency_bias"), "precipitation_24h", "frequency_bias", CTX)
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


# --- final review: R23-R25, footers, station map ----------------------------------


def _relative(table):
    return figures.make_scorecard_fig(table, "temperature_2m", "rmse", True, CTX)


def test_relative_colour_range_is_capped_at_50_percent():
    """R25: one huge percent change neither widens the scale nor loses its value."""
    calm = _table()
    wild = calm.copy()
    hit = (wild["store"] == "gefs") & (wild["lead_day"] == 2)
    wild.loc[hit, "rmse"] = 20.0  # ~ -1500 % against the baseline
    h_calm, h_wild = _heat(_relative(calm)), _heat(_relative(wild))
    assert h_wild.zmax == h_calm.zmax == 50 and h_wild.zmin == -50
    row = list(h_wild.y).index(GRID_STORES["gefs"]["label"])
    assert h_wild.customdata[row][1][1].startswith("-1")  # true value in hover
    assert float(h_wild.customdata[row][1][1].split()[0]) < -1000
    ticks = list(h_wild.colorbar.ticktext)
    assert ticks[0] == "≤−50 %" and ticks[-1] == "≥50 %"
    assert "capped at ±50 %" in figures.figure_text(_relative(wild))


def test_lead_day_ticks_show_the_fewest_runs_of_the_column():
    table = _table()
    table.loc[(table["store"] == "gefs") & (table["lead_day"] == 5), "n_inits"] = 3
    h_fig = figures.make_scorecard_fig(table, "temperature_2m", "rmse", False, CTX)
    ticks = dict(zip(h_fig.layout.xaxis.tickvals, h_fig.layout.xaxis.ticktext))
    assert ticks[5] == "Day 5<br>3 runs" and ticks[4] == "Day 4<br>30 runs"


def test_footer_states_valid_hours_run_count_and_both_sources():
    text = figures.figure_text(
        figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX)
    )
    for phrase in ("scored at 00/06/12/18 UTC valid times",
                   "long leads rest on fewer runs: the window ends at the last observation",
                   "GHCNh", "forecasts: dynamical.org archive"):
        assert phrase in text, phrase
    z = figures.figure_text(
        figures.make_scorecard_fig(_table("z500"), "z500", "rmse", False, CTX)
    )
    assert "forecasts: dynamical.org archive" in z and "00/06/12/18" not in z


def test_bias_correction_and_spread_skill_are_defined_where_shown():
    bc_def = "bias-corrected: minus each station×model mean error over the window"
    ss_def = "√((M+1)/M · mean variance) / RMSE of the ensemble mean, ideal 1"

    def text(score, fn=figures.make_scorecard_fig):
        table = _table(score=score)
        if fn is figures.make_scorecard_fig:
            fig = fn(table, "temperature_2m", score, False, CTX)
        else:
            fig = fn(table, "temperature_2m", score, CTX)
        return figures.figure_text(fig).lower()

    ss_def = ss_def.lower()

    assert bc_def in text("rmse_bc") and bc_def in text("mae_bc", figures.make_score_lead_fig)
    assert ss_def in text("spread_skill", figures.make_score_lead_fig)
    assert bc_def not in text("rmse") and ss_def not in text("rmse")


def test_station_map_of_a_score_the_model_cannot_have_is_a_note():
    df = pd.DataFrame({
        "variable": "temperature_2m", "store": "gfs", "station": ["A", "B"],
        "lat": [48.1, 50.9], "lon": [11.5, 6.9], "name": ["Munich", "Cologne"],
        "spread_skill": [np.nan, np.nan], "n": [100, 90],
    })
    fig = figures.make_station_score_map(df, "temperature_2m", "spread_skill", CTX)
    assert not [t for t in fig.data if t.name == "stations"]
    note = " ".join(a.text for a in fig.layout.annotations)
    assert GRID_STORES["gfs"]["label"] in note and "ensemble" in note


def test_cell_on_few_runs_says_so_in_hover():
    table = _table(low=())
    hit = (table["store"] == "gefs") & (table["lead_day"] == 9)
    table.loc[hit, ["n_inits", "low_n"]] = [3, True]
    h = _heat(figures.make_scorecard_fig(table, "temperature_2m", "rmse", False, CTX))
    row = list(h.y).index(GRID_STORES["gefs"]["label"])
    assert h.customdata[row][8][4] == " (<5 runs)"


# --- Task 6: display modes, significance marks, diagnostic figures --------------

MODE_HC = (
    "height-corrected with ERA5 orography (a proxy for each model's own), 0.0065 K/m"
)
MODE_REP = (
    "accounting for representativeness (ECMWF TM865 perturbed ensemble, "
    "ΔA = native grid spacing)"
)
SIG_FOOTER = (
    "• significant at 95 % (moving-block bootstrap over runs, paired against "
    "ECMWF IFS ENS)"
)


def _sig_rows(table, score="rmse", sig_by_cell=None):
    """lead_relative rows: % change plus diff/lo/hi/sig for `score`."""
    from utils.scorecard import relative_to

    rel = relative_to(table, "ifs_ens")
    rel[f"diff_{score}"] = 0.2
    rel[f"lo_{score}"] = 0.1
    rel[f"hi_{score}"] = 0.3
    rel[f"sig_{score}"] = 0.0
    for (store, day), value in (sig_by_cell or {}).items():
        hit = (rel["store"] == store) & (rel["lead_day"] == day)
        rel.loc[hit, f"sig_{score}"] = value
    rel.loc[rel["store"] == "ifs_ens", f"sig_{score}"] = np.nan
    return rel


def _sig_fig(sig_by_cell, relative=True):
    table = _table(low=())
    rel = _sig_rows(table, sig_by_cell=sig_by_cell)
    return figures.make_scorecard_fig(
        table, "temperature_2m", "rmse", relative, _ctx(relative_rows=rel)
    )


def test_significant_cells_get_a_dot_and_others_none():
    fig = _sig_fig({("gefs", 2): 1.0, ("gefs", 3): 0.0, ("gefs", 4): np.nan})
    (dots,) = [t for t in fig.data if t.name == "significant at 95 %"]
    gefs = GRID_STORES["gefs"]["label"]
    assert list(zip(dots.x, dots.y)) == [(2, gefs)]
    assert SIG_FOOTER in figures.figure_text(fig)


def test_significance_hover_distinguishes_not_assessed():
    fig = _sig_fig({("gefs", 2): 1.0, ("gefs", 3): 0.0, ("gefs", 4): np.nan})
    h = _heat(fig)
    row = list(h.y).index(GRID_STORES["gefs"]["label"])
    sig, no, nan = (h.customdata[row][d - 1][5] for d in (2, 3, 4))
    assert sig.startswith("significant at 95 %") and "[+0.1, +0.3] °C" in sig
    assert no.startswith("not significant") and "°C" in no
    assert "not assessed" in nan and "°C" not in nan
    assert "pairs both models share" in sig
    assert "%{customdata[5]}" in h.hovertemplate


def test_no_significance_columns_no_dots_no_footer():
    fig = figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", True, CTX)
    assert not [t for t in fig.data if t.name == "significant at 95 %"]
    assert "moving-block" not in figures.figure_text(fig)


def test_sig_frame_argument_marks_cells():
    table = _table(low=())
    sig = pd.DataFrame(
        {
            "variable": "temperature_2m",
            "store": ["gefs", "gefs"],
            "lead_day": [5, 6],
            "diff": [0.4, 0.1],
            "lo": [0.2, -0.1],
            "hi": [0.6, 0.3],
            "sig": [1.0, 0.0],
        }
    )
    fig = figures.make_scorecard_fig(
        table, "temperature_2m", "rmse", True, CTX, sig=sig
    )
    (dots,) = [t for t in fig.data if t.name == "significant at 95 %"]
    assert list(dots.x) == [5]


@pytest.mark.parametrize(
    "mode, phrase",
    [
        ("raw", "Mode: raw"),
        ("height_corrected", MODE_HC),
        ("representativeness", MODE_REP),
    ],
)
def test_scorecard_footer_names_the_mode(mode, phrase):
    table = _table().assign(mode=mode, mode_applied=True)
    text = figures.figure_text(
        figures.make_scorecard_fig(
            table, "temperature_2m", "rmse", False, CTX, mode=mode
        )
    )
    assert phrase in text
    if mode == "height_corrected":
        assert "no height correction" not in text


def test_mode_not_applied_says_shown_raw_never_claims_the_mode():
    table = _table("wind_speed_10m").assign(mode="height_corrected", mode_applied=False)
    text = figures.figure_text(
        figures.make_scorecard_fig(
            table, "wind_speed_10m", "rmse", False, CTX, mode="height_corrected"
        )
    )
    assert "RMSE shown raw in this mode" in text
    assert MODE_HC not in text


def test_contingency_score_is_shown_raw_under_representativeness():
    table = _table("precipitation_24h", "ets").assign(
        mode="representativeness", mode_applied=True
    )
    text = figures.figure_text(
        figures.make_scorecard_fig(
            table, "precipitation_24h", "ets", False, CTX, mode="representativeness"
        )
    )
    assert "ETS shown raw in this mode" in text


def test_persistence_skill_under_representativeness_notes_unperturbed():
    table = _table(score="crpss_persist").assign(
        mode="representativeness", mode_applied=True
    )
    text = figures.figure_text(
        figures.make_scorecard_fig(
            table,
            "temperature_2m",
            "crpss_persist",
            False,
            CTX,
            mode="representativeness",
        )
    )
    assert "persistence is not perturbed" in text


def test_precipitation_footer_names_the_station_window():
    table = _table("precipitation_24h", "mae", days=range(2, 16))
    text = figures.figure_text(
        figures.make_scorecard_fig(table, "precipitation_24h", "mae", False, CTX)
    )
    assert "24 h totals 06–06 UTC reported by the station" in text


def test_every_extra_score_has_a_label_and_direction():
    for var, spec in SCORECARD_VARIABLES.items():
        for score in spec["extra_scores"]:
            label, unit = figures._score_text(score, var)
            assert label and score in figures.SCORE_DIRECTION
    label, _ = figures._score_text("brier_5", "precipitation_24h")
    assert "5 mm" in label
    assert "10 mm" in figures._score_text("roc_area_10", "precipitation_24h")[0]


# --- diagnostic tables ---------------------------------------------------------


def _lead_diag(stores=("ifs_ens", "gefs", "gfs"), mode="raw", applied=True):
    rows = []
    for store in stores:
        det = GRID_STORES[store].get("kind") == "deterministic"
        for d in range(1, 11):
            rows.append(
                {
                    "kind": "lead",
                    "mode": mode,
                    "mode_applied": applied,
                    "period_days": 14,
                    "variable": "temperature_2m",
                    "store": store,
                    "lead_day": d,
                    "error_sd": 1.0 + 0.2 * d,
                    "spread_error": np.nan if det else 0.8,
                    "n": 500,
                    "n_inits": 14,
                    "low_n": False,
                }
            )
    return pd.DataFrame(rows)


def _diurnal(mode="raw"):
    rows = []
    for store in ("ifs_ens", "gefs"):
        for g in ("1-3", "4-7", "8-15"):
            for h in (0, 6, 12, 18):
                rows.append(
                    {
                        "kind": "diurnal",
                        "mode": mode,
                        "mode_applied": True,
                        "variable": "temperature_2m",
                        "store": store,
                        "lead_group": g,
                        "valid_hour": h,
                        "bias": 0.1 * h / 6 - 0.2,
                        "rmse": 1.5,
                        "error_sd": 1.4,
                        "n": 300,
                        "n_inits": 14,
                        "low_n": False,
                    }
                )
    return pd.DataFrame(rows)


def _rank(mode="raw", stores=("ifs_ens", "gefs", "aifs_ens")):
    rows = []
    for store in stores:
        for day in (1, 3):
            for b in range(10):
                count = 50.0 + (20 if b in (0, 9) else 0)
                rows.append(
                    {
                        "kind": "rank",
                        "mode": mode,
                        "mode_applied": True,
                        "variable": "temperature_2m",
                        "store": store,
                        "lead_day": day,
                        "bin": b,
                        "count": count,
                        "n": 580.0,
                        "freq": count / 580.0,
                        "n_inits": 14,
                    }
                )
    return pd.DataFrame(rows)


def _reliability(mode="raw", applied=True):
    rows = []
    for store in ("ifs_ens", "gefs"):
        for t in (1.0, 5.0):
            for g in ("1-3", "4-7"):
                for b in range(10):
                    rows.append(
                        {
                            "kind": "reliability",
                            "mode": mode,
                            "mode_applied": applied,
                            "variable": "precipitation_24h",
                            "store": store,
                            "threshold": t,
                            "lead_group": g,
                            "bin": b,
                            "bin_lo": b / 10,
                            "bin_hi": (b + 1) / 10,
                            "p_mean": b / 10 + 0.05,
                            "obs_freq": min(1.0, b / 10 + 0.1),
                            "count": 1000.0 / (b + 1),
                            "n_inits": 14,
                        }
                    )
    return pd.DataFrame(rows)


def _roc(mode="raw"):
    rows = []
    for store, area in (("ifs_ens", 0.83), ("gefs", 0.78)):
        for t in (1.0, 5.0):
            for g in ("1-3", "4-7"):
                for j in range(6):
                    rows.append(
                        {
                            "kind": "roc",
                            "mode": mode,
                            "mode_applied": True,
                            "variable": "precipitation_24h",
                            "store": store,
                            "threshold": t,
                            "lead_group": g,
                            "j": j,
                            "prob_threshold": j / 5,
                            "pod": 1 - (j / 5) ** 0.5,
                            "pofd": (1 - j / 5) ** 2,
                            "area": area,
                            "n": 400,
                        }
                    )
    return pd.DataFrame(rows)


DIAG = {
    "spread_error": lambda t, mode: figures.make_spread_error_fig(
        t, "temperature_2m", mode
    ),
    "diurnal": lambda t, mode: figures.make_diurnal_fig(
        t, "temperature_2m", "bias", mode
    ),
    "rank": lambda t, mode: figures.make_rank_histogram_fig(
        t, "temperature_2m", 1, mode
    ),
    "reliability": lambda t, mode: figures.make_reliability_fig(
        t, "precipitation_24h", 1.0, "1-3", mode
    ),
    "roc": lambda t, mode: figures.make_roc_fig(
        t, "precipitation_24h", 1.0, "1-3", mode
    ),
}
TABLES = {
    "spread_error": _lead_diag,
    "diurnal": _diurnal,
    "rank": _rank,
    "reliability": _reliability,
    "roc": _roc,
}


@pytest.mark.parametrize("kind", list(DIAG))
@pytest.mark.parametrize(
    "mode, phrase",
    [
        ("raw", "Mode: raw"),
        ("representativeness", MODE_REP),
    ],
)
def test_diagnostic_footer_names_the_mode_and_hover_has_units(kind, mode, phrase):
    fig = DIAG[kind](TABLES[kind](mode=mode), mode)
    assert phrase in figures.figure_text(fig)
    data = [t for t in fig.data if t.hovertemplate]
    assert data
    unit = {
        "spread_error": "°C",
        "diurnal": "°C",
        "rank": "% of ranks",
        "reliability": ":.0%}",
        "roc": ":.0%}",
    }[kind]
    assert all(unit in t.hovertemplate for t in data), kind


@pytest.mark.parametrize("kind", list(DIAG))
def test_diagnostic_empty_table_is_no_data(kind):
    fig = DIAG[kind](TABLES[kind]().iloc[0:0], "raw")
    assert not fig.data
    assert "No data available" in [a.text for a in fig.layout.annotations]


@pytest.mark.parametrize("kind", list(DIAG))
def test_diagnostic_ignores_other_modes(kind):
    fig = DIAG[kind](TABLES[kind](mode="representativeness"), "raw")
    assert not fig.data


def test_mode_not_applied_in_a_diagnostic_says_shown_raw():
    fig = figures.make_reliability_fig(
        _reliability(mode="height_corrected", applied=False),
        "precipitation_24h",
        1.0,
        "1-3",
        "height_corrected",
    )
    text = figures.figure_text(fig)
    assert "shown raw in this mode" in text and MODE_HC not in text


def test_spread_error_solid_error_dashed_spread_in_store_colour():
    fig = figures.make_spread_error_fig(_lead_diag(), "temperature_2m", "raw")
    ifs = GRID_STORES["ifs_ens"]["label"]
    err = [t for t in fig.data if t.meta == ("ifs_ens", "error_sd")]
    spr = [t for t in fig.data if t.meta == ("ifs_ens", "spread")]
    assert len(err) == 1 and len(spr) == 1
    assert err[0].line.dash in (None, "solid") and spr[0].line.dash == "dash"
    assert err[0].line.color == spr[0].line.color == figures._store_color("ifs_ens")
    assert np.allclose(spr[0].y, 0.8 * np.asarray(err[0].y))
    assert ifs in err[0].name
    # deterministic GFS: error SD only, no spread or ratio
    assert not [
        t for t in fig.data if t.meta and t.meta[0] == "gfs" and t.meta[1] != "error_sd"
    ]
    ratio = [t for t in fig.data if t.meta == ("ifs_ens", "ratio")]
    assert len(ratio) == 1 and ratio[0].yaxis == "y2"
    assert any(s.y0 == 1 and s.y1 == 1 for s in fig.layout.shapes)


def test_diurnal_one_line_per_store_per_panel_at_synoptic_hours():
    fig = figures.make_diurnal_fig(_diurnal(), "temperature_2m", "bias", "raw")
    lines = [t for t in fig.data if t.type == "scatter"]
    assert len(lines) == 2 * 3
    assert all(list(t.x) == [0, 6, 12, 18] for t in lines)
    assert sum(t.showlegend is not False for t in lines) == 2
    assert any(s.y0 == 0 for s in fig.layout.shapes)
    sd = figures.make_diurnal_fig(_diurnal(), "temperature_2m", "error_sd", "raw")
    assert "Error SD" in figures.figure_text(sd)


def test_rank_histogram_one_panel_per_store_flat_reference_and_caption():
    fig = figures.make_rank_histogram_fig(_rank(), "temperature_2m", 1, "raw")
    bars = [t for t in fig.data if t.type == "bar"]
    assert len(bars) == 3 and len({t.xaxis for t in bars}) == 3
    assert all(len(t.x) == 10 for t in bars)
    flat = [s for s in fig.layout.shapes if s.y0 == s.y1]
    assert len(flat) == 3 and all(np.isclose(s.y0, 10.0) for s in flat)  # 1/bins in %
    text = figures.figure_text(fig)
    assert "U-shape = under-dispersive, ∩ = over-dispersive, slope = bias" in text
    assert "day 1" in text.lower()


def test_reliability_has_diagonal_sized_points_and_sharpness_inset():
    fig = figures.make_reliability_fig(
        _reliability(), "precipitation_24h", 5.0, "1-3", "raw"
    )
    pts = [t for t in fig.data if t.meta and t.meta[1] == "reliability"]
    assert len(pts) == 2
    sizes = np.asarray(pts[0].marker.size, dtype=float)
    assert sizes[0] > sizes[-1]  # bin 0 has the most forecasts
    sharp = [t for t in fig.data if t.meta and t.meta[1] == "sharpness"]
    assert len(sharp) == 2 and sharp[0].xaxis == "x2"
    assert any(
        s.x0 == 0 and s.y0 == 0 and s.x1 == 1 and s.y1 == 1 for s in fig.layout.shapes
    )
    text = figures.figure_text(fig)
    assert "≥ 5 mm" in text and "24 h totals 06–06 UTC reported by the station" in text
    assert "count" in pts[0].hovertemplate


def test_roc_curve_per_store_with_area_in_legend_and_diagonal():
    fig = figures.make_roc_fig(_roc(), "precipitation_24h", 1.0, "1-3", "raw")
    curves = [t for t in fig.data if t.type == "scatter" and t.name]
    names = [t.name for t in curves]
    assert any(GRID_STORES["ifs_ens"]["label"] in n and "0.83" in n for n in names)
    assert any(GRID_STORES["gefs"]["label"] in n and "0.78" in n for n in names)
    assert any(
        s.x0 == 0 and s.y0 == 0 and s.x1 == 1 and s.y1 == 1 for s in fig.layout.shapes
    )
    assert "≥ 1 mm" in figures.figure_text(fig)


def test_lead_and_station_figures_name_their_mode():
    table = _table().assign(mode="height_corrected", mode_applied=True)
    lead = figures.make_score_lead_fig(
        table, "temperature_2m", "rmse", CTX, mode="height_corrected"
    )
    assert MODE_HC in figures.figure_text(lead)
    assert not figures.make_score_lead_fig(table, "temperature_2m", "rmse", CTX).data
    df = pd.DataFrame(
        {
            "variable": "temperature_2m",
            "store": "ifs_ens",
            "station": ["A"],
            "lat": [48.1],
            "lon": [11.5],
            "rmse": [1.0],
            "n": [100],
            "mode": "representativeness",
            "mode_applied": True,
        }
    )
    station = figures.make_station_score_map(
        df, "temperature_2m", "rmse", CTX, mode="representativeness"
    )
    assert MODE_REP in figures.figure_text(station)


# --- Task 6 fix round 1 ---------------------------------------------------------


def test_sig_rows_of_another_variable_never_mark_this_one():
    table = _table(low=())
    sig = pd.DataFrame(
        {
            "variable": ["wind_speed_10m", "temperature_2m"],
            "store": ["gefs", "gefs"],
            "lead_day": [5, 5],
            "diff": [0.4, 0.1],
            "lo": [0.2, -0.1],
            "hi": [0.6, 0.3],
            "sig": [1.0, 0.0],
        }
    )
    fig = figures.make_scorecard_fig(
        table, "temperature_2m", "rmse", True, CTX, sig=sig
    )
    (dots,) = [t for t in fig.data if t.name == "significant at 95 %"]
    assert len(dots.x) == 0
    h = _heat(fig)
    row = list(h.y).index(GRID_STORES["gefs"]["label"])
    assert h.customdata[row][4][5].startswith("not significant")


def test_long_footer_is_wrapped_and_keeps_every_clause():
    table = _rank(mode="representativeness")
    ctx = _ctx(period_days=14, height_corrected_excluded=3)
    fig = figures.make_rank_histogram_fig(
        table, "temperature_2m", 1, "representativeness", context=ctx
    )
    (note,) = [a for a in fig.layout.annotations if a.name == figures.PROVENANCE]
    lines = note.text.split("<br>")
    assert all(len(line) <= 110 for line in lines)
    flat = " ".join(lines)
    for clause in (
        MODE_REP,
        "temperature centred on the height-corrected value",
        "3 stations without elevation or orography excluded",
        "U-shape = under-dispersive, ∩ = over-dispersive, slope = bias",
    ):
        assert clause in flat, clause


def test_wrap_footer_never_drops_words():
    long = " · ".join(["alpha beta gamma delta"] * 12) + " · " + "x" * 30
    out = figures._wrap_footer([long, "short"])
    assert all(len(line) <= 110 for line in out)
    assert out[-1] == "short"
    assert (
        " ".join(out[:-1]).replace(" · ", " ").split()
        == long.replace(" · ", " ").split()
    )


def test_precipitation_footer_names_the_window_not_synoptic_hours():
    precip = figures.figure_text(
        figures.make_roc_fig(_roc(), "precipitation_24h", 1.0, "1-3", "raw")
    )
    assert "Every model scored on 24 h windows ending 06 UTC" in precip
    assert "00/06/12/18 UTC valid times" not in precip
    t2m = figures.figure_text(
        figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX)
    )
    assert "scored at 00/06/12/18 UTC valid times" in t2m


def test_reliability_axes_padded_and_sharpness_on_decades():
    fig = figures.make_reliability_fig(
        _reliability(), "precipitation_24h", 1.0, "1-3", "raw"
    )
    assert fig.layout.yaxis.range[1] > 1.0
    assert fig.layout.yaxis2.type == "log" and fig.layout.yaxis2.dtick == 1
    assert "%" in fig.layout.yaxis2.tickformat


# --- Final review fixes (Rulings V9, V10; findings 3–6) ----------------------------


def test_low_n_cells_never_get_a_significance_dot():
    table = _table(low=(("gefs", 2),))
    rel = _sig_rows(table, sig_by_cell={("gefs", 2): 1.0, ("gefs", 5): 1.0})
    fig = figures.make_scorecard_fig(
        table, "temperature_2m", "rmse", True, _ctx(relative_rows=rel)
    )
    (dots,) = [t for t in fig.data if t.name == "significant at 95 %"]
    assert list(dots.x) == [5]
    h = _heat(fig)
    row = list(h.y).index(GRID_STORES["gefs"]["label"])
    assert "not assessed" in h.customdata[row][1][5]
    text = figures.figure_text(fig)
    assert "assessed only with ≥ 10 common runs" in text


def test_spread_error_fig_omits_deterministic_spread_and_says_so():
    table = _lead_diag(mode="representativeness")
    table.loc[table["store"] == "gfs", "spread_error"] = 0.5  # an old file
    fig = figures.make_spread_error_fig(table, "temperature_2m", "representativeness")
    assert not [
        t for t in fig.data if t.meta and t.meta[0] == "gfs" and t.meta[1] != "error_sd"
    ]
    text = figures.figure_text(fig)
    assert "deterministic models: error SD only, no spread" in text


@pytest.mark.parametrize("score", ["ets", "hss", "frequency_bias", "pod", "far", "ets_5"])
def test_contingency_scores_say_they_use_the_ensemble_mean(score):
    table = _table(variable="precipitation_24h", score=score, low=())
    fig = figures.make_scorecard_fig(table, "precipitation_24h", score, False, CTX)
    assert "from the ensemble mean ≥ threshold" in figures.figure_text(fig)
    det = _table(variable="precipitation_24h", score=score, stores=("icon_eu",),
                 low=())
    fig = figures.make_scorecard_fig(det, "precipitation_24h", score, False, CTX)
    assert "ensemble mean" not in figures.figure_text(fig)


def test_precipitation_lead_axis_names_day_two_as_t30():
    table = _table(variable="precipitation_24h", score="mae", low=())
    fig = figures.make_scorecard_fig(table, "precipitation_24h", "mae", False, CTX)
    assert fig.layout.xaxis.title.text == (
        "Lead day (day 2 = T+30, 24 h ending 06 UTC)"
    )
    line = figures.make_score_lead_fig(table, "precipitation_24h", "mae", CTX)
    assert "day 2 = T+30" in line.layout.xaxis.title.text
    temp = figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX)
    assert temp.layout.xaxis.title.text == "Lead day (day 1 = 0–24 h)"


def test_footer_names_store_grids_and_orography_proxy():
    fig = figures.make_scorecard_fig(_table(), "temperature_2m", "rmse", False, CTX)
    text = figures.figure_text(fig)
    assert "each store's own grid (0.25°; DWD ICON-EU 0.0625°)" in text
    hc = figures.make_scorecard_fig(
        _table(), "temperature_2m", "rmse", False, CTX, mode="height_corrected"
    )
    assert "including ICON-EU's finer 0.0625° grid" in figures.figure_text(hc)
    rep = figures.make_scorecard_fig(
        _table(), "temperature_2m", "rmse", False, CTX, mode="representativeness"
    )
    assert "nearest-cell displacement is not in the perturbation model" in (
        figures.figure_text(rep)
    )


def test_cells_under_sig_min_inits_get_no_dot_even_from_an_old_file():
    table = _table(low=()).assign(n_inits=7)
    rel = _sig_rows(table, sig_by_cell={("gefs", 2): 1.0})
    fig = figures.make_scorecard_fig(
        table, "temperature_2m", "rmse", True, _ctx(relative_rows=rel)
    )
    (dots,) = [t for t in fig.data if t.name == "significant at 95 %"]
    assert len(dots.x) == 0
    h = _heat(fig)
    row = list(h.y).index(GRID_STORES["gefs"]["label"])
    assert "fewer than 10 common runs" in h.customdata[row][1][5]
