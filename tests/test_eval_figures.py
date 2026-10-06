import numpy as np
import pandas as pd

from utils.figures import figure_text, make_eval_runs_fig, make_scores_fig
from utils.settings import EVAL_VARIABLES, GRID_STORES

SCORES = {
    "ifs_ens": pd.DataFrame({"lead_hours": [24.0, 48.0], "n": [10, 8], "bias": [0.1, 0.3], "mae": [1, 1.2],
                             "rmse": [1.1, 1.5], "crps": [0.6, 0.8], "spread": [0.9, 1.2], "spread_skill": [0.8, 0.8]}),
    "aifs_ens": pd.DataFrame({"lead_hours": [24.0], "n": [10], "bias": [-0.2], "mae": [0.9],
                              "rmse": [1.0], "crps": [0.5], "spread": [0.8], "spread_skill": [0.8]}),
}
CONTEXT = {"station": "PARIS-MONTSOURIS (FRM00007156) · 3.9 km · 77 m", "grid_point": (48.75, 2.25),
           "inits": [pd.Timestamp("2026-09-30"), pd.Timestamp("2026-09-21")], "unit": "°C"}


def test_one_line_per_model_named_by_model():
    fig = make_scores_fig(SCORES, "crps", EVAL_VARIABLES["temperature_2m"], CONTEXT)
    names = [t.name for t in fig.data]
    assert names == [GRID_STORES["ifs_ens"]["label"], GRID_STORES["aifs_ens"]["label"]]


def test_hover_states_run_count_and_unit():
    fig = make_scores_fig(SCORES, "rmse", EVAL_VARIABLES["temperature_2m"], CONTEXT)
    assert "n=" in fig.data[0].hovertemplate and "°C" in fig.data[0].hovertemplate


def test_subtitle_carries_provenance():
    title = figure_text(make_scores_fig(SCORES, "bias", EVAL_VARIABLES["temperature_2m"], CONTEXT))
    for part in ("PARIS-MONTSOURIS", "48.75", "GHCNh", "2026-09-21", "2026-09-30"):
        assert part in title


def test_runs_fig_draws_obs_and_each_model():
    times = pd.date_range("2026-09-30 03:00", periods=3, freq="3h")
    matched = {s: pd.DataFrame([{"init_time": pd.Timestamp("2026-09-30"), "valid_time": t, "lead_hours": 3.0 * (i + 1),
                                  "member": m, "value": 10.0 + m, "obs": 11.0} for i, t in enumerate(times) for m in range(3)])
               for s in ("ifs_ens", "aifs_ens")}
    obs = pd.DataFrame({"valid_time": times.tz_localize("UTC"), "value": [11.0, 11.5, 12.0]})
    fig = make_eval_runs_fig(matched, obs, EVAL_VARIABLES["temperature_2m"], CONTEXT)
    names = " ".join(t.name for t in fig.data)
    assert "Observed" in names and "ECMWF IFS ENS" in names and "ECMWF AIFS ENS" in names


def test_spread_skill_not_drawn_for_a_deterministic_model():
    scores = {**SCORES, "gfs": SCORES["aifs_ens"]}
    fig = make_scores_fig(scores, "spread_skill", EVAL_VARIABLES["temperature_2m"], CONTEXT)
    assert all(GRID_STORES["gfs"]["label"] not in t.name for t in fig.data)


def test_subtitle_names_each_models_grid_cell():
    ctx = {**CONTEXT, "grid_points": {"ifs_ens": (48.75, 2.25), "icon_eu": (48.8125, 2.3125)}}
    title = figure_text(make_scores_fig(SCORES, "crps", EVAL_VARIABLES["temperature_2m"], ctx))
    assert "48.81°N" in title and "0.0625°" in title and "0.25°" in title


def test_newest_runs_tab_draws_verified_runs_and_observed_colour():
    from utils.figures import make_eval_runs_fig
    from utils.settings import OBSERVED_COLOR

    rows = []
    for day in range(1, 6):
        init = pd.Timestamp(f"2026-09-{day:02d}")
        for lead in (24.0, 48.0):
            for m in range(3):
                # Runs from 4 Sep on verify after the last report (obs NaN).
                rows.append({"init_time": init, "valid_time": init + pd.Timedelta(hours=lead),
                             "lead_hours": lead, "member": m, "value": 1.0 + m,
                             "obs": 1.0 if day <= 3 else np.nan})
    matched = {"ifs_ens": pd.DataFrame(rows)}
    obs = pd.DataFrame({"valid_time": pd.date_range("2026-09-02", periods=3, freq="D", tz="UTC"), "value": 1.0})
    fig = make_eval_runs_fig(matched, obs, EVAL_VARIABLES["temperature_2m"], CONTEXT)
    runs = {t.hovertemplate.split(" run ")[1][:10] for t in fig.data if t.hovertemplate and " run " in t.hovertemplate}
    assert runs == {"2026-09-01", "2026-09-02", "2026-09-03"}
    observed = next(t for t in fig.data if t.name == "Observed")
    assert observed.line.color == OBSERVED_COLOR


def test_new_scores_have_labels_and_brier_is_unitless():
    scores = {k: v.assign(sd_error=0.5, brier_1=0.1) for k, v in SCORES.items()}
    sd = make_scores_fig(scores, "sd_error", EVAL_VARIABLES["temperature_2m"], CONTEXT)
    assert "SD of error" in sd.layout.yaxis.title.text and "°C" in sd.layout.yaxis.title.text
    brier = make_scores_fig(scores, "brier_1", EVAL_VARIABLES["precipitation_24h"], {**CONTEXT, "unit": "mm"})
    assert "Brier" in brier.layout.yaxis.title.text and "1 mm" in brier.layout.yaxis.title.text
    assert "(mm)" not in brier.layout.yaxis.title.text


def test_gridded_reference_is_named_in_footer_and_line():
    ctx = {**CONTEXT, "reference": "control_t0"}
    text = figure_text(make_scores_fig(SCORES, "rmse", EVAL_VARIABLES["temperature_2m"], ctx))
    assert "IFS control T+0" in text and "GHCNh" not in text
