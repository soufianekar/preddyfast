"""Weather · Evaluation: the station scorecard, scored offline.

Reads only the precomputed aggregates (`scorecard.load_aggregates`), written by
the scheduled job (`scripts/precompute.py --scorecard`) and mirrored from the
Dataset as `eval/scorecard/aggregates.parquet`; a request filters that table
and hands it to the figure factories. Nothing is aggregated or read per
request, so no visitor's choice reaches the archives.
"""

from __future__ import annotations

import inspect
import json

import dash
import dash_mantine_components as dmc
import pandas as pd
import pyarrow as pa
from dash import Input, Output, State, callback, dcc, html, no_update

from components.page_shell import page_shell
from components.shared_state import url_sync
from utils import scorecard
from utils.figures import (
    CONFIG,
    make_diurnal_fig,
    make_empty_fig,
    make_rank_histogram_fig,
    make_reliability_fig,
    make_roc_fig,
    make_score_lead_fig,
    make_scorecard_fig,
    make_spread_error_fig,
    make_station_score_map,
    score_label,
)
from utils.settings import (
    EVAL_REFERENCES,
    GRID_STORES,
    SCORE_DIRECTION,
    SCORECARD_BOOTSTRAP,
    SCORECARD_DAYS,
    SCORECARD_DETERMINISTIC_DRAWS,
    SCORECARD_INIT_HOUR,
    SCORECARD_LEAD_GROUPS,
    SCORECARD_LEAD_STEP_HOURS,
    SCORECARD_MIN_INITS,
    SCORECARD_MODES,
    SCORECARD_PERIODS,
    SCORECARD_SIG_MIN_INITS,
    SCORECARD_STATIONS,
    SCORECARD_STORES,
    SCORECARD_THRESHOLD_SCORES,
    SCORECARD_VARIABLES,
)
from utils.url_state import query_choice, query_list

dash.register_page(
    __name__, path="/evaluation", title="Forecast scorecard · preddyfast"
)

_BASELINE = "ifs_ens"
_DEFAULT_VARIABLE = "temperature_2m"
_DEFAULT_PERIOD = SCORECARD_DAYS
_PERIODS = {str(p) for p in SCORECARD_PERIODS}
_DEFAULT_LEAD = "3"
_LEADS = tuple(str(d) for d in scorecard.STATION_LEAD_DAYS)
_MISSING = "No scorecard has been published yet."
_DEFAULT_MODE = "raw"
_MODE_LABELS = {
    "raw": "Raw",
    "height_corrected": "Height-corrected",
    "representativeness": "Representativeness (TM865)",
}
_DIURNAL_SCORES = {"bias": "Bias", "error_sd": "Error SD"}
# The variable whose events the reliability / ROC panels draw, and its
# thresholds as keys.
_EVENT_SPEC = next(s for s in SCORECARD_VARIABLES.values() if s["thresholds"])
_THRESHOLDS = tuple(scorecard.threshold_key(t) for t in _EVENT_SPEC["thresholds"])
_GROUPS = tuple(f"{a}-{b}" for a, b in SCORECARD_LEAD_GROUPS)
# The pair count under which `aggregate` flags a cell low_n (its own default).
_MIN_N = inspect.signature(scorecard.aggregate).parameters["min_n"].default
_SCORE_NAMES = {
    "rmse": "RMSE",
    "mae": "MAE",
    "bias": "Bias (mean error)",
    "crps": "fair CRPS",
    "rmse_bc": "RMSE, bias-corrected",
    "mae_bc": "MAE, bias-corrected",
    "spread_skill": "Spread / skill",
    "ets": "ETS",
    "hss": "HSS",
    "frequency_bias": "Frequency bias",
    "pod": "POD",
    "far": "FAR",
}


def _score_label(variable: str, score: str) -> str:
    """Score name with its direction, e.g. "RMSE (lower is better)"."""
    spec = SCORECARD_VARIABLES[variable]
    if score == "brier":
        window = f"/{spec['window_hours']} h" if spec["window_hours"] else ""
        name = f"fair Brier ≥ {spec['threshold']:g} {spec['unit']}{window}"
    elif score in _SCORE_NAMES:
        name = _SCORE_NAMES[score]
    else:
        name = score_label(score, variable)
    ideal = SCORE_DIRECTION[score]
    hint = (
        "lower is better"
        if ideal == "lower"
        else "higher is better" if ideal == "higher" else f"ideal {ideal:g}"
    )
    return f"{name} ({hint})"


def _offered_scores(variable: str | None) -> tuple[str, ...]:
    """Every score the page offers for a variable: `scores` then `extra_scores`."""
    if variable not in SCORECARD_VARIABLES:
        return ()
    spec = SCORECARD_VARIABLES[variable]
    return (*spec["scores"], *spec["extra_scores"])


# Per-threshold bases that repeat a general score at the headline threshold.
_HEADLINE_TWINS = ("ets", "fbi", "pod", "far", "brier")


def _threshold_scores(variable: str, threshold: float) -> list[str]:
    """The variable's per-threshold scores at one threshold (`{base}_{t}`).

    At the headline threshold (`spec["threshold"]`) the bases the general
    scores already cover (ets, frequency bias, pod, far, brier) are left out.
    """
    spec = SCORECARD_VARIABLES[variable]
    offered = set(spec["extra_scores"])
    key = scorecard.threshold_key(threshold)
    headline = spec["threshold"] is not None and float(threshold) == float(
        spec["threshold"]
    )
    return [
        f"{s}_{key}"
        for s in SCORECARD_THRESHOLD_SCORES
        if f"{s}_{key}" in offered and not (headline and s in _HEADLINE_TWINS)
    ]


def _score_options(variable: str | None) -> list[dict]:
    """The variable's scores, labelled with their direction.

    A variable with event thresholds gets one group of its general scores and
    one group per threshold; the others a flat list.
    """
    if variable not in SCORECARD_VARIABLES:
        return []
    spec = SCORECARD_VARIABLES[variable]

    def option(s: str) -> dict:
        return {"value": s, "label": _score_label(variable, s)}

    if not spec["thresholds"]:
        return [option(s) for s in _offered_scores(variable)]
    per_threshold = {t: _threshold_scores(variable, t) for t in spec["thresholds"]}
    split = {
        f"{s}_{scorecard.threshold_key(t)}"
        for t in spec["thresholds"]
        for s in SCORECARD_THRESHOLD_SCORES
    }
    general = [s for s in _offered_scores(variable) if s not in split]
    headline = f"{spec['threshold']:g} {spec['unit']}"
    return [
        {
            "group": f"Amounts · headline events ≥ {headline}",
            "items": [option(s) for s in general],
        },
        *(
            {
                "group": f"≥ {t:g} {spec['unit']}",
                "items": [option(s) for s in scores],
            }
            for t, scores in per_threshold.items()
            if scores
        ),
    ]


def _flat_score_options(variable: str | None) -> list[dict]:
    """`_score_options` without its groups."""
    return [
        item
        for entry in _score_options(variable)
        for item in (entry["items"] if "group" in entry else [entry])
    ]


def _obs_mode_data(variable: str | None) -> list[dict]:
    """The Observations control: every mode, those the variable lacks disabled."""
    offered = (
        scorecard.obs_modes(variable)
        if variable in SCORECARD_VARIABLES
        else SCORECARD_MODES
    )
    return [
        {"value": m, "label": _MODE_LABELS[m], "disabled": m not in offered}
        for m in SCORECARD_MODES
    ]


def _variable_options(agg: pd.DataFrame | None) -> list[dict]:
    """Station-verified variables, plus any other the aggregates hold (Z500)."""
    present = set(agg["variable"]) if agg is not None and not agg.empty else set()
    return [
        {"value": key, "label": spec["label"]}
        for key, spec in SCORECARD_VARIABLES.items()
        if spec["obs"] is not None or key in present
    ]


def _store_options(meta: dict | None, only: list[str] | None = None) -> list[dict]:
    """Models the aggregates hold (all configured ones when there is no file)."""
    held = set((meta or {}).get("stores") or SCORECARD_STORES)
    if only is not None:
        held &= set(only)
    return [
        {"value": s, "label": GRID_STORES[s]["label"]}
        for s in SCORECARD_STORES
        if s in held
    ]


def _absent_note(meta: dict, variable: str, stores: list[str]) -> str | None:
    """Name the selected models the card leaves out because they lack the variable."""
    lacking = set((meta.get("lacking") or {}).get(variable, ()))
    absent = [GRID_STORES[s]["label"] for s in stores if s in lacking]
    if not absent:
        return None
    label = SCORECARD_VARIABLES[variable]["label"]
    holds = "holds" if len(absent) == 1 else "hold"
    return f"Not in this card: {', '.join(absent)} — {holds} no {label} in the archive."


def _select(
    agg: pd.DataFrame,
    kind: str,
    variable: str,
    period: int,
    stores: list[str],
    mode: str = _DEFAULT_MODE,
) -> pd.DataFrame:
    """Rows of one kind, variable, period and display mode for the chosen models."""
    keep = (
        (agg["kind"] == kind)
        & (agg["variable"] == variable)
        & (agg["period_days"] == period)
        & agg["store"].isin(stores)
        & (agg["mode"] == mode)
    )
    return agg.loc[keep]


def _context(
    meta: dict, rows: pd.DataFrame, days: int, absent_note: str | None
) -> dict:
    """Figure context shared by every view: the window and the mode metadata."""
    return {
        "n_stations": int(rows["n_stations"].max()) if len(rows) else 0,
        "period_days": days,
        "init_hour": SCORECARD_INIT_HOUR,
        "end": pd.Timestamp(meta["end"]),
        "baseline": _BASELINE,
        "absent_note": absent_note,
        "mode_raw_scores": meta.get("mode_raw_scores"),
        "height_corrected_excluded": meta.get("height_corrected_excluded"),
    }


def _lead_options(
    agg: pd.DataFrame | None, variable: str | None, period: str | None, stores: list
) -> list[str]:
    """Station-map lead days some selected model has rows for in the period.

    All of `_LEADS` when nothing is held (the map then says there is no data).
    """
    if agg is None or agg.empty or variable is None or period not in _PERIODS:
        return list(_LEADS)
    rows = _select(agg, "station", variable, int(period), stores)
    held = {str(int(d)) for d in rows["lead_day"].dropna()}
    return [d for d in _LEADS if d in held] or list(_LEADS)


def _lead_data(leads: list[str]) -> list[dict]:
    return [{"value": d, "label": f"Day {d}"} for d in leads]


def _load() -> tuple[tuple[pd.DataFrame, dict] | None, str | None]:
    """The aggregates and an error text; a damaged file is reported, not raised."""
    try:
        return scorecard.load_aggregates(), None
    except (pa.ArrowException, OSError, ValueError, json.JSONDecodeError) as err:
        return None, f"The published scorecard could not be read ({err})."


def _graph(graph_id: str, height: int = 620) -> dmc.Paper:
    return dmc.Paper(
        dcc.Graph(
            id=graph_id,
            figure=make_empty_fig("Loading scorecard"),
            config=CONFIG,
            style={"height": f"{height}px"},
        ),
        withBorder=True,
        radius="md",
        p="xs",
    )


def _labelled(label: str, control) -> dmc.Stack:
    return dmc.Stack([dmc.Text(label, size="sm", fw=500), control], gap=4)


def _segmented(control_id: str, data: list[dict], value: str) -> dmc.SegmentedControl:
    return dmc.SegmentedControl(
        id=control_id, data=data, value=value, fullWidth=True, size="xs"
    )


def _method_lines() -> list[str]:
    """The Method note, one item per method, worded from the settings in use."""
    precip = SCORECARD_VARIABLES["precipitation_24h"]
    hours = "/".join(
        f"{h:02d}"
        for h in range(
            SCORECARD_INIT_HOUR % SCORECARD_LEAD_STEP_HOURS,
            24,
            SCORECARD_LEAD_STEP_HOURS,
        )
    )
    spacing = ", ".join(
        f"{GRID_STORES[s]['label']} {GRID_STORES[s]['native_km']:g} km"
        for s in SCORECARD_STORES
    )
    res = [GRID_STORES[s]["res_deg"] for s in SCORECARD_STORES]
    common = max(set(res), key=res.count)
    grids = "; ".join(
        [
            f"{common:g}°",
            *(
                f"{GRID_STORES[s]['label']} {GRID_STORES[s]['res_deg']:g}°"
                for s in SCORECARD_STORES
                if GRID_STORES[s]["res_deg"] != common
            ),
        ]
    )
    finer = " and ".join(
        f"{GRID_STORES[s]['label'].split(' ', 1)[-1]}'s finer "
        f"{GRID_STORES[s]['res_deg']:g}° grid"
        for s in SCORECARD_STORES
        if GRID_STORES[s]["res_deg"] < 0.25
    )
    boot = SCORECARD_BOOTSTRAP
    # Variables no station observes: verified at the reference's valid times
    # only (`scorecard.reference_leads`: the archived inits' hour).
    references = [
        f"{spec['label']}: verified against "
        f"{EVAL_REFERENCES[spec['reference']]['label']} "
        f"({EVAL_REFERENCES[spec['reference']]['description']}) at "
        f"{SCORECARD_INIT_HOUR:02d} UTC valid times only, at each station's grid "
        "cell; the reference favours IFS and AIFS, which start from the same "
        "ECMWF analysis."
        for spec in SCORECARD_VARIABLES.values()
        if spec["reference"] != "stations"
    ]
    return [
        "Model values: the nearest grid cell in each store's own grid "
        f"({grids}), never interpolated.",
        f"Valid times against stations: {hours} UTC for "
        f"{SCORECARD_INIT_HOUR:02d} UTC runs, the "
        "coarsest step every model shares, so each lead day holds the same hours "
        f"of the diurnal cycle. {precip['label']}: {precip['window_hours']} h "
        f"totals ending {precip['window_end_hour']:02d} UTC as reported by the "
        "station (GHCNh), never hourly sums; a day without a report gives no pair.",
        *references,
        "Ensembles: fair CRPS and fair Brier score (Ferro 2014), so ensembles of "
        "different size compare; a deterministic model's CRPS is its absolute "
        "error.",
        "Observations · Raw (default): the observation as reported, the model "
        "value as it is.",
        "Observations · Height-corrected (2 m temperature only): t* = t + "
        "0.0065 K/m · (e_model − e_station) (ECMWF TM865 Eq. 2), with e_model from "
        "ERA5 orography as a proxy for each model's own; stations without an "
        "elevation or an orography cell are excluded and counted in the footer. "
        "ERA5 orography (0.25°) is used for every model"
        + (f", including {finer}." if finer else "."),
        "Observations · Representativeness: the perturbed-ensemble method of ECMWF "
        "Technical Memorandum 865 (Ben Bouallègue 2020) — members perturbed by the "
        "observation uncertainty, with ΔA the model's native grid spacing "
        f"({spacing}); a deterministic model is expanded to "
        f"{SCORECARD_DETERMINISTIC_DRAWS} perturbed draws for its CRPS, Brier "
        "score, rank histogram, reliability and ROC — but no spread for a "
        "deterministic model in any mode: the draws' variance is observation "
        "noise. ΔA is the native model spacing; nearest-cell displacement is not "
        "in the perturbation model. Contingency scores stay raw in every mode.",
        "Contingency scores (ETS, HSS, POD, FAR, frequency bias) use the ensemble "
        "mean ≥ threshold (a deterministic model's own value). The mean is "
        "smoothed, so they are biased high at low thresholds and low at high "
        "ones; the fair Brier score and ROC area are their probabilistic "
        "counterparts.",
        "Significance (relative view, dot = significant): paired circular "
        f"moving-block bootstrap over runs, blocks of {boot['block_days']} days, "
        f"{boot['resamples']} resamples, {boot['level'] * 100:g} % interval, "
        f"each pair weighted equally, against {GRID_STORES[_BASELINE]['label']} "
        "on the pairs both models share; assessed only with ≥ "
        f"{SCORECARD_SIG_MIN_INITS} common runs (fewer give too few blocks for "
        "the interval), and never marked on a cell with too few pairs or runs.",
        "Persistence: the observation 24·⌈lead/24⌉ h before the valid time; skill "
        "against it uses only pairs that have one.",
        f"Too few: a cell resting on fewer than {_MIN_N} pairs or "
        f"{SCORECARD_MIN_INITS} runs carries a cross and “(n<min)” in the hover.",
    ]


def _method_note() -> dmc.Accordion:
    return dmc.Accordion(
        dmc.AccordionItem(
            [
                dmc.AccordionControl("Method"),
                dmc.AccordionPanel(
                    dmc.List(
                        [dmc.ListItem(line) for line in _method_lines()],
                        size="sm",
                        spacing=4,
                    )
                ),
            ],
            value="method",
        ),
        id="sc-method",
        value=None,
        variant="contained",
        mt="sm",
    )


def _diagnostics_panel(
    diurnal_score: str, rank_lead: str, threshold: str, lead_group: str
) -> dmc.Stack:
    """The Diagnostics tab: spread–error, diurnal, rank, reliability + ROC."""
    return dmc.Stack(
        [
            html.Div(id="sc-diag-note"),
            _graph("sc-spread", 520),
            _labelled(
                "Diurnal cycle of",
                _segmented(
                    "sc-diurnal-score",
                    [{"value": k, "label": v} for k, v in _DIURNAL_SCORES.items()],
                    diurnal_score,
                ),
            ),
            _graph("sc-diurnal", 420),
            _labelled(
                "Rank histogram at",
                _segmented("sc-rank-lead", _lead_data(list(_LEADS)), rank_lead),
            ),
            _graph("sc-rank", 560),
            dmc.SimpleGrid(
                [
                    _labelled(
                        f"Event ({_EVENT_SPEC['label']})",
                        _segmented(
                            "sc-threshold",
                            [
                                {"value": t, "label": f"≥ {t} {_EVENT_SPEC['unit']}"}
                                for t in _THRESHOLDS
                            ],
                            threshold,
                        ),
                    ),
                    _labelled(
                        "Lead days",
                        _segmented(
                            "sc-lead-group",
                            [
                                {"value": g, "label": g.replace("-", "–")}
                                for g in _GROUPS
                            ],
                            lead_group,
                        ),
                    ),
                ],
                cols={"base": 1, "sm": 2},
            ),
            dmc.SimpleGrid(
                [_graph("sc-reliability", 480), _graph("sc-roc", 480)],
                cols={"base": 1, "md": 2},
            ),
        ],
        gap="sm",
    )


def layout(**query_params):
    loaded, load_error = _load()
    agg, meta = loaded if loaded else (None, None)
    variables = _variable_options(agg)
    variable = query_choice(
        query_params, "variable", [o["value"] for o in variables], _DEFAULT_VARIABLE
    )
    scores = [o["value"] for o in _flat_score_options(variable)]
    score = query_choice(query_params, "score", scores, scores[0])
    obs_mode = query_choice(
        query_params, "obs", scorecard.obs_modes(variable), _DEFAULT_MODE
    )
    diurnal_score = query_choice(query_params, "diurnal", _DIURNAL_SCORES, "bias")
    rank_lead = query_choice(query_params, "rank", _LEADS, _DEFAULT_LEAD)
    threshold = query_choice(query_params, "thr", _THRESHOLDS, _THRESHOLDS[0])
    lead_group = query_choice(query_params, "group", _GROUPS, _GROUPS[0])
    period = query_choice(
        query_params,
        "period",
        [str(p) for p in SCORECARD_PERIODS],
        str(_DEFAULT_PERIOD),
    )
    display = query_choice(
        query_params, "display", ("absolute", "relative"), "relative"
    )
    stores = _store_options(meta)
    models = query_list(
        query_params,
        "models",
        [o["value"] for o in stores],
        [o["value"] for o in stores],
    )
    leads = _lead_options(agg, variable, period, models)
    lead = query_choice(
        query_params,
        "lead",
        leads,
        _DEFAULT_LEAD if _DEFAULT_LEAD in leads else leads[0],
    )
    map_model = query_choice(
        query_params, "mapmodel", models, models[0] if models else None
    )
    base_label = GRID_STORES[_BASELINE]["label"]

    return page_shell(
        title="Forecast scorecard",
        description=(
            "Archived forecasts scored against station observations at "
            + (
                f"{len(SCORECARD_STATIONS)} major European city airports "
                if SCORECARD_STATIONS
                else "selected stations "
            )
            + "(NOAA GHCNh), precomputed over the latest "
            + f"{SCORECARD_DAYS} days of runs with observations. Models are compared, "
            + "never pooled."
        ),
        sidebar=[
            html.Div(id="sc-current-url", style={"display": "none"}),
            "What",
            dmc.Select(
                id="sc-variable",
                label="Variable",
                data=variables,
                value=variable,
                allowDeselect=False,
            ),
            dmc.Select(
                id="sc-score",
                label="Score",
                data=_score_options(variable),
                value=score,
                allowDeselect=False,
            ),
            dmc.Stack(
                [
                    dmc.Text("Observations", size="sm", fw=500),
                    dmc.SegmentedControl(
                        id="sc-obs-mode",
                        data=_obs_mode_data(variable),
                        value=obs_mode,
                        orientation="vertical",
                        fullWidth=True,
                        size="xs",
                    ),
                ],
                gap=4,
            ),
            "When",
            dmc.Stack(
                [
                    dmc.Text("Period (days to the newest run)", size="sm", fw=500),
                    dmc.SegmentedControl(
                        id="sc-period",
                        data=[
                            {"value": str(p), "label": f"{p} d"}
                            for p in SCORECARD_PERIODS
                        ],
                        value=period,
                        fullWidth=True,
                        size="xs",
                    ),
                ],
                gap=4,
            ),
            "Display",
            dmc.SegmentedControl(
                id="sc-display",
                data=[
                    {"value": "absolute", "label": "Absolute"},
                    {"value": "relative", "label": f"vs {base_label}"},
                ],
                value=display,
                fullWidth=True,
                size="xs",
            ),
            "Models",
            dmc.MultiSelect(
                id="sc-models",
                label="Models",
                data=stores,
                value=models,
                description=f"vs {base_label} always includes {base_label}.",
            ),
            dmc.Select(
                id="sc-map-model",
                label="Map model",
                data=_store_options(meta, models),
                value=map_model,
                allowDeselect=False,
                description="Station map: one model at a time.",
            ),
            dmc.Stack(
                [
                    dmc.Text("Lead day for the station map", size="sm", fw=500),
                    dmc.SegmentedControl(
                        id="sc-lead",
                        data=_lead_data(leads),
                        value=lead,
                        fullWidth=True,
                        size="xs",
                    ),
                ],
                gap=4,
            ),
            dmc.Text(
                "The full workbench scores any GHCNh station on request; this public "
                "demo shows a fixed network, scored offline by a scheduled job.",
                size="xs",
                c="dimmed",
            ),
        ],
        main=[
            html.Div(
                (
                    dmc.Alert(load_error or _MISSING, color="red", mb="xs")
                    if load_error or loaded is None
                    else []
                ),
                id="sc-alert-container",
            ),
            dmc.Text(id="sc-provenance", size="sm", c="dimmed", mb="xs"),
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Scorecard", value="card"),
                            dmc.TabsTab("By lead", value="lead"),
                            dmc.TabsTab("Stations", value="stations"),
                            dmc.TabsTab("Diagnostics", value="diagnostics"),
                        ]
                    ),
                    dmc.TabsPanel(_graph("sc-heat"), value="card", pt="sm"),
                    dmc.TabsPanel(_graph("sc-lines"), value="lead", pt="sm"),
                    dmc.TabsPanel(_graph("sc-map"), value="stations", pt="sm"),
                    dmc.TabsPanel(
                        _diagnostics_panel(
                            diurnal_score, rank_lead, threshold, lead_group
                        ),
                        value="diagnostics",
                        pt="sm",
                    ),
                ],
                value="card",
                keepMounted=True,
            ),
            _method_note(),
        ],
        graph_id="sc-heat",
        share_target="sc-current-url",
    )


@callback(
    Output("sc-score", "data"),
    Output("sc-score", "value"),
    Input("sc-variable", "value"),
    State("sc-score", "value"),
)
def options_for_variable(variable, score):
    """Offer the variable's scores; clear a score it does not have."""
    options = _score_options(variable)
    if not options:
        return no_update, no_update
    valid = [o["value"] for o in _flat_score_options(variable)]
    return options, score if score in valid else valid[0]


@callback(
    Output("sc-obs-mode", "data"),
    Output("sc-obs-mode", "value"),
    Input("sc-variable", "value"),
    State("sc-obs-mode", "value"),
)
def options_for_obs_mode(variable, current):
    """Disable the modes the variable lacks; a disabled choice falls back to raw."""
    if variable not in SCORECARD_VARIABLES:
        return no_update, no_update
    offered = scorecard.obs_modes(variable)
    return _obs_mode_data(variable), current if current in offered else _DEFAULT_MODE


@callback(
    Output("sc-map-model", "data"),
    Output("sc-map-model", "value"),
    Input("sc-models", "value"),
    State("sc-map-model", "value"),
)
def options_for_map_model(models, current):
    """Map model is one of the selected models; keep the current one if still
    selected."""
    chosen = [s for s in SCORECARD_STORES if s in (models or [])]
    options = _store_options(None, chosen)
    if not chosen:
        return options, None
    return options, current if current in chosen else chosen[0]


@callback(
    Output("sc-lead", "data"),
    Output("sc-lead", "value"),
    Input("sc-variable", "value"),
    Input("sc-period", "value"),
    Input("sc-models", "value"),
    State("sc-lead", "value"),
)
def options_for_lead(variable, period, models, current):
    """Offer only lead days the selected models have station rows for."""
    loaded, load_error = _load()
    if load_error or loaded is None:
        return no_update, no_update
    stores = [s for s in SCORECARD_STORES if s in (models or [])]
    leads = _lead_options(loaded[0], variable, period, stores)
    if current in leads:
        return _lead_data(leads), current
    return _lead_data(leads), _DEFAULT_LEAD if _DEFAULT_LEAD in leads else leads[0]


@callback(
    Output("sc-heat", "figure"),
    Output("sc-lines", "figure"),
    Output("sc-map", "figure"),
    Output("sc-alert-container", "children"),
    Output("sc-provenance", "children"),
    Input("sc-variable", "value"),
    Input("sc-score", "value"),
    Input("sc-period", "value"),
    Input("sc-display", "value"),
    Input("sc-models", "value"),
    Input("sc-lead", "value"),
    Input("sc-map-model", "value"),
    Input("sc-obs-mode", "value"),
)
def update_scorecard(
    variable, score, period, display, models, lead, map_model, obs_mode
):
    """Filter the precomputed aggregates and draw the three views."""
    if variable not in SCORECARD_VARIABLES or score not in _offered_scores(variable):
        return (no_update,) * 5
    if period not in _PERIODS or lead not in _LEADS or obs_mode not in SCORECARD_MODES:
        return (no_update,) * 5
    loaded, load_error = _load()
    if load_error:
        empty = make_empty_fig("Scorecard file unreadable")
        return empty, empty, empty, dmc.Alert(load_error, color="red", mb="xs"), ""
    if loaded is None:
        empty = make_empty_fig("No scorecard published yet")
        return empty, empty, empty, dmc.Alert(_MISSING, color="red", mb="xs"), ""
    agg, meta = loaded
    stores = [s for s in SCORECARD_STORES if s in (models or [])]
    provenance = (
        f"{meta.get('n_stations', '?')} stations · runs {meta.get('start', '?')} to "
        f"{meta.get('end', '?')} · computed {meta.get('generated', '?')}"
    )
    if not stores:
        empty = make_empty_fig("Select at least one model")
        return empty, empty, empty, [], provenance
    absent_note = _absent_note(meta, variable, stores)
    note = dmc.Alert(absent_note, color="yellow", mb="xs") if absent_note else []
    days = int(period)
    relative = display == "relative"
    mode = obs_mode
    try:
        lead_rows = _select(agg, "lead", variable, days, stores, mode)
        card_stores = (
            [*stores, _BASELINE] if relative and _BASELINE not in stores else stores
        )
        card_rows = _select(agg, "lead", variable, days, card_stores, mode)
        context = _context(meta, lead_rows, days, absent_note)
        heat_ctx = {
            **context,
            "relative_rows": _select(
                agg, "lead_relative", variable, days, card_stores, mode
            ),
        }
        heat = make_scorecard_fig(
            card_rows, variable, score, relative, heat_ctx, mode=mode
        )
        lines = make_score_lead_fig(lead_rows, variable, score, context, mode=mode)
        model = map_model if map_model in stores else stores[0]
        stations = _select(agg, "station", variable, days, [model], mode)
        stations = stations[stations["lead_day"] == int(lead)]
        station_ctx = {**context, "lead": f"day {lead}, {GRID_STORES[model]['label']}"}
        station_map = make_station_score_map(
            stations, variable, score, station_ctx, mode=mode
        )
    except (KeyError, ValueError) as err:
        empty = make_empty_fig("No data available")
        alert = dmc.Alert(
            f"The scorecard could not be drawn: {err}", color="red", mb="xs"
        )
        return empty, empty, empty, alert, provenance
    return heat, lines, station_map, note, provenance


@callback(
    Output("sc-spread", "figure"),
    Output("sc-diurnal", "figure"),
    Output("sc-rank", "figure"),
    Output("sc-reliability", "figure"),
    Output("sc-roc", "figure"),
    Output("sc-diag-note", "children"),
    Input("sc-variable", "value"),
    Input("sc-period", "value"),
    Input("sc-obs-mode", "value"),
    Input("sc-models", "value"),
    Input("sc-diurnal-score", "value"),
    Input("sc-rank-lead", "value"),
    Input("sc-threshold", "value"),
    Input("sc-lead-group", "value"),
)
def update_diagnostics(
    variable, period, obs_mode, models, diurnal_score, rank_lead, threshold, lead_group
):
    """Draw the four Diagnostics panels from the precomputed aggregates."""
    if (
        variable not in SCORECARD_VARIABLES
        or period not in _PERIODS
        or obs_mode not in SCORECARD_MODES
        or diurnal_score not in _DIURNAL_SCORES
        or rank_lead not in _LEADS
        or threshold not in _THRESHOLDS
        or lead_group not in _GROUPS
    ):
        return (no_update,) * 6
    loaded, load_error = _load()
    if load_error:
        empty = make_empty_fig("Scorecard file unreadable")
        return (empty,) * 5 + (dmc.Alert(load_error, color="red", mb="xs"),)
    if loaded is None:
        empty = make_empty_fig("No scorecard published yet")
        return (empty,) * 5 + (dmc.Alert(_MISSING, color="red", mb="xs"),)
    agg, meta = loaded
    stores = [s for s in SCORECARD_STORES if s in (models or [])]
    if not stores:
        return (make_empty_fig("Select at least one model"),) * 5 + ([],)
    mode = obs_mode
    days = int(period)
    notes = scorecard.diagnostic_notes(variable, stores, mode)
    absent_note = _absent_note(meta, variable, stores)

    def rows(kind: str) -> pd.DataFrame:
        return _select(agg, kind, variable, days, stores, mode)

    try:
        context = _context(meta, rows("lead"), days, absent_note)
        spread = (
            make_empty_fig(notes["spread_error"])
            if notes["spread_error"]
            else make_spread_error_fig(rows("lead"), variable, mode, context=context)
        )
        diurnal = (
            make_empty_fig(notes["diurnal"])
            if notes["diurnal"]
            else make_diurnal_fig(
                rows("diurnal"), variable, diurnal_score, mode, context=context
            )
        )
        rank = make_rank_histogram_fig(
            rows("rank"), variable, int(rank_lead), mode, context=context
        )
        if notes["reliability"]:
            reliability = roc = make_empty_fig(notes["reliability"])
        else:
            reliability = make_reliability_fig(
                rows("reliability"),
                variable,
                float(threshold),
                lead_group,
                mode,
                context=context,
            )
            roc = make_roc_fig(
                rows("roc"),
                variable,
                float(threshold),
                lead_group,
                mode,
                context=context,
            )
    except (KeyError, ValueError) as err:
        empty = make_empty_fig("No data available")
        alert = dmc.Alert(
            f"The diagnostics could not be drawn: {err}", color="red", mb="xs"
        )
        return (empty,) * 5 + (alert,)
    shown = [n for n in (absent_note, notes["deterministic"]) if n]
    note = [dmc.Alert(n, color="yellow", mb="xs") for n in shown]
    return spread, diurnal, rank, reliability, roc, note


# Control id -> query key, in the order written to the URL.
URL_KEYS: dict[str, str] = {
    "sc-variable": "variable",
    "sc-score": "score",
    "sc-obs-mode": "obs",
    "sc-period": "period",
    "sc-display": "display",
    "sc-models": "models",
    "sc-lead": "lead",
    "sc-map-model": "mapmodel",
    "sc-diurnal-score": "diurnal",
    "sc-rank-lead": "rank",
    "sc-threshold": "thr",
    "sc-lead-group": "group",
}
url_sync("sc-current-url", URL_KEYS)
