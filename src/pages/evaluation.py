"""Layout + callbacks for Weather · Evaluation: archived forecasts vs station obs.

The user picks a place, one of the nearest GHCNh stations that has data,
a variable, one or more archived models and how many recent runs to score.
Each model is scored on its own members, per lead time, against the same
observations; the lines are compared, never combined.

Reads are bounded the same way as the area view: the run count and models
are checked against the read budget before anything is fetched, and a past
run's point extract is cached for good, so re-scoring re-reads nothing.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from components.location_search import location_search
from components.model_select import model_options
from components.page_shell import page_shell
from components.shared_state import url_sync
from utils import status
from utils.earthmover_api import GridDataError, read_estimate, store_variables
from utils.evaluation import evaluate_point, persist_scores, station_precip_window
from utils.figures import CONFIG, make_empty_fig, make_eval_runs_fig, make_scores_fig
from utils.observations import ObservationError, available_stations
from utils.settings import (
    EVAL_INIT_HOUR,
    EVAL_LEAD_STEP_DEFAULT,
    EVAL_LEAD_STEPS,
    EVAL_REFERENCES,
    EVAL_RUNS_DEFAULT,
    EVAL_RUNS_MAX,
    EVAL_VARIABLES,
    GRID_STORE_DEFAULT,
    GRID_STORES,
    GRID_VARIABLES,
    READ_BUDGET_MB,
)
from utils.url_state import query_choice, query_list

dash.register_page(__name__, path="/evaluation", title="Forecast evaluation · asifast")

_VARIABLE_OPTIONS = [{"value": k, "label": v["label"]} for k, v in EVAL_VARIABLES.items()]

_SCORE_OPTIONS = [
    {"value": "crps", "label": "CRPS"},
    {"value": "rmse", "label": "RMSE"},
    {"value": "mae", "label": "MAE"},
    {"value": "bias", "label": "Bias (mean error)"},
    {"value": "sd_error", "label": "SD of error"},
    {"value": "spread_skill", "label": "Spread / skill"},
]
_REFERENCE_OPTIONS = [
    {"value": k, "label": v["label"]} for k, v in EVAL_REFERENCES.items()
]
_STEP_OPTIONS = [{"value": str(h), "label": f"{h} h"} for h in EVAL_LEAD_STEPS]


def _score_options(variable: str | None) -> list[dict]:
    """Scores offered for a variable: the WMO set, plus Brier per threshold where defined."""
    thresholds = EVAL_VARIABLES.get(variable or "", {}).get("thresholds", ())
    return _SCORE_OPTIONS + [
        {"value": f"brier_{t:g}", "label": f"Brier ≥ {t:g} mm"} for t in thresholds
    ]


def _reference_options(variable: str | None) -> list[dict]:
    """T+0 references are refused for accumulated variables (T+0 precipitation is
    zero); stations for a field no station observes (Z500)."""
    spec = EVAL_VARIABLES.get(variable or "", {})
    accumulated = bool(spec.get("accum_hours"))
    unobserved = bool(spec) and spec.get("obs") is None

    def usable(key: str) -> bool:
        if key == "stations":
            return not unobserved
        if EVAL_REFERENCES[key].get("accumulated"):
            return accumulated  # IMERG: precipitation only
        return not accumulated  # a T+0 has no precipitation yet

    return [{**o, "disabled": not usable(o["value"])} for o in _REFERENCE_OPTIONS]


def _usable_reference(variable: str | None, reference: str | None) -> str:
    """The reference if it applies to the variable, else the first that does."""
    usable = [o["value"] for o in _reference_options(variable) if not o["disabled"]]
    return reference if reference in usable else usable[0]
_DEFAULT_VARIABLE = "temperature_2m"


def _station_label(row: dict) -> str:
    return f"{row['name']} · {row['dist_km']:.1f} km · {row['elev_m']:.0f} m"


def _graph(graph_id: str, message: str) -> dmc.Paper:
    return dmc.Paper(
        dmc.Box(
            [
                dmc.LoadingOverlay(id=f"{graph_id}-overlay", visible=False, zIndex=10),
                dcc.Graph(
                    id=graph_id,
                    figure=make_empty_fig(message),
                    config=CONFIG,
                    style={"height": "560px"},
                ),
            ],
            pos="relative",
        ),
        withBorder=True,
        radius="md",
        p="xs",
    )


def layout(**query_params):
    variable = query_choice(query_params, "variable", EVAL_VARIABLES, _DEFAULT_VARIABLE)
    models = query_list(query_params, "models", GRID_STORES, [GRID_STORE_DEFAULT])
    score = query_choice(query_params, "score", [o["value"] for o in _score_options(variable)], "crps")
    reference = query_choice(query_params, "ref", EVAL_REFERENCES, "stations")
    reference = _usable_reference(variable, reference)
    step = query_choice(query_params, "step", [str(h) for h in EVAL_LEAD_STEPS], str(EVAL_LEAD_STEP_DEFAULT))
    try:
        runs = min(EVAL_RUNS_MAX, max(5, int(query_params.get("runs", EVAL_RUNS_DEFAULT))))
    except ValueError:
        runs = EVAL_RUNS_DEFAULT

    return page_shell(
        title="Forecast evaluation",
        description=(
            "Recent archived runs scored against hourly station observations "
            "(NOAA GHCNh), per lead time. Each model is scored on its own members."
        ),
        sidebar=[
            html.Div(id="ev-current-url", style={"display": "none"}),
            "Where",
            location_search(),
            dcc.Store(id="ev-stations"),
            # Re-requests while runs are still to be read; each request reads
            # at most EVAL_COLD_READS_MAX chunks, inside the worker timeout.
            dcc.Interval(id="ev-more", interval=1500, disabled=True),
            dmc.Select(
                id="ev-station-select",
                label="Station",
                description="Nearest GHCNh stations with data this year (within 50 km).",
                data=[],
                value=query_params.get("station"),
                allowDeselect=False,
            ),
            "What",
            dmc.Select(
                id="ev-variable-select",
                label="Variable",
                data=_VARIABLE_OPTIONS,
                value=variable,
                allowDeselect=False,
            ),
            "Models",
            dmc.MultiSelect(
                id="ev-model-select",
                label="Models",
                description="Only runs every selected model has are scored.",
                data=model_options(),
                value=models,
                maxValues=3,
            ),
            "Score",
            dmc.NumberInput(
                id="ev-runs",
                label=f"Runs scored ({EVAL_INIT_HOUR:02d} UTC)",
                min=5,
                max=EVAL_RUNS_MAX,
                step=5,
                value=runs,
            ),
            dmc.Select(
                id="ev-reference",
                label="Verify against",
                description="Stations, or a model's T+0 at the station's grid cell.",
                data=_reference_options(variable),
                value=reference,
                allowDeselect=False,
            ),
            dmc.Stack(
                [
                    dmc.Text("Lead step", size="sm", fw=500),
                    dmc.SegmentedControl(
                        id="ev-lead-step",
                        data=_STEP_OPTIONS,
                        value=step,
                        fullWidth=True,
                        size="xs",
                    ),
                ],
                gap=4,
            ),
            dmc.Select(
                id="ev-score",
                label="Score",
                data=_score_options(variable),
                value=score,
                allowDeselect=False,
            ),
        ],
        main=[
            html.Div(id="ev-alert-container"),
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Scores by lead", value="scores"),
                            dmc.TabsTab("Newest runs", value="runs"),
                        ]
                    ),
                    dmc.TabsPanel(
                        _graph("ev-scores-graph", "Pick a location, then a station"),
                        value="scores",
                        pt="sm",
                    ),
                    dmc.TabsPanel(
                        _graph("ev-runs-graph", "Pick a location, then a station"),
                        value="runs",
                        pt="sm",
                    ),
                ],
                value="scores",
                keepMounted=True,
            ),
        ],
        graph_id="ev-scores-graph",
        share_target="ev-current-url",
    )


@callback(
    Output("ev-stations", "data"),
    Output("ev-station-select", "data"),
    Output("ev-station-select", "value"),
    Output("ev-alert-container", "children", allow_duplicate=True),
    Input("location-store", "data"),
    State("ev-station-select", "value"),
    prevent_initial_call="initial_duplicate",
)
def find_stations(location, current):
    """Nearest stations that have a file this year; keep the URL's if offered."""
    if not location:
        return no_update, no_update, no_update, no_update
    lat, lon = round(location["lat"], 2), round(location["lon"], 2)
    try:
        near = available_stations(lat, lon, pd.Timestamp.now(tz="UTC").year)
    except ObservationError as exc:
        return [], [], None, dmc.Alert(str(exc), color="red", mb="xs")
    if near.empty:
        alert = dmc.Alert(
            "No GHCNh station with data this year lies within 50 km of this place.",
            color="yellow",
            mb="xs",
        )
        return [], [], None, alert
    records = near.to_dict("records")
    options = [{"value": r["id"], "label": _station_label(r)} for r in records]
    ids = {r["id"] for r in records}
    return records, options, current if current in ids else records[0]["id"], []


@callback(
    Output("ev-runs", "description"),
    Input("ev-model-select", "value"),
    Input("ev-runs", "value"),
)
def runs_cost(models, runs):
    """What a cold read of this many runs costs, stated before it is asked for."""
    stores = tuple(m for m in models or [] if m in GRID_STORES)
    if not stores or not runs:
        return no_update
    try:
        # A point is one tile wherever it is, so any cell gives the cost.
        chunks, mb = read_estimate(stores, (45.0, 45.0, 0.0, 0.0), n_inits=int(runs))
    except GridDataError:
        return no_update
    batches = " in batches" if mb > READ_BUDGET_MB else ""
    return f"First read ~{mb:,.0f} MB ({chunks} chunks){batches}; cached after."


@callback(
    Output("ev-model-select", "data"),
    Input("ev-variable-select", "value"),
)
def models_for_variable(variable):
    """Grey out models that do not archive the variable; a store that will not open stays offered."""
    spec = EVAL_VARIABLES.get(variable)
    if not spec:
        return no_update
    lacking = set()
    for store in GRID_STORES:
        try:
            if spec["grid"] not in store_variables(store):
                lacking.add(store)
        except GridDataError:
            continue
    return model_options(frozenset(lacking))


@callback(
    Output("ev-score", "data"),
    Output("ev-score", "value"),
    Output("ev-reference", "data"),
    Output("ev-reference", "value"),
    Input("ev-variable-select", "value"),
    State("ev-score", "value"),
    State("ev-reference", "value"),
)
def options_for_variable(variable, score, reference):
    """Brier only where thresholds exist; T+0 references only for instantaneous fields."""
    scores = _score_options(variable)
    refs = _reference_options(variable)
    score_value = score if score in {o["value"] for o in scores} else "crps"
    ref_value = _usable_reference(variable, reference)
    return (
        scores,
        score_value if score_value != score else no_update,
        refs,
        ref_value if ref_value != reference else no_update,
    )


@callback(
    Output("ev-station-select", "data", allow_duplicate=True),
    Input("ev-variable-select", "value"),
    Input("ev-stations", "data"),
    prevent_initial_call=True,
)
def stations_for_variable(variable, stations):
    """For precipitation, say what each station can verify and list those first."""
    if not stations:
        return no_update
    hours = EVAL_VARIABLES.get(variable or "", {}).get("accum_hours")
    if not hours:
        return [{"value": r["id"], "label": _station_label(r)} for r in stations]
    rows = []
    for r in stations:
        try:
            best = station_precip_window(r["id"], hours)
        except ObservationError:
            best = None
        note = (
            f" · {hours} h ✓ "
            + (f"reported, ending {best['end_hour']:02d} UTC" if best["source"] == "reported" else "complete hours")
            if best
            else f" · no {hours} h windows"
        )
        rows.append((best is None, {"value": r["id"], "label": _station_label(r) + note}))
    return [option for _, option in sorted(rows, key=lambda t: t[0])]


# Overlay on at once, client-side; the server callback turns it off.
dash.clientside_callback(
    "function(s, v, m, r, ref, step) { return [Boolean(s), Boolean(s)]; }",
    Output("ev-scores-graph-overlay", "visible", allow_duplicate=True),
    Output("ev-runs-graph-overlay", "visible", allow_duplicate=True),
    Input("ev-station-select", "value"),
    Input("ev-variable-select", "value"),
    Input("ev-model-select", "value"),
    Input("ev-runs", "value"),
    Input("ev-reference", "value"),
    Input("ev-lead-step", "value"),
    prevent_initial_call=True,
)


@callback(
    Output("ev-scores-graph", "figure"),
    Output("ev-runs-graph", "figure"),
    Output("ev-alert-container", "children"),
    Output("ev-scores-graph-overlay", "visible"),
    Output("ev-runs-graph-overlay", "visible"),
    Output("ev-more", "disabled"),
    Input("ev-station-select", "value"),
    Input("ev-variable-select", "value"),
    Input("ev-model-select", "value"),
    Input("ev-runs", "value"),
    Input("ev-score", "value"),
    Input("ev-reference", "value"),
    Input("ev-lead-step", "value"),
    Input("ev-more", "n_intervals"),
    State("ev-stations", "data"),
)
def update_evaluation(station_id, variable, models, runs, score, reference, step, _more, stations):
    stores = tuple(m for m in models or [] if m in GRID_STORES)
    station = next((s for s in stations or [] if s["id"] == station_id), None)
    if not station or variable not in EVAL_VARIABLES or not stores or not runs:
        return no_update, no_update, no_update, False, False, True
    n_runs = min(EVAL_RUNS_MAX, max(5, int(runs)))

    reference = reference if reference in EVAL_REFERENCES else "stations"
    lead_step = int(step) if step else EVAL_LEAD_STEP_DEFAULT
    if score not in {o["value"] for o in _score_options(variable)}:
        return no_update, no_update, no_update, False, False, True  # options in transit
    try:
        with status.activity("Evaluation"):
            out = evaluate_point(
                station, variable, stores, n_runs, reference=reference, lead_step=lead_step
            )
            persist_scores(out, station, variable)
    except (GridDataError, ObservationError) as exc:
        empty = make_empty_fig("No data available")
        alert = dmc.Alert(str(exc), color="red", mb="xs")
        return empty, empty, alert, False, False, True

    alerts = [dmc.Alert(msg, color="yellow", mb="xs") for msg in out["errors"]]
    if out["pending"]:
        alerts.append(
            dmc.Alert(
                f"Scored {len(out['inits'])} of {len(out['inits']) + out['pending']} "
                "runs; reading the rest (each run is read once, then kept).",
                color="blue",
                mb="xs",
                withCloseButton=False,
            )
        )
    spec = out["spec"]
    grid_meta = GRID_VARIABLES[spec["grid"]]
    unit = grid_meta.get("accum_unit", grid_meta["unit"]) if spec["accum_hours"] else grid_meta["unit"]
    scores = {s: r["scores"] for s, r in out["results"].items()}
    if all(df.empty for df in scores.values()):
        if reference != "stations":
            why = f"no {EVAL_REFERENCES[reference]['label']} run is valid at a {lead_step} h lead yet"
        elif spec["accum_hours"]:
            why = (
                "this station does not report every hour of precipitation, so no "
                f"complete {spec['accum_hours']} h total exists"
            )
        else:
            why = "no observation lies within 15 min of any forecast step"
        alerts.append(dmc.Alert(f"Nothing to score: {why}.", color="yellow", mb="xs"))
    context = {
        "station": f"{station['name']} ({station['id']}) · {station['dist_km']:.1f} km · "
        f"{station['elev_m']:.0f} m",
        "grid_point": out["grid_point"],
        "grid_points": out["grid_points"],
        "inits": out["inits"],
        "obs_end": out["obs_end"],
        "precip_source": out.get("precip_source"),
        "reference": reference,
        "unit": unit,
    }
    scores_fig = make_scores_fig(scores, score, spec, context)
    runs_fig = make_eval_runs_fig(
        {s: r["matched"] for s, r in out["results"].items()}, out["obs"], spec, context
    )
    return scores_fig, runs_fig, alerts, False, False, not out["pending"]


url_sync(
    "ev-current-url",
    {
        "location-store": None,
        "ev-station-select": "station",
        "ev-variable-select": "variable",
        "ev-model-select": "models",
        "ev-runs": "runs",
        "ev-reference": "ref",
        "ev-lead-step": "step",
        "ev-score": "score",
    },
)
