"""Weather · Evaluation: the station scorecard, scored offline.

Reads only the precomputed aggregates (`scorecard.load_aggregates`), written by
the scheduled job (`scripts/precompute.py --scorecard`) and mirrored from the
Dataset as `eval/scorecard/aggregates.parquet`; a request filters that table
and hands it to the figure factories. Nothing is aggregated or read per
request, so no visitor's choice reaches the archives.
"""

from __future__ import annotations

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
    make_empty_fig,
    make_score_lead_fig,
    make_scorecard_fig,
    make_station_score_map,
)
from utils.settings import (
    GRID_STORES,
    SCORE_DIRECTION,
    SCORECARD_DAYS,
    SCORECARD_INIT_HOUR,
    SCORECARD_PERIODS,
    SCORECARD_STATIONS,
    SCORECARD_STORES,
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
    else:
        name = _SCORE_NAMES[score]
    ideal = SCORE_DIRECTION[score]
    hint = (
        "lower is better"
        if ideal == "lower"
        else "higher is better" if ideal == "higher" else f"ideal {ideal:g}"
    )
    return f"{name} ({hint})"


def _score_options(variable: str | None) -> list[dict]:
    """The variable's scores, labelled with their direction."""
    if variable not in SCORECARD_VARIABLES:
        return []
    return [
        {"value": s, "label": _score_label(variable, s)}
        for s in SCORECARD_VARIABLES[variable]["scores"]
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
    agg: pd.DataFrame, kind: str, variable: str, period: int, stores: list[str]
) -> pd.DataFrame:
    """Rows of one kind, variable and period for the chosen models (raw mode)."""
    keep = (
        (agg["kind"] == kind)
        & (agg["variable"] == variable)
        & (agg["period_days"] == period)
        & agg["store"].isin(stores)
        & (agg["mode"] == "raw")
    )
    return agg.loc[keep]


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


def _graph(graph_id: str) -> dmc.Paper:
    return dmc.Paper(
        dcc.Graph(
            id=graph_id,
            figure=make_empty_fig("Loading scorecard"),
            config=CONFIG,
            style={"height": "620px"},
        ),
        withBorder=True,
        radius="md",
        p="xs",
    )


def layout(**query_params):
    loaded, load_error = _load()
    agg, meta = loaded if loaded else (None, None)
    variables = _variable_options(agg)
    variable = query_choice(
        query_params, "variable", [o["value"] for o in variables], _DEFAULT_VARIABLE
    )
    score = query_choice(
        query_params,
        "score",
        [o["value"] for o in _score_options(variable)],
        _score_options(variable)[0]["value"],
    )
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
                        ]
                    ),
                    dmc.TabsPanel(_graph("sc-heat"), value="card", pt="sm"),
                    dmc.TabsPanel(_graph("sc-lines"), value="lead", pt="sm"),
                    dmc.TabsPanel(_graph("sc-map"), value="stations", pt="sm"),
                ],
                value="card",
                keepMounted=True,
            ),
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
    valid = [o["value"] for o in options]
    return options, score if score in valid else valid[0]


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
)
def update_scorecard(variable, score, period, display, models, lead, map_model):
    """Filter the precomputed aggregates and draw the three views."""
    if (
        variable not in SCORECARD_VARIABLES
        or score not in SCORECARD_VARIABLES[variable]["scores"]
    ):
        return (no_update,) * 5
    if period not in _PERIODS or lead not in _LEADS:
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
    try:
        lead_rows = _select(agg, "lead", variable, days, stores)
        card_stores = (
            [*stores, _BASELINE] if relative and _BASELINE not in stores else stores
        )
        card_rows = _select(agg, "lead", variable, days, card_stores)
        context = {
            "n_stations": int(lead_rows["n_stations"].max()) if len(lead_rows) else 0,
            "period_days": days,
            "init_hour": SCORECARD_INIT_HOUR,
            "end": pd.Timestamp(meta["end"]),
            "baseline": _BASELINE,
            "absent_note": absent_note,
        }
        heat_ctx = {
            **context,
            "relative_rows": _select(agg, "lead_relative", variable, days, card_stores),
        }
        heat = make_scorecard_fig(card_rows, variable, score, relative, heat_ctx)
        lines = make_score_lead_fig(lead_rows, variable, score, context)
        model = map_model if map_model in stores else stores[0]
        stations = _select(agg, "station", variable, days, [model])
        stations = stations[stations["lead_day"] == int(lead)]
        station_ctx = {**context, "lead": f"day {lead}, {GRID_STORES[model]['label']}"}
        station_map = make_station_score_map(stations, variable, score, station_ctx)
    except (KeyError, ValueError) as err:
        empty = make_empty_fig("No data available")
        alert = dmc.Alert(
            f"The scorecard could not be drawn: {err}", color="red", mb="xs"
        )
        return empty, empty, empty, alert, provenance
    return heat, lines, station_map, note, provenance


url_sync(
    "sc-current-url",
    {
        "sc-variable": "variable",
        "sc-score": "score",
        "sc-period": "period",
        "sc-display": "display",
        "sc-models": "models",
        "sc-lead": "lead",
        "sc-map-model": "mapmodel",
    },
)
