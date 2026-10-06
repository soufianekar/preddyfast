"""Weather · Evaluation: archived forecasts vs observations, scored offline.

The full workbench scores any GHCNh station on request. A public site cannot
afford that read (up to a few GB of point runs per new station), so here a
fixed set of stations is scored by the scheduled job (`utils/eval_snapshot.py`)
with the workbench's own `evaluate_point` and figure factories, and this page
only loads the finished figures. Each model is still scored on its own
members, per lead time, against the same observations; never combined.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html, no_update

from components.page_shell import page_shell
from components.shared_state import url_sync
from utils.eval_snapshot import load_figure, load_index, score_options
from utils.figures import CONFIG, make_empty_fig
from utils.settings import EVAL_REFERENCES, EVAL_RUNS_DEFAULT, EVAL_VARIABLES

dash.register_page(__name__, path="/evaluation", title="Forecast evaluation · preddyfast")

_HIDDEN = {"display": "none"}


def _graph(graph_id: str, message: str) -> dmc.Paper:
    return dmc.Paper(
        dcc.Graph(
            id=graph_id,
            figure=make_empty_fig(message),
            config=CONFIG,
            style={"height": "560px"},
        ),
        withBorder=True,
        radius="md",
        p="xs",
    )


def _station_options(index: dict) -> list[dict]:
    return [
        {"value": sid, "label": f"{s['name']} ({sid}) · {s['elev_m']:.0f} m"}
        for sid, s in index["stations"].items()
    ]


def _variable_options(station: dict | None) -> list[dict]:
    held = (station or {}).get("variables", {})
    return [{"value": k, "label": v["label"]} for k, v in EVAL_VARIABLES.items() if k in held]


def _reference_options(station: dict | None, variable: str | None) -> list[dict]:
    held = (station or {}).get("variables", {}).get(variable or "", {})
    return [{"value": k, "label": EVAL_REFERENCES[k]["label"]} for k in EVAL_REFERENCES if k in held]


def _pick(options: list[dict], wanted: str | None) -> str | None:
    values = [o["value"] for o in options]
    return wanted if wanted in values else (values[0] if values else None)


def layout(**query_params):
    index = load_index()
    stations = _station_options(index)
    station_id = _pick(stations, query_params.get("station"))
    station = index["stations"].get(station_id)
    variables = _variable_options(station)
    variable = _pick(variables, query_params.get("variable"))
    references = _reference_options(station, variable)
    reference = _pick(references, query_params.get("ref"))
    scores = score_options(variable)
    score = _pick(scores, query_params.get("score") or "crps")

    return page_shell(
        title="Forecast evaluation",
        description=(
            f"The last {EVAL_RUNS_DEFAULT} 00 UTC runs of archived ensembles scored "
            "against hourly station observations (NOAA GHCNh), satellite "
            "precipitation (IMERG) or a T+0 reference, per lead time. Scored "
            "offline by a scheduled job."
        ),
        sidebar=[
            html.Div(id="ev-current-url", style=_HIDDEN),
            dmc.Select(id="ev-station-select", label="Station", data=stations, value=station_id,
                       allowDeselect=False),
            dmc.Select(id="ev-variable-select", label="Variable", data=variables, value=variable,
                       allowDeselect=False),
            dmc.Select(id="ev-reference", label="Verified against", data=references, value=reference,
                       allowDeselect=False),
            dmc.Select(id="ev-score", label="Score", data=scores, value=score, allowDeselect=False),
            dmc.Text(id="ev-provenance", size="xs", c="dimmed"),
            dmc.Text(
                "The full workbench scores any station, model set and run count on "
                "request; this public demo shows a fixed set.",
                size="xs",
                c="dimmed",
            ),
        ],
        main=[
            html.Div(
                id="ev-alert-container",
                children=(
                    dmc.Alert("No evaluation has been published yet.", color="yellow",
                              withCloseButton=False)
                    if not stations
                    else None
                ),
            ),
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Scores by lead", value="scores"),
                            dmc.TabsTab("Newest runs", value="runs"),
                        ]
                    ),
                    dmc.TabsPanel(_graph("ev-scores-graph", "Choose a station."), value="scores"),
                    dmc.TabsPanel(_graph("ev-runs-graph", "Choose a station."), value="runs"),
                ],
                value="scores",
            ),
        ],
        graph_id="ev-scores-graph",
        share_target="ev-current-url",
    )


@callback(
    Output("ev-variable-select", "data"),
    Output("ev-variable-select", "value"),
    Input("ev-station-select", "value"),
    Input("ev-variable-select", "value"),
)
def variables_for_station(station_id, variable):
    """Offer only the variables drawn for the station."""
    options = _variable_options(load_index()["stations"].get(station_id))
    return options, _pick(options, variable)


@callback(
    Output("ev-reference", "data"),
    Output("ev-reference", "value"),
    Output("ev-score", "data"),
    Output("ev-score", "value"),
    Input("ev-station-select", "value"),
    Input("ev-variable-select", "value"),
    Input("ev-reference", "value"),
    Input("ev-score", "value"),
)
def options_for_variable(station_id, variable, reference, score):
    """References drawn for the variable, and the scores it offers (Brier for precipitation)."""
    station = load_index()["stations"].get(station_id)
    references = _reference_options(station, variable)
    scores = score_options(variable)
    return references, _pick(references, reference), scores, _pick(scores, score)


@callback(
    Output("ev-scores-graph", "figure"),
    Output("ev-runs-graph", "figure"),
    Output("ev-provenance", "children"),
    Output("ev-alert-container", "children", allow_duplicate=True),
    Input("ev-station-select", "value"),
    Input("ev-variable-select", "value"),
    Input("ev-reference", "value"),
    Input("ev-score", "value"),
    prevent_initial_call="initial_duplicate",
)
def show_evaluation(station_id, variable, reference, score):
    """Load the saved figures for the current choice."""
    index = load_index()
    station = index["stations"].get(station_id)
    entry = (station or {}).get("variables", {}).get(variable or "", {}).get(reference or "")
    if not station or not entry or not score:
        return no_update, no_update, "", no_update
    scores = load_figure(station_id, variable, reference, score)
    runs = load_figure(station_id, variable, reference, "runs")
    if scores is None or runs is None:
        empty = make_empty_fig("No data available")
        alert = dmc.Alert("This figure has not been published.", color="yellow", mb="xs")
        return empty, empty, "", alert
    alerts = [dmc.Alert(msg, color="yellow", mb="xs") for msg in entry.get("errors", [])]
    if entry.get("empty"):
        alerts.append(dmc.Alert("Nothing could be scored for this choice.", color="yellow", mb="xs"))
    provenance = (
        f"{' and '.join(station['stores'])} · {entry['runs']} runs to "
        f"{entry.get('newest_init') or '—'} · {station['lead_step']} h lead steps · "
        f"published {index.get('generated', '—')}"
    )
    return go.Figure(scores), go.Figure(runs), provenance, alerts


url_sync(
    "ev-current-url",
    {
        "ev-station-select": "station",
        "ev-variable-select": "variable",
        "ev-reference": "ref",
        "ev-score": "score",
    },
)
