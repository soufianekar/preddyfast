"""Layout + all callbacks for the ensemble meteogram page."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from components.location_search import location_search
from components.model_select import model_options
from components.page_shell import page_shell
from components.run_select import LATEST, register_run_options, run_select
from components.shared_state import publish_variable, url_sync
from utils import status
from utils.climatology import validate_ensemble_selection
from utils.earthmover_api import (
    GridDataError,
    archive_meteogram,
    archive_systems,
    run_in_calendar,
    store_horizon,
    stores_calendar,
)
from utils.figures import CONFIG, make_empty_fig, make_meteogram_fig
from utils.openmeteo_api import (
    fetch_deterministic,
    fetch_deterministic_accumulated,
    fetch_deterministic_synoptic,
    fetch_ensemble,
    fetch_ensemble_accumulated,
    fetch_ensemble_synoptic,
)
from utils.settings import DETERMINISTIC_MODELS, ENSEMBLE_MODELS, GRID_STORES, VARIABLES
from utils.text import in_sentence
from utils.url_state import query_choice, query_list

dash.register_page(__name__, path="/meteogram", title="Forecast at a location · asifast")

_VARIABLE_OPTIONS = [{"value": k, "label": v["label"]} for k, v in VARIABLES.items()]
_ALL_SYSTEMS = {**ENSEMBLE_MODELS, **DETERMINISTIC_MODELS}
_SYSTEM_OPTIONS = [
    {"group": "Ensembles", "items": [{"value": k, "label": v["label"]} for k, v in ENSEMBLE_MODELS.items()]},
    {
        "group": "Deterministic",
        "items": [{"value": k, "label": v["label"]} for k, v in DETERMINISTIC_MODELS.items()],
    },
]
_FORECAST_DAY_OPTIONS = [{"value": str(d), "label": f"{d} days"} for d in (3, 5, 7, 10, 14, 16)]
# Past runs reach further with GEFS (35 d).
_PAST_DAY_CHOICES = (3, 5, 7, 10, 14, 16, 21, 28, 35)
_MAX_MODELS = 3

# ECMWF ENS (its control drawn as the bold line), AI ENS, and the two
# deterministic ECMWF runs.
_DEFAULT_SYSTEMS = ["ecmwf_ifs025", "ecmwf_aifs025", "aifs_single", "ecmwf_ifs_hres"]
_DEFAULT_VARIABLE = "temperature_2m_mean"

_LATEST_LABEL = "Latest (all systems)"
_SYSTEMS_LATEST = "Each model's newest run, from Open-Meteo; deterministic models draw one dashed line."
_SYSTEMS_PAST = "Archived models, up to three; only runs they all hold are offered."

_HIDDEN = {"display": "none"}
_SHOWN = {"display": "block"}


def _default_models(systems: list[str]) -> list[str]:
    """The archived stores behind the selected systems, else IFS ENS."""
    return [_ALL_SYSTEMS[s]["archive_store"] for s in archive_systems(systems)]


def _past_day_options(models: list[str] | None) -> list[dict]:
    """Horizons up to the furthest-reaching selected model."""
    stores = [m for m in models or [] if m in GRID_STORES]
    try:
        reach = max((store_horizon(m) for m in stores), default=360.0) / 24.0
    except GridDataError:  # a store that will not open: offer the common 15 days
        reach = 15.0
    return [{"value": str(d), "label": f"{d} days"} for d in _PAST_DAY_CHOICES if d <= reach]


def _accum_options(variable: str | None) -> tuple[list[dict], str | None]:
    """Accumulation-period options for a variable, empty if it is not accumulated."""
    meta = VARIABLES.get(variable or "", {})
    if not meta.get("is_accumulated"):
        return [], None
    return (
        [{"value": str(h), "label": f"{h} h"} for h in meta["accum_periods"]],
        str(meta["accum_default"]),
    )


def _accumulated_meta(var_meta: dict, accum_hours: int | None) -> dict:
    """Variable metadata re-labelled for the chosen window.

    A copy, never a mutation: `VARIABLES` is module-level shared state and
    editing it here would leak the window into every other page.
    """
    if not accum_hours:
        return var_meta
    return {
        **var_meta,
        "unit": var_meta.get("accum_unit", var_meta["unit"]),
        "label": f"{var_meta['label']} ({accum_hours} h total)",
    }


def layout(**query_params):
    variable = query_choice(query_params, "variable", VARIABLES, _DEFAULT_VARIABLE)
    systems = query_list(query_params, "systems", _ALL_SYSTEMS, _DEFAULT_SYSTEMS)
    models = query_list(query_params, "models", GRID_STORES, _default_models(systems))[
        :_MAX_MODELS
    ]
    days = query_choice(query_params, "days", [str(d) for d in _PAST_DAY_CHOICES], "7")
    accum_options, accum_default = _accum_options(variable)
    accum = query_choice(
        query_params, "accum", [o["value"] for o in accum_options], accum_default
    )
    # Any ISO run is accepted here; the select's options arrive on mount, and
    # an unknown run is reported by the fetch rather than silently replaced.
    run = query_params.get("run") or LATEST

    return page_shell(
        title="Forecast at a location",
        description=(
            "Probabilistic short-range outlook. Overlaid systems are never "
            "averaged together — spread is shown within each system."
        ),
        sidebar=[
            html.Div(id="mg-current-url", style={"display": "none"}),
            "Where",
            location_search(),
            "When",
            run_select(
                "mg-run-select",
                _LATEST_LABEL,
                "Latest: Open-Meteo's systems. Past run: any archived model, at "
                "its nearest grid cell.",
                run,
            ),
            "What",
            dmc.Select(
                id="mg-variable-select",
                label="Variable",
                data=_VARIABLE_OPTIONS,
                value=variable,
            ),
            "Models",
            html.Div(
                dmc.MultiSelect(
                    id="mg-systems-select",
                    label="Models",
                    description=_SYSTEMS_LATEST,
                    data=_SYSTEM_OPTIONS,
                    value=systems,
                ),
                id="mg-systems-wrap",
                style=_SHOWN if run == LATEST else _HIDDEN,
            ),
            html.Div(
                dmc.MultiSelect(
                    id="mg-models-select",
                    label="Models",
                    description=_SYSTEMS_PAST,
                    data=model_options(),
                    value=models,
                    maxValues=_MAX_MODELS,
                ),
                id="mg-models-wrap",
                style=_HIDDEN if run == LATEST else _SHOWN,
            ),
            dcc.Store(id="mg-archive-stores", data=models if run != LATEST else None),
            "Range",
            dmc.Select(
                id="mg-days-select",
                label="Forecast horizon",
                data=_FORECAST_DAY_OPTIONS if run == LATEST else _past_day_options(models),
                value=days,
            ),
            html.Div(
                dmc.Select(
                    id="mg-accum-select",
                    label="Accumulation period",
                    description="Non-overlapping windows, from the hourly ensemble.",
                    data=accum_options,
                    value=accum,
                ),
                id="mg-accum-wrap",
                style=_SHOWN if accum_options else _HIDDEN,
            ),
        ],
        main=[
            html.Div(id="mg-alert-container"),
            dmc.Paper(
                dcc.Loading(
                    dcc.Graph(
                        id="meteogram-graph",
                        figure=make_empty_fig("Pick a location on the map"),
                        config=CONFIG,
                        style={"height": "620px"},
                    ),
                    type="dot",
                ),
                withBorder=True,
                radius="md",
                p="xs",
            ),
        ],
        graph_id="meteogram-graph",
        share_target="mg-current-url",
    )


@callback(
    Output("mg-accum-wrap", "style"),
    Output("mg-accum-select", "data"),
    Output("mg-accum-select", "value"),
    Input("mg-variable-select", "value"),
    State("mg-accum-select", "value"),
)
def switch_variable(variable, current):
    """Show the accumulation control only for variables that are accumulated."""
    options, default = _accum_options(variable)
    if not options:
        return _HIDDEN, [], None
    # On mount the layout has already restored the value from the URL.
    if ctx.triggered_id is None and current in {o["value"] for o in options}:
        return _SHOWN, options, current
    return _SHOWN, options, default


@callback(
    Output("meteogram-graph", "figure"),
    Output("mg-alert-container", "children"),
    Input("mg-run-select", "value"),
    Input("location-store", "data"),
    Input("mg-variable-select", "value"),
    Input("mg-systems-select", "value"),
    Input("mg-days-select", "value"),
    Input("mg-accum-select", "value"),
    Input("mg-models-select", "value"),
)
def update_meteogram(run, location, variable, systems, days_str, accum, models):
    if not all([location, variable, days_str]):
        return no_update, no_update
    with status.activity("Forecast at a location"):
        return _meteogram(run, location, variable, systems, days_str, accum, models)


def _meteogram(run, location, variable, systems, days_str, accum, models):
    """The meteogram callback's work, inside its status activity."""

    lat, lon = round(location["lat"], 2), round(location["lon"], 2)
    forecast_days = int(days_str)

    if run and run != LATEST:
        stores = tuple(m for m in models or [] if m in GRID_STORES)
        if not stores:
            return no_update, no_update
        return _archive_meteogram(
            run, location, lat, lon, variable, forecast_days, accum, stores
        )
    if not systems:
        return no_update, no_update

    alerts, days_by_system, valid_systems = validate_ensemble_selection(
        systems, lat, lon, forecast_days
    )
    alert_children = [
        dmc.Alert(a["message"], color=a["color"], mb="xs", withCloseButton=False)
        for a in alerts
    ]

    if not valid_systems:
        alert_children.append(
            dmc.Alert("None of the selected systems cover this location.", color="red", mb="xs")
        )
        return make_empty_fig("No data available"), alert_children

    # The control and the variable can disagree for one render after a switch.
    is_accumulated = VARIABLES[variable].get("is_accumulated", False)
    accum_hours = int(accum) if (accum and is_accumulated) else None

    dfs_by_system = {}
    # Z500 and other upper-air fields: instantaneous values at 00/12 UTC.
    synoptic = bool(VARIABLES[variable].get("synoptic_hours"))
    deterministic = tuple(s for s in valid_systems if s in DETERMINISTIC_MODELS)
    if deterministic:
        # One request to the longest reach; a shorter model returns nulls past its end.
        used_days = max(days_by_system[s] for s in deterministic)
        try:
            if synoptic:
                got = fetch_deterministic_synoptic(lat, lon, variable, deterministic, used_days)
            elif accum_hours:
                got = fetch_deterministic_accumulated(
                    lat, lon, variable, deterministic, used_days, accum_hours
                )
            else:
                got = fetch_deterministic(lat, lon, variable, deterministic, used_days)
        except Exception as exc:  # noqa: BLE001 — network/SDK errors surfaced to the user
            got = None
            alert_children.append(
                dmc.Alert(f"Could not fetch the deterministic models ({exc}).", color="red", mb="xs")
            )
        got = got or {}
        missing = [DETERMINISTIC_MODELS[s]["label"] for s in deterministic if s not in got]
        if missing:
            alert_children.append(
                dmc.Alert(
                    f"No {in_sentence(VARIABLES[variable]['label'])} from {', '.join(missing)} here.",
                    color="yellow",
                    mb="xs",
                )
            )
        dfs_by_system.update(got)
    for system in valid_systems:
        if system in DETERMINISTIC_MODELS:
            continue
        try:
            if synoptic:
                df = fetch_ensemble_synoptic(lat, lon, variable, system, days_by_system[system])
                if df["value"].notna().any():
                    dfs_by_system[system] = df
                else:
                    alert_children.append(
                        dmc.Alert(
                            f"No {in_sentence(VARIABLES[variable]['label'])} from "
                            f"{ENSEMBLE_MODELS[system]['label']} here.",
                            color="yellow",
                            mb="xs",
                        )
                    )
            elif accum_hours:
                # Sub-daily windows can only come from the hourly endpoint —
                # the daily one is fixed at 24 h sums.
                dfs_by_system[system] = fetch_ensemble_accumulated(
                    lat, lon, variable, system, days_by_system[system], accum_hours
                )
            else:
                dfs_by_system[system] = fetch_ensemble(
                    lat, lon, variable, system, days_by_system[system]
                )
        except Exception as exc:  # noqa: BLE001 — network/SDK errors surfaced to the user
            alert_children.append(
                dmc.Alert(
                    f"Could not fetch {ENSEMBLE_MODELS[system]['label']} ({exc}).",
                    color="red",
                    mb="xs",
                )
            )

    if variable in ("temperature_2m_max", "temperature_2m_min"):
        alert_children.append(
            dmc.Alert(
                "Daily extremes are Open-Meteo's, taken from hourly values (UTC days). Between "
                "a model's own 3- or 6-hourly steps those are interpolated, so the peak between "
                "steps is missed — for every ensemble here, and for IFS HRES past T+90 h.",
                color="blue",
                mb="xs",
                withCloseButton=False,
            )
        )
    if not dfs_by_system:
        alert_children.append(dmc.Alert("No data available for this selection.", color="red", mb="xs"))
        return make_empty_fig("No data available"), alert_children

    var_meta = _accumulated_meta(VARIABLES[variable], accum_hours)
    system_labels = {k: _ALL_SYSTEMS[k]["label"] for k in dfs_by_system}
    fig = make_meteogram_fig(dfs_by_system, system_labels, var_meta, location["name"])

    return fig, alert_children


def _archive_meteogram(run, location, lat, lon, variable, forecast_days, accum, stores):
    """A past run: every selected archived model, from the same init.

    Each model is read from its own store and drawn as its own series; a
    store that cannot serve the request is reported and the rest still draw.
    """
    is_accumulated = VARIABLES[variable].get("is_accumulated", False)
    accum_hours = int(accum) if (accum and is_accumulated) else None
    try:
        # A run the selected models do not all hold is a picker in transit.
        if not run_in_calendar(stores_calendar(stores), run):
            return no_update, no_update
    except GridDataError as exc:
        return make_empty_fig("No data available"), dmc.Alert(str(exc), color="red", mb="xs")

    dfs, labels, clauses, alerts, sources = {}, {}, [], [], []
    var_meta, steps = None, []
    for store in stores:
        spec = GRID_STORES[store]
        try:
            df, var_meta, meta = archive_meteogram(
                lat, lon, variable, run, forecast_days, accum_hours, store=store
            )
        except GridDataError as exc:
            alerts.append(dmc.Alert(str(exc), color="yellow", mb="xs"))
            continue
        deterministic = spec["kind"] == "deterministic"
        synoptic_hours = VARIABLES[variable].get("synoptic_hours")
        if synoptic_hours:
            # Same synoptic hours as the latest-run view (Z500 at 00/12 UTC).
            df = df[df["date"].dt.hour.isin(synoptic_hours)].reset_index(drop=True)
        dfs[store] = df
        labels[store] = f"{spec['label']}{' (deterministic)' if deterministic else ''}"
        size = "deterministic" if deterministic else f"{meta['n_members']} members"
        cell_lat, cell_lon = meta["grid_point"]
        clauses.append(
            f"{spec['label']} run {pd.Timestamp(meta['init_time']):%Y-%m-%d %H} UTC · "
            f"{size} · {spec['res_deg']:g}° cell {cell_lat:.2f}°N {cell_lon:.2f}°E"
        )
        steps.append(f"{spec['label']}: {meta['steps']}")
        if meta.get("daily_source"):
            sources.append(
                f"{spec['label']} from its own step "
                + ("maxima" if variable.endswith("_max") else "minima")
                + " (every model time step)"
                if meta["daily_source"] == "window"
                else f"{spec['label']} from its {meta['steps']} samples, which miss the peak between them"
            )

    if not dfs:
        return make_empty_fig("No data available"), alerts

    if sources:
        alerts.append(
            dmc.Alert(
                f"Daily {'maximum' if variable.endswith('_max') else 'minimum'} (UTC days): "
                + "; ".join(sources) + ".",
                color="blue",
                mb="xs",
                withCloseButton=False,
            )
        )
    elif not VARIABLES[variable]["accumulate"] and not VARIABLES[variable].get("synoptic_hours"):
        alerts.append(
            dmc.Alert(
                f"The archive has no daily {in_sentence(VARIABLES[variable]['label'])}; "
                f"showing {in_sentence(var_meta['label'])} at its native steps "
                f"({'; '.join(steps)}).",
                color="blue",
                mb="xs",
                withCloseButton=False,
            )
        )
    subtitle = f"{location['name']} · past run from the dynamical.org archive"
    fig = make_meteogram_fig(
        dfs, labels, var_meta, location["name"], subtitle=subtitle, provenance=clauses
    )
    return fig, alerts


@callback(
    Output("mg-systems-wrap", "style"),
    Output("mg-models-wrap", "style"),
    Output("mg-models-select", "value"),
    Output("mg-archive-stores", "data"),
    Output("mg-days-select", "data"),
    Input("mg-run-select-mode", "value"),
    Input("mg-models-select", "value"),
    State("mg-systems-select", "value"),
)
def switch_mode(mode, models, systems):
    """Latest shows Open-Meteo systems; Past shows archived models.

    Entering Past with no model chosen starts from the stores behind the
    selected systems. The run picker follows the models' common calendar.
    """
    if mode == LATEST:
        return _SHOWN, _HIDDEN, no_update, no_update, _FORECAST_DAY_OPTIONS
    chosen = [m for m in models or [] if m in GRID_STORES] or _default_models(systems)
    value = chosen if chosen != (models or []) else no_update
    return _HIDDEN, _SHOWN, value, chosen, _past_day_options(chosen)


register_run_options("mg-run-select", store_input=("mg-archive-stores", "data"))

publish_variable("mg-variable-select", "daily", _DEFAULT_VARIABLE)

url_sync(
    "mg-current-url",
    {
        "location-store": None,
        "mg-run-select": "run",
        "mg-variable-select": "variable",
        "mg-systems-select": "systems",
        "mg-models-select": "models",
        "mg-days-select": "days",
        "mg-accum-select": "accum",
    },
)
