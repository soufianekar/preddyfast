"""Layout + all callbacks for the daily climate page."""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from components.export_button import export_button
from components.location_search import location_search
from components.page_shell import page_shell
from components.shared_state import publish_variable, url_sync
from utils import status
from utils.climatology import (
    SEASONS,
    compute_acc_pct,
    compute_daily_pct,
    compute_monthly,
    compute_seasonal,
    compute_trend,
    make_summary,
    validate_selection,
)
from utils.figures import (
    CONFIG,
    make_accumulation_fig,
    make_anomaly_bar_fig,
    make_daily_fig,
    make_empty_fig,
    make_monthly_fig,
    make_seasonal_fig,
)
from utils.openmeteo_api import fetch_forecast, fetch_historical
from utils.settings import CLIM_PERIODS, MODELS, VARIABLES
from utils.url_state import query_choice

dash.register_page(__name__, path="/", title="Climate at a location · asifast")

# Variables with `climate: False` (Z500) have no reanalysis series here.
_CLIMATE_VARIABLES = {k: v for k, v in VARIABLES.items() if v.get("climate", True)}
_VARIABLE_OPTIONS = [{"value": k, "label": v["label"]} for k, v in _CLIMATE_VARIABLES.items()]
_MODEL_OPTIONS = [{"value": k, "label": v["label"]} for k, v in MODELS.items()]
_CLIM_PERIOD_OPTIONS = [{"value": p, "label": p} for p in CLIM_PERIODS]

# The selector's own default. Used to tell "the user chose 1991-2020" apart
# from "the user has not touched this", which decides whether a variable's
# `clim_period_acc` applies — see `_acc_clim_period`.
_DEFAULT_CLIM_PERIOD = "1991-2020"
_DEFAULT_VARIABLE = "temperature_2m_mean"
_CURRENT_YEAR = pd.Timestamp.now().year
_YEAR_OPTIONS = [{"value": str(y), "label": str(y)} for y in range(_CURRENT_YEAR, 1980, -1)]

_PLOT_TYPE_LABELS = {
    "acc": "Year to date",
    "monthly": "Monthly",
    "seasonal": "Seasonal",
    "daily": "Daily values",
    "daily_bar": "Daily anomaly",
}
_SEASON_OPTIONS = [{"value": k, "label": k} for k in SEASONS]


def _plot_type_options(var_meta: dict) -> list[dict]:
    """Views that suit a variable.

    Year to date only for accumulated variables. Daily views only where a
    daily distribution means something — not for precipitation or snowfall,
    whose daily percentiles collapse to zero below the median.
    """
    keys = ["monthly", "seasonal"]
    if var_meta["accumulate"]:
        keys.insert(0, "acc")
    if var_meta.get("daily_views", True):
        keys += ["daily", "daily_bar"]
    return [{"value": k, "label": _PLOT_TYPE_LABELS[k]} for k in keys]


def _latest_season() -> str:
    """Season of the most recent month the reanalysis has fully covered."""
    month = (pd.Timestamp.now() - pd.Timedelta(days=35)).month
    return next(k for k, months in SEASONS.items() if month in months)


def _acc_clim_period(var_meta: dict, clim_period: str) -> tuple[int, int, str]:
    """Reference period for the YTD accumulation bands, and its label.

    A YTD accumulation and a daily value are not asking the same question of
    the climatology, and the project fixes different defaults for them:
    1981-2020 for accumulation, 1991-2020 for daily. `clim_period_acc` in
    `settings.VARIABLES` carries the former per variable.

    An explicit choice in the selector overrides it — but only an explicit
    one. While the selector sits at its default, the daily plot is showing
    1991-2020 and the accumulation plot is showing the variable's own
    accumulation baseline, and each says so in its own subtitle.

    Returns
    -------
    tuple[int, int, str]
        Start year, end year, and the "YYYY-YYYY" label to print.
    """
    if clim_period == _DEFAULT_CLIM_PERIOD and "clim_period_acc" in var_meta:
        start, end = var_meta["clim_period_acc"]
    else:
        start, end = (int(x) for x in clim_period.split("-"))
    return start, end, f"{start}-{end}"


def _df_to_store(df: pd.DataFrame) -> dict:
    """Serialize a date/value/doy frame for `dcc.Store`, ISO-stringifying dates."""
    df = df.copy()
    if "date" in df.columns:
        df["date"] = df["date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return df.to_dict(orient="split")


def _df_from_store(payload: dict | None) -> pd.DataFrame:
    if payload is None:
        return pd.DataFrame()
    df = pd.DataFrame(**payload)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def layout(**query_params):
    variable = query_choice(query_params, "variable", _CLIMATE_VARIABLES, _DEFAULT_VARIABLE)
    year = query_choice(
        query_params, "year", [o["value"] for o in _YEAR_OPTIONS], str(_CURRENT_YEAR)
    )
    model = query_choice(query_params, "model", MODELS, "era5_seamless")
    clim_period = query_choice(
        query_params, "clim_period", CLIM_PERIODS, _DEFAULT_CLIM_PERIOD
    )
    plot_type_options = _plot_type_options(VARIABLES[variable])
    season = query_choice(query_params, "season", SEASONS, _latest_season())
    plot_type = query_choice(
        query_params,
        "plot_type",
        [o["value"] for o in plot_type_options],
        plot_type_options[0]["value"],
    )

    return page_shell(
        title="Climate at a location",
        description=(
            "How this year compares with the reference climate — day by day, "
            "month by month, season by season."
        ),
        sidebar=[
            html.Div(id="current-url", style={"display": "none"}),
            "Where",
            location_search(),
            "What",
            dmc.Select(
                id="variable-select",
                label="Variable",
                data=_VARIABLE_OPTIONS,
                value=variable,
            ),
            "Reference",
            dmc.SimpleGrid(
                [
                    dmc.Select(
                        id="year-select",
                        label="Year",
                        data=_YEAR_OPTIONS,
                        value=year,
                    ),
                    dmc.Select(
                        id="clim-period-select",
                        label="Climatology period",
                        data=_CLIM_PERIOD_OPTIONS,
                        value=clim_period,
                    ),
                ],
                cols=2,
                spacing="sm",
            ),
            dmc.Select(
                id="model-select",
                label="Data source",
                data=_MODEL_OPTIONS,
                value=model,
            ),
        ],
        main=[
            html.Div(id="alert-container"),
            dmc.Text(id="summary-text", mb="md", fw=500),
            dcc.Store(id="climate-data-store"),
            dmc.SegmentedControl(
                id="plot-type-select",
                data=plot_type_options,
                value=plot_type,
                size="xs",
                fullWidth=True,
                mb="xs",
            ),
            html.Div(
                dmc.SegmentedControl(
                    id="season-select",
                    data=_SEASON_OPTIONS,
                    value=season,
                    size="xs",
                ),
                id="season-wrap",
                style={"display": "block" if plot_type == "seasonal" else "none"},
            ),
            dmc.Space(h="xs"),
            dcc.Loading(
                dcc.Graph(
                    id="climate-graph",
                    config=CONFIG,
                    figure=make_empty_fig("Pick a location on the map"),
                ),
                type="dot",
            ),
            dmc.Space(h="md"),
            export_button(),
        ],
        graph_id="climate-graph",
        share_target="current-url",
    )


@callback(
    Output("climate-data-store", "data"),
    Output("alert-container", "children"),
    Output("summary-text", "children"),
    Output("plot-type-select", "data"),
    Output("plot-type-select", "value"),
    Input("location-store", "data"),
    Input("variable-select", "value"),
    Input("year-select", "value"),
    Input("model-select", "value"),
    Input("clim-period-select", "value"),
    State("plot-type-select", "value"),
)
def update_data(location, variable, year_str, model, clim_period, current_plot_type):
    if not all([location, variable, year_str, model, clim_period]):
        return no_update, no_update, no_update, no_update, no_update

    year = int(year_str)
    lat, lon = round(location["lat"], 2), round(location["lon"], 2)
    clim_start, clim_end = (int(x) for x in clim_period.split("-"))

    var_meta = VARIABLES[variable]
    plot_type_options = _plot_type_options(var_meta)
    valid_types = [o["value"] for o in plot_type_options]
    plot_type_value = (
        current_plot_type if current_plot_type in valid_types else valid_types[0]
    )

    alerts, used_year = validate_selection(model, lat, lon, year)
    alert_children = [
        dmc.Alert(a["message"], color=a["color"], mb="xs", withCloseButton=False)
        for a in alerts
    ]

    try:
        with status.activity("Climate history"):
            df = fetch_historical(lat, lon, variable, model)
    except Exception as exc:  # noqa: BLE001 — network/SDK errors surfaced to the user
        alert_children.append(
            dmc.Alert(
                f"Could not fetch data from Open-Meteo ({exc}). "
                "Try ERA5-Seamless if this persists.",
                color="red",
                mb="xs",
            )
        )
        return "empty", alert_children, no_update, plot_type_options, plot_type_value

    if df.empty:
        alert_children.append(dmc.Alert("No data available for this selection.", color="red", mb="xs"))
        return "empty", alert_children, no_update, plot_type_options, plot_type_value

    pct_daily = compute_daily_pct(df, clim_start, clim_end)
    current = df[df["date"].dt.year == used_year].copy()

    forecast = None
    if used_year == pd.Timestamp.now().year:
        try:
            forecast = fetch_forecast(lat, lon, variable)
        except Exception as exc:  # noqa: BLE001 — forecast is best-effort
            alert_children.append(
                dmc.Alert(f"Forecast extension unavailable ({exc}).", color="yellow", mb="xs")
            )

    pct_acc = None
    acc_clim_period = None
    trend_slope = None
    if var_meta["accumulate"]:
        acc_start, acc_end, acc_clim_period = _acc_clim_period(var_meta, clim_period)
        pct_acc = compute_acc_pct(df, acc_start, acc_end)
        _, _, trend_slope = compute_trend(df, accumulate=True)

    reference = (clim_start, clim_end)
    monthly = compute_monthly(df, used_year, reference, var_meta["accumulate"])
    seasonal = {}
    for season in SEASONS:
        table, season_stats = compute_seasonal(df, season, reference, var_meta["accumulate"])
        seasonal[season] = {
            "table": _df_to_store(table.rename_axis("season_year")),
            "stats": season_stats,
        }

    summary = make_summary(
        df, current, var_meta["label"], used_year, var_meta["accumulate"],
        var_meta["direction_labels"],
    )

    data = {
        "variable": variable,
        "model_label": MODELS[model]["label"],
        "clim_period": clim_period,
        "acc_clim_period": acc_clim_period,
        "year": used_year,
        "trend_slope": trend_slope,
        "pct_daily": _df_to_store(pct_daily),
        "pct_acc": _df_to_store(pct_acc) if pct_acc is not None else None,
        "current": _df_to_store(current),
        "forecast": _df_to_store(forecast) if forecast is not None and not forecast.empty else None,
        "monthly": _df_to_store(monthly),
        "seasonal": seasonal,
    }

    return data, alert_children, summary, plot_type_options, plot_type_value


@callback(
    Output("climate-graph", "figure"),
    Input("climate-data-store", "data"),
    Input("plot-type-select", "value"),
    Input("season-select", "value"),
    prevent_initial_call=True,
)
def render_figure(data, plot_type, season):
    if not plot_type:
        return no_update
    if not data:
        return make_empty_fig("No data available")
    if data == "empty":
        return make_empty_fig("No data available")

    var_meta = VARIABLES[data["variable"]]
    current = _df_from_store(data["current"])

    if plot_type == "monthly":
        return make_monthly_fig(
            _df_from_store(data["monthly"]), var_meta, data["model_label"],
            data["clim_period"], data["year"],
        )
    if plot_type == "seasonal":
        entry = data["seasonal"][season or "DJF"]
        return make_seasonal_fig(
            _df_from_store(entry["table"]), entry["stats"], season, var_meta,
            data["model_label"], data["clim_period"], data["year"],
        )

    if plot_type == "acc":
        if data["pct_acc"] is None:
            return make_empty_fig("Accumulation not applicable for this variable")
        pct_acc = _df_from_store(data["pct_acc"])
        forecast = _df_from_store(data["forecast"]) if data["forecast"] is not None else None
        return make_accumulation_fig(
            pct_acc, current, forecast, var_meta, data["model_label"],
            data["acc_clim_period"], data["year"], data["trend_slope"],
        )

    pct_daily = _df_from_store(data["pct_daily"])
    if plot_type == "daily_bar":
        return make_anomaly_bar_fig(
            pct_daily, current, var_meta, data["model_label"], data["clim_period"], data["year"]
        )

    return make_daily_fig(
        pct_daily, current, var_meta, data["model_label"], data["clim_period"], data["year"]
    )


@callback(
    Output("season-wrap", "style"),
    Input("plot-type-select", "value"),
)
def show_season_picker(plot_type):
    return {"display": "block" if plot_type == "seasonal" else "none"}


publish_variable("variable-select", "daily", _DEFAULT_VARIABLE)

url_sync(
    "current-url",
    {
        "location-store": None,
        "variable-select": "variable",
        "year-select": "year",
        "model-select": "model",
        "clim-period-select": "clim_period",
        "plot-type-select": "plot_type",
        "season-select": "season",
    },
)
