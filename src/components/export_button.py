"""CSV download button. PNG export uses the Plotly toolbar (see utils/figures.CONFIG)."""

from __future__ import annotations

import dash_mantine_components as dmc
from dash import Input, Output, State, callback, dcc, no_update

from utils.openmeteo_api import fetch_historical


def export_button() -> dmc.Group:
    return dmc.Group(
        [
            dmc.Button("Export CSV", id="export-csv-btn", variant="light"),
            dcc.Download(id="download-csv"),
        ]
    )


@callback(
    Output("download-csv", "data"),
    Input("export-csv-btn", "n_clicks"),
    State("location-store", "data"),
    State("variable-select", "value"),
    State("year-select", "value"),
    State("model-select", "value"),
    prevent_initial_call=True,
)
def export_csv(_, location, variable, year, model):
    if not all([location, variable, year, model]):
        return no_update

    year = int(year)
    df = fetch_historical(round(location["lat"], 2), round(location["lon"], 2), variable, model)
    current = df[df["date"].dt.year == year]
    filename = f"{location['name'].replace(' ', '_')}_{variable}_{year}.csv"
    return dcc.send_data_frame(current.to_csv, filename, index=False)
