"""Location picker: search a place or coordinates, or click the map.

Writes `location-store` (`{lat, lon, name}`), which every point view reads.
Three ways in, one result: a place name (Nominatim, or Mapbox if configured),
typed coordinates ("51.5, -0.12"), or a click. The map pans to the choice,
so a location restored from a shared link or another view is in sight.
"""

from __future__ import annotations

import re

import dash_leaflet as dl
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, ctx, html, no_update

from components.map_base import MARKER_STYLE, base_map, fit_point
from utils.geocoding import GeocodingError, geocode, reverse_geocode
from utils.regions import format_point, wrap_lon

_DEFAULT_CENTER = [50.0, 10.0]
# "51.5, -0.12", "51.5 -0.12", "-33.9,151.2"
_COORDS = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[, ]\s*(-?\d+(?:\.\d+)?)\s*$")


def location_search() -> dmc.Stack:
    """Search box, result list, clickable map, and a readout of the choice."""
    return dmc.Stack(
        [
            dmc.Group(
                [
                    dmc.TextInput(
                        id="loc-query",
                        placeholder="Search a place, or type lat, lon",
                        size="sm",
                        style={"flex": 1},
                        # Sent on Enter or blur only, and that is what triggers
                        # the search: with per-keystroke updates a quick Enter
                        # reached the server before the new text.
                        debounce=True,
                    ),
                    dmc.Button(
                        "Search", id="loc-search-btn", size="sm", variant="light"
                    ),
                ],
                gap="xs",
                wrap="nowrap",
            ),
            html.Div(
                dmc.Select(
                    id="loc-results",
                    placeholder="Choose a match",
                    data=[],
                    size="sm",
                    allowDeselect=False,
                ),
                id="loc-results-wrap",
                style={"display": "none"},
            ),
            dmc.Text(id="loc-search-note", size="xs", c="dimmed"),
            base_map(
                "location-map",
                [dl.LayerGroup(id="location-marker-layer")],
                center=_DEFAULT_CENTER,
                zoom=3,
                height="300px",
            ),
            dmc.Text("…or click the map.", size="xs", c="dimmed"),
            dmc.Text(id="location-label", size="sm", fw=500),
        ],
        gap="xs",
    )


@callback(
    Output("location-store", "data", allow_duplicate=True),
    Output("loc-results", "data"),
    Output("loc-results", "value"),
    Output("loc-results-wrap", "style"),
    Output("loc-search-note", "children"),
    Input("loc-query", "value"),
    Input("loc-query", "n_submit"),
    prevent_initial_call=True,
)
def search(query, _submit):
    """Coordinates go straight in; a place name lists its matches.

    Triggered by the text itself, which `debounce=True` delivers on Enter or
    blur. The Search button has no callback of its own: clicking it blurs
    the input, which sends the text. Wiring the click separately made it
    race the blur and search the previous text.
    """
    hidden = {"display": "none"}
    if not query or not query.strip():
        return no_update, [], None, hidden, ""

    match = _COORDS.match(query)
    if match:
        lat, lon = float(match.group(1)), float(match.group(2))
        if not -90 <= lat <= 90:
            return no_update, [], None, hidden, "Latitude must be between -90 and 90."
        lat, lon = round(lat, 2), round(wrap_lon(lon), 2)
        return (
            {"lat": lat, "lon": lon, "name": reverse_geocode(lat, lon)},
            [],
            None,
            hidden,
            "",
        )

    try:
        results = geocode(query.strip())
    except GeocodingError as exc:
        return no_update, [], None, hidden, str(exc)
    if not results:
        return no_update, [], None, hidden, f"No place found for “{query.strip()}”."

    # Nominatim often returns a city and its municipality as two hits with
    # the same name; one entry per distinct label is enough.
    options, seen = [], set()
    for r in results:
        label = r["name"] + (f" — {r['detail']}" if r.get("detail") else "")
        if label in seen:
            continue
        seen.add(label)
        options.append({"value": f"{r['lat']:.4f}|{r['lon']:.4f}|{r['name']}", "label": label})
    if len(options) == 1:
        lat, lon, name = options[0]["value"].split("|", 2)
        store = {"lat": round(float(lat), 2), "lon": round(float(lon), 2), "name": name}
        return store, [], None, hidden, ""
    return (
        no_update,
        options,
        None,
        {"display": "block"},
        f"{len(options)} matches — choose one.",
    )


@callback(
    Output("location-store", "data", allow_duplicate=True),
    Output("loc-results-wrap", "style", allow_duplicate=True),
    Output("loc-search-note", "children", allow_duplicate=True),
    Input("loc-results", "value"),
    prevent_initial_call=True,
)
def pick_result(value):
    if not value:
        return no_update, no_update, no_update
    lat, lon, name = value.split("|", 2)
    store = {"lat": round(float(lat), 2), "lon": round(float(lon), 2), "name": name}
    return store, {"display": "none"}, ""


@callback(
    Output("location-store", "data", allow_duplicate=True),
    Input("location-map", "clickData"),
    prevent_initial_call=True,
)
def pick_location(click_data):
    if not click_data:
        return no_update

    latlng = click_data["latlng"]
    lat = round(max(-90.0, min(90.0, latlng["lat"])), 2)
    lon = round(wrap_lon(latlng["lng"]), 2)
    return {"lat": lat, "lon": lon, "name": reverse_geocode(lat, lon)}


@callback(
    Output("location-marker-layer", "children"),
    Output("location-label", "children"),
    Output("location-map", "viewport"),
    Input("location-store", "data"),
    State("location-map", "zoom"),
)
def show_location(location, zoom):
    """Draw the stored location and bring it into view; also runs on mount."""
    if not location:
        return [], "No location chosen yet.", no_update

    lat, lon = location["lat"], location["lon"]
    marker = dl.CircleMarker(
        center=[lat, lon],
        radius=7,
        **MARKER_STYLE,
        children=dl.Tooltip(location["name"]),
    )
    # Keep the user's zoom after a click; zoom in from the opening world view.
    target_zoom = zoom if ctx.triggered_id and zoom and zoom >= 5 else 6
    return (
        [marker],
        f"{location['name']} · {format_point(lat, lon)}",
        fit_point(lat, lon, target_zoom),
    )
