"""Area picker for Climate › Maps: a standard region, or any box.

Same gestures and look as `components/bbox_select` (both build on
`components/map_base`), but not the same component: that one snaps to the
0.25° archive grid and caps boxes at 10° because it drives an area average;
this one has to express a global extent over 0.4° and 0.1° Copernicus grids,
so neither constraint applies.

The four edge inputs are the single source of truth. A region fills them; a
drawn or typed box that is no longer a region's box turns the menu to
"Custom box", so the menu never names a region that is not what will be
retrieved. The active area is outlined — AR6 regions by their polygon — and
framed on the map.
"""

from __future__ import annotations

import dash_leaflet as dl
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, ctx, dcc, no_update

from components.map_base import BOX_STYLE, REGION_STYLE, base_map, fit_box
from utils.regions import (
    box_from_geojson,
    clamp_box,
    format_box,
    grouped_options,
    polygon,
)
from utils.settings import MAP_AREA_PRESETS

CUSTOM = "custom"
_EDGES = ("lat_min", "lat_max", "lon_min", "lon_max")
_EDGE_LABELS = {
    "lat_max": "North",
    "lat_min": "South",
    "lon_min": "West",
    "lon_max": "East",
}

_DEFAULT_PRESET = "ecmwf_europe"
_PRESET_OPTIONS = grouped_options(
    extra=[{"value": CUSTOM, "label": "Custom box (draw on the map)"}]
)


def _edge_input(
    edge: str, bounds: tuple[float, float, float, float]
) -> dmc.NumberInput:
    is_lat = edge.startswith("lat")
    return dmc.NumberInput(
        id=f"ma-{edge}",
        label=_EDGE_LABELS[edge],
        value=bounds[_EDGES.index(edge)],
        min=-90.0 if is_lat else -180.0,
        max=90.0 if is_lat else 180.0,
        clampBehavior="strict",
        step=1.0,
        decimalScale=2,
        size="xs",
    )


def map_area_select() -> dmc.Stack:
    """Region menu, map with a rectangle tool, the four edges, and the store."""
    bounds = MAP_AREA_PRESETS[_DEFAULT_PRESET]
    return dmc.Stack(
        [
            dmc.Select(
                id="ma-preset-select",
                label="Area",
                description="A standard region (ECMWF, WMO, IPCC AR6), or draw a box.",
                data=_PRESET_OPTIONS,
                value=_DEFAULT_PRESET,
                searchable=True,
                allowDeselect=False,
                size="xs",
            ),
            base_map(
                "ma-map",
                [
                    dl.FeatureGroup(
                        [
                            dl.EditControl(
                                id="ma-edit-control",
                                position="topright",
                                draw={
                                    "rectangle": {"shapeOptions": BOX_STYLE},
                                    "polygon": False,
                                    "polyline": False,
                                    "circle": False,
                                    "circlemarker": False,
                                    "marker": False,
                                },
                                # `remove` stays enabled: the auto-clear drives
                                # the edit toolbar's remove mode, and
                                # leaflet-draw only registers a mode it has been
                                # given. Commanding a disabled one throws on mount.
                                edit={"edit": False, "remove": True},
                            )
                        ]
                    ),
                    dl.LayerGroup(id="ma-rect-layer"),
                ],
                center=[(bounds[0] + bounds[1]) / 2, (bounds[2] + bounds[3]) / 2],
                zoom=2,
                height="260px",
            ),
            dmc.Grid(
                [
                    dmc.GridCol(_edge_input(edge, bounds), span=3)
                    for edge in ("lat_max", "lat_min", "lon_min", "lon_max")
                ],
                gutter="xs",
            ),
            dmc.Text(id="ma-label", size="xs", c="dimmed"),
            dcc.Store(id="map-area-store"),
        ],
        gap="xs",
    )


@callback(
    Output("ma-lat_min", "value"),
    Output("ma-lat_max", "value"),
    Output("ma-lon_min", "value"),
    Output("ma-lon_max", "value"),
    Output("ma-edit-control", "editToolbar"),
    Output("ma-preset-select", "value", allow_duplicate=True),
    Input("ma-edit-control", "geojson"),
    Input("ma-preset-select", "value"),
    prevent_initial_call=True,
)
def edges_from_draw_or_preset(geojson, preset):
    """Write a dragged rectangle or a chosen region into the four edges.

    Both sources land here so whichever acted last wins outright. A drawing
    also turns the menu to "Custom box".
    """
    if ctx.triggered_id == "ma-preset-select":
        if preset not in MAP_AREA_PRESETS:
            return (no_update,) * 6
        return (*MAP_AREA_PRESETS[preset], no_update, no_update)

    box = box_from_geojson(geojson)
    if box is None:
        return (no_update,) * 6
    # `n_clicks` must advance for dash-leaflet to act on the clear command.
    clear = {
        "mode": "remove",
        "action": "clear all",
        "n_clicks": len(geojson["features"]),
    }
    return (*box, clear, CUSTOM)


@callback(
    Output("map-area-store", "data"),
    Output("ma-rect-layer", "children"),
    Output("ma-label", "children"),
    Output("ma-map", "viewport"),
    Output("ma-preset-select", "value", allow_duplicate=True),
    Input("ma-lat_min", "value"),
    Input("ma-lat_max", "value"),
    Input("ma-lon_min", "value"),
    Input("ma-lon_max", "value"),
    State("ma-preset-select", "value"),
    prevent_initial_call="initial_duplicate",
)
def sync_area(lat_min, lat_max, lon_min, lon_max, preset):
    """Publish the ordered box, outline and frame it, and keep the menu honest.

    Edges are ordered and clamped here, so a north value typed below the
    south value widens the box rather than producing a request the API
    would reject minutes later. When the edges are no longer the selected
    region's box, the menu switches to "Custom box".
    """
    if any(value is None for value in (lat_min, lat_max, lon_min, lon_max)):
        return (no_update,) * 5

    edges = clamp_box(lat_min, lat_max, lon_min, lon_max)
    region_box = MAP_AREA_PRESETS.get(preset or "")
    matches = region_box is not None and all(
        abs(a - b) < 0.01 for a, b in zip(region_box, edges, strict=True)
    )
    region = preset if matches else None
    data = dict(zip(_EDGES, edges, strict=True))
    data["preset"] = region

    if edges == MAP_AREA_PRESETS["global"]:
        # The global extent is sent as *no* `area` key at all, so the label
        # says that rather than implying a -180/180 subset was requested.
        label, shapes = "Global — retrieved without an area subset", []
    else:
        label = format_box(edges)
        vertices = polygon(region)
        if vertices is not None:
            shapes = [
                dl.Polygon(
                    positions=[[lat, lon] for lon, lat in vertices], **REGION_STYLE
                )
            ]
        else:
            style = REGION_STYLE if region else BOX_STYLE
            shapes = [
                dl.Rectangle(
                    bounds=[[edges[0], edges[2]], [edges[1], edges[3]]], **style
                )
            ]
    menu = no_update if (matches or preset == CUSTOM) else CUSTOM
    return data, shapes, label, fit_box(edges), menu
