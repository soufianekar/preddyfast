"""Area picker for the forecast area view: a standard region, or a box.

One component owns both halves of the choice, so they cannot disagree:

- the **region** select lists standard regions (ECMWF, WMO, IPCC AR6) plus
  "Custom box";
- the **box** is drawn on the map or typed as four edges, snapped to the
  0.25° archive grid and clamped to `BBOX_MAX_SPAN_DEG`.

Drawing a box switches the select to "Custom box" — a drawing that changed
nothing because a region was still selected is the confusion this replaces.
Only the active choice is outlined (purple region or red box), and the map
frames it. The edge inputs always show the box that will actually be read,
snapped and clamped, never the raw drag.

`bbox-store` holds the custom box; `bbox-region-select` says which of the
two is in use.
"""

from __future__ import annotations

import dash_leaflet as dl
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from components.map_base import BOX_STYLE, REGION_STYLE, base_map, fit_box
from utils.earthmover_api import snap_bbox
from utils.regions import REGIONS, box_from_geojson, format_box, polygon
from utils.settings import BBOX_MAX_SPAN_DEG

CUSTOM = "custom"
# Opening box: over the Alps, wide enough to show the tool is regional.
_DEFAULT_BOUNDS = (45.0, 50.0, 6.0, 11.0)  # lat_min, lat_max, lon_min, lon_max
_EDGES = ("lat_min", "lat_max", "lon_min", "lon_max")
_EDGE_LABELS = {
    "lat_max": "North",
    "lat_min": "South",
    "lon_min": "West",
    "lon_max": "East",
}


def _edge_input(edge: str, value: float) -> dmc.NumberInput:
    is_lat = edge.startswith("lat")
    return dmc.NumberInput(
        id=f"bbox-{edge}",
        label=_EDGE_LABELS[edge],
        value=value,
        min=-90.0 if is_lat else -180.0,
        max=90.0 if is_lat else 180.0,
        clampBehavior="strict",
        step=0.25,
        decimalScale=2,
        size="xs",
    )


def bbox_select(
    bounds: dict[str, float] | None = None,
    region: str = CUSTOM,
    region_options: list | None = None,
) -> dmc.Stack:
    """Region select, map with a rectangle tool, and the box's four edges.

    Parameters
    ----------
    bounds
        Box edges from a shared link. When given they replace the session's
        box: `edges_from_store` stands down and `sync_bbox` publishes them.
    region
        Initially selected region key, or `CUSTOM`.
    region_options
        `dmc.Select` data for the region list (the page decides which
        regions are small enough to offer).
    """
    start = tuple(bounds[e] for e in _EDGES) if bounds else _DEFAULT_BOUNDS
    return dmc.Stack(
        [
            dmc.Select(
                id="bbox-region-select",
                label="Area",
                description="A standard region, or draw your own box on the map.",
                data=region_options or [{"value": CUSTOM, "label": "Custom box"}],
                value=region,
                searchable=True,
                allowDeselect=False,
            ),
            base_map(
                "bbox-map",
                [
                    # The draw tool writes here; cleared after every draw.
                    dl.FeatureGroup(
                        [
                            dl.EditControl(
                                id="bbox-edit-control",
                                position="topright",
                                # Rectangle only: the store holds lat/lon boxes,
                                # and offering a polygon would promise a shape
                                # it cannot represent.
                                draw={
                                    "rectangle": {"shapeOptions": BOX_STYLE},
                                    "polygon": False,
                                    "polyline": False,
                                    "circle": False,
                                    "circlemarker": False,
                                    "marker": False,
                                },
                                # `remove` must stay enabled: the auto-clear
                                # drives the edit toolbar's remove mode, and
                                # leaflet-draw only registers modes it was
                                # given — commanding a disabled one throws
                                # "Cannot read properties of undefined
                                # (reading 'handler')" on mount.
                                edit={"edit": False, "remove": True},
                            )
                        ]
                    ),
                    dl.LayerGroup(id="bbox-shape-layer"),
                ],
                center=[(start[0] + start[1]) / 2, (start[2] + start[3]) / 2],
                zoom=4,
            ),
            html.Div(
                dmc.Stack(
                    [
                        dmc.Text(
                            "Drag with the square tool (top right), or type the edges.",
                            size="xs",
                            c="dimmed",
                        ),
                        dmc.Grid(
                            [
                                dmc.GridCol(
                                    _edge_input(e, start[_EDGES.index(e)]), span=3
                                )
                                for e in ("lat_max", "lat_min", "lon_min", "lon_max")
                            ],
                            gutter="xs",
                        ),
                    ],
                    gap=4,
                ),
                id="bbox-edges-wrap",
            ),
            dmc.Text(id="bbox-label", size="sm", fw=500),
            dcc.Store(id="bbox-seed", data=bounds is not None),
        ],
        gap="xs",
    )


@callback(
    Output("bbox-lat_min", "value"),
    Output("bbox-lat_max", "value"),
    Output("bbox-lon_min", "value"),
    Output("bbox-lon_max", "value"),
    Input("bbox-seed", "data"),
    State("bbox-store", "data"),
    State("bbox-lat_min", "value"),
    State("bbox-lat_max", "value"),
    State("bbox-lon_min", "value"),
    State("bbox-lon_max", "value"),
)
def edges_from_store(seeded, store, lat_min, lat_max, lon_min, lon_max):
    """Load the session's box into the edge inputs when the page mounts.

    Triggered by `bbox-seed`, which never changes after mount, rather than by
    `bbox-store`: with the store as an input this and `sync_bbox` form a
    cycle, Dash cannot order a cycle, and `sync_bbox` publishes the defaults
    over the stored box first. One-way, `sync_bbox` waits for this callback.

    Returns the edges as values even when there is nothing to load. An
    all-`no_update` return would cancel the dependent initial call, and
    `sync_bbox` would never publish the box. Stands down when a shared link
    seeded the edges.
    """
    if seeded or not store:
        return lat_min, lat_max, lon_min, lon_max
    return tuple(store[e] for e in _EDGES)


@callback(
    Output("bbox-lat_min", "value", allow_duplicate=True),
    Output("bbox-lat_max", "value", allow_duplicate=True),
    Output("bbox-lon_min", "value", allow_duplicate=True),
    Output("bbox-lon_max", "value", allow_duplicate=True),
    Output("bbox-edit-control", "editToolbar"),
    Output("bbox-region-select", "value", allow_duplicate=True),
    Input("bbox-edit-control", "geojson"),
    prevent_initial_call=True,
)
def bounds_from_draw(geojson):
    """A drawn rectangle becomes the custom box, and the custom box is selected.

    The drawing itself is cleared at once; the box is redrawn from the
    (snapped) edges, so exactly one rectangle is ever on the map.
    """
    box = box_from_geojson(geojson)
    if box is None:
        return (no_update,) * 6
    n_features = len(geojson["features"])
    # `n_clicks` must advance for dash-leaflet to act on the clear command.
    clear = {"mode": "remove", "action": "clear all", "n_clicks": n_features}
    return (*box, clear, CUSTOM)


@callback(
    Output("bbox-store", "data"),
    Output("bbox-lat_min", "value", allow_duplicate=True),
    Output("bbox-lat_max", "value", allow_duplicate=True),
    Output("bbox-lon_min", "value", allow_duplicate=True),
    Output("bbox-lon_max", "value", allow_duplicate=True),
    Input("bbox-lat_min", "value"),
    Input("bbox-lat_max", "value"),
    Input("bbox-lon_min", "value"),
    Input("bbox-lon_max", "value"),
    # Runs on mount too. Its inputs' primary writer is `edges_from_store`,
    # so Dash still orders it after that callback.
    prevent_initial_call="initial_duplicate",
)
def sync_bbox(lat_min, lat_max, lon_min, lon_max):
    """Snap the edges to the grid and publish the box.

    The only writer of `bbox-store`. When snapping or the span limit moved an
    edge, the inputs are rewritten with the box actually used, so what is
    typed, what is drawn and what is read never disagree. The rewrite
    re-enters once and stops: a snapped box snaps to itself.
    """
    if any(v is None for v in (lat_min, lat_max, lon_min, lon_max)):
        return (no_update,) * 5

    # What the inputs show now, compared against the box actually used below.
    raw = (float(lat_min), float(lat_max), float(lon_min), float(lon_max))
    lat_min, lat_max, lon_min, lon_max = _anchor_typed_edge(ctx.triggered_id, *raw)
    snapped = snap_bbox(lat_min, lat_max, lon_min, lon_max)
    changed = any(abs(a - b) > 1e-6 for a, b in zip(snapped, raw, strict=True))
    edges = snapped if changed else (no_update,) * 4
    return dict(zip(_EDGES, snapped, strict=True)), *edges


def _anchor_typed_edge(
    edited: str | None, lat_min: float, lat_max: float, lon_min: float, lon_max: float
) -> tuple[float, float, float, float]:
    """Keep a typed edge where it was typed when the box would exceed the limit.

    `snap_bbox` clamps an oversized box about its centre, which is right for
    a drag but wrong for typing: entering East = 60 on a box at 150°W would
    slide the whole box into the Atlantic. Here the opposite edge gives way.
    """
    limit = BBOX_MAX_SPAN_DEG
    if edited == "bbox-lon_max" and lon_max - lon_min > limit:
        lon_min = lon_max - limit
    elif edited == "bbox-lon_min" and lon_max - lon_min > limit:
        lon_max = lon_min + limit
    elif edited == "bbox-lat_max" and lat_max - lat_min > limit:
        lat_min = lat_max - limit
    elif edited == "bbox-lat_min" and lat_max - lat_min > limit:
        lat_max = lat_min + limit
    return lat_min, lat_max, lon_min, lon_max


@callback(
    Output("bbox-shape-layer", "children"),
    Output("bbox-map", "viewport"),
    Output("bbox-edges-wrap", "style"),
    Output("bbox-label", "children"),
    Input("bbox-store", "data"),
    Input("bbox-region-select", "value"),
)
def show_selection(box, region):
    """Outline whichever choice is active, frame it, and describe it."""
    if region and region != CUSTOM and region in REGIONS:
        info = REGIONS[region]
        bbox = info["bbox"]
        vertices = polygon(region)
        shape = (
            dl.Polygon(positions=[[lat, lon] for lon, lat in vertices], **REGION_STYLE)
            if vertices is not None
            else dl.Rectangle(
                bounds=[[bbox[0], bbox[2]], [bbox[1], bbox[3]]], **REGION_STYLE
            )
        )
        label = f"{info['label']} · {format_box(bbox)}"
        return [shape], fit_box(bbox), {"display": "none"}, label

    if not box:
        return [], no_update, {"display": "block"}, ""
    edges = tuple(box[e] for e in _EDGES)
    rectangle = dl.Rectangle(
        bounds=[[edges[0], edges[2]], [edges[1], edges[3]]], **BOX_STYLE
    )
    label = f"Custom box · {format_box(edges)}"
    if max(edges[1] - edges[0], edges[3] - edges[2]) >= BBOX_MAX_SPAN_DEG:
        label += (
            f" — drawn boxes are limited to {BBOX_MAX_SPAN_DEG:g}° (snapped outward "
            "to the 0.25° grid); choose a standard region for a larger area"
        )
    return [rectangle], fit_box(edges), {"display": "block"}, label
