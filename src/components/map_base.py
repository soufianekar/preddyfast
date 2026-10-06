"""The base map every picker is built on: grey tiles, one world, shared styles.

Every point or area picker in the app uses `base_map`, so they look and
behave the same:

- OpenStreetMap tiles shown in light grey (CSS filter on the tile layer),
  which keep a drawn selection legible;
- the view is held to one copy of the world (`maxBounds` with full
  viscosity, tiles with `noWrap`), so a click or a drag can never report a
  longitude of 200° after the map has been panned sideways;
- selections are styled alike: the red box being drawn or edited, the
  purple outline of a standard region, the blue location marker.

`fit_box` / `fit_point` give the `viewport` value that frames a selection,
so a restored, typed or searched selection is brought into view.
"""

from __future__ import annotations

import dash_leaflet as dl

from utils.settings import BASEMAP_ATTRIBUTION, BASEMAP_TILE_URL

WORLD_BOUNDS = [[-90.0, -180.0], [90.0, 180.0]]

# Red, unfilled: the box the user drew or typed. Unfilled so it does not
# hide the ground being selected.
BOX_STYLE = {"color": "#c23b32", "weight": 2, "fill": False, "fillOpacity": 0}
# Purple outline with a faint fill: a standard region, which is not editable.
REGION_STYLE = {"color": "#6a1b9a", "weight": 2, "fill": True, "fillOpacity": 0.06}
MARKER_STYLE = {
    "color": "#ffffff",
    "weight": 2,
    "fillColor": "#1f5fae",
    "fillOpacity": 1.0,
}


def base_map(
    map_id: str, layers: list, center, zoom: int, height: str = "300px"
) -> dl.Map:
    """A grey, non-wrapping world map with the given layers on top."""
    return dl.Map(
        [
            dl.TileLayer(
                url=BASEMAP_TILE_URL,
                attribution=BASEMAP_ATTRIBUTION,
                noWrap=True,
                bounds=WORLD_BOUNDS,
                className="basemap-grey",
            ),
            *layers,
        ],
        id=map_id,
        center=center,
        zoom=zoom,
        minZoom=1,
        maxBounds=WORLD_BOUNDS,
        maxBoundsViscosity=1.0,
        worldCopyJump=False,
        # Keeps `zoom` current, so a pan-to can keep the user's zoom level.
        trackViewport=True,
        style={"height": height, "width": "100%"},
    )


def fit_box(box: tuple[float, float, float, float]) -> dict:
    """`viewport` value framing a lat/lon box."""
    lat_min, lat_max, lon_min, lon_max = box
    return {
        "bounds": [[lat_min, lon_min], [lat_max, lon_max]],
        "transition": "flyToBounds",
    }


def fit_point(lat: float, lon: float, zoom: int = 6) -> dict:
    """`viewport` value centring a point."""
    return {"center": [lat, lon], "zoom": zoom, "transition": "flyTo"}
