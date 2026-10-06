"""Rebuild `src/data/natural_earth_lines.npz` from upstream Natural Earth GeoJSON.

The maps page draws its own coastlines: a Plotly heatmap lives on cartesian
axes, and only `geo` subplots have built-in coastlines — and they cannot
render a heatmap. Rather than take a cartopy dependency (and its GEOS/PROJ
stack) for two polyline layers, the line work is flattened once, here, into
NaN-separated lon/lat arrays that drop straight into a `Scattergl` trace.

Two resolutions are kept. 110 m is enough for a hemisphere or more; 50 m is
what makes a regional map legible, and its extra ~80k vertices are clipped to
the view before they reach the browser.

Run from the repository root; needs network access. The output is committed,
so this only has to be re-run to change resolutions or layers.

    python scripts/build_natural_earth.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np

BASE_URL = (
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson"
)
OUTPUT = (
    Path(__file__).resolve().parent.parent / "src" / "data" / "natural_earth_lines.npz"
)

# Layer name in the archive -> upstream file. The names are what
# `utils.figures._basemap_traces` looks up, as `{layer}_{resolution}_{lon|lat}`.
LAYERS: dict[str, str] = {
    "coastline_110": "ne_110m_coastline.geojson",
    "borders_110": "ne_110m_admin_0_boundary_lines_land.geojson",
    "coastline_50": "ne_50m_coastline.geojson",
    "borders_50": "ne_50m_admin_0_boundary_lines_land.geojson",
}


def _rings(geometry: dict) -> list[list[list[float]]]:
    """Every coordinate ring in a line geometry, ignoring anything non-linear."""
    kind = geometry.get("type")
    if kind == "LineString":
        return [geometry["coordinates"]]
    if kind == "MultiLineString":
        return geometry["coordinates"]
    return []


def _flatten(collection: dict) -> tuple[np.ndarray, np.ndarray]:
    """Flatten a FeatureCollection into NaN-separated lon/lat arrays.

    One NaN pair after each ring keeps the polylines distinct inside a single
    trace, which is what makes the whole layer one `Scattergl` rather than
    hundreds.
    """
    lon: list[float] = []
    lat: list[float] = []

    for feature in collection["features"]:
        for ring in _rings(feature["geometry"]):
            lon.extend([point[0] for point in ring] + [np.nan])
            lat.extend([point[1] for point in ring] + [np.nan])

    return np.asarray(lon, dtype=np.float32), np.asarray(lat, dtype=np.float32)


def main() -> None:
    arrays: dict[str, np.ndarray] = {}

    for layer, filename in LAYERS.items():
        url = f"{BASE_URL}/{filename}"
        print(f"fetching {url}")
        with urllib.request.urlopen(url, timeout=120) as response:
            collection = json.load(response)

        lon, lat = _flatten(collection)
        arrays[f"{layer}_lon"] = lon
        arrays[f"{layer}_lat"] = lat
        print(f"  {layer}: {lon.size} vertices")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **arrays)
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
