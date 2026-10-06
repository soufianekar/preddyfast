"""Standard regions: ECMWF and WMO verification areas, IPCC AR6 reference regions.

Every area preset in the app comes from here, so a region is always one a
reader can look up. Sources:

- ECMWF: IFS scorecard verification areas (sites.ecmwf.int/ifs/scorecards).
- WMO: Manual on the GDPFS (WMO-No. 485), standard verification areas, §6.2.
- IPCC AR6 / CMIP6: WGI reference regions v4 (Iturbide et al. 2020), bundled
  in `src/data/ar6_regions.json` by `scripts/build_regions.py`.

A region is a box (`lat_min, lat_max, lon_min, lon_max`); AR6 regions also
carry their polygon, used for outlines and for area averages. Everything is
in -180..180 longitude with `lon_min < lon_max`, so a region that crosses the
dateline cannot be represented and is left out rather than silently split.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_AR6_PATH = Path(__file__).resolve().parent.parent / "data" / "ar6_regions.json"

GLOBAL_KEY = "global"

# ECMWF scorecard areas. "N Pacific" (145°E–130°W) crosses the dateline and is
# omitted; see the module docstring.
_ECMWF = {
    "ecmwf_nh": ("N. Hemisphere extratropics", (20.0, 90.0, -180.0, 180.0)),
    "ecmwf_sh": ("S. Hemisphere extratropics", (-90.0, -20.0, -180.0, 180.0)),
    "ecmwf_tropics": ("Tropics", (-20.0, 20.0, -180.0, 180.0)),
    "ecmwf_europe": ("Europe", (35.0, 75.0, -12.5, 42.5)),
    "ecmwf_n_atlantic": ("North Atlantic", (25.0, 65.0, -70.0, -10.0)),
    "ecmwf_n_america": ("North America", (25.0, 60.0, -120.0, -75.0)),
    "ecmwf_e_asia": ("East Asia", (25.0, 60.0, 102.5, 150.0)),
    "ecmwf_arctic": ("Arctic", (60.0, 90.0, -180.0, 180.0)),
    "ecmwf_antarctic": ("Antarctic", (-90.0, -60.0, -180.0, 180.0)),
}

# WMO GDPFS areas not already covered above (its hemispheric and tropical
# bands are identical to ECMWF's).
_WMO = {
    "wmo_north_america": ("North America", (25.0, 60.0, -145.0, -50.0)),
    "wmo_europe_n_africa": ("Europe/North Africa", (25.0, 70.0, -10.0, 28.0)),
    "wmo_asia": ("Asia", (25.0, 65.0, 60.0, 145.0)),
    "wmo_australia_nz": ("Australia/New Zealand", (-55.0, -10.0, 90.0, 180.0)),
}

GROUP_LABELS = {
    "global": "Global",
    "ecmwf": "ECMWF verification areas",
    "wmo": "WMO verification areas (GDPFS)",
    "ar6_land": "IPCC AR6 land regions (CMIP6)",
    "ar6_ocean": "IPCC AR6 ocean regions (CMIP6)",
}


def _load() -> dict[str, dict]:
    regions: dict[str, dict] = {
        GLOBAL_KEY: {
            "label": "Global",
            "group": "global",
            "bbox": (-90.0, 90.0, -180.0, 180.0),
            "polygon": None,
            "source": "",
        }
    }
    for key, (label, bbox) in _ECMWF.items():
        regions[key] = {
            "label": label,
            "group": "ecmwf",
            "bbox": bbox,
            "polygon": None,
            "source": "ECMWF verification area",
        }
    for key, (label, bbox) in _WMO.items():
        regions[key] = {
            "label": label,
            "group": "wmo",
            "bbox": bbox,
            "polygon": None,
            "source": "WMO GDPFS verification area",
        }
    if _AR6_PATH.is_file():
        for region in json.loads(_AR6_PATH.read_text())["regions"]:
            vertices = np.asarray(region["vertices"], dtype=float)
            lons, lats = vertices[:, 0], vertices[:, 1]
            regions[f"ar6_{region['acronym']}"] = {
                "label": f"{region['name']} ({region['acronym']})",
                "group": (
                    "ar6_land" if region["surface"].startswith("Land") else "ar6_ocean"
                ),
                "bbox": (
                    float(lats.min()),
                    float(lats.max()),
                    float(lons.min()),
                    float(lons.max()),
                ),
                "polygon": region["vertices"],
                "source": f"IPCC AR6 reference region {region['acronym']}",
            }
    return regions


REGIONS: dict[str, dict] = _load()
REGION_BOXES: dict[str, tuple[float, float, float, float]] = {
    key: region["bbox"] for key, region in REGIONS.items()
}
REGION_LABELS: dict[str, str] = {
    key: (
        region["label"]
        if region["group"] in ("global", "ar6_land", "ar6_ocean")
        else f"{region['label']} ({region['group'].upper()})"
    )
    for key, region in REGIONS.items()
}


def grouped_options(
    keys: list[str] | None = None, extra: list[dict] | None = None
) -> list:
    """Region choices grouped by source, in the shape `dmc.Select(data=...)` takes."""
    wanted = set(keys) if keys is not None else None
    groups: list = list(extra or [])
    for group, title in GROUP_LABELS.items():
        items = [
            {"value": key, "label": region["label"]}
            for key, region in REGIONS.items()
            if region["group"] == group and (wanted is None or key in wanted)
        ]
        if items:
            groups.append({"group": title, "items": items})
    return groups


def polygon(key: str | None) -> np.ndarray | None:
    """`(n, 2)` lon/lat vertices of a region, or None for a plain box."""
    region = REGIONS.get(key or "")
    if not region or region["polygon"] is None:
        return None
    return np.asarray(region["polygon"], dtype=float)


def polygon_mask(
    key: str | None, lat: np.ndarray, lon: np.ndarray
) -> np.ndarray | None:
    """Boolean `(lat, lon)` mask of grid cells whose centres fall in the region.

    None for regions without a polygon (a box needs no mask beyond the read).
    """
    vertices = polygon(key)
    if vertices is None:
        return None
    from matplotlib.path import Path as MplPath

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    inside = MplPath(vertices).contains_points(points, radius=1e-9)
    return inside.reshape(lat_grid.shape)


# ---------------------------------------------------------------------------
# Geometry helpers shared by the map pickers (pure, no Dash)
# ---------------------------------------------------------------------------

Box = tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max


def wrap_lon(lon: float) -> float:
    """Longitude folded into -180..180 (a map panned sideways reports 200°E)."""
    wrapped = ((float(lon) + 180.0) % 360.0) - 180.0
    return 180.0 if wrapped == -180.0 and lon > 0 else wrapped


def clamp_box(lat_min: float, lat_max: float, lon_min: float, lon_max: float) -> Box:
    """Order a box's edges and keep it inside -90..90 / -180..180.

    Longitudes are clamped, not wrapped: a box dragged across the dateline
    cannot be represented as one -180..180 box, so it stops at the edge.
    """
    lat_min, lat_max = sorted((float(lat_min), float(lat_max)))
    lon_min, lon_max = sorted((float(lon_min), float(lon_max)))
    return (
        max(-90.0, lat_min),
        min(90.0, lat_max),
        max(-180.0, lon_min),
        min(180.0, lon_max),
    )


def box_from_geojson(geojson: dict | None) -> Box | None:
    """The newest drawn rectangle in a leaflet-draw FeatureCollection, clamped."""
    features = (geojson or {}).get("features") or []
    if not features:
        return None
    ring = features[-1]["geometry"]["coordinates"][0]
    lons = [point[0] for point in ring]
    lats = [point[1] for point in ring]
    return tuple(round(v, 2) for v in clamp_box(min(lats), max(lats), min(lons), max(lons)))


def _lat(value: float) -> str:
    return f"{abs(value):g}°{'N' if value >= 0 else 'S'}"


def _lon(value: float) -> str:
    return f"{abs(value):g}°{'E' if value >= 0 else 'W'}"


def format_point(lat: float, lon: float) -> str:
    """'51.51°N, 0.13°W'."""
    return f"{_lat(round(lat, 2))}, {_lon(round(lon, 2))}"


def format_box(box: Box) -> str:
    """'45–50°N, 10°W–40°E (5° × 50°)', with hemisphere letters, never signs."""
    lat_min, lat_max, lon_min, lon_max = box
    lats = (
        f"{abs(lat_min):g}–{_lat(lat_max)}"
        if (lat_min >= 0) == (lat_max >= 0)
        else f"{_lat(lat_min)}–{_lat(lat_max)}"
    )
    lons = (
        f"{abs(lon_min):g}–{_lon(lon_max)}"
        if (lon_min >= 0) == (lon_max >= 0)
        else f"{_lon(lon_min)}–{_lon(lon_max)}"
    )
    return f"{lats}, {lons} ({lat_max - lat_min:g}° × {lon_max - lon_min:g}°)"
