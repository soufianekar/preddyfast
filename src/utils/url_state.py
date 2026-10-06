"""Pure helpers for turning URL query params into control values."""

from __future__ import annotations

from collections.abc import Collection, Mapping

_BBOX_EDGES = ("lat_min", "lat_max", "lon_min", "lon_max")


def query_choice(
    params: Mapping[str, str], key: str, valid: Collection[str], default: str | None
) -> str | None:
    """Return `params[key]` if it is one of `valid`, else `default`."""
    value = params.get(key)
    return value if value in valid else default


def query_list(
    params: Mapping[str, str],
    key: str,
    valid: Collection[str],
    default: list[str],
) -> list[str]:
    """Return the comma-separated `params[key]` filtered to `valid`, else `default`."""
    items = [v for v in (params.get(key) or "").split(",") if v in valid]
    return items or default


def parse_location(params: Mapping[str, str]) -> dict | None:
    """Build a `location-store` payload from `lat`/`lon`/`name`, or None."""
    try:
        return {
            "lat": float(params["lat"]),
            "lon": float(params["lon"]),
            "name": params["name"],
        }
    except (KeyError, ValueError):
        return None


def parse_bbox(params: Mapping[str, str]) -> dict | None:
    """Build a `bbox-store` payload from the four edge params, or None."""
    try:
        return {edge: float(params[edge]) for edge in _BBOX_EDGES}
    except (KeyError, ValueError):
        return None
