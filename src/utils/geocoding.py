"""Nominatim (+ optional Mapbox) geocoding. No other module imports `requests`."""

from __future__ import annotations

import time

import requests

from utils.cache import cache
from utils.settings import MAPBOX_KEY, NOMINATIM_USER_AGENT

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_REVERSE_URL = "https://nominatim.openstreetmap.org/reverse"
MAPBOX_URL = "https://api.mapbox.com/geocoding/v5/mapbox.places"


class GeocodingError(RuntimeError):
    """The geocoding service could not be reached; message is user-facing."""


_SLOT_KEY = "nominatim-slot"


def _nominatim_slot(wait_s: float = 3.0) -> None:
    """Take the site's one-per-second Nominatim slot, shared by all workers.

    Nominatim's usage policy allows one request per second from an
    application, not per visitor. A request that cannot get the slot within
    `wait_s` is refused with a message rather than queued indefinitely.
    """
    deadline = time.monotonic() + wait_s
    while not cache.add(_SLOT_KEY, 1, timeout=1):
        if time.monotonic() > deadline:
            raise GeocodingError("Place search is busy; try again in a moment.")
        time.sleep(0.25)


@cache.memoize(timeout=86400)
def geocode(query: str) -> list[dict]:
    """Geocode a free-text query to candidate locations.

    Uses Mapbox if `MAPBOX_KEY` is configured, otherwise falls back to the
    free Nominatim endpoint.

    Returns
    -------
    list[dict]
        Each dict has keys `name`, `lat`, `lon`.
    """
    try:
        if MAPBOX_KEY:
            return _mapbox_geocode(query)
        return _nominatim_geocode(query)
    except requests.RequestException as exc:
        raise GeocodingError(f"Place search is unavailable right now ({exc}).") from exc


def _nominatim_geocode(query: str) -> list[dict]:
    params = {"q": query, "format": "json", "limit": 5}
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    _nominatim_slot()
    response = requests.get(NOMINATIM_URL, params=params, headers=headers, timeout=5)
    response.raise_for_status()
    return [
        {
            "name": result["display_name"].split(",")[0],
            "detail": ", ".join(result["display_name"].split(",")[1:3]).strip(),
            "lat": float(result["lat"]),
            "lon": float(result["lon"]),
        }
        for result in response.json()
    ]


@cache.memoize(timeout=86400)
def reverse_geocode(lat: float, lon: float) -> str:
    """Resolve a map-clicked coordinate to a short place name.

    Falls back to a formatted coordinate string if Nominatim has no match
    or the request fails.
    """
    params = {"lat": lat, "lon": lon, "format": "json", "zoom": 10}
    headers = {"User-Agent": NOMINATIM_USER_AGENT}
    try:
        _nominatim_slot()
        response = requests.get(
            NOMINATIM_REVERSE_URL, params=params, headers=headers, timeout=5
        )
        response.raise_for_status()
        payload = response.json()
        address = payload.get("address", {})
        name = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("county")
            or payload.get("display_name", "").split(",")[0]
        )
        return name or f"{lat:.2f}, {lon:.2f}"
    except (requests.RequestException, GeocodingError):
        return f"{lat:.2f}, {lon:.2f}"


def _mapbox_geocode(query: str) -> list[dict]:
    url = f"{MAPBOX_URL}/{query}.json"
    params = {"access_token": MAPBOX_KEY, "limit": 5}
    response = requests.get(url, params=params, timeout=5)
    response.raise_for_status()
    return [
        {
            "name": feature["text"],
            "detail": feature.get("place_name", "").split(",", 1)[-1].strip(),
            "lat": feature["center"][1],
            "lon": feature["center"][0],
        }
        for feature in response.json().get("features", [])
    ]
