"""Station observations for forecast evaluation: NOAA GHCNh hourly.

Files are fetched over HTTPS with `urllib` (no `requests` outside the two
modules allowed it) into a disk cache under `OBS_CACHE_DIR`, then parsed
with pandas. A station-year file is ~3 MB; past years are kept for good,
the current year is re-fetched after `OBS_CURRENT_YEAR_MAX_AGE_H`.

Quality control drops only values whose code means a failure (doc Table 3,
see `GHCNH_FAILED_*` in settings): any of GHCNh's own integrated-check
letters, or a legacy code that its source defines as suspect, erroneous or
removed. Legacy codes that mean "passed" (ISD-style 1 and 5) are kept —
dropping them would discard a third of some stations' synoptic reports.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from utils.cache import cache
from utils.settings import (
    GHCNH_BASE_URL,
    GHCNH_FAILED_INTEGRATED_QC,
    GHCNH_FAILED_LEGACY_DEFAULT,
    GHCNH_FAILED_LEGACY_QC,
    GHCNH_FIELDS,
    OBS_CACHE_DIR,
    OBS_CURRENT_YEAR_MAX_AGE_H,
    OBS_STATION_LIST_MAX_AGE_H,
    OBS_TIMEOUT_S,
)

_EARTH_RADIUS_KM = 6371.0


class ObservationError(RuntimeError):
    """An observation file could not be fetched or read; message is user-facing."""


def _cached_download(url: str, path: Path, max_age_h: float | None) -> Path:
    """Download `url` to `path` unless a copy younger than `max_age_h` exists.

    `max_age_h=None` keeps an existing copy forever (a past year's file).
    """
    if path.exists() and (
        max_age_h is None or time.time() - path.stat().st_mtime < max_age_h * 3600
    ):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=OBS_TIMEOUT_S) as resp:
            tmp.write_bytes(resp.read())
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ObservationError(f"GHCNh has no file at {url}.") from exc
        raise ObservationError(f"GHCNh download failed ({exc.code}).") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if path.exists():
            return path  # a stale copy beats no answer
        raise ObservationError(f"GHCNh is unreachable ({exc}).") from exc
    tmp.replace(path)
    return path


def station_list() -> pd.DataFrame:
    """Every GHCNh station: `id, name, lat, lon, elev_m, country`."""
    path = _cached_download(
        f"{GHCNH_BASE_URL}/doc/ghcnh-station-list.csv",
        Path(OBS_CACHE_DIR) / "ghcnh-station-list.csv",
        OBS_STATION_LIST_MAX_AGE_H,
    )
    raw = pd.read_csv(path, dtype=str)
    return pd.DataFrame(
        {
            "id": raw["GHCN_ID"].str.strip(),
            "name": raw["NAME"].str.strip(),
            "lat": pd.to_numeric(raw["LATITUDE"], errors="coerce"),
            "lon": pd.to_numeric(raw["LONGITUDE"], errors="coerce"),
            "elev_m": pd.to_numeric(raw["ELEVATION"], errors="coerce"),
            "country": raw["ISO_CODE"].fillna("").str.strip(),
        }
    ).dropna(subset=["lat", "lon"])


def nearest_stations(
    lat: float, lon: float, n: int = 5, max_km: float = 50.0
) -> pd.DataFrame:
    """The `n` stations nearest a point within `max_km`, nearest first, with `dist_km`."""
    st = station_list()
    phi1, phi2 = np.deg2rad(lat), np.deg2rad(st["lat"].to_numpy())
    dphi = phi2 - phi1
    dlmb = np.deg2rad(st["lon"].to_numpy() - lon)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlmb / 2) ** 2
    st = st.assign(dist_km=2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)))
    near = st[st["dist_km"] <= max_km].sort_values("dist_km")
    return near.head(n).reset_index(drop=True)


def _failed_qc(quality: pd.Series, source: pd.Series) -> pd.Series:
    """True where a value's quality code marks it as failed, read per source."""
    code = quality.fillna("").astype(str).str.strip()
    src = source.fillna("").astype(str).str.strip()
    integrated = code.isin(GHCNH_FAILED_INTEGRATED_QC)
    legacy = [
        c in GHCNH_FAILED_LEGACY_QC.get(s, GHCNH_FAILED_LEGACY_DEFAULT)
        for c, s in zip(code, src, strict=True)
    ]
    return integrated | pd.Series(legacy, index=code.index)


def _year_file_url(station_id: str, year: int) -> str:
    return f"{GHCNH_BASE_URL}/access/by-year/{year}/psv/GHCNh_{station_id}_{year}.psv"


@cache.memoize(timeout=86400)
def _year_exists(station_id: str, year: int) -> bool:
    """Whether GHCNh has a file for this station-year (a HEAD request, no body)."""
    req = urllib.request.Request(_year_file_url(station_id, year), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=OBS_TIMEOUT_S) as resp:
            return resp.status == 200
    except urllib.error.HTTPError:
        return False
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ObservationError(f"GHCNh is unreachable ({exc}).") from exc


# Candidates checked before giving up: the nearest stations are often
# city-network sites with no recent file.
_MAX_CANDIDATES = 25


def available_stations(
    lat: float, lon: float, year: int, n: int = 5, max_km: float = 50.0
) -> pd.DataFrame:
    """The `n` nearest stations that have a file for `year`, nearest first."""
    near = nearest_stations(lat, lon, n=_MAX_CANDIDATES, max_km=max_km)
    keep = [i for i, sid in enumerate(near["id"]) if _year_exists(sid, year)][:n]
    return near.iloc[keep].reset_index(drop=True)


@cache.memoize(timeout=3600)
def fetch_station_year(station_id: str, year: int) -> pd.DataFrame:
    """One station-year of hourly observations, QC-filtered, in display units.

    Returns
    -------
    pd.DataFrame
        `valid_time` (UTC) plus one column per `GHCNH_FIELDS` value
        (`temperature_2m`, …, `precipitation_1h`); NaN where not reported or
        dropped by quality control.
    """
    current = year >= pd.Timestamp.now(tz="UTC").year
    path = _cached_download(
        _year_file_url(station_id, year),
        Path(OBS_CACHE_DIR) / f"GHCNh_{station_id}_{year}.psv",
        OBS_CURRENT_YEAR_MAX_AGE_H if current else None,
    )
    wanted = ["DATE"]
    for field in GHCNH_FIELDS:
        wanted += [
            field,
            f"{field}_Quality_Code",
            f"{field}_Measurement_Code",
            f"{field}_Source_Code",
        ]
    try:
        raw = pd.read_csv(
            path, sep="|", usecols=lambda c: c in wanted, dtype=str, low_memory=False
        )
    except (ValueError, pd.errors.ParserError) as exc:
        raise ObservationError(
            f"Could not read GHCNh file for {station_id} ({exc})."
        ) from exc

    out = pd.DataFrame({"valid_time": pd.to_datetime(raw["DATE"], utc=True)})
    for field, name in GHCNH_FIELDS.items():
        if field not in raw:
            out[name] = np.nan
            continue
        values = pd.to_numeric(raw[field], errors="coerce")
        measure = raw.get(f"{field}_Measurement_Code", pd.Series("", index=raw.index))
        if field.startswith("precipitation"):
            # "T" is a trace: precipitation fell but below the gauge's resolution.
            values = values.mask(measure.fillna("").str.strip() == "T", 0.0)
        none = pd.Series(np.nan, index=raw.index)
        flagged = _failed_qc(
            raw.get(f"{field}_Quality_Code", none), raw.get(f"{field}_Source_Code", none)
        )
        out[name] = values.mask(flagged)
    return out.sort_values("valid_time").reset_index(drop=True)


def last_obs_time(station_id: str, column: str) -> pd.Timestamp | None:
    """The newest report of `column` at a station, naive UTC; None if never reported.

    GHCNh publishes with a lag of days, so runs newer than this have nothing
    to verify yet. Early in January the current-year file may not exist or be
    empty; the previous year's last report then stands.
    """
    now_year = pd.Timestamp.now(tz="UTC").year
    for year in (now_year, now_year - 1):
        try:
            frame = fetch_station_year(station_id, year)
        except ObservationError:
            continue
        reported = frame.loc[frame[column].notna(), "valid_time"]
        if not reported.empty:
            return pd.Timestamp(reported.max()).tz_convert("UTC").tz_localize(None)
    return None


def reported_totals(obs: pd.DataFrame, hours: int) -> pd.DataFrame:
    """Station-reported totals over the `hours` ending at each report: `valid_time, value`."""
    column = f"precip_total_{hours}h"
    if column not in obs:
        return pd.DataFrame(columns=["valid_time", "value"])
    have = obs.dropna(subset=[column])
    return pd.DataFrame(
        {"valid_time": have["valid_time"].to_numpy(), "value": have[column].to_numpy()}
    ).reset_index(drop=True)


def precip_inventory(obs: pd.DataFrame, hours: int) -> list[dict]:
    """The `hours` precipitation windows a station can supply, most numerous first.

    Two sources: windows of complete hourly reports (every hour of the window
    reported), and totals the station reports itself over that window. Each
    entry is `{source, hours, end_hour, count}`; empty when nothing over the
    period can be verified (a station reporting only 4 times a day, say).
    """
    out = []
    reported = reported_totals(obs, hours)
    if not reported.empty:
        ends = pd.to_datetime(reported["valid_time"], utc=True)
        for end_hour, count in ends.dt.hour.value_counts().items():
            out.append({"source": "reported", "hours": hours, "end_hour": int(end_hour), "count": int(count)})

    hourly = obs.dropna(subset=["precipitation_1h"])
    if not hourly.empty:
        per_hour = (
            pd.to_datetime(hourly["valid_time"], utc=True).dt.ceil("h").drop_duplicates()
        )
        have = pd.Series(1, index=pd.DatetimeIndex(per_hour).sort_values())
        grid = pd.date_range(have.index.min(), have.index.max(), freq="h")
        filled = have.reindex(grid, fill_value=0)
        complete = filled.rolling(hours).sum() == hours
        ends = complete[complete].index
        for end_hour, count in pd.Series(ends.hour).value_counts().items():
            out.append({"source": "hourly", "hours": hours, "end_hour": int(end_hour), "count": int(count)})
    return sorted(out, key=lambda e: (-e["count"], e["source"] != "reported", e["end_hour"]))
