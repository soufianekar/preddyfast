"""Station network for the scorecard: selection, thinning, observations.

The fixed `SCORECARD_STATIONS` (else GHCNh stations of `SCORECARD_COUNTRY`,
thinned by farthest-point distance so dense clusters do not dominate) are
screened for completeness over the scorecard window. All times are UTC; nothing
is localized.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

from utils import observations
from utils.settings import (
    SCORECARD_COUNTRY,
    SCORECARD_DIR,
    SCORECARD_FIXED_PRECIP_COMPLETENESS,
    SCORECARD_MAX_STATIONS,
    SCORECARD_STATIONS,
    SCORECARD_VARIABLES,
)

logger = logging.getLogger(__name__)

_EARTH_RADIUS_KM = 6371.0
_SYNOPTIC_HOURS = (0, 6, 12, 18)
_TOLERANCE = pd.Timedelta("15min")
_SYNOP_TYPE = "FM-12"
_SELECT_COLUMNS = [
    "id",
    "name",
    "lat",
    "lon",
    "elev_m",
    "ok_temperature",
    "ok_precip",
    "ok_wind",
]
# scorecard variable key -> ok_* column in the selected network
_OK_COLUMN = {
    "temperature_2m": "ok_temperature",
    "precipitation_24h": "ok_precip",
    "wind_speed_10m": "ok_wind",
}


# Bump when `select_network` / `completeness` change in a way the settings
# below do not show (it is part of the rule fingerprint).
SELECTION_VERSION = 1
# SCORECARD_VARIABLES keys the selector reads to decide a station's ok_* flag.
_RULE_KEYS = (
    "obs",
    "obs_reported",
    "window_hours",
    "window_end_hour",
    "hourly_fallback",
    "completeness",
    "tolerance",
    "reference",
)


def rule_fingerprint() -> str:
    """Hash of everything that decides a station's `ok_*` flags.

    A frozen network is only valid for the rule it was selected under; the
    hash is saved beside it and compared by `scorecard.refresh`. JSON with
    sorted keys, so dict ordering never changes it.
    """
    rule = {
        "version": SELECTION_VERSION,
        "variables": {
            key: {k: spec.get(k) for k in _RULE_KEYS}
            for key, spec in SCORECARD_VARIABLES.items()
        },
        "fixed_precip_completeness": SCORECARD_FIXED_PRECIP_COMPLETENESS,
        "stations": sorted(SCORECARD_STATIONS),
    }
    text = json.dumps(rule, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def _obs_columns() -> list[str]:
    """Observation columns the scorecard uses, taken from SCORECARD_VARIABLES."""
    cols: list[str] = []
    for spec in SCORECARD_VARIABLES.values():
        for key in ("obs", "obs_reported"):
            if spec[key] and spec[key] not in cols:
                cols.append(spec[key])
    return cols


def _utc(t: pd.Timestamp | str) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


def haversine_km(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Pairwise great-circle distances in km between points (n x n matrix)."""
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    lmb = np.deg2rad(np.asarray(lon, dtype=float))
    dphi = phi[:, None] - phi[None, :]
    dlmb = lmb[:, None] - lmb[None, :]
    cos_phi = np.cos(phi)
    a = (
        np.sin(dphi / 2) ** 2
        + cos_phi[:, None] * cos_phi[None, :] * np.sin(dlmb / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def thin_by_distance(stations: pd.DataFrame, n: int) -> pd.DataFrame:
    """Greedy farthest-point selection of `n` stations by haversine distance.

    Starts from the two most distant stations, then repeatedly adds the station
    whose nearest chosen neighbour is farthest. Returns all rows when there are
    `n` or fewer. Deterministic (ties go to the first row).
    """
    if len(stations) <= n:
        return stations.reset_index(drop=True)
    if n <= 0:
        return stations.iloc[0:0].reset_index(drop=True)
    dist = haversine_km(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    if n == 1:
        return stations.iloc[[0]].reset_index(drop=True)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    chosen = [int(min(i, j)), int(max(i, j))]
    nearest = np.minimum(dist[chosen[0]], dist[chosen[1]])
    while len(chosen) < n:
        nearest[chosen] = -1.0
        k = int(np.argmax(nearest))
        chosen.append(k)
        nearest = np.minimum(nearest, dist[k])
    return stations.iloc[sorted(chosen)].reset_index(drop=True)


def _snap(obs: pd.DataFrame, column: str) -> pd.Series:
    """One value per whole UTC hour for `column`, indexed by that hour.

    The report nearest the hour within +-15 min wins; SYNOP (FM-12) beats METAR
    when both are eligible. A report more than 15 min from a full hour is not
    assigned to any hour (an hourly total closes the hour it is nearest to).
    """
    rt = obs.get(f"{column}_report_type", pd.Series("", index=obs.index))
    d = pd.DataFrame(
        {
            "t": pd.to_datetime(obs["valid_time"], utc=True),
            "v": obs[column],
            "not_synop": rt.fillna("").astype(str).str.strip() != _SYNOP_TYPE,
        }
    ).dropna(subset=["v"])
    if d.empty:
        return pd.Series(dtype=float)
    hour = d["t"].dt.round("h")
    d["hour"] = hour
    d["delta"] = (d["t"] - hour).abs()
    d = d[d["delta"] <= _TOLERANCE]
    d = d.sort_values(["not_synop", "delta"], kind="stable").drop_duplicates("hour")
    return d.set_index("hour")["v"].sort_index()


def dedupe_hourly(obs: pd.DataFrame) -> pd.DataFrame:
    """Observations on whole UTC hours, one value per hour and variable.

    Parameters
    ----------
    obs
        `fetch_station_year`-style frame: `valid_time` (UTC) plus value columns
        and optional `<column>_report_type`.

    Returns
    -------
    pd.DataFrame
        `valid_time` (the hour, UTC) plus the scorecard observation columns
        (NaN where none): instants (`temperature_2m`, `wind_speed_10m`) and
        `precipitation_1h` and `precip_total_24h` take the report nearest the
        hour within +-15 min, SYNOP preferred over METAR (accumulations cover
        the period ending at `valid_time`). Rows with no value are dropped.
    """
    cols = {}
    for col in _obs_columns():
        if col in obs:
            cols[col] = _snap(obs, col)
        else:
            cols[col] = pd.Series(dtype=float)
    out = pd.DataFrame(cols)
    out.index.name = "valid_time"
    out = out.reindex(columns=_obs_columns())
    out = out.dropna(how="all").sort_index().reset_index()
    if out.empty:
        out["valid_time"] = pd.to_datetime(out["valid_time"], utc=True)
    return out


def _synoptic_ends(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    t = pd.date_range(start.ceil("h"), end.floor("h"), freq="h", tz="UTC")
    return t[t.hour.isin(_SYNOPTIC_HOURS)]


def completeness(
    obs: pd.DataFrame, variable: str, start: pd.Timestamp, end: pd.Timestamp
) -> float:
    """Share of 00/06/12/18 UTC synoptic hours (or windows) with a QC'd value.

    A window variable with `window_end_hour` counts only windows ending then
    (24 h precipitation: the 06 UTC reports, one per day).

    `variable` is a `SCORECARD_VARIABLES` key. Instants count a value within
    +-15 min of the hour (after QC and per-hour dedupe). A precipitation window
    ending at the hour counts when a station-reported total exists there or,
    unless the variable sets `hourly_fallback` False, every hour of the window
    has a reported hourly value.
    """
    spec = SCORECARD_VARIABLES[variable]
    if spec["obs"] is None:
        raise ValueError(f"{variable} has no station observations")
    start, end = _utc(start), _utc(end)
    hours = _synoptic_ends(start, end)
    end_hour = spec.get("window_end_hour")
    if end_hour is not None:
        hours = hours[hours.hour == end_hour]
    if len(hours) == 0:
        return 0.0
    hourly = dedupe_hourly(obs)
    present = hourly.set_index("valid_time")
    if not spec["window_hours"]:
        if spec["obs"] not in present:
            return 0.0
        have = present[spec["obs"]].dropna().index
        return float(hours.isin(have).sum() / len(hours))
    w = spec["window_hours"]
    reported = (
        present[spec["obs_reported"]].dropna().index
        if spec["obs_reported"] in present
        else pd.DatetimeIndex([], tz="UTC")
    )
    hourly_idx = (
        present[spec["obs"]].dropna().index
        if spec["obs"] in present
        else pd.DatetimeIndex([], tz="UTC")
    )
    ok = hours.isin(reported)
    if not spec.get("hourly_fallback", True):
        return float(ok.sum() / len(hours))
    complete = np.ones(len(hours), dtype=bool)
    for k in range(w):
        complete &= (hours - pd.Timedelta(hours=k)).isin(hourly_idx)
    return float((ok | complete).sum() / len(hours))


def _station_frames(
    station_id: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """Raw QC'd observations over every calendar year the window touches.

    A station-year GHCNh does not hold is skipped; an unreachable or unreadable
    source raises `ObservationError`.
    """
    frames = []
    for year in range(start.year, end.year + 1):
        try:
            frames.append(observations.fetch_station_year(station_id, year))
        except observations.ObservationNotFound:
            continue
    if not frames:
        return pd.DataFrame({"valid_time": pd.DatetimeIndex([], tz="UTC")})
    return pd.concat(frames, ignore_index=True)


def network_obs(
    station_ids: list[str], start: pd.Timestamp, end: pd.Timestamp
) -> dict[str, pd.DataFrame]:
    """QC'd, deduplicated observations per station over [start, end], UTC.

    Reads every calendar year the window touches. Each frame has `valid_time`
    (whole UTC hours, tz-aware, within the window) and the columns
    `temperature_2m`, `wind_speed_10m`, `precipitation_1h` (hour ending at
    `valid_time`) and `precip_total_24h` (station-reported total over the 24 h
    ending at `valid_time`); NaN where absent. One value per station, variable
    and hour: the report nearest the hour within +-15 min, SYNOP over METAR.
    Rows with no value are dropped; a station with no data gets an empty frame.

    Raises
    ------
    ObservationError
        GHCNh is unreachable or a file cannot be read (a missing station-year
        is not an error).
    """
    start, end = _utc(start), _utc(end)
    out = {}
    for sid in station_ids:
        hourly = dedupe_hourly(_station_frames(sid, start, end))
        keep = (hourly["valid_time"] >= start) & (hourly["valid_time"] <= end)
        out[sid] = hourly[keep].reset_index(drop=True)
    return out


def select_network(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """The scorecard station network for a window.

    With `SCORECARD_STATIONS` set, exactly those stations (no country filter,
    no thinning; precipitation is screened at
    `SCORECARD_FIXED_PRECIP_COMPLETENESS`; one failing temperature is logged
    and excluded). Otherwise candidates are `SCORECARD_COUNTRY` stations with a
    file for the window's start or end year, thinned by distance to
    2 x `SCORECARD_MAX_STATIONS` before any
    download. Those are screened per variable by completeness; the stations
    passing temperature are thinned to `SCORECARD_MAX_STATIONS`.

    Returns
    -------
    pd.DataFrame
        `id, name, lat, lon, elev_m, ok_temperature, ok_precip, ok_wind`. A
        station enters a variable's card only where its `ok_*` is True. The
        network is gated on temperature: precipitation and wind cards are
        subsets of it, and `ok_temperature` is kept for a uniform schema.

    Raises
    ------
    ObservationError
        GHCNh is unreachable or unreadable (the caller shows a `dmc.Alert`);
        a transient failure never silently shrinks the network.
    """
    start, end = _utc(start), _utc(end)
    st = observations.station_list()
    fixed = bool(SCORECARD_STATIONS)
    if fixed:
        st = st[st["id"].isin(SCORECARD_STATIONS)]
        st = (
            st.set_index("id")
            .loc[[i for i in SCORECARD_STATIONS if i in set(st["id"])]]
            .reset_index()
        )
        for sid in set(SCORECARD_STATIONS) - set(st["id"]):
            logger.warning(
                "Configured station %s is not in the GHCNh station list", sid
            )
    else:
        st = st[st["country"] == SCORECARD_COUNTRY]
        years = sorted({end.year, start.year})
        st = st[
            [any(observations._year_exists(sid, y) for y in years) for sid in st["id"]]
        ]
        st = thin_by_distance(st.reset_index(drop=True), 2 * SCORECARD_MAX_STATIONS)

    rows = []
    for rec in st.itertuples(index=False):
        frame = _station_frames(rec.id, start, end)
        row = {c: getattr(rec, c) for c in ("id", "name", "lat", "lon", "elev_m")}
        for key, col in _OK_COLUMN.items():
            need = SCORECARD_VARIABLES[key]["completeness"]
            if fixed and key == "precipitation_24h":
                need = SCORECARD_FIXED_PRECIP_COMPLETENESS
            row[col] = bool(completeness(frame, key, start, end) >= need)
        rows.append(row)
    net = pd.DataFrame(rows, columns=_SELECT_COLUMNS)
    if fixed:
        for sid in net.loc[~net["ok_temperature"], "id"]:
            logger.warning(
                "Fixed station %s fails the temperature completeness screen: excluded",
                sid,
            )
    net = net[net["ok_temperature"]].reset_index(drop=True)
    if fixed:
        return net
    return thin_by_distance(net, SCORECARD_MAX_STATIONS)


def _network_path() -> Path:
    return Path(SCORECARD_DIR) / "stations.parquet"


def _window_path() -> Path:
    return _network_path().with_name("stations.window.json")


def _write_atomic(path: Path, write) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    write(tmp)
    tmp.replace(path)


def save_network(
    network: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    configured: tuple[str, ...] = (),
) -> Path:
    """Freeze the selected network to `stations.parquet`, its window beside it.

    The window goes in `stations.window.json`, so an empty network (screened,
    nobody passed) still carries it. The window is written last: a reader that
    finds it finds the parquet too. `configured` is the `SCORECARD_STATIONS`
    list the selection was made for, so a changed list can be noticed; the
    sidecar also carries `rule_hash` (`rule_fingerprint`), so a changed
    qualification rule can be.
    """
    path = _network_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_atomic(path, lambda p: network.to_parquet(p, index=False))
    window = {
        "start": _utc(start).isoformat(),
        "end": _utc(end).isoformat(),
        "configured": sorted(configured),
        "rule_hash": rule_fingerprint(),
    }
    _write_atomic(_window_path(), lambda p: p.write_text(json.dumps(window)))
    return path


def load_network() -> tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp] | None:
    """The frozen network and its (start, end) window; None if never screened.

    An empty frame (with the window) means the window was screened and no
    station passed.
    """
    path, wpath = _network_path(), _window_path()
    if not path.exists() or not wpath.exists():
        return None
    window = json.loads(wpath.read_text())
    return (
        pd.read_parquet(path),
        pd.Timestamp(window["start"]),
        pd.Timestamp(window["end"]),
    )


def load_configured() -> tuple[str, ...] | None:
    """The `SCORECARD_STATIONS` the saved network was selected for (sorted).

    Empty for the general mode or a sidecar written before the key existed;
    None when nothing is saved.
    """
    wpath = _window_path()
    if not wpath.exists():
        return None
    return tuple(sorted(json.loads(wpath.read_text()).get("configured", [])))


def load_rule_hash() -> str | None:
    """The `rule_fingerprint` the saved network was selected under.

    None when nothing is saved or the sidecar predates the key.
    """
    wpath = _window_path()
    if not wpath.exists():
        return None
    return json.loads(wpath.read_text()).get("rule_hash")
