"""Gridded ensemble reads from the dynamical.org Icechunk store on AWS Open Data.

Scope is deliberately narrow: this module serves the *regional* page only —
small lat/lon boxes reduced to a per-member spatial mean. Point features stay
on Open-Meteo (see `docs/icechunk-handoff.md` §1). Never read a point series
from here.

The public functions return tidy DataFrames in the same shape the rest of the
app already consumes, so `figures.py` and `climatology.py` need no knowledge
of xarray or Zarr.

Two constraints shape the code:

* **Fork safety.** Gunicorn runs with `--preload`, and Icechunk/object-store
  session handles and their connection pools do not reliably survive `fork()`.
  The repository handle is therefore opened lazily on first use and tagged
  with the PID that opened it; a handle inherited across a fork is discarded
  and reopened in the child.
* **The cache boundary sits after the reduction.** `@cache.memoize()` cannot
  serialise an xarray object, so every cached function here returns a
  DataFrame (plus a plain-dict metadata payload), never a Dataset or DataArray.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import pickle
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from utils import status
from utils.cache import cache
from utils.climatology import accumulate_over_window, daily_extreme
from utils.regions import REGIONS, polygon, polygon_mask
from utils.settings import (
    ANALYSIS_STORES,
    BBOX_MAX_SPAN_DEG,
    BBOX_MIN_SPAN_DEG,
    DETERMINISTIC_MODELS,
    DYNAMICAL_STAC_TIMEOUT_S,
    DYNAMICAL_STAC_URL,
    ENSEMBLE_MODELS,
    GRID_RESOLUTION_DEG,
    GRID_STORE_DEFAULT,
    GRID_STORES,
    GRID_VARIABLES,
    POINT_CACHE_DIR,
    READ_BUDGET_MB,
    REGION_MAX_CELLS,
    VARIABLE_FAMILIES,
)
from utils.text import in_sentence
from utils.variable_family import family_of


class GridDataError(RuntimeError):
    """Raised when the grid store cannot satisfy a request.

    Carries a message already suitable for display in a `dmc.Alert`.
    """


_log = logging.getLogger(__name__)

# Lazily-opened, per-process store handles, keyed by `GRID_STORES` key.
# `_dataset_pid` records which process opened them so a handle inherited
# through gunicorn's fork is never reused. Guarded by a lock because Dash
# serves callbacks from a thread pool.
_datasets: dict[str, object] = {}
_dataset_pid: int | None = None
_dataset_lock = threading.Lock()


@functools.cache
def store_href(store: str) -> tuple[str, str]:
    """The store's Icechunk href and AWS region, from the dynamical.org catalogue.

    Looked up once per process. Plain HTTP (stdlib), so it is fork-safe; no
    `requests` here by project rule. When the catalogue is unreachable or no
    longer lists the store, the pinned `fallback_href` is used and a warning
    logged — the store may still be there under the old version.

    Returns
    -------
    tuple[str, str]
        `(href, region)`, href as `s3://bucket/prefix`.
    """
    if store not in GRID_STORES:
        raise GridDataError(f"Unknown gridded store '{store}'.")
    spec = GRID_STORES[store]
    url = f"{DYNAMICAL_STAC_URL}/{spec['stac_id']}/collection.json"
    try:
        with urllib.request.urlopen(url, timeout=DYNAMICAL_STAC_TIMEOUT_S) as resp:
            asset = json.load(resp)["assets"]["icechunk"]
        href = asset["href"]
        region = (
            asset.get("xarray:storage_options", {})
            .get("client_kwargs", {})
            .get("region_name", spec["region"])
        )
    except (urllib.error.URLError, OSError, ValueError, KeyError) as exc:
        _log.warning(
            "dynamical.org catalogue lookup for %s failed (%s); using pinned %s",
            spec["stac_id"],
            exc,
            spec["fallback_href"],
        )
        return spec["fallback_href"], spec["region"]
    if href.rstrip("/") != spec["fallback_href"].rstrip("/"):
        _log.warning(
            "dynamical.org catalogue moved %s to %s (pinned fallback is %s)",
            spec["stac_id"],
            href,
            spec["fallback_href"],
        )
    return href, region


def _split_s3(href: str) -> tuple[str, str]:
    """`s3://bucket/prefix/` -> `(bucket, prefix)`."""
    if not href.startswith("s3://"):
        raise GridDataError(f"Unsupported store address '{href}'.")
    bucket, _, prefix = href[len("s3://") :].partition("/")
    return bucket, prefix.rstrip("/")


def _open_dataset(store: str = GRID_STORE_DEFAULT):
    """Open one Icechunk store, reopening if inherited across a fork.

    Imports of `icechunk`/`xarray` are function-local: they are heavy, and
    keeping them out of module scope means importing this module (as `app.py`
    does at startup, before the fork) never opens a network session.

    Returns
    -------
    xarray.Dataset
        Lazily-backed dataset for the store's catalogued version.
    """
    global _dataset_pid

    spec = GRID_STORES.get(store, {})
    if spec.get("source_store"):
        # A one-member view of another store (IFS Control = IFS ENS member 0).
        # The member axis is kept so chunk geometry stays the source's: the
        # view downloads exactly the chunks the source would. Built once per
        # process: `isel` on a whole dataset makes a graph for every variable.
        base = _open_dataset(spec["source_store"])
        with _dataset_lock:
            if store not in _datasets:
                _datasets[store] = base.isel(ensemble_member=[spec["member"]])
            return _datasets[store]

    with _dataset_lock:
        if _dataset_pid != os.getpid():
            _datasets.clear()
            _dataset_pid = os.getpid()
        if store in _datasets:
            return _datasets[store]

        import icechunk
        import xarray as xr

        href, region = store_href(store)
        bucket, prefix = _split_s3(href)
        try:
            storage = icechunk.s3_storage(
                bucket=bucket, prefix=prefix, region=region, anonymous=True
            )
            repo = icechunk.Repository.open(storage=storage)
            session = repo.readonly_session("main")
            ds = xr.open_zarr(session.store, consolidated=False)
        except (OSError, ValueError, RuntimeError) as exc:
            raise GridDataError(
                f"Could not open the gridded ensemble store ({exc})."
            ) from exc

        _datasets[store] = ds
        return ds


def _as_ensemble(ds):
    """A deterministic array (or dataset) as a one-member ensemble.

    Every reader indexes members; one member keeps a single code path. Applied
    to the selected array, not on open: expanding a whole dataset builds a
    graph for every variable (~2 s for AIFS Single). The figures check
    `GRID_STORES[store]["kind"]` and never draw spread for it.
    """
    if "ensemble_member" in ds.dims:
        return ds
    return ds.expand_dims(ensemble_member=[0])


def physical_stores(stores: tuple[str, ...]) -> tuple[str, ...]:
    """The stores actually read, in order: a view counts as its source, once."""
    out: list[str] = []
    for store in stores:
        source = GRID_STORES[store].get("source_store", store)
        if source not in out:
            out.append(source)
    return tuple(out)


def _member_subset(df: pd.DataFrame, meta: dict, store: str) -> tuple[pd.DataFrame, dict]:
    """A view store's series from its source's: its one member, its own labels.

    The source's frame and meta come from caches and are never mutated.
    """
    member = GRID_STORES[store]["member"]
    return df[df["member"] == member].reset_index(drop=True), {
        **meta,
        "n_members": 1,
        "store": store,
        "model_label": GRID_STORES[store]["model_label"],
    }


def _cube_member(cube: dict, member: int, store: str) -> dict:
    """A view store's frame cube: one member sliced out of the source's cube."""
    index = cube["members"].index(member)
    return {
        **cube,
        "cube": np.ascontiguousarray(cube["cube"][:, index : index + 1]),
        "members": [member],
        "store": store,
    }


def _chunk_dims(dims: tuple, chunks: tuple, deterministic: bool) -> dict:
    """Pair an encoding's chunk sizes with the dimensions they were written for.

    The encoding describes the stored array; a member axis added on open is
    not in it.
    """
    if len(dims) == len(chunks):
        # Every dimension was stored, including a member axis kept by a view.
        return dict(zip(dims, chunks, strict=True))
    stored = [d for d in dims if not (deterministic and d == "ensemble_member")]
    return dict(zip(stored, chunks, strict=True))


def _check_step_type(ds, variable: str) -> str:
    """The store's `step_type` for a variable, refused unless the app accepts it.

    The unit conversion and any accumulation depend on it: a mean rate, a
    total since init and a step maximum all plot plausibly when mistaken for
    one another. Variables that declare nothing accept only `instant`. The
    same variable may legitimately differ between stores (cloud cover is
    `instant` in IFS, `avg` in GEFS), so the actual type is returned for the
    caller to convert and label by.
    """
    accepted = GRID_VARIABLES[variable].get("accepted_step_types", ("instant",))
    actual = ds[variable].attrs.get("step_type")
    if actual not in accepted:
        raise GridDataError(
            f"'{variable}' is archived with step_type '{actual}', but the app "
            f"accepts {', '.join(accepted)}. Refusing to convert it."
        )
    return actual


def _window_phrase(step_type: str | None) -> str | None:
    """How a step type is worded in a subtitle; `None` for instantaneous fields."""
    return {
        "avg": "mean over each preceding step",
        "max": "maximum over each preceding step",
        "accum": "total since the previous step",
    }.get(step_type or "instant")


def _require_variable(ds, variable: str, store: str) -> str:
    """Refuse a variable the store does not hold or holds as another step type.

    Returns the store's step type for it.
    """
    if variable not in ds.data_vars:
        raise GridDataError(
            f"{GRID_STORES[store]['label']} does not archive "
            f"{in_sentence(GRID_VARIABLES[variable]['label'])}. Choose another variable "
            "or model."
        )
    return _check_step_type(ds, variable)


def _source(store: str) -> str:
    """The store whose data a store reads: itself, or a view's source."""
    return GRID_STORES[store].get("source_store", store)


def store_variables(store: str = GRID_STORE_DEFAULT) -> list[str]:
    """`GRID_VARIABLES` keys present in a store, in settings order."""
    return _store_variables(_source(store))


def _store_variables(store: str) -> list[str]:
    ds = _open_dataset(store)
    return [v for v in GRID_VARIABLES if v in ds.data_vars]


def _require_init(store: str, init: pd.Timestamp) -> None:
    """Refuse an init the store does not hold, naming the store's own cycle.

    A 06 UTC run asked of IFS ENS (00 UTC only) would otherwise surface as
    a bare selection error.
    """
    inits = available_inits(store)
    if init not in set(inits):
        hours = "/".join(f"{h:02d}" for h in sorted({t.hour for t in inits}))
        raise GridDataError(
            f"{GRID_STORES[store]['label']} has no run at {init:%Y-%m-%d %H:%M} UTC "
            f"(it holds {hours} UTC runs, {inits[0]:%Y-%m-%d} to {inits[-1]:%Y-%m-%d})."
        )


def _describe_steps(lead_hours: np.ndarray) -> str:
    """Lead-step geometry in words: '3 h to T+144, 6 h after' or '6 h'."""
    steps = np.diff(lead_hours)
    if steps.size == 0:
        return "single step"
    parts, start = [], 0
    for i in range(1, steps.size + 1):
        if i == steps.size or steps[i] != steps[start]:
            parts.append((steps[start], lead_hours[i]))
            start = i
    if len(parts) == 1:
        return f"{parts[0][0]:g} h"
    head = [f"{step:g} h to T+{end:g}" for step, end in parts[:-1]]
    return ", ".join(head) + f", {parts[-1][0]:g} h after"


def store_steps(store: str = GRID_STORE_DEFAULT) -> str:
    """A store's lead-step geometry in words, read from its `lead_time` axis."""
    ds = _open_dataset(store)
    lead = pd.to_timedelta(ds["lead_time"].values).total_seconds() / 3600.0
    return _describe_steps(np.asarray(lead))


def step_window(store: str, variable: str) -> str | None:
    """Subtitle clause naming a step-averaged variable's window, from the store.

    `None` for an instantaneous field: it is valid at its lead and saying
    otherwise would be noise.
    """
    ds = _open_dataset(store)
    if variable not in ds.data_vars:
        return None
    phrase = _window_phrase(ds[variable].attrs.get("step_type"))
    return f"{phrase} ({store_steps(store)})" if phrase else None


def _buildable(lead_hours: np.ndarray, period: float) -> bool:
    """Whether some lead `L` has `L - period` as a lead too (a complete window)."""
    leads = set(np.asarray(lead_hours, dtype=float).tolist())
    return any(h - period in leads for h in leads if h >= period)


def buildable_periods(stores: tuple[str, ...], periods) -> list:
    """The accumulation periods every store's steps can build."""
    return [
        p for p in periods if all(_buildable(store_geometry(s)["lead_hours"], p) for s in stores)
    ]


def store_horizon(store: str) -> float:
    """A store's last lead, in hours."""
    return float(store_geometry(store)["lead_hours"].max())


def _geometry_from_coords(
    lat: np.ndarray,
    lon: np.ndarray,
    lead_hours: np.ndarray,
    chunks: dict,
    n_members: int,
) -> dict:
    """A store's grid, domain, leads and chunking, from its coordinates."""
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    res = float(round(abs(lat[1] - lat[0]), 6))
    lat_range = (float(lat.min()), float(lat.max()))
    lon_range = (float(lon.min()), float(lon.max()))
    is_global = lat_range[1] - lat_range[0] >= 180.0 - res and lon_range[1] - lon_range[0] >= 360.0 - 2 * res
    return {
        "res_deg": res,
        "lat_range": lat_range,
        "lon_range": lon_range,
        "lat_descending": bool(lat[0] > lat[-1]),
        "global": bool(is_global),
        "chunks": dict(chunks),
        "lead_hours": np.asarray(lead_hours, dtype=float),
        "n_members": int(n_members),
    }


_PROBE_FALLBACKS = ("temperature_2m", "precipitation_surface")


def _probe_variable(ds) -> str:
    """The first of `_PROBE_FALLBACKS` the store holds."""
    try:
        return next(v for v in _PROBE_FALLBACKS if v in ds.data_vars)
    except StopIteration:
        raise GridDataError("The store holds none of the variables used to probe it.") from None


def _res(store: str) -> float:
    """A store's grid spacing in degrees (declared; checked in `store_geometry`)."""
    return GRID_STORES[store]["res_deg"]


@functools.cache
def store_geometry(store: str) -> dict:
    """Grid spacing, extent, leads, member count and chunk shape of a store.

    Read from the store once per process (coordinates and encoding only).
    """
    ds = _open_dataset(store)
    da = ds[_probe_variable(ds)]
    deterministic = GRID_STORES[store]["kind"] == "deterministic"
    if da.encoding.get("chunks"):
        chunks = _chunk_dims(da.dims, da.encoding["chunks"], deterministic)
    else:
        chunks = dict(zip(da.dims, da.data.chunksize, strict=True))
    lead = pd.to_timedelta(ds["lead_time"].values).total_seconds() / 3600.0
    geometry = _geometry_from_coords(
        ds["latitude"].values,
        ds["longitude"].values,
        np.asarray(lead),
        chunks,
        1 if deterministic else ds.sizes.get("ensemble_member", 1),
    )
    if abs(geometry["res_deg"] - _res(store)) > 1e-6:
        _log.warning(
            "%s grid is %g°, settings declare %g° — update GRID_STORES",
            store,
            geometry["res_deg"],
            _res(store),
        )
    return geometry


def _format_extent(lat_min, lat_max, lon_min, lon_max) -> str:
    def lat(v):
        return f"{abs(v):g}°{'N' if v >= 0 else 'S'}"

    def lon(v):
        return f"{abs(v):g}°{'E' if v >= 0 else 'W'}"

    return f"{lat(lat_min)}–{lat(lat_max)}, {lon(lon_min)}–{lon(lon_max)}"


def clip_to_store(store: str, bounds: tuple) -> tuple[tuple, str | None]:
    """An area clipped to a limited-area store's domain, with a note when clipped.

    A global store passes everything through. An area entirely outside the
    domain is refused; one partly outside is cut to the overlap, and the note
    says so — the subtitle and an alert must state the extent actually read.
    """
    geometry = store_geometry(store)
    if geometry["global"]:
        return tuple(bounds), None
    lat_min, lat_max, lon_min, lon_max = bounds
    (d_lat0, d_lat1), (d_lon0, d_lon1) = geometry["lat_range"], geometry["lon_range"]
    label = GRID_STORES[store]["label"]
    clipped = (max(lat_min, d_lat0), min(lat_max, d_lat1), max(lon_min, d_lon0), min(lon_max, d_lon1))
    if clipped[0] > clipped[1] or clipped[2] > clipped[3]:
        raise GridDataError(
            f"{label} covers {_format_extent(d_lat0, d_lat1, d_lon0, d_lon1)}; "
            "this area is outside it."
        )
    if clipped == tuple(bounds):
        return tuple(bounds), None
    return clipped, f"{label}: clipped to its domain, {_format_extent(*clipped)}"


def _clamp_span(low: float, high: float, limit: float) -> tuple[float, float]:
    """Shrink an interval about its centre until it fits within `limit`."""
    if high - low <= limit:
        return low, high
    centre = (low + high) / 2.0
    return centre - limit / 2.0, centre + limit / 2.0


def snap_bbox(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    max_span: float = BBOX_MAX_SPAN_DEG,
    res: float = GRID_RESOLUTION_DEG,
) -> tuple[float, float, float, float]:
    """Snap a drawn rectangle to the native grid and clamp it to the size guard.

    The returned bounds are inclusive edges aligned to `GRID_RESOLUTION_DEG`,
    so the cell count implied by the box is exactly what gets read — no
    silent half-cells, which matters because the extent is quoted on the plot.
    Snapping rounds *outward*, so the box read is never smaller than the box
    drawn.

    Edges are accepted in either order, so a rectangle dragged bottom-right to
    top-left behaves the same as one dragged top-left to bottom-right.

    Parameters
    ----------
    lat_min, lat_max : float
        South and north edges in degrees.
    lon_min, lon_max : float
        West and east edges in degrees, in the store's -180..180 convention.

    Returns
    -------
    tuple[float, float, float, float]
        `(lat_min, lat_max, lon_min, lon_max)`, grid-aligned and clamped.
    """
    lat_min, lat_max = sorted((float(lat_min), float(lat_max)))
    lon_min, lon_max = sorted((float(lon_min), float(lon_max)))

    # A degenerate drag (a click, or a zero-width edge) still has to produce a
    # readable box, so widen anything under one cell to the minimum span.
    if lat_max - lat_min < BBOX_MIN_SPAN_DEG:
        lat_max = lat_min + BBOX_MIN_SPAN_DEG
    if lon_max - lon_min < BBOX_MIN_SPAN_DEG:
        lon_max = lon_min + BBOX_MIN_SPAN_DEG

    lat_min, lat_max = _clamp_span(lat_min, lat_max, max_span)
    lon_min, lon_max = _clamp_span(lon_min, lon_max, max_span)

    # Round before floor/ceil so 45.0 / 1.5 = 29.999… does not drop a cell.
    return (
        float(max(-90.0, np.floor(round(lat_min / res, 9)) * res)),
        float(min(90.0, np.ceil(round(lat_max / res, 9)) * res)),
        float(max(-180.0, np.floor(round(lon_min / res, 9)) * res)),
        float(min(180.0 - res, np.ceil(round(lon_max / res, 9)) * res)),
    )


# How many inits to walk back from the newest before giving up. The newest
# run in the store is often still being written, and the store's
# `ingested_forecast_length` coordinate is NaT for *every* init — it is never
# populated, so it cannot be used to detect that. Completeness is therefore
# established from the data actually read, not from metadata.
_MAX_INIT_FALLBACK = 4


def available_inits(store: str = GRID_STORE_DEFAULT) -> list[pd.Timestamp]:
    """All init times in a store, oldest first (a view has its source's).

    Metadata only — the coordinate is small and this does not touch a data
    chunk.
    """
    return _store_inits(_source(store))


@cache.memoize(timeout=1800)
def _store_inits(store: str) -> list[pd.Timestamp]:
    ds = _open_dataset(store)
    return [pd.Timestamp(t) for t in pd.to_datetime(ds["init_time"].values)]


def _latest_candidates(store: str) -> list[pd.Timestamp]:
    """The newest few inits, newest first, minus those still being written.

    Four-a-day stores (GFS, AIFS Single, ICON-EU) usually have their newest
    run in flight; skipping it from the calendar's one-chunk probe saves a
    full read of the area just to find it incomplete. The data read still
    decides, so a run the probe missed is still walked past.
    """
    newest = available_inits(store)[-_MAX_INIT_FALLBACK:][::-1]
    try:
        unfinished = {pd.Timestamp(t) for t in run_calendar(store)["unfinished"]}
    except GridDataError:
        unfinished = set()
    return [t for t in newest if t not in unfinished] or newest


def latest_init(store: str = GRID_STORE_DEFAULT) -> pd.Timestamp:
    """Newest init time present in a store.

    Note this run may still be partially written; `fetch_bbox_ensemble` falls
    back to older inits when the data confirms that.
    """
    inits = available_inits(store)
    if not inits:
        raise GridDataError("The gridded ensemble store contains no forecast runs.")
    return inits[-1]


# A finished run is told from one still being written by one chunk at the
# final lead; the first tile is the cheapest address. The variable is the
# store's first `_PROBE_FALLBACKS` entry.


def _run_finished(ds, init: pd.Timestamp) -> bool:
    """True when the final lead of one tile of `init` has landed (one chunk read)."""
    da = ds[_probe_variable(ds)].sel(init_time=init)
    tile = {dim: slice(0, 1) for dim in ("latitude", "longitude") if dim in da.dims}
    try:
        member = {"ensemble_member": 0} if "ensemble_member" in da.dims else {}
        probe = da.isel(lead_time=-1, **member, **tile).compute()
    except (OSError, ValueError, RuntimeError):
        return False
    return not bool(probe.isnull().all())


def _merge_calendars(cals: list[dict]) -> dict:
    """Runs every calendar has: hours intersect per day, unfinished runs union.

    Same shape as `run_calendar`, so the picker draws a merged calendar
    exactly like a single store's. Comparing models only makes sense from the
    same init, so a day or hour one store lacks is unavailable for all.
    """
    days = None
    for cal in cals:
        mine = {d: set(h) for d, h in cal["days"].items()}
        if days is None:
            days = mine
        else:
            days = {d: days[d] & mine[d] for d in days.keys() & mine.keys()}
            days = {d: h for d, h in days.items() if h}
    days = {d: sorted(h) for d, h in sorted((days or {}).items())}
    hours = sorted(set.intersection(*(set(c["hours"]) for c in cals)))
    if days:
        first, last = min(days), max(days)
    else:
        first, last = cals[0]["first"], cals[0]["last"]
    span = pd.date_range(first, last, freq="D").strftime("%Y-%m-%d")
    return {
        "first": first,
        "last": last,
        "hours": hours,
        "days": days,
        "missing": [d for d in span if d not in days],
        "unfinished": sorted({t for c in cals for t in c["unfinished"]}),
    }


@cache.memoize(timeout=1800)
def common_run_calendar(stores: tuple[str, ...]) -> dict:
    """`run_calendar` over several stores: only the inits they all hold."""
    return _merge_calendars([run_calendar(s) for s in stores])


def run_calendar(store: str = GRID_STORE_DEFAULT) -> dict:
    """Which runs a store can serve, shaped for the run picker (see `_store_calendar`).

    A view (IFS Control) serves exactly its source's runs: same calendar, same
    cache entry, no probes of its own.
    """
    return _store_calendar(_source(store))


@cache.memoize(timeout=1800)
def _store_calendar(store: str) -> dict:
    """Which runs a store can serve, shaped for the run picker.

    The init hours are those the store actually holds (IFS ENS: 00 UTC only;
    AIFS ENS: four a day), never assumed. The newest few runs are probed for
    completeness — `ingested_forecast_length` is never populated — and a run
    still being written is listed as unfinished so the picker can disable it.
    Older runs are taken as complete; the fetch still checks the data it reads.

    Returns
    -------
    dict
        JSON-able: `first` / `last` (ISO dates), `hours` (cycle hours seen),
        `days` (ISO date -> available hours), `missing` (dates in range with
        no servable run), `unfinished` (ISO init times).
    """
    inits = available_inits(store)
    if not inits:
        raise GridDataError("The gridded ensemble store contains no forecast runs.")

    ds = _open_dataset(store)
    unfinished = []
    for init in inits[-_MAX_INIT_FALLBACK:][::-1]:
        if _run_finished(ds, init):
            break
        unfinished.append(init)

    days: dict[str, list[int]] = {}
    for init in inits:
        if init not in unfinished:
            days.setdefault(f"{init:%Y-%m-%d}", []).append(int(init.hour))

    span = pd.date_range(inits[0].normalize(), inits[-1].normalize(), freq="D")
    return {
        "first": f"{inits[0]:%Y-%m-%d}",
        "last": f"{inits[-1]:%Y-%m-%d}",
        "hours": sorted({int(t.hour) for t in inits}),
        "days": days,
        "missing": [f"{d:%Y-%m-%d}" for d in span if f"{d:%Y-%m-%d}" not in days],
        "unfinished": [f"{t:%Y-%m-%dT%H:%M}" for t in unfinished],
    }


def _select_box(ds, variable: str, init: pd.Timestamp, bounds: tuple):
    """Lazily select one init's lat/lon box for a variable.

    Returns the un-reduced `(ensemble_member, lead_time, latitude, longitude)`
    DataArray; nothing is read until the caller computes.
    """
    lat_min, lat_max, lon_min, lon_max = bounds

    try:
        # Latitude is stored descending (90 -> -90), so the slice runs max->min.
        da = ds[variable].sel(
            init_time=init,
            latitude=slice(lat_max, lat_min),
            longitude=slice(lon_min, lon_max),
        )
    except (KeyError, ValueError) as exc:
        raise GridDataError(f"Could not select that box or init time ({exc}).") from exc

    if da.sizes.get("latitude", 0) == 0 or da.sizes.get("longitude", 0) == 0:
        raise GridDataError("The selected box contains no grid cells.")

    return _as_ensemble(da)


def _read_regional_mean(
    ds, variable: str, init: pd.Timestamp, bounds: tuple, region: str | None = None
):
    """Read one init's box and reduce it to a `cos(lat)`-weighted mean per member.

    With a `region`, cells whose centres fall outside its polygon get zero
    weight, so an AR6 region is averaged over its own shape, not its box.

    Returns
    -------
    tuple
        `(reduced, selected)` — the computed `(ensemble_member, lead_time)`
        DataArray, and the pre-reduction selection (kept for its cell counts).
    """
    da = _select_box(ds, variable, init, bounds)

    # cos(lat) weighting: 0.25 deg cells shrink in area towards the poles, so
    # an unweighted mean would over-count high-latitude rows.
    weights = np.cos(np.deg2rad(da["latitude"]))
    mask = polygon_mask(region, da["latitude"].values, da["longitude"].values)
    if mask is not None:
        import xarray as xr  # function-local, as in `_open_dataset`

        if not mask.any():
            raise GridDataError("No grid cell centre falls inside that region.")
        weights = weights * xr.DataArray(
            mask.astype(float),
            coords={"latitude": da["latitude"], "longitude": da["longitude"]},
            dims=("latitude", "longitude"),
        )
    reduced = da.weighted(weights).mean(("latitude", "longitude"))

    try:
        return reduced.compute(), da
    except (OSError, ValueError, RuntimeError) as exc:
        raise GridDataError(f"Failed reading the gridded ensemble ({exc}).") from exc


def _is_complete(regional) -> bool:
    """True when a run has been written all the way out to its final lead.

    A partially-ingested run reduces to NaN at the leads whose chunks have
    not landed yet. `.weighted().mean()` propagates that: if every cell in
    the box is NaN the weighted mean is NaN, so checking the final lead is
    enough to reject a run still in flight.
    """
    if regional.isnull().all():
        return False
    return not bool(regional.isel(lead_time=-1).isnull().all())


@cache.memoize(timeout=1800)
def fetch_bbox_ensemble(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    variable: str,
    init_time: str | None = None,
    accum_hours: int | None = None,
    store: str = GRID_STORE_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """Area-averaged ensemble series for one variable over a small lat/lon box.

    Space is reduced *first, per member* — a `cos(lat)`-weighted mean over the
    box — leaving 51 physically self-consistent realisations of a regional
    average. Percentiles are taken across members afterwards, by
    `climatology.compute_member_pct`. The reverse order (per-gridpoint
    percentiles, then averaging) describes a patchwork that corresponds to no
    member and is systematically wider at the tails; see
    `docs/icechunk-handoff.md` §11.

    Cached for 30 minutes, matching `fetch_ensemble`: the store gains new
    runs through the day and a stale init should not persist.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max : float
        Rectangle edges as drawn, rounded to 2 decimal places by the caller.
        Snapped to the grid and clamped to the size guard by `snap_bbox`.
    variable : str
        Key into `utils.settings.GRID_VARIABLES` (a store variable name).
    init_time : str or None
        ISO init timestamp. `None` selects the latest complete run.
    accum_hours : int or None
        Accumulation window in hours for an accumulated variable. `None`
        leaves the series as archived — per-step mean rates. Leads that
        cannot carry a complete window are dropped, so the series starts at
        T+`accum_hours`.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The frame has columns `valid_time`, `lead_hours`, `member`, `value`,
        `variable`, one row per (member, lead). The dict carries everything
        the plot subtitle must state: `init_time`, `n_members`, `n_lat`,
        `n_lon`, the snapped `bbox`, the resolved unit, and `accum_hours`.

    Notes
    -----
    Accumulation is applied *after* the spatial reduction. Both operations are
    linear, so the order does not change the answer, and doing it second keeps
    the expensive read untouched.
    """
    bounds = snap_bbox(lat_min, lat_max, lon_min, lon_max, res=_res(store))
    return _ensemble_series(bounds, variable, init_time, accum_hours, store=store)


@cache.memoize(timeout=1800)
def fetch_region_ensemble(
    region: str,
    variable: str,
    init_time: str | None = None,
    accum_hours: int | None = None,
    store: str = GRID_STORE_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """Ensemble series averaged over a standard region (see `utils.regions`).

    Same as `fetch_bbox_ensemble`, but the box is the region's own, exempt
    from the 10° drawing limit, and cells outside an AR6 polygon get zero
    weight. Large regions are refused above `REGION_MAX_CELLS`: the read time
    grows with the chunks touched (~18 s cold for W. & C. Europe).
    """
    if region not in REGIONS:
        raise GridDataError(f"Unknown region '{region}'.")
    res = _res(store)
    bounds = snap_bbox(*REGIONS[region]["bbox"], max_span=360.0, res=res)
    n_cells = round((bounds[1] - bounds[0]) / res + 1) * round(
        (bounds[3] - bounds[2]) / res + 1
    )
    if n_cells > REGION_MAX_CELLS:
        raise GridDataError(
            f"{REGIONS[region]['label']} spans {n_cells:,} grid cells — too large "
            f"to read interactively (limit {REGION_MAX_CELLS:,}). Pick a smaller "
            "region."
        )
    return _ensemble_series(bounds, variable, init_time, accum_hours, region, store)


@cache.memoize(timeout=1800)
def fetch_point_ensemble(
    lat: float,
    lon: float,
    variable: str,
    init_time: str | None = None,
    accum_hours: int | None = None,
    store: str = GRID_STORE_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """Ensemble series at the grid cell nearest a point, for any archived run.

    Used by the point meteogram when a past run is chosen: Open-Meteo's
    ensemble API only serves the latest run. The value is the nearest 0.25°
    cell, not an interpolation, and the caller says so.
    """
    res = _res(store)
    lat_c = float(np.clip(np.round(lat / res) * res, -90.0, 90.0))
    lon_c = float(((np.round(lon / res) * res + 180.0) % 360.0) - 180.0)
    df, meta = _ensemble_series(
        (lat_c, lat_c, lon_c, lon_c), variable, init_time, accum_hours, store=store
    )
    meta["grid_point"] = (lat_c, lon_c)
    return df, meta


def _tiles(
    lat_idx: tuple[int, int],
    lon_idx: tuple[int, int],
    tile: int = 32,
    lon_tile: int | None = None,
) -> int:
    """Chunks a lat/lon index range touches (inclusive ends)."""
    lon_tile = lon_tile or tile
    rows = lat_idx[1] // tile - lat_idx[0] // tile + 1
    cols = lon_idx[1] // lon_tile - lon_idx[0] // lon_tile + 1
    return rows * cols


def _chunk_count(
    geometry: dict,
    lat_idx: tuple[int, int],
    lon_idx: tuple[int, int],
    horizon_hours: float | None,
) -> int:
    """Chunks one init of one variable touches: tiles x lead blocks x member blocks.

    Leads count from T+0, so the leads up to a horizon are a prefix of the
    axis. Stores that keep the whole lead axis (or all members) in one chunk
    get one block for it; GEFS splits leads into ~8-day blocks.
    """
    chunks = geometry["chunks"]
    tiles = _tiles(lat_idx, lon_idx, chunks["latitude"], chunks["longitude"])
    leads = geometry["lead_hours"]
    kept = int((leads <= horizon_hours).sum()) if horizon_hours else len(leads)
    lead_blocks = -(-max(kept, 1) // chunks.get("lead_time", len(leads)))
    members = geometry.get("n_members", 1)
    member_blocks = -(-members // chunks.get("ensemble_member", members))
    return tiles * lead_blocks * member_blocks


def _store_chunks(
    store: str, bounds: tuple, horizon_hours: float | None = None
) -> tuple[int, int]:
    """`(chunks touched, bytes per chunk)` for one init of one variable.

    From the store's own geometry and chunk encoding, never assumed.
    """
    geometry = store_geometry(store)
    ds = _open_dataset(store)
    lat = np.asarray(ds["latitude"].values)
    lon = np.asarray(ds["longitude"].values)
    lat_min, lat_max, lon_min, lon_max = bounds
    rows = sorted(int(np.abs(lat - v).argmin()) for v in (lat_min, lat_max))
    cols = sorted(int(np.abs(lon - v).argmin()) for v in (lon_min, lon_max))
    n = _chunk_count(geometry, tuple(rows), tuple(cols), horizon_hours)
    itemsize = ds[_probe_variable(ds)].dtype.itemsize
    nbytes = int(np.prod(list(geometry["chunks"].values()))) * itemsize
    return n, nbytes


def read_estimate(
    stores: tuple[str, ...],
    bounds: tuple,
    n_inits: int = 1,
    horizon_hours: float | None = None,
) -> tuple[int, float]:
    """`(chunks, MB)` a cold read of `bounds` would touch, over models and inits."""
    total_chunks, total_bytes = 0, 0
    # A view (IFS Control) shares its source's reads: count the source once.
    for store in physical_stores(stores):
        tiles, nbytes = _store_chunks(store, bounds, horizon_hours)
        total_chunks += tiles * n_inits
        total_bytes += tiles * n_inits * nbytes
    return total_chunks, total_bytes / 1e6


def check_read_budget(
    stores: tuple[str, ...],
    bounds: tuple,
    n_inits: int = 1,
    horizon_hours: float | None = None,
) -> None:
    """Refuse a request whose chunk reads exceed `READ_BUDGET_MB`, before reading.

    Only store metadata is consulted. The cost is tiles x inits x models;
    the message states it in MB so the choice of what to drop is informed.
    """
    total_chunks, total_mb = read_estimate(stores, bounds, n_inits, horizon_hours)
    if total_mb > READ_BUDGET_MB:
        n = len(stores)
        raise GridDataError(
            f"This needs ~{total_mb:,.0f} MB across {n} model"
            f"{'s' if n != 1 else ''} ({total_chunks} chunks) — over the "
            f"{READ_BUDGET_MB:,.0f} MB interactive limit. Pick fewer models, "
            "fewer runs or a smaller area."
        )


def area_bounds(area_key: str) -> tuple:
    """Snapped bounds for a `box:` key or a region key."""
    if area_key.startswith("box:"):
        return snap_bbox(*(float(v) for v in area_key[4:].split(",")))
    if area_key not in REGIONS:
        raise GridDataError(f"Unknown region '{area_key}'.")
    return snap_bbox(*REGIONS[area_key]["bbox"], max_span=360.0)


def fetch_area_multi(
    area_key: str,
    variable: str,
    init_time: str | None,
    accum_hours: int | None,
    stores: tuple[str, ...],
) -> tuple[dict[str, tuple[pd.DataFrame, dict]], list[str]]:
    """The same area series from several stores, each read and reduced on its own.

    Never pooled: each store keeps its own members, percentiles and init. A
    store that cannot serve the request (a variable it does not archive, an
    init it does not hold) is reported in the error list and the others are
    still returned, in the order given.

    Parameters
    ----------
    area_key : str
        A `REGIONS` key, or `"box:lat_min,lat_max,lon_min,lon_max"` with
        edges rounded to 2 dp — the same key `frame_cube` takes.

    Returns
    -------
    tuple
        `({store: (df, meta)}, [error message, ...])`.
    """
    check_read_budget(stores, area_bounds(area_key))
    results, errors = {}, []
    for n, store in enumerate(stores, start=1):
        status.report("Reading archive", n, len(stores), GRID_STORES[store]["label"])
        try:
            if area_key.startswith("box:"):
                edges = [float(v) for v in area_key[4:].split(",")]
                results[store] = fetch_bbox_ensemble(
                    *edges, variable, init_time, accum_hours, store=store
                )
            else:
                results[store] = fetch_region_ensemble(
                    area_key, variable, init_time, accum_hours, store=store
                )
        except GridDataError as exc:
            errors.append(str(exc))
            continue
        if results[store][1].get("clip_note"):
            errors.append(results[store][1]["clip_note"])
    return results, errors


def _ensemble_series(
    bounds: tuple,
    variable: str,
    init_time: str | None,
    accum_hours: int | None,
    region: str | None = None,
    store: str = GRID_STORE_DEFAULT,
) -> tuple[pd.DataFrame, dict]:
    """Shared read for boxes, regions and points; see `fetch_bbox_ensemble`."""
    if variable not in GRID_VARIABLES:
        raise GridDataError(f"Unknown grid variable '{variable}'.")
    source = GRID_STORES[store].get("source_store")
    if source:
        df, meta = _ensemble_series(bounds, variable, init_time, accum_hours, region, store=source)
        return _member_subset(df, meta, store)
    bounds, clip_note = clip_to_store(store, bounds)

    ds = _open_dataset(store)
    step_type = _require_variable(ds, variable, store)

    if init_time:
        candidates = [pd.Timestamp(init_time)]
        _require_init(store, candidates[0])
    else:
        # Newest first, walking back over runs that are still being written.
        candidates = _latest_candidates(store)

    regional = None
    for candidate in candidates:
        attempt, da = _read_regional_mean(ds, variable, candidate, bounds, region)
        if _is_complete(attempt):
            regional, init, selected = attempt, candidate, da
            break

    if regional is None:
        raise GridDataError(
            "No completed forecast run covers that box — the most recent runs in "
            "the store are still being written."
        )

    da = selected
    lat_min, lat_max, lon_min, lon_max = bounds
    meta_var = GRID_VARIABLES[variable]
    values = regional.transpose("ensemble_member", "lead_time").values
    values = values * meta_var["scale"] + meta_var["offset"]

    members = regional["ensemble_member"].values
    lead = pd.to_timedelta(regional["lead_time"].values)
    valid_time = pd.to_datetime(np.asarray(regional["valid_time"].values))

    df = pd.DataFrame(
        {
            "valid_time": np.tile(valid_time, len(members)),
            "lead_hours": np.tile(lead.total_seconds() / 3600.0, len(members)),
            "member": np.repeat(members, len(lead)),
            "value": values.reshape(-1),
        }
    )
    df["variable"] = variable
    df = df.dropna(subset=["value"])

    if df.empty:
        raise GridDataError("No data available for that box, variable, and init time.")

    unit = meta_var["unit"]
    if accum_hours:
        if not meta_var.get("is_accumulated"):
            raise GridDataError(
                f"'{variable}' is not an accumulated variable — it cannot be "
                "totalled over a window."
            )
        df = accumulate_over_window(
            df,
            accum_hours,
            step_type=step_type,
        )
        if df.empty:
            raise GridDataError(
                f"No lead time in this run carries a complete {accum_hours} h "
                "accumulation."
            )
        # The quantity has changed from a rate to a total, so the unit has to
        # change with it. Leaving "mm/h" on a 24 h total is off by 24.
        unit = meta_var.get("accum_unit", unit)

    meta = {
        "init_time": init,
        "n_members": int(len(members)),
        "n_lat": int(da.sizes["latitude"]),
        "n_lon": int(da.sizes["longitude"]),
        "bbox": (lat_min, lat_max, lon_min, lon_max),
        "resolution_deg": _res(store),
        "step_window": step_window(store, variable),
        "clip_note": clip_note,
        "store": store,
        "model_label": GRID_STORES[store]["model_label"],
        "unit": unit,
        "accum_hours": accum_hours,
        "region": region,
        "region_label": REGIONS[region]["label"] if region else None,
        "n_cells_in_region": (
            int(
                polygon_mask(
                    region, da["latitude"].values, da["longitude"].values
                ).sum()
            )
            if region and polygon(region) is not None
            else None
        ),
    }
    return df, meta


def eval_inits(
    stores: tuple[str, ...],
    n_runs: int,
    hour: int | None = None,
    before: pd.Timestamp | None = None,
) -> list[pd.Timestamp]:
    """The newest `n_runs` finished inits every store holds, newest first.

    `hour` keeps one cycle only (e.g. 00 UTC), so scores compare like with
    like: an ensemble's error depends on the time of day its leads verify.
    `before` drops inits after it (the last observation): a run that starts
    after the last report has nothing to verify and would take a slot.
    """
    cal = stores_calendar(tuple(stores))
    unfinished = {pd.Timestamp(t) for t in cal["unfinished"]}
    if before is not None:
        before = pd.Timestamp(before)
        if before.tzinfo is not None:
            before = before.tz_convert("UTC").tz_localize(None)
    inits = [
        pd.Timestamp(f"{day}T{h:02d}:00")
        for day, hours in cal["days"].items()
        for h in hours
        if hour is None or h == hour
    ]
    inits = sorted(
        (t for t in inits if t not in unfinished and (before is None or t <= before)),
        reverse=True,
    )
    return inits[:n_runs]


def run_in_calendar(cal: dict, run: str | None) -> bool:
    """Whether a run value can be served by a calendar: "latest", or a finished init it holds."""
    if not run or run == "latest":
        return True
    try:
        t = pd.Timestamp(run)
    except ValueError:
        return False
    return (
        t.hour in cal["days"].get(f"{t:%Y-%m-%d}", [])
        and f"{t:%Y-%m-%dT%H:%M}" not in set(cal["unfinished"])
    )


def stores_calendar(stores: tuple[str, ...]) -> dict:
    """`run_calendar` for one store, `common_run_calendar` (sorted key) for several."""
    if len(stores) == 1:
        return run_calendar(stores[0])
    return common_run_calendar(tuple(sorted(stores)))


def resolve_run(stores: tuple[str, ...], run: str | None) -> str | None:
    """The init a request reads: one common init for several models, always.

    "Latest" for one model stays `None` (that store's newest complete run).
    For several it becomes their newest common finished init, so the models
    are compared from the same start and the animation's panels share valid
    times; each store's own newest run may differ by hours.
    """
    if run and run != "latest":
        return run
    if len(stores) < 2:
        return None
    newest = eval_inits(tuple(sorted(stores)), 1)
    if not newest:
        raise GridDataError("The selected models have no finished run in common.")
    return f"{newest[0]:%Y-%m-%dT%H:%M}"


def _read_point_run(lat, lon, variable, init_time, accum_hours, store):
    """The uncached point read; one chunk of the store per call."""
    return fetch_point_ensemble.uncached(lat, lon, variable, init_time, accum_hours, store)


def _point_cache_path(
    lat: float, lon: float, variable: str, init_time: str, accum_hours: int | None, store: str
) -> Path:
    """Disk path for one past run's extract, keyed by the grid cell it reads."""
    res = _res(store)
    lat_c = float(np.clip(np.round(lat / res) * res, -90.0, 90.0))
    lon_c = float(((np.round(lon / res) * res + 180.0) % 360.0) - 180.0)
    init = pd.Timestamp(init_time).strftime("%Y%m%dT%H%M")
    name = f"{store}__{variable}__{lat_c:+.2f}_{lon_c:+.2f}__{init}__{accum_hours or 0}.pkl"
    return Path(POINT_CACHE_DIR) / name


def point_run_cached(
    lat: float, lon: float, variable: str, init_time: str, accum_hours: int | None, store: str
) -> bool:
    """Whether a past run's extract is already on disk (reading it is free)."""
    store = GRID_STORES[store].get("source_store", store)
    return _point_cache_path(lat, lon, variable, init_time, accum_hours, store).exists()


def _point_run(lat, lon, variable, init_time, accum_hours, store):
    """A past run's point extract, read once and then kept on disk for good.

    Not in the flask cache: its entries share keys with `fetch_point_ensemble`
    and its size threshold evicts never-expiring entries first. A view store
    (IFS Control) is cut from its source's extract, so both share one file.
    """
    source = GRID_STORES[store].get("source_store")
    if source:
        df, meta = _point_run(lat, lon, variable, init_time, accum_hours, source)
        return _member_subset(df, meta, store)
    path = _point_cache_path(lat, lon, variable, init_time, accum_hours, store)
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    result = _read_point_run(lat, lon, variable, init_time, accum_hours, store)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.part")
    with tmp.open("wb") as fh:
        pickle.dump(result, fh)
    tmp.replace(path)
    return result


def point_archive(
    store: str,
    variable: str,
    lat: float,
    lon: float,
    inits: list[pd.Timestamp],
    accum_hours: int | None,
) -> pd.DataFrame:
    """One store's ensemble at the cell nearest a point, for several past runs.

    Each run costs one chunk read the first time (~18 MB for IFS ENS) and
    nothing after. Call `check_read_budget` first.

    Returns
    -------
    pd.DataFrame
        `init_time, valid_time, lead_hours, member, value`.
    """
    frames = []
    for n, init in enumerate(inits, start=1):
        status.report("Reading runs", n, len(inits), GRID_STORES[store]["label"])
        df, _ = _point_run(lat, lon, variable, f"{init:%Y-%m-%dT%H:%M}", accum_hours, store)
        frames.append(df.assign(init_time=pd.Timestamp(init)))
    if not frames:
        return pd.DataFrame(columns=["init_time", "valid_time", "lead_hours", "member", "value"])
    return pd.concat(frames, ignore_index=True)


# Shown for a past run when nothing archive-backed is selected.
_ARCHIVE_FALLBACK_SYSTEM = "ecmwf_ifs025"


def archive_systems(selected: list[str] | None) -> list[str]:
    """The selected Open-Meteo systems that have an archive, in selection order.

    A past run can only come from a system with an `archive_store`; when none
    of the selection has one, IFS ENS stands in rather than drawing nothing.
    """
    kept = [
        s for s in selected or []
        if "archive_store" in (ENSEMBLE_MODELS.get(s) or DETERMINISTIC_MODELS.get(s) or {})
    ]
    return kept or [_ARCHIVE_FALLBACK_SYSTEM]


def archive_meteogram(
    lat: float,
    lon: float,
    daily_variable: str,
    init_time: str,
    forecast_days: int,
    accum_hours: int | None = None,
    store: str = GRID_STORE_DEFAULT,
) -> tuple[pd.DataFrame, dict, dict]:
    """A past run of one archived model at a point, for `figures.make_meteogram_fig`.

    The point pages speak in Open-Meteo's daily variables; the archive holds
    3–6-hourly fields. The variable is mapped through its family (2 m
    temperature, precipitation, …); one with no archived equivalent is
    refused rather than approximated.

    Returns
    -------
    tuple[pd.DataFrame, dict, dict]
        Frame with `date`, `value`, `member`; the grid variable's metadata,
        relabelled for an accumulation window; and the fetch metadata.
    """
    family = family_of("daily", daily_variable)
    members = VARIABLE_FAMILIES.get(family or "", {}).get("grid")
    if not members:
        archived = ", ".join(v["label"] for v in GRID_VARIABLES.values())
        raise GridDataError(
            f"Past runs come from the {GRID_STORES[store]['label']} archive, which "
            f"holds: {archived}. This variable is not among them — choose the "
            "latest run instead."
        )
    grid_variable = members[0]
    if daily_variable in _DAILY_EXTREMES:
        return _archive_daily_extreme(lat, lon, daily_variable, init_time, forecast_days, store)
    var_meta = GRID_VARIABLES[grid_variable]
    accum = accum_hours if var_meta.get("is_accumulated") else None

    df, meta = fetch_point_ensemble(lat, lon, grid_variable, init_time, accum, store)
    df = df[df["lead_hours"] <= forecast_days * 24].rename(columns={"valid_time": "date"})
    if accum:
        # Disjoint windows, as on the latest-run meteogram: the store's windows
        # roll (one ending at every step), and boxes drawn from overlapping
        # windows share most of their hours. Windows ending at multiples of the
        # period from the init are exactly the disjoint ones.
        df = df[df["lead_hours"] % accum == 0]
        var_meta = {
            **var_meta,
            "unit": var_meta.get("accum_unit", var_meta["unit"]),
            "label": f"{var_meta['label']} ({accum} h total)",
        }
    meta["grid_variable"] = grid_variable
    meta["steps"] = store_steps(store)
    return df[["date", "value", "member"]], var_meta, meta


# Daily max/min on the meteogram: from the model's own step extremes where the
# store archives them (GEFS/GFS), else from instantaneous samples (IFS/AIFS).
_DAILY_EXTREMES: dict[str, tuple[str, str]] = {
    "temperature_2m_max": ("max", "maximum_temperature_2m"),
    "temperature_2m_min": ("min", "minimum_temperature_2m"),
}


def _archive_daily_extreme(lat, lon, variable, init_time, forecast_days, store):
    """A past run's daily max/min per member; `meta["daily_source"]` says how it was made."""
    kind, window_var = _DAILY_EXTREMES[variable]
    windowed = window_var in store_variables(store)
    grid_variable = window_var if windowed else "temperature_2m"
    df, meta = fetch_point_ensemble(lat, lon, grid_variable, init_time, None, store)
    df = df[df["lead_hours"] <= forecast_days * 24].rename(columns={"valid_time": "date"})
    daily = daily_extreme(df, kind, step_type=kind if windowed else "instant")
    word = "Max" if kind == "max" else "Min"
    var_meta = {**GRID_VARIABLES["temperature_2m"], "label": f"{word} temperature (2 m, daily)"}
    meta["grid_variable"] = grid_variable
    meta["steps"] = store_steps(store)
    meta["daily_source"] = "window" if windowed else "samples"
    return daily[["date", "value", "member"]], var_meta, meta


# Cells per animation frame before thinning. Frames are static PNGs, so the
# limit is render time, not payload; ECMWF Europe (~36k cells) is unthinned.
FRAME_CELL_BUDGET = 250_000


def _complete_run(ds, variable: str, bounds: tuple, init_time: str | None, store: str):
    """The requested run, or the newest one fully written, as a lazy box."""
    if init_time:
        candidates = [pd.Timestamp(init_time)]
        _require_init(store, candidates[0])
    else:
        candidates = _latest_candidates(store)
    for candidate in candidates:
        da = _select_box(ds, variable, candidate, bounds)
        # The last lead of member 0 is the cheapest completeness probe: a run
        # still being written has NaN chunks at its far leads.
        probe = da.isel(lead_time=-1, ensemble_member=0).compute()
        if not bool(probe.isnull().all()):
            return candidate, da
    raise GridDataError(
        "No completed forecast run covers that area — the most recent runs in the "
        "store are still being written."
    )


@functools.lru_cache(maxsize=3)
def frame_cube(
    area_key: str,
    variable: str,
    init_time: str | None,
    step_hours: int,
    accum_hours: int | None,
    store: str = GRID_STORE_DEFAULT,
    horizon_hours: int | None = None,
) -> dict:
    """Every member at a subset of leads, thinned for display, held in memory.

    The expensive read happens once per (area, variable, run, step, window);
    every statistic the map offers — mean, a percentile, a
    percentile range — is then computed from this cube in memory, so changing
    it does not re-read the archive. An in-process LRU (three cubes — one per
    model of a comparison) rather than the shared cache: the cube is too
    large to pickle through the file cache on every change, and a miss only
    costs a re-read.

    `horizon_hours` trims the leads *before* the values are pulled into
    memory. It does not reduce the download — a chunk holds the whole lead
    axis — but it cuts decode time and the cube's size (T+72 h keeps ~30% of
    an IFS run).

    Parameters
    ----------
    area_key
        A `utils.regions` key, or "box:lat_min,lat_max,lon_min,lon_max".
    step_hours
        Spacing of the animation frames in lead time.
    accum_hours
        For an accumulated variable, each frame is the member's total over
        the window ending at that lead (`cum(L) - cum(L-P)`); leads without a
        complete window are dropped, as on the time series.

    Returns
    -------
    dict
        `cube` float32 `(frame, member, lat, lon)` in display units, NaN
        outside a region polygon; `lead_hours`, `valid_time`, `lat`, `lon`,
        `members`, `init_time`, `stride`, `bounds`, `region`, `unit`.
    """
    if variable not in GRID_VARIABLES:
        raise GridDataError(f"Unknown grid variable '{variable}'.")
    source = GRID_STORES[store].get("source_store")
    if source:
        cube = frame_cube(
            area_key, variable, init_time, step_hours, accum_hours, source, horizon_hours
        )
        return _cube_member(cube, GRID_STORES[store]["member"], store)
    region = None
    if area_key.startswith("box:"):
        bounds = snap_bbox(*(float(v) for v in area_key[4:].split(",")), res=_res(store))
    else:
        region = area_key
        bounds = snap_bbox(*REGIONS[region]["bbox"], max_span=360.0, res=_res(store))
    bounds, clip_note = clip_to_store(store, bounds)

    ds = _open_dataset(store)
    _require_variable(ds, variable, store)
    init, da = _complete_run(ds, variable, bounds, init_time, store)
    meta_var = GRID_VARIABLES[variable]
    if horizon_hours:
        all_h = pd.to_timedelta(da["lead_time"].values).total_seconds() / 3600.0
        da = da.isel(lead_time=np.flatnonzero(np.asarray(all_h) <= horizon_hours))

    leads = pd.to_timedelta(da["lead_time"].values)
    lead_h = np.asarray(leads.total_seconds() / 3600.0)
    try:
        # (member, lead, lat, lon), read once. Everything below is in memory.
        values = da.transpose("ensemble_member", "lead_time", "latitude", "longitude").values
    except (OSError, ValueError, RuntimeError) as exc:
        raise GridDataError(f"Failed reading the gridded field ({exc}).") from exc
    values = values.astype(np.float32) * meta_var["scale"] + meta_var["offset"]
    unit = meta_var["unit"]

    if accum_hours:
        if not meta_var.get("is_accumulated"):
            raise GridDataError(f"'{variable}' cannot be totalled over a window.")
        # Mean rate over the preceding step x step length = amount per step.
        duration = np.diff(lead_h, prepend=0.0).astype(np.float32)
        cum = np.cumsum(np.nan_to_num(values) * duration[None, :, None, None], axis=1)
        index = {h: i for i, h in enumerate(lead_h)}
        keep, totals = [], []
        for i, h in enumerate(lead_h):
            start = h - accum_hours
            if start < 0 or (start > 0 and start not in index):
                continue
            base = cum[:, index[start]] if start > 0 else 0.0
            keep.append(i)
            totals.append(cum[:, i] - base)
        if not keep:
            raise GridDataError(f"No lead in this run carries a complete {accum_hours} h total.")
        values = np.stack(totals, axis=1)
        lead_h = lead_h[keep]
        valid_all = np.asarray(da["valid_time"].values)[keep]
        unit = meta_var.get("accum_unit", unit)
    else:
        valid_all = np.asarray(da["valid_time"].values)

    frames = [i for i, h in enumerate(lead_h) if h > 0 and h % step_hours == 0]
    if not frames:
        raise GridDataError(f"No lead time falls on a {step_hours} h step.")
    values = values[:, frames]

    lat = np.asarray(da["latitude"].values, dtype=float)
    lon = np.asarray(da["longitude"].values, dtype=float)
    mask = polygon_mask(region, lat, lon)
    if mask is not None:
        values = np.where(mask[None, None], values, np.nan)

    cells = lat.size * lon.size
    stride = max(1, int(np.ceil(np.sqrt(cells / FRAME_CELL_BUDGET))))
    values = values[:, :, ::stride, ::stride]
    lat, lon = lat[::stride], lon[::stride]
    order = np.argsort(lat)  # stored descending; plot ascending

    return {
        "cube": np.ascontiguousarray(values[:, :, order].transpose(1, 0, 2, 3)),
        "lead_hours": lead_h[frames].astype(int).tolist(),
        "valid_time": pd.to_datetime(valid_all[frames]).tolist(),
        "lat": lat[order],
        "lon": lon,
        "members": da["ensemble_member"].values.tolist(),
        "init_time": init,
        "stride": stride,
        "bounds": bounds,
        "region": region,
        "unit": unit,
        "store": store,
        "clip_note": clip_note,
    }


def frame_statistic(
    cube: dict, statistic: str, pct: float, pct_range: tuple
) -> tuple[np.ndarray, str]:
    """Reduce a frame cube over members; returns `(values, description)`.

    `statistic` is "mean", "percentile" or "range" (P_hi − P_lo). Computed on
    the cells inside the area only (outside a region polygon the cube is NaN
    for every member), with plain `np.percentile` — several times faster
    than `nanpercentile`, and no empty-slice warnings.
    """
    data = cube["cube"]  # (frame, member, lat, lon)
    n_frames, n, n_lat, n_lon = data.shape
    flat = data.reshape(n_frames, n, n_lat * n_lon)
    inside = np.isfinite(flat[0, 0])
    out = np.full((n_frames, n_lat * n_lon), np.nan, dtype=np.float32)
    cells = flat[:, :, inside]

    if statistic == "percentile":
        out[:, inside] = np.percentile(cells, pct, axis=1)
        description = f"{pct:g}th percentile across {n} members"
    elif statistic == "range":
        low, high = pct_range
        bounds = np.percentile(cells, [low, high], axis=1)
        out[:, inside] = bounds[1] - bounds[0]
        description = f"P{high:g} − P{low:g} spread across {n} members"
    else:
        out[:, inside] = cells.mean(axis=1)
        description = f"ensemble mean of {n} members"
    return out.reshape(n_frames, n_lat, n_lon), description


@cache.memoize(timeout=1800)
def fetch_bbox_field(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
    variable: str,
    lead_hours: int,
    member: int | None = None,
    init_time: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Map-shaped read: the 2-D field over a box at one lead time.

    The spatial counterpart to `fetch_bbox_ensemble` — that one collapses
    space and keeps members, this one keeps space and collapses (or picks)
    members. Same store, same box guard, same init fallback.

    `member=None` returns the ensemble mean, which is a legitimate reduction
    *within* one system (only cross-system means are ruled out). It is
    smoother than any member and must be labelled as such on the plot.

    Parameters
    ----------
    lat_min, lat_max, lon_min, lon_max : float
        Rectangle edges, rounded to 2 decimal places by the caller. Snapped
        to the grid and clamped to the size guard by `snap_bbox`.
    variable : str
        Key into `utils.settings.GRID_VARIABLES`.
    lead_hours : int
        Forecast lead in hours from the init time. Snapped to the nearest
        lead present in the store.
    member : int or None
        Ensemble member to map. `None` maps the across-member mean.
    init_time : str or None
        ISO init timestamp. `None` selects the latest complete run.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The frame is tidy: one row per grid cell with `latitude`,
        `longitude`, `value`, `variable`. The dict carries `init_time`,
        `valid_time`, `lead_hours` (as resolved), `reduction`, `n_members`,
        `n_lat`, `n_lon`, `bbox`, `resolution_deg`, and `unit`.
    """
    if variable not in GRID_VARIABLES:
        raise GridDataError(f"Unknown grid variable '{variable}'.")

    ds = _open_dataset()
    _require_variable(ds, variable, GRID_STORE_DEFAULT)

    bounds = snap_bbox(lat_min, lat_max, lon_min, lon_max)

    inits = available_inits()
    if init_time:
        requested = pd.Timestamp(init_time)
        if requested not in set(inits):
            raise GridDataError(
                f"Init {requested:%Y-%m-%d %H:%M} UTC is not in the store "
                f"(latest is {inits[-1]:%Y-%m-%d %H:%M} UTC)."
            )
        candidates = [requested]
    else:
        candidates = _latest_candidates(GRID_STORE_DEFAULT)

    lead = pd.Timedelta(hours=int(lead_hours))
    field = None
    for candidate in candidates:
        da = _select_box(ds, variable, candidate, bounds)

        # `method="nearest"` is what lets an off-grid lead (T+5) snap to a real
        # one (T+6), but on its own it would also clamp T+10000 to the final
        # lead and plot it as if it had been asked for. Range-check first, so
        # snapping stays a convenience and never becomes a silent substitution.
        leads = pd.to_timedelta(da["lead_time"].values)
        if lead < leads.min() or lead > leads.max():
            horizon = int(leads.max().total_seconds() // 3600)
            raise GridDataError(
                f"Lead time {lead_hours} h is outside this run's horizon "
                f"(0-{horizon} h)."
            )

        try:
            at_lead = da.sel(lead_time=lead, method="nearest")
        except (KeyError, ValueError) as exc:
            raise GridDataError(
                f"Could not select lead time {lead_hours} h ({exc})."
            ) from exc

        n_members = int(at_lead.sizes["ensemble_member"])
        if member is None:
            reduced = at_lead.mean("ensemble_member")
            reduction = f"ensemble mean of {n_members} members"
        else:
            if member not in set(at_lead["ensemble_member"].values.tolist()):
                raise GridDataError(
                    f"Member {member} is not in this run (0–{n_members - 1})."
                )
            reduced = at_lead.sel(ensemble_member=member)
            reduction = f"member {member} of {n_members}"

        try:
            computed = reduced.compute()
        except (OSError, ValueError, RuntimeError) as exc:
            raise GridDataError(f"Failed reading the gridded field ({exc}).") from exc

        # An init still being written returns all-NaN chunks at unwritten
        # leads; same completeness rule as the time-series path, applied to
        # the one lead actually asked for.
        if not bool(computed.isnull().all()):
            field, init, selected = computed, candidate, at_lead
            break

    if field is None:
        raise GridDataError(
            "No completed forecast run covers that box and lead time — the most "
            "recent runs in the store are still being written."
        )

    meta_var = GRID_VARIABLES[variable]
    lat = np.asarray(field["latitude"].values, dtype=float)
    lon = np.asarray(field["longitude"].values, dtype=float)
    values = field.transpose("latitude", "longitude").values
    values = values * meta_var["scale"] + meta_var["offset"]

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    df = pd.DataFrame(
        {
            "latitude": lat_grid.reshape(-1),
            "longitude": lon_grid.reshape(-1),
            "value": values.reshape(-1),
        }
    )
    df["variable"] = variable
    df = df.dropna(subset=["value"])

    if df.empty:
        raise GridDataError("No data available for that box, variable, and lead time.")

    resolved_lead = pd.Timedelta(field["lead_time"].values)
    lat_min, lat_max, lon_min, lon_max = bounds
    meta = {
        "init_time": init,
        "valid_time": pd.Timestamp(np.asarray(field["valid_time"].values).item()),
        "lead_hours": int(resolved_lead.total_seconds() // 3600),
        "reduction": reduction,
        "n_members": int(selected.sizes["ensemble_member"]),
        "n_lat": int(selected.sizes["latitude"]),
        "n_lon": int(selected.sizes["longitude"]),
        "bbox": (lat_min, lat_max, lon_min, lon_max),
        "resolution_deg": GRID_RESOLUTION_DEG,
        "step_window": step_window(GRID_STORE_DEFAULT, variable),
        "unit": meta_var["unit"],
    }
    return df, meta


def _open_analysis(key: str):
    """Open an analysis store (`ANALYSIS_STORES`), once per process, fork-safe like `_open_dataset`."""
    global _dataset_pid
    name = f"analysis:{key}"
    with _dataset_lock:
        if _dataset_pid != os.getpid():
            _datasets.clear()
            _dataset_pid = os.getpid()
        if name in _datasets:
            return _datasets[name]
        import icechunk
        import xarray as xr

        spec = ANALYSIS_STORES[key]
        bucket, prefix = _split_s3(spec["href"])
        try:
            storage = icechunk.s3_storage(
                bucket=bucket, prefix=prefix, region=spec["region"], anonymous=True
            )
            session = icechunk.Repository.open(storage=storage).readonly_session("main")
            ds = xr.open_zarr(session.store, consolidated=False)
        except (OSError, ValueError, RuntimeError) as exc:
            raise GridDataError(f"Could not open {spec['label']} ({exc}).") from exc
        _datasets[name] = ds
        return ds


def _imerg_month(lat_c: float, lon_c: float, month: pd.Timestamp) -> pd.DataFrame:
    """One month of IMERG half-hour amounts (mm) at a cell, from disk once read.

    A past month is kept for good; the current month is re-read after 3 h
    (IMERG Late lands ~14 h after observation).
    """
    path = Path(POINT_CACHE_DIR) / f"imerg__{lat_c:+.2f}_{lon_c:+.2f}__{month:%Y%m}.pkl"
    current = month == pd.Timestamp.now().normalize().replace(day=1)
    if path.exists() and not (current and time.time() - path.stat().st_mtime > 3 * 3600):
        with path.open("rb") as fh:
            return pickle.load(fh)
    spec = ANALYSIS_STORES["imerg"]
    ds = _open_analysis("imerg")
    end = month + pd.offsets.MonthBegin(1)
    try:
        da = ds[spec["variable"]].sel(latitude=lat_c, longitude=lon_c, method="nearest")
        da = da.sel(time=slice(month, end - pd.Timedelta(seconds=1)))
        values = da.values
    except (OSError, ValueError, RuntimeError, KeyError) as exc:
        raise GridDataError(f"Failed reading {spec['label']} ({exc}).") from exc
    # Mean rate (kg m-2 s-1 = mm/s) over the half hour x its length = mm.
    frame = pd.DataFrame(
        {"time": pd.to_datetime(da["time"].values), "mm": values * spec["step_seconds"]}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.part")
    with tmp.open("wb") as fh:
        pickle.dump(frame, fh)
    tmp.replace(path)
    return frame


def imerg_halfhours(lat: float, lon: float, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """IMERG half-hour amounts (`time, mm`, start-stamped) at the 0.1° cell nearest a point."""
    res = ANALYSIS_STORES["imerg"]["res_deg"]
    # Cell centres sit at .05: snap to them.
    lat_c = float(np.clip(np.floor(lat / res) * res + res / 2, -89.95, 89.95))
    lon_c = float(((np.floor(lon / res) * res + res / 2 + 180.0) % 360.0) - 180.0)
    months = pd.date_range(
        pd.Timestamp(start).normalize().replace(day=1), pd.Timestamp(end), freq="MS"
    )
    frames = [_imerg_month(lat_c, lon_c, m) for m in months]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "mm"])
    return out[(out["time"] >= pd.Timestamp(start)) & (out["time"] < pd.Timestamp(end))].reset_index(drop=True)
