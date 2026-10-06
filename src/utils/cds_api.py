"""Copernicus CDS / ADS retrievals for the maps page.

Third data source in the app, and the only one that serves global fields and
atmospheric composition. Scope is map-shaped reads: one 2-D field per request.
Point series stay on Open-Meteo and regional box averages stay on the
Icechunk store — nothing here should be used for either.

Two things make this module unlike `openmeteo_api` and `earthmover_api`.

**Latency.** A CDS or ADS retrieval is queued server-side and can take
anywhere from a few seconds to tens of minutes. That is far longer than a
Dash callback can hold, and longer than the 120 s `proxy_read_timeout` in the
deployment notes. Retrievals therefore run on a worker *thread* and the page
polls for completion. A thread, not a process: the fork hazard documented in
`app.py` applies to background callbacks too, and `cdsapi` holds a `requests`
session that must not be inherited across a fork.

**Job state is shared, not process-local.** Gunicorn runs three workers, and
the poll that follows a submit may land on a different worker from the one
that submitted. Job records therefore live in the shared `flask_caching`
backend, and the downloaded file lives on disk keyed by request hash — so
any worker can serve a job any other worker started, and a repeat of the same
request is answered from disk without touching the API at all.

The cache boundary sits after parsing: `@cache.memoize()` cannot serialise an
xarray object, so everything crossing it is a tidy DataFrame plus a plain-dict
metadata payload.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
import zipfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from utils.cache import cache
from utils.climatology import SEASONS
from utils.settings import (
    CAMS_EUROPE_MODELS,
    CDS_CACHE_DIR,
    CDS_DATASETS,
    CDS_JOB_TIMEOUT_S,
    CDS_STORES,
    CDS_VARIABLES,
    CLIMATE_INDICES,
    INDEX_GRID_DEG,
    INDEX_SOURCES,
    INDEX_START_YEAR,
    INDEX_TARGETS,
    MAP_AREA_GLOBAL_KEY,
    MAP_AREA_PRESETS,
    MONTHLY_ACCUM_MODES,
    REGRID_ABOVE_DEG,
    REGRID_MONTHLY_DEG,
)

if TYPE_CHECKING:  # xarray is imported lazily; see `_open`.
    import xarray as xr


class CDSDataError(RuntimeError):
    """Raised when a CDS/ADS request cannot be satisfied.

    Carries a message already suitable for display in a `dmc.Alert`.
    """


# Display budget. A global 0.25° ERA5 field is 1440x721 = 1.04M cells, which
# serialises to roughly 10 MB of JSON and stalls the browser. Fields above
# this are strided down for display and the *effective* resolution is carried
# in the metadata, so the plot never claims a resolution it is not showing.
MAX_DISPLAY_CELLS: int = 250_000

_JOB_KEY_PREFIX = "cdsjob:"
_JOB_TTL_S = 86_400

# Terminal job states. Anything else means the page should keep polling.
JOB_DONE = "completed"
JOB_FAILED = "failed"

# Threads started by *this* process, so a repeated submit does not launch a
# second retrieval of the same request within one worker. Cross-worker
# duplication is prevented by the shared job record instead.
_live_jobs: set[str] = set()
_live_lock = threading.Lock()

# CDS *rejects* — does not queue — requests beyond a small per-user limit per
# dataset ("Number queued requests for this dataset is temporarily limited").
# A gallery product or an anomaly is two requests on one dataset, so a second
# click would bounce. Retrievals therefore wait here for a slot instead. The
# limit is per process: gunicorn workers do not share it, so the friendly
# message in `_run_job` still covers the cross-worker case.
_DATASET_SLOTS = 2
_dataset_slots: dict[str, threading.BoundedSemaphore] = {}
_CDS_QUEUE_LIMIT_TEXT = "Number queued requests for this dataset is temporarily limited"


# ---------------------------------------------------------------------------
# Credentials and clients
# ---------------------------------------------------------------------------


def _rc_credentials(rc_file: str | None) -> tuple[str | None, str | None]:
    """Read `url:`/`key:` from a `.cdsapirc`-style file, if it exists."""
    if not rc_file:
        return None, None

    path = Path(rc_file).expanduser()
    if not path.is_file():
        return None, None

    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        name, _, value = line.partition(":")
        values[name.strip()] = value.strip()

    return values.get("url"), values.get("key")


def credentials(store: str) -> tuple[str, str] | None:
    """Resolve `(url, key)` for a store, or `None` if no token is configured.

    The environment wins over the rc file, so a deployment can override a
    developer's `~/.cdsapirc` without editing it. The ADS has no rc file of
    its own — `cdsapi` reads exactly one, and it points at the CDS.

    Parameters
    ----------
    store : str
        Key into `utils.settings.CDS_STORES` — `"cds"` or `"ads"`.

    Returns
    -------
    tuple[str, str] or None
        The endpoint URL and personal access token, or `None` when the token
        cannot be found. A missing token is an expected state, reported in
        the UI, not an error.
    """
    if store not in CDS_STORES:
        raise CDSDataError(f"Unknown data store '{store}'.")

    meta = CDS_STORES[store]
    rc_url, rc_key = _rc_credentials(meta["rc_file"])

    key = os.environ.get(meta["key_env"]) or rc_key
    url = meta["url"] or rc_url

    if not key or not url:
        return None
    return url, key


def credentials_message(store: str) -> str:
    """One-paragraph instruction for configuring a store whose token is missing."""
    meta = CDS_STORES[store]
    if meta["rc_file"]:
        how = (
            f"Put your token in {meta['rc_file']} as `url:` and `key:` lines, "
            f"or set {meta['key_env']} in the environment."
        )
    else:
        how = (
            f"Set {meta['key_env']} in the environment — cdsapi reads only one "
            f"rc file and it points at the CDS, so the {meta['short']} token "
            f"has nowhere else to live."
        )
    return (
        f"No {meta['label']} ({meta['short']}) token is configured. "
        f"Get a Personal Access Token from {meta['register_url']}. {how}"
    )


def _client(store: str):
    """Build a `cdsapi.Client` for one store.

    The import is function-local: `cdsapi` pulls in `requests` and opens a
    session, and importing this module (as the page does at startup, before
    gunicorn forks) must not do either.

    `wait_until_complete=True` is deliberate — the blocking wait happens on
    the job thread, where blocking is the point, and letting `cdsapi` own the
    polling keeps its backoff, retry, and licence-acceptance error handling.
    """
    import cdsapi

    resolved = credentials(store)
    if resolved is None:
        raise CDSDataError(credentials_message(store))

    url, key = resolved
    return cdsapi.Client(
        url=url, key=key, quiet=True, progress=False, wait_until_complete=True
    )


# ---------------------------------------------------------------------------
# Request building
# ---------------------------------------------------------------------------


def _area_key(area: tuple[float, float, float, float] | None) -> list[float] | None:
    """Convert `(lat_min, lat_max, lon_min, lon_max)` to the CDS `[N, W, S, E]` order.

    Returns `None` for the global extent, which omits `area` from the request
    entirely: an explicit -180/180 box is subsetted by MARS and can shave the
    wrap-around column, leaving a one-cell white stripe down the map.
    """
    if area is None:
        return None

    lat_min, lat_max, lon_min, lon_max = area
    if (lat_min, lat_max, lon_min, lon_max) == MAP_AREA_PRESETS[MAP_AREA_GLOBAL_KEY]:
        return None

    return [round(lat_max, 2), round(lon_min, 2), round(lat_min, 2), round(lon_max, 2)]


def build_request(
    dataset_key: str,
    variable: str,
    area: tuple[float, float, float, float] | None,
    date: str | None = None,
    hour: str | None = None,
    year: int | None = None,
    month: int | None = None,
    years: list[int] | None = None,
    leadtime_hour: int | None = None,
    init_hour: str | None = None,
    cams_model: str | None = None,
    accum_hours: int | None = None,
    pressure_level: str | None = None,
    grid: float | None = None,
    months: list[int] | None = None,
) -> dict:
    """Assemble the request dict for one dataset.

    Every dataset in the catalogue has its own required keys — ERA5 wants
    `product_type` and a year/month/day/time split, CAMS global wants a date
    range and `leadtime_hour`, CAMS Europe wants a `model` and a `level`. The
    branching lives here rather than in the page so the page only has to know
    which *controls* a dataset declares.

    Parameters
    ----------
    dataset_key : str
        Key into `utils.settings.CDS_DATASETS`.
    variable : str
        CDS/ADS variable name (the request name, not the NetCDF short name).
    area : tuple or None
        `(lat_min, lat_max, lon_min, lon_max)`. `None` or the global preset
        omits `area`.
    date : str or None
        ISO date, for datasets whose controls include `date`.
    hour : str or None
        `"HH:00"`, for the ERA5 hourly dataset.
    year, month : int or None
        For `era5_monthly`.
    years : list[int] or None
        Reference-period years, for the `era5_monthly` anomaly request.
    leadtime_hour : int or None
        Forecast lead in hours, for the CAMS datasets.
    init_hour : str or None
        `"00:00"` or `"12:00"`, for CAMS global.
    cams_model : str or None
        Key into `utils.settings.CAMS_EUROPE_MODELS`.
    accum_hours : int or None
        Accumulation window in hours for an accumulated variable. Anything
        above the native hourly step widens the request to cover the whole
        window, including the hour past midnight that a 00-24 UTC day needs.
    pressure_level : str or None
        hPa level, for `era5_monthly_pressure`.
    months : list[int] or None
        Several calendar months at once, for a seasonal request; overrides
        `month` for the monthly datasets.
    grid : float or None
        Output spacing in degrees. CDS regrids server-side, which is what
        keeps a multi-year global reference small. `None` keeps the native
        grid.

    Returns
    -------
    dict
        Request body, ready for `cdsapi.Client.retrieve`.
    """
    if dataset_key not in CDS_DATASETS:
        raise CDSDataError(f"Unknown dataset '{dataset_key}'.")

    meta = CDS_DATASETS[dataset_key]
    # The catalogue key is not always the CDS name: pressure-level variables
    # are keyed by name and level ("geopotential_500") but requested as
    # `geopotential` + `pressure_level`.
    var_meta = CDS_VARIABLES.get(dataset_key, {}).get(variable, {})
    request: dict = {"variable": [var_meta.get("request_variable", variable)]}
    pressure_level = pressure_level or var_meta.get("pressure_level")

    if dataset_key == "era5_single_levels":
        stamp = pd.Timestamp(date)
        request.update(
            {
                "product_type": ["reanalysis"],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
        )
        if accum_hours and accum_hours > 1:
            # A 24 h total for day D runs 00-24 UTC. ERA5's hourly field at
            # hour H is the accumulation over (H-1, H], so that window is
            # hours 01..23 of D *plus* hour 00 of D+1 — summing 00..23 of D
            # instead silently gives (23 UTC D-1, 23 UTC D], a day shifted an
            # hour early. Both days are requested and the 24 correct fields
            # are selected on read.
            end = stamp + pd.Timedelta(days=1)
            request["date"] = [f"{stamp:%Y-%m-%d}/{end:%Y-%m-%d}"]
            request["time"] = [f"{h:02d}:00" for h in range(24)]
        else:
            request.update(
                {
                    "year": [f"{stamp.year:04d}"],
                    "month": [f"{stamp.month:02d}"],
                    "day": [f"{stamp.day:02d}"],
                    "time": [hour or "12:00"],
                }
            )
    elif dataset_key in ("era5_monthly", "era5_monthly_pressure"):
        request_years = [str(y) for y in (years or [year])]
        request.update(
            {
                "product_type": [meta["product_type"]],
                "year": request_years,
                "month": [f"{int(m):02d}" for m in (months or [month])],
                # Monthly means carry a single nominal time; it is not an hour
                # of day and must not be offered as one.
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "unarchived",
            }
        )
        if dataset_key == "era5_monthly_pressure":
            if not pressure_level:
                raise CDSDataError("A pressure-level request needs a level.")
            request["pressure_level"] = [str(pressure_level)]
    elif dataset_key == "cams_global":
        request.update(
            {
                "date": [f"{date}/{date}"],
                "time": [init_hour or "00:00"],
                "leadtime_hour": [str(int(leadtime_hour or 0))],
                "type": ["forecast"],
                "data_format": "netcdf_zip",
            }
        )
    elif dataset_key == "cams_europe":
        if cams_model not in CAMS_EUROPE_MODELS:
            raise CDSDataError(f"Unknown CAMS Europe model '{cams_model}'.")
        request.update(
            {
                "model": [cams_model],
                "date": [f"{date}/{date}"],
                "type": ["forecast"],
                "time": ["00:00"],
                "leadtime_hour": [str(int(leadtime_hour or 0))],
                # Surface. The dataset is levelled, and omitting this returns
                # every level rather than failing.
                "level": ["0"],
                "data_format": "netcdf_zip",
            }
        )
    else:  # pragma: no cover — guarded by the catalogue check above
        raise CDSDataError(f"No request builder for '{dataset_key}'.")

    area_box = _area_key(area)
    if area_box is not None:
        request["area"] = area_box

    if grid is not None:
        request["grid"] = [float(grid), float(grid)]

    return request


def monthly_grid(dataset_key: str, area: tuple | None) -> float | None:
    """Output grid for a monthly request, coarsened only for very large areas.

    A monthly anomaly's reference is 30 fields; at 0.25° over the globe that
    is ~125 MB. Areas wider than `REGRID_ABOVE_DEG` in either direction are
    requested at `REGRID_MONTHLY_DEG` instead, and the map says so.
    """
    if dataset_key not in ("era5_monthly", "era5_monthly_pressure"):
        return None
    lat_min, lat_max, lon_min, lon_max = area or MAP_AREA_PRESETS[MAP_AREA_GLOBAL_KEY]
    if max(lat_max - lat_min, lon_max - lon_min) > REGRID_ABOVE_DEG:
        return REGRID_MONTHLY_DEG
    return None


def index_requests(
    index_key: str,
    month: int,
    reference: tuple[int, int],
    target: str,
    latest: pd.Timestamp,
) -> dict[str, tuple[str, dict]]:
    """Every retrieval one climate index needs, by role.

    The index source (SST, MSLP or z500) is requested for every month from
    `INDEX_START_YEAR`, split into complete years and the current partial
    year — asking CDS for months that do not exist yet fails the request.
    The correlation target is one calendar month over the reference years,
    over the index's fixed map area.

    Returns
    -------
    dict[str, tuple[str, dict]]
        `source_full`, `source_recent` (absent in January before any month
        of the year is out) and `target`, each `(dataset_key, request)`.
    """
    index = CLIMATE_INDICES[index_key]
    source = INDEX_SOURCES[index["source"]]
    if target not in INDEX_TARGETS:
        raise CDSDataError(f"Unknown correlation target '{target}'.")

    all_months = [f"{m:02d}" for m in range(1, 13)]

    def source_request(years: list[int], months: list[str]) -> dict:
        request = build_request(
            source["dataset"],
            source["variable"],
            source["area"],
            month=1,
            years=years,
            grid=INDEX_GRID_DEG,
        )
        request["month"] = months
        return request

    roles = {
        "source_full": (
            source["dataset"],
            source_request(list(range(INDEX_START_YEAR, latest.year)), all_months),
        ),
    }
    if latest.month >= 1 and latest.year > INDEX_START_YEAR:
        roles["source_recent"] = (
            source["dataset"],
            source_request([latest.year], all_months[: latest.month]),
        )

    area = MAP_AREA_PRESETS[index["map_area"]]
    years = list(range(reference[0], reference[1] + 1))
    roles["target"] = (
        "era5_monthly",
        build_request(
            "era5_monthly",
            target,
            area,
            month=month,
            years=years,
            grid=monthly_grid("era5_monthly", area),
        ),
    )
    return roles


def request_hash(dataset: str, request: dict) -> str:
    """Stable short hash of a dataset + request pair, used as job id and filename."""
    payload = json.dumps({"dataset": dataset, "request": request}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Job runner
# ---------------------------------------------------------------------------


def _cache_dir() -> Path:
    path = Path(CDS_CACHE_DIR)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _target_path(job_id: str) -> Path:
    return _cache_dir() / f"{job_id}.nc"


def _write_job(job_id: str, **fields) -> dict:
    record = cache.get(_JOB_KEY_PREFIX + job_id) or {}
    record.update(fields)
    record["updated_at"] = time.time()
    cache.set(_JOB_KEY_PREFIX + job_id, record, timeout=_JOB_TTL_S)
    return record


def job_status(job_id: str) -> dict:
    """Current state of a retrieval.

    Returns
    -------
    dict
        `state` is one of `"queued"`, `"running"`, `"completed"`, `"failed"`,
        or `"unknown"` when no record exists and no file is on disk. A
        completed record carries `path`; a failed one carries `message`.
    """
    record = cache.get(_JOB_KEY_PREFIX + job_id)
    if record:
        return record

    # A worker that never saw the submit can still serve it from disk: the
    # filename *is* the request hash, so the file's presence is proof the
    # request completed at some point.
    path = _target_path(job_id)
    if path.is_file():
        return {"state": JOB_DONE, "path": str(path), "message": ""}

    return {"state": "unknown", "message": "No such retrieval.", "path": ""}


def _extract_netcdf(archive: Path, destination: Path) -> None:
    """Pull the single NetCDF out of a `netcdf_zip` response, or move it as-is.

    CAMS responses come back zipped; ERA5 responses do not. The download
    target is the same either way, so the format is detected from the file's
    magic number rather than from the request.
    """
    if not zipfile.is_zipfile(archive):
        shutil.move(str(archive), str(destination))
        return

    with zipfile.ZipFile(archive) as zf:
        members = [name for name in zf.namelist() if name.endswith(".nc")]
        if not members:
            raise CDSDataError("The response archive contained no NetCDF file.")
        with zf.open(members[0]) as source, destination.open("wb") as sink:
            shutil.copyfileobj(source, sink)

    archive.unlink(missing_ok=True)


def _run_job(job_id: str, store: str, dataset: str, request: dict) -> None:
    """Retrieve one request on a worker thread and record the outcome.

    Downloads to a temporary file in the same directory and moves it into
    place only once complete, so a partial download can never be mistaken for
    a cached result by another worker reading the directory.
    """
    target = _target_path(job_id)
    scratch = None

    try:
        _write_job(job_id, state="running", message="Request submitted to the queue.")
        client = _client(store)

        handle, scratch_name = tempfile.mkstemp(dir=str(_cache_dir()), suffix=".part")
        os.close(handle)
        scratch = Path(scratch_name)

        with _live_lock:
            slot = _dataset_slots.setdefault(
                dataset, threading.BoundedSemaphore(_DATASET_SLOTS)
            )
        _write_job(job_id, state="queued", message="Waiting for a free Copernicus slot.")
        with slot:
            _write_job(job_id, state="running", message="Request submitted to the queue.")
            client.retrieve(dataset, request, str(scratch))
        _extract_netcdf(scratch, target)
        scratch = None
        # What this file holds, so a later request for a smaller area can be
        # cut from it instead of queued again (`find_enclosing`).
        target.with_suffix(".json").write_text(json.dumps({"dataset": dataset, "request": request}))

        _write_job(job_id, state=JOB_DONE, path=str(target), message="")
    except CDSDataError as exc:
        _write_job(job_id, state=JOB_FAILED, message=str(exc), path="")
    except Exception as exc:  # noqa: BLE001
        # `cdsapi` raises bare `Exception` for every server-side refusal —
        # unaccepted licences, invalid parameter combinations, queue errors —
        # so there is no narrower type to catch here. The message is the only
        # thing that distinguishes them, and it goes straight to the user.
        message = str(exc)
        if _CDS_QUEUE_LIMIT_TEXT in message:
            message = (
                "Copernicus rejected the request: too many requests for this "
                "dataset are already queued under this account (possibly from "
                "another worker or session). Wait for them to finish and press "
                "again — completed files are cached and will not be re-requested."
            )
        _write_job(job_id, state=JOB_FAILED, message=message, path="")
    finally:
        if scratch is not None:
            scratch.unlink(missing_ok=True)
        with _live_lock:
            _live_jobs.discard(job_id)


def _encloses(outer: list | None, inner: list) -> bool:
    """Whether a CDS `[N, W, S, E]` area (None = global) contains another."""
    if outer is None:
        return True
    n, w, s_, e = outer
    n2, w2, s2, e2 = inner
    if w > e or w2 > e2:  # a box across the dateline: only the global field is trusted
        return False
    return n >= n2 and s_ <= s2 and w <= w2 and e >= e2


def find_enclosing(dataset: str, request: dict) -> str | None:
    """A finished retrieval identical to `request` except for an area that contains it.

    Everything but `area` must match — including `grid`, so a regridded global
    field never stands in for a native-resolution box. The smallest enclosing
    area wins. A request with no area (the global field) is only ever served by
    its own exact file.
    """
    want = request.get("area")
    if want is None:
        return None
    rest = {k: v for k, v in request.items() if k != "area"}
    best, best_size = None, None
    for sidecar in _cache_dir().glob("*.json"):
        if not sidecar.with_suffix(".nc").is_file():
            continue
        try:
            held = json.loads(sidecar.read_text())
        except (OSError, ValueError):
            continue
        cached = held.get("request", {})
        if held.get("dataset") != dataset or {k: v for k, v in cached.items() if k != "area"} != rest:
            continue
        area = cached.get("area")
        if not _encloses(area, want):
            continue
        size = 360.0 * 180.0 if area is None else (area[0] - area[2]) * (area[3] - area[1])
        if best_size is None or size < best_size:
            best, best_size = sidecar.stem, size
    return best


def submit_or_reuse(dataset_key: str, request: dict) -> tuple[str, bool]:
    """`submit`, unless a cached larger field already holds this area: `(job_id, cut_from_cache)`."""
    dataset = CDS_DATASETS[dataset_key]["dataset"]
    if not _target_path(request_hash(dataset, request)).is_file():
        enclosing = find_enclosing(dataset, request)
        if enclosing:
            _write_job(enclosing, state=JOB_DONE, path=str(_target_path(enclosing)), message="")
            return enclosing, True
    return submit(dataset_key, request), False


def _crop(da, lat_name: str, lon_name: str, crop: tuple | None):
    """Cut `(lat_min, lat_max, lon_min, lon_max)` out of a field (lon_min > lon_max wraps)."""
    if crop is None:
        return da
    lat_min, lat_max, lon_min, lon_max = crop
    lat = np.asarray(da[lat_name].values)
    lon = np.asarray(da[lon_name].values)
    keep_lat = (lat >= lat_min) & (lat <= lat_max)
    keep_lon = (
        (lon >= lon_min) & (lon <= lon_max)
        if lon_min <= lon_max
        else (lon >= lon_min) | (lon <= lon_max)
    )
    if not keep_lat.any() or not keep_lon.any():
        raise CDSDataError("The area lies outside the cached field it was to be cut from.")
    return da.isel({lat_name: np.flatnonzero(keep_lat), lon_name: np.flatnonzero(keep_lon)})


def submit(dataset_key: str, request: dict) -> str:
    """Start (or join) a retrieval and return its job id.

    Idempotent by request content: an identical request already on disk
    returns immediately as completed, and one already running is joined
    rather than duplicated. Submitting is therefore safe to do on every click.

    Raises
    ------
    CDSDataError
        If the store has no token configured, so the page can report that
        before a job is ever queued.
    """
    meta = CDS_DATASETS[dataset_key]
    store = meta["store"]
    dataset = meta["dataset"]

    if credentials(store) is None:
        raise CDSDataError(credentials_message(store))

    job_id = request_hash(dataset, request)

    if _target_path(job_id).is_file():
        _write_job(job_id, state=JOB_DONE, path=str(_target_path(job_id)), message="")
        return job_id

    record = cache.get(_JOB_KEY_PREFIX + job_id) or {}
    if record.get("state") in ("queued", "running"):
        age = time.time() - record.get("updated_at", 0)
        if age < CDS_JOB_TIMEOUT_S:
            return job_id
        # A record left behind by a worker that died mid-retrieval would
        # otherwise strand this request forever.
        _write_job(
            job_id, state=JOB_FAILED, message="The retrieval timed out.", path=""
        )

    with _live_lock:
        if job_id in _live_jobs:
            return job_id
        _live_jobs.add(job_id)

    _write_job(
        job_id,
        state="queued",
        message="Waiting for the Copernicus queue.",
        path="",
        submitted_at=time.time(),
        dataset=dataset,
        store=store,
    )
    threading.Thread(
        target=_run_job,
        args=(job_id, store, dataset, request),
        name=f"cds-{job_id}",
        daemon=True,
    ).start()

    return job_id


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Coordinate names that are not latitude/longitude and are safe to reduce
# away. `expver` is the ERA5/ERA5T marker: a request spanning the boundary
# returns both, which breaks any code assuming a 2-D field.
_LAT_NAMES = ("latitude", "lat")
_LON_NAMES = ("longitude", "lon")
_TIME_NAMES = (
    "valid_time",
    "time",
    "forecast_period",
    "forecast_reference_time",
    "step",
)


def _axis_name(obj, candidates: tuple[str, ...]) -> str | None:
    for name in candidates:
        if name in obj.dims or name in obj.coords:
            return name
    return None


def _resolve_variable(ds, var_meta: dict, request_name: str) -> str:
    """Find the field in the file, trying declared short names then falling back.

    The CDS request name is almost never the NetCDF variable name, and the
    mapping has changed across CDS generations, so a file is matched against
    the declared candidates first and only then against its own contents.
    """
    for candidate in var_meta.get("nc_names", ()):
        if candidate in ds.data_vars:
            return candidate

    lowered = request_name.lower()
    for name in ds.data_vars:
        if name.lower() in lowered or lowered.startswith(name.lower()):
            return name

    fields = [name for name in ds.data_vars if ds[name].ndim >= 2]
    if not fields:
        raise CDSDataError("The retrieved file contains no gridded field.")
    return fields[0]


def _check_step_type(da, var_meta: dict, nc_name: str) -> None:
    """Verify the file's GRIB stepType matches what the variable's scale assumes.

    Accumulated fields are the easiest thing in this module to be wrong about
    by an exact factor, and a wrong factor still produces a perfectly
    plausible-looking map. ERA5's hourly product stores `accum` (a total over
    the step) while the monthly product stores `avgad` (an average *daily*
    accumulation) — treating the latter as the former, or vice versa, is a
    silent factor of 24.

    The check is advisory in one direction only: a file that does not carry
    the attribute at all is accepted, since not every source writes it. A file
    that carries a *different* one is refused, because that means the scale
    being applied was derived for a different quantity.
    """
    expected = var_meta.get("expected_step_type")
    if not expected:
        return

    actual = da.attrs.get("GRIB_stepType") or da.attrs.get("step_type")
    if actual and actual != expected:
        raise CDSDataError(
            f"'{nc_name}' has GRIB stepType '{actual}' but the unit conversion "
            f"for this variable assumes '{expected}'. Refusing to scale it — "
            "the result would be wrong by a constant factor with nothing to "
            "show for it."
        )


def _normalise_longitude(da, lon_name: str):
    """Wrap longitudes to -180..180 and re-sort.

    CAMS Europe stores its domain as 335°E..45°E, which wraps the prime
    meridian: plotted as-is the map is cut in half and reassembled in the
    wrong order.
    """
    lons = np.asarray(da[lon_name].values, dtype=float)
    if lons.max() <= 180.0:
        return da
    return da.assign_coords({lon_name: (((lons + 180.0) % 360.0) - 180.0)}).sortby(
        lon_name
    )


def _reduce_to_field(da, lat_name: str, lon_name: str) -> tuple[object, dict]:
    """Collapse everything that is not latitude/longitude down to a single 2-D field.

    Length-1 dimensions (`level`, `expver` on a clean request, the nominal
    monthly `time`) are squeezed. A remaining time-like dimension of length
    greater than one means the response carried more steps than were asked
    for; the first is taken and the fact is recorded so the caller can say so
    rather than silently presenting one step as the answer.
    """
    notes: dict = {}

    for dim in list(da.dims):
        if dim in (lat_name, lon_name):
            continue
        if da.sizes[dim] == 1:
            da = da.squeeze(dim, drop=True)

    for dim in list(da.dims):
        if dim in (lat_name, lon_name):
            continue
        notes["extra_dim"] = f"{dim} ({da.sizes[dim]} steps, first shown)"
        da = da.isel({dim: 0}, drop=True)

    return da, notes


def _round_spacing(spacing: float | None) -> float | None:
    """Round a grid spacing to three significant figures.

    Coordinates come back as float32, so a nominal 0.1° grid measures
    0.0999755859375. Left alone that reaches the plot subtitle, where it reads
    as false precision about a number the dataset states plainly.
    """
    if spacing is None or spacing <= 0:
        return None
    return float(f"{spacing:.3g}")


def _stride(da, lat_name: str, lon_name: str) -> tuple[object, float | None]:
    """Thin a field to the display budget, returning it with its effective spacing.

    Striding, not averaging: an averaged cell is a quantity the dataset does
    not contain, and the hover readout would then report a number no gridbox
    holds. The effective spacing is returned so the plot can state the
    resolution it is actually showing.
    """
    n_cells = da.sizes[lat_name] * da.sizes[lon_name]
    if n_cells <= MAX_DISPLAY_CELLS:
        return da, None

    step = int(np.ceil(np.sqrt(n_cells / MAX_DISPLAY_CELLS)))
    thinned = da.isel(
        {lat_name: slice(None, None, step), lon_name: slice(None, None, step)}
    )

    lons = np.asarray(thinned[lon_name].values, dtype=float)
    spacing = _round_spacing(
        float(np.abs(np.diff(lons)).min()) if lons.size > 1 else None
    )
    return thinned, spacing


def _open(path: str):
    import xarray as xr

    try:
        return xr.open_dataset(path)
    except (OSError, ValueError) as exc:
        raise CDSDataError(f"Could not read the retrieved file ({exc}).") from exc


def _first_datetime(source, name: str) -> pd.Timestamp | None:
    """First value of a coordinate, but only if it really is a datetime.

    The dtype check is the point. CAMS Europe stores forecast *lead hours* in
    a coordinate called `time` — plain floats, 0 to 96 — and
    `pd.Timestamp(0.0)` happily returns the Unix epoch. Presenting 1970-01-01
    as a validity time would be a fabricated label, so a non-datetime axis
    yields nothing and the caller falls back to what the page requested.
    """
    if name not in source.coords:
        return None

    values = np.atleast_1d(np.asarray(source[name].values))
    if values.size == 0 or not np.issubdtype(values.dtype, np.datetime64):
        return None

    stamp = pd.Timestamp(values.flat[0])
    return None if pd.isna(stamp) else stamp


def _first_timedelta(source, name: str) -> pd.Timedelta | None:
    """First value of a coordinate, but only if it really is a duration."""
    if name not in source.coords:
        return None

    values = np.atleast_1d(np.asarray(source[name].values))
    if values.size == 0 or not np.issubdtype(values.dtype, np.timedelta64):
        return None

    delta = pd.Timedelta(values.flat[0])
    return None if pd.isna(delta) else delta


def _valid_time(da, ds) -> pd.Timestamp | None:
    """Best trustworthy validity timestamp for the selected field.

    Ordered by how directly each source answers the question. `valid_time` is
    the answer outright. Failing that, CAMS global files carry the init as
    `forecast_reference_time` and the lead as `forecast_period`, which sum to
    it — taking the reference time alone would label a T+120 h field with its
    init time. `time` is tried last and only when it is genuinely a datetime.
    """
    for source in (da, ds):
        stamp = _first_datetime(source, "valid_time")
        if stamp is not None:
            return stamp

    for source in (da, ds):
        base = _first_datetime(source, "forecast_reference_time")
        if base is None:
            continue
        lead = _first_timedelta(da, "forecast_period") or _first_timedelta(
            ds, "forecast_period"
        )
        return base + lead if lead is not None else base

    for source in (da, ds):
        stamp = _first_datetime(source, "time")
        if stamp is not None:
            return stamp

    return None


def _resolved_unit(
    var_meta: dict, accum_hours: int | None, monthly_mode: str | None
) -> str:
    """Display unit after any accumulation or re-expression has been applied.

    The unit is part of the answer, not decoration: a 24 h total labelled
    "mm/h" and a month total labelled "mm/day" are both wrong by a factor the
    reader has no way to spot.
    """
    if accum_hours and accum_hours > 1:
        return var_meta.get("accum_unit", var_meta["unit"])

    if monthly_mode and var_meta.get("is_monthly_rate"):
        spec = MONTHLY_ACCUM_MODES.get(monthly_mode, {})
        if monthly_mode != "per_day":
            # "mm/day" -> "mm per month": the per-day denominator is consumed
            # by the multiplication, so it must not survive into the label.
            base = var_meta["unit"].split("/day")[0]
            return f"{base}{spec.get('unit_suffix', '')}"

    return var_meta["unit"]


def monthly_rate_factor(mode: str, stamp: pd.Timestamp | None) -> tuple[float, str]:
    """Multiplier and unit suffix for re-expressing a monthly mean rate.

    A monthly mean of an hourly accumulation is a *rate*, already scaled to a
    per-day figure by the variable's `scale`. It cannot be accumulated over a
    6 or 12 h window — there is no sub-monthly information left in it. What a
    reader does want is to see it as a daily mean or as a month total, which
    is a unit change, not an accumulation.

    The month total uses the calendar length of the month the field belongs
    to, never a nominal 30: February and July differ by 10%, which is larger
    than most of the anomalies being looked at.

    Returns
    -------
    tuple[float, str]
        Factor to apply to the per-day value, and the suffix to append to the
        per-day unit.
    """
    spec = MONTHLY_ACCUM_MODES.get(mode)
    if spec is None or mode == "per_day":
        return 1.0, ""

    if stamp is None or pd.isna(stamp):
        raise CDSDataError(
            "The file carries no month, so a monthly total cannot be formed "
            "from a daily mean."
        )
    days = int(pd.Timestamp(stamp).days_in_month)
    return float(days), f" (× {days} days)"


def _sum_window(da, time_name: str, window_start: pd.Timestamp, hours: int):
    """Sum an hourly accumulation over `(window_start, window_start + hours]`.

    Right-closed on purpose. ERA5's hourly accumulation at hour H covers
    `(H-1, H]`, so the fields belonging to a 00-24 UTC day are 01:00 through
    00:00 of the following day — the left edge is excluded and the right edge
    included. Getting this inclusive on the wrong side shifts the whole day by
    an hour, which is invisible in a map and wrong in every total.
    """
    window_end = window_start + pd.Timedelta(hours=hours)
    stamps = pd.to_datetime(np.asarray(da[time_name].values))
    mask = (stamps > window_start) & (stamps <= window_end)

    if int(mask.sum()) != hours:
        raise CDSDataError(
            f"The retrieved file covers {int(mask.sum())} of the {hours} hours "
            f"needed for the window ending {window_end:%Y-%m-%d %H:%M} UTC."
        )

    return da.isel({time_name: np.flatnonzero(mask)}).sum(time_name, keep_attrs=True)


@cache.memoize(timeout=3600)
def load_field(
    path: str,
    dataset_key: str,
    variable: str,
    accum_hours: int | None = None,
    window_start: str | None = None,
    monthly_mode: str | None = None,
    crop: tuple | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Parse a retrieved NetCDF into a tidy field frame.

    Applies, in order: variable resolution by short name, longitude
    unwrapping, reduction of every non-spatial dimension, unit conversion, and
    thinning to the display budget.

    Parameters
    ----------
    path : str
        File written by a completed retrieval.
    dataset_key : str
        Key into `utils.settings.CDS_DATASETS`.
    variable : str
        CDS/ADS request name, keying `utils.settings.CDS_VARIABLES`.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The frame has one row per grid cell: `latitude`, `longitude`,
        `value`. The dict carries what the plot subtitle must state —
        `valid_time`, `n_lat`, `n_lon`, `extent`, `resolution_deg`,
        `display_resolution_deg`, `unit`, the NetCDF variable actually read,
        and `accum_hours`/`accum_window` when a window was summed.
    """
    var_meta = CDS_VARIABLES[dataset_key][variable]

    ds = _open(path)
    try:
        nc_name = _resolve_variable(ds, var_meta, variable)
        da = ds[nc_name]
        _check_step_type(da, var_meta, nc_name)

        lat_name = _axis_name(da, _LAT_NAMES)
        lon_name = _axis_name(da, _LON_NAMES)
        if lat_name is None or lon_name is None:
            raise CDSDataError(
                f"'{nc_name}' has no latitude/longitude axes ({list(da.dims)})."
            )

        da = _normalise_longitude(da, lon_name)
        da = _crop(da, lat_name, lon_name, crop)

        notes: dict = {}
        accum_window = None
        if accum_hours and accum_hours > 1:
            time_name = _axis_name(da, _TIME_NAMES)
            if time_name is None or time_name not in da.dims:
                raise CDSDataError(
                    "The retrieved file has no time axis to accumulate over."
                )
            start = pd.Timestamp(window_start)
            da = _sum_window(da, time_name, start, accum_hours)
            accum_window = (start, start + pd.Timedelta(hours=accum_hours))

        da, reduce_notes = _reduce_to_field(da, lat_name, lon_name)
        notes.update(reduce_notes)
        valid = accum_window[1] if accum_window else _valid_time(da, ds)

        native_lons = np.asarray(da[lon_name].values, dtype=float)
        native_res = _round_spacing(
            float(np.abs(np.diff(native_lons)).min()) if native_lons.size > 1 else None
        )

        da, display_res = _stride(da, lat_name, lon_name)
        values = da.transpose(lat_name, lon_name).values.astype(float)
        values = values * var_meta["scale"] + var_meta["offset"]

        monthly_factor, monthly_suffix = 1.0, ""
        if monthly_mode and var_meta.get("is_monthly_rate"):
            monthly_factor, monthly_suffix = monthly_rate_factor(monthly_mode, valid)
            values = values * monthly_factor

        lat = np.asarray(da[lat_name].values, dtype=float)
        lon = np.asarray(da[lon_name].values, dtype=float)
    finally:
        ds.close()

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
        raise CDSDataError("The retrieved field is entirely missing over this area.")

    meta = {
        "valid_time": valid,
        "n_lat": int(lat.size),
        "n_lon": int(lon.size),
        "extent": (
            float(lat.min()),
            float(lat.max()),
            float(lon.min()),
            float(lon.max()),
        ),
        "resolution_deg": native_res or CDS_DATASETS[dataset_key]["resolution_deg"],
        "display_resolution_deg": display_res,
        "unit": _resolved_unit(var_meta, accum_hours, monthly_mode),
        "nc_variable": nc_name,
        "accum_hours": accum_hours,
        "accum_window": accum_window,
        "monthly_mode": monthly_mode if var_meta.get("is_monthly_rate") else None,
        "monthly_note": monthly_suffix,
        **notes,
    }
    return df, meta


@cache.memoize(timeout=3600)
def load_monthly_anomaly(
    field_path: str,
    reference_path: str,
    dataset_key: str,
    variable: str,
    monthly_mode: str | None = None,
    crop: tuple | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Field minus the same calendar month averaged over the reference period.

    Both files are parsed through `load_field`, so the reference is thinned
    identically to the field and the two grids line up cell for cell. The
    reference request asks for one month across N years, so the time mean
    taken here is a mean over years of that one month — which is what the
    subtitle claims and what an anomaly against a climatological month means.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Same shape as `load_field`; the metadata gains `is_anomaly=True` and
        `reference_years`.
    """
    # Both sides must be expressed over the same window, or the difference is
    # between two different quantities.
    field, meta = load_field(
        field_path, dataset_key, variable, monthly_mode=monthly_mode, crop=crop
    )
    reference, ref_meta = _load_reference_mean(
        reference_path, dataset_key, variable, monthly_mode=monthly_mode, crop=crop
    )

    merged = field.merge(
        reference.rename(columns={"value": "reference"}),
        on=["latitude", "longitude"],
        how="inner",
        suffixes=("", "_ref"),
    )
    if merged.empty:
        raise CDSDataError(
            "The field and its reference period do not share a grid — retrieve "
            "both over the same area."
        )

    merged["value"] = merged["value"] - merged["reference"]
    out = merged[["latitude", "longitude", "value"]].copy()
    out["variable"] = variable

    meta = {
        **meta,
        "is_anomaly": True,
        "reference_years": ref_meta["reference_years"],
        "n_lat": int(out["latitude"].nunique()),
        "n_lon": int(out["longitude"].nunique()),
    }
    return out, meta


@cache.memoize(timeout=3600)
def _load_reference_mean(
    path: str, dataset_key: str, variable: str, monthly_mode: str | None = None,
    crop: tuple | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Mean over years of a single calendar month, as a tidy frame.

    Separate from `load_field` because the reference file keeps its time
    dimension — `_reduce_to_field` would take the first year and call it the
    climatology.
    """
    var_meta = CDS_VARIABLES[dataset_key][variable]

    ds = _open(path)
    try:
        nc_name = _resolve_variable(ds, var_meta, variable)
        da = ds[nc_name]
        _check_step_type(da, var_meta, nc_name)

        lat_name = _axis_name(da, _LAT_NAMES)
        lon_name = _axis_name(da, _LON_NAMES)
        if lat_name is None or lon_name is None:
            raise CDSDataError(f"'{nc_name}' has no latitude/longitude axes.")

        da = _normalise_longitude(da, lon_name)
        da = _crop(da, lat_name, lon_name, crop)

        time_name = _axis_name(da, _TIME_NAMES)
        years: list[int] = []
        month: int | None = None
        if time_name is not None and time_name in da.dims:
            stamps = pd.DatetimeIndex(pd.to_datetime(np.asarray(da[time_name].values)))
            years = sorted({int(y) for y in stamps.year})
            months = {int(m) for m in stamps.month}
            # Every field in a reference request is the same calendar month;
            # more than one means the request was built wrong and the mean
            # below would be across months, not across years.
            if len(months) != 1:
                raise CDSDataError(
                    f"The reference file spans {len(months)} calendar months; "
                    "a climatological month mean needs exactly one."
                )
            month = months.pop()
            da = da.mean(time_name, keep_attrs=True)

        da, _ = _reduce_to_field(da, lat_name, lon_name)
        da, _ = _stride(da, lat_name, lon_name)

        values = da.transpose(lat_name, lon_name).values.astype(float)
        values = values * var_meta["scale"] + var_meta["offset"]

        if monthly_mode and var_meta.get("is_monthly_rate"):
            # The reference is a mean over N years of one calendar month, so
            # that month's length is what re-expresses it — taken from the
            # month itself, not from whichever year happened to be first.
            month_stamp = pd.Timestamp(year=2001, month=int(month), day=1) if month else None
            factor, _ = monthly_rate_factor(monthly_mode, month_stamp)
            values = values * factor

        lat = np.asarray(da[lat_name].values, dtype=float)
        lon = np.asarray(da[lon_name].values, dtype=float)
    finally:
        ds.close()

    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    df = pd.DataFrame(
        {
            "latitude": lat_grid.reshape(-1),
            "longitude": lon_grid.reshape(-1),
            "value": values.reshape(-1),
        }
    ).dropna(subset=["value"])

    if not years:
        raise CDSDataError(
            "The reference file carries no time axis, so no climatological "
            "mean can be taken from it."
        )

    return df, {"reference_years": (min(years), max(years), len(years))}


@cache.memoize(timeout=3600)
def load_series(path: str, spec: dict, crop: tuple | None = None) -> xr.DataArray:
    """Every time step of a monthly file, as a `(time, latitude, longitude)` array.

    For the static gallery, whose products need the individual years (a
    correlation) or a month and its reference kept apart — `load_field` takes
    the first step and `_load_reference_mean` averages them away.

    Parameters
    ----------
    path : str
        File written by a completed retrieval.
    spec : dict
        `nc_names`, `scale`, `offset`, and optionally `expected_step_type` —
        a `GALLERY_PRODUCTS` entry, or one of its `targets`.

    Returns
    -------
    xr.DataArray
        Converted to the display unit, longitudes in -180..180, latitude
        ascending, with a `time` dimension even for a single month.
    """
    ds = _open(path)
    try:
        nc_name = _resolve_variable(ds, spec, spec["nc_names"][0])
        da = ds[nc_name]
        _check_step_type(da, spec, nc_name)

        lat_name = _axis_name(da, _LAT_NAMES)
        lon_name = _axis_name(da, _LON_NAMES)
        time_name = _axis_name(da, _TIME_NAMES)
        if lat_name is None or lon_name is None:
            raise CDSDataError(f"'{nc_name}' has no latitude/longitude axes.")
        if time_name is None or time_name not in da.dims:
            raise CDSDataError(f"'{nc_name}' has no time axis.")

        da = _normalise_longitude(da, lon_name)
        da = _crop(da, lat_name, lon_name, crop)
        for dim in list(da.dims):
            if dim not in (lat_name, lon_name, time_name) and da.sizes[dim] == 1:
                da = da.squeeze(dim, drop=True)
        extra = [d for d in da.dims if d not in (lat_name, lon_name, time_name)]
        if extra:
            raise CDSDataError(
                f"'{nc_name}' carries unexpected dimensions {extra}; refusing to "
                "guess which slice was meant."
            )

        da = (
            da.rename({lat_name: "latitude", lon_name: "longitude", time_name: "time"})
            .transpose("time", "latitude", "longitude")
            .sortby("latitude")
            .load()
        )
    finally:
        ds.close()

    out = da * spec["scale"] + spec["offset"]
    out.attrs = {}
    return out


def _seasonal_means(da, season: str, var_meta: dict, monthly_mode: str | None):
    """Complete seasons of a monthly series, one value per season-year.

    Months are weighted by their length: a plain mean of three monthly means
    over-weights February. With `monthly_mode="per_month"` a rate is
    expressed as the season total instead (sum of month rate × days).
    DJF belongs to the year of its January.
    """
    import xarray as xr

    stamps = pd.DatetimeIndex(pd.to_datetime(da["time"].values))
    months = SEASONS[season]
    keep = np.isin(stamps.month, months)
    stamps = stamps[keep]
    season_year = stamps.year + ((stamps.month == 12) & (season == "DJF")).astype(int)
    da = da.isel(time=np.flatnonzero(keep)).assign_coords(
        season_year=("time", np.asarray(season_year))
    )
    days = xr.DataArray(np.asarray(stamps.days_in_month, dtype=float), dims="time")
    weighted = (da * days).groupby("season_year").sum("time")
    n_months = da["time"].groupby("season_year").count()
    total_days = days.assign_coords(season_year=da["season_year"]).groupby("season_year").sum()
    as_total = monthly_mode == "per_month" and var_meta.get("is_monthly_rate")
    values = weighted if as_total else weighted / total_days
    return values.where(n_months == 3, drop=True)


@cache.memoize(timeout=3600)
def load_seasonal(
    field_path: str,
    dataset_key: str,
    variable: str,
    season: str,
    year: int,
    reference_path: str | None = None,
    reference: tuple[int, int] | None = None,
    monthly_mode: str | None = None,
    crop: tuple | None = None,
) -> tuple[pd.DataFrame, dict]:
    """A seasonal mean (or anomaly) map from monthly files, as `load_field` returns.

    Parameters
    ----------
    field_path
        The three months of the season (for DJF, a request spanning both
        years; only the right three are used).
    reference_path, reference
        When given, the result is the anomaly against the mean of the same
        season over the reference years, complete seasons only.
    """
    var_meta = CDS_VARIABLES[dataset_key][variable]
    spec = {key: var_meta.get(key) for key in ("nc_names", "scale", "offset", "expected_step_type")}

    seasons = _seasonal_means(load_series(field_path, spec, crop), season, var_meta, monthly_mode)
    if year not in seasons["season_year"].values:
        raise CDSDataError(f"The retrieved file does not hold a complete {season} {year}.")
    field = seasons.sel(season_year=year, drop=True)

    meta: dict = {}
    unit = var_meta["unit"]
    if monthly_mode == "per_month" and var_meta.get("is_monthly_rate"):
        unit = f"{unit.split('/day')[0]} per season"
    if reference_path:
        ref = _seasonal_means(load_series(reference_path, spec, crop), season, var_meta, monthly_mode)
        years = ref["season_year"].values
        ref = ref.sel(season_year=years[(years >= reference[0]) & (years <= reference[1])])
        if ref.sizes["season_year"] < 10:
            raise CDSDataError("Fewer than 10 complete reference seasons were retrieved.")
        field = field - ref.mean("season_year")
        used = ref["season_year"].values
        meta["is_anomaly"] = True
        meta["reference_years"] = (int(used.min()), int(used.max()), int(len(used)))
        if not (monthly_mode == "per_month" and var_meta.get("is_monthly_rate")):
            unit = var_meta.get("anomaly_unit", unit)

    lons = np.asarray(field["longitude"].values, dtype=float)
    native = _round_spacing(float(np.abs(np.diff(lons)).min()) if lons.size > 1 else None)
    field, display = _stride(field, "latitude", "longitude")
    lat = np.asarray(field["latitude"].values, dtype=float)
    lon = np.asarray(field["longitude"].values, dtype=float)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    df = pd.DataFrame(
        {
            "latitude": lat_grid.reshape(-1),
            "longitude": lon_grid.reshape(-1),
            "value": field.transpose("latitude", "longitude").values.reshape(-1),
        }
    ).dropna(subset=["value"])
    if df.empty:
        raise CDSDataError("The seasonal field is entirely missing over this area.")

    start_month = SEASONS[season][0]
    meta.update(
        {
            "valid_time": pd.Timestamp(year - (season == "DJF"), start_month, 1),
            "n_lat": int(lat.size),
            "n_lon": int(lon.size),
            "extent": (float(lat.min()), float(lat.max()), float(lon.min()), float(lon.max())),
            "resolution_deg": native or CDS_DATASETS[dataset_key]["resolution_deg"],
            "display_resolution_deg": display,
            "unit": unit,
            "monthly_mode": monthly_mode if var_meta.get("is_monthly_rate") else None,
            "season": season,
        }
    )
    return df, meta
