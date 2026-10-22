"""Precomputed preset animations: written by `scripts/precompute.py`, served by URL.

The job renders each preset (`PRECOMPUTE_*` in settings) with the same code
the page uses (`map_products.forecast_animation`), dark theme only, and writes
WebP frames under `PRECOMPUTE_DIR` with a `latest.json` manifest. The page
asks `preset_animation` first: a request that matches a preset whose run is
still the newest gets frame URLs and renders nothing; anything else — a custom
box, another palette, projection or step, several models, a past run — is
rendered live as before.

Only the default palette is precomputed: its levels are fixed per parameter,
so a precomputed frame and a live one mean the same colours.
"""

from __future__ import annotations

import base64
import copy
import functools
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from utils.earthmover_api import (
    GridDataError,
    area_bounds,
    available_inits,
    frame_cube,
    frame_statistic,
    read_estimate,
    resolve_run,
    run_calendar,
    store_variables,
)
from utils import hub_sync
from utils.map_products import forecast_animation
from utils.settings import (
    GRID_STORES,
    PRECOMPUTE_DIR,
    PRECOMPUTE_HORIZON_HOURS,
    PRECOMPUTE_REGIONS,
    PRECOMPUTE_STATISTICS,
    PRECOMPUTE_STEP_HOURS,
    PRECOMPUTE_VARIABLES,
)

_log = logging.getLogger(__name__)

_MANIFEST = "latest.json"
_PRESET_VIEW = {"projection": "auto", "extent_key": "data", "palette": "default"}


def _stamp(init: pd.Timestamp) -> str:
    return f"{pd.Timestamp(init):%Y%m%d%H}"


def _root() -> Path:
    return Path(PRECOMPUTE_DIR)


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.part")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_animation(
    store: str,
    init: pd.Timestamp,
    region: str,
    variable: str,
    statistic: str,
    animation: dict,
    accum_hours: int | None = None,
) -> dict:
    """Write one single-model animation's frames as WebP; return its manifest entry."""
    folder = Path("frames", store, _stamp(init), region, variable, statistic)
    frames = []
    for frame in animation["frames"]:
        png = base64.b64decode(frame["srcs"][0].split(",", 1)[1])
        with Image.open(io.BytesIO(png)) as img:
            buf = io.BytesIO()
            img.save(buf, format="WEBP", quality=85, method=6)
        rel = folder / f"T{int(frame['lead_hours']):03d}.webp"
        _atomic_bytes(_root() / rel, buf.getvalue())
        frames.append({"lead_hours": frame["lead_hours"], "valid": frame["valid"], "path": rel.as_posix()})
    return {
        "frames": frames,
        "caption": animation["caption"],
        "filename": animation["filename"],
        "panel": animation["panels"][0],
        "accum_hours": accum_hours,
    }


def write_fields(
    store: str,
    init: pd.Timestamp,
    region: str,
    variable: str,
    statistic: str,
    values: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    leads: list[int],
    valid: list,
    unit: str,
    description: str,
) -> str:
    """Save a preset's statistic fields (frame, lat, lon; float32) beside its frames.

    A smaller area inside the preset region is then drawn from these, at the
    resolution the preset holds, with no archive read (`fields_covering`).
    """
    rel = Path("frames", store, _stamp(init), region, variable, statistic, "fields.npz")
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        values=np.asarray(values, dtype=np.float32),
        lat=np.asarray(lat, dtype=float),
        lon=np.asarray(lon, dtype=float),
        leads=np.asarray(leads, dtype=int),
        valid=np.asarray(pd.to_datetime(valid).astype("datetime64[s]").astype(np.int64)),
        unit=np.asarray(unit),
        description=np.asarray(description),
    )
    _atomic_bytes(_root() / rel, buf.getvalue())
    return rel.as_posix()


@functools.lru_cache(maxsize=8)
def _load_fields(path: str, mtime: float) -> dict:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def fields_covering(
    bounds: tuple,
    variable: str,
    init_time: str | None,
    step_hours: int,
    horizon_hours: int,
    accum_hours: int | None,
    statistic: str,
    store: str,
) -> dict | None:
    """A frame cube cut from a preset's saved fields, or None to read the archive.

    Shaped like `earthmover_api.frame_cube`'s, with the statistic as its single
    "member" (`precomputed_statistic`, `description`), so the animation renders
    it unchanged. Only when the run is the preset's, the step and window match,
    the statistic was precomputed, and the area lies inside the preset region.
    """
    if step_hours != PRECOMPUTE_STEP_HOURS:
        return None
    manifest = load_manifest()
    entry = manifest["stores"].get(store)
    if not entry:
        return None
    lat_min, lat_max, lon_min, lon_max = bounds
    init = None  # asked for only once a product could serve: it reads the store's calendar
    for region in PRECOMPUTE_REGIONS:
        product = entry["products"].get(f"{region}/{variable}/{statistic}")
        if not product or not product.get("fields") or product.get("accum_hours") != accum_hours:
            continue
        path = _root() / product["fields"]
        if not path.exists():
            continue
        data = _load_fields(str(path), path.stat().st_mtime)
        lat, lon = data["lat"], data["lon"]
        if not (lat.min() <= lat_min and lat_max <= lat.max() and lon.min() <= lon_min and lon_max <= lon.max()):
            continue
        init = pd.Timestamp(init_time) if init_time else current_init((store,), None)
        if init is None or not is_current(manifest, store, init):
            return None
        keep_lat = np.flatnonzero((lat >= lat_min) & (lat <= lat_max))
        keep_lon = np.flatnonzero((lon >= lon_min) & (lon <= lon_max))
        frames = np.flatnonzero(data["leads"] <= horizon_hours)
        if not keep_lat.size or not keep_lon.size or not frames.size:
            return None
        values = data["values"][np.ix_(frames, keep_lat, keep_lon)].astype(np.float32)
        return {
            "cube": values[:, None, :, :],
            "lead_hours": data["leads"][frames].astype(int).tolist(),
            "valid_time": pd.to_datetime(data["valid"][frames], unit="s").tolist(),
            "lat": lat[keep_lat],
            "lon": lon[keep_lon],
            "members": [0],
            "init_time": init,
            "stride": 1,
            "bounds": (float(lat[keep_lat].min()), float(lat[keep_lat].max()),
                       float(lon[keep_lon].min()), float(lon[keep_lon].max())),
            "region": None,
            "unit": str(data["unit"]),
            "store": store,
            "clip_note": None,
            "precomputed_statistic": statistic,
            "description": str(data["description"]),
            "cut_from_larger": bool(keep_lat.size < lat.size or keep_lon.size < lon.size),
        }
    return None


def load_manifest() -> dict:
    """The manifest, or an empty one."""
    hub_sync.sync()
    path = _root() / _MANIFEST
    if not path.exists():
        return {"stores": {}}
    return json.loads(path.read_text())


def save_manifest(manifest: dict) -> None:
    manifest = {**manifest, "generated": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M}Z"}
    _atomic_bytes(_root() / _MANIFEST, json.dumps(manifest, indent=1).encode())


def record(
    manifest: dict,
    store: str,
    init: pd.Timestamp,
    region: str,
    variable: str,
    statistic: str,
    entry: dict,
) -> None:
    """Add an entry; a newer run of the store replaces all of its older entries."""
    iso = f"{pd.Timestamp(init):%Y-%m-%dT%H:%M}"
    current = manifest["stores"].get(store)
    if not current or current["init"] != iso:
        current = manifest["stores"][store] = {"init": iso, "products": {}}
    current["products"][f"{region}/{variable}/{statistic}"] = entry


def is_current(manifest: dict, store: str, init: pd.Timestamp) -> bool:
    entry = manifest["stores"].get(store)
    return bool(entry) and entry["init"] == f"{pd.Timestamp(init):%Y-%m-%dT%H:%M}"


def merge_manifest(base: dict, part: dict) -> dict:
    """`base` with each of `part`'s stores where `part` holds a run at least as new.

    The publish job merges every producer job's manifest into the one mirrored
    from the Dataset: a job that failed or skipped a store leaves that store's
    published entry (and so its frames) in place.
    """
    out = {"stores": copy.deepcopy(base.get("stores", {}))}
    for store, entry in part.get("stores", {}).items():
        held = out["stores"].get(store)
        if held is None or entry["init"] >= held["init"]:
            out["stores"][store] = copy.deepcopy(entry)
    return out


def current_init(stores: tuple[str, ...], run: str | None) -> pd.Timestamp | None:
    """The init a live request would draw: the run asked for, or the newest finished."""
    resolved = resolve_run(stores, run)
    if resolved:
        return pd.Timestamp(resolved)
    unfinished = set(run_calendar(stores[0])["unfinished"])
    finished = [t for t in available_inits(stores[0]) if f"{t:%Y-%m-%dT%H:%M}" not in unfinished]
    return finished[-1] if finished else None


def preset_animation(
    area_key: str,
    variable: str,
    run: str | None,
    step_hours: int,
    horizon_hours: int,
    accum_hours: int | None,
    statistic: str,
    projection: str,
    extent_key: str,
    palette: str,
    stores: tuple[str, ...],
    base_url: str,
    levels: str = "auto",
) -> dict | None:
    """A precomputed animation shaped like `forecast_animation`'s, or None to render live."""
    view = {"projection": projection, "extent_key": extent_key, "palette": palette}
    if (
        len(stores) != 1
        or levels == "fitted"  # frames were drawn with Default levels
        or area_key not in PRECOMPUTE_REGIONS
        or step_hours != PRECOMPUTE_STEP_HOURS
        or view != _PRESET_VIEW
    ):
        return None
    store = stores[0]
    manifest = load_manifest()
    entry = manifest["stores"].get(store)
    if not entry:
        return None
    product = entry["products"].get(f"{area_key}/{variable}/{statistic}")
    init = current_init(stores, run)
    if not product or init is None or not is_current(manifest, store, init):
        return None
    if product.get("accum_hours") != accum_hours:
        return None
    frames = [
        {
            "lead_hours": f["lead_hours"],
            "valid": f["valid"],
            "srcs": [f"{base_url}{f['path']}"],
        }
        for f in product["frames"]
        if f["lead_hours"] <= horizon_hours
    ]
    if not frames:
        return None
    return {
        "panels": [product["panel"]],
        "frames": frames,
        "errors": [],
        "caption": (
            f"{len(frames)} frames, T+{frames[0]['lead_hours']} to T+{frames[-1]['lead_hours']} h"
            f" · {product['caption'].split(' · ', 1)[1]}"
            if " · " in product["caption"]
            else product["caption"]
        ),
        "filename": product["filename"],
        "precomputed": True,
    }


def _statistics(store: str, variable: str) -> tuple[str, ...]:
    if GRID_STORES[store]["kind"] == "deterministic":
        return ("mean",)  # one member: its field
    return PRECOMPUTE_STATISTICS.get(variable, ("mean",))


def refresh(stores: tuple[str, ...], manifest: dict, force: bool = False) -> int:
    """Render presets for each store's newest finished run; returns animations written.

    Region, then variable, then store: a view (IFS Control) renders straight
    after its source, while the source's frame cube is still in `frame_cube`'s
    small in-process LRU — otherwise the view would read every chunk again.
    A store whose run is already in the manifest is skipped; a preset a store
    cannot serve (a variable it lacks) is logged and skipped. The manifest is
    saved after each animation, so an interrupted job keeps what it finished.
    """
    todo, held = {}, {}
    for store in stores:
        init = current_init((store,), None)
        if init is None:
            _log.warning("%s: no finished run", store)
        elif is_current(manifest, store, init) and not force:
            _log.info("%s: %s already precomputed", store, init)
        else:
            todo[store], held[store] = init, set(store_variables(store))
    # Sources first, each view right after it.
    order = sorted(todo, key=lambda s: (list(stores).index(GRID_STORES[s].get("source_store", s)),
                                        GRID_STORES[s].get("source_store") is not None))
    written = 0
    for region in PRECOMPUTE_REGIONS:
        for variable, accum in PRECOMPUTE_VARIABLES.items():
            for store in order:
                if variable not in held[store]:
                    _log.info("%s: no %s, skipped", store, variable)
                    continue
                init = todo[store]
                for statistic in _statistics(store, variable):
                    try:
                        animation = forecast_animation(
                            region, variable, f"{init:%Y-%m-%dT%H:%M}", PRECOMPUTE_STEP_HOURS,
                            PRECOMPUTE_HORIZON_HOURS, accum, statistic, 90.0, (10.0, 90.0),
                            "auto", "data", "default", (store,), "dark",
                        )
                    except GridDataError as exc:
                        _log.warning("%s %s %s %s: %s", store, region, variable, statistic, exc)
                        continue
                    entry = write_animation(store, init, region, variable, statistic, animation, accum)
                    # The statistic's fields too: a smaller area inside this
                    # region is then drawn from them with no archive read. The
                    # cube is still in frame_cube's LRU from the render above.
                    cube = frame_cube(
                        region, variable, f"{init:%Y-%m-%dT%H:%M}", PRECOMPUTE_STEP_HOURS, accum,
                        store, PRECOMPUTE_HORIZON_HOURS,
                    )
                    values, description = frame_statistic(cube, statistic, 90.0, (10.0, 90.0))
                    entry["fields"] = write_fields(
                        store, init, region, variable, statistic, values, cube["lat"], cube["lon"],
                        cube["lead_hours"], cube["valid_time"], cube["unit"], description,
                    )
                    record(manifest, store, init, region, variable, statistic, entry)
                    save_manifest(manifest)
                    written += 1
                    _log.info("%s %s %s %s: %d frames", store, region, variable, statistic,
                              len(entry["frames"]))
    return written


def refresh_store(store: str, manifest: dict, force: bool = False) -> int:
    """`refresh` for one store."""
    return refresh((store,), manifest, force)


def estimate(stores: tuple[str, ...]) -> list[tuple[str, str, float]]:
    """`(store, region, MB)` one cycle's presets read per variable, from store metadata."""
    out = []
    for store in stores:
        shared = GRID_STORES[store].get("source_store") in stores
        for region in PRECOMPUTE_REGIONS:
            if shared:
                out.append((store, region, 0.0))  # cut from its source's cube
                continue
            _, mb = read_estimate((store,), area_bounds(region), horizon_hours=PRECOMPUTE_HORIZON_HOURS)
            out.append((store, region, mb))
    return out


_CONTENT_TYPES = {".webp": "image/webp", ".json": "application/json", ".parquet": "application/octet-stream"}


def upload(client, bucket: str, roots: tuple[Path, ...] | None = None) -> int:
    """Copy the precomputed tree (and any extra roots) to an S3-compatible bucket.

    `client` is a boto3 S3 client (Cloudflare R2 speaks the S3 API). The
    manifest goes last, so a reader never finds it pointing at frames not
    yet uploaded. Returns the number of files sent.
    """
    root = _root()
    files = [p for p in root.rglob("*") if p.is_file() and not p.name.endswith(".part")]
    manifest = root / _MANIFEST
    ordered = [p for p in files if p != manifest] + ([manifest] if manifest.exists() else [])
    sent = 0
    for path in ordered:
        key = path.relative_to(root).as_posix()
        client.upload_file(
            str(path), bucket, key,
            ExtraArgs={"ContentType": _CONTENT_TYPES.get(path.suffix, "application/octet-stream")},
        )
        sent += 1
    for extra in roots or ():
        for path in Path(extra).rglob("*.parquet"):
            key = f"{Path(extra).name.lstrip('.')}/{path.relative_to(extra).as_posix()}"
            client.upload_file(str(path), bucket, key, ExtraArgs={"ContentType": _CONTENT_TYPES[".parquet"]})
            sent += 1
    return sent
