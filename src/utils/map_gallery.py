"""Maps gallery: a fixed set of ERA5 monthly maps, retrieved and drawn offline.

`scripts/precompute.py --maps` builds each item of `MAP_GALLERY` with the
page code of the full workbench — `cds_api.build_request` for the requests,
`map_products.field_static` for the drawing — dark theme, Default style, and
writes PNGs under `PRECOMPUTE_DIR/maps/` with a `gallery.json` manifest. The
public page only shows those images, so no visitor's request ever reaches
Copernicus and the CDS token lives only where the job runs.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
from pathlib import Path

import pandas as pd

from utils import hub_sync
from utils.settings import (
    CDS_DATASETS,
    CDS_STORES,
    CDS_VARIABLES,
    MAP_AREA_PRESET_LABELS,
    MAP_AREA_PRESETS,
    MAP_GALLERY,
    MAP_GALLERY_DATASET,
    MAP_GALLERY_REFERENCE,
    PRECOMPUTE_DIR,
)

_log = logging.getLogger(__name__)

_SUBDIR = "maps"
_MANIFEST = "gallery.json"
# The static view every gallery image is drawn in (the page's defaults).
_VIEW = {"projection": "auto", "extent_key": "data", "palette": "default", "levels": "default"}


def _root() -> Path:
    return Path(PRECOMPUTE_DIR) / _SUBDIR


def gallery_month(now: pd.Timestamp | None = None) -> pd.Timestamp:
    """First day of the newest month the gallery dataset can serve, by its lag."""
    now = pd.Timestamp.now() if now is None else pd.Timestamp(now)
    lag = CDS_DATASETS[MAP_GALLERY_DATASET]["lag_days"]
    return (now.normalize() - pd.Timedelta(days=lag)).to_period("M").to_timestamp()


def gallery_job(key: str, month: pd.Timestamp) -> tuple[dict, dict[str, dict]]:
    """The page's job dict for one gallery item, and its CDS requests by role.

    Mirrors what the full workbench's Maps page builds for a month/year
    retrieval, so `map_products.field_static` draws it exactly as it would there.
    """
    from utils.cds_api import build_request, monthly_grid

    item = MAP_GALLERY[key]
    dataset_key, variable = MAP_GALLERY_DATASET, item["variable"]
    box = MAP_AREA_PRESETS[item["area"]]
    grid = monthly_grid(dataset_key, box)
    month = pd.Timestamp(month)
    reference = [int(p) for p in MAP_GALLERY_REFERENCE.split("-")]

    requests = {
        "field": build_request(
            dataset_key, variable, box, year=month.year, month=month.month, grid=grid
        )
    }
    if item["anomaly"]:
        requests["reference"] = build_request(
            dataset_key,
            variable,
            box,
            month=month.month,
            years=list(range(reference[0], reference[1] + 1)),
            grid=grid,
        )

    store_meta = CDS_STORES[CDS_DATASETS[dataset_key]["store"]]
    job = {
        "kind": "field",
        "season": None,
        "year": month.year,
        "reference": reference if item["anomaly"] else None,
        "dataset": dataset_key,
        "variable": variable,
        "crop": None,
        "grid": grid,
        "accum_hours": None,
        "monthly_mode": item.get("monthly_mode"),
        "window_start": None,
        "request_meta": {
            "store_label": f"{store_meta['label']} ({store_meta['short']})",
            "region": item["area"],
            "region_label": MAP_AREA_PRESET_LABELS.get(item["area"]),
            "date": None,
            "hour": None,
            "init_time": None,
            "leadtime_hour": None,
            "cams_model": None,
            "month_label": f"{month:%B %Y} monthly mean",
            "valid_time_hint": None,
            "accumulation": CDS_VARIABLES[dataset_key][variable].get("accumulation"),
        },
        "anomaly": bool(item["anomaly"]),
    }
    return job, requests


def retrieve(requests: dict[str, dict], poll_s: float = 15.0, timeout_s: float = 3 * 3600) -> dict:
    """Queue every request at CDS and wait for all of them: `{role: path}`.

    For the offline job only — it blocks for as long as the queue takes.
    Downloads are cached on disk by request hash, so a rerun is free.
    """
    from utils.cds_api import JOB_DONE, JOB_FAILED, CDSDataError, job_status, submit

    ids = {role: submit(MAP_GALLERY_DATASET, request) for role, request in requests.items()}
    deadline = time.time() + timeout_s
    while True:
        states = {role: job_status(job_id) for role, job_id in ids.items()}
        failed = [s for s in states.values() if s["state"] in (JOB_FAILED, "unknown")]
        if failed:
            raise CDSDataError(failed[0].get("message") or "The retrieval failed.")
        if all(s["state"] == JOB_DONE for s in states.values()):
            return {role: s["path"] for role, s in states.items()}
        if time.time() > deadline:
            raise CDSDataError("Gave up waiting for the Copernicus queue.")
        time.sleep(poll_s)


def _png_bytes(data_uri: str) -> bytes:
    return base64.b64decode(data_uri.split(",", 1)[1])


def _atomic_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.{os.getpid()}.part")
    tmp.write_bytes(data)
    tmp.replace(path)


def load_gallery() -> dict:
    """The manifest, or an empty one."""
    hub_sync.sync()
    path = _root() / _MANIFEST
    if not path.exists():
        return {"items": {}}
    return json.loads(path.read_text())


def save_gallery(manifest: dict) -> None:
    manifest = {**manifest, "generated": f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M}Z"}
    _atomic_bytes(_root() / _MANIFEST, json.dumps(manifest, indent=1).encode())


def record(manifest: dict, key: str, month: pd.Timestamp, path: str, caption: str) -> None:
    """Add or replace one item's entry; `path` is relative to `PRECOMPUTE_DIR`."""
    manifest["items"][key] = {
        "label": MAP_GALLERY[key]["label"],
        "month": f"{pd.Timestamp(month):%Y-%m}",
        "path": path,
        "caption": caption,
    }


def is_current(manifest: dict, key: str, month: pd.Timestamp) -> bool:
    entry = manifest["items"].get(key)
    return bool(entry) and entry["month"] == f"{pd.Timestamp(month):%Y-%m}"


def gallery_entries() -> list[dict]:
    """Manifest entries in `MAP_GALLERY` order, each with its `key`."""
    items = load_gallery()["items"]
    return [{"key": key, **items[key]} for key in MAP_GALLERY if key in items]


def refresh(force: bool = False, keys: list[str] | None = None) -> int:
    """Retrieve and draw every gallery item not yet drawn for the newest month.

    Returns the number of images written. An item that fails is logged and
    skipped; its previous image (an older month) stays in the manifest.
    """
    from utils.cds_api import CDSDataError
    from utils.map_products import field_static

    month = gallery_month()
    manifest = load_gallery()
    written = 0
    for key in keys or list(MAP_GALLERY):
        if not force and is_current(manifest, key, month):
            continue
        job, requests = gallery_job(key, month)
        try:
            out = field_static({**job, "paths": retrieve(requests)}, theme="dark", **_VIEW)
        except (CDSDataError, ValueError) as exc:
            _log.warning("gallery %s: %s", key, exc)
            continue
        rel = f"{_SUBDIR}/{key}-{month:%Y%m}.png"
        _atomic_bytes(Path(PRECOMPUTE_DIR) / rel, _png_bytes(out["src"]))
        old = manifest["items"].get(key, {}).get("path")
        record(manifest, key, month, rel, out["caption"])
        save_gallery(manifest)
        if old and old != rel:
            (Path(PRECOMPUTE_DIR) / old).unlink(missing_ok=True)
        written += 1
    return written
