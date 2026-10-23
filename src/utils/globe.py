"""Global-field export for the static globe page.

Data texture encoding: one 16-bit code per cell in an RGBA PNG. R is the high
byte, G the low byte, B is 0, and alpha is always 255 (premultiplied alpha
would corrupt R and G). ``q = round((v - offset) / scale)`` is clipped to
0..65534; 65535 is the missing code (NaN). Decoding is ``offset + q * scale``.
"""

import io
import json
import shutil
from pathlib import Path

import contourpy
import numpy as np
import pandas as pd
from PIL import Image

from utils.earthmover_api import (
    GRID_VARIABLES,
    GridDataError,
    _complete_run,
    _open_dataset,
    _require_variable,
    _select_box,
    check_read_budget,
    window_totals,
)
from utils.precomputed import _atomic_bytes, _stamp
from utils.settings import (
    GLOBE_HORIZON_HOURS,
    GLOBE_LAYERS,
    GLOBE_READ_BUDGET_MB,
    GLOBE_STEP_HOURS,
    GLOBE_STORE,
    GLOBE_STRIDE,
    GRID_STORES,
    MAP_PALETTES,
    PRECOMPUTE_DIR,
)

MISSING = 65535


def encode16(values: np.ndarray, offset: float, scale: float) -> np.ndarray:
    """Pack an ``(ny, nx)`` float field into ``(ny, nx, 4)`` uint8 RGBA."""
    v = np.asarray(values, dtype=np.float64)
    nan = np.isnan(v)
    q = np.rint((np.where(nan, 0.0, v) - offset) / scale)
    q = np.clip(q, 0, MISSING - 1).astype(np.uint32)
    q[nan] = MISSING
    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = q >> 8
    rgba[..., 1] = q & 255
    rgba[..., 3] = 255
    return rgba


def decode16(rgba: np.ndarray, offset: float, scale: float) -> np.ndarray:
    """Inverse of :func:`encode16`; the missing code becomes NaN (float32)."""
    q = rgba[..., 0].astype(np.uint32) * 256 + rgba[..., 1].astype(np.uint32)
    out = (offset + q * scale).astype(np.float32)
    out[q == MISSING] = np.nan
    return out


def png_bytes(rgba: np.ndarray) -> bytes:
    """Lossless RGBA PNG, with no colour profile and no gamma chunk."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8), "RGBA").save(
        buf, format="PNG", optimize=True
    )
    return buf.getvalue()


def isobars(
    values: np.ndarray, lat: np.ndarray, lon: np.ndarray, interval: float
) -> dict:
    """Isolines every multiple of ``interval`` as a GeoJSON FeatureCollection.

    A cyclic column (the first column again at ``lon[0] + 360``) is appended
    before contouring, so lines that cross the seam run on to +-180 at the
    same latitude on both sides instead of stopping a cell short.
    """
    v = np.asarray(values, dtype=np.float64)
    lon_x = np.append(lon, lon[0] + 360.0)
    z = np.concatenate([v, v[:, :1]], axis=1)
    z = np.ma.masked_invalid(z)
    if z.count() == 0:
        return {"type": "FeatureCollection", "features": []}
    gen = contourpy.contour_generator(lon_x, np.asarray(lat), z)
    first = int(np.ceil(float(z.min()) / interval))
    last = int(np.floor(float(z.max()) / interval))
    features = []
    for k in range(first, last + 1):
        level = k * interval
        for line in gen.lines(level):
            if len(line) < 2:
                continue
            coords = [[round(float(x), 2), round(float(y), 2)] for x, y in line]
            features.append(
                {
                    "type": "Feature",
                    "properties": {"value": float(level)},
                    "geometry": {"type": "LineString", "coordinates": coords},
                }
            )
    return {"type": "FeatureCollection", "features": features}


GLOBAL_BOUNDS = (-90.0, 90.0, -180.0, 179.75)
_MANIFEST = "manifest.json"


def _open_global(store: str):
    """The store's dataset (a seam for tests)."""
    return _open_dataset(store)


def _iso(t: pd.Timestamp) -> str:
    return f"{pd.Timestamp(t):%Y-%m-%dT%H:%M:%S}"


def read_global(layer: str, init: pd.Timestamp | None = None) -> dict:
    """One layer of member 0, strided to the drawn grid, for every export lead.

    The stride is applied lazily, so only every ``GLOBE_STRIDE``-th cell is
    decoded into memory. The budget is checked on the store's metadata first.

    Parameters
    ----------
    layer
        Key of ``GLOBE_LAYERS``.
    init
        Run to read; ``None`` takes the newest fully written one.

    Returns
    -------
    dict
        ``values`` float32 ``(frame, lat, lon)`` in display units (lat 90 to
        -90), ``lat``, ``lon``, ``lead_hours``, ``valid_time``, ``init_time``,
        ``unit`` and ``window_hours`` (``None`` unless a window total).
    """
    spec = GLOBE_LAYERS[layer]
    variable = spec["variable"]
    meta = GRID_VARIABLES[variable]
    check_read_budget(
        (GLOBE_STORE,),
        GLOBAL_BOUNDS,
        horizon_hours=GLOBE_HORIZON_HOURS,
        budget_mb=GLOBE_READ_BUDGET_MB,
    )
    ds = _open_global(GLOBE_STORE)
    _require_variable(ds, variable, GLOBE_STORE)
    if init is None:
        init, da = _complete_run(ds, variable, GLOBAL_BOUNDS, None, GLOBE_STORE)
    else:
        da = _select_box(ds, variable, pd.Timestamp(init), GLOBAL_BOUNDS)
    da = da.isel(ensemble_member=0)
    all_h = np.asarray(pd.to_timedelta(da["lead_time"].values).total_seconds()) / 3600
    window = spec.get("accum_hours")
    if window:
        # Window totals need every step up to the horizon, not just the frames.
        pick = all_h <= GLOBE_HORIZON_HOURS
    else:
        pick = (all_h <= GLOBE_HORIZON_HOURS) & (all_h % GLOBE_STEP_HOURS == 0)
    da = da.isel(
        lead_time=np.flatnonzero(pick),
        latitude=slice(None, None, GLOBE_STRIDE),
        longitude=slice(None, None, GLOBE_STRIDE),
    )
    lead_h = all_h[pick]
    try:
        raw = da.transpose("lead_time", "latitude", "longitude").values
    except (OSError, ValueError, RuntimeError) as exc:
        raise GridDataError(f"Failed reading the global field ({exc}).") from exc
    values = raw.astype(np.float32) * meta["scale"] + meta["offset"]
    del raw
    unit = meta["unit"]
    if window:
        if not meta.get("is_accumulated"):
            raise GridDataError(f"'{variable}' cannot be totalled over a window.")
        values, keep = window_totals(values, lead_h, window, lead_axis=0)
        lead_h = lead_h[keep]
        unit = meta.get("accum_unit", unit)
        frames = np.flatnonzero(lead_h % GLOBE_STEP_HOURS == 0)
        values, lead_h = values[frames], lead_h[frames]
    init = pd.Timestamp(init)
    return {
        "values": np.ascontiguousarray(values, dtype=np.float32),
        "lat": np.asarray(da["latitude"].values, dtype=float),
        "lon": np.asarray(da["longitude"].values, dtype=float),
        "lead_hours": [int(h) for h in lead_h],
        "valid_time": [init + pd.Timedelta(hours=int(h)) for h in lead_h],
        "init_time": init,
        "unit": unit,
        "window_hours": window,
    }


def _root() -> Path:
    return Path(PRECOMPUTE_DIR)


def load_manifest() -> dict:
    """The globe manifest, or an empty one."""
    path = _root() / "globe" / _MANIFEST
    return json.loads(path.read_text()) if path.exists() else {}


def save_manifest(m: dict) -> None:
    """Replace the manifest atomically."""
    _atomic_bytes(_root() / "globe" / _MANIFEST, json.dumps(m, indent=1).encode())


def _write_layer(layer: str, out: dict, stamp_dir: Path) -> list[dict]:
    """Write one layer's frames; returns their manifest entries."""
    spec = GLOBE_LAYERS[layer]
    interval = MAP_PALETTES[GRID_VARIABLES[spec["variable"]]["palette"]].get("isolines")
    shaded = spec["kind"] == "shaded"
    frames = []
    for i, lead in enumerate(out["lead_hours"]):
        name = f"T{lead:03d}.{'png' if shaded else 'geojson'}"
        if shaded:
            rgba = encode16(out["values"][i], spec["offset"], spec["scale"])
            data = png_bytes(rgba)
        else:
            gj = isobars(out["values"][i], out["lat"], out["lon"], interval)
            data = json.dumps(gj, separators=(",", ":")).encode()
        _atomic_bytes(stamp_dir / layer / name, data)
        rel = Path("globe", GLOBE_STORE, stamp_dir.name, layer, name)
        frames.append(
            {
                "lead_hours": lead,
                "valid_time": _iso(out["valid_time"][i]),
                "path": rel.as_posix(),
            }
        )
    return frames


def _layer_entry(layer: str, out: dict, frames: list[dict]) -> dict:
    spec = GLOBE_LAYERS[layer]
    meta = GRID_VARIABLES[spec["variable"]]
    label = (
        meta.get("accum_label", meta["label"]) if out["window_hours"] else meta["label"]
    )
    if out["window_hours"]:
        label = f"{label} ({out['window_hours']} h total)"
    entry = {"kind": spec["kind"], "label": label, "unit": out["unit"]}
    if spec["kind"] == "shaded":
        entry |= {
            "palette": meta["palette"],
            "offset": spec["offset"],
            "scale": spec["scale"],
            "window_hours": out["window_hours"],
        }
    else:
        entry["interval"] = MAP_PALETTES[meta["palette"]]["isolines"]
    return entry | {"frames": frames}


def refresh(
    manifest: dict, force: bool = False, failures: list[str] | None = None
) -> int:
    """Export the newest finished run unless the manifest already holds it.

    The run is resolved once and every layer reads that init. The manifest is
    replaced only after every layer is written; a failed layer is appended to
    ``failures`` and the previous manifest stays.

    Returns
    -------
    int
        Files written (0 when current or failed).
    """
    failures = failures if failures is not None else []
    first = GLOBE_LAYERS[next(iter(GLOBE_LAYERS))]["variable"]
    try:
        ds = _open_global(GLOBE_STORE)
        init, _ = _complete_run(ds, first, GLOBAL_BOUNDS, None, GLOBE_STORE)
    except GridDataError as exc:
        failures.append(f"globe: {exc}")
        return 0
    if not force and manifest.get("init_time") == _iso(init):
        return 0
    stamp_dir = _root() / "globe" / GLOBE_STORE / _stamp(init)
    layers, grid, written = {}, None, 0
    for layer in GLOBE_LAYERS:
        try:
            out = read_global(layer, init)
            frames = _write_layer(layer, out, stamp_dir)
        except (GridDataError, OSError, ValueError) as exc:
            failures.append(f"globe {layer}: {exc}")
            if manifest.get("init_time") != _iso(init):
                shutil.rmtree(stamp_dir, ignore_errors=True)
            return 0
        layers[layer] = _layer_entry(layer, out, frames)
        grid = grid or {
            "res_deg": GRID_STORES[GLOBE_STORE]["res_deg"] * GLOBE_STRIDE,
            "native_res_deg": GRID_STORES[GLOBE_STORE]["res_deg"],
            "lon0": float(out["lon"][0]),
            "lat0": float(out["lat"][0]),
            "nx": int(out["lon"].size),
            "ny": int(out["lat"].size),
        }
        written += len(frames)
        del out
    save_manifest(
        {
            "store": GLOBE_STORE,
            "label": GRID_STORES[GLOBE_STORE]["label"],
            "init_time": _iso(init),
            "grid": grid,
            "layers": layers,
            "written": _iso(pd.Timestamp.now(tz="UTC")),
        }
    )
    return written
