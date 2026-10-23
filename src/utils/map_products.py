"""Maps-page products: a finished retrieval in, figures and PNGs out.

Glue between `cds_api` (files on disk), `climatology` (numbers), and the two
renderers — `figures` (interactive Plotly) and `static_figures` (matplotlib
PNG) — so the page callbacks only hand over a job record. No API calls here:
every input is a file a completed retrieval wrote.

A job record is what the page stores once all its retrievals have landed:
`kind` is "field" (one CDS/ADS variable, optionally an anomaly) or "index"
(a climate variability index), plus the request details and `paths`.
"""

from __future__ import annotations

import base64
import threading

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr
from matplotlib.ticker import MaxNLocator

from utils import status
from utils.cache import cache
from utils.cds_api import load_field, load_monthly_anomaly, load_seasonal, load_series
from utils.climatology import compute_index, correlation_map, index_for_month
from utils.earthmover_api import (
    GridDataError,
    area_bounds,
    check_read_budget,
    frame_cube,
    frame_statistic,
)
from utils.figures import (
    make_correlation_map_fig,
    make_field_map_fig,
    make_index_timeseries_fig,
    map_subtitle,
    palette_levels,
    window_hours,
)
from utils.regions import REGIONS, polygon
from utils.settings import (
    CDS_DATASETS,
    CDS_VARIABLES,
    CLIMATE_INDICES,
    GRID_STORE_DEFAULT,
    GRID_STORES,
    GRID_VARIABLES,
    INDEX_GRID_DEG,
    INDEX_SOURCES,
    INDEX_START_YEAR,
    INDEX_TARGETS,
    MAP_AREA_PRESET_LABELS,
    MAP_AREA_PRESETS,
    MAP_PALETTES,
    STATIC_PALETTES,
    STATIC_PROJECTIONS,
)
from utils.static_figures import (
    render_field_map,
    render_index_timeseries,
    resolve_projection,
)

# One animation renders at a time per process: matplotlib holds the GIL for
# most of a frame, and two overlapping animations each ran ~4x slower than
# one after the other (measured: 137 s instead of ~25 s).
_RENDER_LOCK = threading.Lock()

# Candidate steps for automatic levels, per power of ten.
_NICE_STEPS = (1.0, 2.0, 2.5, 5.0)


def _png_uri(png: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(png).decode()}"


def _grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tidy lat/lon/value frame to a regular grid, latitude ascending."""
    table = df.pivot_table(index="latitude", columns="longitude", values="value")
    table = table.sort_index().sort_index(axis=1)
    return table.columns.to_numpy(), table.index.to_numpy(), table.to_numpy()


def _symmetric_levels(values: np.ndarray, n_side: int = 8) -> np.ndarray:
    """Zero-centred levels on a round step, reaching the 98th pct of |value|.

    Robust rather than the absolute peak, so one extreme cell does not wash
    the rest of the map out; `extend="both"` still colours what lies beyond.
    """
    finite = np.abs(values[np.isfinite(values)])
    reach = float(np.percentile(finite, 98)) if finite.size else 1.0
    reach = reach or 1.0
    raw = reach / n_side
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (*_NICE_STEPS, 10.0) if m * magnitude >= raw)
    n = int(np.ceil(reach / step))
    return np.arange(-n, n + 1) * step


def _auto_levels(values: np.ndarray) -> np.ndarray:
    """About 12 round levels across the 2nd–98th percentile of the field."""
    finite = values[np.isfinite(values)]
    lo, hi = np.percentile(finite, [2, 98]) if finite.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    return MaxNLocator(nbins=12).tick_values(lo, hi)


def _view_extent(extent_key: str, data_extent: tuple) -> tuple:
    return tuple(MAP_AREA_PRESETS.get(extent_key, data_extent))


def _static_footer(projection: str, extent: tuple, palette: str, levels: str = "default") -> str:
    kind = resolve_projection(projection, extent)
    fitted = " · levels fitted to this area (P2–P98)" if levels == "fitted" else ""
    return (
        f"Static map · {STATIC_PROJECTIONS.get(kind, kind)} · "
        f"palette: {STATIC_PALETTES.get(palette, palette)}{fitted}"
    )


# ---------------------------------------------------------------------------
# Fields
# ---------------------------------------------------------------------------


def _load_field_result(job: dict) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    """The field (or anomaly) frame and metadata for a completed field job.

    An anomaly of a variable marked `anomaly_overlay` also returns the
    month's absolute field, which is contoured over the shaded anomaly.
    """
    dataset_key, variable = job["dataset"], job["variable"]
    var_meta = CDS_VARIABLES[dataset_key][variable]
    paths = job["paths"]
    absolute = None
    # A smaller area served from a cached larger retrieval is cut out here,
    # before thinning, so it is drawn at the resolution the file holds.
    crop = tuple(job["crop"]) if job.get("crop") else None

    if job.get("season"):
        df, meta = load_seasonal(
            paths["field"],
            dataset_key,
            variable,
            job["season"],
            job["year"],
            reference_path=paths.get("reference") if job.get("anomaly") else None,
            reference=tuple(job["reference"]) if job.get("anomaly") else None,
            monthly_mode=job.get("monthly_mode"),
            crop=crop,
        )
        if job.get("anomaly") and var_meta.get("anomaly_overlay"):
            absolute, _ = load_seasonal(
                paths["field"], dataset_key, variable, job["season"], job["year"], crop=crop
            )
    elif job.get("anomaly"):
        df, meta = load_monthly_anomaly(
            paths["field"],
            paths["reference"],
            dataset_key,
            variable,
            monthly_mode=job.get("monthly_mode"),
            crop=crop,
        )
        if var_meta.get("anomaly_overlay"):
            absolute, _ = load_field(paths["field"], dataset_key, variable, crop=crop)
    else:
        df, meta = load_field(
            paths["field"],
            dataset_key,
            variable,
            accum_hours=job.get("accum_hours"),
            window_start=job.get("window_start"),
            monthly_mode=job.get("monthly_mode"),
            crop=crop,
        )

    if job.get("grid"):
        meta = {**meta, "regridded_from": CDS_DATASETS[dataset_key]["resolution_deg"]}
    if crop:
        meta = {**meta, "cut_from_cache": True}
    return df, meta, absolute


def field_figure(job: dict) -> go.Figure:
    """Interactive Plotly map of a completed field job."""
    df, meta, absolute = _load_field_result(job)
    return make_field_map_fig(
        df,
        CDS_VARIABLES[job["dataset"]][job["variable"]],
        CDS_DATASETS[job["dataset"]],
        meta,
        job["request_meta"],
        overlay=absolute,
        outline=polygon(job["request_meta"].get("region")),
    )


@cache.memoize(timeout=86_400)
def field_static(
    job: dict,
    projection: str,
    extent_key: str,
    palette: str,
    theme: str = "dark",
    levels: str = "auto",
) -> dict:
    """Static PNG of a completed field job.

    Memoized on the job (whose paths are request hashes) and the three style
    choices, so flipping back to a projection already drawn is instant.

    Returns
    -------
    dict
        `src` (PNG data URI), `filename`, `caption`.
    """
    dataset_key, variable = job["dataset"], job["variable"]
    var_meta = CDS_VARIABLES[dataset_key][variable]
    df, meta, absolute = _load_field_result(job)
    lon, lat, values = _grid(df)
    is_anomaly = bool(meta.get("is_anomaly"))
    unit = meta.get("unit") or var_meta.get("anomaly_unit" if is_anomaly else "unit")
    style = MAP_PALETTES[var_meta["palette"]]

    shading: dict | None
    isolines = None
    notes: list[str] = []
    # Auto: a field cut from a larger cached one is fitted, so its tails show.
    mode = resolve_levels(levels, bool(job.get("crop")))
    if is_anomaly or "isolines" in style:
        mode = "default"  # anomalies stay symmetric; isolines keep their interval
    if is_anomaly:
        cmap = (
            var_meta.get("anomaly_colorscale", "RdBu_r")
            if palette == "default"
            else palette
        )
        shading = {"levels": _symmetric_levels(values), "cmap": cmap, "extend": "both"}
        if absolute is not None:
            _, _, abs_values = _grid(absolute)
            isolines = {"values": abs_values, "interval": style.get("isolines", 5.0)}
    elif palette == "default" and "isolines" in style:
        shading = None
        isolines = {"interval": style["isolines"], "color": style["color"]}
    elif palette == "default" and mode == "fitted":
        colors = style["colors"].split("/")
        shading = {"levels": fitted_levels(values, len(colors)), "colors": colors, "extend": "both"}
    elif palette == "default":
        fixed = palette_levels(style, window_hours(meta))
        colors = style["colors"].split("/")
        shading = {"levels": fixed, "colors": colors}
        peak = float(np.nanmax(values))
        if len(colors) < len(fixed) and peak > fixed[-1]:
            notes.append(
                f"colour scale capped at {fixed[-1]:g}; field peaks at {peak:.3g}"
            )
    else:
        shading = {"levels": _auto_levels(values), "cmap": palette, "extend": "both"}
        if "isolines" in style:
            isolines = {"interval": style["isolines"], "color": "#111111"}

    subtitle = map_subtitle(CDS_DATASETS[dataset_key], meta, job["request_meta"])
    subtitle = subtitle.replace("<br>", "\n")
    if notes:
        subtitle += "\n" + " · ".join(notes)
    extent = _view_extent(extent_key, meta["extent"])
    # The retrieved region's polygon, else the viewed region's, if it has one.
    outline = polygon(job["request_meta"].get("region"))
    if outline is None:
        outline = polygon(extent_key)
    png = render_field_map(
        lon,
        lat,
        values,
        extent=extent,
        projection=projection,
        shading=shading,
        colorbar_label=(
            f"{'anomaly' if is_anomaly else var_meta['label']} ({unit})"
            if unit
            else var_meta.get("colorbar_title", var_meta["label"])
        ),
        title=f"{var_meta['label']}{' anomaly' if is_anomaly else ''}",
        subtitle=subtitle,
        footer=_static_footer(projection, extent, palette, mode),
        isolines=isolines,
        outline=outline,
        theme=theme,
    )
    finite = values[np.isfinite(values)]
    caption = (
        f"Field ranges {finite.min():.4g} to {finite.max():.4g} {unit}."
        if finite.size
        else "The field is empty over this area."
    )
    return {
        "src": _png_uri(png),
        "filename": f"{dataset_key}-{variable}{'-anomaly' if is_anomaly else ''}.png",
        "caption": caption,
    }


# ---------------------------------------------------------------------------
# Climate indices
# ---------------------------------------------------------------------------


@cache.memoize(timeout=86_400)
def load_index(
    source_paths: tuple[str, ...], index_key: str, reference: tuple[int, int]
) -> pd.DataFrame:
    """Monthly index frame from its source file(s), full years plus recent."""
    index = CLIMATE_INDICES[index_key]
    source = INDEX_SOURCES[index["source"]]
    parts = [load_series(path, source) for path in source_paths]
    field = xr.concat(parts, dim="time").sortby("time") if len(parts) > 1 else parts[0]
    return compute_index(field, index, reference)


def _index_correlation(job: dict) -> dict:
    index_key, month = job["index"], job["month"]
    reference = tuple(job["reference"])
    index = CLIMATE_INDICES[index_key]
    frame = load_index(_source_paths(job), index_key, reference)
    # A decadal index (AMO) is correlated through its monthly values; its
    # 121-month running mean has no year-to-year variation left to correlate.
    column = "monthly" if (index.get("smooth") or 0) > 12 else "value"
    series = index_for_month(frame, month, column)
    target = load_series(job["paths"]["target"], INDEX_TARGETS[job["target"]])
    r, p, n_years = correlation_map(target, series)
    return {
        "lon": r["longitude"].values,
        "lat": r["latitude"].values,
        "r": r.values,
        "p": p.values,
        "n": n_years,
    }


def _source_paths(job: dict) -> tuple[str, ...]:
    paths = job["paths"]
    return tuple(
        paths[role] for role in ("source_full", "source_recent") if role in paths
    )


def _index_texts(job: dict, frame: pd.DataFrame, n_years: int | None = None) -> dict:
    index = CLIMATE_INDICES[job["index"]]
    target = INDEX_TARGETS[job["target"]]["label"]
    month_name = pd.Timestamp(2001, job["month"], 1).strftime("%B")
    first, last = job["reference"]
    area_key = index["map_area"]
    source = INDEX_SOURCES[index["source"]]
    dataset = CDS_DATASETS[source["dataset"]]["label"]
    start, end = frame.index[0], frame["monthly"].dropna().index[-1]
    return {
        "ts_title": index["label"],
        "ts_subtitle": (
            f"{start:%b %Y}–{end:%b %Y} · anomalies vs {first}–{last} · "
            f"{index['definition']}"
        ),
        "ts_footer": (
            f"{dataset} · Copernicus CDS · {INDEX_GRID_DEG:g}° grid (regridded by "
            f"CDS) · ERA5-derived proxy, record from {INDEX_START_YEAR}"
        ),
        "map_title": f"{month_name} {target.lower()} vs {index['label']}",
        "map_subtitle": (
            f"Pearson r over {first}–{last}"
            + (f" ({n_years} {month_name}s)" if n_years else "")
            + " · stippled p < 0.05, two-sided t-test, no autocorrelation "
            f"correction · area fixed to {MAP_AREA_PRESET_LABELS.get(area_key, area_key)}"
        ),
    }


def index_figures(job: dict) -> tuple[go.Figure, go.Figure]:
    """Interactive time series and correlation map for a completed index job."""
    frame = load_index(_source_paths(job), job["index"], tuple(job["reference"]))
    corr = _index_correlation(job)
    texts = _index_texts(job, frame, corr["n"])
    ts = make_index_timeseries_fig(
        frame, CLIMATE_INDICES[job["index"]], texts["ts_title"], texts["ts_subtitle"]
    )
    area = MAP_AREA_PRESETS[CLIMATE_INDICES[job["index"]]["map_area"]]
    corr_fig = make_correlation_map_fig(
        corr["lon"],
        corr["lat"],
        corr["r"],
        corr["p"],
        area,
        texts["map_title"],
        texts["map_subtitle"],
    )
    return ts, corr_fig


@cache.memoize(timeout=86_400)
def index_static(
    job: dict, projection: str, extent_key: str, palette: str, theme: str = "dark"
) -> dict:
    """Static correlation map and time series PNGs for a completed index job.

    Returns
    -------
    dict
        `map_src`, `ts_src`, `map_filename`, `ts_filename`, `caption`.
    """
    index = CLIMATE_INDICES[job["index"]]
    frame = load_index(_source_paths(job), job["index"], tuple(job["reference"]))
    corr = _index_correlation(job)
    texts = _index_texts(job, frame, corr["n"])
    area = MAP_AREA_PRESETS[index["map_area"]]
    extent = _view_extent(extent_key, area)

    map_png = render_field_map(
        corr["lon"],
        corr["lat"],
        corr["r"],
        extent=extent,
        projection=projection,
        shading={
            "levels": np.arange(-0.9, 0.91, 0.1),
            "cmap": "RdBu_r" if palette == "default" else palette,
            "extend": "both",
        },
        colorbar_label="Pearson r",
        title=texts["map_title"],
        subtitle=texts["map_subtitle"],
        footer=f"{texts['ts_footer']} · {_static_footer(projection, extent, palette)}",
        stipple=np.isfinite(corr["p"]) & (corr["p"] < 0.05),
        theme=theme,
    )
    ts_png = render_index_timeseries(
        frame,
        index,
        title=texts["ts_title"],
        subtitle=texts["ts_subtitle"],
        footer=texts["ts_footer"],
        theme=theme,
    )

    window = index.get("smooth")
    latest = (frame["value"] if window and window <= 12 else frame["monthly"]).dropna()
    significant = float(np.mean(np.isfinite(corr["p"]) & (corr["p"] < 0.05)))
    caption = (
        f"Latest {index['label']}: {latest.iloc[-1]:+.2f} {index['unit']} "
        f"({latest.index[-1]:%B %Y}). Correlation r ranges "
        f"{np.nanmin(corr['r']):+.2f} to {np.nanmax(corr['r']):+.2f}; "
        f"{significant:.0%} of cells significant at p < 0.05."
    )
    return {
        "map_src": _png_uri(map_png),
        "ts_src": _png_uri(ts_png),
        "map_filename": f"{job['index']}-correlation-{job['target']}-{job['month']:02d}.png",
        "ts_filename": f"{job['index']}-timeseries.png",
        "caption": caption,
    }


# ---------------------------------------------------------------------------
# Forecast animations (Weather · Area average)
# ---------------------------------------------------------------------------


def _range_levels(values: np.ndarray) -> np.ndarray:
    """Round levels from 0 to the 98th percentile of a spread field."""
    finite = values[np.isfinite(values)]
    top = float(np.percentile(finite, 98)) if finite.size else 1.0
    return MaxNLocator(nbins=10).tick_values(0.0, top or 1.0)


def fitted_levels(values: np.ndarray, n_bands: int) -> np.ndarray:
    """`n_bands + 1` round, evenly spaced edges covering the area's P2–P98.

    For the "Fitted to area" levels: the Default palette's colours spread over
    the values this area actually holds, so its tails are not one colour.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.arange(n_bands + 1, dtype=float)
    lo, hi = np.percentile(finite, [2, 98])
    if hi <= lo:
        hi = lo + 1.0
    raw = (hi - lo) / n_bands
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (1.0, 2.0, 2.5, 5.0, 10.0) if m * magnitude >= raw)
    start = np.floor(lo / step) * step
    # Widen by whole steps until the top edge clears P98 (rounding the start down can fall short).
    n = n_bands
    while start + n * step < hi:
        n += 1
    return start + step * np.arange(n + 1)


def resolve_levels(choice: str | None, cut_from_larger: bool) -> str:
    """"auto" fits a field cut from a larger one (its tails would sit in one band)."""
    if choice in ("default", "fitted"):
        return choice
    return "fitted" if cut_from_larger else "default"


def _shared_levels(
    values_by_store: dict[str, np.ndarray],
    statistic: str,
    palette: str,
    style: dict,
    var_meta: dict,
    accum_hours: int | None,
    unit: str,
    levels: str = "default",
) -> tuple[dict | None, dict | None, str]:
    """One colour scale for every model's frames: `(shading, isolines, bar_label)`.

    Fitted levels (spread, colormaps) come from all models' values together,
    so a colour means the same value in every panel and every frame. Default
    styles keep their fixed levels.
    """
    values = np.concatenate([v.ravel() for v in values_by_store.values()])
    isolines = None
    if statistic == "range":
        shading = {
            "levels": _range_levels(values),
            "cmap": "YlOrRd" if palette == "default" else palette,
            "extend": "max",
        }
        return shading, None, f"ensemble spread ({unit})"
    if palette == "default" and "isolines" in style:
        isolines = {
            "interval": var_meta.get("isoline_interval", style["isolines"]),
            "color": style["color"],
        }
        return None, isolines, unit
    if palette == "default" and levels == "fitted":
        colors = style["colors"].split("/")
        return {"levels": fitted_levels(values, len(colors)), "colors": colors, "extend": "both"}, None, unit
    if palette == "default":
        fixed = palette_levels(style, float(accum_hours or 1))
        return {"levels": fixed, "colors": style["colors"].split("/")}, None, unit
    shading = {"levels": _auto_levels(values), "cmap": palette, "extend": "both"}
    if "isolines" in style:
        isolines = {
            "interval": var_meta.get("isoline_interval", style["isolines"]),
            "color": "#111111",
        }
    return shading, isolines, unit


@cache.memoize(timeout=1800)
def forecast_animation(
    area_key: str,
    variable: str,
    init_time: str | None,
    step_hours: int,
    horizon_hours: int,
    accum_hours: int | None,
    statistic: str,
    pct: float,
    pct_range: tuple[float, float],
    projection: str,
    extent_key: str,
    palette: str,
    stores: tuple[str, ...] = (GRID_STORE_DEFAULT,),
    theme: str = "dark",
    levels: str = "auto",
    budget_mb: float | None = None,
) -> dict:
    """Static map frames of an ensemble statistic, one panel per model.

    Every panel and frame uses the same levels — default bands, or levels
    fitted to all models' frames together for a colormap — so a colour means
    the same value everywhere. Frames are the leads every model has; each
    model's statistic is taken over its own members only. Rendered
    sequentially: matplotlib contends on the GIL, and four threads were 4x
    slower than one.

    Returns
    -------
    dict
        `panels`: `[{store, label}]`; `frames`: `[{lead_hours, valid, srcs}]`
        with one PNG data URI per panel; `caption`; `filename`.

    `budget_mb` replaces the interactive read budget (the preset job's own).
    """
    from utils.precomputed import fields_covering  # precomputed imports this module

    # A precomputed preset that covers the area, run, step and statistic is cut
    # from its saved fields: no archive read, and no read budget to spend.
    bounds = area_bounds(area_key)
    cubes = {}
    for s in stores:
        own = "mean" if GRID_STORES[s].get("kind") == "deterministic" else statistic
        cut = fields_covering(bounds, variable, init_time, step_hours, horizon_hours, accum_hours, own, s)
        if cut is not None:
            cubes[s] = cut
    to_read = tuple(s for s in stores if s not in cubes)
    # Frames stop at the horizon, so only its lead chunks are read (GEFS).
    if to_read:
        check_read_budget(to_read, bounds, horizon_hours=horizon_hours, budget_mb=budget_mb)
    # A model that cannot serve this (no such variable, no such run) loses its
    # panel, not the animation: the others still render, and say why.
    errors = []
    for n, s in enumerate(to_read, start=1):
        status.report("Reading model fields", n, len(to_read), GRID_STORES[s]["label"])
        try:
            cubes[s] = frame_cube(
                area_key, variable, init_time, step_hours, accum_hours, s, horizon_hours
            )
        except GridDataError as exc:
            errors.append(str(exc))
            continue
        if cubes[s].get("clip_note"):
            errors.append(cubes[s]["clip_note"])
    if not cubes:
        raise GridDataError(" ".join(errors) or "No model can serve this animation.")
    stores = tuple(s for s in stores if s in cubes)
    common = sorted(set.intersection(*(set(c["lead_hours"]) for c in cubes.values())))
    common = [h for h in common if h <= horizon_hours]
    if not common:
        raise GridDataError(f"No frame within the first {horizon_hours} h common to every model.")

    if statistic == "range":
        for s in [s for s in cubes if GRID_STORES[s].get("kind") == "deterministic"]:
            errors.append(f"{GRID_STORES[s]['label']} is deterministic: no spread to map.")
            del cubes[s]
        if not cubes:
            raise GridDataError("A deterministic model has no ensemble spread to map.")
        stores = tuple(cubes)
    values_by_store, keep_by_store = {}, {}
    for s, cube in cubes.items():
        keep = [cube["lead_hours"].index(h) for h in common]
        # A deterministic run has one member: its mean, any percentile and
        # its field are the same, and its spread is not zero but undefined.
        own = "mean" if GRID_STORES[s].get("kind") == "deterministic" else statistic
        if cube.get("precomputed_statistic"):
            values, store_description = cube["cube"][:, 0], cube["description"]
        else:
            values, store_description = frame_statistic(cube, own, pct, pct_range)
        if own == statistic:
            description = store_description
        values_by_store[s], keep_by_store[s] = values[keep], keep
    if not any(GRID_STORES[s].get("kind") != "deterministic" for s in cubes):
        description = "deterministic field"

    var_meta = GRID_VARIABLES[variable]
    style = MAP_PALETTES[var_meta["palette"]]
    first = next(iter(cubes.values()))
    # Auto: a sub-area cut from a preset's saved fields is fitted.
    mode = resolve_levels(levels, any(c.get("cut_from_larger") for c in cubes.values()))
    shading, isolines, bar_label = _shared_levels(
        values_by_store, statistic, palette, style, var_meta, accum_hours, first["unit"], levels=mode
    )

    region = first["region"]
    area = REGIONS[region]["label"] if region else "custom box"
    extent = _view_extent(extent_key, tuple(first["bounds"]))
    label = (
        f"{var_meta.get('accum_label', var_meta['label'])} ({accum_hours} h total)"
        if accum_hours
        else var_meta["label"]
    )
    outline = polygon(region)

    rendered = {}
    with _RENDER_LOCK:
        for s, cube in cubes.items():
            thinned = f" · drawn every {cube['stride']} cells" if cube["stride"] > 1 else ""
            rendered[s] = _render_frames(
                cube,
                values_by_store[s],
                keep_by_store[s],
                extent,
                projection,
                shading,
                bar_label,
                label,
                description,
                area,
                pd.Timestamp(cube["init_time"]),
                thinned,
                palette,
                isolines,
                outline,
                theme,
                "fitted" if shading and shading.get("extend") == "both" and "colors" in shading else "default",
            )

    lead_frames = rendered[stores[0]]
    frames = [
        {
            "lead_hours": f["lead_hours"],
            "valid": f["valid"],
            "srcs": [rendered[s][n]["src"] for s in stores],
        }
        for n, f in enumerate(lead_frames)
    ]
    inits = " · ".join(
        f"{GRID_STORES[s]['label']} {pd.Timestamp(c['init_time']):%Y-%m-%d %H} UTC"
        for s, c in cubes.items()
    )
    init = pd.Timestamp(first["init_time"])
    return {
        "panels": [
            {
                "store": s,
                "label": GRID_STORES[s]["label"]
                + (" (deterministic, field)" if GRID_STORES[s].get("kind") == "deterministic" else ""),
            }
            for s in stores
        ],
        "frames": frames,
        "errors": errors,
        "caption": (
            f"{len(frames)} frames, T+{frames[0]['lead_hours']} to "
            f"T+{frames[-1]['lead_hours']} h · {description} · runs {inits}."
        ),
        # The player prefixes each panel's store.
        "filename": f"{variable}-{statistic}-{init:%Y%m%d%H}",
    }


def _render_frames(
    cube,
    values,
    keep,
    extent,
    projection,
    shading,
    bar_label,
    label,
    description,
    area,
    init,
    thinned,
    palette,
    isolines,
    outline,
    theme="dark",
    levels="default",
) -> list[dict]:
    frames = []
    for n, i in enumerate(keep):
        status.report("Rendering frames", n + 1, len(keep), GRID_STORES[cube["store"]]["label"])
        valid = pd.Timestamp(cube["valid_time"][i])
        lead = cube["lead_hours"][i]
        png = render_field_map(
            cube["lon"],
            cube["lat"],
            values[n],
            extent=extent,
            projection=projection,
            shading=shading,
            colorbar_label=bar_label,
            title=f"{label} — {description}",
            subtitle=(
                f"T+{lead} h · valid {valid:%a %d %b %Y %H:%M} UTC\n"
                f"{area} · {GRID_STORES[cube['store']]['model_label']} · init {init:%Y-%m-%d %H:%M} UTC · "
                f"{GRID_STORES[cube['store']]['res_deg']:g}° grid{thinned}"
            ),
            footer=_static_footer(projection, extent, palette, levels),
            isolines=isolines,
            outline=outline,
            theme=theme,
        )
        frames.append(
            {
                "src": _png_uri(png),
                "lead_hours": lead,
                "valid": f"{valid:%a %d %b %H:%M} UTC",
            }
        )
    return frames
