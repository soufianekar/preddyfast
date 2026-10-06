"""Static (matplotlib/cartopy) maps and index charts for the maps page.

PNG bytes out, no API calls. Figures are built with the object API —
`Figure` + `FigureCanvasAgg`, never `pyplot` — because pyplot keeps global
state and Dash serves callbacks from several threads at once.

Coastlines and borders come from the bundled Natural Earth lines, never from
`cartopy.feature`: that downloads shapefiles on first use, and a server
without outbound access would fail the render on the first map.
"""

from __future__ import annotations

import io
import re
import textwrap
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.path as mpath
import numpy as np
import pandas as pd
from cartopy.util import add_cyclic_point
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from utils.settings import (
    NATURAL_EARTH_FILE,
    THEMES,
)

_PC = ccrs.PlateCarree()
_NATURAL_EARTH_PATH = (
    Path(__file__).resolve().parent.parent / "data" / NATURAL_EARTH_FILE
)
_DPI = 110

Extent = tuple[float, float, float, float]  # lat_min, lat_max, lon_min, lon_max


# ---------------------------------------------------------------------------
# Projection and framing
# ---------------------------------------------------------------------------


def _is_global(extent: Extent) -> bool:
    lat_min, lat_max, lon_min, lon_max = extent
    return (lon_max - lon_min) >= 300 and (lat_max - lat_min) >= 120


def resolve_projection(choice: str, extent: Extent) -> str:
    """Concrete projection key for a choice, resolving "auto" by extent.

    A hemisphere cap gets polar stereographic, the globe Robinson, a box
    that straddles the equator Plate Carrée (Lambert conformal degenerates
    when its standard parallels have opposite signs), anything else Lambert.
    """
    if choice != "auto":
        return choice
    lat_min, lat_max, lon_min, lon_max = extent
    wraps = (lon_max - lon_min) >= 300
    if wraps and lat_min >= 0 and lat_max > 60:
        return "north_polar"
    if wraps and lat_max <= 0 and lat_min < -60:
        return "south_polar"
    if wraps:
        return "robinson"
    if lat_min < 0 < lat_max:
        return "platecarree"
    return "lambert"


def _crs(kind: str, extent: Extent):
    lat_min, lat_max, lon_min, lon_max = extent
    lon_c = 0.0 if (lon_max - lon_min) >= 300 else (lon_min + lon_max) / 2
    lat_c = (lat_min + lat_max) / 2
    if kind == "north_polar":
        return ccrs.NorthPolarStereo(central_longitude=-10)
    if kind == "south_polar":
        return ccrs.SouthPolarStereo(central_longitude=0)
    if kind == "robinson":
        return ccrs.Robinson(central_longitude=lon_c)
    if kind == "mollweide":
        return ccrs.Mollweide(central_longitude=lon_c)
    if kind == "orthographic":
        return ccrs.Orthographic(central_longitude=lon_c, central_latitude=lat_c)
    if kind == "mercator":
        return ccrs.Mercator(central_longitude=lon_c)
    if kind == "lambert":
        # Standard parallels a third of the way in from each edge. A box that
        # crosses the equator keeps to the hemisphere holding its centre —
        # parallels of opposite sign make the cone degenerate.
        if lat_min < 0 < lat_max:
            lat_min, lat_max = (0.5, lat_max) if lat_c >= 0 else (lat_min, -0.5)
        span = lat_max - lat_min
        return ccrs.LambertConformal(
            central_longitude=lon_c,
            central_latitude=lat_c,
            standard_parallels=(lat_min + span / 3, lat_max - span / 3),
        )
    return ccrs.PlateCarree(central_longitude=lon_c)


def _frame(ax, kind: str, extent: Extent) -> None:
    """Set the view once every artist is drawn.

    Polar caps are framed by the projected radius of their outer latitude,
    not `set_extent([-180, 180, ...])`: with a central longitude other than
    0 the ±180 edges coincide and the projected box collapses to a sliver.
    """
    lat_min, lat_max, lon_min, lon_max = extent
    if kind in ("north_polar", "south_polar"):
        edge = lat_min if kind == "north_polar" else lat_max
        if kind == "north_polar" and edge <= 0:
            edge = 20.0
        if kind == "south_polar" and edge >= 0:
            edge = -20.0
        proj = ax.projection
        x, y = proj.transform_point(proj.proj4_params.get("lon_0", 0.0), edge, _PC)
        radius = float(np.hypot(x, y))
        ax.set_xlim(-radius, radius)
        ax.set_ylim(-radius, radius)
        theta = np.linspace(0, 2 * np.pi, 200)
        circle = mpath.Path(np.vstack([np.sin(theta), np.cos(theta)]).T * 0.5 + 0.5)
        ax.set_boundary(circle, transform=ax.transAxes)
    elif kind == "orthographic" or (
        kind in ("robinson", "mollweide") and _is_global(extent)
    ):
        ax.set_global()
    else:
        if kind == "mercator":
            lat_min, lat_max = max(lat_min, -80.0), min(lat_max, 80.0)
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=_PC)


def _lines(scale: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Bundled coastline and border polylines (NaN-separated lon/lat)."""
    if not _NATURAL_EARTH_PATH.is_file():
        return {}
    with np.load(_NATURAL_EARTH_PATH) as archive:
        return {
            kind: (archive[f"{kind}_{scale}_lon"], archive[f"{kind}_{scale}_lat"])
            for kind in ("coastline", "borders")
            if f"{kind}_{scale}_lon" in archive.files
        }


def _mpl_color(value: str):
    """A theme colour in a form matplotlib accepts: CSS `rgba(r, g, b, a)` -> tuple."""
    m = re.fullmatch(r"rgba\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*([\d.]+)\s*\)", value)
    if m:
        r, g, b, a = m.groups()
        return (int(r) / 255, int(g) / 255, int(b) / 255, float(a))
    return value


def isoline_color(isolines: dict, shading: dict | None, tokens: dict) -> str:
    """Isolines over shading keep their own colour; on the bare panel they take the ink.

    The Default isobars are black, as drawn on a white chart; on the dark
    panel black lines would all but vanish.
    """
    if shading is None:
        return tokens["ink"]
    return isolines.get("color", "k")


def _tokens(theme: str) -> dict:
    """Theme colours for a static map; field colours never depend on the theme."""
    return {k: _mpl_color(v) for k, v in THEMES.get(theme, THEMES["dark"]).items()}


def _basemap(ax, kind: str, extent: Extent, theme: str = "dark") -> None:
    t = _tokens(theme)
    lat_min, lat_max, lon_min, lon_max = extent
    span = max(lat_max - lat_min, lon_max - lon_min)
    for name, (lon, lat) in _lines("110" if span > 60 else "50").items():
        # Clip polar line work to its cap: vertices in the far hemisphere
        # project to enormous coordinates and would drag the axes with them.
        if kind == "north_polar":
            lat = np.where(lat >= min(lat_min, 20.0) - 5, lat, np.nan)
        elif kind == "south_polar":
            lat = np.where(lat <= max(lat_max, -20.0) + 5, lat, np.nan)
        coast = name == "coastline"
        ax.plot(
            lon,
            lat,
            transform=_PC,
            color=t["coast"] if coast else t["border"],
            linewidth=0.6 if coast else 0.3,
            zorder=3,
        )
    labels = kind in ("platecarree", "mercator", "lambert")
    gl = ax.gridlines(
        draw_labels=labels, linewidth=0.3, color=t["rule"], x_inline=False, y_inline=False
    )
    if labels:
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 8, "color": t["dim"]}


def _cyclic(
    lon: np.ndarray, lat: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Close the longitude seam of a global field so contours do not gap."""
    if lon.size > 1 and lon.max() - lon.min() > 350:
        values, lon = add_cyclic_point(values, coord=lon)
    return lon, lat, values


# ---------------------------------------------------------------------------
# Maps
# ---------------------------------------------------------------------------


def render_field_map(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    *,
    extent: Extent,
    projection: str,
    shading: dict | None,
    colorbar_label: str,
    title: str,
    subtitle: str,
    footer: str,
    isolines: dict | None = None,
    stipple: np.ndarray | None = None,
    outline: np.ndarray | None = None,
    theme: str = "dark",
) -> bytes:
    """A gridded field as a filled-contour map, on the theme's panel colour.

    Parameters
    ----------
    lon, lat, values
        Regular grid, latitude ascending, `values` shaped `(lat, lon)`.
    extent
        View `(lat_min, lat_max, lon_min, lon_max)`.
    projection
        Key from `settings.STATIC_PROJECTIONS`, or "auto".
    shading
        `{"levels", "colors"}` for the default stepped bands — as many colours as
        levels means an open top band — or `{"levels", "cmap", "extend"}` for
        a matplotlib colormap. None draws no shading (isolines only).
    isolines
        `{"interval", "color"}` and optionally `values` — labelled contours
        of `values`, or of the shaded field when absent.
    stipple
        Boolean `(lat, lon)` mask hatched with dots (significance).
    outline
        `(n, 2)` lon/lat vertices of a region polygon, drawn closed.
    theme
        "dark" or "light": ground, text and line work. Field colours are the
        same in both; unshaded areas show the ground.
    """
    t = _tokens(theme)
    kind = resolve_projection(projection, extent)
    round_map = kind in ("north_polar", "south_polar", "orthographic")
    fig = Figure(
        figsize=(8.5, 8.8) if round_map else (11, 7.0),
        layout="constrained",
        facecolor=t["panel"],
    )
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1, projection=_crs(kind, extent))
    ax.set_facecolor(t["panel"])

    clon, clat, cvalues = _cyclic(lon, lat, values)

    if shading is not None:
        levels = np.asarray(shading["levels"], dtype=float)
        if "colors" in shading and shading.get("extend") == "both":
            # Fitted levels: the palette's colours resampled to the bands, and
            # its end colours below and above, so every value is coloured.
            from matplotlib.colors import LinearSegmentedColormap

            ramp = LinearSegmentedColormap.from_list("fitted", list(shading["colors"]))
            filled = ax.contourf(
                clon, clat, cvalues, levels=levels, cmap=ramp, extend="both", transform=_PC
            )
        elif "colors" in shading:
            colors = list(shading["colors"])
            open_top = len(colors) == len(levels)
            # `colors=` makes matplotlib build the ListedColormap and
            # BoundaryNorm itself, so uneven levels (0.5/2/4/10… mm) map one
            # colour per band. Below the lowest level stays unshaded, as
            # the default style draws it; an open top band is the "over" colour.
            filled = ax.contourf(
                clon,
                clat,
                cvalues,
                levels=levels,
                colors=colors[: len(levels) - 1],
                extend="max" if open_top else "neither",
                transform=_PC,
            )
            if open_top:
                filled.cmap.set_over(colors[-1])
                filled.changed()
        else:
            filled = ax.contourf(
                clon,
                clat,
                cvalues,
                levels=levels,
                cmap=shading["cmap"],
                extend=shading.get("extend", "both"),
                transform=_PC,
            )
        stride = max(1, int(np.ceil(len(levels) / 14)))
        ticks = levels[::stride]
        bar = fig.colorbar(
            filled,
            ax=ax,
            orientation="horizontal",
            shrink=0.75,
            pad=0.03,
            ticks=ticks,
            spacing="uniform",
        )
        bar.set_label(colorbar_label, color=t["ink"])
        # `.6g`, not `.3g`: three significant figures turn 1004 hPa into "1e+03".
        bar.ax.set_xticklabels([f"{v:.6g}" for v in ticks], fontsize=8, color=t["dim"])
        bar.ax.tick_params(colors=t["dim"])
        bar.outline.set_edgecolor(t["rule"])

    if isolines is not None:
        own = isolines.get("values")
        ilon, ilat, ivalues = (
            (clon, clat, cvalues) if own is None else _cyclic(lon, lat, own)
        )
        step = isolines["interval"]
        finite = ivalues[np.isfinite(ivalues)]
        if finite.size:
            contour_levels = np.arange(
                np.floor(finite.min() / step) * step,
                np.ceil(finite.max() / step) * step + step / 2,
                step,
            )
            lines = ax.contour(
                ilon,
                ilat,
                ivalues,
                levels=contour_levels,
                colors=isoline_color(isolines, shading, t),
                linewidths=0.8,
                transform=_PC,
            )
            ax.clabel(lines, fmt="%g", fontsize=7, inline_spacing=2)

    if stipple is not None:
        slon, slat, smask = _cyclic(lon, lat, stipple.astype(float))
        ax.contourf(
            slon,
            slat,
            smask,
            levels=[0.5, 1.5],
            colors="none",
            hatches=["...."],
            transform=_PC,
        )

    _basemap(ax, kind, extent, theme)
    if outline is not None:
        closed = np.vstack([outline, outline[:1]])
        ax.plot(
            closed[:, 0],
            closed[:, 1],
            transform=_PC,  # AR6 edges are straight in lat/lon, as the mask is
            color=t["occl"],
            linewidth=1.6,
            zorder=4,
        )
    _frame(ax, kind, extent)
    if "geo" in ax.spines:
        ax.spines["geo"].set_edgecolor(t["rule"])
    return _finish(fig, ax, title, subtitle, footer, theme)


def _finish(
    fig: Figure, ax, title: str, subtitle: str, footer: str, theme: str = "dark"
) -> bytes:
    """Title, provenance subtitle and footer, then the PNG bytes.

    The subtitle is wrapped here rather than by matplotlib so its line count
    is known: it sits between title and axes, and the title is padded by that
    many lines — otherwise a long provenance string grows up over the title.
    """
    chars = int(fig.get_figwidth() * 14)
    lines = [
        wrapped
        for paragraph in subtitle.split("\n")
        for wrapped in (textwrap.wrap(paragraph, chars) or [""])
    ]
    ax.text(
        0.0,
        1.012,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=8.5,
        color=_tokens(theme)["dim"],
        va="bottom",
        ha="left",
        linespacing=1.3,
    )
    ax.set_title(
        title,
        fontsize=12,
        loc="left",
        fontweight="bold",
        pad=10 + 14 * len(lines),
        color=_tokens(theme)["ink"],
    )
    footer = "\n".join(textwrap.wrap(footer, int(chars * 1.1)))
    fig.supxlabel(footer, fontsize=7.5, color=_tokens(theme)["dim"], x=0.01, ha="left")
    buffer = io.BytesIO()
    fig.savefig(
        buffer, format="png", dpi=_DPI, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Index time series
# ---------------------------------------------------------------------------


def render_index_timeseries(
    frame: pd.DataFrame,
    index: dict,
    *,
    title: str,
    subtitle: str,
    footer: str,
    theme: str = "dark",
) -> bytes:
    """Monthly index as red/blue bars, with a long running mean as a line.

    Short smoothing (ONI's 3 months) is the index itself, so the bars show
    it; long smoothing (AMO's 121 months) is a low-pass view, drawn over the
    monthly bars. Thresholds (El Niño ±0.5 °C, IOD ±0.4 °C) are dashed.
    """
    t = _tokens(theme)
    fig = Figure(figsize=(11, 4.6), layout="constrained", facecolor=t["panel"])
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(t["panel"])
    ax.tick_params(colors=t["dim"])
    for spine in ax.spines.values():
        spine.set_edgecolor(t["rule"])

    window = index.get("smooth")
    bars = frame["value"] if window and window <= 12 else frame["monthly"]
    colors = np.where(bars.to_numpy() >= 0, "#d6604d", "#4393c3")
    ax.bar(bars.index, bars.to_numpy(), width=31, color=colors, linewidth=0)
    if window and window > 12:
        ax.plot(
            frame.index,
            frame["value"],
            color=t["ink"],
            linewidth=1.8,
            label=f"{window}-month running mean",
        )
        ax.legend(loc="upper left", fontsize=8, frameon=False, labelcolor=t["ink"])

    for threshold, label in zip(
        index.get("thresholds", ()), index.get("threshold_labels", ()), strict=False
    ):
        ax.axhline(threshold, color=t["dim"], linewidth=0.8, linestyle="--")
        ax.text(
            bars.index[0],
            threshold,
            f" {label}",
            fontsize=7.5,
            color=t["dim"],
            va="bottom" if threshold > 0 else "top",
        )
    ax.axhline(0, color=t["zero"], linewidth=0.6)

    latest = bars.dropna()
    if len(latest):
        ax.annotate(
            f"{latest.index[-1]:%b %Y}: {latest.iloc[-1]:+.2f} {index['unit']}",
            xy=(latest.index[-1], latest.iloc[-1]),
            xytext=(-8, 12),
            textcoords="offset points",
            ha="right",
            fontsize=8.5,
            fontweight="bold",
            color=t["ink"],
        )
    ax.set_ylabel(index["unit"], color=t["dim"])
    ax.set_xlim(frame.index[0], frame.index[-1] + pd.Timedelta(days=62))
    ax.grid(axis="y", color=t["grid"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return _finish(fig, ax, title, subtitle, footer, theme)
