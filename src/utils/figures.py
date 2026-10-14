"""All go.Figure factories. No API calls here."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from utils.climatology import SEASONS, accumulate, calendar_date
from utils.regions import format_box, format_point
from utils.settings import (
    CLIM_COLORS,
    DETERMINISTIC_MODELS,
    EVAL_REFERENCES,
    FALLBACK_SYSTEM_STYLE,
    GRID_STORES,
    MAP_BORDER_COLOR,
    MAP_COASTLINE_COLOR,
    MAP_PALETTES,
    NATURAL_EARTH_COARSE_ABOVE_DEG,
    NATURAL_EARTH_FILE,
    OBSERVED_COLOR,
    SCORE_DIRECTION,
    SCORECARD_LEAD_STEP_HOURS,
    SCORECARD_MIN_INITS,
    SCORECARD_STORES,
    SCORECARD_VARIABLES,
    SYSTEM_STYLES,
)

CONFIG: dict = {
    "displayModeBar": True,
    "toImageButtonOptions": {
        "format": "png",
        "width": 1400,
        "height": 700,
        "filename": "climate_dashboard",
    },
    "scrollZoom": False,
}

# Transparent paper and plot area: a figure sits on the page's own panel colour
# in either theme. Text, grid and legend colours are applied in the browser by
# `assets/plot_theme.js` from the theme tokens; data colours are set here and
# chosen to read on both themes (see SYSTEM_PALETTE).
_TRANSPARENT = "rgba(0,0,0,0)"

_LAYOUT_DEFAULTS: dict = {
    "template": "plotly_white",
    "paper_bgcolor": _TRANSPARENT,
    "plot_bgcolor": _TRANSPARENT,
    "hovermode": "x unified",
    # Below the plot, as in `_METEOGRAM_LAYOUT`: above it, the legend shares
    # the top margin with the two-line title and wraps over the subtitle once
    # the page sidebar narrows the plot.
    "legend": {"orientation": "h", "yanchor": "top", "y": -0.18, "x": 0},
    "margin": {"t": 80, "r": 20, "l": 60, "b": 110},
}


# Meteogram-specific layout. Deliberately not `_LAYOUT_DEFAULTS`: that sets
# `hovermode="x unified"`, which collapses every jittered member point at a
# given date into one tooltip and defeats per-member hover.
#
# The legend goes *below* the plot, not above it: each system contributes two
# entries (members + control), so an above-plot horizontal legend wraps to
# several rows and overwrites the two-line title sharing the top margin.
_METEOGRAM_LAYOUT: dict = {
    "template": "plotly_white",
    "hovermode": "closest",
    "paper_bgcolor": _TRANSPARENT,
    "plot_bgcolor": _TRANSPARENT,
    "height": 620,
    "legend": {
        "orientation": "h",
        "yanchor": "top",
        "y": -0.16,
        "xanchor": "left",
        "x": 0,
        "bgcolor": _TRANSPARENT,
    },
    "margin": {"t": 80, "r": 24, "l": 64, "b": 110},
}

# Box fill alpha and member-point/control-marker alphas.
_BOX_FILL_ALPHA = 0.3
_CONTROL_FILL_ALPHA = 0.7

# Box width as a fraction of one x-axis step, when a single system is shown.
_BOX_WIDTH_FRAC = 0.35


# Time-series frame: a bold title, one context line (`title.subtitle`), the
# legend under the x axis, then the provenance footer — one line per model
# (init, members, grid). Text sizes step 16 / 12 / 11 px; `plot_theme.js`
# gives the subtitle and the footer (annotation named PROVENANCE) the theme's
# muted text colour. Positions are in pixels from a fixed figure height, so
# the legend and footer never overlap however many models are drawn.
PROVENANCE = "provenance"
# The legend height is estimated here (exports, first paint) and re-measured
# in the browser by `mxFrameFit`, which reads the geometry from `layout.meta`.
_TS_TOP_PX = 74            # title + subtitle + padding
_TS_AXIS_PX = {"date": 64, "linear": 50}  # tick labels (two lines on dates) + axis title
_TS_LEGEND_ROW_PX = 29        # measured: a 12 px legend row plus its gap
_TS_GAP_PX = 8             # between the legend and the footer
_TS_FOOTER_LINE_PX = 15
_TS_PAD_PX = 10
_TS_NOMINAL_WIDTH_PX = 860  # plot width assumed when wrapping the legend


def _legend_rows(fig: go.Figure) -> int:
    """Rows a horizontal legend wraps to at the nominal width (~40 px + 6.5 px a char per entry)."""
    names = [t.name for t in fig.data if t.showlegend is not False and t.name]
    rows, used = 1, 0.0
    for name in names:
        width = 40 + 6.5 * len(name)
        if used and used + width > _TS_NOMINAL_WIDTH_PX:
            rows, used = rows + 1, 0.0
        used += width
    return rows


def _frame_timeseries(
    fig: go.Figure,
    title: str,
    context: str,
    provenance: list[str],
    height: int,
    x_kind: str = "date",
) -> None:
    """Lay out title, context line, legend and provenance footer on a time-series figure.

    Parameters
    ----------
    title : str
        Plain words; drawn bold, 16 px.
    context : str
        One line: where, how it was reduced, which window (12 px, muted).
    provenance : list[str]
        One line per model or source (11 px, muted), under the legend. Kept in
        the figure so a PNG export carries it.
    height : int
        Figure height in px; matches the page's graph height.
    x_kind : str
        "date" (two-line tick labels) or "linear".
    """
    axis_px = _TS_AXIS_PX[x_kind]
    footer_top = axis_px + _legend_rows(fig) * _TS_LEGEND_ROW_PX + _TS_GAP_PX
    footer_px = len(provenance) * _TS_FOOTER_LINE_PX
    bottom = footer_top + footer_px + _TS_PAD_PX
    plot_px = height - _TS_TOP_PX - bottom
    fig.update_layout(
        height=height,
        meta={
            "mx_frame": {
                "axis_px": axis_px,
                "gap_px": _TS_GAP_PX,
                "line_px": _TS_FOOTER_LINE_PX,
                "lines": len(provenance),
                "pad_px": _TS_PAD_PX,
            }
        },
        margin={"t": _TS_TOP_PX, "b": bottom, "l": 64, "r": 24},
        title={
            "text": f"<b>{title}</b>",
            "font": {"size": 16},
            "subtitle": {"text": context, "font": {"size": 12}},
            "x": 0,
            "xref": "paper",
            "xanchor": "left",
            "y": 1,
            "yref": "container",
            "yanchor": "top",
            "pad": {"t": 12},
        },
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -axis_px / plot_px,
            "yanchor": "top",
            "bgcolor": _TRANSPARENT,
        },
    )
    if provenance:
        fig.add_annotation(
            name=PROVENANCE,
            text="<br>".join(provenance),
            xref="paper",
            yref="paper",
            x=0,
            xanchor="left",
            y=0,
            yanchor="top",
            yshift=-footer_top,
            align="left",
            showarrow=False,
            font={"size": 11},
        )


def figure_text(fig: go.Figure) -> str:
    """Every word around the plot: title, subtitle and provenance footer."""
    title = fig.layout.title
    parts = [title.text or "", (title.subtitle.text if title.subtitle else "") or ""]
    parts += [a.text for a in fig.layout.annotations if a.name == PROVENANCE]
    return " ".join(parts)


def _subtitle(model_label: str, clim_period: str, var_meta: dict | None = None) -> str:
    note = (var_meta or {}).get("source_note")
    return f"Source: {model_label}{f' ({note})' if note else ''} · Climatology: {clim_period}"


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert `#rrggbb` to an `rgba(...)` string at the given alpha."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i : i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r}, {g}, {b}, {alpha})"


def _system_offsets(systems: list[str], step: pd.Timedelta) -> dict[str, pd.Timedelta]:
    """Symmetric side-by-side x-offsets for N ensemble systems sharing one axis.

    Side by side: the group spans `min(0.6, 0.2 * n)` of one x-axis
    step, so two systems sit close together and many systems stay inside one
    step's worth of space. A single system is centred on the tick.

    Parameters
    ----------
    systems : list[str]
        Ensemble keys in draw order.
    step : pd.Timedelta
        Spacing between adjacent x positions, derived from the data.

    Returns
    -------
    dict[str, pd.Timedelta]
        Ensemble key -> offset to add to each date.
    """
    n = len(systems)
    if n == 0:
        return {}
    if n == 1:
        return {systems[0]: pd.Timedelta(0)}

    total_span = min(0.6, 0.2 * n)
    return {
        system: (-total_span / 2 + idx * total_span / (n - 1)) * step
        for idx, system in enumerate(systems)
    }


def _x_step(dfs_by_system: dict[str, pd.DataFrame]) -> pd.Timedelta:
    """Median spacing between forecast dates, across all systems.

    Derived from the response rather than assumed daily, so the box width and
    offsets stay correct if the endpoint ever returns sub-daily steps.
    """
    dates = pd.Index(
        sorted({d for df in dfs_by_system.values() for d in df["date"].unique()})
    )
    if len(dates) < 2:
        return pd.Timedelta(days=1)
    return pd.Timedelta(pd.Series(dates).diff().dropna().median())


def _add_envelope(fig: go.Figure, pct: pd.DataFrame, x: pd.Index) -> None:
    """Add the p10-90 / p25-75 / median envelope, outermost band first.

    `fill='tonexty'` fills down to the *previous* trace, so bounds must be
    added outer-upper, outer-lower, inner-upper, inner-lower, median.
    """
    fig.add_trace(
        go.Scatter(
            x=x, y=pct["p90"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=pct["p10"],
            fill="tonexty", fillcolor=CLIM_COLORS["outer_band"],
            line=dict(width=0), name="10th–90th pct", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=pct["p75"], mode="lines",
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=pct["p25"],
            fill="tonexty", fillcolor=CLIM_COLORS["inner_band"],
            line=dict(width=0), name="25th–75th pct", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=pct["p50"],
            mode="lines",
            line=dict(color=CLIM_COLORS["median"], width=1.5, dash="dash"),
            name="Median", hoverinfo="skip",
        )
    )


def _calendar_x(days: pd.Index, year: int) -> list[pd.Timestamp]:
    """Calendar days 1-365 as dates in `year`, for a date axis."""
    return [calendar_date(year, d) for d in days]


def _year_axis(fig: go.Figure, year: int) -> None:
    """Month-labelled date axis spanning the calendar year."""
    fig.update_xaxes(
        title_text="",
        tickformat="%b",
        dtick="M1",
        ticklabelmode="period",
        range=[pd.Timestamp(year, 1, 1), pd.Timestamp(year, 12, 31)],
    )


def _trend_text(slope: float, unit: str, accumulated: bool) -> str:
    what = "annual total" if accumulated else "annual mean"
    return f"Trend in {what}: {slope:+.2f} {unit} per decade"


def make_empty_fig(message: str) -> go.Figure:
    """A blank figure carrying only an annotation, for N/A or empty-data cases."""
    fig = go.Figure()
    fig.add_annotation(
        text=message, xref="paper", yref="paper", x=0.5, y=0.5,
        showarrow=False, font=dict(size=14, color="#888"),
    )
    fig.update_layout(**_LAYOUT_DEFAULTS, xaxis_visible=False, yaxis_visible=False)
    return fig


def make_accumulation_fig(
    pct_acc: pd.DataFrame,
    current: pd.DataFrame,
    forecast: pd.DataFrame | None,
    var_meta: dict,
    model_label: str,
    clim_period: str,
    year: int,
    trend_slope: float | None = None,
) -> go.Figure:
    """YTD cumulative value vs. historical percentile bands.

    Parameters
    ----------
    pct_acc : pd.DataFrame
        Indexed by DOY, columns p10/p25/p50/p75/p90 of cumulative value.
    current : pd.DataFrame
        Selected year's observed series (columns date, value, doy).
    forecast : pd.DataFrame | None
        10-day forecast extension (current year only), or None.
    var_meta : dict
        Entry from `utils.settings.VARIABLES`.
    year : int
        Selected year, for the trace name and title.
    trend_slope : float | None
        Linear trend across annual means, in units/year, from
        `utils.climatology.compute_trend`. Shown as an annotation when given.
    """
    unit = var_meta["unit"]
    label = var_meta["label"]

    fig = go.Figure()
    _add_envelope(fig, pct_acc, _calendar_x(pct_acc.index, year))

    current = current.sort_values("date").copy()
    current["ytd"] = accumulate(current["value"])
    fig.add_trace(
        go.Scatter(
            x=current["date"], y=current["ytd"],
            mode="lines", line=dict(color=CLIM_COLORS["current"], width=2),
            name=str(year),
            hovertemplate="%{x|%d %b}: %{y:.1f} " + unit + "<extra></extra>",
        )
    )

    # The extension has to continue from the last day the record is *complete*
    # through, not from the last row. Those differ: the archive returns rows up
    # to yesterday while the reanalysis lags several days, so the final rows are
    # NaN and `iloc[-1]` is NaN — which silently turned the whole extension into
    # NaN and drew nothing at all.
    observed = current.dropna(subset=["ytd"])
    if forecast is not None and not forecast.empty and not observed.empty:
        anchor = observed.iloc[-1]
        # Forecast days at or before the anchor would double-count against days
        # already accumulated, so only the genuinely new ones are appended.
        forecast = forecast[forecast["date"] > anchor["date"]].sort_values("date").copy()

    if forecast is not None and not forecast.empty:
        last_ytd = observed["ytd"].iloc[-1] if not observed.empty else 0.0
        forecast["ytd"] = last_ytd + accumulate(forecast["value"])

        # Join the two lines explicitly from the anchor, so the dotted segment
        # starts where the solid one ends instead of appearing to float.
        if not observed.empty:
            anchor = observed.iloc[-1]
            forecast = pd.concat(
                [
                    pd.DataFrame(
                        {
                            "doy": [anchor["doy"]],
                            "ytd": [anchor["ytd"]],
                            "date": [anchor["date"]],
                        }
                    ),
                    forecast[["doy", "ytd", "date"]],
                ],
                ignore_index=True,
            )

        gap_days = int((forecast["date"].iloc[1] - forecast["date"].iloc[0]).days) - 1
        name = "Forecast (+10d)"
        if gap_days > 0:
            # Days between the end of the reanalysis and the start of the
            # forecast are in neither series. Saying so beats drawing a
            # straight line across them and calling it an accumulation.
            name += f" — {gap_days}d gap not accumulated"

        fig.add_trace(
            go.Scatter(
                x=forecast["date"], y=forecast["ytd"],
                mode="lines",
                line=dict(color=CLIM_COLORS["forecast"], width=2, dash="dot"),
                name=name,
                hovertemplate="%{x|%d %b}: %{y:.1f} " + unit + "<extra></extra>",
            )
        )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"{label} — YTD Accumulation<br>"
            f"<sup>{_subtitle(model_label, clim_period, var_meta)}</sup>"
        ),
        yaxis_title=f"Cumulative {label.lower()} ({unit})",
    )
    _year_axis(fig, year)

    if trend_slope is not None and np.isfinite(trend_slope):
        fig.add_annotation(
            text=_trend_text(trend_slope, unit, accumulated=True),
            xref="paper", yref="paper", x=1, y=1.08, xanchor="right",
            showarrow=False, font=dict(size=11, color=CLIM_COLORS["trend"]),
        )

    return fig


def make_daily_fig(
    pct_daily: pd.DataFrame,
    current: pd.DataFrame,
    var_meta: dict,
    model_label: str,
    clim_period: str,
    year: int,
) -> go.Figure:
    """Day-by-day absolute value vs. historical percentile bands.

    Days outside P10/P90 are colored darker to flag extremes.
    """
    unit = var_meta["unit"]
    label = var_meta["label"]

    fig = go.Figure()
    _add_envelope(fig, pct_daily, _calendar_x(pct_daily.index, year))

    current = current.sort_values("date").copy()
    p10 = pct_daily.loc[current["doy"].values, "p10"].values
    p90 = pct_daily.loc[current["doy"].values, "p90"].values
    above_p90 = current["value"].values > p90
    below_p10 = current["value"].values < p10

    colors = [
        CLIM_COLORS["extreme_hot"] if is_hot
        else CLIM_COLORS["extreme_cold"] if is_cold
        else CLIM_COLORS["current"]
        for is_hot, is_cold in zip(above_p90, below_p10)
    ]

    fig.add_trace(
        go.Scatter(
            x=current["date"], y=current["value"],
            mode="markers+lines",
            marker=dict(color=colors, size=5),
            line=dict(color=CLIM_COLORS["current"], width=1.5),
            name=str(year),
            hovertemplate="%{x|%d %b}: %{y:.1f} " + unit + "<extra></extra>",
        )
    )

    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"{label} — Daily values<br>"
            f"<sup>{_subtitle(model_label, clim_period, var_meta)} · bands pooled over ±7 days</sup>"
        ),
        yaxis_title=f"{label} ({unit})",
    )
    _year_axis(fig, year)
    return fig


def _step_window_note(meta: dict) -> str:
    """Trailing subtitle clause naming a field's step window, if it has one.

    Only step-averaged or step-maximum fields carry one, derived from the
    store's own lead axis by `earthmover_api.step_window`. An instantaneous
    field is valid at its lead time and saying otherwise would be noise.
    """
    window = meta.get("step_window")
    return f" · {window}" if window else ""


def _bbox_label(meta: dict) -> str:
    """Human-readable box extent: cell counts, degrees, and native resolution."""
    extent = format_box(tuple(meta["bbox"]))
    if meta.get("region_label"):
        cells = meta.get("n_cells_in_region") or meta["n_lat"] * meta["n_lon"]
        return (
            f"{meta['region_label']} · {meta['resolution_deg']:g}° gridbox mean over "
            f"{cells:,} cells ({extent})"
        )
    return (
        f"{meta['resolution_deg']:g}° gridbox mean, {meta['n_lat']}×{meta['n_lon']} cells "
        f"({extent})"
    )


def _store_color(store: str) -> str:
    """The store's colour (its Open-Meteo twin's where it has one)."""
    return SYSTEM_STYLES.get(store, FALLBACK_SYSTEM_STYLE)["color"]


def make_regional_multi_fig(results: dict[str, tuple], var_meta: dict) -> go.Figure:
    """Area-averaged ensemble spread for one or more models, never pooled.

    Each model gets its own colour: a P10–P90 and a P25–P75 band, a bold
    ensemble-mean line, and its members as one trace. With several models the
    members start hidden (click the legend to show them): three models'
    spaghetti over each other hides the bands, which are the comparison.

    Parameters
    ----------
    results : dict
        `{store: (df, pct, meta)}` — the tidy member frame and `meta` from
        `earthmover_api.fetch_bbox_ensemble`, and `pct` from
        `climatology.compute_member_pct`. Insertion order is drawing order.
    var_meta : dict
        Entry from `utils.settings.GRID_VARIABLES`.

    Notes
    -----
    The context line states the extent once (when every model read the same
    cells), the weighting and the window; the footer gives per model its
    label, init, member count, grid and step window — spread is
    uninterpretable without the run age.
    """
    first_meta = next(iter(results.values()))[2]
    accum_hours = first_meta.get("accum_hours")
    if accum_hours:
        unit = first_meta.get("unit", var_meta["unit"])
        label = f"{var_meta.get('accum_label', var_meta['label'])} ({accum_hours} h total)"
    else:
        unit = var_meta["unit"]
        label = var_meta["label"]
    several = sum(GRID_STORES[s].get("kind") != "deterministic" for s in results) > 1

    fig = go.Figure()
    clauses = []
    # Each store snaps the area to its own grid (and ICON-EU clips it), so the
    # extent is stated once only when every model read the same cells.
    extents = {store: _bbox_label(meta) for store, (_, _, meta) in results.items()}
    shared = len(set(extents.values())) == 1
    for store, (df, pct, meta) in results.items():
        name = GRID_STORES[store]["label"]
        own_extent = "" if shared else f" · {extents[store]}"
        color = _store_color(store)
        x = pct.index
        if GRID_STORES[store].get("kind") == "deterministic":
            # One run, one line: no members to spread, so no bands to draw.
            series = df.groupby("valid_time")["value"].mean().sort_index()
            fig.add_trace(
                go.Scatter(
                    x=series.index, y=series.to_numpy(), mode="lines",
                    line={"color": color, "width": 2.5, "dash": "dash"},
                    name=f"{name} (deterministic)", legendgroup=f"{store}-line",
                    hovertemplate=(
                        f"{name}<br>%{{x|%a %d %b %H:%M}}<br>%{{y:.1f}} {unit}<extra></extra>"
                    ),
                )
            )
            init = pd.Timestamp(meta["init_time"])
            window = "" if accum_hours else _step_window_note(meta)
            clip = f" · {meta['clip_note']}" if meta.get("clip_note") else ""
            clauses.append(
                f"{GRID_STORES[store]['model_label']} init {init:%Y-%m-%d %H} UTC · "
                f"deterministic · {meta['resolution_deg']:g}°{own_extent}{window}{clip}"
            )
            continue
        for lo, hi, alpha, band in (("p10", "p90", 0.15, "P10–P90"), ("p25", "p75", 0.30, "P25–P75")):
            fig.add_trace(
                go.Scatter(
                    x=x, y=pct[hi], mode="lines", line={"width": 0},
                    legendgroup=f"{store}-band", showlegend=False, hoverinfo="skip",
                    name=f"{name} {band}",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x, y=pct[lo], mode="lines", line={"width": 0}, fill="tonexty",
                    fillcolor=_hex_to_rgba(color, alpha), legendgroup=f"{store}-band",
                    showlegend=band == "P10–P90", name=f"{name} {band}" if band != "P10–P90" else f"{name} P10–P90 / P25–P75",
                    hoverinfo="skip",
                )
            )

        # Members as one trace, broken by None between members: one legend
        # entry and one payload entry per model instead of 51.
        xs, ys, ids = [], [], []
        for member, group in df.sort_values("valid_time").groupby("member", sort=True):
            xs += [*group["valid_time"], None]
            ys += [*group["value"], None]
            ids += [member] * len(group) + [None]
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, customdata=ids, mode="lines",
                line={"color": _hex_to_rgba(color, 0.25), "width": 1},
                name=f"{name} members ({meta['n_members']})",
                legendgroup=f"{store}-members",
                visible="legendonly" if several else True,
                hovertemplate=(
                    f"{name} member %{{customdata}}<br>%{{x|%a %d %b %H:%M}}<br>"
                    f"%{{y:.1f}} {unit}<extra></extra>"
                ),
            )
        )

        mean = df.groupby("valid_time")["value"].mean().sort_index()
        fig.add_trace(
            go.Scatter(
                x=mean.index, y=mean.to_numpy(), mode="lines",
                line={"color": color, "width": 3},
                name=f"{name} ensemble mean", legendgroup=f"{store}-mean",
                hovertemplate=(
                    f"{name} ensemble mean<br>%{{x|%a %d %b %H:%M}}<br>"
                    f"%{{y:.1f}} {unit}<extra></extra>"
                ),
            )
        )
        init = pd.Timestamp(meta["init_time"])
        window = "" if accum_hours else _step_window_note(meta)
        clip = f" · {meta['clip_note']}" if meta.get("clip_note") else ""
        clauses.append(
            f"{GRID_STORES[store]['model_label']} init {init:%Y-%m-%d %H} UTC · "
            f"{meta['n_members']} members · {meta['resolution_deg']:g}°{own_extent}{window}{clip}"
        )

    head = f" · {accum_hours} h rolling accumulation, ending at each lead" if accum_hours else ""
    where = extents[next(iter(results))] if shared else "each model on its own grid"
    context = f"{where} · cos(lat)-weighted mean · percentiles across each model's own members{head}"
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        yaxis_title=f"{label} ({unit})",
        xaxis_title="Valid time (UTC)",
    )
    _frame_timeseries(fig, f"Regional {label}", context, clauses, height=620)
    # "closest", not the shared "x unified": unified lists every member at
    # the hovered time; closest names the one line under the cursor.
    fig.update_layout(hovermode="closest", hoverdistance=12)
    return fig


def make_regional_ensemble_fig(
    df: pd.DataFrame,
    pct: pd.DataFrame,
    var_meta: dict,
    meta: dict,
    model_label: str | None = None,
) -> go.Figure:
    """One model's area-averaged spread; see `make_regional_multi_fig`.

    `model_label` is accepted for old callers and ignored: the label comes
    from `meta["store"]`.
    """
    return make_regional_multi_fig({meta.get("store", "ifs_ens"): (df, pct, meta)}, var_meta)


# Map colorscales, keyed by grid variable. Sequential (one hue, light->dark)
# for magnitudes with a natural zero; diverging with a neutral midpoint for
# fields read as departures either side of a reference. No rainbow anywhere —
# it manufactures banding that is not in the data.
_MAP_COLORSCALES: dict[str, tuple[str, float | None]] = {
    "temperature_2m": ("RdBu_r", 0.0),
    "temperature_850hpa": ("RdBu_r", 0.0),
    "dew_point_temperature_2m": ("RdBu_r", 0.0),
    "precipitation_surface": ("Blues", None),
    "wind_gust_10m": ("Purples", None),
    "total_cloud_cover_atmosphere": ("Greys", None),
    "pressure_reduced_to_mean_sea_level": ("Viridis", None),
    "geopotential_height_500hpa": ("Viridis", None),
}
_MAP_FALLBACK_COLORSCALE = ("Viridis", None)


def make_regional_map_fig(
    df: pd.DataFrame,
    var_meta: dict,
    meta: dict,
    model_label: str,
) -> go.Figure:
    """Map of a gridded field over the selected box at one lead time.

    Cells are drawn at their native spacing on a linear lat/lon plot with the
    x-axis locked to the y-axis by `1/cos(lat)`, so a 0.25° cell is square in
    ground distance rather than in degrees. No smoothing or interpolation is
    applied: the store's gridbox values are what gets shown.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy field frame from `utils.earthmover_api.fetch_bbox_field`, with
        columns `latitude`, `longitude`, `value`.
    var_meta : dict
        Entry from `utils.settings.GRID_VARIABLES`.
    meta : dict
        Metadata payload from `fetch_bbox_field` — init and valid time, lead,
        the member reduction, cell counts, and snapped box extent.
    model_label : str
        `utils.settings.GRID_MODEL_LABEL`.

    Notes
    -----
    As on the time-series figure, the subtitle carries the provenance the plot
    is meaningless without: which members it reduces, the box extent and grid
    resolution, the model, and the init time alongside the valid time — a
    field is uninterpretable without knowing how old the run behind it is.
    """
    unit = var_meta["unit"]
    label = var_meta["label"]
    init = pd.Timestamp(meta["init_time"])
    valid = pd.Timestamp(meta["valid_time"])

    grid = df.pivot_table(index="latitude", columns="longitude", values="value")
    colorscale, midpoint = _MAP_COLORSCALES.get(
        df["variable"].iloc[0], _MAP_FALLBACK_COLORSCALE
    )

    fig = go.Figure(
        go.Heatmap(
            x=grid.columns.to_numpy(),
            y=grid.index.to_numpy(),
            z=grid.to_numpy(),
            colorscale=colorscale,
            # `zmid` anchors the neutral midpoint to the physically meaningful
            # value (0 °C) without forcing symmetric limits — a summer field
            # would otherwise spend half the ramp on temperatures not present.
            zmid=midpoint,
            zsmooth=False,
            colorbar=dict(title=f"{unit}", thickness=14, outlinewidth=0),
            hovertemplate=(
                "%{y:.2f}°N, %{x:.2f}°E<br>%{z:.1f} " + unit + "<extra></extra>"
            ),
        )
    )

    lat_min, lat_max, _, _ = meta["bbox"]
    mid_lat = (lat_min + lat_max) / 2.0
    aspect = 1.0 / max(np.cos(np.deg2rad(mid_lat)), 0.05)

    subtitle = (
        f"{meta['reduction']} · T+{meta['lead_hours']} h · "
        f"valid {valid:%Y-%m-%d %H:%M} UTC{_step_window_note(meta)}<br>"
        f"{_bbox_label(meta)} · {model_label} · init {init:%Y-%m-%d %H:%M} UTC"
    )
    layout = {**_LAYOUT_DEFAULTS, "hovermode": "closest"}
    fig.update_layout(
        **layout,
        title=dict(text=f"<b>{label}</b><br><sup>{subtitle}</sup>", x=0.01),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        height=640,
    )
    fig.update_yaxes(scaleanchor="x", scaleratio=aspect, constrain="domain")
    return fig


def make_meteogram_fig(
    dfs_by_system: dict[str, pd.DataFrame],
    system_labels: dict[str, str],
    var_meta: dict,
    location_name: str,
    subtitle: str | None = None,
    provenance: list[str] | None = None,
) -> go.Figure:
    """Overlay one or more ensemble systems on a single meteogram as box/whisker plots.

    Parameters
    ----------
    dfs_by_system : dict[str, pd.DataFrame]
        Ensemble key -> member frame (columns date, value, member) from
        `utils.openmeteo_api.fetch_ensemble`. One entry per selected system.
    system_labels : dict[str, str]
        Ensemble key -> display label, from `utils.settings.ENSEMBLE_MODELS`.
    var_meta : dict
        Entry from `utils.settings.VARIABLES`.
    location_name : str
        For the plot subtitle.
    subtitle : str, optional
        The context line; defaults to the place and the time of the latest runs.
    provenance : list[str], optional
        Footer lines, one per system (run, members, grid cell).

    Notes
    -----
    A forecast-evolution plot. Each system gets its
    own color: perturbed members (1..N) are drawn as one `go.Box` per system,
    with every member shown as a jittered point over a translucent box
    (`boxpoints="all"`), so inter-model spread and intra-model spread are
    both legible. Plotly aggregates per unique `x` (date), so one trace
    yields one box per forecast day and a single legend row. The control run
    (member 0) is overlaid as a bold line with star markers.

    Setting an explicit box `width` disables Plotly's automatic `boxmode`
    offsetting, so systems are positioned manually via `_system_offsets`.
    Never average values across systems — only within one system's members.
    """
    unit = var_meta["unit"]
    label = var_meta["label"]

    drawn = [s for s, df in dfs_by_system.items() if not df.empty]
    step = _x_step({s: dfs_by_system[s] for s in drawn}) if drawn else pd.Timedelta(days=1)
    offsets = _system_offsets(drawn, step)

    # Widen boxes up to that fraction, but never past the gap between
    # adjacent systems — otherwise 3+ systems overlap into an unreadable mess.
    if len(drawn) > 1:
        gap = min(0.6, 0.2 * len(drawn)) / (len(drawn) - 1)
        width_frac = min(_BOX_WIDTH_FRAC, 0.9 * gap)
    else:
        width_frac = _BOX_WIDTH_FRAC
    box_width = width_frac * step.total_seconds() * 1000.0  # date axis width is in ms

    fig = go.Figure()
    for system in drawn:
        df = dfs_by_system[system]
        style = SYSTEM_STYLES.get(system, FALLBACK_SYSTEM_STYLE)
        color = style["color"]
        system_label = system_labels[system]
        offset = offsets[system]
        first_date = df["date"].min()

        if (
            GRID_STORES.get(system, {}).get("kind") == "deterministic"
            or system in DETERMINISTIC_MODELS
        ):
            # One run: one line, no boxes, no "mean" of a single member.
            run = df.sort_values("date")
            fig.add_trace(
                go.Scatter(
                    x=run["date"] + offset,
                    y=run["value"],
                    mode="lines+markers",
                    line={"color": color, "width": 2.5, "dash": "dash"},
                    marker={"color": color, "size": 5},
                    legendgroup=f"{system}-line",
                    name=system_label if "deterministic" in system_label
                    else f"{system_label} (deterministic)",
                    customdata=(run["date"] - first_date).dt.days,
                    hovertemplate=(
                        "%{x|%a %d %b}<br>Lead time: +%{customdata}d<br>"
                        f"%{{y:.1f}} {unit}<extra>{system_label}</extra>"
                    ),
                )
            )
            continue

        perturbed = df[df["member"] != 0].sort_values(["date", "member"])
        if not perturbed.empty:
            lead_days = (perturbed["date"] - first_date).dt.days
            fig.add_trace(
                go.Box(
                    x=perturbed["date"] + offset,
                    y=perturbed["value"],
                    name=f"{system_label} members",
                    legendgroup=f"{system}-members",
                    width=box_width,
                    fillcolor=_hex_to_rgba(color, _BOX_FILL_ALPHA),
                    marker=dict(color=color, size=4, symbol=style["symbol"]),
                    line=dict(color=color, width=1.2),
                    boxpoints="all",
                    jitter=0.3,
                    pointpos=0,
                    whiskerwidth=0.6,
                    customdata=list(
                        zip(
                            perturbed["date"].dt.strftime("%a %d %b"),
                            lead_days,
                            perturbed["member"],
                        )
                    ),
                    hovertemplate=(
                        f"System: {system_label}<br>"
                        "Date: %{customdata[0]}<br>"
                        "Lead time: +%{customdata[1]}d<br>"
                        "Member: %{customdata[2]}<br>"
                        f"Value: %{{y:.1f}} {unit}<extra></extra>"
                    ),
                )
            )

        # Ensemble mean over this system's members only — never across systems.
        mean = df.groupby("date")["value"].mean().sort_index()
        if not mean.empty:
            fig.add_trace(
                go.Scatter(
                    x=mean.index + offset,
                    y=mean.to_numpy(),
                    mode="lines",
                    line=dict(color=color, width=3),
                    legendgroup=f"{system}-mean",
                    name=f"{system_label} ensemble mean",
                    customdata=(mean.index - first_date).days,
                    hovertemplate=(
                        "%{x|%a %d %b}<br>Lead time: +%{customdata}d<br>"
                        f"Mean: %{{y:.1f}} {unit}<extra>{system_label} mean</extra>"
                    ),
                )
            )

        control = df[df["member"] == 0].sort_values("date")
        if not control.empty:
            lead_days = (control["date"] - first_date).dt.days
            fig.add_trace(
                go.Scatter(
                    x=control["date"] + offset,
                    y=control["value"],
                    mode="lines+markers",
                    line=dict(color=color, width=1.5, dash=style["dash"]),
                    marker=dict(
                        symbol="star",
                        size=11,
                        color=_hex_to_rgba(color, _CONTROL_FILL_ALPHA),
                        line=dict(color=color, width=1.5),
                    ),
                    legendgroup=f"{system}-control",
                    name=f"{system_label} control",
                    visible="legendonly",
                    showlegend=True,
                    customdata=list(
                        zip(control["date"].dt.strftime("%a %d %b"), lead_days)
                    ),
                    hovertemplate=(
                        "Date: %{customdata[0]}<br>"
                        "Lead time: +%{customdata[1]}d<br>"
                        f"Value: %{{y:.1f}} {unit}"
                        f"<extra>{system_label} control</extra>"
                    ),
                )
            )

    fig.update_layout(
        **_METEOGRAM_LAYOUT,
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor="white",
            gridwidth=1,
            tickangle=0,
        ),
        yaxis=dict(
            title=f"{label} ({unit})",
            showgrid=True,
            gridcolor="white",
            gridwidth=1,
        ),
    )
    context = subtitle or (
        f"{location_name} · latest runs as of "
        f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:00} UTC"
    )
    _frame_timeseries(
        fig, f"{label} — ensemble meteogram", context, provenance or [],
        height=_METEOGRAM_LAYOUT["height"],
    )
    return fig


def make_anomaly_bar_fig(
    pct_daily: pd.DataFrame,
    current: pd.DataFrame,
    var_meta: dict,
    model_label: str,
    clim_period: str,
    year: int,
) -> go.Figure:
    """Signed anomaly relative to the median, as a bar chart."""
    unit = var_meta["unit"]
    label = var_meta["label"]

    merged = current.sort_values("date").copy()
    p10 = pct_daily.loc[merged["doy"].values, "p10"].values
    p50 = pct_daily.loc[merged["doy"].values, "p50"].values
    p90 = pct_daily.loc[merged["doy"].values, "p90"].values
    merged["anomaly"] = merged["value"] - p50
    extreme = (merged["value"].values > p90) | (merged["value"].values < p10)

    colors = [
        CLIM_COLORS["extreme_hot"] if (anomaly > 0 and is_extreme)
        else CLIM_COLORS["current"] if anomaly > 0
        else CLIM_COLORS["extreme_cold"] if is_extreme
        else CLIM_COLORS["median"]
        for anomaly, is_extreme in zip(merged["anomaly"], extreme)
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=merged["date"], y=merged["anomaly"],
            marker_color=colors,
            name="Anomaly vs median",
            hovertemplate="%{x|%b %d}: %{y:+.1f} " + unit + "<extra></extra>",
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"{label} — Daily anomaly vs median<br>"
            f"<sup>{_subtitle(model_label, clim_period, var_meta)}</sup>"
        ),
        yaxis_title=f"Anomaly ({unit})",
    )
    _year_axis(fig, year)
    return fig


_MONTH_NAMES = [pd.Timestamp(2001, m, 1).strftime("%b") for m in range(1, 13)]


def _category_color(value: float, row: pd.Series) -> str:
    """Tercile-and-tail colour of a value against its reference percentiles."""
    if not np.isfinite(value):
        return "rgba(0,0,0,0)"
    if value > row["p90"]:
        return CLIM_COLORS["extreme_hot"]
    if value > row["p75"]:
        return CLIM_COLORS["current"]
    if value < row["p10"]:
        return CLIM_COLORS["extreme_cold"]
    if value < row["p25"]:
        return CLIM_COLORS["median"]
    return "#7f7f7f"


def make_monthly_fig(
    monthly: pd.DataFrame,
    var_meta: dict,
    model_label: str,
    clim_period: str,
    year: int,
) -> go.Figure:
    """This year's months against the reference distribution of each month.

    The reference is drawn as a floating 10th–90th and 25th–75th percentile
    bar per month with the median ticked; the year is a marker coloured by
    where it falls. A month still in progress is an open marker compared
    over the same days only, and its hover says how many.
    """
    unit = var_meta["unit"]
    label = var_meta["label"]
    accumulated = var_meta["accumulate"]
    what = "total" if accumulated else "mean"
    x = _MONTH_NAMES

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=x, y=monthly["p90"] - monthly["p10"], base=monthly["p10"],
            marker_color=CLIM_COLORS["outer_band"], width=0.62,
            name="10th–90th pct", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Bar(
            x=x, y=monthly["p75"] - monthly["p25"], base=monthly["p25"],
            marker_color=CLIM_COLORS["inner_band"], width=0.62,
            name="25th–75th pct", hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x, y=monthly["p50"], mode="markers",
            marker=dict(symbol="line-ew", size=26, line=dict(width=2, color=CLIM_COLORS["median"])),
            name="Median", hoverinfo="skip",
        )
    )

    notes = []
    for month, row in monthly.iterrows():
        if row["n_days"] and not row["complete"]:
            notes.append(f"{row['n_days']} of {row['days_in_month']} days")
        else:
            notes.append("full month" if row["n_days"] else "")
    extra = (
        monthly["pct_of_normal"].map(lambda v: f"{v:.0f} % of normal" if np.isfinite(v) else "")
        if accumulated
        else monthly["anomaly"].map(lambda v: f"{v:+.1f} {unit} vs mean" if np.isfinite(v) else "")
    )
    colors = [_category_color(v, row) for v, (_, row) in zip(monthly["value"], monthly.iterrows(), strict=True)]
    fig.add_trace(
        go.Scatter(
            x=x, y=monthly["value"], mode="markers",
            marker=dict(
                size=13,
                color=[c if row["complete"] else "white" for c, (_, row) in zip(colors, monthly.iterrows(), strict=True)],
                line=dict(width=2.5, color=colors),
            ),
            name=str(year),
            customdata=np.stack([extra, notes, monthly["percentile"]], axis=-1),
            hovertemplate=(
                "%{x} " + str(year) + ": %{y:.1f} " + unit + "<br>%{customdata[0]}"
                "<br>above %{customdata[2]:.0f} % of reference years"
                "<br>%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        barmode="overlay",
        title=dict(
            text=f"{label} — Monthly {what}s, {year}<br>"
            f"<sup>{_subtitle(model_label, clim_period, var_meta)} · open marker: month in "
            "progress, compared over the same days</sup>"
        ),
        yaxis_title=f"Monthly {what} ({unit})",
    )
    fig.update_layout(hovermode="closest")
    return fig


def make_seasonal_fig(
    seasonal: pd.DataFrame,
    season_stats: dict,
    season: str,
    var_meta: dict,
    model_label: str,
    clim_period: str,
    year: int,
) -> go.Figure:
    """One season's anomaly in every year, the selected year highlighted.

    Anomalies are against the reference mean of complete seasons; a season
    still in progress is compared over the same days and drawn hatched. The
    band is ±1 standard deviation of the reference seasons.
    """
    unit = var_meta["unit"]
    label = var_meta["label"]
    accumulated = var_meta["accumulate"]
    what = "total" if accumulated else "mean"
    data = seasonal.dropna(subset=["anomaly"])
    std = season_stats["std"]

    colors = np.where(data["anomaly"] >= 0, CLIM_COLORS["current"], CLIM_COLORS["median"])
    line_colors = [CLIM_COLORS["record"] if y == year else "rgba(0,0,0,0)" for y in data.index]
    hover_extra = (
        data["pct_of_normal"].map(lambda v: f"{v:.0f} % of normal")
        if accumulated
        else pd.Series("", index=data.index)
    )
    fig = go.Figure()
    fig.add_hrect(
        y0=-std, y1=std, fillcolor=CLIM_COLORS["outer_band"], line_width=0, layer="below"
    )
    fig.add_trace(
        go.Bar(
            x=data.index, y=data["anomaly"],
            marker=dict(
                color=colors,
                line=dict(color=line_colors, width=2.5),
                pattern=dict(shape=["/" if not c else "" for c in data["complete"]]),
            ),
            name=f"{season} anomaly",
            customdata=np.stack([data["value"], hover_extra, data["n_days"], data["season_days"]], axis=-1),
            hovertemplate=(
                season + " %{x}: %{y:+.2f} " + unit + " (%{customdata[0]:.1f} " + unit + ")"
                "<br>%{customdata[1]}<br>%{customdata[2]} of %{customdata[3]} days"
                "<extra></extra>"
            ),
        )
    )
    slope = season_stats.get("slope_per_decade", float("nan"))
    complete = data[data["complete"]]
    if len(complete) >= 3 and np.isfinite(slope):
        years = complete.index.to_numpy(dtype=float)
        fit = np.polyfit(years, complete["anomaly"].to_numpy(), 1)
        fig.add_trace(
            go.Scatter(
                x=years, y=np.polyval(fit, years), mode="lines",
                line=dict(color=CLIM_COLORS["trend"], width=2),
                name=f"Trend {slope:+.2f} {unit}/decade", hoverinfo="skip",
            )
        )
    months = "–".join(pd.Timestamp(2001, m, 1).strftime("%b") for m in SEASONS[season])
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        title=dict(
            text=f"{label} — {season} ({months}) {what} anomaly<br>"
            f"<sup>{_subtitle(model_label, clim_period, var_meta)} · band: ±1σ of reference "
            f"seasons · outlined: {year} · hatched: season in progress</sup>"
        ),
        yaxis_title=f"Anomaly ({unit})",
        xaxis_title="Season year" + (" (DJF: year of Jan–Feb)" if season == "DJF" else ""),
    )
    fig.update_layout(hovermode="closest", bargap=0.15)
    return fig

# ---------------------------------------------------------------------------
# Field maps (CDS / ADS)
# ---------------------------------------------------------------------------

# Natural Earth line work, loaded once per process and memoised in-module.
# The map draws its own coastlines because a Plotly heatmap lives on cartesian
# axes: only `geo` subplots have built-in coastlines, and they cannot render a
# heatmap. Built by `scripts/build_natural_earth.py`.
_NATURAL_EARTH_PATH = (
    Path(__file__).resolve().parent.parent / "data" / NATURAL_EARTH_FILE
)
_natural_earth_cache: dict[str, np.ndarray] | None = None

# Fields larger than a hemisphere get the 110 m lines; anything regional gets
# 50 m, where the extra 60k vertices actually resolve something.
_NE_LAYER_SUFFIX = {True: "110", False: "50"}


def _natural_earth() -> dict[str, np.ndarray]:
    """Load the bundled line work, or an empty mapping if it is not present.

    A missing file degrades the map to a bare heatmap rather than failing the
    callback: the field is the data, the coastlines are context.
    """
    global _natural_earth_cache

    if _natural_earth_cache is not None:
        return _natural_earth_cache

    if not _NATURAL_EARTH_PATH.is_file():
        _natural_earth_cache = {}
        return _natural_earth_cache

    try:
        with np.load(_NATURAL_EARTH_PATH) as archive:
            _natural_earth_cache = {name: archive[name] for name in archive.files}
    except (OSError, ValueError):
        _natural_earth_cache = {}

    return _natural_earth_cache


def _clip_lines(
    lon: np.ndarray, lat: np.ndarray, extent: tuple[float, float, float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Blank out line vertices outside the map extent.

    Out-of-view points become NaN rather than being dropped, which both keeps
    the polyline breaks intact and stops Plotly drawing a chord straight
    across the map between two distant in-view segments. The extent is padded
    by one cell's worth so lines reach the plot edge instead of stopping short.
    """
    lat_min, lat_max, lon_min, lon_max = extent
    pad_lat = max((lat_max - lat_min) * 0.02, 0.5)
    pad_lon = max((lon_max - lon_min) * 0.02, 0.5)

    inside = (
        (lon >= lon_min - pad_lon)
        & (lon <= lon_max + pad_lon)
        & (lat >= lat_min - pad_lat)
        & (lat <= lat_max + pad_lat)
    )
    if not inside.any():
        return np.array([]), np.array([])

    clipped_lon = np.where(inside, lon, np.nan)
    clipped_lat = np.where(inside, lat, np.nan)
    return clipped_lon, clipped_lat


def _basemap_traces(extent: tuple[float, float, float, float]) -> list[go.Scattergl]:
    """Coastline and national-border traces clipped to the extent."""
    lines = _natural_earth()
    if not lines:
        return []

    lat_min, lat_max, lon_min, lon_max = extent
    span = max(lat_max - lat_min, lon_max - lon_min)
    suffix = _NE_LAYER_SUFFIX[span > NATURAL_EARTH_COARSE_ABOVE_DEG]

    traces: list[go.Scattergl] = []
    for layer, color, width in (
        ("borders", MAP_BORDER_COLOR, 0.5),
        ("coastline", MAP_COASTLINE_COLOR, 0.9),
    ):
        key = f"{layer}_{suffix}"
        if f"{key}_lon" not in lines:
            continue

        lon, lat = _clip_lines(lines[f"{key}_lon"], lines[f"{key}_lat"], extent)
        if lon.size == 0:
            continue

        traces.append(
            go.Scattergl(
                x=lon,
                y=lat,
                mode="lines",
                line=dict(color=color, width=width),
                hoverinfo="skip",
                showlegend=False,
                name=layer,
            )
        )
    return traces


def _map_resolution_label(meta: dict) -> str:
    """Resolution text that states what is *drawn*, not only what was retrieved.

    A global 0.25° field exceeds the browser's cell budget and is strided down
    for display. Saying "0.25°" for a picture drawn at 0.5° would be exactly
    the silent label/code mismatch the project forbids, so both numbers appear
    whenever they differ.
    """
    native = meta.get("resolution_deg")
    display = meta.get("display_resolution_deg")

    if native is None:
        return "resolution unknown"
    if meta.get("regridded_from"):
        return f"{native:g}° grid (regridded by CDS from {meta['regridded_from']:g}°)"
    if display and abs(display - native) > 1e-6:
        return f"{native:g}° native, drawn at {display:g}° (thinned for display)"
    return f"{native:g}° native grid"


def _extent_label(extent: tuple[float, float, float, float]) -> str:
    lat_min, lat_max, lon_min, lon_max = extent
    return f"{lat_min:g}–{lat_max:g}°N, {lon_min:g}–{lon_max:g}°E"


def window_hours(meta: dict) -> float:
    """Length of the window a displayed accumulation or flux is totalled over."""
    mode = meta.get("monthly_mode")
    if mode == "per_month" and meta.get("season"):
        start = pd.Timestamp(meta["valid_time"])
        return 24.0 * (start + pd.DateOffset(months=3) - start).days
    if mode == "per_month":
        return 24.0 * pd.Timestamp(meta["valid_time"]).days_in_month
    if mode == "per_day":
        return 24.0
    return float(meta.get("accum_hours") or 1)


def palette_levels(palette: dict, window_hours: float) -> np.ndarray:
    """Band edges of a default-style palette, in the displayed unit.

    A `flux_w_m2` palette is stated as a mean flux and converted to the energy
    over the window, MJ/m² = W/m² × 3600 s/h × hours / 1e6. The flux range
    itself depends on the window: a daily mean never approaches the noon peak
    an hourly field reaches.
    """
    if "range" in palette:
        spec = palette["range"]
        if palette.get("flux_w_m2") and window_hours < 24:
            spec = palette.get("range_sub_daily", spec)
        start, stop, step = spec
        levels = np.arange(start, stop + step / 2, step)
    else:
        levels = np.array([float(x) for x in palette["levels"].split("/")])
    if palette.get("flux_w_m2"):
        levels = levels * 3600.0 * window_hours / 1e6
    return levels


def _stepped_heatmap(
    grid: pd.DataFrame, levels: np.ndarray, palette: dict, colorbar: dict
) -> tuple[go.Heatmap, float | None]:
    """Heatmap shaded in discrete bands, with every band equally wide on the bar.

    Default-style legends give each band the same width however uneven the levels
    (0.5/2/4/10/25… mm), so the field is drawn in band-index space and the
    real values ride along in `customdata` for hover. Below the lowest level
    is left unshaded.

    Returns
    -------
    tuple[go.Heatmap, float or None]
        The trace, and the top level when the top band is bounded — values
        above it are clipped into that band — or None for an open top band.
    """
    colors = palette["colors"].split("/")
    values = grid.to_numpy()
    n_levels = len(levels)
    open_top = len(colors) == n_levels
    n_bands = len(colors)

    index = np.interp(values, levels, np.arange(n_levels, dtype=float))
    if open_top:
        index = np.where(values > levels[-1], n_levels - 0.5, index)
    index = np.where(values < levels[0], np.nan, index)

    scale = []
    for k, color in enumerate(colors):
        scale += [[k / n_bands, color], [(k + 1) / n_bands, color]]

    if palette.get("flux_w_m2"):
        # Converted edges land on values like 0.288; label round numbers
        # instead, placed at their position within the bands.
        tick_values = _round_ticks(levels[0], levels[-1])
        tick_pos = np.interp(tick_values, levels, np.arange(n_levels, dtype=float))
    else:
        tick_pos = np.arange(0, n_levels, max(1, int(np.ceil(n_levels / 14))))
        tick_values = levels[tick_pos]
    trace = go.Heatmap(
        z=index,
        customdata=values,
        colorscale=scale,
        zmin=0.0,
        zmax=float(n_bands),
        colorbar={
            **colorbar,
            "tickvals": tick_pos,
            "ticktext": [f"{v:.3g}" for v in tick_values],
        },
    )
    return trace, None if open_top else float(levels[-1])


def _round_ticks(low: float, high: float, target: int = 8) -> np.ndarray:
    """About `target` ticks on a 1/2/2.5/5 × 10^k step between two values."""
    raw = (high - low) / target
    magnitude = 10.0 ** np.floor(np.log10(raw))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw)
    return np.arange(np.ceil(low / step) * step, high + step / 2, step)


def _isoline_trace(grid: pd.DataFrame, palette: dict, hover: str) -> go.Contour:
    """Labelled isolines at a fixed interval, unshaded — the default MSLP look."""
    values = grid.to_numpy()
    step = palette["isolines"]
    return go.Contour(
        z=values,
        contours=dict(
            start=np.floor(np.nanmin(values) / step) * step,
            end=np.ceil(np.nanmax(values) / step) * step,
            size=step,
            coloring="none",
            showlabels=True,
            labelfont=dict(size=10, color=palette["color"]),
        ),
        line=dict(color=palette["color"], width=2),
        showscale=False,
        hovertemplate=hover + "<extra></extra>",
    )


def make_field_map_fig(
    df: pd.DataFrame,
    var_meta: dict,
    dataset_meta: dict,
    meta: dict,
    request_meta: dict,
    overlay: pd.DataFrame | None = None,
    outline: np.ndarray | None = None,
) -> go.Figure:
    """Map one retrieved CDS/ADS field.

    Cells are drawn at their own spacing with no smoothing or interpolation,
    and the y-axis is locked to the x-axis by `1/cos(lat)` so a gridbox is
    square in ground distance rather than in degrees — the same treatment the
    regional map gets, for the same reason.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy field frame from `utils.cds_api.load_field`, with columns
        `latitude`, `longitude`, `value`.
    var_meta : dict
        Entry from `utils.settings.CDS_VARIABLES[dataset]`.
    dataset_meta : dict
        Entry from `utils.settings.CDS_DATASETS`.
    meta : dict
        Metadata payload from the loader — valid time, extent, cell counts,
        native and drawn resolution.
    request_meta : dict
        What was asked for, as the page assembled it: `store_label`, and
        whichever of `date`, `hour`, `init_time`, `leadtime_hour`,
        `cams_model`, `month_label` apply. Anything absent is simply not
        stated.

    Notes
    -----
    Every element of provenance the field is uninterpretable without goes in
    the subtitle: the dataset and the store it came from, the validity time,
    the init time and lead for a forecast, the model for the CAMS regional
    ensemble, the reference period for an anomaly, and the grid resolution
    actually drawn.
    """
    is_anomaly = bool(meta.get("is_anomaly"))
    # The loader resolves the unit once the accumulation window or monthly
    # re-expression has been applied, so its answer wins over the variable's
    # static one. Falling back to `anomaly_unit` here would relabel a month
    # total as a daily mean.
    unit = meta.get("unit") or (
        var_meta.get("anomaly_unit", var_meta["unit"])
        if is_anomaly
        else var_meta["unit"]
    )
    unit_suffix = f" {unit}" if unit else ""
    label = var_meta["label"]

    grid = df.pivot_table(index="latitude", columns="longitude", values="value")
    hover = "%{y:.2f}°N, %{x:.2f}°E<br>"
    colorbar = dict(
        title=var_meta.get("colorbar_title") or unit or label,
        thickness=14,
        outlinewidth=0,
    )
    palette = MAP_PALETTES[var_meta["palette"]]

    if is_anomaly:
        # An anomaly is a departure either side of zero whatever the field, so
        # the sequential scale a magnitude uses would hide the sign. Limits are
        # made symmetric about zero as well — otherwise the neutral colour sits
        # off-centre and a mild warm anomaly reads the same as a strong one.
        extreme = float(np.nanmax(np.abs(grid.to_numpy())))
        field = go.Heatmap(
            z=grid.to_numpy(),
            colorscale=var_meta.get("anomaly_colorscale", "RdBu_r"),
            zmid=0.0,
            zmin=-extreme,
            zmax=extreme,
            colorbar=colorbar,
            hovertemplate=hover + "%{z:.3g}" + unit_suffix + "<extra></extra>",
        )
    elif "isolines" in palette:
        field = _isoline_trace(grid, palette, hover + "%{z:.1f}" + unit_suffix)
    else:
        levels = palette_levels(palette, window_hours(meta))
        field, top = _stepped_heatmap(grid, levels, palette, colorbar)
        field.hovertemplate = (
            hover + "%{customdata:.3g}" + unit_suffix + "<extra></extra>"
        )
        # Fixed levels keep successive maps comparable; the cost is that a
        # bounded top band swallows everything above it, so an exceedance is
        # invisible unless the subtitle says the scale is capped.
        peak = float(np.nanmax(grid.to_numpy()))
        if top is not None and peak > top:
            meta = {**meta, "scale_capped": (top, peak)}

    field.update(
        x=grid.columns.to_numpy(),
        y=grid.index.to_numpy(),
        name=label,
        showlegend=False,
    )
    if isinstance(field, go.Heatmap):
        field.zsmooth = False
    fig = go.Figure(field)

    # The absolute field contoured over its own anomaly — the 500 hPa chart:
    # shading says how unusual, contours say where the flow is.
    if overlay is not None:
        over = overlay.pivot_table(index="latitude", columns="longitude", values="value")
        step = MAP_PALETTES[var_meta["palette"]].get("isolines", 5.0)
        fig.add_trace(
            go.Contour(
                x=over.columns.to_numpy(),
                y=over.index.to_numpy(),
                z=over.to_numpy(),
                contours=dict(size=step, coloring="none", showlabels=True),
                line=dict(color="#111111", width=1.2),
                showscale=False,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    extent = meta["extent"]
    for trace in _basemap_traces(extent):
        fig.add_trace(trace)
    if outline is not None:
        fig.add_trace(_outline_trace(outline))

    # Regulatory thresholds, drawn as contours over the field rather than left
    # to the colour scale — an exceedance is a yes/no question and reading it
    # off a continuous ramp is guesswork.
    if not is_anomaly:
        for level, threshold_label in var_meta.get("thresholds", ()):
            fig.add_trace(
                go.Contour(
                    x=grid.columns.to_numpy(),
                    y=grid.index.to_numpy(),
                    z=grid.to_numpy(),
                    contours=dict(
                        start=level,
                        end=level,
                        size=1,
                        coloring="none",
                        showlabels=False,
                    ),
                    line=dict(color="#111111", width=1.2, dash="dot"),
                    showscale=False,
                    hoverinfo="skip",
                    showlegend=True,
                    name=f"{threshold_label} ({level:g}{unit_suffix})",
                )
            )

    lat_min, lat_max, _, _ = extent
    mid_lat = (lat_min + lat_max) / 2.0
    aspect = 1.0 / max(np.cos(np.deg2rad(mid_lat)), 0.05)

    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        title=dict(
            text=f"<b>{label}{' anomaly' if is_anomaly else ''}</b><br>"
            f"<sup>{map_subtitle(dataset_meta, meta, request_meta)}</sup>",
            x=0.01,
        ),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
        margin={"t": 96, "r": 20, "l": 60, "b": 60},
        height=680,
    )
    fig.update_xaxes(range=[extent[2], extent[3]], constrain="domain")
    fig.update_yaxes(
        range=[lat_min, lat_max],
        scaleanchor="x",
        scaleratio=aspect,
        constrain="domain",
    )
    return fig


def map_subtitle(dataset_meta: dict, meta: dict, request_meta: dict) -> str:
    """Two-line provenance string for a field map.

    Ordered by what a reader needs first: what is being shown and when it is
    valid, then where it came from and how it was produced.
    """
    parts: list[str] = []

    # The file's own timestamp wins; the hint is what the page asked for, and
    # covers the datasets whose time axis carries lead hours rather than dates.
    valid = meta.get("valid_time") or request_meta.get("valid_time_hint")
    if request_meta.get("month_label"):
        parts.append(request_meta["month_label"])
    elif valid is not None and not pd.isna(valid):
        parts.append(f"valid {pd.Timestamp(valid):%Y-%m-%d %H:%M} UTC")

    if request_meta.get("init_time"):
        lead = request_meta.get("leadtime_hour")
        lead_text = f" · T+{int(lead)} h" if lead is not None else ""
        parts.append(f"init {request_meta['init_time']} UTC{lead_text}")
    elif request_meta.get("leadtime_hour") is not None:
        parts.append(f"T+{int(request_meta['leadtime_hour'])} h from 00 UTC")

    if meta.get("is_anomaly"):
        first, last, count = meta["reference_years"]
        parts.append(
            f"anomaly vs {first}–{last} ({count} years, same "
            f"{'season' if meta.get('season') else 'calendar month'})"
        )

    if request_meta.get("cams_model"):
        parts.append(request_meta["cams_model"])

    if request_meta.get("region_label") and request_meta.get("region") != "global":
        parts.append(request_meta["region_label"])

    # An accumulated field is not valid *at* an instant, it is a total over a
    # window. Printing only "valid 12:00 UTC" invites reading a precipitation
    # map as a rate at noon.
    if request_meta.get("accumulation"):
        parts.append(request_meta["accumulation"])

    if meta.get("extra_dim"):
        parts.append(f"response carried {meta['extra_dim']}")

    if meta.get("scale_capped"):
        cap, peak = meta["scale_capped"]
        parts.append(f"colour scale capped at {cap:g}; field peaks at {peak:.3g}")

    source = (
        f"{dataset_meta['label']} · {request_meta.get('store_label', '')} · "
        f"{_map_resolution_label(meta)} · {meta['n_lat']}×{meta['n_lon']} cells "
        f"({_extent_label(meta['extent'])})"
        + (" · cut from a cached larger retrieval" if meta.get("cut_from_cache") else "")
    )
    return f"{' · '.join(parts)}<br>{source}"


def _outline_trace(vertices: np.ndarray) -> go.Scatter:
    """A region's polygon, closed, as a thin line over the map."""
    closed = np.vstack([vertices, vertices[:1]])
    return go.Scatter(
        x=closed[:, 0],
        y=closed[:, 1],
        mode="lines",
        line=dict(color="#6a1b9a", width=2),
        hoverinfo="skip",
        showlegend=False,
    )


def make_index_timeseries_fig(
    frame: pd.DataFrame, index: dict, title: str, subtitle: str
) -> go.Figure:
    """Monthly climate index: signed bars, plus the long running mean if any.

    Mirrors `static_figures.render_index_timeseries`: short smoothing (ONI)
    is the index and is what the bars show; long smoothing (AMO) is drawn as
    a line over the monthly bars.
    """
    window = index.get("smooth")
    bars = frame["value"] if window and window <= 12 else frame["monthly"]
    unit = index["unit"]
    fig = go.Figure(
        go.Bar(
            x=bars.index,
            y=bars.to_numpy(),
            marker_color=np.where(bars.to_numpy() >= 0, "#d6604d", "#4393c3"),
            marker_line_width=0,
            name=index["label"],
            hovertemplate="%{x|%b %Y}: %{y:+.2f} " + unit + "<extra></extra>",
        )
    )
    if window and window > 12:
        fig.add_trace(
            go.Scatter(
                x=frame.index,
                y=frame["value"],
                mode="lines",
                line=dict(color="#111111", width=2),
                name=f"{window}-month running mean",
                hovertemplate="%{x|%b %Y}: %{y:+.2f} " + unit + "<extra></extra>",
            )
        )
    for threshold, label in zip(
        index.get("thresholds", ()), index.get("threshold_labels", ()), strict=False
    ):
        fig.add_hline(
            y=threshold,
            line=dict(color="#555555", dash="dash", width=1),
            annotation_text=label,
            annotation_position="top left" if threshold > 0 else "bottom left",
        )
    fig.update_layout(
        template="plotly_white",
        title=dict(text=f"<b>{title}</b><br><sup>{subtitle}</sup>", x=0.01),
        yaxis_title=unit,
        bargap=0,
        hovermode="x",
        showlegend=bool(window and window > 12),
        legend=dict(orientation="h", yanchor="top", y=-0.12, x=0),
        margin={"t": 80, "r": 20, "l": 60, "b": 50},
        height=420,
    )
    return fig


def make_correlation_map_fig(
    lon: np.ndarray,
    lat: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
    extent: tuple[float, float, float, float],
    title: str,
    subtitle: str,
) -> go.Figure:
    """Pearson r on native cells, ±1 fixed, with p < 0.05 cells marked by dots."""
    fig = go.Figure(
        go.Heatmap(
            x=lon,
            y=lat,
            z=r,
            customdata=p,
            colorscale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            zmid=0.0,
            zsmooth=False,
            colorbar=dict(title="r", thickness=14, outlinewidth=0),
            hovertemplate=(
                "%{y:.1f}°N, %{x:.1f}°E<br>r = %{z:+.2f}<br>p = %{customdata:.3f}"
                "<extra></extra>"
            ),
        )
    )
    # One cell in four (every second row *and* column): a dot pattern that
    # reads as stippling. Thinning rows alone leaves solid rows of markers
    # that read as hatching.
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    significant = np.isfinite(p) & (p < 0.05)
    keep = np.zeros_like(significant)
    keep[::2, ::2] = True
    significant &= keep
    fig.add_trace(
        go.Scattergl(
            x=lon_grid[significant],
            y=lat_grid[significant],
            mode="markers",
            marker=dict(size=2, color="#111111"),
            hoverinfo="skip",
            name="p < 0.05",
        )
    )
    for trace in _basemap_traces(extent):
        fig.add_trace(trace)

    lat_min, lat_max, lon_min, lon_max = extent
    aspect = 1.0 / max(np.cos(np.deg2rad((lat_min + lat_max) / 2.0)), 0.05)
    fig.update_layout(
        template="plotly_white",
        hovermode="closest",
        title=dict(text=f"<b>{title}</b><br><sup>{subtitle}</sup>", x=0.01),
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        showlegend=False,
        margin={"t": 96, "r": 20, "l": 60, "b": 60},
        height=620,
    )
    fig.update_xaxes(range=[lon_min, lon_max], constrain="domain")
    fig.update_yaxes(
        range=[lat_min, lat_max], scaleanchor="x", scaleratio=aspect, constrain="domain"
    )
    return fig


_SCORE_LABELS: dict[str, tuple[str, bool]] = {
    # score key -> (axis label, carries the variable's unit)
    "crps": ("CRPS (fair)", True),
    "rmse": ("RMSE of the ensemble mean", True),
    "mae": ("MAE of the ensemble mean", True),
    "bias": ("Bias of the ensemble mean (forecast − observed)", True),
    "sd_error": ("SD of error of the ensemble mean", True),
    "spread_skill": ("Spread / RMSE (1 = well dispersed)", False),
}


def _score_label(score: str, unit: str) -> tuple[str, bool]:
    """(axis label, carries the unit); Brier keys are `brier_<threshold>`."""
    if score.startswith("brier_"):
        return f"Brier score (≥ {score.removeprefix('brier_')} {unit})", False
    return _SCORE_LABELS[score]


# Matches the Evaluation page's graph height.
_EVAL_HEIGHT = 560


def _eval_context(context: dict) -> str:
    """The evaluation context line: station and the runs scored."""
    inits = sorted(context["inits"])
    return (
        f"{context['station']} · {len(inits)} runs {inits[0]:%Y-%m-%d %H} → "
        f"{inits[-1]:%Y-%m-%d %H} UTC"
    )


def _eval_provenance(spec: dict, context: dict) -> list[str]:
    """Footer lines: each model's grid cell, then the observations used."""
    points = context.get("grid_points") or {None: context["grid_point"]}
    lines = [
        (f"{GRID_STORES[s]['label']}: " if s else "")
        + f"nearest {GRID_STORES[s]['res_deg'] if s else 0.25:g}° cell {format_point(lat, lon)}, "
        "no height correction"
        for s, (lat, lon) in points.items()
    ]
    reference = context.get("reference", "stations")
    if reference != "stations":
        ref = EVAL_REFERENCES[reference]
        where = "the station's 0.1° cell" if reference == "imerg" else "each model's cell"
        window = f", {spec['accum_hours']} h totals of every half-hour" if reference == "imerg" else ""
        lines.append(
            f"Verified against {ref['label']} ({ref['description']}) at {where}{window}"
            + (" — favours IFS and AIFS, which start from the IFS analysis" if reference == "control_t0" else "")
        )
        return lines
    source = context.get("precip_source")
    if spec["accum_hours"] and source and source["source"] == "reported":
        end = source["end_hour"]
        window = f" · {spec['accum_hours']} h totals reported at {end:02d} UTC ({end:02d}–{end:02d} UTC)"
    elif spec["accum_hours"]:
        window = f" · {spec['accum_hours']} h totals, complete hours only"
    else:
        window = ""
    last = (
        f", last report {context['obs_end']:%Y-%m-%d %H:%M} UTC"
        if context.get("obs_end") is not None
        else ""
    )
    reported = spec["accum_hours"] and source and source["source"] == "reported"
    kind = "GHCNh station totals" if reported else "GHCNh hourly, ±15 min"
    lines.append(f"Observations: {kind}, QC-passed{last}{window}")
    return lines


def make_scores_fig(scores: dict, score: str, spec: dict, context: dict) -> go.Figure:
    """One score against lead time, one line per model; marker size follows `n`.

    `scores` is `{store: verification.scores_by_lead frame}`. Each model is
    scored on its own members; the lines are compared, never combined.
    """
    axis, with_unit = _score_label(score, context["unit"])
    unit = context["unit"] if with_unit else ""
    fig = go.Figure()
    n_max = max((int(df["n"].max()) for df in scores.values() if not df.empty), default=1)
    for store, df in scores.items():
        if df.empty:
            continue
        if score == "spread_skill" and GRID_STORES[store].get("kind") == "deterministic":
            continue  # one member has no spread
        color = _store_color(store)
        fig.add_trace(
            go.Scatter(
                x=df["lead_hours"],
                y=df[score],
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"color": color, "size": 4 + 8 * df["n"] / max(n_max, 1)},
                customdata=df["n"],
                name=GRID_STORES[store]["label"],
                hovertemplate=(
                    f"{GRID_STORES[store]['label']}<br>T+%{{x:.0f}} h<br>"
                    f"%{{y:.2f}} {unit}<br>n=%{{customdata}} runs<extra></extra>"
                ),
            )
        )
    if score == "bias":
        fig.add_hline(y=0, line={"color": "#888888", "width": 1})
    if score == "spread_skill":
        fig.add_hline(y=1, line={"color": "#888888", "width": 1, "dash": "dash"})
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        xaxis_title="Lead time (h)",
        yaxis_title=f"{axis}{f' ({unit})' if unit else ''}",
    )
    _frame_timeseries(
        fig, f"{spec['label']} — {axis}", _eval_context(context),
        _eval_provenance(spec, context), height=_EVAL_HEIGHT, x_kind="linear",
    )
    fig.update_layout(hovermode="closest")
    return fig


def make_eval_runs_fig(
    matched: dict, obs: pd.DataFrame, spec: dict, context: dict, n_runs: int = 3
) -> go.Figure:
    """The newest verified runs' ensemble P10–P90 and mean per model, under the observations.

    "Verified" means at least one lead has an observation: the newest runs
    usually verify after the station's last report (GHCNh lags by days).
    """
    unit = context["unit"]
    fig = go.Figure()
    shown = set()
    for store, df in matched.items():
        color = _store_color(store)
        name = GRID_STORES[store]["label"]
        verified = df.dropna(subset=["obs"])["init_time"].unique() if "obs" in df else df["init_time"].unique()
        for init in sorted(verified)[-n_runs:]:
            run = df[df["init_time"] == init]
            q = run.groupby("valid_time")["value"].quantile([0.1, 0.9]).unstack()
            mean = run.groupby("valid_time")["value"].mean()
            first = store not in shown
            fig.add_trace(go.Scatter(x=q.index, y=q[0.9], mode="lines", line={"width": 0},
                                     legendgroup=f"{store}-band", showlegend=False, hoverinfo="skip",
                                     name=f"{name} P90"))
            fig.add_trace(go.Scatter(x=q.index, y=q[0.1], mode="lines", line={"width": 0},
                                     fill="tonexty", fillcolor=_hex_to_rgba(color, 0.15),
                                     legendgroup=f"{store}-band", showlegend=first,
                                     name=f"{name} P10–P90", hoverinfo="skip"))
            fig.add_trace(go.Scatter(
                x=mean.index, y=mean.to_numpy(), mode="lines", line={"color": color, "width": 2},
                legendgroup=f"{store}-mean", showlegend=first, name=f"{name} ensemble mean",
                hovertemplate=(f"{name} run {pd.Timestamp(init):%Y-%m-%d %H} UTC<br>"
                               f"%{{x|%a %d %b %H:%M}}<br>%{{y:.1f}} {unit}<extra></extra>"),
            ))
            shown.add(store)
    if not obs.empty:
        lo = min(df["valid_time"].min() for df in matched.values()) if matched else None
        o = obs.dropna(subset=["value"]).copy()
        # Matched forecast times are naive UTC (see `verification.match_obs`).
        o["valid_time"] = pd.to_datetime(o["valid_time"], utc=True).dt.tz_localize(None)
        if lo is not None:
            o = o[o["valid_time"] >= lo]
        reference = context.get("reference", "stations")
        ref_name = "Observed" if reference == "stations" else EVAL_REFERENCES[reference]["label"]
        fig.add_trace(go.Scatter(
            x=o["valid_time"], y=o["value"], mode="lines" if reference == "stations" else "lines+markers",
            line={"color": OBSERVED_COLOR, "width": 2}, name=ref_name,
            hovertemplate=f"{ref_name}<br>%{{x|%a %d %b %H:%M}}<br>%{{y:.1f}} {unit}<extra></extra>",
        ))
    fig.update_layout(
        **_LAYOUT_DEFAULTS,
        xaxis_title="Valid time (UTC)",
        yaxis_title=f"{spec['label']} ({unit})",
    )
    _frame_timeseries(
        fig, f"{spec['label']} — newest verified runs vs observed", _eval_context(context),
        _eval_provenance(spec, context), height=_EVAL_HEIGHT,
    )
    fig.update_layout(hovermode="closest")
    return fig


# --------------------------------------------------------------------------
# Station scorecard
# --------------------------------------------------------------------------
# score -> (label, carries the variable's unit)
_SCORECARD_SCORES: dict[str, tuple[str, bool]] = {
    "rmse": ("RMSE", True),
    "mae": ("MAE", True),
    "bias": ("Bias (forecast − observed)", True),
    "crps": ("Fair CRPS", True),
    "brier": ("Fair Brier score", False),
    "rmse_bc": ("RMSE, bias-corrected", True),
    "mae_bc": ("MAE, bias-corrected", True),
    "spread_skill": ("Spread / RMSE (1 = well dispersed)", False),
    "ets": ("ETS", False),
    "hss": ("HSS", False),
    "frequency_bias": ("Frequency bias (1 = unbiased)", False),
    "pod": ("POD (hit rate)", False),
    "far": ("FAR (false alarm ratio)", False),
}
# Cells with too few pairs: a cross, neutral enough to read on any cell colour.
_NEUTRAL_MARK = "rgba(128, 128, 128, 0.95)"
_SCORECARD_DETERMINISTIC_SKIP = ("spread_skill", "brier")
_SCORECARD_HEIGHT = 420
# R25: the relative colour range is fixed; the true percent stays in the hover.
_RELATIVE_CAP_PCT = 50.0
_FORECAST_SOURCE = "forecasts: dynamical.org archive"
_BIAS_CORRECTED_NOTE = (
    "bias-corrected: minus each station×model mean error over the window, "
    "all leads, in-sample"
)
_SPREAD_SKILL_NOTE = (
    "spread/skill = √((M+1)/M · mean variance) / RMSE of the ensemble mean, ideal 1"
)
_FEW_RUNS_NOTE = (
    "long leads rest on fewer runs: the window ends at the last observation"
)


def _score_text(score: str, variable: str) -> tuple[str, str]:
    """(label, unit text) of a scorecard score for one variable ("" if unitless)."""
    label, with_unit = _SCORECARD_SCORES[score]
    return label, (SCORECARD_VARIABLES[variable]["unit"] if with_unit else "")


def _is_deterministic(store: str) -> bool:
    return GRID_STORES[store].get("kind") == "deterministic"


def _scorecard_frame_text(
    variable: str, score: str, stores: list[str], context: dict, relative: bool = False
) -> tuple[str, list[str]]:
    """Context line and footer lines shared by every scorecard figure.

    Parameters
    ----------
    variable, score : str
        Keys of `SCORECARD_VARIABLES` and `SCORE_DIRECTION`.
    stores : list of str
        Models drawn; the CRPS = MAE note appears only when a deterministic one is.
    context : dict
        `n_stations`, `period_days`, `init_hour`, `end`, `baseline`; optional
        `min_n` (default 50), `tolerance` (overrides the matching text) and
        `absent_note` (a last footer line naming models left out of the card).
    relative : bool
        The colour is a percent change against `context["baseline"]`.
    """
    spec = SCORECARD_VARIABLES[variable]
    context_line = (
        f"{context['n_stations']} stations · last {context['period_days']} days to "
        f"{pd.Timestamp(context['end']):%Y-%m-%d} · {context['init_hour']:02d} UTC runs"
    )
    if spec["reference"] == "control_t0":
        match = (
            "Verified against a pseudo-analysis: IFS control T+0, favours IFS and "
            "AIFS · 00 UTC valid times, at each station's grid cell"
        )
        sampling = _FEW_RUNS_NOTE[0].upper() + _FEW_RUNS_NOTE[1:]
    else:
        if spec["window_hours"] and not spec.get("hourly_fallback", True):
            match = context.get("tolerance") or (
                f"Observations: GHCNh station-reported {spec['window_hours']} h "
                f"totals ending {spec['window_end_hour']:02d} UTC — never hourly sums"
            )
        elif spec["window_hours"]:
            match = context.get("tolerance") or (
                f"Observations: GHCNh {spec['window_hours']} h totals ending "
                "00/06/12/18 UTC — reported where available, else complete "
                "hourly sums"
            )
        else:
            tol = spec["tolerance"].replace("min", " min")
            match = context.get("tolerance") or (
                f"Observations: GHCNh, matched within ±{tol}"
            )
        hours = "/".join(
            f"{h:02d}"
            for h in range(
                context["init_hour"] % SCORECARD_LEAD_STEP_HOURS,
                24,
                SCORECARD_LEAD_STEP_HOURS,
            )
        )
        sampling = f"Every model scored at {hours} UTC valid times · {_FEW_RUNS_NOTE}"
    lines = [
        f"{match} · {_FORECAST_SOURCE}",
        (
            "Model values: nearest grid cell, no height correction; coastal cells "
            "may be partly sea · common inits: every model scored on the same runs"
        ),
        sampling,
    ]
    notes = []
    if score == "crps":
        notes.append("fair CRPS (ensembles scored on their own members)")
        if any(_is_deterministic(s) for s in stores):
            notes.append("CRPS of a deterministic model equals its MAE")
    elif score == "brier":
        notes.append(f"fair Brier score at ≥ {spec['threshold']:g} {spec['unit']}")
    elif score in ("rmse_bc", "mae_bc"):
        notes.append(_BIAS_CORRECTED_NOTE)
    elif score == "spread_skill":
        notes.append(_SPREAD_SKILL_NOTE)
    if relative:
        base = GRID_STORES[context["baseline"]]["label"]
        notes.append(
            f"colour: % change vs {base}, positive = better, capped at "
            f"±{_RELATIVE_CAP_PCT:g} %"
        )
    if notes:
        lines.append(" · ".join(notes)[0].upper() + " · ".join(notes)[1:])
    if context.get("absent_note"):
        lines.append(context["absent_note"])
    return context_line, lines


def _score_scale(score: str, values: np.ndarray) -> dict:
    """Colour-scale kwargs for absolute values: the better end keeps one colour."""
    direction = SCORE_DIRECTION[score]
    finite = values[np.isfinite(values)]
    if not isinstance(direction, str):
        half = float(np.abs(finite - direction).max()) if finite.size else 1.0
        half = half or 1.0
        return {
            "colorscale": "RdBu_r",
            "zmid": direction,
            "zmin": direction - half,
            "zmax": direction + half,
        }
    return {"colorscale": "YlGnBu_r" if direction == "lower" else "YlGnBu"}


def make_scorecard_fig(
    table: pd.DataFrame, variable: str, score: str, relative: bool, context: dict
) -> go.Figure:
    """Models × lead days heatmap of one score.

    Parameters
    ----------
    table : pandas.DataFrame
        `scorecard.aggregate(..., by=("store", "lead_day"))`; rows of other
        variables are ignored.
    variable, score : str
        `SCORECARD_VARIABLES` key and a `SCORE_DIRECTION` key.
    relative : bool
        True: colour is the percent change against `context["baseline"]`
        (`scorecard.relative_to`), diverging, zero-centred and symmetric, blue =
        better. False: the absolute score, the better end always the same colour.
    context : dict
        See `_scorecard_frame_text`. Optional `relative_rows`: stored
        `lead_relative` rows (same keys as `table`) used for the percent colour
        instead of recomputing it from `table`.

    Notes
    -----
    A missing cell (a model without that horizon) is blank, never zero. Cells
    with `low_n` carry a cross and "(n<min)" in the hover.
    """
    # Lazy: scorecard pulls in scores/xarray/earthmover, which figures must not.
    from utils.scorecard import relative_to

    table = table[table["variable"] == variable]
    if table.empty:
        return make_empty_fig("No data available")
    stores = [s for s in SCORECARD_STORES if s in set(table["store"])]
    days = list(range(1, 16))
    label, unit = _score_text(score, variable)
    min_n = context.get("min_n", 50)

    def grid(frame: pd.DataFrame, column: str) -> np.ndarray:
        wide = frame.pivot_table(
            index="store", columns="lead_day", values=column, aggfunc="first"
        )
        return wide.reindex(index=stores, columns=days).to_numpy(dtype=float)

    stored = context.get("relative_rows")
    if stored is not None:
        stored = stored[stored["variable"] == variable]
    pct_table = (
        stored if stored is not None else relative_to(table, context["baseline"])
    )
    absolute, pct = grid(table, score), grid(pct_table, score)
    z = pct if relative else absolute
    n = grid(table, "n")
    n_inits = grid(table, "n_inits")
    low = grid(table.assign(low_n=table["low_n"].astype(float)), "low_n") == 1.0

    def fmt(a: np.ndarray, spec: str, suffix: str = "") -> np.ndarray:
        return np.where(
            np.isfinite(a),
            np.vectorize(lambda v: f"{v:{spec}}{suffix}")(np.nan_to_num(a)),
            "–",
        )

    few_runs = np.isfinite(n_inits) & (n_inits < SCORECARD_MIN_INITS)
    flag = np.where(
        low & (n < min_n),
        f" (n<{min_n})",
        np.where(low & few_runs, f" (<{SCORECARD_MIN_INITS} runs)", ""),
    )
    custom = np.stack(
        [
            fmt(absolute, ".3g", f" {unit}" if unit else ""),
            fmt(pct, "+.1f", " %"),
            fmt(n, ".0f"),
            fmt(n_inits, ".0f"),
            flag,
        ],
        axis=-1,
    )
    labels = [GRID_STORES[s]["label"] for s in stores]
    base_label = GRID_STORES[context["baseline"]]["label"]
    if relative:
        labels = [
            f"{lab} (baseline)" if s == context["baseline"] else lab
            for s, lab in zip(stores, labels)
        ]
        # R25: fixed ±cap (target-valued scores near their ideal explode the
        # percent); the true value stays in the hover's customdata.
        cap = _RELATIVE_CAP_PCT
        z = np.clip(z, -cap, cap)
        scale = {"colorscale": "RdBu", "zmid": 0, "zmin": -cap, "zmax": cap}
        ticks = np.linspace(-cap, cap, 5)
        bar = {
            "title": {"text": f"% better than {base_label}", "side": "right"},
            "tickvals": list(ticks),
            "ticktext": [
                f"≤−{cap:g} %",
                *(f"{t:+g} %".replace("-", "−") for t in ticks[1:-1]),
                f"≥{cap:g} %",
            ],
        }
    else:
        scale = _score_scale(score, z)
        direction = SCORE_DIRECTION[score]
        hint = (
            "lower is better"
            if direction == "lower"
            else (
                "higher is better"
                if direction == "higher"
                else f"closer to {direction:g} is better"
            )
        )
        bar = {
            "title": {
                "text": f"{label}{f' ({unit})' if unit else ''} · {hint}",
                "side": "right",
            }
        }
    fig = go.Figure(
        go.Heatmap(
            x=days,
            y=labels,
            z=np.where(np.isfinite(z), z, np.nan),
            customdata=custom,
            hoverongaps=False,
            xgap=1,
            ygap=1,
            colorbar=bar,
            hovertemplate=(
                "%{y}<br>Lead day %{x}<br>"
                f"{label}: %{{customdata[0]}}<br>"
                f"vs {base_label}: %{{customdata[1]}}<br>"
                "n=%{customdata[2]} pairs, %{customdata[3]} inits%{customdata[4]}"
                "<extra></extra>"
            ),
            **scale,
        )
    )
    if low.any():
        rows, cols = np.nonzero(low & np.isfinite(z))
        fig.add_trace(
            go.Scatter(
                x=[days[c] for c in cols],
                y=[labels[r] for r in rows],
                mode="markers",
                marker={
                    "symbol": "x",
                    "size": 9,
                    "color": _NEUTRAL_MARK,
                    "line": {"width": 1},
                },
                name="n below minimum",
                hoverinfo="skip",
            )
        )
    # R24: each column names its fewest runs (minimum n_inits over the models).
    with np.errstate(all="ignore"):
        runs = np.where(
            np.isfinite(n_inits).any(axis=0),
            np.nanmin(np.where(np.isfinite(n_inits), n_inits, np.inf), axis=0),
            np.nan,
        )
    ticktext = [
        (
            f"Day {d}<br>{r:.0f} run{'s' if r != 1 else ''}"
            if np.isfinite(r)
            else f"Day {d}"
        )
        for d, r in zip(days, runs)
    ]
    fig.update_layout(
        **{
            k: v
            for k, v in _LAYOUT_DEFAULTS.items()
            if k not in ("hovermode", "legend", "margin")
        },
        hovermode="closest",
        xaxis={
            "title": "Lead day (day 1 = 0–24 h)",
            "tickvals": days,
            "ticktext": ticktext,
        },
        yaxis={"autorange": "reversed"},
    )
    head, foot = _scorecard_frame_text(variable, score, stores, context, relative)
    _frame_timeseries(
        fig,
        f"{SCORECARD_VARIABLES[variable]['label']} — {label}",
        head,
        foot,
        height=_SCORECARD_HEIGHT + 30 * len(stores),
        x_kind="date",  # two-line ticks: day and run count
    )
    return fig


def make_score_lead_fig(
    df: pd.DataFrame, variable: str, score: str, context: dict
) -> go.Figure:
    """One score against lead day, one line per model (never pooled).

    Deterministic models are skipped for spread/skill and Brier (one member). A
    dashed line marks the ideal value of spread/skill (1), frequency bias (1)
    and bias (0).
    """
    df = df[df["variable"] == variable]
    if df.empty:
        return make_empty_fig("No data available")
    label, unit = _score_text(score, variable)
    fig = go.Figure()
    stores = []
    n_max = max(float(df["n"].max()), 1.0) if len(df) else 1.0
    for store in SCORECARD_STORES:
        sub = df[df["store"] == store].sort_values("lead_day")
        if sub.empty or sub[score].notna().sum() == 0:
            continue
        if score in _SCORECARD_DETERMINISTIC_SKIP and _is_deterministic(store):
            continue
        stores.append(store)
        color = _store_color(store)
        name = GRID_STORES[store]["label"]
        fig.add_trace(
            go.Scatter(
                x=sub["lead_day"],
                y=sub[score],
                mode="lines+markers",
                line={"color": color, "width": 2},
                marker={"color": color, "size": 4 + 6 * sub["n"] / n_max},
                customdata=np.stack([sub["n"], sub["n_inits"]], axis=-1),
                name=name,
                hovertemplate=(
                    f"{name}<br>Lead day %{{x}}<br>%{{y:.3g}} {unit}<br>"
                    "n=%{customdata[0]} pairs, %{customdata[1]} inits<extra></extra>"
                ),
            )
        )
    ideal = SCORE_DIRECTION[score]
    if not isinstance(ideal, str):
        fig.add_hline(
            y=ideal, line={"color": _NEUTRAL_MARK, "width": 1, "dash": "dash"}
        )
    fig.update_layout(
        **{
            k: v
            for k, v in _LAYOUT_DEFAULTS.items()
            if k not in ("hovermode", "legend", "margin")
        },
        hovermode="closest",
        xaxis={"title": "Lead day (day 1 = 0–24 h)", "dtick": 1},
        yaxis_title=f"{label}{f' ({unit})' if unit else ''}",
    )
    head, foot = _scorecard_frame_text(variable, score, stores, context)
    _frame_timeseries(
        fig,
        f"{SCORECARD_VARIABLES[variable]['label']} — {label}",
        head,
        foot,
        height=_SCORECARD_HEIGHT + 60,
        x_kind="linear",
    )
    return fig


def make_station_score_map(
    df: pd.DataFrame, variable: str, score: str, context: dict
) -> go.Figure:
    """One dot per station coloured by one score, on the bundled coastlines.

    Parameters
    ----------
    df : pandas.DataFrame
        One model at one lead day: `aggregate(by=("store", "station"))` on rows
        filtered to that lead, joined with `lat`, `lon` and optionally `name`.
        The caller chooses the lead and the model; both go in `context["lead"]`
        (text, e.g. "day 3, ECMWF IFS ENS") and appear in the title.
    """
    df = df[df["variable"] == variable].dropna(subset=["lat", "lon"])
    if df.empty:
        return make_empty_fig("No data available")
    label, unit = _score_text(score, variable)
    if not np.isfinite(df[score].to_numpy(dtype=float)).any():
        stores = list(df["store"].unique()) if "store" in df else []
        if (
            score in _SCORECARD_DETERMINISTIC_SKIP
            and stores
            and all(_is_deterministic(s) for s in stores)
        ):
            models = ", ".join(GRID_STORES[s]["label"] for s in stores)
            return make_empty_fig(
                f"{label} needs an ensemble: {models} is deterministic"
            )
        return make_empty_fig("No data available")
    names = df["name"] if "name" in df else df["station"]
    custom = np.stack([df["station"], names, df["n"]], axis=-1)
    stores = list(df["store"].unique()) if "store" in df else []
    fig = go.Figure()
    extent = (
        (
            float(df["lat"].min()) - 0.5,
            float(df["lat"].max()) + 0.5,
            float(df["lon"].min()) - 0.5,
            float(df["lon"].max()) + 0.5,
        )
        if len(df)
        else (47.0, 55.5, 5.5, 15.5)
    )
    for trace in _basemap_traces(extent):
        fig.add_trace(trace)
    values = df[score].to_numpy(dtype=float)
    scale = _score_scale(score, values)
    fig.add_trace(
        go.Scatter(
            x=df["lon"],
            y=df["lat"],
            mode="markers",
            name="stations",
            marker={
                "size": 9,
                "color": values,
                "line": {"width": 0.5, "color": _NEUTRAL_MARK},
                "colorbar": {
                    "title": {
                        "text": f"{label}{f' ({unit})' if unit else ''}",
                        "side": "right",
                    }
                },
                "cmin": scale.get("zmin"),
                "cmax": scale.get("zmax"),
                "cmid": scale.get("zmid"),
                "colorscale": scale["colorscale"],
            },
            customdata=custom,
            hovertemplate=(
                "%{customdata[1]} (%{customdata[0]})<br>"
                f"{label}: %{{marker.color:.3g}} {unit}<br>"
                "n=%{customdata[2]} pairs<extra></extra>"
            ),
        )
    )
    lat_min, lat_max, lon_min, lon_max = extent
    aspect = 1.0 / max(np.cos(np.deg2rad((lat_min + lat_max) / 2.0)), 0.05)
    fig.update_layout(
        **{
            k: v
            for k, v in _LAYOUT_DEFAULTS.items()
            if k not in ("hovermode", "legend", "margin")
        },
        hovermode="closest",
        showlegend=False,
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
    )
    fig.update_xaxes(range=[lon_min, lon_max], constrain="domain")
    fig.update_yaxes(
        range=[lat_min, lat_max], scaleanchor="x", scaleratio=aspect, constrain="domain"
    )
    head, foot = _scorecard_frame_text(variable, score, stores, context)
    lead = f" · {context['lead']}" if context.get("lead") else ""
    _frame_timeseries(
        fig,
        f"{SCORECARD_VARIABLES[variable]['label']} — {label} by station{lead}",
        head,
        foot,
        height=620,
        x_kind="linear",
    )
    return fig
