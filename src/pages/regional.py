"""Layout + all callbacks for the regional area-averaged ensemble page.

This is the only page backed by the gridded Icechunk store. It answers a
question the point pages structurally cannot: how uncertain is the *regional
average* of a variable over an area, across ensemble members.

The area is chosen by dragging a rectangle on the map (see
`components/bbox_select.py`), which is the natural gesture for a quantity
defined over an extent — a point marker would misrepresent what is being
computed.

It is deliberately kept separate from the meteogram. The two draw different
quantities from different models — one gridbox interpolated to a point
(Open-Meteo) versus a weighted mean of many gridboxes (ECMWF IFS or AIFS ENS
from the dynamical.org archive) — and presenting them on one axis would invite the reader to treat
the difference as signal rather than as two different definitions.
"""

from __future__ import annotations

import dash
import dash_mantine_components as dmc
from dash import Input, Output, State, callback, ctx, dcc, html, no_update

from components.bbox_select import CUSTOM as _CUSTOM
from components.bbox_select import bbox_select
from components.model_select import model_options
from components.page_shell import page_shell
from components.run_select import LATEST, register_run_options, run_select
from components.shared_state import publish_variable, url_sync
from utils import status
from utils.climatology import compute_member_pct
from utils.earthmover_api import (
    GridDataError,
    buildable_periods,
    fetch_area_multi,
    resolve_run,
    run_in_calendar,
    store_horizon,
    store_variables,
    stores_calendar,
)
from utils.figures import CONFIG, make_empty_fig, make_regional_multi_fig
from utils.map_products import forecast_animation
from utils.precomputed import preset_animation
from utils.regions import REGIONS, grouped_options
from utils.settings import (
    MAX_MODELS,
    GRID_RESOLUTION_DEG,
    GRID_STORE_DEFAULT,
    GRID_STORES,
    GRID_VARIABLES,
    LEVEL_MODES,
    PRECOMPUTE_URL,
    REGION_MAX_CELLS,
    STATIC_PALETTES,
    STATIC_PROJECTIONS,
)
from utils.url_state import parse_bbox, query_choice, query_list

dash.register_page(__name__, path="/regional", title="Forecast for an area · preddyfast")

# Derived variables (10 m wind speed from u/v) exist for the scorecard's
# station reads only; the area readers never build them.
_VARIABLE_OPTIONS = [
    {"value": k, "label": v["label"]}
    for k, v in GRID_VARIABLES.items()
    if "derived_from" not in v
]

_DEFAULT_VARIABLE = "temperature_2m"
# Several models may be drawn together, always from the same init (the run
# picker offers only inits they all hold) and never pooled across models.
_HORIZON_CHOICES = (72, 120, 168, 240, 360, 480, 720, 840, 1104)
_MAX_MODELS = MAX_MODELS


def _stores(model) -> tuple[str, ...]:
    """The selected stores, known keys only, in selection order."""
    chosen = model if isinstance(model, list) else [model]
    return tuple(k for k in chosen if k in GRID_STORES)


def _area_key(bbox: dict | None, region: str | None) -> str | None:
    """A region key, or the drawn box as `box:` edges rounded to 2 dp."""
    if region and region != _CUSTOM:
        return region
    if not bbox:
        return None
    edges = (round(bbox[k], 2) for k in ("lat_min", "lat_max", "lon_min", "lon_max"))
    return "box:" + ",".join(f"{e}" for e in edges)


def _model_description(stores: tuple[str, ...]) -> str:
    """Each selected model's one-line description."""
    return " ".join(
        f"{GRID_STORES[k]['label']}: {GRID_STORES[k]['description']}" for k in stores
    )
_LATEST_LABEL = "Latest complete"
# Regions small enough to read interactively; see `REGION_MAX_CELLS`.
_REGION_KEYS = [_CUSTOM] + [
    key
    for key, region in REGIONS.items()
    if (region["bbox"][1] - region["bbox"][0]) / GRID_RESOLUTION_DEG
    * (region["bbox"][3] - region["bbox"][2]) / GRID_RESOLUTION_DEG
    <= REGION_MAX_CELLS
]
_REGION_OPTIONS = grouped_options(
    _REGION_KEYS, extra=[{"value": _CUSTOM, "label": "Custom box (draw on the map)"}]
)
_PROJECTION_OPTIONS = [{"value": k, "label": v} for k, v in STATIC_PROJECTIONS.items()]
_PALETTE_OPTIONS = [{"value": k, "label": v} for k, v in STATIC_PALETTES.items()]
_LEVEL_OPTIONS = [{"value": k, "label": v} for k, v in LEVEL_MODES.items()]
_EXTENT_OPTIONS = grouped_options(extra=[{"value": "data", "label": "Forecast area"}])

_HIDDEN = {"display": "none"}
_SHOWN = {"display": "block"}


def _accum_options(
    variable: str | None, stores: tuple[str, ...] = ()
) -> tuple[list[dict], str | None]:
    """Accumulation-period options for a variable, empty if it is not accumulated.

    The periods come from the variable, then only those every selected
    store's steps can build: a store stepping 24 h can total 24 h but not 6 h.
    """
    meta = GRID_VARIABLES.get(variable or "", {})
    if not meta.get("is_accumulated"):
        return [], None
    periods = list(meta["accum_periods"])
    if stores:
        try:
            periods = buildable_periods(stores, periods)
        except GridDataError:
            pass
    if not periods:
        return [], None
    default = meta["accum_default"] if meta["accum_default"] in periods else periods[-1]
    options = [{"value": str(h), "label": f"{h} h"} for h in periods]
    return options, str(default)


def _horizon_options(stores: tuple[str, ...]) -> list[dict]:
    """Animation horizons up to the furthest-reaching selected model."""
    try:
        reach = max((store_horizon(s) for s in stores), default=360.0)
    except GridDataError:
        reach = 360.0
    return [
        {"value": str(h), "label": f"T+{h} h ({h // 24} days)"}
        for h in _HORIZON_CHOICES
        if h <= reach
    ]


def _frame_slot(i: int, image_style: dict) -> html.Div:
    """One model's frame: its name, the image, and a download link."""
    return html.Div(
        [
            dmc.Text(id=f"rg-anim-name-{i}", size="sm", fw=500),
            html.Img(
                id=f"rg-anim-img-{i}",
                alt="Forecast map frame",
                style={**image_style, "display": "block"},
            ),
            html.A(
                dmc.Button("Download this frame (PNG)", variant="light", size="xs"),
                id=f"rg-anim-download-{i}",
                style={"display": "none"},
            ),
        ],
        id=f"rg-anim-slot-{i}",
        style={"display": "none"},
    )


def _overlaid(graph, overlay_id: str) -> dmc.Paper:
    """A graph with a loading overlay over it.

    DMC 2.x LoadingOverlay is a sibling overlay driven by `visible`, not a
    wrapper — it takes no children, and needs a positioned ancestor to cover.
    """
    return dmc.Paper(
        dmc.Box(
            [
                dmc.LoadingOverlay(
                    id=overlay_id,
                    visible=False,
                    overlayProps={"blur": 2},
                    loaderProps={"type": "dots"},
                ),
                graph,
            ],
            pos="relative",
        ),
        withBorder=True,
        radius="md",
        p="xs",
    )


def _series_panel() -> list:
    return [
        _overlaid(
            dcc.Graph(
                id="regional-graph",
                figure=make_empty_fig("Drag a box on the map"),
                config=CONFIG,
                style={"height": "620px"},
            ),
            "rg-loading-overlay",
        )
    ]


def _map_panel() -> list:
    """Map tab: statistic, map style, frames, then the static-frame player."""
    image_style = {"width": "100%", "maxWidth": "980px", "display": "none"}
    return [
        dmc.Group(
            [
                dmc.SegmentedControl(
                    id="rg-stat",
                    data=[
                        {"value": "mean", "label": "Ensemble mean"},
                        {"value": "percentile", "label": "Percentile"},
                        {"value": "range", "label": "Percentile range"},
                    ],
                    value="mean",
                    size="xs",
                ),
                html.Div(
                    dmc.NumberInput(
                        id="rg-pct",
                        label="Percentile",
                        value=90,
                        min=1,
                        max=99,
                        step=1,
                        size="xs",
                        w=110,
                    ),
                    id="rg-pct-wrap",
                    style=_HIDDEN,
                ),
                html.Div(
                    dmc.Stack(
                        [
                            dmc.Text("Range (P low – P high)", size="xs", fw=500),
                            dmc.RangeSlider(
                                id="rg-pct-range",
                                value=[10, 90],
                                min=0,
                                max=100,
                                step=5,
                                minRange=5,
                                w=220,
                                marks=[
                                    {"value": v, "label": f"{v}"}
                                    for v in (0, 25, 50, 75, 100)
                                ],
                            ),
                        ],
                        gap=4,
                        pb="md",
                    ),
                    id="rg-range-wrap",
                    style=_HIDDEN,
                ),
            ],
            align="flex-end",
            gap="md",
            mb="xs",
        ),
        dmc.SimpleGrid(
            [
                dmc.Select(
                    id="rg-projection",
                    label="Projection",
                    data=_PROJECTION_OPTIONS,
                    value="auto",
                    allowDeselect=False,
                    size="xs",
                ),
                dmc.Select(
                    id="rg-extent",
                    label="Extent",
                    data=_EXTENT_OPTIONS,
                    value="data",
                    searchable=True,
                    allowDeselect=False,
                    size="xs",
                ),
                dmc.Select(
                    id="rg-palette",
                    label="Palette",
                    data=_PALETTE_OPTIONS,
                    value="default",
                    allowDeselect=False,
                    size="xs",
                ),
                dmc.Select(
                    id="rg-levels",
                    label="Levels",
                    description="Auto fits an area cut from a larger field.",
                    data=_LEVEL_OPTIONS,
                    value="auto",
                    allowDeselect=False,
                    size="xs",
                ),
                dmc.Select(
                    id="rg-step",
                    label="Frame every",
                    data=[{"value": str(h), "label": f"{h} h"} for h in (6, 12, 24)],
                    value="24",
                    allowDeselect=False,
                    size="xs",
                ),
                dmc.Select(
                    id="rg-horizon",
                    label="Up to",
                    data=[
                        {"value": str(h), "label": f"T+{h} h ({h // 24} days)"}
                        for h in _HORIZON_CHOICES[:5]
                    ],
                    # Three days by default: frames render at ~1 s each, per model.
                    value="72",
                    allowDeselect=False,
                    size="xs",
                ),
            ],
            cols={"base": 2, "sm": 6},
            spacing="xs",
            mb="xs",
        ),
        dmc.Group(
            [
                dmc.Button("Build animation", id="rg-anim-build", size="xs"),
                dmc.Text(
                    "Each frame is a rendered map (~1 s); the first animation of a run "
                    "also reads every member. Finished animations are cached.",
                    size="xs",
                    c="dimmed",
                ),
            ],
            gap="md",
            mb="xs",
        ),
        dcc.Store(id="rg-anim-store"),
        dcc.Interval(id="rg-anim-tick", interval=900, disabled=True),
        _overlaid(
            dmc.Stack(
                [
                    dmc.Group(
                        [
                            dmc.Button(
                                "Play", id="rg-anim-play", size="xs", variant="light",
                                disabled=True,
                            ),
                            dmc.Slider(
                                id="rg-anim-slider",
                                min=0,
                                max=1,
                                value=0,
                                step=1,
                                disabled=True,
                                style={"flex": 1},
                                label=None,
                            ),
                            dmc.Text(id="rg-anim-label", size="sm", fw=500, w=230),
                        ],
                        gap="md",
                        wrap="nowrap",
                    ),
                    dmc.Text(
                        "Open this tab after choosing an area.",
                        id="rg-anim-caption",
                        size="sm",
                        c="dimmed",
                    ),
                    # One slot per model (up to _MAX_MODELS), side by side on
                    # wide screens, stacked on narrow ones; one slider drives all.
                    dmc.SimpleGrid(
                        [_frame_slot(i, image_style) for i in range(_MAX_MODELS)],
                        id="rg-anim-panels",
                        cols={"base": 1, "md": 1},
                        spacing="xs",
                    ),
                ],
                gap="xs",
                p="xs",
                mih=320,
            ),
            "rg-map-overlay",
        ),
    ]


def layout(**query_params):
    variable = query_choice(
        query_params,
        "variable",
        [o["value"] for o in _VARIABLE_OPTIONS],
        _DEFAULT_VARIABLE,
    )
    accum_options, accum_default = _accum_options(variable)
    accum = query_choice(
        query_params, "accum", [o["value"] for o in accum_options], accum_default
    )

    region = query_choice(query_params, "region", _REGION_KEYS, _CUSTOM)
    models = query_list(query_params, "model", GRID_STORES, [GRID_STORE_DEFAULT])[
        :_MAX_MODELS
    ]
    run = query_params.get("run") or LATEST

    return page_shell(
        title="Forecast for an area",
        description=(
            "Ensemble spread of an area average — a standard region or a box you "
            "draw. Space is averaged per member first, so every line is a "
            "self-consistent realisation of the area mean."
        ),
        sidebar=[
            html.Div(id="rg-current-url", style={"display": "none"}),
            "Where",
            bbox_select(parse_bbox(query_params), region, _REGION_OPTIONS),
            dmc.Text(
                "Standard regions are averaged over their own shape; large ones "
                "take up to a minute on first read.",
                size="xs",
                c="dimmed",
            ),
            "Models",
            dmc.MultiSelect(
                id="rg-model-select",
                label="Models",
                description=_model_description(tuple(models)),
                data=model_options(),
                value=models,
                maxValues=_MAX_MODELS,
            ),
            "When",
            run_select(
                "rg-run-select",
                _LATEST_LABEL,
                "Any archived run; dates and hours without one are disabled.",
                run,
            ),
            "What",
            dmc.Select(
                id="rg-variable-select",
                label="Variable",
                data=_VARIABLE_OPTIONS,
                value=variable,
            ),
            # Only rendered for accumulated variables; `switch_variable`
            # drives the wrapper.
            html.Div(
                dmc.Select(
                    id="rg-accum-select",
                    label="Accumulation period",
                    description=(
                        "Total over the window ending at each lead time. Leads "
                        "shorter than the window are dropped."
                    ),
                    data=accum_options,
                    value=accum,
                ),
                id="rg-accum-wrap",
                style=_SHOWN if accum_options else _HIDDEN,
            ),
        ],
        main=[
            dmc.Alert(
                "This view reads ECMWF IFS or AIFS ENS from the dynamical.org archive. The "
                "location view's latest run comes from Open-Meteo (other models, other "
                "runs), so values will not match it; neither resolves anything finer "
                "than the 0.25° grid.",
                color="yellow",
                mb="xs",
                withCloseButton=False,
            ),
            html.Div(id="rg-alert-container"),
            dmc.Tabs(
                [
                    dmc.TabsList(
                        [
                            dmc.TabsTab("Time series", value="series"),
                            dmc.TabsTab("Map & animation", value="map"),
                        ]
                    ),
                    dmc.TabsPanel(_series_panel(), value="series", pt="sm"),
                    dmc.TabsPanel(_map_panel(), value="map", pt="sm"),
                ],
                id="rg-view-tabs",
                value="series",
                keepMounted=True,
            ),
        ],
        graph_id="regional-graph",
        share_target="rg-current-url",
    )


# Show the overlay the instant an input changes, client-side, so the user gets
# feedback during the ~10-14 s cold read without a server round-trip. The
# server callback below turns it off again when the figure lands. This
# replaces `background=True` + `running=`: see the note in `app.py` — the
# gridded stack cannot be driven from a forked background worker.
dash.clientside_callback(
    "function(bbox, variable, accum, region, run, model) { return Boolean(bbox && variable); }",
    Output("rg-loading-overlay", "visible", allow_duplicate=True),
    Input("bbox-store", "data"),
    Input("rg-variable-select", "value"),
    Input("rg-accum-select", "value"),
    Input("bbox-region-select", "value"),
    Input("rg-run-select", "value"),
    Input("rg-model-select", "value"),
    prevent_initial_call=True,
)


@callback(
    Output("rg-accum-wrap", "style"),
    Output("rg-accum-select", "data"),
    Output("rg-accum-select", "value"),
    Input("rg-variable-select", "value"),
    Input("rg-model-select", "value"),
    State("rg-accum-select", "value"),
)
def switch_variable(variable, model, current):
    """Show the accumulation control only for variables that are accumulated.

    Clearing the value on the way out matters: a stale 24 h left behind after
    switching to temperature would total an instantaneous field, which
    `fetch_bbox_ensemble` rejects outright rather than quietly computing.
    """
    options, default = _accum_options(variable, _stores(model))
    if not options:
        return _HIDDEN, [], None
    # Keep the URL's or the user's window when the new choice can build it.
    if ctx.triggered_id != "rg-variable-select" and current in {o["value"] for o in options}:
        return _SHOWN, options, current
    return _SHOWN, options, default


@callback(
    Output("rg-horizon", "data"),
    Output("rg-horizon", "value"),
    Input("rg-model-select", "value"),
    State("rg-horizon", "value"),
)
def switch_horizons(model, current):
    """Offer horizons the selected models reach; keep the current one if possible."""
    options = _horizon_options(_stores(model))
    values = {o["value"] for o in options}
    return options, current if current in values else "72"


@callback(
    Output("regional-graph", "figure"),
    Output("rg-alert-container", "children"),
    Output("rg-loading-overlay", "visible"),
    Input("bbox-store", "data"),
    Input("rg-variable-select", "value"),
    Input("rg-accum-select", "value"),
    Input("bbox-region-select", "value"),
    Input("rg-run-select", "value"),
    Input("rg-model-select", "value"),
)
def update_regional(bbox, variable, accum, region, run, model):
    stores = _stores(model)
    area_key = _area_key(bbox, region)
    if not area_key or not variable or not stores:
        return no_update, no_update, False

    is_accumulated = GRID_VARIABLES[variable].get("is_accumulated", False)
    # The control and the variable can disagree for one render after a switch,
    # since they are written by different callbacks. Trust the variable.
    accum_hours = int(accum) if (accum and is_accumulated) else None
    if is_accumulated and accum_hours is None:
        return no_update, no_update, False

    try:
        # A run the selected models do not all hold is a picker in transit
        # (a model was just added): wait for its recomposed value.
        if not run_in_calendar(stores_calendar(stores), run):
            return no_update, no_update, False
        init_time = resolve_run(stores, run)
        with status.activity("Area average"):
            results, errors = fetch_area_multi(
                area_key, variable, init_time, accum_hours, stores
            )
    except GridDataError as exc:  # over the read budget: nothing was read
        alert = dmc.Alert(str(exc), color="red", mb="xs")
        return make_empty_fig("No data available"), alert, False
    alerts = [
        dmc.Alert(msg, color="yellow" if results else "red", mb="xs") for msg in errors
    ]
    if not results:
        return make_empty_fig("No data available"), alerts, False

    drawn = {
        store: (df, compute_member_pct(df), meta) for store, (df, meta) in results.items()
    }
    fig = make_regional_multi_fig(drawn, GRID_VARIABLES[variable])
    return fig, alerts, False


@callback(
    Output("rg-pct-wrap", "style"),
    Output("rg-range-wrap", "style"),
    Input("rg-stat", "value"),
)
def show_stat_controls(statistic):
    return (
        _SHOWN if statistic == "percentile" else _HIDDEN,
        _SHOWN if statistic == "range" else _HIDDEN,
    )


# Built on the button, not on every control change: an animation is ~10
# rendered frames, and a render started per click would pile up. Opening the
# tab builds once if nothing has been built yet.
_MAP_TRIGGERS = (
    Input("rg-anim-build", "n_clicks"),
    Input("rg-view-tabs", "value"),
    # A theme switch re-renders an animation already on screen (cached per
    # theme, so switching back is free).
    Input("theme-store", "data"),
)
_MAP_STATES = (
    State("rg-anim-store", "data"),
    State("bbox-store", "data"),
    State("rg-variable-select", "value"),
    State("rg-accum-select", "value"),
    State("bbox-region-select", "value"),
    State("rg-run-select", "value"),
    State("rg-model-select", "value"),
    State("rg-stat", "value"),
    State("rg-pct", "value"),
    State("rg-pct-range", "value"),
    State("rg-step", "value"),
    State("rg-horizon", "value"),
    State("rg-projection", "value"),
    State("rg-extent", "value"),
    State("rg-palette", "value"),
    State("rg-levels", "value"),
)

dash.clientside_callback(
    """
    function(clicks, tab, theme, store) {
        const trigger = dash_clientside.callback_context.triggered_id;
        if (trigger === 'theme-store') { return Boolean(store); }
        return trigger === 'rg-anim-build' || (tab === 'map' && !store);
    }
    """,
    Output("rg-map-overlay", "visible", allow_duplicate=True),
    *_MAP_TRIGGERS,
    State("rg-anim-store", "data"),
    prevent_initial_call=True,
)


def _rethemed_args(built: dict, theme: str) -> tuple:
    """The on-screen animation's build arguments with a new theme.

    JSON turned its tuples into lists; the memo key needs them hashable.
    """
    args = list(built["args"])
    if len(args) == 14:  # built before the Levels control: Auto
        args.append("auto")
    args[8] = tuple(args[8])
    args[12] = tuple(args[12])
    args[13] = theme
    return tuple(args)


@callback(
    Output("rg-anim-store", "data"),
    Output("rg-anim-slider", "max"),
    Output("rg-anim-slider", "marks"),
    Output("rg-anim-slider", "value"),
    Output("rg-anim-caption", "children"),
    Output("rg-alert-container", "children", allow_duplicate=True),
    Output("rg-map-overlay", "visible"),
    *_MAP_TRIGGERS,
    *_MAP_STATES,
    prevent_initial_call=True,
)
def update_map(
    _clicks,
    tab,
    theme,
    built,
    bbox,
    variable,
    accum,
    region,
    run,
    model,
    statistic,
    pct,
    pct_range,
    step,
    horizon,
    projection,
    extent_key,
    palette,
    levels,
):
    """Render the animation frames on the button, or on first opening the tab."""
    skip = (no_update,) * 6
    has_area = bool(bbox) or (region and region != _CUSTOM)
    first_open = ctx.triggered_id == "rg-view-tabs" and tab == "map" and not built
    rethemed = ctx.triggered_id == "theme-store" and bool(built)
    if not (ctx.triggered_id == "rg-anim-build" or first_open or rethemed):
        return *skip, no_update
    if rethemed and built.get("precomputed"):
        return *skip, False  # precomputed frames are drawn for the dark theme only
    if not variable or not has_area:
        return *skip, False

    try:
        if rethemed and built.get("args"):
            # Re-render what is on screen, not whatever the controls say now.
            args = _rethemed_args(built, theme or "dark")
        else:
            is_accumulated = GRID_VARIABLES[variable].get("is_accumulated", False)
            accum_hours = int(accum) if (accum and is_accumulated) else None
            if region and region != _CUSTOM:
                area_key = region
            else:
                area_key = "box:" + ",".join(
                    f"{round(bbox[k], 2)}"
                    for k in ("lat_min", "lat_max", "lon_min", "lon_max")
                )
            stores = _stores(model) or (GRID_STORE_DEFAULT,)
            args = (
                area_key,
                variable,
                resolve_run(stores, run),
                int(step or 24),
                int(horizon or 72),
                accum_hours,
                statistic or "mean",
                float(pct or 90),
                tuple(pct_range or (10, 90)),
                projection or "auto",
                extent_key or "data",
                palette or "default",
                stores,
                theme or "dark",
                levels or "auto",
            )
        preset = None
        if not rethemed:
            # A preset whose run is still the newest is served as files: no read, no render.
            preset = preset_animation(
                args[0], variable, run, args[3], args[4], args[5], args[6], args[9],
                args[10], args[11], args[12], base_url=PRECOMPUTE_URL, levels=args[14],
            )
        if preset is None:
            with status.activity("Map & animation"):
                result = {**forecast_animation(*args), "args": list(args)}
        else:
            result = preset
    except GridDataError as exc:
        alert = dmc.Alert(str(exc), color="red", mb="xs")
        return None, 1, [], 0, "No animation for this selection.", alert, False

    frames = result["frames"]
    every = max(1, len(frames) // 8)
    marks = [
        {"value": i, "label": f"+{f['lead_hours']}"}
        for i, f in enumerate(frames)
        if i % every == 0
    ]
    alerts = [dmc.Alert(m, color="yellow", mb="xs") for m in result.get("errors", [])]
    return result, max(1, len(frames) - 1), marks, 0, result["caption"], alerts, False


# Player, entirely client-side: the frames are already in the browser.
dash.clientside_callback(
    """
    function(index, store) {
        const none = window.dash_clientside.no_update;
        const hidden = {display: 'none'};
        const n = __N__;
        const out = [];
        if (!store || !store.frames || !store.frames.length) {
            for (let j = 0; j < n; j++) { out.push(hidden, '', none, none, none, hidden); }
            return [...out, '', true, true, none];
        }
        const i = Math.min(index || 0, store.frames.length - 1);
        const f = store.frames[i];
        const panels = store.panels;
        for (let j = 0; j < n; j++) {
            if (j < panels.length) {
                out.push({display: 'block'}, panels[j].label, f.srcs[j], f.srcs[j],
                         `${panels[j].store}-${store.filename}-T${f.lead_hours}.png`,
                         {display: 'inline-block'});
            } else {
                out.push(hidden, '', none, none, none, hidden);
            }
        }
        const cols = {base: 1, md: Math.max(1, panels.length)};
        return [...out, `T+${f.lead_hours} h · ${f.valid}`, false, false, cols];
    }
    """.replace("__N__", str(_MAX_MODELS)),
    *(
        output
        for i in range(_MAX_MODELS)
        for output in (
            Output(f"rg-anim-slot-{i}", "style"),
            Output(f"rg-anim-name-{i}", "children"),
            Output(f"rg-anim-img-{i}", "src"),
            Output(f"rg-anim-download-{i}", "href"),
            Output(f"rg-anim-download-{i}", "download"),
            Output(f"rg-anim-download-{i}", "style"),
        )
    ),
    Output("rg-anim-label", "children"),
    Output("rg-anim-play", "disabled"),
    Output("rg-anim-slider", "disabled"),
    Output("rg-anim-panels", "cols"),
    Input("rg-anim-slider", "value"),
    Input("rg-anim-store", "data"),
)

dash.clientside_callback(
    """
    function(clicks, disabled) {
        const playing = !disabled;
        return [playing, playing ? 'Play' : 'Pause'];
    }
    """,
    Output("rg-anim-tick", "disabled"),
    Output("rg-anim-play", "children"),
    Input("rg-anim-play", "n_clicks"),
    State("rg-anim-tick", "disabled"),
    prevent_initial_call=True,
)

dash.clientside_callback(
    """
    function(n, index, store) {
        if (!store || !store.frames) { return window.dash_clientside.no_update; }
        return ((index || 0) + 1) % store.frames.length;
    }
    """,
    Output("rg-anim-slider", "value", allow_duplicate=True),
    Input("rg-anim-tick", "n_intervals"),
    State("rg-anim-slider", "value"),
    State("rg-anim-store", "data"),
    prevent_initial_call=True,
)


register_run_options("rg-run-select", store_input="rg-model-select")


@callback(
    Output("rg-model-select", "description"),
    Output("rg-variable-select", "data"),
    Output("rg-variable-select", "value", allow_duplicate=True),
    Input("rg-model-select", "value"),
    State("rg-variable-select", "value"),
    prevent_initial_call="initial_duplicate",
)
def switch_model(model, variable):
    """Offer the variables any selected model archives (AIFS has no gust).

    A model that lacks the chosen variable is reported by the fetch, and the
    others still draw; only a variable no selected model holds is disabled.
    """
    stores = _stores(model)
    if not stores:
        return no_update, no_update, no_update
    held = set()
    for store in stores:
        try:
            held |= set(store_variables(store))
        except GridDataError:
            continue
    if not held:
        return _model_description(stores), no_update, no_update
    data = [
        {**option, "disabled": option["value"] not in held}
        for option in _VARIABLE_OPTIONS
    ]
    value = no_update
    if variable not in held:
        value = _DEFAULT_VARIABLE if _DEFAULT_VARIABLE in held else min(held)
    return _model_description(stores), data, value


publish_variable("rg-variable-select", "grid", _DEFAULT_VARIABLE)

url_sync(
    "rg-current-url",
    {
        "bbox-store": None,
        "bbox-region-select": "region",
        "rg-model-select": "model",
        "rg-run-select": "run",
        "rg-variable-select": "variable",
        "rg-accum-select": "accum",
    },
)
