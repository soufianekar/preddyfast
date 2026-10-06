"""Layout + all callbacks for the Copernicus maps page (CDS reanalysis, ADS composition).

The only page that draws a *field* from Copernicus rather than a series: one
2-D map per retrieval, from ERA5 on the Climate Data Store or from the CAMS
forecasts on the Atmosphere Data Store.

It works differently from every other page, and has to. A CDS or ADS request
is queued server-side and takes anywhere from seconds to tens of minutes — far
beyond what a callback can hold open, and beyond the 120 s
`proxy_read_timeout` the deployment notes set. So the page submits, then
polls: "Retrieve map" starts a worker thread in `utils.cds_api` and a
`dcc.Interval` asks for its state until a file lands. Re-requesting something
already retrieved is answered from the on-disk cache without touching the API.

Every control is declared by the dataset catalogue rather than by this module.
A dataset lists the controls it needs (`date`, `hour`, `leadtime`, ...) and
the page shows exactly those, so adding a dataset is a `settings.py` edit.
"""

from __future__ import annotations

import urllib.parse

import dash
import dash_mantine_components as dmc
import pandas as pd
from dash import Input, Output, State, callback, dcc, html, no_update

from components.map_area_select import map_area_select
from components.page_shell import page_shell
from utils import status
from utils.cds_api import (
    JOB_DONE,
    JOB_FAILED,
    CDSDataError,
    build_request,
    credentials,
    credentials_message,
    index_requests,
    job_status,
    monthly_grid,
    submit,
    submit_or_reuse,
)
from utils.climatology import SEASONS
from utils.figures import CONFIG, make_empty_fig
from utils.map_products import (
    field_figure,
    field_static,
    index_figures,
    index_static,
)
from utils.regions import grouped_options
from utils.settings import (
    CAMS_EUROPE_MODELS,
    CDS_DATASETS,
    CDS_STORES,
    CDS_VARIABLE_NAMESPACE,
    CDS_VARIABLES,
    CLIM_PERIODS,
    CLIMATE_INDICES,
    INDEX_TARGETS,
    LEVEL_MODES,
    MAP_AREA_PRESET_LABELS,
    MAP_AREA_PRESETS,
    MONTHLY_ACCUM_DEFAULT,
    MONTHLY_ACCUM_MODES,
    STATIC_PALETTES,
    STATIC_PROJECTIONS,
)
from utils.url_state import query_choice
from utils.variable_family import publish, resolve

dash.register_page(__name__, path="/maps", title="Climate maps & indices · asifast")

# Climate indices are offered as one more "dataset": its variables are the
# indices, and its controls are the correlation month, reference and target.
_INDEX_DATASET = "climate_indices"
_INDEX_DATASET_LABEL = "Climate variability indices (ERA5-derived)"
_DEFAULT_INDEX = "oni"

_DATASET_OPTIONS = [
    {"value": k, "label": v["label"]} for k, v in CDS_DATASETS.items()
] + [{"value": _INDEX_DATASET, "label": _INDEX_DATASET_LABEL}]
_INDEX_TARGET_OPTIONS = [{"value": k, "label": v["label"]} for k, v in INDEX_TARGETS.items()]
_PROJECTION_OPTIONS = [{"value": k, "label": v} for k, v in STATIC_PROJECTIONS.items()]
_PALETTE_OPTIONS = [{"value": k, "label": v} for k, v in STATIC_PALETTES.items()]
_LEVEL_OPTIONS = [{"value": k, "label": v} for k, v in LEVEL_MODES.items()]
_EXTENT_OPTIONS = grouped_options(extra=[{"value": "data", "label": "Data area"}])
_DEFAULT_DATASET = "era5_single_levels"

_HOUR_OPTIONS = [
    {"value": f"{h:02d}:00", "label": f"{h:02d}:00 UTC"} for h in range(24)
]
_SEASON_MONTHS_LABEL = {"DJF": "Dec–Feb", "MAM": "Mar–May", "JJA": "Jun–Aug", "SON": "Sep–Nov"}


def _season_label(season: str, year: int) -> str:
    if season == "DJF":
        return f"DJF {year} (Dec {year - 1}–Feb {year}) seasonal mean"
    return f"{season} {year} seasonal mean"


_MONTH_OPTIONS = [
    {"value": str(m), "label": pd.Timestamp(2000, m, 1).strftime("%B")}
    for m in range(1, 13)
]
# Months and seasons in one picker for the monthly datasets.
_PERIOD_OPTIONS = [
    {"group": "Month", "items": _MONTH_OPTIONS},
    {
        "group": "Season (3-month mean)",
        "items": [
            {"value": key, "label": f"{key} ({_SEASON_MONTHS_LABEL[key]})"} for key in SEASONS
        ],
    },
]
_REFERENCE_OPTIONS = [{"value": p, "label": p} for p in CLIM_PERIODS]

# Every control the catalogue can ask for, and the wrapper id that shows or
# hides it. Declared once so the visibility callback and the layout cannot
# drift apart.
_CONTROL_WRAPPERS: dict[str, str] = {
    "date": "mp-wrap-date",
    "hour": "mp-wrap-hour",
    "month_year": "mp-wrap-month-year",
    "init_hour": "mp-wrap-init-hour",
    "leadtime": "mp-wrap-leadtime",
    "cams_model": "mp-wrap-cams-model",
    "anomaly": "mp-wrap-anomaly",
    "index": "mp-wrap-index",
}
# The area picker is hidden for the indices, whose map areas are fixed.
_AREA_WRAPPER = "mp-wrap-area"

# Owned by `switch_variable`, not by `switch_dataset`: whether an accumulation
# period applies depends on the variable, not on the dataset that serves it.
_ACCUM_WRAPPER = "mp-wrap-accum"

_HIDDEN = {"display": "none"}
_SHOWN = {"display": "block"}


def _default_date(dataset_key: str) -> str:
    """Most recent date a dataset can plausibly serve, as an ISO string."""
    if dataset_key == _INDEX_DATASET:
        dataset_key = "era5_monthly"
    lag = CDS_DATASETS[dataset_key]["lag_days"]
    return (pd.Timestamp.now(tz="UTC").normalize() - pd.Timedelta(days=lag)).strftime(
        "%Y-%m-%d"
    )


def _variable_options(dataset_key: str) -> list[dict]:
    if dataset_key == _INDEX_DATASET:
        return [{"value": k, "label": v["label"]} for k, v in CLIMATE_INDICES.items()]
    return [
        {"value": k, "label": v["label"]} for k, v in CDS_VARIABLES[dataset_key].items()
    ]


def _leadtime_options(dataset_key: str) -> list[dict]:
    hours = CDS_DATASETS[dataset_key].get("leadtime_hours", (0,))
    return [{"value": str(h), "label": f"T+{h} h"} for h in hours]


def _accum_options(dataset_key: str, variable: str | None) -> tuple[list[dict], str | None]:
    """Accumulation-period options for a variable, empty if it is not accumulated.

    The options live on the variable because they depend on the native time
    step behind it: an hourly reanalysis field can be totalled over an hour or
    a day, a 3-hourly forecast over 6/12/24 h, and a monthly mean over neither.
    """
    meta = CDS_VARIABLES.get(dataset_key, {}).get(variable or "", {})

    if meta.get("is_monthly_rate"):
        # A monthly mean holds no sub-monthly information, so the control
        # offers the window the rate is *expressed* over, not one to
        # accumulate across.
        return (
            [{"value": k, "label": v["label"]} for k, v in MONTHLY_ACCUM_MODES.items()],
            MONTHLY_ACCUM_DEFAULT,
        )

    if not meta.get("is_accumulated"):
        return [], None

    labels = {1: "1 h (native step)", 24: "24 h (00–24 UTC)"}
    options = [
        {"value": str(h), "label": labels.get(h, f"{h} h")}
        for h in meta["accum_periods"]
    ]
    return options, str(meta["accum_default"])


def _control(component, wrapper_id: str, visible: bool) -> html.Div:
    return html.Div(component, id=wrapper_id, style=_SHOWN if visible else _HIDDEN)


def _static_panel() -> list:
    """Static tab: style controls over one or two PNGs and their downloads."""
    image_style = {"width": "100%", "maxWidth": "980px", "display": "none"}
    return [
        dmc.SimpleGrid(
            [
                dmc.Select(
                    id="st-projection",
                    label="Projection",
                    data=_PROJECTION_OPTIONS,
                    value="auto",
                    allowDeselect=False,
                    size="xs",
                ),
                dmc.Select(
                    id="st-extent",
                    label="Extent",
                    data=_EXTENT_OPTIONS,
                    value="data",
                    searchable=True,
                    allowDeselect=False,
                    size="xs",
                ),
                dmc.Select(
                    id="st-palette",
                    label="Palette",
                    data=_PALETTE_OPTIONS,
                    value="default",
                    allowDeselect=False,
                    size="xs",
                ),
                dmc.Select(
                    id="st-levels",
                    label="Levels",
                    description="Auto fits an area cut from a larger field.",
                    data=_LEVEL_OPTIONS,
                    value="auto",
                    allowDeselect=False,
                    size="xs",
                ),
            ],
            cols={"base": 1, "sm": 4},
            spacing="xs",
            mb="xs",
        ),
        html.Div(id="st-alert-container"),
        dcc.Loading(
            dmc.Stack(
                [
                    dmc.Text(
                        "Choose a field and press Retrieve map.",
                        id="st-caption",
                        size="sm",
                        c="dimmed",
                    ),
                    html.Img(id="st-image", alt="Static map", style=image_style),
                    html.Img(id="st-image-2", alt="Index time series", style=image_style),
                    dmc.Group(
                        [
                            html.A(
                                dmc.Button("Download map PNG", variant="light", size="xs"),
                                id="st-download",
                                style={"display": "none"},
                            ),
                            html.A(
                                dmc.Button(
                                    "Download time series PNG", variant="light", size="xs"
                                ),
                                id="st-download-2",
                                style={"display": "none"},
                            ),
                        ],
                        gap="xs",
                    ),
                ],
                gap="xs",
            ),
            type="dot",
        ),
    ]


def _interactive_panel() -> list:
    """Interactive tab: the Plotly map, plus the index time series when shown."""
    return [
        dmc.Paper(
            dcc.Graph(
                id="map-graph",
                figure=make_empty_fig("Choose a field and press Retrieve map"),
                config=CONFIG,
                style={"height": "680px"},
            ),
            withBorder=True,
            radius="md",
            p="xs",
        ),
        html.Div(
            dmc.Paper(
                dcc.Graph(id="mp-ts-graph", config=CONFIG, style={"height": "440px"}),
                withBorder=True,
                radius="md",
                p="xs",
                mt="sm",
            ),
            id="mp-ts-wrap",
            style=_HIDDEN,
        ),
    ]


def layout(**query_params):
    dataset = _DEFAULT_DATASET
    controls = CDS_DATASETS[dataset]["controls"]
    # Only the nav link's variable is read here; a full shared link is
    # restored by `restore_from_url`, which sets the dataset first.
    variable = query_choice(
        query_params,
        "variable",
        CDS_VARIABLES[dataset],
        CDS_DATASETS[dataset]["default_variable"],
    )

    return page_shell(
        title="Maps & indices",
        description=(
            "Gridded fields retrieved on demand from the Climate Data Store "
            "(ERA5) and the Atmosphere Data Store (CAMS). Requests are queued by "
            "Copernicus — the first retrieval of a given field can take minutes; "
            "repeats are served from cache."
        ),
        sidebar=[
            dcc.Location(id="_mp_location"),
            html.Div(id="mp-current-url", style={"display": "none"}),
            dcc.Store(id="mp-job-store"),
            dcc.Interval(id="mp-poll", interval=2000, disabled=True),
            # Relay: `switch_dataset` writes the new dataset's default extent
            # here and `apply_default_area` forwards it to the picker, so the
            # picker keeps sole ownership of its own inputs.
            html.Div(id="mp-preset-sync", style={"display": "none"}),
                dmc.Select(
                    id="mp-dataset-select",
                    allowDeselect=False,
                    label="Dataset",
                    data=_DATASET_OPTIONS,
                    value=dataset,
                ),
                dmc.Select(
                    id="mp-variable-select",
                    allowDeselect=False,
                    label="Variable",
                    data=_variable_options(dataset),
                    value=variable,
                ),
                _control(
                    dmc.DatePickerInput(
                        id="mp-date-input",
                        label="Date",
                        value=_default_date(dataset),
                        valueFormat="YYYY-MM-DD",
                    ),
                    _CONTROL_WRAPPERS["date"],
                    "date" in controls,
                ),
                _control(
                    dmc.Select(
                        id="mp-hour-select",
                        allowDeselect=False,
                        label="Hour (UTC)",
                        data=_HOUR_OPTIONS,
                        value="12:00",
                    ),
                    _CONTROL_WRAPPERS["hour"],
                    "hour" in controls,
                ),
                _control(
                    dmc.Select(
                        id="mp-init-hour-select",
                        allowDeselect=False,
                        label="Run init (UTC)",
                        data=[
                            {"value": "00:00", "label": "00 UTC"},
                            {"value": "12:00", "label": "12 UTC"},
                        ],
                        value="00:00",
                    ),
                    _CONTROL_WRAPPERS["init_hour"],
                    "init_hour" in controls,
                ),
                _control(
                    dmc.Select(
                        id="mp-leadtime-select",
                        allowDeselect=False,
                        label="Lead time",
                        data=_leadtime_options("cams_global"),
                        value="0",
                    ),
                    _CONTROL_WRAPPERS["leadtime"],
                    "leadtime" in controls,
                ),
                _control(
                    dmc.Select(
                        id="mp-cams-model-select",
                        allowDeselect=False,
                        label="Regional model",
                        data=[
                            {"value": k, "label": v} for k, v in CAMS_EUROPE_MODELS.items()
                        ],
                        value="ensemble",
                    ),
                    _CONTROL_WRAPPERS["cams_model"],
                    "cams_model" in controls,
                ),
                _control(
                    dmc.SimpleGrid(
                        [
                            dmc.Select(
                                id="mp-month-select",
                                allowDeselect=False,
                                label="Month or season",
                                data=_PERIOD_OPTIONS,
                                value="7",
                            ),
                            dmc.NumberInput(
                                id="mp-year-input",
                                label="Year",
                                value=pd.Timestamp.now().year - 1,
                                min=1940,
                                max=pd.Timestamp.now().year,
                                step=1,
                            ),
                        ],
                        cols=2,
                        spacing="xs",
                    ),
                    _CONTROL_WRAPPERS["month_year"],
                    "month_year" in controls,
                ),
                _control(
                    dmc.Select(
                        id="mp-accum-select",
                        allowDeselect=False,
                        label="Accumulation",
                        data=[],
                        value=None,
                    ),
                    _ACCUM_WRAPPER,
                    False,
                ),
                _control(
                    dmc.Stack(
                        [
                            dmc.Switch(
                                id="mp-anomaly-switch",
                                label="Show anomaly",
                                checked=False,
                                size="sm",
                            ),
                            dmc.Select(
                                id="mp-reference-select",
                                allowDeselect=False,
                                label="Reference period",
                                data=_REFERENCE_OPTIONS,
                                value="1991-2020",
                                size="xs",
                            ),
                        ],
                        gap="xs",
                    ),
                    _CONTROL_WRAPPERS["anomaly"],
                    "anomaly" in controls,
                ),
                _control(
                    dmc.Stack(
                        [
                            dmc.Text(
                                CLIMATE_INDICES[_DEFAULT_INDEX]["definition"],
                                id="mp-index-definition",
                                size="xs",
                                c="dimmed",
                            ),
                            dmc.SimpleGrid(
                                [
                                    dmc.Select(
                                        id="mp-index-month",
                                        label="Correlation month",
                                        data=_MONTH_OPTIONS,
                                        value="1",
                                        allowDeselect=False,
                                    ),
                                    dmc.Select(
                                        id="mp-index-reference",
                                        label="Reference period",
                                        data=_REFERENCE_OPTIONS,
                                        value="1991-2020",
                                        allowDeselect=False,
                                    ),
                                ],
                                cols=2,
                                spacing="xs",
                            ),
                            dmc.Select(
                                id="mp-index-target",
                                label="Correlate with",
                                data=_INDEX_TARGET_OPTIONS,
                                value=CLIMATE_INDICES[_DEFAULT_INDEX][
                                    "default_target"
                                ],
                                allowDeselect=False,
                            ),
                            dmc.Text(
                                id="mp-index-area",
                                size="xs",
                                c="dimmed",
                            ),
                        ],
                        gap="xs",
                    ),
                    _CONTROL_WRAPPERS["index"],
                    False,
                ),
                html.Div(map_area_select(), id=_AREA_WRAPPER),
                dmc.Group(
                    [
                        dmc.Button("Retrieve map", id="mp-submit-btn", variant="filled"),
                        dmc.Text(id="mp-status", size="sm", c="dimmed"),
                    ],
                    gap="md",
                ),
        ],
        main=[
            html.Div(id="mp-note-container"),
            html.Div(id="mp-alert-container"),
            dcc.Store(id="mp-result-store"),
            dcc.Store(id="mp-interactive-rendered"),
            dmc.Box(
                [
                    # DMC 2.x LoadingOverlay is a sibling overlay driven by
                    # `visible`, not a wrapper, and needs a positioned
                    # ancestor to cover. It covers both tabs while retrieving.
                    dmc.LoadingOverlay(
                        id="mp-loading-overlay",
                        visible=False,
                        overlayProps={"blur": 2},
                        loaderProps={"type": "dots"},
                    ),
                    dmc.Tabs(
                        [
                            dmc.TabsList(
                                [
                                    dmc.TabsTab("Static map", value="static"),
                                    dmc.TabsTab("Interactive", value="interactive"),
                                ]
                            ),
                            dmc.TabsPanel(_static_panel(), value="static", pt="sm"),
                            dmc.TabsPanel(
                                _interactive_panel(), value="interactive", pt="sm"
                            ),
                        ],
                        id="mp-view-tabs",
                        value="static",
                        # Kept mounted: the interactive graph and the static
                        # controls are callback targets whichever tab shows.
                        keepMounted=True,
                    ),
                ],
                pos="relative",
                mih=360,
            ),
        ],
        graph_id="map-graph",
        share_target="mp-current-url",
    )


@callback(
    *[Output(wrapper, "style") for wrapper in _CONTROL_WRAPPERS.values()],
    Output("mp-variable-select", "data"),
    Output("mp-variable-select", "value"),
    Output("mp-leadtime-select", "data"),
    Output("mp-leadtime-select", "value"),
    Output("mp-date-input", "value"),
    Output("mp-preset-sync", "children"),
    Output("mp-note-container", "children"),
    Output(_AREA_WRAPPER, "style"),
    Input("mp-dataset-select", "value"),
    prevent_initial_call=True,
)
def switch_dataset(dataset_key):
    """Re-derive every control from the catalogue when the dataset changes.

    The variable list, lead times, default date and control visibility all
    belong to the dataset, so they are reset together — leaving a CAMS
    variable selected after switching to ERA5 would build a request the API
    rejects several minutes later.
    """
    if dataset_key == _INDEX_DATASET:
        styles = [_SHOWN if name == "index" else _HIDDEN for name in _CONTROL_WRAPPERS]
        note = dmc.Alert(
            "Monthly indices computed from ERA5 back to 1950, each a proxy for the "
            "agency index of the same name. The correlation map covers a fixed "
            "area per index.",
            title=_INDEX_DATASET_LABEL,
            color="blue",
            variant="light",
            mb="xs",
            withCloseButton=False,
        )
        return (
            *styles,
            _variable_options(dataset_key),
            _DEFAULT_INDEX,
            [],
            no_update,
            no_update,
            no_update,
            note,
            _HIDDEN,
        )

    if dataset_key not in CDS_DATASETS:
        return (no_update,) * (len(_CONTROL_WRAPPERS) + 8)

    meta = CDS_DATASETS[dataset_key]
    controls = meta["controls"]
    styles = [_SHOWN if name in controls else _HIDDEN for name in _CONTROL_WRAPPERS]

    leadtimes = _leadtime_options(dataset_key) if "leadtime" in controls else []
    leadtime_value = leadtimes[0]["value"] if leadtimes else no_update

    note = dmc.Alert(
        meta["note"],
        title=meta["label"],
        color="blue",
        variant="light",
        mb="xs",
        withCloseButton=False,
    )

    return (
        *styles,
        _variable_options(dataset_key),
        meta["default_variable"],
        leadtimes,
        leadtime_value,
        _default_date(dataset_key),
        meta["default_area"],
        note,
        _SHOWN,
    )


@callback(
    Output("mp-index-definition", "children"),
    Output("mp-index-target", "value"),
    Output("mp-index-area", "children"),
    Input("mp-variable-select", "value"),
    State("mp-dataset-select", "value"),
)
def switch_index(variable, dataset_key):
    """Describe the chosen index and preset its most telling correlation target."""
    index = CLIMATE_INDICES.get(variable or "")
    if dataset_key != _INDEX_DATASET or index is None:
        return no_update, no_update, no_update
    area = MAP_AREA_PRESET_LABELS.get(index["map_area"], index["map_area"])
    return index["definition"], index["default_target"], f"Correlation area: {area}"


@callback(
    Output(_ACCUM_WRAPPER, "style"),
    Output("mp-accum-select", "data"),
    Output("mp-accum-select", "value"),
    Input("mp-dataset-select", "value"),
    Input("mp-variable-select", "value"),
)
def switch_variable(dataset_key, variable):
    """Show the accumulation control only for variables that are accumulated.

    Clearing the value on the way out matters: a 24 h left behind after
    switching from precipitation to temperature would widen the request to two
    days and sum an instantaneous field into nonsense.
    """
    options, default = _accum_options(dataset_key, variable)
    if not options:
        return _HIDDEN, [], None
    return _SHOWN, options, default


@callback(
    Output("ma-preset-select", "value", allow_duplicate=True),
    Input("mp-preset-sync", "children"),
    prevent_initial_call=True,
)
def apply_default_area(preset):
    """Move the area picker to the new dataset's default extent.

    Routed through a hidden relay rather than written directly by
    `switch_dataset`, because the picker owns that input and two callbacks
    writing one property need `allow_duplicate` on the losing one — keeping
    the relay here makes which callback that is explicit.
    """
    return preset or no_update


def _valid_time_hint(
    dataset_key: str,
    date: str | None,
    hour: str | None,
    init_hour: str | None,
    leadtime: str | None,
) -> str | None:
    """When the field is valid, derived from the controls rather than the file.

    Only a fallback: `cds_api.load_field` reports the file's own timestamp
    whenever the file carries a real one. This covers the datasets whose time
    axis holds lead hours instead — reading a date out of those would invent
    one.

    Returned as an ISO string because it travels through a `dcc.Store`, and a
    value that round-trips as itself is worth more than one that depends on
    what the serialiser happens to do with a `Timestamp`.
    """
    controls = CDS_DATASETS[dataset_key]["controls"]
    if not date or "date" not in controls:
        return None

    base = pd.Timestamp(date)
    if "init_hour" in controls and init_hour:
        base += pd.Timedelta(hours=int(init_hour.split(":")[0]))
    elif "hour" in controls and hour:
        base += pd.Timedelta(hours=int(hour.split(":")[0]))

    if "leadtime" in controls and leadtime is not None:
        base += pd.Timedelta(hours=int(leadtime))

    return base.isoformat()


def _resolve_area(dataset_key: str, area: dict | None) -> tuple[tuple, list]:
    """Order the box, clamp it to a dataset's domain, and report any clamping.

    Returns
    -------
    tuple
        `((lat_min, lat_max, lon_min, lon_max), alerts)`. The alerts are
        `dmc.Alert` children, empty when the drawn box needed no adjustment.
    """
    if not area:
        return MAP_AREA_PRESETS[CDS_DATASETS[dataset_key]["default_area"]], []

    box = (
        float(area["lat_min"]),
        float(area["lat_max"]),
        float(area["lon_min"]),
        float(area["lon_max"]),
    )

    domain = CDS_DATASETS[dataset_key].get("domain")
    if domain is None:
        return box, []

    d_lat_min, d_lat_max, d_lon_min, d_lon_max = domain
    clamped = (
        max(box[0], d_lat_min),
        min(box[1], d_lat_max),
        max(box[2], d_lon_min),
        min(box[3], d_lon_max),
    )

    if clamped[0] >= clamped[1] or clamped[2] >= clamped[3]:
        return box, [
            dmc.Alert(
                f"{CDS_DATASETS[dataset_key]['label']} covers "
                f"{d_lat_min:g}–{d_lat_max:g}°N, {d_lon_min:g}–{d_lon_max:g}°E only, "
                "and the selected area lies entirely outside it.",
                color="red",
                mb="xs",
            )
        ]

    if clamped != box:
        return clamped, [
            dmc.Alert(
                f"The selected area was clipped to the "
                f"{CDS_DATASETS[dataset_key]['label']} domain "
                f"({d_lat_min:g}–{d_lat_max:g}°N, {d_lon_min:g}–{d_lon_max:g}°E).",
                color="yellow",
                mb="xs",
                withCloseButton=False,
            )
        ]

    return clamped, []


@callback(
    Output("mp-job-store", "data"),
    Output("mp-poll", "disabled"),
    Output("mp-alert-container", "children"),
    Output("mp-status", "children"),
    Output("mp-loading-overlay", "visible", allow_duplicate=True),
    Input("mp-submit-btn", "n_clicks"),
    State("mp-dataset-select", "value"),
    State("mp-variable-select", "value"),
    State("map-area-store", "data"),
    State("mp-date-input", "value"),
    State("mp-hour-select", "value"),
    State("mp-month-select", "value"),
    State("mp-year-input", "value"),
    State("mp-init-hour-select", "value"),
    State("mp-leadtime-select", "value"),
    State("mp-cams-model-select", "value"),
    State("mp-anomaly-switch", "checked"),
    State("mp-reference-select", "value"),
    State("mp-accum-select", "value"),
    State("mp-index-month", "value"),
    State("mp-index-reference", "value"),
    State("mp-index-target", "value"),
    prevent_initial_call=True,
)
def start_retrieval(
    _clicks,
    dataset_key,
    variable,
    area,
    date,
    hour,
    month,
    year,
    init_hour,
    leadtime,
    cams_model,
    anomaly,
    reference,
    accum,
    index_month,
    index_reference,
    index_target,
):
    """Validate the form, queue the retrieval(s), and hand the poller their ids.

    An anomaly needs two retrievals — the month itself and its reference
    period — and they queue together rather than in sequence, because CDS
    serves two concurrent requests per user and running them back to back
    would double an already long wait.
    """
    if not dataset_key or not variable:
        return no_update, True, no_update, "", False

    if dataset_key == _INDEX_DATASET:
        return _start_index(variable, index_month, index_reference, index_target)

    meta = CDS_DATASETS[dataset_key]
    controls = meta["controls"]
    store_meta = CDS_STORES[meta["store"]]

    if credentials(meta["store"]) is None:
        return (
            no_update,
            True,
            dmc.Alert(
                credentials_message(meta["store"]),
                title=f"{store_meta['short']} token missing",
                color="red",
                mb="xs",
            ),
            "",
            False,
        )

    box, alerts = _resolve_area(dataset_key, area)
    if any(alert.color == "red" for alert in alerts):
        return no_update, True, alerts, "", False

    wants_anomaly = bool(anomaly) and "anomaly" in controls
    # The month picker also offers the four seasons (see `_MONTH_OPTIONS`).
    season = month if month in SEASONS else None
    month_value = int(month) if month and not season else None
    year_value = int(year) if year else None
    if season and year_value is None:
        alert = dmc.Alert("Choose a year.", color="yellow", mb="xs")
        return no_update, True, [*alerts, alert], "", False
    if "month_year" in controls and not season and (month_value is None or year_value is None):
        alert = dmc.Alert("Choose a month and a year.", color="yellow", mb="xs")
        return no_update, True, [*alerts, alert], "", False
    if "date" in controls and not date:
        alert = dmc.Alert("Choose a date.", color="yellow", mb="xs")
        return no_update, True, [*alerts, alert], "", False

    var_meta = CDS_VARIABLES[dataset_key][variable]
    # The control and the variable can disagree for one render after a switch,
    # since different callbacks write them. Trust the variable.
    accum_hours = int(accum) if (accum and var_meta.get("is_accumulated")) else None
    monthly_mode = accum if (accum and var_meta.get("is_monthly_rate")) else None
    grid = monthly_grid(dataset_key, box)

    try:
        request = build_request(
            dataset_key,
            variable,
            box,
            date=date,
            hour=hour,
            year=year_value,
            month=month_value,
            # A DJF season needs December of the previous year.
            years=(
                [year_value - 1, year_value] if season == "DJF" else [year_value]
            )
            if season
            else None,
            months=list(SEASONS[season]) if season else None,
            leadtime_hour=int(leadtime) if leadtime is not None else None,
            init_hour=init_hour,
            cams_model=cams_model,
            accum_hours=accum_hours,
            grid=grid,
        )
        # A cached larger field holding this area is cut, not re-retrieved.
        field_id, cut = submit_or_reuse(dataset_key, request)
        jobs = {"field": field_id}

        if wants_anomaly:
            first, last = (int(part) for part in reference.split("-"))
            jobs["reference"], ref_cut = submit_or_reuse(
                dataset_key,
                build_request(
                    dataset_key,
                    variable,
                    box,
                    month=month_value,
                    years=list(
                        range(first - 1 if season == "DJF" else first, last + 1)
                    ),
                    months=list(SEASONS[season]) if season else None,
                    grid=grid,
                ),
            )
    except (CDSDataError, ValueError, KeyError) as exc:
        return (
            no_update,
            True,
            [*alerts, dmc.Alert(str(exc), color="red", mb="xs")],
            "",
            False,
        )

    job = {
        "kind": "field",
        "season": season,
        "year": year_value,
        "reference": [int(p) for p in reference.split("-")] if wants_anomaly else None,
        "dataset": dataset_key,
        "variable": variable,
        "jobs": jobs,
        "crop": list(box) if (cut or (wants_anomaly and ref_cut)) else None,
        "grid": grid,
        "accum_hours": accum_hours,
        "monthly_mode": monthly_mode,
        # The window opens at 00 UTC on the chosen date; `load_field` closes it
        # 24 h later, right-inclusive, per ERA5's accumulation convention.
        "window_start": f"{date}T00:00:00" if accum_hours and date else None,
        "request_meta": {
            "store_label": f"{store_meta['label']} ({store_meta['short']})",
            # The standard region the box came from, if the user picked one
            # and did not then redraw it; named in the subtitle and outlined.
            "region": (area or {}).get("preset"),
            "region_label": MAP_AREA_PRESET_LABELS.get((area or {}).get("preset") or ""),
            "date": date if "date" in controls else None,
            "hour": hour if "hour" in controls else None,
            "init_time": (
                f"{date} {init_hour}" if "init_hour" in controls and date else None
            ),
            "leadtime_hour": (
                int(leadtime)
                if "leadtime" in controls and leadtime is not None
                else None
            ),
            "cams_model": (
                CAMS_EUROPE_MODELS.get(cams_model) if "cams_model" in controls else None
            ),
            "month_label": (
                _season_label(season, year_value)
                if season
                else f"{pd.Timestamp(year_value, month_value, 1):%B %Y} monthly mean"
                if "month_year" in controls and year_value and month_value
                else None
            ),
            # Fallback validity time, used only when the file cannot state its
            # own. CAMS Europe keeps forecast lead hours in a coordinate named
            # `time`, so the loader refuses to read a date out of it rather
            # than reporting the Unix epoch.
            "valid_time_hint": _valid_time_hint(
                dataset_key, date, hour, init_hour, leadtime
            ),
            # Only accumulated variables carry this; instantaneous ones are
            # genuinely valid at an instant and must not be annotated as if
            # they were totals over a window.
            "accumulation": (
                f"{accum_hours} h accumulation, {date} 00–24 UTC"
                if accum_hours and accum_hours > 1
                else var_meta.get("accumulation")
            ),
        },
        "anomaly": wants_anomaly,
    }

    return job, False, alerts, "Queued at Copernicus…", True


def _start_index(index_key, month, reference, target):
    """Queue the index source record and its correlation target."""
    if index_key not in CLIMATE_INDICES or not month or not reference or not target:
        alert = dmc.Alert(
            "Choose an index, a correlation month, a reference period and a target.",
            color="yellow",
            mb="xs",
        )
        return no_update, True, alert, "", False
    if credentials("cds") is None:
        alert = dmc.Alert(credentials_message("cds"), color="red", mb="xs")
        return no_update, True, alert, "", False

    first, last = (int(part) for part in reference.split("-"))
    latest = pd.Timestamp(_default_date("era5_monthly"))
    try:
        roles = index_requests(index_key, int(month), (first, last), target, latest)
        jobs = {role: submit(dataset, request) for role, (dataset, request) in roles.items()}
    except (CDSDataError, ValueError, KeyError) as exc:
        return no_update, True, dmc.Alert(str(exc), color="red", mb="xs"), "", False

    job = {
        "kind": "index",
        "index": index_key,
        "month": int(month),
        "reference": [first, last],
        "target": target,
        "jobs": jobs,
    }
    return job, False, [], "Queued at Copernicus…", True


@callback(
    Output("mp-result-store", "data"),
    Output("mp-poll", "disabled", allow_duplicate=True),
    Output("mp-alert-container", "children", allow_duplicate=True),
    Output("mp-status", "children", allow_duplicate=True),
    Output("mp-loading-overlay", "visible"),
    Input("mp-poll", "n_intervals"),
    State("mp-job-store", "data"),
    prevent_initial_call=True,
)
def poll_retrieval(_n, job):
    """Wait for every retrieval of the job, then publish it for rendering.

    Runs on the `dcc.Interval` rather than inline with the submit, because a
    Copernicus request can outlast any callback the proxy will hold open.
    Drawing is left to `render_static` / `render_interactive`, so changing a
    projection or palette re-renders without touching the retrieval.
    """
    if not job:
        return no_update, True, no_update, "", False

    states = {role: job_status(job_id) for role, job_id in job["jobs"].items()}

    failed = [s for s in states.values() if s["state"] == JOB_FAILED]
    if failed:
        status.track("cds:" + "-".join(sorted(job["jobs"].values())), "Copernicus retrieval",
                     "Failed", state="failed")
        alert = dmc.Alert(
            failed[0].get("message") or "The retrieval failed.",
            title="Copernicus request failed",
            color="red",
            mb="xs",
        )
        return no_update, True, alert, "", False

    if any(s["state"] == "unknown" for s in states.values()):
        # The worker holding the thread has gone (a restart, or a request
        # routed to a process that never saw the submit and found no file).
        alert = dmc.Alert(
            "The retrieval is no longer being tracked. Press “Retrieve map” again "
            "— anything already downloaded is served from cache.",
            color="yellow",
            mb="xs",
            withCloseButton=False,
        )
        return no_update, True, alert, "", False

    pending = [s for s in states.values() if s["state"] != JOB_DONE]
    key = "cds:" + "-".join(sorted(job["jobs"].values()))
    if pending:
        done = len(states) - len(pending)
        message = pending[0].get("message", "Retrieving…")
        status.track(key, "Copernicus retrieval", message, done, len(states))
        return no_update, False, no_update, f"{done}/{len(states)} files · {message}", True

    status.track(key, "Copernicus retrieval", "Retrieved", state="done")
    result = {**job, "paths": {role: s["path"] for role, s in states.items()}}
    return result, True, [], "Retrieved.", False


@callback(
    Output("st-image", "src"),
    Output("st-image", "style"),
    Output("st-image-2", "src"),
    Output("st-image-2", "style"),
    Output("st-caption", "children"),
    Output("st-download", "href"),
    Output("st-download", "download"),
    Output("st-download", "style"),
    Output("st-download-2", "href"),
    Output("st-download-2", "download"),
    Output("st-download-2", "style"),
    Output("st-alert-container", "children"),
    Input("mp-result-store", "data"),
    Input("st-projection", "value"),
    Input("st-extent", "value"),
    Input("st-palette", "value"),
    Input("st-levels", "value"),
    Input("theme-store", "data"),
    prevent_initial_call=True,
)
def render_static(result, projection, extent_key, palette, levels, theme):
    """Draw the static tab; re-runs on a style change without re-retrieving."""
    if not result:
        return (no_update,) * 12

    shown = {"width": "100%", "maxWidth": "980px", "display": "block"}
    link = {"display": "inline-block"}
    try:
        if result["kind"] == "index":
            out = index_static(result, projection, extent_key, palette, theme or "dark")
            return (
                out["map_src"], shown, out["ts_src"], shown, out["caption"],
                out["map_src"], out["map_filename"], link,
                out["ts_src"], out["ts_filename"], link,
                [],
            )
        out = field_static(result, projection, extent_key, palette, theme or "dark", levels or "auto")
    except (CDSDataError, ValueError) as exc:
        alert = dmc.Alert(str(exc), color="red", mb="xs")
        return (*(no_update,) * 11, alert)

    return (
        out["src"], shown, "", _HIDDEN, out["caption"],
        out["src"], out["filename"], link,
        "", "", _HIDDEN,
        [],
    )


@callback(
    Output("map-graph", "figure"),
    Output("mp-ts-graph", "figure"),
    Output("mp-ts-wrap", "style"),
    Output("mp-interactive-rendered", "data"),
    Output("mp-alert-container", "children", allow_duplicate=True),
    Input("mp-result-store", "data"),
    Input("mp-view-tabs", "value"),
    State("mp-interactive-rendered", "data"),
    prevent_initial_call=True,
)
def render_interactive(result, tab, rendered):
    """Draw the interactive tab, lazily: only when it is shown, once per result.

    A global field is ~10 MB of Plotly JSON; building it for users who stay
    on the static tab would be pure waste.
    """
    if not result or tab != "interactive":
        return (no_update,) * 5
    key = str(sorted(result["jobs"].items()))
    if rendered == key:
        return (no_update,) * 5
    try:
        if result["kind"] == "index":
            ts_fig, map_fig = index_figures(result)
            return map_fig, ts_fig, _SHOWN, key, no_update
        return field_figure(result), no_update, _HIDDEN, key, no_update
    except (CDSDataError, ValueError) as exc:
        alert = dmc.Alert(str(exc), color="red", mb="xs")
        return no_update, no_update, no_update, no_update, alert


# Point the shell's save-PNG tool at the Plotly map only on the interactive
# tab, and nudge Plotly to size itself: a graph first drawn in a hidden panel
# measures zero width until the window reports a resize.
dash.clientside_callback(
    """
    function(tab) {
        setTimeout(() => window.dispatchEvent(new Event('resize')), 60);
        return tab === 'interactive' ? 'map-graph' : 'st-image';
    }
    """,
    Output("shell-graph", "data"),
    Input("mp-view-tabs", "value"),
)


@callback(
    Output("variable-store", "data", allow_duplicate=True),
    Input("mp-variable-select", "value"),
    State("mp-dataset-select", "value"),
    State("variable-store", "data"),
    prevent_initial_call=True,
)
def publish_variable(variable, dataset_key, store):
    """Record the choice for other pages; CAMS species belong to no family.

    Skips the value the page opened with, as `shared_state.publish_variable`
    does — including a dataset's default after a dataset switch, which is
    not a variable choice.
    """
    namespace = CDS_VARIABLE_NAMESPACE.get(dataset_key, dataset_key)
    opened_with = resolve(store, namespace) or CDS_DATASETS.get(
        dataset_key, {}
    ).get("default_variable")
    if not variable or variable == opened_with:
        return no_update
    return publish(store, namespace, variable)


dash.clientside_callback(
    """
    function(dataset, variable, area, date, hour, month, year, lead, init, model, anomaly, reference) {
        const p = new URLSearchParams({
            dataset: dataset || '', variable: variable || '',
            date: date || '', hour: hour || '', month: month || '', year: year || '',
            lead: (lead === null || lead === undefined) ? '' : lead,
            init: init || '', model: model || '',
            anomaly: anomaly ? '1' : '', reference: reference || ''
        });
        if (area) {
            p.set('lat_min', area.lat_min); p.set('lat_max', area.lat_max);
            p.set('lon_min', area.lon_min); p.set('lon_max', area.lon_max);
        }
        window.history.replaceState(null, '', '?' + p.toString());
        return window.location.href;
    }
    """,
    Output("mp-current-url", "children"),
    Input("mp-dataset-select", "value"),
    Input("mp-variable-select", "value"),
    Input("map-area-store", "data"),
    Input("mp-date-input", "value"),
    Input("mp-hour-select", "value"),
    Input("mp-month-select", "value"),
    Input("mp-year-input", "value"),
    Input("mp-leadtime-select", "value"),
    Input("mp-init-hour-select", "value"),
    Input("mp-cams-model-select", "value"),
    Input("mp-anomaly-switch", "checked"),
    Input("mp-reference-select", "value"),
)


@callback(
    Output("mp-dataset-select", "value"),
    Output("mp-variable-select", "value", allow_duplicate=True),
    Output("mp-date-input", "value", allow_duplicate=True),
    Output("mp-hour-select", "value"),
    Output("mp-month-select", "value"),
    Output("mp-year-input", "value"),
    Output("mp-leadtime-select", "value", allow_duplicate=True),
    Output("mp-init-hour-select", "value"),
    Output("mp-cams-model-select", "value"),
    Output("mp-anomaly-switch", "checked"),
    Output("mp-reference-select", "value"),
    Output("ma-lat_min", "value", allow_duplicate=True),
    Output("ma-lat_max", "value", allow_duplicate=True),
    Output("ma-lon_min", "value", allow_duplicate=True),
    Output("ma-lon_max", "value", allow_duplicate=True),
    Input("_mp_location", "search"),
    prevent_initial_call="initial_duplicate",
)
def restore_from_url(search):
    """Restore every control from the query string.

    The map itself is deliberately *not* re-retrieved on load: a shared link
    would otherwise queue a Copernicus request for whoever opened it. The
    controls are restored and the user presses the button.
    """
    if not search:
        return (no_update,) * 15

    params = dict(urllib.parse.parse_qsl(search.lstrip("?")))

    def value(name):
        return params[name] if params.get(name) else no_update

    def number(name):
        return float(params[name]) if params.get(name) else no_update

    return (
        value("dataset"),
        value("variable"),
        value("date"),
        value("hour"),
        value("month"),
        int(params["year"]) if params.get("year") else no_update,
        value("lead"),
        value("init"),
        value("model"),
        params.get("anomaly") == "1",
        value("reference"),
        number("lat_min"),
        number("lat_max"),
        number("lon_min"),
        number("lon_max"),
    )
