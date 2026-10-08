"""Central definitions for variables, models, climatology periods, and colors.

Nothing else in the codebase hardcodes variable names, model strings, or
colors — the UI, callbacks, and figure factories all derive from these dicts.
"""

from __future__ import annotations

import os

from utils.regions import GLOBAL_KEY, REGION_BOXES, REGION_LABELS

# Accumulation periods offered to the user, in hours.
#
# The accumulation-period control appears only for
# variables that are accumulated, and the options have to be compatible with
# the data's own time step — offering a 6 h accumulation of an hourly
# reanalysis field is fine, offering it for a 24 h-stepped one is not.
#
# Medium-range forecasts carry sub-daily steps, so 6/12/24 all resolve. The
# reanalysis map path is hourly, where the useful choices are the native hour
# and the day. Monthly means are excluded deliberately: a monthly mean is
# already a mean *rate*, and an accumulation window inside it has no meaning —
# it is re-expressed per day or per month instead (see MONTHLY_ACCUM_MODES).
ACCUM_PERIODS_FORECAST: tuple[int, ...] = (6, 12, 24)
ACCUM_PERIODS_REANALYSIS: tuple[int, ...] = (1, 24)
ACCUM_PERIOD_DEFAULT_FORECAST: int = 24
ACCUM_PERIOD_DEFAULT_REANALYSIS: int = 24

# Monthly means of an accumulated field are a mean of the hourly accumulation.
# The only honest choices are which window to express that rate over, so the
# control offers exactly that rather than pretending to accumulate.
MONTHLY_ACCUM_MODES: dict[str, dict] = {
    "per_day": {"label": "Mean per day", "hours": 24.0, "unit_suffix": "/day"},
    "per_month": {
        "label": "Month total",
        # Resolved per month from the calendar, not a fixed 30 — see
        # `cds_api.monthly_accumulation_factor`.
        "hours": None,
        "unit_suffix": " per month",
    },
}
MONTHLY_ACCUM_DEFAULT: str = "per_day"

VARIABLES: dict[str, dict] = {
    "precipitation_sum": {
        # Zero-inflated: most days are 0, so daily percentile bands collapse
        # to 0 below the median and a daily anomaly flags every wet day. Only
        # monthly, seasonal and year-to-date views are offered.
        "daily_views": False,
        "label": "Precipitation",
        "unit": "mm",
        "accumulate": True,
        "clim_period_acc": (1981, 2020),
        "clim_period_daily": (1991, 2020),
        "colorscale": "Blues",
        "positive_is_warm": None,
        "direction_labels": ("wettest", "driest"),
        "is_accumulated": True,
        "accum_periods": ACCUM_PERIODS_FORECAST,
        "accum_default": ACCUM_PERIOD_DEFAULT_FORECAST,
        "accum_unit": "mm",
        "hourly_name": "precipitation",
        "hourly_scale": 1.0,
    },
    "temperature_2m_max": {
        "label": "Max Temperature",
        # Open-Meteo's ERA5 daily max is the max of the hourly analysed 2t: the
        # route ECMWF recommends (the forecast mx2t/mn2t carry the model bias).
        "source_note": "daily max of hourly analysed 2 m temperature",
        "unit": "°C",
        "accumulate": False,
        "clim_period_daily": (1991, 2020),
        "colorscale": "RdBu_r",
        "positive_is_warm": True,
        "direction_labels": ("warmest", "coolest"),
    },
    "temperature_2m_min": {
        "label": "Min Temperature",
        # Open-Meteo's ERA5 daily min is the min of the hourly analysed 2t: the
        # route ECMWF recommends (the forecast mx2t/mn2t carry the model bias).
        "source_note": "daily min of hourly analysed 2 m temperature",
        "unit": "°C",
        "accumulate": False,
        "clim_period_daily": (1991, 2020),
        "colorscale": "RdBu_r",
        "positive_is_warm": True,
        "direction_labels": ("warmest", "coolest"),
    },
    "temperature_2m_mean": {
        "label": "Mean Temperature (2m)",
        "unit": "°C",
        "accumulate": False,
        "clim_period_daily": (1991, 2020),
        "colorscale": "RdBu_r",
        "positive_is_warm": True,
        "direction_labels": ("warmest", "coolest"),
    },
    "snowfall_sum": {
        # Zero-inflated: most days are 0, so daily percentile bands collapse
        # to 0 below the median and a daily anomaly flags every wet day. Only
        # monthly, seasonal and year-to-date views are offered.
        "daily_views": False,
        "label": "Snowfall",
        "unit": "cm",
        "accumulate": True,
        "clim_period_acc": (1981, 2020),
        "clim_period_daily": (1991, 2020),
        "colorscale": "Blues",
        "positive_is_warm": None,
        "direction_labels": ("snowiest", "least snowy"),
        "is_accumulated": True,
        "accum_periods": ACCUM_PERIODS_FORECAST,
        "accum_default": ACCUM_PERIOD_DEFAULT_FORECAST,
        "accum_unit": "cm",
        "hourly_name": "snowfall",
        "hourly_scale": 1.0,
    },
    "wind_speed_10m_max": {
        "label": "Max Wind Speed",
        "unit": "km/h",
        "accumulate": False,
        "clim_period_daily": (1991, 2020),
        "colorscale": "Purples",
        "positive_is_warm": None,
        "direction_labels": ("windiest", "calmest"),
    },
    "shortwave_radiation_sum": {
        "label": "Solar Radiation",
        "unit": "MJ/m²",
        "accumulate": True,
        "clim_period_acc": (1981, 2020),
        "clim_period_daily": (1991, 2020),
        "colorscale": "YlOrRd",
        "positive_is_warm": True,
        "direction_labels": ("sunniest", "dullest"),
        "is_accumulated": True,
        "accum_periods": ACCUM_PERIODS_FORECAST,
        "accum_default": ACCUM_PERIOD_DEFAULT_FORECAST,
        "accum_unit": "MJ/m²",
        "hourly_name": "shortwave_radiation",
        "hourly_scale": 0.0036,
    },
    "et0_fao_evapotranspiration": {
        "label": "Evapotranspiration (ET₀)",
        "unit": "mm",
        "accumulate": True,
        "clim_period_acc": (1981, 2020),
        "clim_period_daily": (1991, 2020),
        "colorscale": "Greens",
        "positive_is_warm": None,
        "direction_labels": ("highest", "lowest"),
        "is_accumulated": True,
        "accum_periods": ACCUM_PERIODS_FORECAST,
        "accum_default": ACCUM_PERIOD_DEFAULT_FORECAST,
        "accum_unit": "mm",
        "hourly_name": "et0_fao_evapotranspiration",
        "hourly_scale": 1.0,
    },
    # Meteogram only (`climate: False`): the archive API serves no pressure
    # levels, and Z500 is read at the synoptic hours, not as a daily mean.
    "geopotential_height_500hPa": {
        "label": "Geopotential height (500 hPa)",
        "unit": "m",
        "accumulate": False,
        "climate": False,
        "hourly_name": "geopotential_height_500hPa",
        "hourly_scale": 1.0,
        "synoptic_hours": (0, 12),
    },
}

MODELS: dict[str, dict] = {
    "era5_seamless": {
        "label": "ERA5-Seamless (recommended)",
        "min_year": 1940,
        "max_year": None,
        "europe_only": False,
        "warn_consistency": False,
    },
    "era5_land": {
        "label": "ERA5-Land",
        "min_year": 1950,
        "max_year": None,
        "europe_only": False,
        "warn_consistency": False,
    },
    "era5": {
        "label": "ERA5",
        "min_year": 1940,
        "max_year": None,
        "europe_only": False,
        "warn_consistency": False,
    },
    "cerra": {
        "label": "CERRA (Europe, 1985–2021)",
        "min_year": 1985,
        "max_year": 2021,
        "europe_only": True,
        "warn_consistency": False,
    },
    "ecmwf_ifs": {
        "label": "ECMWF IFS (2017–present)",
        "min_year": 2017,
        "max_year": None,
        "europe_only": False,
        "warn_consistency": False,
    },
    "ecmwf_ifs_analysis_long_window": {
        "label": "IFS Analysis Long-Window (2024–present)",
        "min_year": 2024,
        "max_year": None,
        "europe_only": False,
        "warn_consistency": False,
    },
    "best_match": {
        "label": "Best Match (ERA5 → IFS from 2017)",
        "min_year": 1940,
        "max_year": None,
        "europe_only": False,
        "warn_consistency": True,
    },
}

ENSEMBLE_MODELS: dict[str, dict] = {
    "icon_seamless": {"label": "ICON Seamless", "members": 40, "max_forecast_days": 7, "europe_only": False},
    "icon_global": {"label": "ICON Global", "members": 40, "max_forecast_days": 7, "europe_only": False},
    "icon_eu": {"label": "ICON EU", "members": 40, "max_forecast_days": 5, "europe_only": True},
    "icon_d2": {"label": "ICON D2", "members": 20, "max_forecast_days": 2, "europe_only": True},
    "gfs_seamless": {"label": "GFS Ensemble", "members": 31, "max_forecast_days": 16, "europe_only": False},
    "gfs025": {"label": "GFS Ensemble 0.25°", "members": 31, "max_forecast_days": 35, "europe_only": False, "archive_store": "gefs"},
    "gfs05": {"label": "GFS Ensemble 0.5°", "members": 31, "max_forecast_days": 35, "europe_only": False},
    "aigefs025": {"label": "AI-GEFS 0.25°", "members": 31, "max_forecast_days": 16, "europe_only": False},
    # `archive_store` names the GRID_STORES entry holding this system's past
    # runs. The key stays the same in both modes, so a model keeps its colour
    # whether the run comes from Open-Meteo (latest) or the archive (past).
    "ecmwf_ifs025": {"label": "ECMWF ENS", "members": 51, "max_forecast_days": 15, "europe_only": False, "archive_store": "ifs_ens"},
    "ecmwf_aifs025": {"label": "ECMWF AI ENS", "members": 51, "max_forecast_days": 15, "europe_only": False, "archive_store": "aifs_ens"},
    "gem_global": {"label": "GEM Global Ensemble", "members": 21, "max_forecast_days": 16, "europe_only": False},
    "bom_access_global": {"label": "BoM ACCESS-GE", "members": 18, "max_forecast_days": 10, "europe_only": False},
    "ukmo_global_ensemble_20km": {"label": "UKMO Global 20km", "members": 18, "max_forecast_days": 7, "europe_only": False},
    "ukmo_uk_ensemble_2km": {"label": "UKMO UK 2km", "members": 3, "max_forecast_days": 2, "europe_only": True},
    "meteoswiss_icon_ch1": {"label": "MeteoSwiss ICON CH1", "members": 11, "max_forecast_days": 3, "europe_only": True},
    "meteoswiss_icon_ch2": {"label": "MeteoSwiss ICON CH2", "members": 21, "max_forecast_days": 5, "europe_only": True},
}

# Qualitative base palette for ensemble systems overlaid on one meteogram.
#
# One distinct hue per entry in ENSEMBLE_MODELS, so no two systems ever share a
# color. The first eight are the original validated hues and keep their slots —
# existing systems do not repaint. The second eight were chosen by farthest-point
# search in CIELAB, maximising the minimum pairwise ΔE against the first eight,
# against each other, and under simulated deuteranopia and protanopia, subject to
# ΔE >= 40 from the #eaeaf2 plot background so nothing washes out.
#
# Extending the palette did not cost separation. Whole-set minimum pairwise ΔE:
# normal 22.0, deuteranopia 3.9, protanopia 4.6 — identical to the original
# eight-hue set, because both worst pairs are pre-existing pairs *within* the
# first eight (magenta/violet, and green/yellow at 13.8).
#
# Known limitation, unchanged by this palette: #e87ba4 (magenta) and #4a3aa7
# (violet) collapse under deuteranopia and protanopia (ΔE ~4). They are not
# separated by dash either, since both fall in cycle 0. Fixing that means
# re-picking one of the original eight hues.
#
# Re-validated 2026-10-02 for the dark and light themes: every hue keeps at
# least 3:1 contrast against both panel colours (#15263D, #FFFFFF), tested in
# tests/test_palette_contrast.py. Eleven hues kept their hue and moved only in
# lightness to the mid band that reads on both (luminance ~0.15–0.30).
#
# Do not add per-system hex values anywhere else — figure code reads SYSTEM_STYLES.
SYSTEM_PALETTE: tuple[str, ...] = (
    "#2a78d6",  # blue
    "#008300",  # green
    "#e56a98",  # magenta
    "#c48500",  # yellow
    "#1aa674",  # aqua
    "#eb6834",  # orange
    "#7060c8",  # violet
    "#e34948",  # red
    "#829f0f",  # pale lime
    "#cd7a6a",  # salmon (was maroon; re-picked 2026-10-02, >= 31 ΔE from all)
    "#6c8040",  # olive
    "#4aa50c",  # bright green
    "#b88c3d",  # tan
    "#bf3997",  # plum
    "#b15912",  # dark brown
    "#008242",  # dark teal-green
)

# Fallback for a system key missing from ENSEMBLE_MODELS — only reachable if
# validate_ensemble_selection was bypassed. Gray so it reads as "unstyled"
# rather than silently impersonating a real system's slot.
FALLBACK_SYSTEM_STYLE: dict[str, str] = {
    "color": "#888888",
    "dash": "solid",
    "symbol": "circle",
}

# Secondary encoding, cycled only if ENSEMBLE_MODELS ever outgrows the palette.
# Order matters: the first entry reads as "primary". With one hue per system the
# cycle index is always 0 today, so every system draws solid/circle and color
# alone carries the identity.
#
# Two channels are needed because they attach to different trace types:
# `go.Box.line` accepts only color and width (no dash), so the member boxes are
# separated by marker symbol, while the control `go.Scatter` uses line dash.
SYSTEM_DASHES: tuple[str, ...] = ("solid", "dash", "dot", "dashdot")
SYSTEM_SYMBOLS: tuple[str, ...] = ("circle", "diamond", "square", "triangle-up")


def _build_system_styles() -> dict[str, dict[str, str]]:
    """Assign each ensemble system a stable, unique (color, dash, symbol) slot.

    The slot index comes from the system's position in `ENSEMBLE_MODELS`, not
    from its position in the user's current selection — so deselecting a system
    never repaints the ones that remain.
    """
    styles: dict[str, dict[str, str]] = {}
    n_colors = len(SYSTEM_PALETTE)
    for index, system in enumerate(ENSEMBLE_MODELS):
        cycle = index // n_colors
        styles[system] = {
            "color": SYSTEM_PALETTE[index % n_colors],
            "dash": SYSTEM_DASHES[cycle % len(SYSTEM_DASHES)],
            "symbol": SYSTEM_SYMBOLS[cycle % len(SYSTEM_SYMBOLS)],
        }
    return styles


SYSTEM_STYLES: dict[str, dict[str, str]] = _build_system_styles()

# Archived stores draw in their Open-Meteo twin's colour, so a model looks
# the same in Latest and Past modes and on every page. Stores without a twin
# get their own hue (re-validated for the dark and light themes in B2).
# Farthest-point search in CIELAB among hues passing both themes at 3:1:
# each is >= 32 ΔE from every palette hue and from the others.
_STORE_ONLY_COLORS: dict[str, str] = {
    "ifs_control": "#3499b2",
    "aifs_single": "#b90bf9",
    "gfs": "#fa33df",
    "icon_eu": "#f90b0b",
}

# Station observations on the evaluation plots: a neutral, so it never reads as
# another model. The lightest grey still at 3:1 on the light panel, which puts
# it at ~5:1 on the dark default (tests/test_palette_contrast.py).
OBSERVED_COLOR: str = "#87919D"

# CERRA / Europe-domain spatial guard: approx lon 20W-45E, lat 20N-75N.
# Reused for any europe_only entry in MODELS or ENSEMBLE_MODELS.
CERRA_LON_MIN, CERRA_LON_MAX = -20.0, 45.0
CERRA_LAT_MIN, CERRA_LAT_MAX = 20.0, 75.0

# --------------------------------------------------------------------------
# Gridded (Icechunk / AWS Open Data) backend — regional ensemble page only.
#
# Point features stay on Open-Meteo; this store serves only the grid-shaped
# feature Open-Meteo structurally cannot (see docs/icechunk-handoff.md §1).
#
# Stores are addressed by their dynamical.org STAC id. dynamical asks users to
# resolve the Icechunk href from the catalogue rather than hardcode it: stores
# are re-versioned (GFS forecast is already at v0.2.7) and old versions go.
# `earthmover_api.store_href` looks the href up once per process and falls back
# to `fallback_href` — the address the catalogue gave on 2026-10-02 — when the
# catalogue is unreachable. The buckets also hold `vbackfill-test-*` and
# `vpara0` beside the real release, so never glob a prefix.
#
# Everything a store's runs look like — init hours, lead steps, member count —
# is read from the store itself, never declared here.
# --------------------------------------------------------------------------
DYNAMICAL_STAC_URL: str = os.environ.get("DYNAMICAL_STAC_URL", "https://stac.dynamical.org")
DYNAMICAL_STAC_TIMEOUT_S: float = 5.0

GRID_STORES: dict[str, dict] = {
    "ifs_ens": {
        "members": 51,
        "res_deg": 0.25,
        "label": "ECMWF IFS ENS",
        "kind": "ensemble",
        "family": "ECMWF",
        # Shown on every plot. The stores are models *different* from anything
        # the point pages draw — the label must say so, since a meteogram and
        # a regional box for the same place come from different runs.
        "model_label": "ECMWF IFS ENS (dynamical.org, AWS Open Data)",
        "description": "Physics-based ensemble.",
        "stac_id": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "fallback_href": (
            "s3://dynamical-ecmwf-ifs-ens/"
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk"
        ),
        "region": "us-west-2",
    },
    # The IFS ENS control (unperturbed member 0), read from the IFS ENS store:
    # `source_store` + `member` make it a view, so it shares every download
    # and cache entry with the ENS and is drawn and scored as deterministic.
    "ifs_control": {
        "members": 1,
        "res_deg": 0.25,
        "label": "ECMWF IFS Control",
        "kind": "deterministic",
        "family": "ECMWF",
        "model_label": "ECMWF IFS ENS control member (dynamical.org, AWS Open Data)",
        "description": "The unperturbed member of the IFS ensemble.",
        "source_store": "ifs_ens",
        "member": 0,
        "stac_id": "ecmwf-ifs-ens-forecast-15-day-0-25-degree",
        "fallback_href": (
            "s3://dynamical-ecmwf-ifs-ens/"
            "ecmwf-ifs-ens-forecast-15-day-0-25-degree/v0.1.0.icechunk"
        ),
        "region": "us-west-2",
    },
    "aifs_ens": {
        "members": 51,
        "res_deg": 0.25,
        "label": "ECMWF AIFS ENS",
        "kind": "ensemble",
        "family": "ECMWF",
        "model_label": "ECMWF AIFS ENS (dynamical.org, AWS Open Data)",
        # Same 0.25° grid, 51 members and (1, lead, member, 32, 32) chunks as
        # IFS ENS, so the box-size guard and read cost carry over. Fewer
        # variables: no 10 m gust.
        "description": "Machine-learned ensemble (ECMWF AIFS).",
        "stac_id": "ecmwf-aifs-ens-forecast",
        "fallback_href": (
            "s3://dynamical-ecmwf-aifs-ens/ecmwf-aifs-ens-forecast/v0.1.0.icechunk"
        ),
        "region": "us-west-2",
    },
    "aifs_single": {
        "members": 1,
        "res_deg": 0.25,
        "label": "ECMWF AIFS Single",
        "kind": "deterministic",
        "family": "ECMWF",
        "model_label": "ECMWF AIFS Single (dynamical.org, AWS Open Data)",
        "description": "Machine-learned deterministic forecast, four runs a day.",
        "stac_id": "ecmwf-aifs-single-forecast",
        "fallback_href": (
            "s3://dynamical-ecmwf-aifs-single/ecmwf-aifs-single-forecast/v0.1.0.icechunk"
        ),
        "region": "us-west-2",
    },
    # NOAA GEFS: 31 members, 0.25°, 35 days, 00 UTC only. Its chunks split
    # the lead axis (~8 days each), so a shorter horizon reads less.
    "gefs": {
        "members": 31,
        "res_deg": 0.25,
        "label": "NOAA GEFS",
        "kind": "ensemble",
        "family": "NOAA",
        "model_label": "NOAA GEFS 35-day (dynamical.org, AWS Open Data)",
        "description": "NOAA's global ensemble, 31 members, out to 35 days.",
        "stac_id": "noaa-gefs-forecast-35-day",
        "fallback_href": (
            "s3://dynamical-noaa-gefs/noaa-gefs-forecast-35-day/v0.2.0.icechunk"
        ),
        "region": "us-west-2",
    },
    "gfs": {
        "members": 1,
        "res_deg": 0.25,
        "label": "NOAA GFS",
        "kind": "deterministic",
        "family": "NOAA",
        "model_label": "NOAA GFS (dynamical.org, AWS Open Data)",
        "description": "NOAA's global deterministic forecast, four runs a day.",
        "stac_id": "noaa-gfs-forecast",
        "fallback_href": "s3://dynamical-noaa-gfs/noaa-gfs-forecast/v0.2.7.icechunk",
        "region": "us-west-2",
    },
    # Limited area: 29.5–70.5°N, 23.5°W–62.5°E at 0.0625°.
    "icon_eu": {
        "members": 1,
        "res_deg": 0.0625,
        "label": "DWD ICON-EU",
        "kind": "deterministic",
        "family": "DWD",
        "model_label": "DWD ICON-EU (dynamical.org, AWS Open Data)",
        "description": "DWD's European deterministic model, 0.0625°, five days.",
        "stac_id": "dwd-icon-eu-forecast-5-day",
        "fallback_href": (
            "s3://dynamical-dwd-icon-eu/dwd-icon-eu-forecast-5-day/v0.2.0.icechunk"
        ),
        "region": "us-west-2",
    },
}
GRID_STORE_DEFAULT: str = "ifs_ens"

for _key in GRID_STORES:
    _twin = next(
        (k for k, m in ENSEMBLE_MODELS.items() if m.get("archive_store") == _key), None
    )
    if _key in SYSTEM_STYLES:
        # Also an Open-Meteo key (icon_eu): the same model family keeps the
        # Open-Meteo system's colour; a store never repaints it.
        continue
    SYSTEM_STYLES[_key] = (
        dict(SYSTEM_STYLES[_twin])
        if _twin
        else {"color": _STORE_ONLY_COLORS[_key], "dash": "solid", "symbol": "circle"}
    )

# Deterministic models for the latest run, from Open-Meteo's forecast API
# (`api_model` ids checked against the live API, 2026-10-04). One with an
# archived store shares the store's key and colour, so it looks the same in
# Latest and Past modes; the others take their centre's ensemble colour and
# are told apart by their dashed line. Météo-France and JMA have no ensemble
# here: own hues by CIELAB farthest-point search (>= 29 ΔE from every hue,
# >= 3:1 on both themes). Horizons are each model's longest Open-Meteo reach.
DETERMINISTIC_MODELS: dict[str, dict] = {
    "ecmwf_ifs_hres": {"label": "ECMWF IFS HRES 9 km", "api_model": "ecmwf_ifs", "max_forecast_days": 15, "europe_only": False, "colour_of": "ecmwf_ifs025"},
    "aifs_single": {"label": "ECMWF AIFS Single", "api_model": "ecmwf_aifs025_single", "max_forecast_days": 15, "europe_only": False, "archive_store": "aifs_single"},
    "gfs": {"label": "NOAA GFS", "api_model": "gfs_seamless", "max_forecast_days": 16, "europe_only": False, "archive_store": "gfs"},
    "dwd_icon": {"label": "DWD ICON", "api_model": "icon_seamless", "max_forecast_days": 7, "europe_only": False, "colour_of": "icon_seamless"},
    "meteofrance": {"label": "Météo-France ARPEGE/AROME", "api_model": "meteofrance_seamless", "max_forecast_days": 4, "europe_only": False, "color": "#f2006d"},
    "ukmo": {"label": "UK Met Office", "api_model": "ukmo_seamless", "max_forecast_days": 7, "europe_only": False, "colour_of": "ukmo_global_ensemble_20km"},
    "gem": {"label": "ECCC GEM", "api_model": "gem_seamless", "max_forecast_days": 10, "europe_only": False, "colour_of": "gem_global"},
    "jma": {"label": "JMA", "api_model": "jma_seamless", "max_forecast_days": 11, "europe_only": False, "color": "#ca6df2"},
}
for _key, _meta in DETERMINISTIC_MODELS.items():
    if _key in SYSTEM_STYLES:  # an archived twin (gfs, aifs_single) already has its colour
        continue
    _color = _meta.get("color") or SYSTEM_STYLES[_meta["colour_of"]]["color"]
    SYSTEM_STYLES[_key] = {"color": _color, "dash": "dash", "symbol": "circle"}

# Point extracts of past runs (Evaluation, past-run meteograms) are kept on
# disk for good — a finished run never changes — outside the flask cache,
# whose size threshold would prune them. Mount it as a volume like .cds-cache.
POINT_CACHE_DIR: str = os.environ.get("POINT_CACHE_DIR", ".point-cache")

GRID_MODEL_LABEL: str = GRID_STORES[GRID_STORE_DEFAULT]["model_label"]

# Native grid spacing of the store, in degrees. Quoted in every subtitle:
# neither page resolves anything finer than this (handoff §12).
GRID_RESOLUTION_DEG: float = 0.25

# Box-size guard. The store is time-optimized: chunks are (1, 85, 51, 32, 32),
# so a 32x32-cell tile (8 deg) is one chunk and ~17.8 MB. A few degrees stays
# in 1-4 chunks; Europe-wide would be ~160 chunks / 2.8 GB and must never be
# reachable from a callback (handoff §4).
BBOX_MAX_SPAN_DEG: float = 10.0
BBOX_MIN_SPAN_DEG: float = GRID_RESOLUTION_DEG
BBOX_DEFAULT_SPAN_DEG: float = 5.0
# Largest standard region the area forecast reads (grid cells at 0.25°). Read
# time grows with the chunks touched: ~3 s for a 5° box, ~18 s cold for AR6
# W. & C. Europe (13k cells); ECMWF Europe (~36k) fits, hemispheres do not.
REGION_MAX_CELLS: int = 36_000
# Interactive read budget, in MB of chunks summed over every model and init a
# request touches, checked before any read. Counted in bytes, not chunks:
# chunk sizes differ ~10x between stores (IFS ENS ~18 MB, GEFS ~2 MB, GFS
# ~6 MB). Whether a shorter horizon saves download depends on the store:
# IFS ENS keeps every lead in one chunk, GEFS splits leads into ~8-day blocks.
# 1100 MB is a 10° box in two IFS-like models, or ECMWF Europe in one. The
# public site runs lower (env): every visitor's read is the same Space's time.
READ_BUDGET_MB: float = float(os.environ.get("READ_BUDGET_MB", "400"))
# Models one request may compare, on the meteogram and the area view.
MAX_MODELS: int = int(os.environ.get("MAX_MODELS", "2"))

# Variables available from the grid store. Keys are the store's own variable
# names — do not reuse the Open-Meteo VARIABLES keys, they do not match.
#
# `scale`/`offset` convert the store's native unit to `unit` for display:
# value_displayed = value_native * scale + offset. Precipitation is archived
# as a rate (kg m-2 s-1 == mm s-1) and MSLP in Pa.
GRID_VARIABLES: dict[str, dict] = {
    # GEFS/GFS only: the extreme over each step from every model time step
    # (like ERA5's mx2t/mn2t). Daily max/min are built from these where held.
    "maximum_temperature_2m": {
        "palette": "default_2t",
        "label": "Max temperature (2 m, over each step)",
        "unit": "°C",
        "native_unit": "degree_Celsius",
        "scale": 1.0,
        "offset": 0.0,
        "accepted_step_types": ("max",),
    },
    "minimum_temperature_2m": {
        "palette": "default_2t",
        "label": "Min temperature (2 m, over each step)",
        "unit": "°C",
        "native_unit": "degree_Celsius",
        "scale": 1.0,
        "offset": 0.0,
        "accepted_step_types": ("min",),
    },
    "temperature_2m": {
        # Default style for static maps (see MAP_PALETTES).
        "palette": "default_2t",
        "label": "Temperature (2 m)",
        "unit": "°C",
        "native_unit": "degree_Celsius",
        "scale": 1.0,
        "offset": 0.0,
    },
    "precipitation_surface": {
        # Default style for static maps (see MAP_PALETTES).
        "palette": "default_tp",
        "label": "Precipitation rate",
        "unit": "mm/h",
        "native_unit": "kg m-2 s-1",
        "scale": 3600.0,
        "offset": 0.0,
        # The store's own attributes say `step_type: avg`, "Average
        # precipitation rate since the previous forecast step". So each value
        # is a mean *rate* over the preceding step, not an accumulation:
        # x3600 converts mm/s to mm/h correctly and no de-accumulation is
        # needed or wanted.
        #
        # The catch is the window. Step lengths differ between stores and can
        # change within a run (IFS ENS: 3 h to T+144, 6 h after), so a point
        # late in the run may average twice as much time as one early in it.
        # `earthmover_api.step_window` derives the window from the store's
        # own `lead_time` and it is stated on the plot.
        #
        # `accepted_step_types` is checked against the store's attribute on
        # every read (`earthmover_api._check_step_type`): a new store or
        # version that archived an accumulation instead would otherwise be
        # converted wrongly and still plot plausibly.
        "accepted_step_types": ("avg",),
        # Drives the accumulation-period control on the regional page. The
        # archived rate is integrated over the chosen window; the result is a
        # total, so it carries its own unit.
        "is_accumulated": True,
        "accum_periods": ACCUM_PERIODS_FORECAST,
        "accum_default": ACCUM_PERIOD_DEFAULT_FORECAST,
        "accum_unit": "mm",
        "accum_label": "Precipitation",
    },
    "dew_point_temperature_2m": {
        # Default style for static maps (see MAP_PALETTES).
        "palette": "default_2t",
        "label": "Dew Point (2 m)",
        "unit": "°C",
        "native_unit": "degree_Celsius",
        "scale": 1.0,
        "offset": 0.0,
    },
    "wind_gust_10m": {
        # Default style for static maps (see MAP_PALETTES).
        "palette": "default_gust",
        "label": "Wind Gust (10 m)",
        "unit": "m/s",
        "native_unit": "m s-1",
        "scale": 1.0,
        "offset": 0.0,
        # Maximum gust over the preceding step, not an instantaneous value.
        "accepted_step_types": ("max",),
    },
    "wind_speed_10m": {
        # Not a store variable: √(u² + v²) per member from the raw components,
        # so an ensemble's wind speed is scored member by member.
        "derived_from": ("wind_u_10m", "wind_v_10m"),
        "palette": "default_gust",
        "label": "Wind Speed (10 m)",
        "unit": "m/s",
        "native_unit": "m s-1",
        "scale": 1.0,
        "offset": 0.0,
        "accepted_step_types": ("instant",),
    },
    "pressure_reduced_to_mean_sea_level": {
        # Default style for static maps (see MAP_PALETTES).
        "palette": "default_msl",
        "label": "Mean Sea Level Pressure",
        "unit": "hPa",
        "native_unit": "Pa",
        "scale": 0.01,
        "offset": 0.0,
    },
    "temperature_850hpa": {
        # Default style for static maps (see MAP_PALETTES).
        "palette": "default_2t",
        "label": "Temperature (850 hPa)",
        "unit": "°C",
        "native_unit": "degree_Celsius",
        "scale": 1.0,
        "offset": 0.0,
    },
    "geopotential_height_500hpa": {
        # Default style for static maps (see MAP_PALETTES).
        "palette": "default_z500",
        # Stored in metres: the default 5 dam isohypse spacing is 50 m here.
        "isoline_interval": 50.0,
        "label": "Geopotential Height (500 hPa)",
        "unit": "m",
        "native_unit": "m",
        "scale": 1.0,
        "offset": 0.0,
    },
    "total_cloud_cover_atmosphere": {
        # Default style for static maps (see MAP_PALETTES).
        "palette": "default_tcc",
        "label": "Total Cloud Cover",
        "unit": "%",
        "native_unit": "percent",
        "scale": 1.0,
        "offset": 0.0,
        # IFS/AIFS archive it instantaneous; GEFS and GFS as a mean over the
        # preceding step. Both are cloud fraction, so both are accepted, and
        # the subtitle names each store's window (`step_window`).
        "accepted_step_types": ("instant", "avg"),
    },
}

# Colors for the regional ensemble figure. Members are drawn as low-opacity
# spaghetti under a bold ensemble mean, matching the meteogram's grammar.
GRID_COLORS: dict[str, str] = {
    "member": "rgba(42,120,214,0.18)",
    "mean": "#2a78d6",
}

# Cross-page variable families. Each page reads a different catalogue
# ("daily" = VARIABLES, used by the climate and meteogram pages; "grid" =
# GRID_VARIABLES; "era5" = the ERA5 entries of CDS_VARIABLES), so a choice on
# one page is carried to the next by family, not by key. The first key per
# namespace is what a page picks on arrival. Only physically comparable
# quantities share a family: daily max wind speed is not a gust, so it has none.
VARIABLE_FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "temperature": {
        "daily": ("temperature_2m_mean", "temperature_2m_max", "temperature_2m_min"),
        "grid": ("temperature_2m",),
        "era5": ("2m_temperature",),
    },
    "precipitation": {
        "daily": ("precipitation_sum",),
        "grid": ("precipitation_surface",),
        "era5": ("total_precipitation",),
    },
    "radiation": {
        "daily": ("shortwave_radiation_sum",),
        "era5": ("surface_solar_radiation_downwards",),
    },
    "dewpoint": {
        "grid": ("dew_point_temperature_2m",),
        "era5": ("2m_dewpoint_temperature",),
    },
    "mslp": {
        "grid": ("pressure_reduced_to_mean_sea_level",),
        "era5": ("mean_sea_level_pressure",),
    },
    "cloud_cover": {
        "grid": ("total_cloud_cover_atmosphere",),
        "era5": ("total_cloud_cover",),
    },
    "gust": {
        "grid": ("wind_gust_10m",),
        "era5": ("10m_wind_gust_since_previous_post_processing",),
    },
    "z500": {
        "daily": ("geopotential_height_500hPa",),
        "grid": ("geopotential_height_500hpa",),
        "era5": ("geopotential_500",),
    },
}

# Maps-page datasets whose variables take part in the families above.
CDS_VARIABLE_NAMESPACE: dict[str, str] = {
    "era5_single_levels": "era5",
    "era5_monthly": "era5",
}

CLIM_PERIODS: list[str] = ["1961-1990", "1981-2010", "1991-2020"]

CLIM_COLORS: dict[str, str] = {
    "outer_band": "rgba(173,216,230,0.25)",
    "inner_band": "rgba(70,130,180,0.35)",
    "median": "#4575b4",
    "current": "#d73027",
    "forecast": "#f46d43",
    "record": "#762a83",
    "trend": "#1a9641",
    "extreme_hot": "#99000d",
    "extreme_cold": "#08306b",
}

URL_BASE_PATHNAME: str = os.environ.get("URL_BASE_PATHNAME", "/")

# Public-site guards (see app.py and utils/openmeteo_api.py).
# Open-Meteo's free tier is 10,000 weighted calls a day per IP, shared here by
# every visitor; requests stop at this budget, leaving headroom for retries.
OPENMETEO_DAILY_BUDGET: float = float(os.environ.get("OPENMETEO_DAILY_BUDGET", "8000"))
# Per-client limit on Dash callbacks, in flask-limiter syntax. A page load
# fires ~10-20 callbacks, so this is set for scripts, not for people clicking.
RATE_LIMIT: str = os.environ.get("RATE_LIMIT", "120 per minute;1500 per hour")
# Proxies in front of the app that append to X-Forwarded-For (Hugging Face: 1).
# The client is the address that many hops from the right; anything left of
# it was written by the client and can be forged.
TRUSTED_PROXY_HOPS: int = int(os.environ.get("TRUSTED_PROXY_HOPS", "1"))

# Nominatim's usage policy asks for an identifying User-Agent (with a contact,
# set on the deployment) and at most one request per second from the site.
NOMINATIM_USER_AGENT = os.environ.get("NOMINATIM_USER_AGENT", "preddyfast/1.0")
MAPBOX_KEY = os.environ.get("MAPBOX_KEY", "")


# --------------------------------------------------------------------------
# Copernicus CDS / ADS map backend — maps page only.
#
# A third data source, alongside Open-Meteo (points) and the Icechunk store
# (regional boxes). It is the only one that serves *global* fields and the
# only one that serves atmospheric composition, which is why it earns its own
# module (`utils/cds_api.py`) rather than being folded into either.
#
# Both stores speak the same `cdsapi` protocol and differ only in endpoint and
# token. CDS credentials come from `~/.cdsapirc` as usual; ADS needs a second
# token, which has nowhere to live in that file — hence the env vars. A
# missing ADS token is a normal, expected state and is reported in the UI, not
# raised at import time.
# --------------------------------------------------------------------------
CDS_STORES: dict[str, dict] = {
    "cds": {
        "label": "Climate Data Store",
        "short": "CDS",
        "url": os.environ.get("CDS_URL", "https://cds.climate.copernicus.eu/api"),
        "key_env": "CDS_KEY",
        "rc_file": "~/.cdsapirc",
        "register_url": "https://cds.climate.copernicus.eu/profile",
    },
    "ads": {
        "label": "Atmosphere Data Store",
        "short": "ADS",
        "url": os.environ.get("ADS_URL", "https://ads.atmosphere.copernicus.eu/api"),
        "key_env": "ADS_KEY",
        # `cdsapi` reads exactly one rc file and it points at the CDS, so the
        # ADS token has to be supplied out of band.
        "rc_file": None,
        "register_url": "https://ads.atmosphere.copernicus.eu/profile",
    },
}

# Where retrieved NetCDF lands. Requests are slow enough (seconds to tens of
# minutes in the queue) that the download is cached on disk by request hash
# and never re-requested; the in-memory cache only holds the parsed frame.
CDS_CACHE_DIR: str = os.environ.get("CDS_CACHE_DIR", ".cds-cache")

# Hard ceiling on a single retrieval, in seconds. A request still queued after
# this is abandoned and reported, rather than holding a worker thread for the
# rest of the process's life.
CDS_JOB_TIMEOUT_S: int = int(os.environ.get("CDS_JOB_TIMEOUT_S", "1800"))

# Named extents for the area picker, as (lat_min, lat_max, lon_min, lon_max).
# "Global" is the sentinel that makes `cds_api` omit the `area` key entirely —
# an explicit -180/180 box would be subsetted by MARS and can shave the
# wrap-around column.
# Area presets are the standard regions in `utils.regions` — ECMWF and WMO
# verification areas and the IPCC AR6 (CMIP6) reference regions — never
# ad-hoc boxes, so every area named in the app can be looked up.
MAP_AREA_PRESETS: dict[str, tuple[float, float, float, float]] = REGION_BOXES
MAP_AREA_PRESET_LABELS: dict[str, str] = REGION_LABELS
MAP_AREA_GLOBAL_KEY: str = GLOBAL_KEY

# Maps gallery (public site): a fixed set of ERA5 monthly maps for the newest
# month the dataset serves, retrieved and drawn offline by
# `scripts/precompute.py --maps` — visitors never queue at Copernicus. Each
# item is one map; anomalies are against `MAP_GALLERY_REFERENCE`.
MAP_GALLERY_DATASET: str = "era5_monthly"
MAP_GALLERY_REFERENCE: str = "1991-2020"
MAP_GALLERY: dict[str, dict] = {
    "t2m_anomaly_europe": {
        "label": "2 m temperature anomaly · Europe",
        "variable": "2m_temperature",
        "area": "ecmwf_europe",
        "anomaly": True,
    },
    "t2m_anomaly_global": {
        "label": "2 m temperature anomaly · Global",
        "variable": "2m_temperature",
        "area": GLOBAL_KEY,
        "anomaly": True,
    },
    "tp_anomaly_europe": {
        "label": "Precipitation anomaly · Europe",
        "variable": "total_precipitation",
        "area": "ecmwf_europe",
        "anomaly": True,
        "monthly_mode": "per_day",
    },
    "tp_anomaly_global": {
        "label": "Precipitation anomaly · Global",
        "variable": "total_precipitation",
        "area": GLOBAL_KEY,
        "anomaly": True,
        "monthly_mode": "per_day",
    },
    "t2m_europe": {
        "label": "2 m temperature · Europe",
        "variable": "2m_temperature",
        "area": "ecmwf_europe",
        "anomaly": False,
    },
}

# Natural Earth line work, bundled rather than fetched: the map figure draws
# its own coastlines because a Plotly heatmap lives on cartesian axes and
# cannot use a `geo` subplot's built-in ones. Built by
# `scripts/build_natural_earth.py`; 110 m for wide extents, 50 m for regional.
# Basemap for every map picker: the standard OpenStreetMap tiles, shown in
# grey by a CSS filter on the tile layer only (`.basemap-grey` in
# assets/shell.css), so selections and markers keep their colour. Free under
# the OSM tile usage policy, with attribution, and no API key. (CARTO's
# Positron was tried first: it now watermarks tiles "API key required".)
BASEMAP_TILE_URL: str = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
BASEMAP_ATTRIBUTION: str = (
    '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> '
    "contributors"
)

NATURAL_EARTH_FILE: str = "natural_earth_lines.npz"
NATURAL_EARTH_COARSE_ABOVE_DEG: float = 60.0
# Interactive (Plotly) map line work is drawn as data, so it cannot follow the
# theme in the browser: mid tones that read on both panels (>= 2.9:1).
MAP_COASTLINE_COLOR: str = "#8A99AA"
MAP_BORDER_COLOR: str = "#6E7F92"

# --------------------------------------------------------------------------
# CDS/ADS dataset catalogue.
#
# Each entry declares which controls the page renders (`controls`), how the
# request is assembled, and what the response means. `nc_names` lists the
# NetCDF short names a variable may come back as — the CDS request name
# (`ozone`) is almost never the name in the file (`o3_conc`), and the mapping
# has changed across CDS generations, so candidates are tried in order before
# falling back to the file's sole data variable.
#
# `scale`/`offset` convert native units to `unit` for display:
# value_displayed = value_native * scale + offset.
# --------------------------------------------------------------------------
CDS_CONTROLS: tuple[str, ...] = (
    "date",
    "hour",
    "month_year",
    "leadtime",
    "init_hour",
    "cams_model",
    "anomaly",
)

CDS_DATASETS: dict[str, dict] = {
    "era5_single_levels": {
        "label": "ERA5 hourly (single levels)",
        "store": "cds",
        "dataset": "reanalysis-era5-single-levels",
        "controls": ("date", "hour"),
        "resolution_deg": 0.25,
        "default_area": "ecmwf_europe",
        "default_variable": "2m_temperature",
        "min_date": "1940-01-01",
        # ERA5 is published ~5 days behind real time; ERA5T fills the gap but
        # the last few days are simply absent.
        "lag_days": 6,
        "note": (
            "ERA5 hourly accumulations (total precipitation, radiation) are "
            "accumulated over the hour ending at the valid time — for "
            "product_type='reanalysis' they are already per-hour and must not "
            "be differenced. The 06/18 UTC accumulation resets that catch users "
            "out belong to the ERA5 forecast and ensemble streams, not to this one."
        ),
    },
    "era5_monthly": {
        "label": "ERA5 monthly means",
        "store": "cds",
        "dataset": "reanalysis-era5-single-levels-monthly-means",
        "controls": ("month_year", "anomaly"),
        "resolution_deg": 0.25,
        "default_area": "ecmwf_europe",
        "default_variable": "2m_temperature",
        "min_date": "1940-01-01",
        "lag_days": 60,
        "product_type": "monthly_averaged_reanalysis",
        "note": (
            "Anomalies are computed against the same calendar month over the "
            "selected reference period, which is retrieved as a second request "
            "— one month × N years. Keep the area small: a global 1991–2020 "
            "reference is 30 global fields."
        ),
    },
    "cams_global": {
        "label": "CAMS global composition forecast",
        "store": "ads",
        "dataset": "cams-global-atmospheric-composition-forecasts",
        "controls": ("date", "init_hour", "leadtime"),
        "resolution_deg": 0.4,
        "default_area": "global",
        "default_variable": "total_aerosol_optical_depth_550nm",
        # The archive holds a rolling window; older dates are on the CAMS
        # reanalysis (EAC4) instead, which is a different dataset.
        "min_date": "2015-01-01",
        "lag_days": 1,
        "init_hours": ("00:00", "12:00"),
        "leadtime_hours": (0, 12, 24, 36, 48, 60, 72, 84, 96, 108, 120),
        "note": (
            "Initialised at 00 and 12 UTC. Lead-time 0 is the analysis, not a "
            "forecast — spread between a 0 h and a 120 h field is forecast "
            "error, not an aerosol trend."
        ),
    },
    "cams_europe": {
        "label": "CAMS European air quality forecast",
        "store": "ads",
        "dataset": "cams-europe-air-quality-forecasts",
        "controls": ("date", "leadtime", "cams_model"),
        "resolution_deg": 0.1,
        "default_area": "ar6_WCE",
        "default_variable": "particulate_matter_2.5um",
        "min_date": "2020-01-01",
        "lag_days": 1,
        "leadtime_hours": tuple(range(0, 97, 3)),
        "domain": (30.0, 72.0, -25.0, 45.0),
        "note": (
            "Regional models on a 0.1° grid covering Europe only. Files store "
            "longitude as 0–360 with the domain wrapping the prime meridian, "
            "and carry a length-1 'level' dimension; both are normalised on "
            "read. The time axis holds forecast lead hours from 00 UTC, not dates."
        ),
    },
    "era5_monthly_pressure": {
        "label": "ERA5 monthly means (pressure levels)",
        "store": "cds",
        "dataset": "reanalysis-era5-pressure-levels-monthly-means",
        "controls": ("month_year", "anomaly"),
        "resolution_deg": 0.25,
        "default_area": "ecmwf_nh",
        "default_variable": "geopotential_500",
        "min_date": "1940-01-01",
        "lag_days": 60,
        "product_type": "monthly_averaged_reanalysis",
        "note": (
            "Upper-air monthly means. Each variable fixes its pressure level. "
            "With the anomaly on, 500 hPa height is shaded as the anomaly with "
            "the month's own heights contoured over it."
        ),
    },
}

# Map palettes for absolute fields: the "Default" style for each parameter,
# taken from ECMWF's Magics styles (github.com/ecmwf/magics,
# share/magics/styles/ecmwf; `source` names the style). Values are that style's own: `levels` (or a
# `(start, stop, step)` `range`) are the band edges, `colors` fill the bands
# between them. One colour per edge instead of per band means the last colour
# is an open-ended band above the top level. Below the lowest level is left
# unshaded, as the source style does. Named colours are resolved from its colour
# table, and alpha is flattened over white.
#
# `isolines` draws contours instead of shading, as the default MSLP style does.
# Anomalies never use these: they stay diverging and zero-centred.
MAP_PALETTES: dict[str, dict] = {
    "default_2t": {
        "source": "sh_all_fM48t56i4",
        "range": (-48.0, 56.0, 4.0),
        "colors": (
            "#000080/#000080/#000080/#000080/#000080/#0000d9/#4000ff/"
            "#8000ff/#0080ff/#00ffff/#00ff80/#80ff00/#daff00/#ffff00/"
            "#fff500/#ffda00/#ffb000/#ff8000/#ff4f00/#ff2500/#ff0a00/"
            "#ff0000/#ff00ff/#ff00ff/#ff00ff/#ff00ff/#ff00ff"
        ),
    },
    "default_tp": {
        "source": "sh_blured_f05t300lst",
        "levels": "0.5/2/4/10/25/50/100/250",
        "colors": (
            "#00ffff/#0080ff/#0000ff/#da00ff/#ff00ff/#ff8000/#ff0000/"
            "#333333"
        ),
    },
    "default_gust": {
        "source": "sh_red_f10t70lst",
        "levels": "10/15/20/25/30/40/50/60/70",
        "colors": (
            "#ffff00/#ffdb00/#ffb600/#ff9200/#ff6d00/#ff4900/#ff2400/"
            "#ff0000"
        ),
    },
    "default_tcc": {
        "source": "transparency_grey_f0t100",
        "levels": "20/30/40/50/60/70/80/90/100",
        "colors": (
            "#f6f6f6/#eeeeee/#e5e5e5/#dcdcdc/#d4d4d4/#cbcbcb/#c2c2c2/"
            "#bababa"
        ),
    },
    "default_ssrd": {
        "source": "sh_shortwave_radiation",
        # Levels are a mean flux in W/m², converted to the displayed
        # window at draw time — see `figures.palette_levels`. The source style's
        # 0-500 W/m² suits daily and monthly means; an hourly field at
        # noon reaches ~1000, so sub-daily windows use the same 51
        # colours over 0-1000 (our extension, not part of the source style).
        "range": (0.0, 500.0, 10.0),
        "range_sub_daily": (0.0, 1000.0, 20.0),
        "flux_w_m2": True,
        "colors": (
            "#f7f7f7/#f4f4f6/#f1f1f5/#eeeef3/#ebebf2/#e8e8f1/#e4e6f0/"
            "#e1e3ef/#dee0ed/#dbddec/#d8daeb/#d4d5e8/#d0d0e6/#cdcce3/"
            "#c9c7e1/#c5c2de/#c1bedc/#bdb9d9/#bab4d7/#b6b0d4/#b2abd2/"
            "#ada5ce/#a8a0ca/#a39ac7/#9e94c3/#998fbf/#9489bb/#8f83b7/"
            "#8a7eb4/#8578b0/#8073ac/#7c6ca8/#7864a5/#745da1/#6f559e/"
            "#6b4e9a/#674696/#623f93/#5e378f/#592f8c/#542788/#502382/"
            "#4c1f7b/#481b75/#44176f/#401369/#3c0f63/#380b5d/#350657/"
            "#310351/#2d004b"
        ),
    },
    "default_aod": {
        "source": "sh_all_aod",
        "levels": "0.1/0.15/0.2/0.25/0.3/0.35/0.4/0.5/0.8/1",
        "colors": (
            "#0000f1/#004cff/#00b1ff/#29ffce/#7dff7a/#ceff29/#ffc400/"
            "#ff6800/#f10800/#800000"
        ),
    },
    "default_pm10": {
        "source": "sh_all_pm10_defra_daqi",
        "levels": "17/34/51/59/67/76/84/92/101",
        "colors": (
            "#31ff00/#31cf00/#ffff00/#ffcf00/#ff9a00/#ff6464/#ff0000/"
            "#990000/#ce30ff"
        ),
    },
    "default_pm2p5": {
        "source": "sh_all_pm2p5_defra_daqi",
        "levels": "12/24/36/42/48/54/59/65/71",
        "colors": (
            "#31ff00/#31cf00/#ffff00/#ffcf00/#ff9a00/#ff6464/#ff0000/"
            "#990000/#ce30ff"
        ),
    },
    "default_no2": {
        "source": "sh_all_no2_surface",
        "levels": "0.01/0.02/0.05/0.1/0.2/0.5/1/2/5/10/15/20/50/100/300",
        "colors": (
            "#9bc7ea/#009ee0/#007bbf/#265ba0/#269170/#6bb25b/#a8c96b/"
            "#fff49b/#f4d138/#eab538/#dd8740/#cc3533/#9b1623/#561919"
        ),
    },
    "default_o3": {
        "source": "sh_all_o3_sfc",
        "levels": "0/10/20/30/40/50/60/70/80/90/100/120/150",
        "colors": (
            "#9bc7ea/#009ee0/#007bbf/#265ba0/#269170/#6bb25b/#a8c96b/"
            "#fff49b/#f4d138/#eab538/#dd8740/#cc3533/#9b1623"
        ),
    },
    "default_so2": {
        "source": "sh_all_so2_surface",
        "levels": "0.01/0.05/0.1/0.5/1/2/5/10/20/50/100/200/500",
        "colors": (
            "#9bc7ea/#009ee0/#007bbf/#265ba0/#269170/#6bb25b/#a8c96b/"
            "#fff49b/#f4d138/#eab538/#dd8740/#cc3533/#9b1623"
        ),
    },
    "default_msl": {
        "source": "ct_blk_i5_t2",
        "isolines": 5.0,
        "color": "#000000",
    },
    "default_sst": {
        "source": "sh_sst_fM2t36i2",
        "range": (-2.0, 36.0, 2.0),
        "colors": (
            "#4a007a/#9c00ff/#cc78ff/#0000ff/#0059ff/#008cff/#0a7d00/"
            "#0abf00/#0aff00/#a19c00/#d6cf00/#ffed00/#a85400/#d66b00/"
            "#ff8500/#bf0a00/#ff0d00/#ff8580/#ffd6d4"
        ),
    },
    # z500 is read as a contour chart, not a shaded one: the default
    # `ct_blk_i5_t2` draws black isohypses every 5 dam.
    "default_z500": {
        "source": "ct_blk_i5_t2",
        "isolines": 5.0,
        "color": "#000000",
    },
}

# Per-dataset variable catalogues. Kept beside the datasets rather than inside
# them only to keep each dict readable.
CDS_VARIABLES: dict[str, dict[str, dict]] = {
    "era5_single_levels": {
        "2m_temperature": {
            "label": "2 m Temperature",
            "unit": "°C",
            "nc_names": ("t2m", "VAR_2T"),
            "scale": 1.0,
            "offset": -273.15,
            # The default fixed 4 °C bands, not a ramp stretched to the field, so
            # the same temperature is the same colour on every map. No zero
            # midpoint: absolute temperature is not a departure from 0 °C.
            "palette": "default_2t",
        },
        "2m_dewpoint_temperature": {
            "label": "2 m Dew Point",
            "unit": "°C",
            "nc_names": ("d2m", "VAR_2D"),
            "scale": 1.0,
            "offset": -273.15,
            "palette": "default_2t",
        },
        "mean_sea_level_pressure": {
            "label": "Mean Sea Level Pressure",
            "unit": "hPa",
            "nc_names": ("msl",),
            "scale": 0.01,
            "offset": 0.0,
            "palette": "default_msl",
        },
        "total_precipitation": {
            "label": "Total Precipitation",
            "unit": "mm",
            "nc_names": ("tp",),
            "scale": 1000.0,
            "expected_step_type": "accum",
            "offset": 0.0,
            "palette": "default_tp",
            # Accumulated, not instantaneous. For product_type="reanalysis"
            # each hourly field is the total over the hour *ending* at the
            # valid time, already de-accumulated by ERA5 — it must not be
            # differenced. The 06/18 UTC accumulation resets that catch users
            # out belong to the forecast and ensemble streams, not this one.
            "accumulation": "accumulated over the hour ending at the valid time",
            "is_accumulated": True,
            "accum_periods": ACCUM_PERIODS_REANALYSIS,
            "accum_default": ACCUM_PERIOD_DEFAULT_REANALYSIS,
            "accum_unit": "mm",
        },
        "total_cloud_cover": {
            "label": "Total Cloud Cover",
            "unit": "%",
            "nc_names": ("tcc",),
            "scale": 100.0,
            "offset": 0.0,
            "palette": "default_tcc",
        },
        "surface_solar_radiation_downwards": {
            "label": "Surface Solar Radiation Down",
            # Per hour, and the unit has to say so. Bare "MJ/m²" reads as a
            # total over an unstated window, which is exactly the ambiguity
            # that makes accumulated fields get misread by a factor of 24.
            "unit": "MJ/m²/h",
            "nc_names": ("ssrd",),
            "scale": 1e-6,
            "expected_step_type": "accum",
            "offset": 0.0,
            "palette": "default_ssrd",
            "accumulation": "accumulated over the hour ending at the valid time",
            "is_accumulated": True,
            "accum_periods": ACCUM_PERIODS_REANALYSIS,
            "accum_default": ACCUM_PERIOD_DEFAULT_REANALYSIS,
            "accum_unit": "MJ/m²",
        },
        "10m_wind_gust_since_previous_post_processing": {
            "label": "10 m Wind Gust",
            "unit": "m/s",
            "nc_names": ("fg10", "i10fg"),
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_gust",
        },
        "sea_surface_temperature": {
            "label": "Sea Surface Temperature",
            "unit": "°C",
            "nc_names": ("sst",),
            "scale": 1.0,
            "offset": -273.15,
            "palette": "default_sst",
        },
    },
    "era5_monthly": {
        "2m_temperature": {
            "label": "2 m Temperature",
            "unit": "°C",
            "nc_names": ("t2m", "VAR_2T"),
            "scale": 1.0,
            "offset": -273.15,
            "palette": "default_2t",
            # An anomaly is a departure either side of zero whatever the field,
            # so the diverging scale and zero midpoint apply to every variable
            # here once `anomaly` is on. See `figures.make_field_map_fig`.
            "anomaly_unit": "°C",
        },
        "mean_sea_level_pressure": {
            "label": "Mean Sea Level Pressure",
            "unit": "hPa",
            "nc_names": ("msl",),
            "scale": 0.01,
            "offset": 0.0,
            "palette": "default_msl",
            "anomaly_unit": "hPa",
        },
        "total_precipitation": {
            "label": "Total Precipitation (mean rate)",
            "unit": "mm/day",
            # The monthly product stores accumulations with GRIB stepType
            # `avgad` — "average daily accumulation" — so the archived value is
            # already metres *per day*, not per hour. x1000 alone gives mm/day.
            #
            # Verified against the file rather than assumed: the July 2023
            # Europe field means 0.00216 m, i.e. 2.16 mm/day, which matches
            # both the known climatology and the same region's hourly product.
            # An extra x24 here (the natural guess, if you think of a monthly
            # mean as averaging the hourly field) puts Europe at 52 mm/day.
            # `expected_step_type` makes the assumption checkable at read time.
            "nc_names": ("tp",),
            "scale": 1000.0,
            "expected_step_type": "avgad",
            "offset": 0.0,
            "palette": "default_tp",
            "anomaly_unit": "mm/day",
            # Dry brown, wet green-teal, white at zero — the usual precipitation
            # anomaly convention; red/blue would read as temperature.
            "anomaly_colorscale": "BrBG",
            "accumulation": "monthly mean of the hourly accumulation, expressed per day",
            # Not an accumulation *window* — a monthly mean is already a mean
            # rate, and totalling 6 h of it would be meaningless. The only
            # real choice is the window the rate is expressed over, which is
            # what MONTHLY_ACCUM_MODES offers.
            "is_monthly_rate": True,
        },
        "total_cloud_cover": {
            "label": "Total Cloud Cover",
            "unit": "%",
            "nc_names": ("tcc",),
            "scale": 100.0,
            "offset": 0.0,
            "palette": "default_tcc",
            "anomaly_unit": "%",
        },
        "surface_solar_radiation_downwards": {
            "label": "Surface Solar Radiation Down",
            "unit": "MJ/m²/day",
            # Same `avgad` convention as monthly precipitation: J/m² per day.
            "nc_names": ("ssrd",),
            "scale": 1e-6,
            "expected_step_type": "avgad",
            "offset": 0.0,
            "palette": "default_ssrd",
            "anomaly_unit": "MJ/m²/day",
            "accumulation": "monthly mean of the hourly accumulation, expressed per day",
            # Not an accumulation *window* — a monthly mean is already a mean
            # rate, and totalling 6 h of it would be meaningless. The only
            # real choice is the window the rate is expressed over, which is
            # what MONTHLY_ACCUM_MODES offers.
            "is_monthly_rate": True,
        },
        "sea_surface_temperature": {
            "label": "Sea Surface Temperature",
            "unit": "°C",
            "nc_names": ("sst",),
            "scale": 1.0,
            "offset": -273.15,
            "palette": "default_sst",
            "anomaly_unit": "°C",
            # Under sea ice ERA5 holds SST near freezing, so a year that melts
            # out early reads as a +10 °C anomaly on Arctic shelves.
            "note": "Under sea ice, ERA5 SST is held near freezing; polar anomalies "
            "there reflect ice cover, not ocean warming.",
        },
    },
    "era5_monthly_pressure": {
        "geopotential_500": {
            "label": "500 hPa Geopotential Height",
            "request_variable": "geopotential",
            "pressure_level": "500",
            "unit": "dam",
            "nc_names": ("z",),
            # m²/s² -> dam: divide by standard gravity, then by 10.
            "scale": 1.0 / 98.0665,
            "offset": 0.0,
            "palette": "default_z500",
            "anomaly_unit": "dam",
            # With the anomaly on, the month's heights are contoured over the
            # shaded anomaly — the standard forecaster's 500 hPa chart.
            "anomaly_overlay": True,
        },
        "temperature_850": {
            "label": "850 hPa Temperature",
            "request_variable": "temperature",
            "pressure_level": "850",
            "unit": "°C",
            "nc_names": ("t",),
            "scale": 1.0,
            "offset": -273.15,
            "palette": "default_2t",
            "anomaly_unit": "°C",
        },
    },
    "cams_global": {
        "total_aerosol_optical_depth_550nm": {
            "label": "Total AOD 550 nm",
            "unit": "",
            "nc_names": ("aod550",),
            "colorbar_title": "AOD 550 nm",
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_aod",
        },
        "dust_aerosol_optical_depth_550nm": {
            "label": "Dust AOD 550 nm",
            "unit": "",
            "nc_names": ("duaod550",),
            "colorbar_title": "AOD 550 nm",
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_aod",
        },
        "organic_matter_aerosol_optical_depth_550nm": {
            "label": "Organic Matter AOD 550 nm",
            "unit": "",
            "nc_names": ("omaod550",),
            "colorbar_title": "AOD 550 nm",
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_aod",
        },
        "sea_salt_aerosol_optical_depth_550nm": {
            "label": "Sea Salt AOD 550 nm",
            "unit": "",
            "nc_names": ("ssaod550",),
            "colorbar_title": "AOD 550 nm",
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_aod",
        },
        "sulphate_aerosol_optical_depth_550nm": {
            "label": "Sulphate AOD 550 nm",
            "unit": "",
            "nc_names": ("suaod550",),
            "colorbar_title": "AOD 550 nm",
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_aod",
        },
        "particulate_matter_2.5um": {
            "label": "PM2.5 (surface)",
            "unit": "µg/m³",
            "nc_names": ("pm2p5",),
            "scale": 1e9,
            "offset": 0.0,
            "palette": "default_pm2p5",
        },
        "particulate_matter_10um": {
            "label": "PM10 (surface)",
            "unit": "µg/m³",
            "nc_names": ("pm10",),
            "scale": 1e9,
            "offset": 0.0,
            "palette": "default_pm10",
        },
    },
    "cams_europe": {
        "particulate_matter_2.5um": {
            "label": "PM2.5",
            "unit": "µg/m³",
            "nc_names": ("pm2p5_conc", "pm2p5"),
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_pm2p5",
        },
        "particulate_matter_10um": {
            "label": "PM10",
            "unit": "µg/m³",
            "nc_names": ("pm10_conc", "pm10"),
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_pm10",
        },
        "ozone": {
            "label": "Ozone (surface)",
            "unit": "µg/m³",
            "nc_names": ("o3_conc", "o3"),
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_o3",
            # EU information (180) and alert (240) thresholds, drawn as
            # contour lines over the field — the numbers the forecast is
            # actually judged against.
            "thresholds": (
                (180.0, "EU information threshold"),
                (240.0, "EU alert threshold"),
            ),
        },
        "nitrogen_dioxide": {
            "label": "Nitrogen Dioxide",
            "unit": "µg/m³",
            "nc_names": ("no2_conc", "no2"),
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_no2",
            "thresholds": ((200.0, "EU hourly limit value"),),
        },
        "sulphur_dioxide": {
            "label": "Sulphur Dioxide",
            "unit": "µg/m³",
            "nc_names": ("so2_conc", "so2"),
            "scale": 1.0,
            "offset": 0.0,
            "palette": "default_so2",
            "thresholds": ((350.0, "EU hourly limit value"),),
        },
    },
}

# Grid CDS regrids a large-area monthly request to. A global 1991-2020
# reference at 0.25° is ~125 MB; at 1° it is a few MB. Only areas wider than
# REGRID_ABOVE_DEG are coarsened, and the map subtitle says when it happened.
REGRID_MONTHLY_DEG: float = 1.0
REGRID_ABOVE_DEG: float = 90.0

# ---------------------------------------------------------------------------
# Climate variability indices, all computed from ERA5 monthly means so the
# app keeps a single data source. Each is a *proxy* for the agency index of
# the same name and is labelled as such; the method is stated on the plot.
# ---------------------------------------------------------------------------

INDEX_START_YEAR: int = 1950
INDEX_GRID_DEG: float = 2.5  # source fields for the indices; plenty for box means

# The fields indices are computed from. One request (per year block) serves
# every index built on the same source.
INDEX_SOURCES: dict[str, dict] = {
    "sst": {
        "dataset": "era5_monthly",
        "variable": "sea_surface_temperature",
        "nc_names": ("sst",),
        "scale": 1.0,
        "offset": -273.15,
        "area": (-60.0, 70.0, -180.0, 180.0),
    },
    "msl": {
        "dataset": "era5_monthly",
        "variable": "mean_sea_level_pressure",
        "nc_names": ("msl",),
        "scale": 0.01,
        "offset": 0.0,
        "area": None,
    },
    "z500": {
        "dataset": "era5_monthly_pressure",
        "variable": "geopotential_500",
        "nc_names": ("z",),
        "scale": 1.0 / 98.0665,
        "offset": 0.0,
        "area": (0.0, 90.0, -180.0, 180.0),
    },
}

# Fields an index can be correlated with, over the index's fixed map area.
INDEX_TARGETS: dict[str, dict] = {
    "2m_temperature": {
        "label": "2 m temperature",
        "nc_names": ("t2m", "VAR_2T"),
        "scale": 1.0,
        "offset": -273.15,
    },
    "total_precipitation": {
        "label": "Precipitation",
        "nc_names": ("tp",),
        "scale": 1000.0,
        "offset": 0.0,
        "expected_step_type": "avgad",
    },
    "mean_sea_level_pressure": {
        "label": "Mean sea level pressure",
        "nc_names": ("msl",),
        "scale": 0.01,
        "offset": 0.0,
    },
}

# `method` picks the computation in `climatology.compute_index`:
#   box_mean      area-weighted box mean anomaly (°C)
#   box_diff      box_mean(a) - box_mean(b)
#   stations      standardised pressure, first point minus second
#   zonal_diff    standardised zonal-mean pressure, first latitude minus second
#   pna           Wallace & Gutzler 4-point standardised z500 combination
# `standardise` divides by the per-calendar-month standard deviation of the
# reference period; `smooth` is a centred running mean in months;
# `detrend` removes a linear trend over the whole record first (AMO).
CLIMATE_INDICES: dict[str, dict] = {
    "oni": {
        "label": "El Niño – Niño 3.4 (ONI-like)",
        "source": "sst",
        "method": "box_mean",
        "boxes": ((-5.0, 5.0, -170.0, -120.0),),
        "smooth": 3,
        "unit": "°C",
        "thresholds": (0.5, -0.5),
        "threshold_labels": ("El Niño", "La Niña"),
        "map_area": "global",
        "default_target": "total_precipitation",
        "definition": (
            "3-month running mean of the Niño 3.4 SST anomaly (5°S–5°N, "
            "170°–120°W) from ERA5, fixed reference period. NOAA's ONI uses "
            "ERSSTv5 and a sliding 30-year base, so values differ slightly."
        ),
    },
    "nino34": {
        "label": "Niño 3.4 SST anomaly (monthly)",
        "source": "sst",
        "method": "box_mean",
        "boxes": ((-5.0, 5.0, -170.0, -120.0),),
        "unit": "°C",
        "thresholds": (0.5, -0.5),
        "threshold_labels": ("warm", "cold"),
        "map_area": "global",
        "default_target": "2m_temperature",
        "definition": "Monthly Niño 3.4 SST anomaly (5°S–5°N, 170°–120°W), ERA5.",
    },
    "nao": {
        "label": "North Atlantic Oscillation (station)",
        "source": "msl",
        "method": "stations",
        "points": ((37.74, -25.68), (64.13, -21.93)),
        "point_labels": ("Ponta Delgada", "Reykjavík"),
        "standardise": True,
        "unit": "σ",
        "map_area": "ecmwf_europe",
        "default_target": "2m_temperature",
        "definition": (
            "Hurrell station index: standardised ERA5 MSLP at Ponta Delgada "
            "minus Reykjavík. An ERA5 proxy, not CPC's EOF-based NAO."
        ),
    },
    "ao": {
        "label": "Arctic Oscillation (zonal-mean)",
        "source": "msl",
        "method": "zonal_diff",
        "latitudes": (35.0, 65.0),
        "standardise": True,
        "unit": "σ",
        "map_area": "ecmwf_nh",
        "default_target": "2m_temperature",
        "definition": (
            "Li & Wang (2003) style: standardised zonal-mean ERA5 MSLP at 35°N "
            "minus 65°N. A proxy for CPC's EOF-based AO."
        ),
    },
    "sam": {
        "label": "Southern Annular Mode (zonal-mean)",
        "source": "msl",
        "method": "zonal_diff",
        "latitudes": (-40.0, -65.0),
        "standardise": True,
        "unit": "σ",
        "map_area": "ecmwf_sh",
        "default_target": "2m_temperature",
        "definition": (
            "Gong & Wang (1999) style: standardised zonal-mean ERA5 MSLP at "
            "40°S minus 65°S (Marshall's index uses stations). Before 1979 ERA5 "
            "has few Southern Hemisphere observations; treat early values and "
            "the long-term trend with caution."
        ),
    },
    "pna": {
        "label": "Pacific–North American pattern",
        "source": "z500",
        "method": "pna",
        "standardise": True,
        "unit": "σ",
        "map_area": "wmo_north_america",
        "default_target": "2m_temperature",
        "definition": (
            "Wallace & Gutzler (1981) 4-point index of standardised ERA5 500 hPa "
            "height: ¼[Z(20°N,160°W) − Z(45°N,165°W) + Z(55°N,115°W) − Z(30°N,85°W)]."
        ),
    },
    "iod": {
        "label": "Indian Ocean Dipole (DMI)",
        "source": "sst",
        "method": "box_diff",
        "boxes": ((-10.0, 10.0, 50.0, 70.0), (-10.0, 0.0, 90.0, 110.0)),
        "unit": "°C",
        "thresholds": (0.4, -0.4),
        "threshold_labels": ("positive IOD", "negative IOD"),
        "map_area": "ecmwf_tropics",
        "default_target": "total_precipitation",
        "definition": (
            "Dipole Mode Index: ERA5 SST anomaly, western box (10°S–10°N, "
            "50–70°E) minus eastern box (10°S–0°, 90–110°E)."
        ),
    },
    "amo": {
        "label": "Atlantic Multidecadal Variability (AMO)",
        "source": "sst",
        "method": "box_mean",
        "boxes": ((0.0, 60.0, -80.0, 0.0),),
        "detrend": True,
        "smooth": 121,
        "unit": "°C",
        "map_area": "ecmwf_nh",
        "default_target": "2m_temperature",
        "definition": (
            "Enfield et al. (2001) style: linearly detrended North Atlantic "
            "(0–60°N, 80°W–0°) ERA5 SST anomaly; bold line is the 121-month "
            "running mean."
        ),
    },
}

# Static map controls. Projections the user can pick; "auto" chooses by
# extent (polar stereographic for a hemisphere cap, Robinson for global,
# Lambert conformal for a region).
STATIC_PROJECTIONS: dict[str, str] = {
    "auto": "Automatic",
    "platecarree": "Plate Carrée (lat/lon)",
    "robinson": "Robinson",
    "mollweide": "Mollweide",
    "lambert": "Lambert conformal",
    "orthographic": "Orthographic (globe)",
    "north_polar": "North polar stereographic",
    "south_polar": "South polar stereographic",
    "mercator": "Mercator",
}
# "default" keeps the variable's default stepped levels; anything else is
# a matplotlib colormap over automatic levels.
# Levels of an absolute field: Default (fixed stepped levels, comparable
# everywhere) or fitted to the drawn area's P2–P98 (tails visible); Auto
# fits a field cut from a larger one.
LEVEL_MODES: dict[str, str] = {"auto": "Auto", "default": "Default", "fitted": "Fitted to area"}

STATIC_PALETTES: dict[str, str] = {
    "default": "Default",
    "viridis": "Viridis",
    "cividis": "Cividis",
    "turbo": "Turbo",
    "RdYlBu_r": "Red–yellow–blue",
    "RdBu_r": "Red–blue (diverging)",
    "BrBG": "Brown–green (diverging)",
    "YlGnBu": "Yellow–green–blue",
    "YlOrRd": "Yellow–orange–red",
    "Greys": "Greys",
}


# Regional models behind `cams-europe-air-quality-forecasts`. The ensemble is
# the median of the individual models and is the operational product; the
# individual models are offered because comparing them is the only way to see
# the regional spread this dataset carries.
CAMS_EUROPE_MODELS: dict[str, str] = {
    "ensemble": "Ensemble (median of all models)",
    "chimere": "CHIMERE (INERIS)",
    "emep": "EMEP (MET Norway)",
    "euradim": "EURAD-IM (Jülich)",
    "lotos": "LOTOS-EUROS (TNO)",
    "match": "MATCH (SMHI)",
    "mocage": "MOCAGE (Météo-France)",
    "silam": "SILAM (FMI)",
    "dehm": "DEHM (Aarhus)",
    "gemaq": "GEM-AQ (IEP-NRI)",
    "minni": "MINNI (ENEA)",
    "monarch": "MONARCH (BSC)",
}


# --------------------------------------------------------------------------
# Observations (Weather · Evaluation) — GHCNh hourly station data, NOAA NCEI.
#
# NOAA ISD on AWS (`noaa-isd-pds`) stopped updating in 2025; GHCNh is its
# successor. Per-station yearly PSV files, ~3 MB, refreshed daily.
# Documentation: .../hourly/doc/ghcnh_DOCUMENTATION.pdf (units: °C to tenths,
# hPa, m/s, mm; Table 3 for quality codes).
# --------------------------------------------------------------------------
GHCNH_BASE_URL: str = os.environ.get(
    "GHCNH_BASE_URL",
    "https://www.ncei.noaa.gov/oa/global-historical-climatology-network/hourly",
)
OBS_CACHE_DIR: str = os.environ.get("OBS_CACHE_DIR", ".obs-cache")
# Precomputed presets (scripts/precompute.py -> utils/precomputed.py). Dark
# theme only, default palette, 24 h frames to T+360. Wind is 10 m gust: the
# archive has no 10 m wind speed (AIFS has no gust and is skipped for it).
PRECOMPUTE_DIR: str = os.environ.get("PRECOMPUTE_DIR", ".precomputed")
# Per-run verification scores (utils/scores_db.py) of `evaluation.evaluate_point`.
# The public site neither writes nor mirrors them: its Evaluation view is the
# station scorecard, which keeps its own state in SCORECARD_DIR.
SCORES_DIR: str = os.environ.get("SCORES_DIR", f"{PRECOMPUTE_DIR}/scores")
# The station scorecard job's working state (per-init rows, frozen network,
# window). Outside PRECOMPUTE_DIR: never uploaded; the workflow keeps it with
# actions/cache. Only its aggregates file is published (utils/scorecard.py).
SCORECARD_DIR: str = os.environ.get("SCORECARD_DIR", ".scorecard")
# The public Hugging Face Dataset the scheduled job uploads to and the app
# mirrors metadata from ("user/name"); unset for a purely local run.
PRECOMPUTE_HF_REPO: str = os.environ.get("PRECOMPUTE_HF_REPO", "")
# How often (s) the app re-mirrors the Dataset's manifests and scores.
PRECOMPUTE_SYNC_S: float = float(os.environ.get("PRECOMPUTE_SYNC_S", "900"))
# Where the browser fetches precomputed images: the Dataset when there is one,
# else the app's own route (same relative paths).
PRECOMPUTE_URL: str = os.environ.get(
    "PRECOMPUTE_URL",
    f"https://huggingface.co/datasets/{PRECOMPUTE_HF_REPO}/resolve/main/"
    if PRECOMPUTE_HF_REPO
    else f"{URL_BASE_PATHNAME}precomputed/",
)
PRECOMPUTE_REGIONS: tuple[str, ...] = ("ecmwf_europe", "ecmwf_n_atlantic")
PRECOMPUTE_STORES: tuple[str, ...] = ("ifs_ens", "aifs_ens", "gefs", "ifs_control", "aifs_single")
# variable -> accumulation window (h), None for instantaneous fields
PRECOMPUTE_VARIABLES: dict[str, int | None] = {
    "temperature_2m": None,
    "precipitation_surface": 24,
    "pressure_reduced_to_mean_sea_level": None,
    "wind_gust_10m": None,
    "geopotential_height_500hpa": None,
}
# Ensemble statistics drawn per variable; deterministic models draw their field.
PRECOMPUTE_STATISTICS: dict[str, tuple[str, ...]] = {"precipitation_surface": ("mean", "range")}
PRECOMPUTE_STEP_HOURS: int = 24
PRECOMPUTE_HORIZON_HOURS: int = 360
OBS_TIMEOUT_S: float = 30.0
OBS_STATION_LIST_MAX_AGE_H: float = 24 * 7
# The current year's file grows through the day; past years never change.
OBS_CURRENT_YEAR_MAX_AGE_H: float = 6.0

# GHCNh column -> the grid variable it verifies. Units already match the grid
# display units (°C, hPa, m/s); `precipitation` is a total "nominally for the
# hour" (doc §IX), kept as a 1 h amount and only ever summed over complete
# windows.
GHCNH_FIELDS: dict[str, str] = {
    "temperature": "temperature_2m",
    "dew_point_temperature": "dew_point_temperature_2m",
    "sea_level_pressure": "pressure_reduced_to_mean_sea_level",
    "wind_gust": "wind_gust_10m",
    # Mean of the last ~10 min before the report, m/s: an instant, like the
    # model's 10 m wind.
    "wind_speed": "wind_speed_10m",
    "precipitation": "precipitation_1h",
    # SYNOP-style totals over the window ending at the report (e.g. DWD's
    # 06–06 UTC daily total at 06 UTC); verifiable where hourly reports have gaps.
    "precipitation_6_hour": "precip_total_6h",
    "precipitation_12_hour": "precip_total_12h",
    "precipitation_24_hour": "precip_total_24h",
}


# Variables the Evaluation page scores. `grid` is the GRID_VARIABLES key read
# from the archive, `obs` the observations column it is matched with, and
# `accum_hours` a forecast window compared with an observed total over the
# same window (complete hours only).
#
# 10 m gust is deliberately absent: the archive holds the maximum over each
# 3–6 h step, a station reports the gust of the last ~10 min, and scoring one
# against the other would read as a model bias that is only a definition.
# GHCNh quality codes that mean a value failed (doc §VI, Table 3). Lower-case
# letters are GHCNh's own integrated checks (spike, streak, neighbour …) on
# temperature, dew point, pressures and wind — any of them is a failure.
# Legacy codes keep their source's meaning, so they are read per source:
# sources 313–346 use 0/1/4/5/9 = passed and 2/3/6/7 = suspect/erroneous;
# sources 220–223/347/348 use 2 = suspect, 3 = erroneous, 5 = removed.
# Upper-case letters (A, U, P, I, M, R) mark values a validator accepted or
# replaced, and are kept.
GHCNH_FAILED_INTEGRATED_QC: frozenset[str] = frozenset("abcdefhiklmnoprstuvwxz")
GHCNH_FAILED_LEGACY_QC: dict[str, frozenset[str]] = {
    **{src: frozenset("2367") for src in ("313", "314", "315", "322", "335", "343", "344", "346")},
    **{src: frozenset("235") for src in ("220", "221", "222", "223", "347", "348")},
}
# A legacy code from a source not listed above: only the codes every listed
# source agrees are failures.
GHCNH_FAILED_LEGACY_DEFAULT: frozenset[str] = frozenset("23")

EVAL_VARIABLES: dict[str, dict] = {
    "temperature_2m": {"label": "2 m temperature", "grid": "temperature_2m", "obs": "temperature_2m", "accum_hours": None},
    "dew_point_temperature_2m": {"label": "2 m dew point", "grid": "dew_point_temperature_2m", "obs": "dew_point_temperature_2m", "accum_hours": None},
    "pressure_reduced_to_mean_sea_level": {"label": "Mean sea level pressure", "grid": "pressure_reduced_to_mean_sea_level", "obs": "pressure_reduced_to_mean_sea_level", "accum_hours": None},
    # Brier thresholds: 24 h totals at or above 1, 5 and 10 mm.
    "precipitation_24h": {"label": "24 h precipitation", "grid": "precipitation_surface", "obs": "precipitation_1h", "accum_hours": 24, "thresholds": (1.0, 5.0, 10.0)},
    # No station observes it: verified against T+0 references only.
    "z500": {"label": "500 hPa geopotential height", "grid": "geopotential_height_500hpa", "obs": None, "accum_hours": None},
}
# What forecasts are verified against. Gridded references are a model's T+0
# at the station's grid cell: instantaneous fields only (T+0 precipitation is
# zero). Verifying IFS and AIFS against the IFS analysis favours them.
EVAL_REFERENCES: dict[str, dict] = {
    "stations": {"label": "Stations", "description": "NOAA GHCNh hourly reports"},
    "control_t0": {"label": "IFS control T+0", "description": "pseudo-analysis, one reference for all", "store": "ifs_control"},
    "own_t0": {"label": "Own T+0", "description": "each model against its own initial state"},
    # Accumulated variables only: satellite precipitation at the station's
    # 0.1° cell, for places without hourly gauges.
    "imerg": {"label": "IMERG (satellite)", "description": "NASA GPM IMERG Late, 0.1°, gauge-adjusted", "accumulated": True},
}
# Gridded analyses read as point time series (not forecast stores: no init or
# lead). IMERG Late: half-hourly mean precipitation rate, stamped at the start
# of each half-hour, ~14 h latency, CC BY 4.0 (dynamical.org, AWS Open Data).
ANALYSIS_STORES: dict[str, dict] = {
    "imerg": {
        "label": "NASA GPM IMERG Late",
        "href": "s3://dynamical-nasa-imerg/nasa-imerg-analysis-late/v0.1.0.icechunk",
        "region": "us-west-2",
        "variable": "precipitation_surface",
        "res_deg": 0.1,
        "step_seconds": 1800,
    },
}
# Station scorecard: archived runs at GHCNh stations over a rolling window of 00 UTC inits. Stores are compared, never pooled.
SCORECARD_STORES: tuple[str, ...] = (
    "ifs_ens", "aifs_ens", "gefs", "ifs_control", "aifs_single", "gfs", "icon_eu",
)
# 14 inits, anchored on the GHCNh observation end rather than today (the feed
# lags ~6 days; see scorecard._window).
SCORECARD_DAYS: int = 14
SCORECARD_INIT_HOUR: int = 0
SCORECARD_MAX_STATIONS: int = 200
# GHCNh's ISO_CODE column is ISO 3166-1 alpha-2: Germany is "DE". ("GM" would
# be Gambia; FIPS uses GM for Germany.)
SCORECARD_COUNTRY: str = "DE"
SCORECARD_PERIODS: tuple[int, ...] = (7, 14)
# Fixed network: major European city airports with a 2026 GHCNh file and
# >= 0.96 temperature/wind completeness over 16-30 Sep 2026. London Heathrow,
# Paris Charles de Gaulle, Berlin Brandenburg, Madrid Barajas, Rome Fiumicino,
# Vienna Schwechat, Athens Eleftherios Venizelos, Istanbul Airport. Warsaw,
# Stockholm, Amsterdam and Copenhagen have no 2026 GHCNh file; the DWD national
# network reaches GHCNh about a year late, so SCORECARD_COUNTRY stations are
# unusable for a current window. Empty: the general mode (country + thinning).
SCORECARD_STATIONS: tuple[str, ...] = (
    "UKI0000EGLL", "FRI0000LFPG", "GMI0000EDDB", "SPMU0098221",
    "ITMU0016242", "AUI0000LOWW", "GRI0000LGAV", "TUI0000LTFM",
)
# Precipitation enters the card for a fixed-list station at this share of its
# 6 h windows (Heathrow 0.55; the rest 0-0.3), instead of the variable's own
# `completeness`. Not used in the general mode.
SCORECARD_FIXED_PRECIP_COMPLETENESS: float = 0.5
# Every store is scored to 15 days at most (GEFS 16-35 d is not scored).
SCORECARD_MAX_LEAD_HOURS: int = 360
# When observations count as "in" for marking an init complete: the median of
# each station's last report, over stations that reported in the
# SCORECARD_OBS_RECENT_DAYS before the newest report, minus a grace for
# stations whose data arrive late. A stopped station or one bad timestamp
# cannot move it; an init with leads past it is rescored when more arrive.
SCORECARD_OBS_RECENT_DAYS: int = 7
SCORECARD_OBS_GRACE_HOURS: int = 48
# Read budget for one station_cube call in the scheduled job (the interactive
# READ_BUDGET_MB would refuse it): the eight stations read one chunk tile
# each, ~0.4 GB per IFS ENS variable and init, and wind reads two components;
# 1000 MB still stops a mistyped network or store from reading gigabytes.
SCORECARD_READ_BUDGET_MB: float = 1000.0
# Raw store components a derived GRID_VARIABLES entry may be built from. They
# are not GRID_VARIABLES entries of their own; every scorecard store holds them
# (measured 2026-10-06 for ifs_ens, aifs_ens, aifs_single, gefs, gfs, icon_eu).
STORE_RAW_COMPONENTS: tuple[str, ...] = ("wind_u_10m", "wind_v_10m")
# Per variable: `grid` the GRID_VARIABLES key(s) read, `obs` the GHCNh-derived
# column (None: verified against `reference` only), `window_hours` the
# accumulation window (None for instants), `threshold` the event threshold in
# the display unit, `scores` what is computed, `completeness` the share of
# 00/06/12/18 UTC synoptic hours (or windows) a station must report to enter
# the card, `obs_reported` the column of station-reported window totals
# (None: not applicable), `tolerance` the obs-to-valid-time match, `reference` what verifies.
SCORECARD_VARIABLES: dict[str, dict] = {
    "temperature_2m": {
        "label": "2 m temperature",
        "unit": "°C",
        "grid": ("temperature_2m",),
        "obs": "temperature_2m",
        "obs_reported": None,
        "window_hours": None,
        "threshold": None,
        "scores": ("rmse", "mae", "bias", "crps", "rmse_bc", "mae_bc", "spread_skill"),
        "completeness": 0.90,
        "tolerance": "15min",
        "reference": "stations",
    },
    "precipitation_6h": {
        "label": "6 h precipitation",
        "unit": "mm",
        "grid": ("precipitation_surface",),
        # Policy: reported 6 h totals first (`obs_reported`), else sums of
        # complete hourly reports (`obs`).
        "obs": "precipitation_1h",
        "obs_reported": "precip_total_6h",
        "window_hours": 6,
        # 0.1 mm/h (the usual "wet hour") over a 6 h window.
        "threshold": 0.1 * 6,
        "scores": ("mae", "bias", "crps", "ets", "hss", "frequency_bias", "brier"),
        "completeness": 0.80,
        "tolerance": None,
        "reference": "stations",
    },
    "wind_speed_10m": {
        "label": "10 m wind speed",
        "unit": "m/s",
        "grid": ("wind_speed_10m",),
        "obs": "wind_speed_10m",
        "obs_reported": None,
        "window_hours": None,
        "threshold": None,
        "scores": ("rmse", "mae", "bias", "crps"),
        "completeness": 0.85,
        "tolerance": "15min",
        "reference": "stations",
    },
    # No station observes it: IFS control T+0 is the pseudo-analysis.
    "z500": {
        "label": "500 hPa geopotential height",
        "unit": "m",
        "grid": ("geopotential_height_500hpa",),
        "obs": None,
        "obs_reported": None,
        "window_hours": None,
        "threshold": None,
        "scores": ("rmse", "bias", "crps"),
        "completeness": None,
        "tolerance": None,
        "reference": "control_t0",
    },
}
# How each score is judged, shared by `relative_to` and the scorecard figures:
# "lower" / "higher" is better, a number is the ideal value (closer is better).
SCORE_DIRECTION: dict[str, str | float] = {
    "rmse": "lower",
    "mae": "lower",
    "crps": "lower",
    "brier": "lower",
    "rmse_bc": "lower",
    "mae_bc": "lower",
    "ets": "higher",
    "hss": "higher",
    "bias": 0.0,
    "frequency_bias": 1.0,
    "spread_skill": 1.0,
}
EVAL_LEAD_STEPS: tuple[int, ...] = (6, 12, 24)
EVAL_LEAD_STEP_DEFAULT: int = 24
# Thirty 00 UTC runs: a month of daily starts, all verifying at the same
# times of day (an ensemble's error depends on when its leads verify).
EVAL_RUNS_DEFAULT: int = 30
EVAL_RUNS_MAX: int = 30
EVAL_INIT_HOUR: int = 0
# Uncached chunks per Evaluation request (a point run is 1 chunk in IFS, 3 in GEFS). A cold chunk
# took up to ~5 s on 2026-10-02, so 8 keeps a request well inside gunicorn's
# 90 s; the page re-requests on an interval until every run is in.
EVAL_COLD_READS_MAX: int = 8


# --------------------------------------------------------------------------
# Themes (dark default, light) — CSS tokens, Plotly and static maps all read
# this table; see components/theme_toggle.py.
# --------------------------------------------------------------------------
# Night ground is the deep blue of an IR satellite night pass, not black;
# the accents are the three synoptic fronts the app already uses for its
# sections (warm: Climate, cold: Weather) plus occluded for focus/selection.
THEMES: dict[str, dict[str, str]] = {
    "dark": {
        "night": "#0E1B2C",
        "panel": "#15263D",
        "raise": "#1D3352",
        "ink": "#E4ECF4",
        "dim": "#8FA3BA",
        "rule": "#294160",
        "warm": "#E8584D",
        "cold": "#5B9BEA",
        "occl": "#A879E0",
        "grid": "rgba(143, 163, 186, 0.22)",
        "zero": "rgba(228, 236, 244, 0.45)",
        "coast": "#C9D4E0",
        "border": "#7D8FA6",
    },
    "light": {
        "night": "#F3F6F9",
        "panel": "#FFFFFF",
        "raise": "#E8EEF4",
        "ink": "#14263B",
        "dim": "#5A6B80",
        "rule": "#D3DCE6",
        "warm": "#C23B32",
        "cold": "#1F5FAE",
        "occl": "#7A4FB8",
        "grid": "rgba(20, 38, 59, 0.10)",
        "zero": "rgba(20, 38, 59, 0.45)",
        "coast": "#2B3A4C",
        "border": "#6B7A8C",
    },
}
