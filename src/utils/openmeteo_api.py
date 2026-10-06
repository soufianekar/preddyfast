"""All Open-Meteo API calls. No other module imports `requests` or the SDK."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from utils import status
from utils.cache import cache
from utils.climatology import accumulate_over_window, calendar_day
from utils.settings import (
    DETERMINISTIC_MODELS,
    HISTORICAL_START_YEAR,
    HISTORY_DIR,
    VARIABLES,
)

_log = logging.getLogger(__name__)

_cache_session = requests_cache.CachedSession(".cache/openmeteo", expire_after=3600)
_retry_session = retry(_cache_session, retries=5, backoff_factor=0.2)
_client = openmeteo_requests.Client(session=_retry_session)

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
# ECMWF IFS at 0.25 deg. Not `ecmwf_ifs04`: that endpoint still accepts the
# request and still returns the right number of days, but every value is
# null — so the forecast extension silently drew nothing rather than
# raising anything the page could report.
FORECAST_MODEL = "ecmwf_ifs025"
ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"


def _response_to_frame(response, variable: str) -> pd.DataFrame:
    """Convert a single Open-Meteo SDK daily response into a tidy DataFrame."""
    daily = response.Daily()
    df = pd.DataFrame(
        {
            "date": pd.date_range(
                start=pd.to_datetime(daily.Time(), unit="s", utc=True),
                end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
                freq=pd.Timedelta(seconds=daily.Interval()),
                inclusive="left",
            ),
            "value": daily.Variables(0).ValuesAsNumpy(),
        }
    )
    df["date"] = df["date"].dt.tz_localize(None)
    df["doy"] = calendar_day(df["date"])
    df["variable"] = variable
    return df


# Free tier: 10,000 weighted calls a day per IP. A 1981→yesterday daily
# series weighs ~1,190 (one per 14 days), so the history is kept on disk for
# good and only its tail is re-asked for (see `history_series`).
_DAILY_CALL_CAP = 10_000
_REVISION_DAYS = 7  # ERA5T is revised into final ERA5 within days
_calls_today: dict[str, float] = {}


def weighted_calls(days: int, n_vars: int = 1) -> float:
    """Open-Meteo's weighted cost of one location: per 14 days and per 10 variables."""
    return max(1.0, days / 14) * max(1.0, n_vars / 10)


def _count_calls(days: int, n_vars: int = 1) -> None:
    """Log this process's running weighted total; warn near the free tier's daily cap."""
    day = f"{_today():%Y-%m-%d}"
    total = _calls_today.get(day, 0.0) + weighted_calls(days, n_vars)
    _calls_today.clear()
    _calls_today[day] = total
    level = logging.WARNING if total > 0.8 * _DAILY_CALL_CAP else logging.INFO
    _log.log(level, "Open-Meteo: ~%.0f weighted calls today in this process", total)


def _today() -> pd.Timestamp:
    return pd.Timestamp.now().normalize()


def _fetch_daily_range(
    lat: float, lon: float, variable: str, model: str, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DataFrame:
    """One archive request for `[start, end]`; columns date, value."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": f"{pd.Timestamp(start):%Y-%m-%d}",
        "end_date": f"{pd.Timestamp(end):%Y-%m-%d}",
        "daily": [variable],
        "models": model,
    }
    responses = _client.weather_api(ARCHIVE_URL, params=params)
    _count_calls((pd.Timestamp(end) - pd.Timestamp(start)).days + 1)
    return _response_to_frame(responses[0], variable)[["date", "value"]]


def _history_path(lat: float, lon: float, variable: str, model: str) -> Path:
    # Keyed by the requested point (2 dp), not snapped to the reanalysis grid:
    # Open-Meteo corrects temperature to the point's own elevation.
    return Path(HISTORY_DIR) / f"{model}__{variable}__{lat:+.2f}_{lon:+.2f}.parquet"


def history_series(lat: float, lon: float, variable: str, model: str) -> pd.DataFrame:
    """The daily series from `HISTORICAL_START_YEAR` to yesterday, kept on disk.

    The first request for a place fetches everything once (~1,190 weighted
    calls). Later requests re-ask only from the earlier of the first empty
    trailing day (the reanalysis lags) and `_REVISION_DAYS` ago — about one
    call — and at most once a day per file.

    Returns
    -------
    pd.DataFrame
        Columns: date, value, doy, variable.
    """
    path = _history_path(lat, lon, variable, model)
    yesterday = _today() - pd.Timedelta("1D")
    start_all = pd.Timestamp(f"{HISTORICAL_START_YEAR}-01-01")
    if path.exists():
        stored = pd.read_parquet(path)
        refreshed = pd.Timestamp(path.stat().st_mtime, unit="s").normalize()  # see `_write`
        if refreshed >= _today():
            new = None
        else:
            values = stored["value"].to_numpy()
            filled = stored["date"][~pd.isna(values)]
            first_gap = filled.max() + pd.Timedelta("1D") if not filled.empty else start_all
            start = max(start_all, min(first_gap, yesterday - pd.Timedelta(days=_REVISION_DAYS - 1)))
            new = _fetch_daily_range(lat, lon, variable, model, start, yesterday) if start <= yesterday else None
        if new is not None:
            stored = (
                pd.concat([stored, new], ignore_index=True)
                .drop_duplicates("date", keep="last")
                .sort_values("date")
                .reset_index(drop=True)
            )
            _write(path, stored)
    else:
        status.report("Fetching the full history once for this place", detail=f"{model}, {variable}")
        stored = _fetch_daily_range(lat, lon, variable, model, start_all, yesterday)
        _write(path, stored)
    df = stored[stored["date"] <= yesterday].reset_index(drop=True)
    df["doy"] = calendar_day(df["date"])
    df["variable"] = variable
    return df


def _write(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.part")
    frame[["date", "value"]].to_parquet(tmp, index=False)
    tmp.replace(path)
    # The file's mtime is its refresh day: `history_series` asks again only
    # on a later day.
    day = _today().timestamp()
    os.utime(path, (day, day))


@cache.memoize(timeout=3600)
def fetch_historical(lat: float, lon: float, variable: str, model: str) -> pd.DataFrame:
    """Fetch the full historical/reanalysis daily series for one variable.

    Served from the on-disk history (`history_series`): the archive is asked
    for the whole series once per place, then only for new and revised days.

    Parameters
    ----------
    lat, lon : float
        Coordinates rounded to 2 decimal places by the caller.
    variable : str
        Key into `utils.settings.VARIABLES`.
    model : str
        Key into `utils.settings.MODELS`.

    Returns
    -------
    pd.DataFrame
        Columns: date, value, doy, variable.
    """
    return history_series(lat, lon, variable, model)


@cache.memoize(timeout=3600)
def fetch_forecast(lat: float, lon: float, variable: str) -> pd.DataFrame:
    """Fetch the 10-day ECMWF-IFS forecast extension for the current year.

    Returns
    -------
    pd.DataFrame
        Columns: date, value, doy, variable.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [variable],
        "forecast_days": 10,
        "models": FORECAST_MODEL,
    }
    responses = _client.weather_api(FORECAST_URL, params=params)
    return _response_to_frame(responses[0], variable)


@cache.memoize(timeout=1800)
def fetch_ensemble_accumulated(
    lat: float,
    lon: float,
    variable: str,
    model: str,
    forecast_days: int,
    accum_hours: int,
) -> pd.DataFrame:
    """Per-member totals of an accumulated variable over a fixed window.

    The daily endpoint only ever offers 24 h sums, so anything shorter has to
    come from the hourly one and be re-totalled here. Windows are built by the
    same `climatology.accumulate_over_window` the regional page uses, so a
    6 h total means the same thing on both pages: the total over the window
    *ending* at the label, with leads too short to fill one dropped.

    Parameters
    ----------
    lat, lon : float
        Coordinates rounded to 2 decimal places by the caller.
    variable : str
        Key into `utils.settings.VARIABLES`. Must have `is_accumulated`.
    model : str
        Key into `utils.settings.ENSEMBLE_MODELS`.
    forecast_days : int
        Horizon in days.
    accum_hours : int
        Accumulation window in hours.

    Returns
    -------
    pd.DataFrame
        Columns `date`, `value`, `member`, `doy`, `variable` — the same shape
        `fetch_ensemble` returns, so the meteogram figure needs no special
        case. `date` is the window's closing time.
    """
    meta = VARIABLES[variable]
    if not meta.get("is_accumulated"):
        raise ValueError(f"'{variable}' is not an accumulated variable.")

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [meta["hourly_name"]],
        "models": model,
        "forecast_days": forecast_days,
    }
    responses = _client.weather_api(ENSEMBLE_URL, params=params)
    # Convert to the display unit *before* summing: the radiation field is a
    # mean flux in W/m², and one hour of it is 0.0036 MJ/m² — summing raw
    # watts would give a number in no unit at all.
    df = _hourly_frame(responses[0].Hourly(), meta["hourly_scale"])
    # Disjoint windows: the meteogram draws one box per window, and
    # overlapping boxes would share most of their members' values.
    return _disjoint_windows(df, accum_hours, variable)


def _hourly_frame(block, scale: float, member: int | None = None) -> pd.DataFrame:
    """`date, value, member` from an SDK hourly block, values in the display unit."""
    times = pd.date_range(
        start=pd.to_datetime(block.Time(), unit="s", utc=True),
        end=pd.to_datetime(block.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=block.Interval()),
        inclusive="left",
    ).tz_localize(None)
    frames = []
    for i in range(block.VariablesLength()):
        var = block.Variables(i)
        frames.append(
            pd.DataFrame(
                {
                    "date": times,
                    "value": var.ValuesAsNumpy() * scale,
                    "member": member if member is not None else var.EnsembleMember(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _disjoint_windows(df: pd.DataFrame, accum_hours: int, variable: str) -> pd.DataFrame:
    """Disjoint `accum_hours` totals of an hourly series, labelled at their end.

    Hours count from the series' first timestamp (Open-Meteo starts at 00 UTC
    of the first day), so windows are aligned on UTC days. With a 1 h step,
    each value is its own step total (`step_type="avg"` x 1 h).
    """
    origin = df["date"].min()
    df = df.assign(lead_hours=(df["date"] - origin).dt.total_seconds() / 3600.0)
    df = accumulate_over_window(df, accum_hours, step_type="avg", disjoint=True)
    df["doy"] = calendar_day(df["date"])
    df["variable"] = variable
    return df[["date", "value", "member", "doy", "variable"]]


def _api_models(keys: tuple[str, ...]) -> list[str]:
    return [DETERMINISTIC_MODELS[k]["api_model"] for k in keys]


@cache.memoize(timeout=1800)
def fetch_deterministic(
    lat: float, lon: float, variable: str, keys: tuple[str, ...], forecast_days: int
) -> dict[str, pd.DataFrame]:
    """Daily values of one variable from several deterministic models, in one request.

    Returns
    -------
    dict[str, pd.DataFrame]
        `{key: frame}` (columns date, value, member=0, doy, variable) for each
        model that returned any value; a model with no data for this variable
        or place is left out rather than drawn as an empty line.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [variable],
        "models": _api_models(keys),
        "forecast_days": forecast_days,
    }
    responses = _client.weather_api(FORECAST_URL, params=params)
    _count_calls(forecast_days, len(keys))
    out = {}
    for key, response in zip(keys, responses, strict=True):
        df = _response_to_frame(response, variable).assign(member=0)
        if df["value"].notna().any():
            out[key] = df[["date", "value", "member", "doy", "variable"]]
    return out


@cache.memoize(timeout=1800)
def fetch_deterministic_accumulated(
    lat: float,
    lon: float,
    variable: str,
    keys: tuple[str, ...],
    forecast_days: int,
    accum_hours: int,
) -> dict[str, pd.DataFrame]:
    """Disjoint window totals from several deterministic models (see `fetch_ensemble_accumulated`)."""
    meta = VARIABLES[variable]
    if not meta.get("is_accumulated"):
        raise ValueError(f"'{variable}' is not an accumulated variable.")
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [meta["hourly_name"]],
        "models": _api_models(keys),
        "forecast_days": forecast_days,
    }
    responses = _client.weather_api(FORECAST_URL, params=params)
    _count_calls(forecast_days, len(keys))
    out = {}
    for key, response in zip(keys, responses, strict=True):
        df = _hourly_frame(response.Hourly(), meta["hourly_scale"], member=0)
        if df["value"].notna().any():
            out[key] = _disjoint_windows(df, accum_hours, variable)
    return out


def _synoptic(df: pd.DataFrame, variable: str) -> pd.DataFrame:
    """The rows at the variable's synoptic hours (instantaneous values, not means)."""
    hours = VARIABLES[variable]["synoptic_hours"]
    df = df[df["date"].dt.hour.isin(hours)].reset_index(drop=True)
    df["doy"] = calendar_day(df["date"])
    df["variable"] = variable
    return df[["date", "value", "member", "doy", "variable"]]


@cache.memoize(timeout=1800)
def fetch_ensemble_synoptic(
    lat: float, lon: float, variable: str, model: str, forecast_days: int
) -> pd.DataFrame:
    """Every member of one ensemble at the synoptic hours (e.g. Z500 at 00/12 UTC)."""
    meta = VARIABLES[variable]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [meta["hourly_name"]],
        "models": model,
        "forecast_days": forecast_days,
    }
    responses = _client.weather_api(ENSEMBLE_URL, params=params)
    return _synoptic(_hourly_frame(responses[0].Hourly(), meta["hourly_scale"]), variable)


@cache.memoize(timeout=1800)
def fetch_deterministic_synoptic(
    lat: float, lon: float, variable: str, keys: tuple[str, ...], forecast_days: int
) -> dict[str, pd.DataFrame]:
    """Several deterministic models at the synoptic hours; models without the field left out."""
    meta = VARIABLES[variable]
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [meta["hourly_name"]],
        "models": _api_models(keys),
        "forecast_days": forecast_days,
    }
    responses = _client.weather_api(FORECAST_URL, params=params)
    _count_calls(forecast_days, len(keys))
    out = {}
    for key, response in zip(keys, responses, strict=True):
        df = _synoptic(_hourly_frame(response.Hourly(), meta["hourly_scale"], member=0), variable)
        if df["value"].notna().any():
            out[key] = df
    return out


@cache.memoize(timeout=1800)
def fetch_ensemble(
    lat: float, lon: float, variable: str, model: str, forecast_days: int
) -> pd.DataFrame:
    """Fetch one ensemble system's daily member spread for one variable.

    One request = one ensemble system — the `/v1/ensemble` endpoint does not
    accept a list of `models` the way `/v1/archive` does. To overlay several
    systems on one meteogram, call this once per selected system.

    Cached for 30 minutes rather than the 1-hour default used for reanalysis
    fetches: ensemble runs are refreshed several times a day, so a shorter
    timeout keeps the meteogram from serving a stale run for too long.

    Returns
    -------
    pd.DataFrame
        Columns: date, value, member, doy, variable. `member` is 0 for the
        control run, 1..N for perturbed members — never hardcode N, it
        varies by system (see `utils.settings.ENSEMBLE_MODELS`).
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": [variable],
        "models": model,
        "forecast_days": forecast_days,
    }
    responses = _client.weather_api(ENSEMBLE_URL, params=params)
    daily = responses[0].Daily()

    dates = pd.date_range(
        start=pd.to_datetime(daily.Time(), unit="s", utc=True),
        end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=daily.Interval()),
        inclusive="left",
    ).tz_localize(None)

    frames = []
    for i in range(daily.VariablesLength()):
        var = daily.Variables(i)
        frames.append(
            pd.DataFrame({"date": dates, "value": var.ValuesAsNumpy(), "member": var.EnsembleMember()})
        )
    df = pd.concat(frames, ignore_index=True)
    df["doy"] = calendar_day(df["date"])
    df["variable"] = variable
    return df
