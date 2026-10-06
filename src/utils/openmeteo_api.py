"""All Open-Meteo API calls. No other module imports `requests` or the SDK."""

from __future__ import annotations

import logging

import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

from utils.cache import cache
from utils.climatology import accumulate_over_window, calendar_day
from utils.settings import (
    DETERMINISTIC_MODELS,
    OPENMETEO_DAILY_BUDGET,
    VARIABLES,
)

_log = logging.getLogger(__name__)

_cache_session = requests_cache.CachedSession(".cache/openmeteo", expire_after=3600)
_retry_session = retry(_cache_session, retries=5, backoff_factor=0.2)
_client = openmeteo_requests.Client(session=_retry_session)

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
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


class OpenMeteoQuotaError(RuntimeError):
    """The site's share of Open-Meteo's free daily calls is spent."""


# Free tier: 10,000 weighted calls a day per IP — and on a public site every
# visitor shares the server's IP. The count lives in the shared cache so all
# workers see one total; once it reaches `OPENMETEO_DAILY_BUDGET` requests are
# refused until the next UTC day rather than letting Open-Meteo start failing.
_QUOTA_KEY = "openmeteo-weighted-calls:"


def weighted_calls(days: int, n_series: int = 1) -> float:
    """Open-Meteo's weighted cost of one location and model: per 14 days, per 10 series.

    For the ensemble API every member counts as a series: Open-Meteo's own
    example (one variable, ICON Seamless's 40 members, 7 days) costs 4.0.
    """
    return max(1.0, days / 14) * max(1.0, n_series / 10)


def calls_today() -> float:
    """Weighted calls this site has made today (UTC), across workers."""
    return float(cache.get(_QUOTA_KEY + f"{_today():%Y-%m-%d}") or 0.0)


def _check_quota() -> None:
    if calls_today() >= OPENMETEO_DAILY_BUDGET:
        raise OpenMeteoQuotaError(
            "This public demo has used today's share of Open-Meteo's free API "
            "calls. Forecasts already fetched are still shown; new ones resume "
            "after 00 UTC. Past runs (from the archive) are not affected."
        )


def _series(response) -> int:
    block = response.Hourly() or response.Daily()
    return block.VariablesLength() if block is not None else 1


def _get(url: str, params: dict, days: int) -> list:
    """One Open-Meteo request: refused over budget, counted from what came back.

    The weight is summed per response (one per model), with the member count
    read from the response — never assumed per system.
    """
    _check_quota()
    responses = _client.weather_api(url, params=params)
    weight = sum(weighted_calls(days, _series(r)) for r in responses)
    key = _QUOTA_KEY + f"{_today():%Y-%m-%d}"
    total = calls_today() + weight  # read-modify-write: approximate across workers
    cache.set(key, total, timeout=2 * 86_400)
    level = logging.WARNING if total > 0.8 * OPENMETEO_DAILY_BUDGET else logging.INFO
    _log.log(level, "Open-Meteo: +%.1f, ~%.0f weighted calls today", weight, total)
    return responses


def _today() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").normalize()


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
    responses = _get(ENSEMBLE_URL, params, forecast_days)
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
    responses = _get(FORECAST_URL, params, forecast_days)
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
    responses = _get(FORECAST_URL, params, forecast_days)
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
    responses = _get(ENSEMBLE_URL, params, forecast_days)
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
    responses = _get(FORECAST_URL, params, forecast_days)
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
    responses = _get(ENSEMBLE_URL, params, forecast_days)
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
