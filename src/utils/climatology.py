"""Percentile computation, accumulation, and trend fitting.

No API calls and no Plotly here — pure pandas/numpy transforms only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from utils.settings import (
    CERRA_LAT_MAX,
    CERRA_LAT_MIN,
    CERRA_LON_MAX,
    CERRA_LON_MIN,
    DETERMINISTIC_MODELS,
    ENSEMBLE_MODELS,
    MODELS,
)

PCT_LEVELS = [0.1, 0.25, 0.5, 0.75, 0.9]
PCT_COLUMNS = ["p10", "p25", "p50", "p75", "p90"]
# Days are indexed on a 365-day calendar (see `calendar_day`), so 1 March is
# day 60 in every year and a leap year does not shift the rest of its dates.
FULL_DOY_INDEX = pd.RangeIndex(1, 366)
# Half-width, in days, of the window pooled around each calendar day for the
# daily percentiles: ±7 days x 30 years ≈ 450 samples per day.
POOL_HALF_WIDTH = 7

SEASONS: dict[str, tuple[int, int, int]] = {
    "DJF": (12, 1, 2),
    "MAM": (3, 4, 5),
    "JJA": (6, 7, 8),
    "SON": (9, 10, 11),
}


def calendar_day(dates: pd.Series) -> pd.Series:
    """Day of a 365-day calendar: 29 Feb shares day 59 with 28 Feb.

    `dayofyear` makes 1 March day 61 in leap years and 60 otherwise, so a
    climatology grouped by it compares each date after February with its
    neighbour in a quarter of the years — and a "same window" ranking in a
    leap year includes one extra day. Folding 29 February into 28 February
    keeps every other date on the same day number every year; the leap day's
    value is not dropped, it is pooled with 28 February.
    """
    doy = dates.dt.dayofyear
    shifted = dates.dt.is_leap_year & (doy >= 60)
    return doy - shifted.astype(int)


def calendar_date(year: int, day: int) -> pd.Timestamp:
    """The date of a 365-day calendar day in `year` (day 60 is always 1 March)."""
    ref = pd.Timestamp(2001, 1, 1) + pd.Timedelta(days=int(day) - 1)
    return pd.Timestamp(year, ref.month, ref.day)


def _full_calendar(pct: pd.DataFrame) -> pd.DataFrame:
    """Every calendar day 1-365 present; a day with no data takes its neighbour."""
    return pct.reindex(FULL_DOY_INDEX).ffill().bfill()


def accumulate(values: pd.Series) -> pd.Series:
    """Running total that stops at the first gap instead of stepping over it.

    `Series.cumsum()` defaults to `skipna=True`, which for an accumulation is
    the wrong kind of forgiving: a missing day emits NaN at its own position
    but the running total simply carries on without it, so every later value
    is short by the missing day and nothing says so. Over the reanalysis lag
    at the end of the current year that is a few days of precipitation
    quietly deleted from the year-to-date total.

    Truncating instead makes the gap visible. The curve is only defined while
    the record behind it is complete, so the total runs to the last day before
    the first gap and is NaN from there on — a line that stops, rather than a
    line that lies.

    Parameters
    ----------
    values : pd.Series
        Daily values in date order. Must already be sorted.

    Returns
    -------
    pd.Series
        Cumulative total up to the first NaN, NaN at and after it. Same index.
    """
    running = values.cumsum()
    first_gap = values.isna()
    if not first_gap.any():
        return running
    return running.where(~first_gap.cummax())


def accumulate_over_window(
    df: pd.DataFrame,
    period_hours: int,
    *,
    lead_col: str = "lead_hours",
    value_col: str = "value",
    group_cols: tuple[str, ...] = ("member",),
    step_type: str = "avg",
    disjoint: bool = False,
) -> pd.DataFrame:
    """Re-express a per-step forecast series as totals over a trailing window.

    By convention the value at lead `L` for an accumulation
    period `P` is the total over `(L-P, L]`, obtained as the difference of two
    accumulations-since-init, `cum(L) - cum(L-P)`. A lead whose `L-P` is not
    itself a step in the run cannot carry a complete period and is dropped
    rather than reported short — which is also what makes `L < P` disappear,
    with no separate rule needed.

    Two source conventions are handled, because getting this backwards is the
    classic way to be wrong by a factor of the step length:

    * `step_type="avg"` — each value is the mean *rate* over the preceding
      step (the Icechunk store's `precipitation_surface`). The accumulation
      since init is the duration-weighted running total, so a run whose steps
      widen from 3 h to 6 h partway through still integrates correctly.
    * `step_type="accum"` — each value is already the total since init, and
      differencing alone is enough.

    Parameters
    ----------
    df : pd.DataFrame
        Tidy frame with a lead column, a value column, and the grouping
        columns. Not modified.
    period_hours : int
        Accumulation window in hours.
    lead_col, value_col : str
        Column names for forecast lead (in hours) and value.
    group_cols : tuple[str, ...]
        Columns identifying an independent series — ensemble member, and
        anything else that must not be summed across.
    step_type : str
        `"avg"` or `"accum"`, per above.
    disjoint : bool
        When False (the default) a window ends at every available step, so
        successive windows overlap — a rolling total, which reads well as a
        line. When True only non-overlapping windows are kept, which is what
        a box plot needs: overlapping boxes share most of their data and
        invite the reader to see structure that is just the window sliding.

    Returns
    -------
    pd.DataFrame
        Same columns, one row per (group, lead) that carries a complete
        window. `value_col` holds the window total.
    """
    if period_hours <= 0:
        raise ValueError("Accumulation period must be positive.")

    out = df.sort_values([*group_cols, lead_col]).copy()
    keys = list(group_cols)

    if step_type == "avg":
        # Step duration from the lead axis itself, never assumed: the run's
        # step widens mid-forecast and a hardcoded interval would silently
        # halve or double everything past the change.
        #
        # The first row's duration is measured from the init, not from the
        # previous row — there isn't one. Filling it with 0 would give the
        # opening step zero weight and quietly discard the first few hours of
        # every run. When lead 0 *is* present this still yields 0, correctly.
        leads = out.groupby(keys, sort=False)[lead_col]
        out["_duration"] = leads.diff().fillna(out[lead_col])
        out["_amount"] = out[value_col] * out["_duration"]
        out["_cum"] = out.groupby(keys, sort=False)["_amount"].cumsum()
    elif step_type == "accum":
        out["_cum"] = out[value_col]
    else:
        raise ValueError(f"Unknown step_type '{step_type}'.")

    # Look up cum(L - P) by joining the frame to itself on the shifted lead.
    # An absent L-P yields NaN, which is precisely the incomplete-window case.
    start = out[[*keys, lead_col, "_cum"]].copy()

    # The accumulation since init is zero at init by definition. That origin is
    # often missing from the data — a "mean since previous step" field has no
    # value at lead 0 and gets dropped as NaN upstream — and without it the
    # first complete window has nothing to subtract from and is lost. Adding it
    # back is what makes a P-hour accumulation start at T+P rather than T+2P.
    origin = out.groupby(keys, sort=False, as_index=False).first()[keys].copy()
    origin[lead_col] = 0.0
    origin["_cum"] = 0.0
    present = set(map(tuple, out.loc[out[lead_col] == 0, keys].to_numpy().tolist()))
    if present:
        mask = [tuple(row) not in present for row in origin[keys].to_numpy().tolist()]
        origin = origin[mask]
    start = pd.concat([origin, start], ignore_index=True)

    start[lead_col] = start[lead_col] + period_hours
    start = start.rename(columns={"_cum": "_cum_start"})

    merged = out.merge(start, on=[*keys, lead_col], how="left")
    merged[value_col] = merged["_cum"] - merged["_cum_start"]
    merged = merged.dropna(subset=[value_col])

    if disjoint:
        # Tile from the first complete window rather than from lead 0, so the
        # bins line up with the first window that actually exists.
        first = merged[lead_col].min()
        merged = merged[(merged[lead_col] - first) % period_hours == 0]

    return merged.drop(
        columns=[c for c in ("_duration", "_amount", "_cum", "_cum_start") if c in merged]
    ).reset_index(drop=True)


def daily_extreme(df: pd.DataFrame, kind: str, step_type: str) -> pd.DataFrame:
    """Daily (UTC) maximum or minimum per member, from complete days only.

    Two sources, following ECMWF's own distinction:

    * `step_type="max"`/`"min"` — the model's extreme over each step, taken
      from every model time step (GEFS/GFS `maximum/minimum_temperature_2m`,
      like ERA5's `mx2t`/`mn2t`). A value stamped T covers `(T - step, T]`, so
      a UTC day is the windows stamped (00:00, 24:00]; complete when those
      windows add up to 24 h.
    * `step_type="instant"` — instantaneous samples (IFS/AIFS 2 m temperature,
      3- or 6-hourly). The day's value is the extreme of the samples stamped in
      [00:00, 24:00), which misses the peak between samples: the caller says so.
      Complete when the first sample is at 00:00 and no gap in the day is
      longer than the gap to the next midnight.

    Parameters
    ----------
    df : pd.DataFrame
        `date` (valid time), `lead_hours`, `member`, `value`.
    kind : str
        "max" or "min".
    step_type : str
        "max", "min" (window extremes) or "instant" (samples).

    Returns
    -------
    pd.DataFrame
        `date` (the day, 00:00), `member`, `value`.
    """
    have = df.dropna(subset=["value"]).sort_values(["member", "lead_hours"]).copy()
    if step_type in ("max", "min"):
        have = have[have["lead_hours"] > 0]
        have["day"] = (have["date"] - pd.Timedelta("1ns")).dt.normalize()
        have["dur"] = have.groupby("member")["lead_hours"].diff().fillna(have["lead_hours"])
        grouped = have.groupby(["member", "day"])
        complete = grouped["dur"].sum().round(6) == 24.0
    else:
        have["day"] = have["date"].dt.normalize()
        have["hour"] = (have["date"] - have["day"]).dt.total_seconds() / 3600.0

        def covers(hours: pd.Series) -> bool:
            h = np.sort(hours.to_numpy())
            inner = np.diff(h)
            if not inner.size:  # one sample (a run's last, at 00 UTC) is not a day
                return False
            return h[0] == 0.0 and (24.0 - h[-1]) <= inner.max()

        grouped = have.groupby(["member", "day"])
        complete = grouped["hour"].apply(covers)
    values = grouped["value"].agg(kind)
    out = values[complete.reindex(values.index, fill_value=False)].reset_index()
    return out.rename(columns={"day": "date"})[["date", "member", "value"]]


def compute_daily_pct(df: pd.DataFrame, clim_start: int, clim_end: int) -> pd.DataFrame:
    """Daily percentile bands from a pooled window around each calendar day.

    Each day's percentiles are taken over every value within ±`POOL_HALF_WIDTH`
    days of it in every reference year (the ETCCDI approach) — about 450
    samples rather than 30. Averaging 30-sample percentiles across
    neighbouring days, as this used to, smooths the curves but not the
    sampling noise in each tail. The window wraps across the year end.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns `date`, `value`, `doy` (a 365-day calendar day).
    clim_start, clim_end : int
        Inclusive reference period years, e.g. 1991, 2020.

    Returns
    -------
    pd.DataFrame
        Indexed by calendar day (1-365), columns p10/p25/p50/p75/p90.
    """
    clim = df[df["date"].dt.year.between(clim_start, clim_end)].dropna(subset=["value"])
    days = clim["doy"].to_numpy()
    values = clim["value"].to_numpy()
    pooled = pd.DataFrame(
        {
            "doy": np.concatenate(
                [
                    (days - 1 + offset) % len(FULL_DOY_INDEX) + 1
                    for offset in range(-POOL_HALF_WIDTH, POOL_HALF_WIDTH + 1)
                ]
            ),
            "value": np.tile(values, 2 * POOL_HALF_WIDTH + 1),
        }
    )
    pct = pooled.groupby("doy")["value"].quantile(PCT_LEVELS).unstack()
    pct.columns = PCT_COLUMNS
    return _full_calendar(pct)


def compute_acc_pct(df: pd.DataFrame, clim_start: int, clim_end: int) -> pd.DataFrame:
    """Compute YTD-accumulation percentile bands over a climatology reference period.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns `date`, `value`, `doy`.
    clim_start, clim_end : int
        Inclusive reference period years, e.g. 1981, 2020.

    Returns
    -------
    pd.DataFrame
        Indexed by DOY (1-366), columns p10/p25/p50/p75/p90 of cumulative value.
    """
    clim_acc = df[df["date"].dt.year.between(clim_start, clim_end)].sort_values("date").copy()
    # Per year, and truncating at the first gap: a year with a missing day in
    # March would otherwise contribute a total that is short by that day for
    # every DOY after it, dragging the whole envelope low with no indication.
    # A truncated year simply stops contributing past its gap, so each DOY's
    # percentiles are taken over the years that are actually complete there.
    year = clim_acc["date"].dt.year
    clim_acc["ytd"] = clim_acc.groupby(year)["value"].transform(accumulate)
    # 28 and 29 February share calendar day 59; the later row carries the
    # total through both, so a leap year's day 59 includes its leap day.
    per_day = clim_acc.groupby([year, clim_acc["doy"]])["ytd"].last().reset_index()
    acc_pct = per_day.groupby("doy")["ytd"].quantile(PCT_LEVELS).unstack()
    acc_pct.columns = PCT_COLUMNS
    return _full_calendar(acc_pct)


def compute_member_pct(df: pd.DataFrame) -> pd.DataFrame:
    """Percentile spread across ensemble members at each valid time.

    Intended for the regional page, whose input frame has already been
    reduced over space *per member* by `earthmover_api.fetch_bbox_ensemble`.
    Each member is therefore still one physically self-consistent
    realisation, and these percentiles describe the spread of the regional
    average — not a per-gridpoint patchwork. The order matters; see
    `docs/icechunk-handoff.md` §11.

    No smoothing is applied, unlike `compute_daily_pct`: a forecast spread is
    not a climatology and each lead time is a distribution in its own right.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns `valid_time`, `member`, `value`.

    Returns
    -------
    pd.DataFrame
        Indexed by `valid_time`, columns p10/p25/p50/p75/p90.
    """
    pct = df.groupby("valid_time")["value"].quantile(PCT_LEVELS).unstack()
    pct.columns = PCT_COLUMNS
    return pct.sort_index()


def compute_trend(
    df: pd.DataFrame, accumulate: bool = False
) -> tuple[pd.Series, pd.Series, float]:
    """Linear trend of annual totals (accumulated variables) or annual means.

    Only complete years count. A partial current year would drag an annual
    total down by the months still to come, and even an annual mean by the
    seasonal cycle it has not yet completed.

    Returns
    -------
    tuple[pd.Series, pd.Series, float]
        Annual values, fitted trend (same index), slope per decade.
    """
    year = df["date"].dt.year
    counts = df.groupby(year)["value"].count()
    days = pd.Series(
        [366 if pd.Timestamp(y, 1, 1).is_leap_year else 365 for y in counts.index],
        index=counts.index,
    )
    complete = counts.index[counts.to_numpy() == days.to_numpy()]
    grouped = df[year.isin(complete)].groupby(year[year.isin(complete)])["value"]
    annual = (grouped.sum() if accumulate else grouped.mean()).dropna()
    if len(annual) < 3:
        return annual, annual * np.nan, float("nan")
    slope, intercept, *_ = stats.linregress(annual.index, annual.values)
    trend = pd.Series(intercept + slope * annual.index, index=annual.index)
    return annual, trend, slope * 10


def _reduce(values: pd.Series, accumulate: bool) -> float:
    return float(values.sum() if accumulate else values.mean())


def compute_monthly(
    df: pd.DataFrame, year: int, reference: tuple[int, int], accumulate: bool
) -> pd.DataFrame:
    """One year's monthly totals (or means) against the reference distribution.

    A month still in progress — or ending in the reanalysis lag — is compared
    like-for-like: the reference years are cut to the same days of the month,
    never to the whole month. Reference years with a gap in the window are
    left out rather than counted short. Months not yet observed still carry
    their climatology, so the chart shows what the rest of the year normally
    brings.

    Returns
    -------
    pd.DataFrame
        Indexed by month 1-12: `value`, `n_days`, `days_in_month`, `complete`,
        p10/p25/p50/p75/p90, `mean`, `anomaly`, `pct_of_normal` (accumulated
        variables only), `percentile` (share of reference years below), and
        `n_years`.
    """
    data = df.assign(
        year=df["date"].dt.year, month=df["date"].dt.month, day=df["date"].dt.day
    )
    this = data[data["year"] == year].dropna(subset=["value"])
    ref = data[data["year"].between(*reference)]

    rows = []
    for month in range(1, 13):
        days_in_month = pd.Timestamp(year, month, 1).days_in_month
        obs = this[this["month"] == month]
        # The window is the days observed from the 1st without a break.
        last_day = int(obs["day"].max()) if not obs.empty else days_in_month
        contiguous = not obs.empty and len(obs) == last_day
        value = _reduce(obs["value"], accumulate) if contiguous else np.nan

        window = ref[(ref["month"] == month) & (ref["day"] <= last_day)]
        per_year = window.groupby("year")["value"].agg(["count", "sum", "mean"])
        expected = np.array(
            [min(last_day, pd.Timestamp(y, month, 1).days_in_month) for y in per_year.index]
        )
        per_year = per_year[per_year["count"].to_numpy() == expected]
        ref_values = per_year["sum"] if accumulate else per_year["mean"]

        row = {
            "month": month,
            "value": value,
            "n_days": last_day if contiguous else 0,
            "days_in_month": days_in_month,
            "complete": bool(contiguous and last_day == days_in_month),
            "mean": ref_values.mean(),
            "n_years": len(ref_values),
        }
        for level, column in zip(PCT_LEVELS, PCT_COLUMNS, strict=True):
            row[column] = ref_values.quantile(level) if len(ref_values) else np.nan
        row["anomaly"] = value - row["mean"]
        row["pct_of_normal"] = (
            100 * value / row["mean"] if accumulate and row["mean"] > 0 else np.nan
        )
        row["percentile"] = (
            100 * float((ref_values < value).mean()) if len(ref_values) else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows).set_index("month")


def compute_seasonal(
    df: pd.DataFrame, season: str, reference: tuple[int, int], accumulate: bool
) -> tuple[pd.DataFrame, dict]:
    """Every year's value for one season, as an anomaly against the reference.

    DJF belongs to the year of its January and February (DJF 2025 is
    December 2024 to February 2025). Complete seasons are compared with the
    reference mean over complete reference seasons. The latest season, if
    still in progress, is compared like-for-like with the same number of
    days from the start of the season in each reference year.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        Per season-year: `value`, `n_days`, `season_days`, `complete`,
        `reference_mean`, `anomaly`, `pct_of_normal`. The dict carries the
        reference `mean` and `std` (complete seasons), `n_years`, and the
        trend over all complete seasons in `slope_per_decade`.
    """
    months = SEASONS[season]
    data = df[df["date"].dt.month.isin(months)].copy()
    data["season_year"] = data["date"].dt.year + (
        (data["date"].dt.month == 12) & (season == "DJF")
    ).astype(int)
    start_month = months[0]

    def season_start(season_year: int) -> pd.Timestamp:
        start_year = season_year - 1 if season == "DJF" else season_year
        return pd.Timestamp(start_year, start_month, 1)

    def season_length(season_year: int) -> int:
        start = season_start(season_year)
        return int((start + pd.DateOffset(months=3) - start).days)

    data["position"] = [
        (d - season_start(y)).days for d, y in zip(data["date"], data["season_year"], strict=True)
    ]
    valid = data.dropna(subset=["value"])

    records = {}
    for season_year, group in valid.groupby("season_year"):
        n = int(group["position"].max()) + 1
        contiguous = len(group) == n and int(group["position"].min()) == 0
        records[season_year] = {
            "value": _reduce(group["value"], accumulate) if contiguous else np.nan,
            "n_days": n if contiguous else 0,
            "season_days": season_length(season_year),
        }
    table = pd.DataFrame.from_dict(records, orient="index").sort_index()
    table["complete"] = table["n_days"] == table["season_days"]

    in_ref = table.index.to_series().between(*reference)
    ref_complete = table[in_ref & table["complete"]]["value"]
    stats_out = {
        "mean": float(ref_complete.mean()),
        "std": float(ref_complete.std(ddof=1)),
        "n_years": len(ref_complete),
    }

    table["reference_mean"] = stats_out["mean"]
    partial = table.index[~table["complete"] & (table["n_days"] > 0)]
    for season_year in partial:
        n = int(table.at[season_year, "n_days"])
        window = valid[(valid["position"] < n)]
        per_year = window.groupby("season_year")["value"].agg(["count", "sum", "mean"])
        per_year = per_year[per_year.index.to_series().between(*reference)]
        per_year = per_year[per_year["count"] == n]
        ref_values = per_year["sum"] if accumulate else per_year["mean"]
        table.at[season_year, "reference_mean"] = float(ref_values.mean())

    table["anomaly"] = table["value"] - table["reference_mean"]
    table["pct_of_normal"] = (
        100 * table["value"] / table["reference_mean"] if accumulate else np.nan
    )

    complete = table[table["complete"]]["value"].dropna()
    if len(complete) >= 3:
        slope, *_ = stats.linregress(complete.index.astype(float), complete.to_numpy())
        stats_out["slope_per_decade"] = float(slope * 10)
    else:
        stats_out["slope_per_decade"] = float("nan")
    return table, stats_out


def validate_selection(model: str, lat: float, lon: float, year: int) -> tuple[list[dict], int]:
    """Validate a model/location/year combination.

    Returns
    -------
    tuple[list[dict], int]
        Alerts (each `{"color": str, "message": str}`), and the possibly
        clamped year to actually use for the fetch.
    """
    meta = MODELS[model]
    alerts: list[dict] = []
    used_year = year

    if meta["europe_only"] and not (
        CERRA_LON_MIN <= lon <= CERRA_LON_MAX and CERRA_LAT_MIN <= lat <= CERRA_LAT_MAX
    ):
        alerts.append(
            {
                "color": "yellow",
                "message": (
                    f"{meta['label']} only covers Europe (lon "
                    f"{CERRA_LON_MIN}–{CERRA_LON_MAX}, lat {CERRA_LAT_MIN}–{CERRA_LAT_MAX}). "
                    "This location falls outside its domain — switch models to fetch data here."
                ),
            }
        )

    if year < meta["min_year"]:
        used_year = meta["min_year"]
        alerts.append(
            {
                "color": "yellow",
                "message": (
                    f"{meta['label']} starts in {meta['min_year']}. "
                    f"Showing {used_year} instead of {year}."
                ),
            }
        )
    elif meta["max_year"] is not None and year > meta["max_year"]:
        used_year = meta["max_year"]
        alerts.append(
            {
                "color": "yellow",
                "message": (
                    f"{meta['label']} ends in {meta['max_year']}. "
                    f"Showing {used_year} instead of {year}."
                ),
            }
        )

    if meta["warn_consistency"]:
        alerts.append(
            {
                "color": "yellow",
                "message": (
                    f"{meta['label']} blends ERA5 (pre-2017) and IFS (2017+) — "
                    "YTD accumulations spanning that boundary may show discontinuities."
                ),
            }
        )

    return alerts, used_year


def validate_ensemble_selection(
    systems: list[str], lat: float, lon: float, forecast_days: int
) -> tuple[list[dict], dict[str, int], list[str]]:
    """Validate a model (ensemble or deterministic)/location/forecast-day combination.

    Returns
    -------
    tuple[list[dict], dict[str, int], list[str]]
        Alerts (each `{"color": str, "message": str}`), each valid model's own
        horizon in days (the request, capped at its `max_forecast_days`), and
        the subset of `systems` actually in-domain and worth fetching
        (out-of-domain systems are dropped, not just warned about).
    """
    alerts: list[dict] = []
    valid_systems: list[str] = []
    days: dict[str, int] = {}

    for system in systems:
        meta = ENSEMBLE_MODELS.get(system) or DETERMINISTIC_MODELS[system]
        if meta["europe_only"] and not (
            CERRA_LON_MIN <= lon <= CERRA_LON_MAX and CERRA_LAT_MIN <= lat <= CERRA_LAT_MAX
        ):
            alerts.append(
                {
                    "color": "yellow",
                    "message": (
                        f"{meta['label']} only covers Europe (lon "
                        f"{CERRA_LON_MIN}–{CERRA_LON_MAX}, lat {CERRA_LAT_MIN}–{CERRA_LAT_MAX}). "
                        "Excluded from the meteogram for this location."
                    ),
                }
            )
            continue
        valid_systems.append(system)
        days[system] = min(forecast_days, meta["max_forecast_days"])

    # Each model runs to its own reach; a short one never shortens the others.
    short = [s for s in valid_systems if days[s] < forecast_days]
    if short:
        names = ", ".join(
            f"{(ENSEMBLE_MODELS.get(s) or DETERMINISTIC_MODELS[s])['label']} ({days[s]} d)" for s in short
        )
        alerts.append({"color": "blue", "message": f"Shorter range, drawn to its end: {names}."})

    return alerts, days, valid_systems


def ordinal(n: int) -> str:
    """Render an integer as an ordinal string, e.g. 3 -> '3rd'."""
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def make_summary(
    df: pd.DataFrame,
    current: pd.DataFrame,
    variable_label: str,
    year: int,
    accumulate: bool,
    direction_labels: tuple[str, str] = ("highest", "lowest"),
) -> str:
    """Build a plain-language auto-summary ranking the current year.

    Parameters
    ----------
    df : pd.DataFrame
        Full historical series (columns date, value, doy).
    current : pd.DataFrame
        Current year's series up to today (columns date, value, doy).
    variable_label : str
        Human-readable variable name, e.g. "Precipitation".
    year : int
        The selected year.
    accumulate : bool
        Whether the variable is accumulable (sum) or not (mean).
    direction_labels : tuple[str, str]
        (above-median word, below-median word) for this variable, e.g.
        `("wettest", "driest")` or `("warmest", "coolest")` — from
        `utils.settings.VARIABLES[variable]["direction_labels"]`.

    Returns
    -------
    str
        e.g. "2025 is tracking as the 3rd driest year on record ..."
    """
    if current.empty:
        return f"No data available yet for {year}."

    # Rank over the window the current year actually *has*, not the window it
    # spans. The archive returns rows up to yesterday but the reanalysis lags
    # several days behind, so the tail is NaN. `sum()` skips those days while
    # every historical year is summed over all of them, which for an
    # accumulated variable hands the current year a total short by the lag —
    # a few days of precipitation, biased in one direction, every time.
    observed = current.dropna(subset=["value"])
    if observed.empty:
        return f"No data available yet for {year}."

    today_doy = observed["doy"].max()
    ytd_val = observed["value"].sum() if accumulate else observed["value"].mean()

    historical = df[df["date"].dt.year < year]

    def _window_total(group: pd.DataFrame) -> float:
        """Total over the same DOY window, or NaN if that year has a gap in it."""
        window = group[group["doy"] <= today_doy]["value"]
        if window.empty or window.isna().any():
            return float("nan")
        return window.sum() if accumulate else window.mean()

    historical_ytd = historical.groupby(historical["date"].dt.year).apply(
        _window_total, include_groups=False
    )
    historical_ytd = historical_ytd.dropna()
    if historical_ytd.empty:
        return f"Not enough historical data to rank {year}."

    high_label, low_label = direction_labels
    n = len(historical_ytd) + 1
    value_pct_rank = round(100 * (historical_ytd <= ytd_val).sum() / n)
    direction = high_label if ytd_val > historical_ytd.median() else low_label
    rank = (
        int((historical_ytd > ytd_val).sum() + 1)
        if direction == high_label
        else int((historical_ytd < ytd_val).sum() + 1)
    )

    first_year = int(historical_ytd.index.min())
    return (
        f"{year} is tracking as the {ordinal(rank)} {direction} year since "
        f"{first_year} ({ordinal(value_pct_rank)} percentile) for "
        f"{variable_label} at this location."
    )


# ---------------------------------------------------------------------------
# Gridded monthly products (static gallery)
# ---------------------------------------------------------------------------


def monthly_anomaly(month: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """One month minus the mean of the same calendar month over the reference.

    Both come from `cds_api.load_series` with a `time` dimension; the month
    file holds a single step.
    """
    if month.sizes["time"] != 1:
        raise ValueError(f"expected one month, got {month.sizes['time']} steps")
    ref_months = set(reference["time"].dt.month.values.tolist())
    this_month = int(month["time"].dt.month.values[0])
    if ref_months != {this_month}:
        raise ValueError(
            f"reference spans months {sorted(ref_months)}, field is month {this_month}"
        )
    return month.isel(time=0, drop=True) - reference.mean("time")


def _box_mean(field: xr.DataArray, box: tuple[float, float, float, float]) -> pd.Series:
    """Area-weighted (cos latitude) mean over a lat/lon box, as a time series."""
    lat_min, lat_max, lon_min, lon_max = box
    sub = field.sel(
        latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
    )
    if sub.sizes["latitude"] == 0 or sub.sizes["longitude"] == 0:
        raise ValueError(f"box {box} contains no grid cells")
    weights = np.cos(np.deg2rad(sub["latitude"]))
    mean = sub.weighted(weights).mean(("latitude", "longitude"))
    return mean.to_series()


def _zonal_mean(field: xr.DataArray, lat: float) -> pd.Series:
    return field.sel(latitude=lat, method="nearest").mean("longitude").to_series()


def _point(field: xr.DataArray, lat: float, lon: float) -> pd.Series:
    return field.sel(latitude=lat, longitude=lon, method="nearest").to_series()


def _calendar_anomaly(
    series: pd.Series, reference: tuple[int, int], standardise: bool
) -> pd.Series:
    """Anomaly against each calendar month's reference mean, optionally in σ."""
    series = series.copy()
    series.index = pd.DatetimeIndex(series.index)
    years = series.index.year
    ref = series[(years >= reference[0]) & (years <= reference[1])]
    if ref.index.year.nunique() < 10:
        raise ValueError("the reference period has fewer than 10 years of data")
    months = series.index.month
    clim = ref.groupby(ref.index.month).mean()
    anomaly = series - clim.reindex(months).to_numpy()
    if standardise:
        spread = ref.groupby(ref.index.month).std(ddof=1)
        anomaly = anomaly / spread.reindex(months).to_numpy()
    return anomaly


# Wallace & Gutzler (1981) PNA centres and signs.
_PNA_POINTS = ((20.0, -160.0, 1), (45.0, -165.0, -1), (55.0, -115.0, 1), (30.0, -85.0, -1))


def compute_index(
    field: xr.DataArray, index: dict, reference: tuple[int, int]
) -> pd.DataFrame:
    """Monthly climate index from a `(time, latitude, longitude)` source field.

    Parameters
    ----------
    field : xr.DataArray
        Source from `cds_api.load_series`, every month of the record.
    index : dict
        A `settings.CLIMATE_INDICES` entry; `method` selects the recipe.
    reference : tuple[int, int]
        Years the anomalies (and σ, for pressure indices) are taken against.

    Returns
    -------
    pd.DataFrame
        Indexed by month start. `value` is the index as defined (smoothed
        when the definition smooths), `monthly` the unsmoothed monthly value.
    """
    method = index["method"]
    standardise = bool(index.get("standardise"))

    def anomaly(series: pd.Series) -> pd.Series:
        return _calendar_anomaly(series, reference, standardise)

    if method == "box_mean":
        monthly = anomaly(_box_mean(field, index["boxes"][0]))
    elif method == "box_diff":
        west, east = index["boxes"]
        monthly = anomaly(_box_mean(field, west)) - anomaly(_box_mean(field, east))
    elif method == "stations":
        (lat_a, lon_a), (lat_b, lon_b) = index["points"]
        monthly = anomaly(_point(field, lat_a, lon_a)) - anomaly(
            _point(field, lat_b, lon_b)
        )
    elif method == "zonal_diff":
        lat_a, lat_b = index["latitudes"]
        monthly = anomaly(_zonal_mean(field, lat_a)) - anomaly(_zonal_mean(field, lat_b))
    elif method == "pna":
        monthly = 0.25 * sum(
            sign * anomaly(_point(field, lat, lon)) for lat, lon, sign in _PNA_POINTS
        )
    else:
        raise ValueError(f"unknown index method '{method}'")

    monthly = monthly.sort_index()
    if index.get("detrend"):
        t = np.arange(len(monthly), dtype=float)
        valid = monthly.notna().to_numpy()
        slope, intercept = np.polyfit(t[valid], monthly.to_numpy()[valid], 1)
        monthly = monthly - (slope * t + intercept)

    window = index.get("smooth")
    value = (
        monthly.rolling(window, center=True, min_periods=window).mean()
        if window
        else monthly
    )
    return pd.DataFrame({"value": value, "monthly": monthly})


def index_for_month(index_frame: pd.DataFrame, month: int, column: str) -> pd.Series:
    """One calendar month of an index, keyed by year, for `correlation_map`."""
    series = index_frame[column]
    picked = series[series.index.month == month].dropna()
    return pd.Series(picked.to_numpy(), index=picked.index.year)


def correlation_map(
    field: xr.DataArray, index: pd.Series
) -> tuple[xr.DataArray, xr.DataArray, int]:
    """Per-cell Pearson correlation of a yearly field series with an index.

    Years are matched by calendar year, so the two need not come from the
    same request. The p-value is the two-sided t-test on r with n-2 degrees
    of freedom — no correction for autocorrelation, which for one calendar
    month a year apart is small; the caller should say so.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, int]
        r, p, and the number of years used.
    """
    field_years = field["time"].dt.year.values
    common = np.intersect1d(field_years, index.index.values)
    if common.size < 10:
        raise ValueError(f"only {common.size} overlapping years; need at least 10")

    y = field.isel(time=np.isin(field_years, common)).values
    x = index.loc[common].to_numpy(dtype=float)
    x = (x - x.mean()) / x.std(ddof=1)

    anomalies = y - y.mean(axis=0)
    spread = anomalies.std(axis=0, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.tensordot(x, anomalies, axes=(0, 0)) / ((common.size - 1) * spread)
        t = r * np.sqrt((common.size - 2) / (1.0 - r**2))
    p = 2 * stats.t.sf(np.abs(t), df=common.size - 2)

    coords = {"latitude": field["latitude"], "longitude": field["longitude"]}
    dims = ("latitude", "longitude")
    return (
        xr.DataArray(r, coords=coords, dims=dims),
        xr.DataArray(p, coords=coords, dims=dims),
        int(common.size),
    )
