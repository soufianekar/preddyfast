"""Forecast verification against station observations, one system at a time.

Every score here is computed per system and per lead. Nothing pools members
or errors across systems: each system's ensemble is its own distribution.

A lead with no matched observation has no score. It is dropped, never
scored as zero error, and `n` says how many runs each score rests on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _naive_utc(times: pd.Series) -> pd.Series:
    """Timestamps as naive UTC, so forecast and observation times compare."""
    times = pd.to_datetime(times)
    if times.dt.tz is not None:
        times = times.dt.tz_convert("UTC").dt.tz_localize(None)
    # One resolution on both sides: xarray gives ns, pandas parsing may give us.
    return times.astype("datetime64[ns]")


def match_obs(
    fc: pd.DataFrame, obs: pd.DataFrame, tolerance: str = "15min"
) -> pd.DataFrame:
    """Attach the nearest observation within `tolerance` to each forecast row.

    Parameters
    ----------
    fc : pd.DataFrame
        `init_time, valid_time, lead_hours, member, value`.
    obs : pd.DataFrame
        `valid_time, value`; rows with NaN values are ignored.

    Returns
    -------
    pd.DataFrame
        `fc` plus `obs`; NaN where no report lies within `tolerance`.
    """
    left = fc.assign(valid_time=_naive_utc(fc["valid_time"])).sort_values("valid_time")
    right = (
        obs.dropna(subset=["value"])
        .assign(valid_time=lambda d: _naive_utc(d["valid_time"]))
        .rename(columns={"value": "obs"})[["valid_time", "obs"]]
        .sort_values("valid_time")
    )
    return pd.merge_asof(
        left,
        right,
        on="valid_time",
        direction="nearest",
        tolerance=pd.Timedelta(tolerance),
    )


def obs_window_total(obs_1h: pd.DataFrame, end_times, hours: int) -> pd.Series:
    """Observed totals over `(end - hours, end]`, NaN unless every hour reported.

    Each report belongs to the hour ending at or after it. Within an hour the
    last report is the hour's total: GHCNh hourly precipitation is a running
    total for the hour when several reports arrive (doc §IX).
    """
    o = obs_1h.dropna(subset=["value"]).copy()
    o["hour"] = _naive_utc(o["valid_time"]).dt.ceil("h")
    per_hour = o.sort_values("valid_time").groupby("hour")["value"].last()

    ends = _naive_utc(pd.Series(list(end_times)))
    totals = []
    for end in ends:
        window = per_hour[
            (per_hour.index > end - pd.Timedelta(hours=hours)) & (per_hour.index <= end)
        ]
        totals.append(float(window.sum()) if len(window) == hours else np.nan)
    return pd.Series(totals, index=pd.DatetimeIndex(ends), name="obs")


def fair_crps(members: np.ndarray, obs: float) -> float:
    """Fair CRPS of an ensemble (Ferro 2014): unbiased for the member count.

    `mean|x_i - y| - sum_ij |x_i - x_j| / (2 m (m - 1))`. A one-member
    "ensemble" has no spread term and reduces to the absolute error.
    """
    x = np.asarray(members, dtype=float)
    m = x.size
    skill = np.mean(np.abs(x - obs))
    if m < 2:
        return float(skill)
    spread = np.abs(x[:, None] - x[None, :]).sum() / (2 * m * (m - 1))
    return float(skill - spread)


def _brier_key(threshold: float) -> str:
    return f"brier_{threshold:g}"


def per_run_scores(matched: pd.DataFrame, thresholds: tuple[float, ...] = ()) -> pd.DataFrame:
    """One row per (run, lead) with every additive part of the scores.

    This is the unit that is stored and aggregated: any period's scores are
    means of these rows, so a longer period never needs a re-read.

    Parameters
    ----------
    matched : pd.DataFrame
        `init_time, lead_hours, member, value, obs` (from `match_obs`).
    thresholds : tuple of float
        Event thresholds for the Brier score, as `value >= threshold`.

    Returns
    -------
    pd.DataFrame
        `init_time, lead_hours, valid_time, n_members, obs, err` (ensemble
        mean minus observation), `abs_err, sq_err, var` (members, n-1),
        `crps` (fair), and `brier_<t>` per threshold. Runs with no
        observation are absent.
    """
    have = matched.dropna(subset=["obs", "value"])
    columns = ["init_time", "lead_hours", "valid_time", "n_members", "obs", "err", "abs_err",
               "sq_err", "var", "crps", *(_brier_key(t) for t in thresholds)]
    if have.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for (lead, init), group in have.groupby(["lead_hours", "init_time"]):
        x = group["value"].to_numpy(dtype=float)
        y = float(group["obs"].iloc[0])
        err = float(x.mean() - y)
        row = {
            "init_time": init,
            "lead_hours": float(lead),
            "valid_time": group["valid_time"].iloc[0] if "valid_time" in group else pd.NaT,
            "n_members": int(x.size),
            "obs": y,
            "err": err,
            "abs_err": abs(err),
            "sq_err": err * err,
            "var": float(x.var(ddof=1)) if x.size > 1 else 0.0,
            "crps": fair_crps(x, y),
        }
        for t in thresholds:
            # Probability from the member fraction, event "at or above".
            row[_brier_key(t)] = (float((x >= t).mean()) - float(y >= t)) ** 2
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def aggregate_scores(per_run: pd.DataFrame, thresholds: tuple[float, ...] = ()) -> pd.DataFrame:
    """Scores per lead from per-run rows: the WMO surface set plus ensemble scores.

    Returns
    -------
    pd.DataFrame
        `lead_hours, n, bias` (ME), `mae, rmse, sd_error` (SD of the errors,
        n-1), `crps, spread, spread_skill`, and `brier_<t>`. Leads with no
        run are absent.
    """
    briers = [_brier_key(t) for t in thresholds]
    columns = ["lead_hours", "n", "bias", "mae", "rmse", "sd_error", "crps", "spread",
               "spread_skill", *briers]
    if per_run.empty:
        return pd.DataFrame(columns=columns)
    grouped = per_run.groupby("lead_hours")
    out = grouped.agg(
        n=("err", "size"),
        bias=("err", "mean"),
        mae=("abs_err", "mean"),
        rmse=("sq_err", lambda v: np.sqrt(np.mean(v))),
        sd_error=("err", lambda e: e.std(ddof=1) if len(e) > 1 else np.nan),
        crps=("crps", "mean"),
        spread=("var", lambda v: np.sqrt(np.mean(v))),
        **{b: (b, "mean") for b in briers},
    )
    out["spread_skill"] = out["spread"] / out["rmse"].replace(0.0, np.nan)
    return out.reset_index()[columns]


def scores_by_lead(matched: pd.DataFrame, thresholds: tuple[float, ...] = ()) -> pd.DataFrame:
    """Scores per lead straight from matched rows (`per_run_scores` then `aggregate_scores`).

    The ensemble mean, variance and CRPS are taken per run (init, lead);
    those are then averaged over runs. Only runs with an observation count.
    Bias is forecast minus observation.
    """
    return aggregate_scores(per_run_scores(matched, thresholds), thresholds)


def common_leads(
    scores: dict[str, pd.DataFrame], step_hours: int, offset: float = 0.0
) -> list[float]:
    """Leads on a `step_hours` grid (shifted by `offset`) that every model with scores has.

    Models step differently (IFS 3 h then 6 h, AIFS 6 h, GEFS 3 h then 6 h);
    comparing them lead for lead needs leads they share. `offset` aligns
    window ends with a station's: 00 UTC runs against 06–06 UTC totals keep
    T+30, T+54, … A model with no scores at all does not veto the others.
    """
    sets = [
        {
            float(h)
            for h in df["lead_hours"]
            if h > 0 and (float(h) - offset) % step_hours == 0
        }
        for df in scores.values()
        if not df.empty
    ]
    return sorted(set.intersection(*sets)) if sets else []


def imerg_window_totals(halfhours: pd.DataFrame, end_times, hours: int) -> pd.Series:
    """IMERG totals (mm) over `(end - hours, end]`, NaN unless every half-hour is present.

    IMERG stamps each half-hour at its *start* (the record begins 1998-01-01
    00:00, the file covering 00:00:00–00:29:59), so a window ending at E holds
    the half-hours starting in [E - hours, E). `halfhours` is `time, mm`.
    """
    amounts = halfhours.dropna(subset=["mm"]).set_index("time")["mm"].sort_index()
    need = hours * 2
    totals = []
    for end in pd.to_datetime(list(end_times)):
        window = amounts[(amounts.index >= end - pd.Timedelta(hours=hours)) & (amounts.index < end)]
        totals.append(float(window.sum()) if len(window) == need else np.nan)
    return pd.Series(totals, index=pd.DatetimeIndex(pd.to_datetime(list(end_times))), name="obs")


def sample_leads(scores: pd.DataFrame, leads: list[float]) -> pd.DataFrame:
    """The rows of a per-lead score frame at `leads` only."""
    return scores[scores["lead_hours"].isin(leads)].reset_index(drop=True)
