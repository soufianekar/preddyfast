"""Station scorecard: one station cube + station observations -> per-pair score rows.

Pure: no Dash, no reads. `match_station_obs` pairs every (station, lead) of a
`earthmover_api.station_cube` with its observation on valid time (UTC, never
local time), or `match_reference` with the IFS control T+0 of the init at
each 00 UTC valid time (Z500); `pair_scores` turns the pairs into one row of
additive score parts each, which `aggregate` reduces over periods and lead-day bins.
"""

from __future__ import annotations

import functools
import json
import logging
import operator
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scores
import xarray as xr

from utils import (
    earthmover_api,
    hub_sync,
    observations,
    scores_db,
    station_network,
    status,
    verification,
)
from utils.earthmover_api import GridDataError
from utils.observations import (
    ObservationError,  # noqa: F401  (re-exported: refresh raises it)
)
from utils.settings import (
    EVAL_REFERENCES,
    GRID_STORES,
    PRECOMPUTE_DIR,
    SCORE_DIRECTION,
    SCORECARD_DAYS,
    SCORECARD_INIT_HOUR,
    SCORECARD_LEAD_STEP_HOURS,
    SCORECARD_MAX_LEAD_HOURS,
    SCORECARD_MIN_INITS,
    SCORECARD_OBS_GRACE_HOURS,
    SCORECARD_OBS_RECENT_DAYS,
    SCORECARD_PERIODS,
    SCORECARD_READ_BUDGET_MB,
    SCORECARD_STATIONS,
    SCORECARD_STORES,
    SCORECARD_VARIABLES,
)

logger = logging.getLogger(__name__)

PAIR_COLUMNS: list[str] = [
    "station",
    "init",
    "lead_hours",
    "valid_time",
    "obs",
    "fc_mean",
    "err",
    "ae",
    "se",
    "crps",
    "spread2",
    "wet_fc",
    "wet_obs",
    "brier",
]
_PAIR = "pair"


def _spec(variable: str) -> dict:
    try:
        return SCORECARD_VARIABLES[variable]
    except KeyError:
        raise ValueError(f"Unknown scorecard variable {variable!r}") from None


def _instant_obs(
    frame: pd.DataFrame, column: str, times: pd.Series, tolerance: str
) -> np.ndarray:
    """The report nearest each time within `tolerance`, NaN where there is none."""
    if column not in frame or frame.empty:
        return np.full(len(times), np.nan)
    fc = pd.DataFrame({"valid_time": times.to_numpy(), "order": np.arange(len(times))})
    reports = pd.DataFrame({"valid_time": frame["valid_time"], "value": frame[column]})
    matched = verification.match_obs(fc, reports, tolerance=tolerance)
    return matched.sort_values("order")["obs"].to_numpy(dtype=float)


def _window_obs(frame: pd.DataFrame, spec: dict, times: pd.Series) -> np.ndarray:
    """Observed window totals ending exactly at each time.

    The station's own reported total first (`obs_reported`); else the sum of
    the window's hourly reports (`obs`), only when every hour reported. Never
    a partial sum.
    """
    hours = spec["window_hours"]
    ends = verification._naive_utc(times).reset_index(drop=True)
    out = np.full(len(ends), np.nan)
    if frame.empty:
        return out

    if spec["obs_reported"] and spec["obs_reported"] in frame:
        reported = observations.reported_totals(frame, hours)
        if not reported.empty:
            by_end = (
                reported.assign(
                    valid_time=verification._naive_utc(reported["valid_time"])
                )
                .groupby("valid_time")["value"]
                .last()
            )
            out = by_end.reindex(pd.DatetimeIndex(ends)).to_numpy(dtype=float)

    if spec["obs"] in frame:
        hourly = pd.DataFrame(
            {"valid_time": frame["valid_time"], "value": frame[spec["obs"]]}
        )
        summed = verification.obs_window_total(hourly, ends, hours).to_numpy(
            dtype=float
        )
        out = np.where(np.isfinite(out), out, summed)
    return out


def match_station_obs(
    cube: xr.DataArray, obs_by_station: dict[str, pd.DataFrame], variable: str
) -> xr.Dataset:
    """Pair a station cube with the stations' observations on valid time.

    Parameters
    ----------
    cube : xarray.DataArray
        From `earthmover_api.station_cube`: dims `(station, member,
        lead_hours)`, coord `valid_time(lead_hours)` (naive UTC), attrs
        `init`, `store`, `variable`, `window_hours`.
    obs_by_station : dict of str to pandas.DataFrame
        From `station_network.network_obs`: per station, `valid_time` (UTC)
        and the internal observation columns. A station missing here has no
        observations.
    variable : str
        `SCORECARD_VARIABLES` key.

    Returns
    -------
    xarray.Dataset
        Dims `(station, lead_hours)` plus `member` for `fc`: `fc` (members),
        `fc_mean` (member mean), `obs` (NaN where none) and `valid_time`.
        Leads beyond `SCORECARD_MAX_LEAD_HOURS` are dropped. Instants take the
        report nearest the valid time within the variable's `tolerance`;
        window variables the reported total ending at the valid time, else a
        complete-hour sum. Attrs carry the cube's plus `scorecard_variable`.

    Raises
    ------
    ValueError
        Unknown variable, or one verified against a reference rather than
        stations (`obs` is None, e.g. Z500).
    """
    spec = _spec(variable)
    if spec["obs"] is None:
        raise ValueError(
            f"{variable!r} has no station observation (reference "
            f"{spec['reference']!r}); it is verified by the reference path"
        )
    cube = cube.sel(lead_hours=cube["lead_hours"] <= SCORECARD_MAX_LEAD_HOURS)
    times = pd.Series(pd.to_datetime(cube["valid_time"].values))

    rows = []
    for sid in cube["station"].values:
        frame = obs_by_station.get(str(sid), pd.DataFrame())
        if spec["window_hours"]:
            rows.append(_window_obs(frame, spec, times))
        else:
            rows.append(_instant_obs(frame, spec["obs"], times, spec["tolerance"]))
    obs = np.vstack(rows) if rows else np.empty((0, cube.sizes["lead_hours"]))

    # station_cube is float32: a float32 member mean can land a few ulp below a
    # threshold the exact mean meets, so thresholds are applied in float64.
    fc = cube.transpose("station", "member", "lead_hours").astype("float64")
    ds = xr.Dataset(
        {
            "fc": fc,
            "fc_mean": fc.mean("member"),
            "obs": (("station", "lead_hours"), obs),
        },
        attrs={**cube.attrs, "scorecard_variable": variable},
    )
    return ds


def reference_leads(cube: xr.DataArray) -> xr.DataArray:
    """The cube at the leads a T+0 reference can verify (R5).

    Leads up to `SCORECARD_MAX_LEAD_HOURS` whose valid time is on the hour
    `SCORECARD_INIT_HOUR` (00 UTC): only those inits are archived for the
    scorecard, so a 12 UTC valid time has no reference and is dropped.
    """
    cube = cube.sel(lead_hours=cube["lead_hours"] <= SCORECARD_MAX_LEAD_HOURS)
    valid = pd.DatetimeIndex(pd.to_datetime(cube["valid_time"].values))
    on_init_hour = (
        (valid.hour == SCORECARD_INIT_HOUR) & (valid.minute == 0) & (valid.second == 0)
    )
    return cube.isel(lead_hours=np.flatnonzero(on_init_hour))


def reference_valid_times(cube: xr.DataArray) -> list[pd.Timestamp]:
    """Valid times (naive UTC) whose T+0 reference init `match_reference` needs."""
    kept = reference_leads(cube)
    return [pd.Timestamp(t) for t in pd.to_datetime(kept["valid_time"].values)]


def match_reference(
    cube: xr.DataArray, references: dict[pd.Timestamp, xr.DataArray], variable: str
) -> xr.Dataset:
    """Pair a station cube with a gridded T+0 reference (Z500 vs IFS control).

    Parameters
    ----------
    cube : xarray.DataArray
        The forecast, from `earthmover_api.station_cube`.
    references : dict of pandas.Timestamp to xarray.DataArray
        Reference station cubes (`EVAL_REFERENCES[...]["store"]`, one member)
        keyed by their own init (naive UTC). The value at valid time V is that
        of the reference init V at lead 0 — never another init's lead. A valid
        time with no init here (not archived yet) or a station absent from it
        gets NaN, so it yields no pair.
    variable : str
        `SCORECARD_VARIABLES` key whose `reference` is not "stations".

    Returns
    -------
    xarray.Dataset
        As `match_station_obs`, at the leads of `reference_leads` only.

    Raises
    ------
    ValueError
        A station-verified variable, or a reference whose `init` attr is not
        the key it is filed under.
    """
    spec = _spec(variable)
    if spec["reference"] == "stations":
        raise ValueError(f"{variable!r} is verified against stations, not a reference")
    cube = reference_leads(cube)
    stations = [str(s) for s in cube["station"].values]
    obs = np.full((len(stations), cube.sizes["lead_hours"]), np.nan)
    for j, valid in enumerate(pd.to_datetime(cube["valid_time"].values)):
        ref = references.get(pd.Timestamp(valid))
        if ref is None:
            continue
        if pd.Timestamp(ref.attrs["init"]) != pd.Timestamp(valid):
            raise ValueError(
                f"Reference filed under {valid} is the run of {ref.attrs['init']}"
            )
        if not (ref["lead_hours"] == 0).any():
            continue
        t0 = ref.sel(lead_hours=0.0).isel(member=0)
        t0 = t0.assign_coords(station=[str(s) for s in t0["station"].values])
        obs[:, j] = t0.reindex(station=stations).values

    fc = cube.transpose("station", "member", "lead_hours").astype("float64")
    return xr.Dataset(
        {
            "fc": fc,
            "fc_mean": fc.mean("member"),
            "obs": (("station", "lead_hours"), obs),
        },
        attrs={**cube.attrs, "scorecard_variable": variable},
    )


def _empty_pairs(dropped: int) -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series(dtype=float) for c in PAIR_COLUMNS})
    out.attrs["dropped_nan_fc"] = dropped
    return out


def pair_scores(matched: xr.Dataset, variable: str) -> pd.DataFrame:
    """One row of score parts per (station, lead_hours, init) with an observation.

    Parameters
    ----------
    matched : xarray.Dataset
        From `match_station_obs`.
    variable : str
        `SCORECARD_VARIABLES` key; its `threshold` and `scores` decide the
        contingency flags and the Brier score.

    Returns
    -------
    pandas.DataFrame
        Columns `PAIR_COLUMNS`. `err` is ensemble mean minus observation, `ae`
        and `se` its absolute and squared value. `crps` is the fair CRPS of
        the members; for a single member (deterministic store) the fair kernel
        divides by M(M-1) = 0, so the ECDF CRPS is used, which for one member
        is exactly |f - o| (CRPS reduces to MAE). `spread2` is the unbiased
        member variance (ddof=1), NaN for one member. `wet_fc` (ensemble mean)
        and `wet_obs` are float 1.0/0.0 for `>= threshold` (inclusive), NaN
        for a variable with no threshold: sum them for contingency counts.
        `brier` is the fair Brier score at the threshold (`>=`) for ensembles
        of a variable whose scores include it, NaN otherwise.

        Pairs with no finite observation give no row (never a zero). A pair
        whose forecast has a non-finite member is not scored either; their
        number is in `attrs["dropped_nan_fc"]` and logged.
    """
    spec = _spec(variable)
    pairs = matched.stack({_PAIR: ("station", "lead_hours")})
    has_obs = np.isfinite(pairs["obs"].values)
    fc_ok = np.isfinite(pairs["fc"].transpose(_PAIR, "member").values).all(axis=1)
    dropped = int((has_obs & ~fc_ok).sum())
    if dropped:
        logger.warning(
            "%s %s %s: %d pair(s) with a non-finite forecast not scored",
            matched.attrs.get("store"),
            variable,
            matched.attrs.get("init"),
            dropped,
        )
    keep = np.flatnonzero(has_obs & fc_ok)
    if keep.size == 0:
        return _empty_pairs(dropped)

    # A plain integer pair index: scores' preserve_dims does not need the MultiIndex.
    station = pairs["station"].values[keep]
    lead = pairs["lead_hours"].values[keep]
    sub = pairs.isel({_PAIR: keep}).drop_vars(["station", "lead_hours", _PAIR])
    sub = sub.assign_coords({_PAIR: np.arange(keep.size)})
    fc, fc_mean, obs = sub["fc"], sub["fc_mean"], sub["obs"]
    n_members = fc.sizes["member"]
    keep_dims = [_PAIR]

    err = scores.continuous.additive_bias(fc_mean, obs, preserve_dims=keep_dims)
    ae = scores.continuous.mae(fc_mean, obs, preserve_dims=keep_dims)
    se = scores.continuous.mse(fc_mean, obs, preserve_dims=keep_dims)
    crps = scores.probability.crps_for_ensemble(
        fc,
        obs,
        "member",
        method="fair" if n_members > 1 else "ecdf",
        preserve_dims=keep_dims,
    )
    nan = np.full(keep.size, np.nan)
    spread2 = fc.var("member", ddof=1).values if n_members > 1 else nan

    threshold = spec["threshold"]
    if threshold is None:
        wet_fc = wet_obs = nan
    else:
        wet_fc = (fc_mean.values >= threshold).astype(float)
        wet_obs = (obs.values >= threshold).astype(float)
    if "brier" in spec["scores"] and threshold is not None and n_members > 1:
        brier = (
            scores.probability.brier_score_for_ensemble(
                fc,
                obs,
                "member",
                threshold,
                preserve_dims=keep_dims,
                fair_correction=True,
                event_threshold_operator=operator.ge,
            )
            .squeeze("threshold", drop=True)
            .values
        )
    else:
        brier = nan

    out = pd.DataFrame(
        {
            "station": station.astype(str),
            "init": pd.Timestamp(matched.attrs["init"]),
            "lead_hours": lead,
            "valid_time": pd.to_datetime(sub["valid_time"].values),
            "obs": obs.values,
            "fc_mean": fc_mean.values,
            "err": err.values,
            "ae": ae.values,
            "se": se.values,
            "crps": crps.values,
            "spread2": spread2,
            "wet_fc": wet_fc,
            "wet_obs": wet_obs,
            "brier": brier,
        },
        columns=PAIR_COLUMNS,
    )
    out.attrs["dropped_nan_fc"] = dropped
    return out


# --- aggregation -------------------------------------------------------------------

SCORE_COLUMNS: list[str] = list(SCORE_DIRECTION)
_COUNT_COLUMNS: list[str] = ["n", "n_inits", "n_stations", "low_n"]


def _div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    """num / den, NaN where the denominator is zero or not finite (never inf)."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    ok = np.isfinite(den) & (den != 0)
    return np.divide(
        num, den, out=np.full(np.broadcast(num, den).shape, np.nan), where=ok
    )


def contingency_scores(
    h: np.ndarray, m: np.ndarray, f: np.ndarray, cn: np.ndarray
) -> dict[str, np.ndarray]:
    """ETS, HSS and frequency bias from summed hits, misses, false alarms and
    correct negatives.

    The dynamical.org formulas, identical to `scores.categorical.
    BinaryContingencyManager` (which takes events, not counts):
    ETS = (H - Hr) / (H + M + F - Hr), Hr = (H + M)(H + F) / n;
    HSS = 2(H·CN - F·M) / [(H + M)(M + CN) + (H + F)(F + CN)];
    FB = (H + F) / (H + M). A zero denominator gives NaN.
    """
    h, m, f, cn = (np.asarray(x, dtype=float) for x in (h, m, f, cn))
    n = h + m + f + cn
    hr = _div((h + m) * (h + f), n)
    return {
        "ets": _div(h - hr, h + m + f - hr),
        "hss": _div(2.0 * (h * cn - f * m), (h + m) * (m + cn) + (h + f) * (f + cn)),
        "frequency_bias": _div(h + f, h + m),
    }


def _select_period(
    rows: pd.DataFrame, period_days: int, end: pd.Timestamp | None
) -> pd.DataFrame:
    """Rows whose init date is within the last `period_days` days up to `end`."""
    days = rows["init"].dt.normalize()
    last = days.max() if end is None else pd.Timestamp(end).normalize()
    first = last - pd.Timedelta(days=period_days - 1)
    return rows[(days >= first) & (days <= last)]


def _keep_common_inits(rows: pd.DataFrame) -> pd.DataFrame:
    """Per variable, only the inits every store present for that variable holds."""
    held = rows.groupby(["variable", "init"])["store"].nunique()
    stores = rows.groupby("variable")["store"].nunique()
    common = held[
        held.to_numpy() == stores.reindex(held.index.get_level_values(0)).to_numpy()
    ]
    keys = pd.MultiIndex.from_arrays([rows["variable"], rows["init"]])
    return rows[keys.isin(common.index)]


def aggregate(
    rows: pd.DataFrame,
    period_days: int,
    by: tuple[str, ...] = ("store", "lead_day"),
    *,
    end: pd.Timestamp | None = None,
    common_inits: bool = True,
    min_n: int = 50,
) -> pd.DataFrame:
    """Scorecard scores per group from per-pair rows.

    Parameters
    ----------
    rows : pandas.DataFrame
        From `scores_db.load_rows`: `PAIR_COLUMNS` plus `store` and `variable`,
        normally the whole `SCORECARD_DAYS` window. Several variables may be
        mixed; they are never pooled.
    period_days : int
        Inits whose date lies within the last `period_days` days, counted back
        from the newest init in `rows` (or `end`), inclusive.
    by : tuple of str
        Group keys besides `variable` (always added): `("store", "lead_day")`,
        `("store", "station")` (pools every lead) or
        `("store", "lead_day", "station")` (select a lead from the result; never
        pre-filter `rows` by lead, which changes the bias-correction offset).
        `lead_day` = ceil(lead_hours/24), so day 1 is (0, 24] h; T+0 pairs
        and leads off the `SCORECARD_LEAD_STEP_HOURS` step are dropped (R23),
        so every store's bin holds the same valid hours.
    end : pandas.Timestamp, optional
        Last init date of the period.
    common_inits : bool
        Per variable, keep only the inits every store in `rows` holds (after
        the period cut), so models are compared on the same cases.
    min_n : int
        Groups with fewer pairs, or resting on fewer than
        `SCORECARD_MIN_INITS` runs (R24), are flagged `low_n`.

    Returns
    -------
    pandas.DataFrame
        One row per group: `variable`, the `by` keys, `n` (pairs), `n_inits`,
        `n_stations`, `low_n`, and every column of `SCORE_COLUMNS`; a score the
        variable does not offer (`SCORECARD_VARIABLES[...]["scores"]`) or that
        is undefined (zero denominator, deterministic spread or Brier) is NaN.

    Notes
    -----
    Pass the whole 90-day window as `rows`: with only a period's rows the bias
    correction (`rmse_bc`, `mae_bc`) uses that period's mean error.

    Scores are computed over all pairs of a group, each pair weighted equally:
    "averaged across initialisation times" with each init contributing its
    pairs, which are near-equal in number per init. RMSE is √(mean se), not a
    mean of per-init RMSEs. `brier` is the mean fair Brier score (per pair it
    may be slightly negative). `rmse_bc`/`mae_bc` subtract each (store,
    station, variable) mean error over **all rows passed in** (the full
    90-day window, not just the period; T+0 excluded), so a 7-day card corrects with the
    same stable station offset as the 90-day one. `spread_skill` is
    √((M+1)/M · mean spread2) / RMSE with M the store's member count
    (`GRID_STORES`), ideal 1; NaN for one member. ETS, HSS and frequency bias
    come from the group's summed contingency counts (`contingency_scores`).
    """
    by = tuple(by)
    keys = ["variable", *[k for k in by if k != "variable"]]
    # R23: one valid-hour set per bin for every store (hourly GFS/ICON-EU,
    # 3-hourly IFS ENS/GEFS, 6-hourly AIFS), before the bias offset.
    lead = rows["lead_hours"].to_numpy(dtype=float)
    on_step = np.isclose(np.mod(lead, SCORECARD_LEAD_STEP_HOURS), 0.0)
    rows = rows[(lead > 0) & on_step].copy()
    rows["lead_day"] = np.ceil(rows["lead_hours"].to_numpy(dtype=float) / 24.0).astype(
        int
    )

    offset = rows.groupby(["store", "station", "variable"])["err"].transform("mean")
    rows["err_bc"] = rows["err"] - offset
    rows["se_bc"] = rows["err_bc"] ** 2
    rows["ae_bc"] = rows["err_bc"].abs()

    if not rows.empty:
        rows = _select_period(rows, period_days, end)
    if common_inits and not rows.empty:
        rows = _keep_common_inits(rows)
    if rows.empty:
        return pd.DataFrame(columns=[*keys, *_COUNT_COLUMNS, *SCORE_COLUMNS])

    has_cat = rows["wet_fc"].notna() & rows["wet_obs"].notna()
    wf = rows["wet_fc"].where(has_cat)
    wo = rows["wet_obs"].where(has_cat)
    rows["h"] = wf * wo
    rows["m"] = (1.0 - wf) * wo
    rows["f"] = wf * (1.0 - wo)
    rows["cn"] = (1.0 - wf) * (1.0 - wo)

    # pandas' groupby mean/sum skip NaN and give NaN for an all-NaN group, no warning.
    g = rows.groupby(keys, sort=True)
    out = g.agg(
        n=("err", "size"),
        n_inits=("init", "nunique"),
        n_stations=("station", "nunique"),
        se=("se", "mean"),
        mae=("ae", "mean"),
        bias=("err", "mean"),
        crps=("crps", "mean"),
        brier=("brier", "mean"),
        se_bc=("se_bc", "mean"),
        mae_bc=("ae_bc", "mean"),
        spread2=("spread2", "mean"),
        h=("h", "sum"),
        m=("m", "sum"),
        f=("f", "sum"),
        cn=("cn", "sum"),
        n_cat=("h", "count"),
    ).reset_index()

    out["rmse"] = np.sqrt(out["se"])
    out["rmse_bc"] = np.sqrt(out["se_bc"])
    if "store" in out:
        members = out["store"].map(lambda s: GRID_STORES.get(s, {}).get("members", 1))
    else:
        members = pd.Series(1, index=out.index)
    members = members.to_numpy(dtype=float)
    ens = members > 1
    factor = np.where(ens, (members + 1.0) / np.where(ens, members, 1.0), np.nan)
    out["spread_skill"] = _div(np.sqrt(factor * out["spread2"].to_numpy()), out["rmse"])

    cat = contingency_scores(out["h"], out["m"], out["f"], out["cn"])
    no_cat = out["n_cat"].to_numpy() == 0
    for name, values in cat.items():
        out[name] = np.where(no_cat, np.nan, values)

    offered = out["variable"].map(
        lambda v: set(SCORECARD_VARIABLES.get(v, {}).get("scores", ()))
    )
    for score in SCORE_COLUMNS:
        mask = offered.map(lambda s, score=score: score in s).to_numpy(dtype=bool)
        out[score] = out[score].where(mask)

    out["low_n"] = (out["n"] < min_n) | (out["n_inits"] < SCORECARD_MIN_INITS)
    return out[[*keys, *_COUNT_COLUMNS, *SCORE_COLUMNS]]


def relative_to(agg: pd.DataFrame, baseline: str = "ifs_ens") -> pd.DataFrame:
    """Percent change of every score against the baseline store, positive = better.

    Parameters
    ----------
    agg : pandas.DataFrame
        From `aggregate` (must hold `store`).
    baseline : str
        The reference store; matched on every other group key present
        (`variable`, `lead_day`, `station`).

    Returns
    -------
    pandas.DataFrame
        `agg` with each score column of `SCORE_DIRECTION` replaced by its
        percent change: lower-is-better (base - model)/|base|; higher-is-better
        (model - base)/|base|; target-valued (|base - t| - |model - t|)/|base - t|,
        all ×100. Baseline rows are 0, or NaN where the baseline sits at its ideal
        (zero denominator); no baseline row gives NaN.
    """
    keys = [k for k in ("variable", "lead_day", "station") if k in agg]
    scores_here = [c for c in SCORE_COLUMNS if c in agg]
    base = agg.loc[agg["store"] == baseline, [*keys, *scores_here]]
    base = base.drop_duplicates(keys)
    merged = agg.merge(base, on=keys, how="left", suffixes=("", "__base"))
    out = agg.copy()
    for score in scores_here:
        model = merged[score].to_numpy(dtype=float)
        ref = merged[f"{score}__base"].to_numpy(dtype=float)
        direction = SCORE_DIRECTION[score]
        if direction == "lower":
            pct = _div(ref - model, np.abs(ref))
        elif direction == "higher":
            pct = _div(model - ref, np.abs(ref))
        else:
            dist_ref = np.abs(ref - direction)
            pct = _div(dist_ref - np.abs(model - direction), dist_ref)
        out[score] = pct * 100.0
    return out


# --- incremental runner ----------------------------------------------------------

_clock = time.monotonic  # replaced in tests


# GHCNh reaches us days late; observations are loaded this far back from today
# so the observation end (and with it the window) can be found.
_OBS_LAG_ALLOWANCE_DAYS = 30


def _window(anchor: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """First and last init date kept: SCORECARD_DAYS inits ending the day
    before `anchor`.

    `anchor` is the observation end (refresh) or today; its date is used.
    """
    day = pd.Timestamp(anchor).normalize()
    return day - pd.Timedelta(days=SCORECARD_DAYS), day - pd.Timedelta(days=1)


def _grid_read(variable: str) -> tuple[str, int | None]:
    """The `(grid variable, accumulation hours)` station_cube is asked for."""
    spec = _spec(variable)
    return spec["grid"][0], spec["window_hours"]


def _store_has(store: str, grid_variable: str) -> bool:
    """Whether a store holds a variable.

    `store_variables` lists only stored `GRID_VARIABLES`; a derived variable
    (10 m wind speed) is built from raw u/v components that no listing
    reports, so it is taken as available (every scorecard store holds them,
    see `STORE_RAW_COMPONENTS`) and station_cube refuses the run if not.
    """
    if "derived_from" in earthmover_api.GRID_VARIABLES.get(grid_variable, {}):
        return True
    return grid_variable in earthmover_api.store_variables(store)


def _needs_scoring(
    meta: dict | None, net_hash: str, obs_end: pd.Timestamp | None
) -> bool:
    """Score when the file is missing, built on another network, or has leads
    still awaiting observations that have since arrived."""
    if meta is None or meta.get("network_hash") != net_hash:
        return True
    if meta.get("complete") or obs_end is None:
        return False
    stored = meta.get("obs_end")
    return stored is None or obs_end > pd.Timestamp(stored)


def _naive_utc(t: pd.Timestamp) -> pd.Timestamp:
    """A timestamp as naive UTC (tz-aware input is converted, naive is taken as UTC)."""
    t = pd.Timestamp(t)
    return t.tz_convert("UTC").tz_localize(None) if t.tzinfo else t


def _obs_end(
    obs: dict[str, pd.DataFrame],
    cap: pd.Timestamp | None = None,
    grace_hours: float = SCORECARD_OBS_GRACE_HOURS,
) -> pd.Timestamp | None:
    """The time up to which the network's observations count as in, naive UTC.

    The median of each station's last report over the stations that reported
    in the `SCORECARD_OBS_RECENT_DAYS` before the newest report, minus
    `SCORECARD_OBS_GRACE_HOURS`. A station that stopped, or one bad timestamp,
    cannot move it, and a station whose data lag by up to the grace still has
    its last leads scored on a later run. Reports after `cap` (the run's end)
    are ignored. None when no station has a report.
    """
    last = []
    for frame in obs.values():
        if not len(frame):
            continue
        times = pd.to_datetime(frame["valid_time"])
        if times.dt.tz is not None:
            times = times.dt.tz_convert("UTC").dt.tz_localize(None)
        if cap is not None:
            times = times[times <= _naive_utc(cap)]
        if len(times):
            last.append(times.max())
    if not last:
        return None
    last = pd.Series(last)
    recent = last[last >= last.max() - pd.Timedelta(days=SCORECARD_OBS_RECENT_DAYS)]
    return recent.median() - pd.Timedelta(hours=grace_hours)


def _anchor(obs: dict[str, pd.DataFrame], day: pd.Timestamp) -> pd.Timestamp:
    """Date the window is anchored on: the day of the robust median last report.

    No grace (R22): the grace only decides which inits are `complete`; applying
    it here would leave the newest inits never scored. `day` when no reports.
    """
    end = _obs_end(obs, cap=day + pd.Timedelta(days=1), grace_hours=0)
    return day if end is None else end.normalize()


def _network(
    start: pd.Timestamp, end: pd.Timestamp, reselect: bool
) -> pd.DataFrame | None:
    """The frozen network, selecting (and saving) one only when it must.

    Selection reads ~GHCNh for the candidates and a new list re-keys the
    station cube cache (a full re-read), so it happens only when none is saved,
    its window ended before `start`, the configured `SCORECARD_STATIONS` differ
    from the saved selection, or `reselect`.
    """
    saved = station_network.load_network()
    configured = tuple(sorted(SCORECARD_STATIONS))
    if saved is not None and not reselect and _naive_utc(saved[2]) >= start:
        if station_network.load_configured() == configured:
            return saved[0]
        why = "SCORECARD_STATIONS changed"
    else:
        why = (
            "reselect requested" if reselect else "none saved or its window is too old"
        )
    logger.info("Selecting the station network (%s)", why)
    net = station_network.select_network(start, end)
    station_network.save_network(net, start, end, configured)
    return net


def refresh(
    stores: list[str] | tuple[str, ...],
    variables: list[str] | tuple[str, ...],
    today: pd.Timestamp,
    *,
    backfill: bool = False,
    max_minutes: float | None = None,
    reselect: bool = False,
) -> int:
    """Score every missing or stale (store, variable, init) of the rolling window.

    Parameters
    ----------
    stores, variables : sequence of str
        `SCORECARD_STORES` and `SCORECARD_VARIABLES` keys. A variable with no
        station observation (Z500) is verified against its `reference` store's
        T+0 (`match_reference`) at every network station, 00 UTC valid times
        only. A store lacking a variable is skipped, logged, and recorded
        (`load_lacking`) so the page can name it.
    today : pandas.Timestamp
        The window is the SCORECARD_DAYS 00 UTC inits ending the day before the
        day of the median last report (GHCNh lags ~6 days), not today. Files
        for older inits are deleted.
    backfill : bool
        Oldest init first (a resumable walk of the whole window); else newest
        first.
    max_minutes : float, optional
        Stop cleanly between (store, variable, init) items once this much time
        has passed; the next call resumes, because skipping is by file state.
    reselect : bool
        Select a fresh station network even if the saved one is current.

    Returns
    -------
    int
        Init files written.

    Raises
    ------
    observations.ObservationError
        GHCNh is unreachable: there is nothing to score against.

    Notes
    -----
    An init is scored when its file is missing, its network hash differs, or it
    is incomplete and newer observations exist than it was scored with.
    Rescoring finds the cube in station_cube's disk cache, so it reads
    nothing. A `GridDataError` (init absent, over budget, incomplete run) is
    logged and the item retried next run.
    """
    started = _clock()
    day = pd.Timestamp(today).normalize()
    obs_start = day - pd.Timedelta(days=SCORECARD_DAYS + _OBS_LAG_ALLOWANCE_DAYS)

    todo_vars = list(variables)
    for variable in todo_vars:
        _spec(variable)  # an unknown variable raises before anything is read

    written = 0
    with status.activity("Scorecard"):
        # The window ends at the observation end (the feed lags), not today.
        # Observations are read once, from `obs_start` to today. With stations
        # known before selection (saved, else the fixed list) they are read
        # first and the window follows from their end; in the general mode with
        # nothing saved the network is selected on a window ending a week ago,
        # then read. A network changed by selection is read for its new ids.
        status.report("Reading observations")
        saved = station_network.load_network()
        same_list = station_network.load_configured() == tuple(
            sorted(SCORECARD_STATIONS)
        )
        if (
            saved is not None
            and not saved[0].empty
            and (same_list or not SCORECARD_STATIONS)
        ):
            probe_ids = [str(i) for i in saved[0]["id"]]
        else:
            probe_ids = list(SCORECARD_STATIONS)
        if probe_ids:
            obs = station_network.network_obs(probe_ids, obs_start, day)
            anchor = _anchor(obs, day)
            first, last = _window(anchor)
            status.report("Selecting stations")
            net = _network(first, anchor, reselect)
        else:
            obs = {}
            guess = day - pd.Timedelta(days=7)
            status.report("Selecting stations")
            net = _network(_window(guess)[0], guess, reselect)
        if net is None or net.empty:
            logger.warning("The station network is empty: nothing to score")
            return 0
        ids = [str(i) for i in net["id"]]
        missing_obs = [i for i in ids if i not in obs]
        if missing_obs:
            obs = {**obs, **station_network.network_obs(missing_obs, obs_start, day)}
        obs = {i: obs[i] for i in ids}
        if not probe_ids:
            anchor = _anchor(obs, day)
            first, last = _window(anchor)
        obs_end = _obs_end(obs, cap=day + pd.Timedelta(days=1))
        logger.info(
            "Scorecard window: inits %s to %s (observations anchor %s)",
            f"{first:%Y-%m-%d}",
            f"{last:%Y-%m-%d}",
            f"{anchor:%Y-%m-%d}",
        )
        removed = scores_db.prune_scorecard(first)
        if removed:
            logger.info(
                "Removed %d scorecard file(s) older than %s",
                removed,
                f"{first:%Y-%m-%d}",
            )

        subsets = {}
        for variable in todo_vars:
            column = station_network._OK_COLUMN.get(variable)
            # A gridded reference has no completeness screen: every station.
            subset = (net if column is None else net[net[column]]).reset_index(
                drop=True
            )
            if subset.empty:
                logger.info(
                    "Skipping %s: no station passed its completeness screen", variable
                )
                continue
            subsets[variable] = (subset, scores_db.station_hash(subset))

        wanted = pd.date_range(first, last, freq="D") + pd.Timedelta(
            hours=SCORECARD_INIT_HOUR
        )
        references = _ReferenceReader(subsets)
        plan, lacking = [], {}
        for store in stores:
            try:
                held = set(earthmover_api.available_inits(store))
            except GridDataError as err:
                logger.warning("Skipping %s: %s", store, err)
                continue
            for variable, (subset, net_hash) in subsets.items():
                grid_variable, _ = _grid_read(variable)
                try:
                    has = _store_has(store, grid_variable)
                except GridDataError as err:
                    logger.warning("Skipping %s: %s", store, err)
                    continue
                lacking.setdefault(variable, {})[store] = not has
                if not has:
                    logger.info(
                        "Skipping %s for %s: the store does not hold it",
                        variable,
                        store,
                    )
                    continue
                missing = [t for t in wanted if t not in held]
                if missing:
                    logger.info(
                        "%s %s: %d init(s) not in the store, skipped",
                        store,
                        variable,
                        len(missing),
                    )
                for init in wanted:
                    if init in held:
                        plan.append((init, store, variable))
        plan.sort(key=lambda item: (item[0], item[1], item[2]), reverse=not backfill)
        _save_window(anchor, lacking)

        for done, (init, store, variable) in enumerate(plan):
            if max_minutes is not None and (_clock() - started) / 60.0 >= max_minutes:
                logger.info(
                    "Time limit of %s min reached: %d of %d item(s) left for the "
                    "next run",
                    max_minutes,
                    len(plan) - done,
                    len(plan),
                )
                break
            status.report(f"{store} {variable} {init:%Y-%m-%d}", done, len(plan))
            subset, net_hash = subsets[variable]
            reference = _spec(variable)["obs"] is None
            # A reference variable's "observations" end at the newest archived
            # reference init: an init is rescored when references arrive.
            item_end = references.end(variable) if reference else obs_end
            meta = scores_db.scorecard_meta(store, variable, init)
            if not _needs_scoring(meta, net_hash, item_end):
                continue
            grid_variable, accum = _grid_read(variable)
            try:
                cube = earthmover_api.station_cube(
                    store,
                    grid_variable,
                    init,
                    subset,
                    accum,
                    budget_mb=SCORECARD_READ_BUDGET_MB,
                )
            except GridDataError as err:
                logger.warning(
                    "%s %s %s not scored (retried next run): %s",
                    store,
                    variable,
                    f"{init:%Y-%m-%d}",
                    err,
                )
                continue
            if reference:
                try:
                    refs, failed = references.get(variable, reference_valid_times(cube))
                except GridDataError as err:
                    logger.warning(
                        "%s %s %s not scored (retried next run): reference: %s",
                        store,
                        variable,
                        f"{init:%Y-%m-%d}",
                        err,
                    )
                    continue
                if failed:
                    # Scored without them, and stamped as verified only up to
                    # the first failed reference, so the next run (whose end is
                    # past it) rescores the init.
                    item_end = min(failed) - pd.Timedelta(hours=1)
                matched = match_reference(cube, refs, variable)
            else:
                matched = match_station_obs(cube, obs, variable)
            rows = pair_scores(matched, variable)
            # The last *scored* lead: a GEFS cube runs to 840 h, scored to 360 h.
            scored = matched["lead_hours"].values <= SCORECARD_MAX_LEAD_HOURS
            last_valid = pd.Timestamp(matched["valid_time"].values[scored].max())
            scores_db.save_scorecard_init(
                store,
                variable,
                init,
                rows,
                obs_end=item_end,
                network_hash=net_hash,
                complete=item_end is not None and last_valid <= item_end,
            )
            written += 1
        status.report("Done", len(plan), len(plan))
    # A pruned init leaves no newer file behind, so staleness alone misses it.
    if written or removed or _aggregates_stale():
        write_aggregates(anchor)
    return written


class _ReferenceReader:
    """T+0 reference cubes for one refresh, each init read at most once.

    Keyed by init (naive UTC); only inits the reference store holds are read,
    so a valid time past the archive simply has no reference. Only lead 0 is
    kept. A failed read (e.g. the newest run still being written) is
    remembered for the run and reported to every item that needs it.
    """

    def __init__(self, subsets: dict[str, tuple[pd.DataFrame, str]]) -> None:
        self._subsets = subsets
        self._held: dict[str, set[pd.Timestamp] | GridDataError] = {}
        self._cubes: dict[tuple[str, pd.Timestamp], xr.DataArray | GridDataError] = {}

    @staticmethod
    def store(variable: str) -> str:
        return EVAL_REFERENCES[_spec(variable)["reference"]]["store"]

    def held(self, variable: str) -> set[pd.Timestamp]:
        store = self.store(variable)
        if store not in self._held:
            try:
                self._held[store] = {
                    t
                    for t in earthmover_api.available_inits(store)
                    if t.hour == SCORECARD_INIT_HOUR
                }
            except GridDataError as err:
                self._held[store] = err
        held = self._held[store]
        if isinstance(held, GridDataError):
            raise held
        return held

    def end(self, variable: str) -> pd.Timestamp | None:
        """The newest reference init held, None when the store is unreadable."""
        try:
            held = self.held(variable)
        except GridDataError:
            return None
        return max(held) if held else None

    def get(
        self, variable: str, valid_times: list[pd.Timestamp]
    ) -> tuple[dict[pd.Timestamp, xr.DataArray], list[pd.Timestamp]]:
        """The reference cube of every held init among `valid_times`, and the
        held inits whose read failed (no reference for those valid times)."""
        held = self.held(variable)
        store = self.store(variable)
        grid_variable, accum = _grid_read(variable)
        subset = self._subsets[variable][0]
        out, failed = {}, []
        for valid in valid_times:
            if valid not in held:
                continue
            key = (variable, valid)
            if key not in self._cubes:
                try:
                    cube = earthmover_api.station_cube(
                        store,
                        grid_variable,
                        valid,
                        subset,
                        accum,
                        budget_mb=SCORECARD_READ_BUDGET_MB,
                    )
                    self._cubes[key] = cube.sel(lead_hours=cube["lead_hours"] == 0)
                except GridDataError as err:
                    logger.warning(
                        "%s %s reference %s not read: %s",
                        store,
                        variable,
                        f"{valid:%Y-%m-%d %H}",
                        err,
                    )
                    self._cubes[key] = err
            got = self._cubes[key]
            if isinstance(got, GridDataError):
                failed.append(valid)
            else:
                out[valid] = got
        return out, failed


def _anchor_path() -> Path:
    return scores_db.scorecard_root() / "window.json"


def _read_window() -> dict:
    try:
        state = json.loads(_anchor_path().read_text())
    except (OSError, ValueError):
        return {}
    return state if isinstance(state, dict) else {}


def _save_window(anchor: pd.Timestamp, lacking: dict[str, dict[str, bool]]) -> None:
    """Persist the window anchor and which stores lack which variable (atomic).

    `lacking` holds this run's findings, `{variable: {store: lacks}}`; stores
    and variables it does not mention keep their saved state.
    """
    saved = {v: set(s) for v, s in _read_window().get("lacking", {}).items()}
    for variable, found in lacking.items():
        stores = saved.setdefault(variable, set())
        stores |= {s for s, lacks in found.items() if lacks}
        stores -= {s for s, lacks in found.items() if not lacks}
    state = {
        "anchor": pd.Timestamp(anchor).strftime("%Y-%m-%d"),
        "lacking": {v: sorted(s, key=_store_order) for v, s in saved.items() if s},
    }
    path = _anchor_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(state))
    tmp.replace(path)


def _store_order(store: str) -> int:
    return (
        SCORECARD_STORES.index(store)
        if store in SCORECARD_STORES
        else len(SCORECARD_STORES)
    )


def load_lacking() -> dict[str, list[str]]:
    """Per variable, the stores the last refreshes found without it (e.g. GFS Z500)."""
    return {v: list(s) for v, s in _read_window().get("lacking", {}).items()}


def window_anchor(today: pd.Timestamp) -> pd.Timestamp:
    """The anchor `refresh` last used, so aggregation matches its window.

    Falls back to the saved network's window end, then to `today`.
    """
    anchor = _read_window().get("anchor")
    if anchor:
        try:
            return pd.Timestamp(anchor)
        except ValueError:
            pass
    saved = station_network.load_network()
    return (
        _naive_utc(saved[2]).normalize()
        if saved is not None
        else pd.Timestamp(today).normalize()
    )


# --- precomputed aggregates (what the pages read) ---------------------------------

AGGREGATES_RELPATH = "eval/scorecard/aggregates.parquet"
_AGG_META_KEY = b"scorecard"
STATION_LEAD_DAYS: tuple[int, ...] = (1, 3, 5, 10)
# Pair columns `aggregate` uses; the rest are dropped on load to bound memory.
_AGG_PAIR_COLUMNS: list[str] = [
    "station",
    "init",
    "lead_hours",
    "store",
    "variable",
    "err",
    "ae",
    "se",
    "crps",
    "spread2",
    "wet_fc",
    "wet_obs",
    "brier",
]


def aggregates_path() -> Path:
    """Where `write_aggregates` puts its single Parquet file.

    `PRECOMPUTE_DIR/eval/scorecard/aggregates.parquet`: the one scorecard file
    published to the Dataset and mirrored by the app (`hub_sync.SYNC_PATTERNS`).
    """
    return Path(PRECOMPUTE_DIR) / AGGREGATES_RELPATH


def _aggregates_stale() -> bool:
    """True when the aggregates file is missing or older than the newest init file."""
    path = aggregates_path()
    if not path.exists():
        return True
    newest = max(
        (p.stat().st_mtime for p in scores_db.scorecard_root().glob("*/*/*.parquet")),
        default=0.0,
    )
    return newest > path.stat().st_mtime


def write_aggregates(today: pd.Timestamp) -> Path:
    """Aggregate the stored window once, for every period, into one Parquet.

    Parameters
    ----------
    today : pandas.Timestamp
        Sets the window (`_window`): pass refresh's anchor (`window_anchor`).

    Returns
    -------
    pathlib.Path
        `aggregates_path()`; not written when there are no rows.

    Notes
    -----
    Long form with a `kind` column: `lead` (absolute, by store and lead day),
    `lead_relative` (the same as percent change against ifs_ens) and `station`
    (by store and station at lead days `STATION_LEAD_DAYS`, with lat/lon/name),
    each for every `period_days` in `SCORECARD_PERIODS`. Rows are read one
    variable at a time (all stores together, so common inits hold). Key-value
    metadata (`scorecard`): n_stations, start, end, generated, stores.
    Written to a temporary name and renamed.
    """
    first, last = _window(today)
    net = station_network.load_network()
    coords = None
    if net is not None and not net[0].empty:
        frame = net[0]
        coords = frame.assign(station=frame["id"].astype(str))[
            ["station", "lat", "lon", "name"]
        ]
    pieces, stores_present, stations, inits = [], set(), set(), []
    for variable in SCORECARD_VARIABLES:
        rows = scores_db.load_rows(SCORECARD_STORES, [variable], first, last)
        if rows.empty:
            continue
        rows = rows[_AGG_PAIR_COLUMNS]
        stores_present |= set(rows["store"].unique())
        stations |= set(rows["station"].astype(str).unique())
        inits += [rows["init"].min(), rows["init"].max()]
        for period in SCORECARD_PERIODS:
            lead = aggregate(rows, period, by=("store", "lead_day"))
            pieces.append(lead.assign(kind="lead", period_days=period))
            pieces.append(
                relative_to(lead).assign(kind="lead_relative", period_days=period)
            )
            # Whole window, all leads: rmse_bc/mae_bc subtract each (store,
            # station) mean error over every lead, as on the other tabs.
            by_station = aggregate(rows, period, by=("store", "lead_day", "station"))
            keep = by_station[by_station["lead_day"].isin(STATION_LEAD_DAYS)]
            if not keep.empty:
                pieces.append(keep.assign(kind="station", period_days=period))
        del rows
    path = aggregates_path()
    if not pieces:
        logger.warning("No scorecard rows to aggregate: %s not written", path)
        return path
    out = pd.concat(pieces, ignore_index=True)
    out["lead_day"] = out["lead_day"].astype("Int64")
    out["station"] = out["station"].astype("string")
    if coords is not None:
        out = out.merge(coords, on="station", how="left")
    else:
        out[["lat", "lon", "name"]] = np.nan
    meta = {
        "n_stations": len(stations),
        "start": f"{min(inits):%Y-%m-%d}",
        "end": f"{max(inits):%Y-%m-%d}",
        "generated": pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
        "stores": [s for s in SCORECARD_STORES if s in stores_present],
        "lacking": load_lacking(),
    }
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), _AGG_META_KEY: json.dumps(meta).encode()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    # `.part`: hub_sync never uploads one, should an upload overlap the write.
    tmp = path.with_suffix(f".{os.getpid()}.part")
    pq.write_table(table, tmp)
    os.replace(tmp, path)
    return path


@functools.lru_cache(maxsize=2)
def _read_aggregates(path: str, mtime_ns: int) -> tuple[pd.DataFrame, dict]:
    """Read the file; `mtime_ns` is only the cache key."""
    table = pq.read_table(path)
    meta = json.loads((table.schema.metadata or {}).get(_AGG_META_KEY, b"{}"))
    return table.to_pandas(), meta


def load_aggregates() -> tuple[pd.DataFrame, dict] | None:
    """The precomputed aggregates and their metadata, or None when never written.

    Memoized on the file's mtime, so a rewrite is picked up without a restart.
    Treat the returned frame as read-only. Mirrors the Dataset first (at most
    once per `PRECOMPUTE_SYNC_S`; nothing without `PRECOMPUTE_HF_REPO`).
    """
    hub_sync.sync()
    path = aggregates_path()
    try:
        mtime = path.stat().st_mtime_ns
    except OSError:
        return None
    return _read_aggregates(str(path), mtime)
