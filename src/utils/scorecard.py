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
import zlib
from collections.abc import Mapping
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
    obs_uncertainty,
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
    SCORECARD_BOOTSTRAP,
    SCORECARD_DAYS,
    SCORECARD_DETERMINISTIC_DRAWS,
    SCORECARD_INIT_HOUR,
    SCORECARD_LEAD_GROUPS,
    SCORECARD_LEAD_STEP_HOURS,
    SCORECARD_MAX_LEAD_HOURS,
    SCORECARD_MIN_INITS,
    SCORECARD_MODES,
    SCORECARD_OBS_GRACE_HOURS,
    SCORECARD_OBS_RECENT_DAYS,
    SCORECARD_PERIODS,
    SCORECARD_READ_BUDGET_MB,
    SCORECARD_RNG_SEED,
    SCORECARD_ROW_VERSION,
    SCORECARD_STATIONS,
    SCORECARD_STORES,
    SCORECARD_VARIABLES,
)

logger = logging.getLogger(__name__)


def threshold_key(t: float) -> str:
    """Column suffix of an event threshold: `g` format (1.0 -> "1", 0.5 -> "0.5")."""
    return f"{float(t):g}"


# Every event threshold of any variable: one column set per threshold, NaN for a
# variable that does not use it, so all stored rows share one layout.
_THRESHOLDS: tuple[float, ...] = tuple(
    sorted(
        {float(t) for v in SCORECARD_VARIABLES.values() for t in v["thresholds"] or ()}
    )
)
_THRESHOLD_PARTS: tuple[str, ...] = (
    "p",
    "p_pert",
    "wet_fc",
    "wet_obs",
    "brier",
    "brier_pert",
)
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
    # Height-corrected 2 m temperature (TM865 Eq. 2, ERA5 orography as proxy).
    "dz",
    "err_hc",
    "crps_hc",
    "rank_hc",
    # TM865 perturbed ensemble (Ferro 2017).
    "crps_pert",
    "spread2_pert",
    "err_pert",
    "rank",
    "rank_pert",
    "n_members",
    "n_members_pert",
    "obs_persist",
    *(f"{part}_{threshold_key(t)}" for t in _THRESHOLDS for part in _THRESHOLD_PARTS),
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
    a partial sum. With `hourly_fallback` False (24 h precipitation) a day with
    no reported total is NaN, whatever the hourly reports say.
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

    if spec.get("hourly_fallback", True) and spec["obs"] in frame:
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
        `fc_mean` (member mean), `obs` (NaN where none), `obs_persist` (the
        observation at `valid_time - persistence_offset(lead)`, matched by the
        same rule, NaN where none) and `valid_time`.
        Leads beyond `SCORECARD_MAX_LEAD_HOURS` are dropped, and for a window
        variable with `window_end_hour` those not ending then. Instants take the
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
    end_hour = spec.get("window_end_hour")
    if spec["window_hours"] and end_hour is not None:
        valid = pd.DatetimeIndex(pd.to_datetime(cube["valid_time"].values))
        ends = (valid.hour == end_hour) & (valid.minute == 0)
        cube = cube.isel(lead_hours=np.flatnonzero(ends))
    times = pd.Series(pd.to_datetime(cube["valid_time"].values))
    persist_times = times - persistence_offset(cube["lead_hours"].values)

    def _match(frame: pd.DataFrame, when: pd.Series) -> np.ndarray:
        if spec["window_hours"]:
            return _window_obs(frame, spec, when)
        return _instant_obs(frame, spec["obs"], when, spec["tolerance"])

    rows, persist = [], []
    for sid in cube["station"].values:
        frame = obs_by_station.get(str(sid), pd.DataFrame())
        rows.append(_match(frame, times))
        persist.append(_match(frame, persist_times))
    empty = np.empty((0, cube.sizes["lead_hours"]))
    obs = np.vstack(rows) if rows else empty
    obs_persist = np.vstack(persist) if persist else empty

    # station_cube is float32: a float32 member mean can land a few ulp below a
    # threshold the exact mean meets, so thresholds are applied in float64.
    fc = cube.transpose("station", "member", "lead_hours").astype("float64")
    ds = xr.Dataset(
        {
            "fc": fc,
            "fc_mean": fc.mean("member"),
            "obs": (("station", "lead_hours"), obs),
            "obs_persist": (("station", "lead_hours"), obs_persist),
        },
        attrs={**cube.attrs, "scorecard_variable": variable},
    )
    return ds


def persistence_offset(lead_hours: np.ndarray) -> pd.TimedeltaIndex:
    """How far back the persistence observation lies: 24·ceil(lead/24) h.

    The same time of day at or before the init (lead 6 h and 30 h of a 00 UTC
    run both use 18 UTC the day before), so persistence never sees the future.
    """
    days = np.ceil(np.asarray(lead_hours, dtype=float) / 24.0)
    return pd.to_timedelta(24.0 * days, unit="h")


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
            # A T+0 reference has no persistence forecast in the scorecard.
            "obs_persist": (("station", "lead_hours"), np.full_like(obs, np.nan)),
        },
        attrs={**cube.attrs, "scorecard_variable": variable},
    )


def _empty_pairs(dropped: int) -> pd.DataFrame:
    out = pd.DataFrame({c: pd.Series(dtype=float) for c in PAIR_COLUMNS})
    out.attrs["dropped_nan_fc"] = dropped
    return out


def item_rng(
    store: str, variable: str, init: pd.Timestamp | str
) -> np.random.Generator:
    """The generator of one (store, variable, init): reproducible, order-free.

    Seeded from `SCORECARD_RNG_SEED` and a CRC-32 of the item (never Python's
    salted `hash`), so a rescore draws the same perturbations and tie-breaks
    whichever items were processed before it.
    """
    key = f"{store}|{variable}|{pd.Timestamp(init):%Y-%m-%dT%H:%M}"
    seq = np.random.SeedSequence([SCORECARD_RNG_SEED, zlib.crc32(key.encode())])
    return np.random.default_rng(seq)


def _crps_rows(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """CRPS per row of `members` (pair, member): fair for M > 1, ECDF for M = 1.

    NaN where the observation or any member is not finite (the `scores` kernel
    would otherwise return a number for a NaN member).
    """
    m = members.shape[1]
    out = scores.probability.crps_for_ensemble(
        xr.DataArray(members, dims=(_PAIR, "member")),
        xr.DataArray(obs, dims=(_PAIR,)),
        "member",
        method="fair" if m > 1 else "ecdf",
        preserve_dims=[_PAIR],
    ).values
    ok = np.isfinite(members).all(axis=1) & np.isfinite(obs)
    return np.where(ok, out, np.nan)


def _brier_rows(
    members: np.ndarray, obs: np.ndarray, thresholds: list[float]
) -> np.ndarray:
    """Fair Brier score (`>=`) per row and threshold, shape (pair, threshold).

    NaN for one member (the fair correction divides by M - 1) and where the
    observation or any member is not finite.
    """
    nan = np.full((members.shape[0], len(thresholds)), np.nan)
    if members.shape[1] < 2 or not thresholds:
        return nan
    out = scores.probability.brier_score_for_ensemble(
        xr.DataArray(members, dims=(_PAIR, "member")),
        xr.DataArray(obs, dims=(_PAIR,)),
        "member",
        list(thresholds),
        preserve_dims=[_PAIR],
        fair_correction=True,
        event_threshold_operator=operator.ge,
    )
    out = out.transpose(_PAIR, "threshold").values.reshape(nan.shape)
    ok = np.isfinite(members).all(axis=1) & np.isfinite(obs)
    return np.where(ok[:, None], out, np.nan)


def _rank_rows(
    members: np.ndarray, obs: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Rank of the observation among the members, 0..M.

    Members below it count; ties take a uniformly random share (0..ties), so an
    exchangeable observation has a flat histogram even for zero-inflated
    precipitation. One draw per row whatever the data, so the stream (and the
    perturbations after it) does not depend on which rows are NaN.
    """
    o = obs[:, None]
    less = (members < o).sum(axis=1)
    ties = (members == o).sum(axis=1)
    rank = less + np.floor(rng.random(obs.size) * (ties + 1))
    ok = np.isfinite(members).all(axis=1) & np.isfinite(obs)
    return np.where(ok, rank, np.nan)


def _event_prob(members: np.ndarray, t: float) -> np.ndarray:
    """Member fraction `>= t`, NaN where any member is not finite."""
    ok = np.isfinite(members).all(axis=1)
    return np.where(ok, (members >= t).mean(axis=1), np.nan)


def pair_scores(
    matched: xr.Dataset,
    variable: str,
    dz: Mapping[str, float] | None = None,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """One row of score parts per (station, lead_hours, init) with an observation.

    Parameters
    ----------
    matched : xarray.Dataset
        From `match_station_obs` (or `match_reference`).
    variable : str
        `SCORECARD_VARIABLES` key; its `threshold`, `thresholds` and `scores`
        decide the contingency flags and the Brier scores.
    dz : mapping of str to float, optional
        Station id -> model minus station elevation (m), `station_dz`; used for
        2 m temperature only. A station missing (or NaN) here gets NaN
        height-corrected and perturbed temperature scores, never a zero
        correction.
    rng : numpy.random.Generator, optional
        Default `item_rng` of the matched attrs `store` and `init`.

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

        Temperature only: `dz`, and `err_hc` / `crps_hc` / `rank_hc` of the
        members plus `0.0065 * dz` (TM865 Eq. 2; `rank_hc` NaN for one member). Temperature, wind and 24 h precipitation:
        `crps_pert` (fair), `spread2_pert` (ddof=1) and `err_pert` (mean minus
        observation) of the TM865 perturbed ensemble at the store's
        `native_km` — temperature centred on the height-corrected value; a
        single member is expanded to `SCORECARD_DETERMINISTIC_DRAWS` draws.
        `rank` (0..`n_members`, NaN for one member) and `rank_pert`
        (0..`n_members_pert`) place the observation among the members, ties
        broken uniformly at random. `obs_persist` is the persistence
        observation of `match_station_obs`. Per threshold t of `thresholds`
        (suffix `threshold_key(t)`): `p_t`, `p_pert_t` member fractions
        `>= t`, `wet_fc_t` / `wet_obs_t` flags, `brier_t` (fair, ensembles
        only) and `brier_pert_t` (fair, perturbed). Columns a variable does
        not use are NaN.

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

    store = str(matched.attrs.get("store", ""))
    if rng is None:
        rng = item_rng(store, variable, matched.attrs["init"])

    # A plain integer pair index: scores' preserve_dims does not need the MultiIndex.
    station = pairs["station"].values[keep].astype(str)
    lead = pairs["lead_hours"].values[keep]
    sub = pairs.isel({_PAIR: keep}).drop_vars(["station", "lead_hours", _PAIR])
    sub = sub.assign_coords({_PAIR: np.arange(keep.size)})
    fc, fc_mean, obs = sub["fc"], sub["fc_mean"], sub["obs"]
    n_members = fc.sizes["member"]
    keep_dims = [_PAIR]
    members = fc.transpose(_PAIR, "member").values
    o = obs.values

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

    # Height correction (2 m temperature only): t* = t + 0.0065 dz.
    is_t2m = variable == "temperature_2m"
    dz_pair = (
        pd.Series(station).map(dict(dz)).to_numpy(dtype=float) if is_t2m and dz else nan
    )
    if is_t2m:
        shift = obs_uncertainty.LAPSE_K_PER_M * dz_pair
        crps_hc = _crps_rows(members + shift[:, None], o)
        err_hc = fc_mean.values + shift - o
    else:
        crps_hc = err_hc = nan

    # Draw order is fixed (raw ranks, perturbation, perturbed ranks, then
    # height-corrected ranks) so the same item always gets the same numbers.
    rank = _rank_rows(members, o, rng) if n_members > 1 else nan
    delta_km = GRID_STORES.get(store, {}).get("native_km")
    pert = None
    if variable in obs_uncertainty.VARIABLES and delta_km is not None:
        pert = obs_uncertainty.perturb(
            members,
            variable,
            float(delta_km),
            rng,
            dz_m=dz_pair if is_t2m else None,
            draws=SCORECARD_DETERMINISTIC_DRAWS if n_members == 1 else None,
        )
    elif variable in obs_uncertainty.VARIABLES:
        logger.warning("%s has no native_km: no perturbed scores", store)
    if pert is not None:
        pert_ok = np.isfinite(pert).all(axis=1)
        crps_pert = _crps_rows(pert, o)
        spread2_pert = np.where(pert_ok, pert.var(axis=1, ddof=1), np.nan)
        err_pert = np.where(pert_ok, pert.mean(axis=1), np.nan) - o
        rank_pert = _rank_rows(pert, o, rng)
        n_members_pert = np.full(keep.size, float(pert.shape[1]))
    else:
        crps_pert = spread2_pert = err_pert = rank_pert = n_members_pert = nan
    # Drawn last, so the raw ranks and the perturbations keep the stream they
    # had before rank_hc existed. NaN where dz is (the shifted members are).
    if is_t2m and n_members > 1:
        rank_hc = _rank_rows(members + shift[:, None], o, rng)
    else:
        rank_hc = nan

    per_threshold = {}
    used = [float(t) for t in spec["thresholds"] or ()]
    brier_t = _brier_rows(members, o, used)
    brier_pert_t = (
        _brier_rows(pert, o, used)
        if pert is not None
        else np.full_like(brier_t, np.nan)
    )
    for t in _THRESHOLDS:
        k = threshold_key(t)
        if t not in used:
            for part in _THRESHOLD_PARTS:
                per_threshold[f"{part}_{k}"] = nan
            continue
        j = used.index(t)
        per_threshold[f"p_{k}"] = _event_prob(members, t)
        per_threshold[f"p_pert_{k}"] = _event_prob(pert, t) if pert is not None else nan
        per_threshold[f"wet_fc_{k}"] = (fc_mean.values >= t).astype(float)
        per_threshold[f"wet_obs_{k}"] = (o >= t).astype(float)
        per_threshold[f"brier_{k}"] = brier_t[:, j]
        per_threshold[f"brier_pert_{k}"] = brier_pert_t[:, j]

    obs_persist = sub["obs_persist"].values if "obs_persist" in sub else nan

    out = pd.DataFrame(
        {
            "station": station,
            "init": pd.Timestamp(matched.attrs["init"]),
            "lead_hours": lead,
            "valid_time": pd.to_datetime(sub["valid_time"].values),
            "obs": o,
            "fc_mean": fc_mean.values,
            "err": err.values,
            "ae": ae.values,
            "se": se.values,
            "crps": crps.values,
            "spread2": spread2,
            "wet_fc": wet_fc,
            "wet_obs": wet_obs,
            "brier": brier,
            "dz": dz_pair if is_t2m else nan,
            "err_hc": err_hc,
            "crps_hc": crps_hc,
            "rank_hc": rank_hc,
            "crps_pert": crps_pert,
            "spread2_pert": spread2_pert,
            "err_pert": err_pert,
            "rank": rank,
            "rank_pert": rank_pert,
            "n_members": float(n_members),
            "n_members_pert": n_members_pert,
            "obs_persist": obs_persist,
            **per_threshold,
        },
        columns=PAIR_COLUMNS,
    )
    out.attrs["dropped_nan_fc"] = dropped
    return out


# --- aggregation -------------------------------------------------------------------

SCORE_COLUMNS: list[str] = list(SCORE_DIRECTION)
# Lead days of the station map and the rank histograms.
STATION_LEAD_DAYS: tuple[int, ...] = (1, 3, 5, 10)
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
    """ETS, HSS, frequency bias, POD and FAR from summed contingency counts.

    Takes summed hits, misses, false alarms and correct negatives.

    The dynamical.org formulas, identical to `scores.categorical.
    BinaryContingencyManager` (which takes events, not counts):
    ETS = (H - Hr) / (H + M + F - Hr), Hr = (H + M)(H + F) / n;
    HSS = 2(H·CN - F·M) / [(H + M)(M + CN) + (H + F)(F + CN)];
    FB = (H + F) / (H + M); POD = H / (H + M); FAR = F / (H + F) (false alarm
    ratio, not rate). A zero denominator gives NaN.
    """
    h, m, f, cn = (np.asarray(x, dtype=float) for x in (h, m, f, cn))
    n = h + m + f + cn
    hr = _div((h + m) * (h + f), n)
    return {
        "ets": _div(h - hr, h + m + f - hr),
        "hss": _div(2.0 * (h * cn - f * m), (h + m) * (m + cn) + (h + f) * (f + cn)),
        "frequency_bias": _div(h + f, h + m),
        "pod": _div(h, h + m),
        "far": _div(f, h + f),
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


# Scores computed from raw columns whatever the mode (Ruling V6): the
# contingency scores come from the raw ensemble mean's event flags. The
# aggregates metadata carries this table so a figure can say so.
_CONTINGENCY_RAW: tuple[str, ...] = (
    "ets",
    "hss",
    "frequency_bias",
    "pod",
    "far",
    *(
        f"{score}_{threshold_key(t)}"
        for t in _THRESHOLDS
        for score in ("ets", "fbi", "pod", "far")
    ),
)
MODE_RAW_SCORES: dict[str, tuple[str, ...]] = {
    mode: _CONTINGENCY_RAW for mode in SCORECARD_MODES
}


def mode_columns(variable: str, mode: str) -> dict[str, str | bool]:
    """Pair columns a display mode reads for one variable.

    Parameters
    ----------
    variable : str
        `SCORECARD_VARIABLES` key.
    mode : str
        One of `SCORECARD_MODES`.

    Returns
    -------
    dict
        `err`, `crps`, `spread2`, `brier` (headline threshold), `rank`,
        `n_members`, the templates `p` / `brier_t` (format with `k=` a
        `threshold_key`), and `applied`: False when the variable has no such
        mode and its raw columns stand in (height correction is 2 m temperature
        only; Z500 has no representativeness model).

    Notes
    -----
    Height-corrected: `err_hc`, `crps_hc` and `rank_hc` (members + 0.0065·dz
    against the observation); the spread is raw (a uniform shift does not
    change it), and so are the event probabilities and the reliability / ROC
    built on them. Representativeness: the TM865 perturbed ensemble for every
    part, with M = `n_members_pert`. The contingency scores (`wet_fc*` of the
    raw ensemble mean) are raw in every mode: `MODE_RAW_SCORES`.
    """
    if mode not in SCORECARD_MODES:
        raise ValueError(f"Unknown scorecard mode {mode!r}")
    cols: dict[str, str | bool] = {
        "err": "err",
        "crps": "crps",
        "spread2": "spread2",
        "brier": "brier",
        "rank": "rank",
        "n_members": "n_members",
        "p": "p_{k}",
        "brier_t": "brier_{k}",
        "applied": mode == "raw",
    }
    if mode == "height_corrected" and variable == "temperature_2m":
        cols.update(err="err_hc", crps="crps_hc", rank="rank_hc", applied=True)
    elif mode == "representativeness" and variable in obs_uncertainty.VARIABLES:
        headline = SCORECARD_VARIABLES.get(variable, {}).get("threshold")
        cols.update(
            err="err_pert",
            crps="crps_pert",
            spread2="spread2_pert",
            brier=(
                f"brier_pert_{threshold_key(headline)}"
                if headline is not None
                else "brier"
            ),
            rank="rank_pert",
            n_members="n_members_pert",
            p="p_pert_{k}",
            brier_t="brier_pert_{k}",
            applied=True,
        )
    return cols


def _store_members(store: str, perturbed: bool) -> float:
    """Member count of a store when a row file lacks it (older row versions)."""
    m = float(GRID_STORES.get(store, {}).get("members", 1))
    return float(SCORECARD_DETERMINISTIC_DRAWS) if perturbed and m <= 1 else m


def _prepare(rows: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Scored leads of `rows` with the mode's generic parts and the bias offset.

    Drops T+0 and leads off `SCORECARD_LEAD_STEP_HOURS` (R23), adds
    `lead_day`, and per variable the `mode_columns` as `_e`, `_se`, `_ae`,
    `_crps`, `_sp2`, `_brier`, `_rank`, `_m`, `_applied`, `_p_{k}`,
    `_brier_{k}`, the persistence error `_pe` (obs_persist - obs, never
    perturbed), and `_e_bc` / `_se_bc` / `_ae_bc` from each (store, station,
    variable) mean of `_e` over every row passed (the window, not the period).
    """
    missing = [c for c in PAIR_COLUMNS if c not in rows]
    if missing:
        rows = rows.assign(**{c: np.nan for c in missing})
    lead = rows["lead_hours"].to_numpy(dtype=float)
    on_step = np.isclose(np.mod(lead, SCORECARD_LEAD_STEP_HOURS), 0.0)
    rows = rows[(lead > 0) & on_step]
    keys = [threshold_key(t) for t in _THRESHOLDS]
    parts = []
    for variable, g in rows.groupby("variable", sort=False):
        cols = mode_columns(str(variable), mode)

        def col(name: str, g: pd.DataFrame = g) -> np.ndarray:
            return g[name].to_numpy(dtype=float)

        e = col(cols["err"])
        raw_err = cols["err"] == "err"
        m = col(cols["n_members"])
        no_m = ~np.isfinite(m)
        if no_m.any():
            perturbed = cols["n_members"] == "n_members_pert"
            fill = g["store"].map(lambda s, p=perturbed: _store_members(s, p))
            m = np.where(no_m, fill.to_numpy(dtype=float), m)
        add = {
            "_e": e,
            "_se": col("se") if raw_err else e**2,
            "_ae": col("ae") if raw_err else np.abs(e),
            "_crps": col(cols["crps"]),
            "_sp2": col(cols["spread2"]),
            "_brier": col(cols["brier"]),
            "_rank": col(cols["rank"]),
            "_m": m,
            "_applied": bool(cols["applied"]),
            "_pe": col("obs_persist") - col("obs"),
        }
        for k in keys:
            add[f"_p_{k}"] = col(str(cols["p"]).format(k=k))
            add[f"_brier_{k}"] = col(str(cols["brier_t"]).format(k=k))
        parts.append(g.assign(**add))
    if parts:
        rows = pd.concat(parts)
    else:
        extra = ["_e", "_se", "_ae", "_crps", "_sp2", "_brier", "_rank", "_m", "_pe"]
        extra += [f"_{p}_{k}" for k in keys for p in ("p", "brier")]
        rows = rows.assign(**{c: np.nan for c in extra}, _applied=False)
    rows["lead_day"] = np.ceil(rows["lead_hours"].to_numpy(dtype=float) / 24.0).astype(
        int
    )
    offset = rows.groupby(["store", "station", "variable"])["_e"].transform("mean")
    rows["_e_bc"] = rows["_e"] - offset
    rows["_se_bc"] = rows["_e_bc"] ** 2
    rows["_ae_bc"] = rows["_e_bc"].abs()
    return rows


def _scored_rows(
    rows: pd.DataFrame,
    period_days: int,
    mode: str,
    end: pd.Timestamp | None,
    common_inits: bool = True,
) -> pd.DataFrame:
    """`_prepare`, then the period cut and (optionally) the common-init filter."""
    rows = _prepare(rows, mode)
    if not rows.empty:
        rows = _select_period(rows, period_days, end)
    if common_inits and not rows.empty:
        rows = _keep_common_inits(rows)
    # A pair the mode cannot score (height correction without an elevation, a
    # row version without the perturbed columns) is not a pair of this mode.
    return rows[np.isfinite(rows["_e"].to_numpy(dtype=float))]


def _lead_group(lead_day: pd.Series) -> pd.Series:
    """`SCORECARD_LEAD_GROUPS` label ("1-3", "4-7", "8-15") of each lead day."""
    out = pd.Series(pd.NA, index=lead_day.index, dtype="string")
    for lo, hi in SCORECARD_LEAD_GROUPS:
        out[lead_day.between(lo, hi)] = f"{lo}-{hi}"
    return out


def _contingency_counts(
    rows: pd.DataFrame, fc: str, obs: str, prefix: str = ""
) -> pd.DataFrame:
    """Add hit / miss / false-alarm / correct-negative parts of two flag columns."""
    has = rows[fc].notna() & rows[obs].notna()
    wf = rows[fc].where(has)
    wo = rows[obs].where(has)
    return rows.assign(
        **{
            f"{prefix}h": wf * wo,
            f"{prefix}m": (1.0 - wf) * wo,
            f"{prefix}f": wf * (1.0 - wo),
            f"{prefix}cn": (1.0 - wf) * (1.0 - wo),
        }
    )


def _roc_table(rows: pd.DataFrame, keys: list[str], k: str) -> pd.DataFrame:
    """ROC points of every group at the member-fraction thresholds p >= j/M.

    Parameters
    ----------
    rows : pandas.DataFrame
        From `_prepare` (carries `_p_{k}`, `_m` and `wet_obs_{k}`).
    keys : list of str
        Group keys.
    k : str
        `threshold_key` of the event threshold.

    Returns
    -------
    pandas.DataFrame
        `keys`, `j` (0..M+1), `prob_threshold` (j/M), `pod`, `pofd`, `area`
        and `n`. j = 0 is (1, 1) and j = M + 1 is (0, 0); the area is the
        trapezoid through every point. Only rows of ensembles (M > 1) enter:
        a deterministic store has no ROC curve except as a perturbed ensemble.
        A group without both events and non-events has NaN POD/POFD/area.
    """
    cols = [*keys, "j", "prob_threshold", "pod", "pofd", "area", "n"]
    p = rows[f"_p_{k}"].to_numpy(dtype=float)
    o = rows[f"wet_obs_{k}"].to_numpy(dtype=float)
    m = rows["_m"].to_numpy(dtype=float)
    ok = np.isfinite(p) & np.isfinite(o) & np.isfinite(m) & (m > 1)
    if not ok.any():
        return pd.DataFrame(columns=cols)
    sub = rows.loc[ok, keys]
    gid = sub.groupby(keys, sort=True).ngroup().to_numpy()
    groups = sub.groupby(keys, sort=True).size().index.to_frame(index=False)
    n_g = int(gid.max()) + 1
    m_ok = m[ok].astype(int)
    count = np.rint(p[ok] * m[ok]).astype(int)
    event = o[ok] > 0.5
    width = int(m_ok.max()) + 2
    ev = np.zeros((n_g, width))
    ne = np.zeros((n_g, width))
    np.add.at(ev, (gid[event], count[event]), 1.0)
    np.add.at(ne, (gid[~event], count[~event]), 1.0)
    # hits(j) = events with count >= j: reverse cumulative sum over counts.
    hits = np.cumsum(ev[:, ::-1], axis=1)[:, ::-1]
    fas = np.cumsum(ne[:, ::-1], axis=1)[:, ::-1]
    n_ev, n_ne = hits[:, :1], fas[:, :1]
    pod = _div(hits, np.broadcast_to(n_ev, hits.shape))
    pofd = _div(fas, np.broadcast_to(n_ne, fas.shape))
    area = 0.5 * np.sum((pofd[:, :-1] - pofd[:, 1:]) * (pod[:, :-1] + pod[:, 1:]), 1)
    m_g = np.zeros(n_g, dtype=int)
    np.maximum.at(m_g, gid, m_ok)
    jj = np.arange(width)
    keep = jj[None, :] <= (m_g[:, None] + 1)
    g_idx, j_idx = np.nonzero(keep)
    out = groups.iloc[g_idx].reset_index(drop=True)
    out["j"] = j_idx
    out["prob_threshold"] = j_idx / m_g[g_idx]
    out["pod"] = pod[g_idx, j_idx]
    out["pofd"] = pofd[g_idx, j_idx]
    out["area"] = area[g_idx]
    out["n"] = (n_ev[:, 0] + n_ne[:, 0])[g_idx].astype(int)
    return out[cols]


def aggregate(
    rows: pd.DataFrame,
    period_days: int,
    by: tuple[str, ...] = ("store", "lead_day"),
    *,
    end: pd.Timestamp | None = None,
    common_inits: bool = True,
    min_n: int = 50,
    mode: str = "raw",
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
    mode : str
        One of `SCORECARD_MODES`: which error, CRPS, spread, Brier and
        probability columns are read (`mode_columns`). `mode_applied` is False
        for a variable the mode does not apply to (its raw scores are given).

    Returns
    -------
    pandas.DataFrame
        One row per group: `variable`, the `by` keys, `n` (pairs), `n_inits`,
        `n_stations`, `low_n`, `mode_applied`, and every column of
        `SCORE_COLUMNS`; a score the variable does not offer
        (`SCORECARD_VARIABLES[...]["scores"]` and `["extra_scores"]`) or that
        is undefined (zero denominator, deterministic spread or Brier) is NaN.

    Notes
    -----
    Pass the whole window as `rows`: with only a period's rows the bias
    correction (`rmse_bc`, `mae_bc`) uses that period's mean error.

    Scores are computed over all pairs of a group, each pair weighted equally:
    "averaged across initialisation times" with each init contributing its
    pairs, which are near-equal in number per init. RMSE is √(mean se), not a
    mean of per-init RMSEs. `brier` is the mean fair Brier score (per pair it
    may be slightly negative). `rmse_bc`/`mae_bc` subtract each (store,
    station, variable) mean of the mode's error over **all rows passed in**
    (T+0 excluded), so a 7-day card corrects with the same stable station
    offset as the 14-day one.

    `error_sd` = √max(mse − bias², 0). `spread_skill` = √(mean (M+1)/M ·
    spread2) / RMSE and `spread_error` the same over `error_sd` (TM865 slide
    19), with M per row `n_members` (or `n_members_pert` when perturbed),
    ideal 1; NaN for one member. Persistence (`obs_persist`, never perturbed):
    `rmse_ss_persist` = 1 − mse / mse_persist and `crpss_persist` = 1 − crps /
    mae_persist (a deterministic forecast's CRPS is its MAE), both over the
    pairs that have a persistence observation only.

    `mode_applied` refers to the error, CRPS, spread, Brier, rank and
    probability parts; the contingency scores are raw in every mode
    (`MODE_RAW_SCORES`, also in the aggregates metadata).

    ETS, HSS, frequency bias, POD and FAR come from the group's summed
    contingency counts (`contingency_scores`) at the headline threshold, and
    `ets_{t}`, `fbi_{t}`, `pod_{t}`, `far_{t}` per threshold, with `brier_{t}`
    the mean fair Brier and `roc_area_{t}` the trapezoid area of `_roc_table`
    (ensembles, or perturbed ensembles, only).
    """
    by = tuple(by)
    keys = ["variable", *[k for k in by if k != "variable"]]
    out_cols = [*keys, *_COUNT_COLUMNS, "mode_applied", *SCORE_COLUMNS]
    rows = _scored_rows(rows, period_days, mode, end, common_inits)
    if rows.empty:
        return pd.DataFrame(columns=out_cols)

    tkeys = [threshold_key(t) for t in _THRESHOLDS]
    rows = _contingency_counts(rows, "wet_fc", "wet_obs")
    for k in tkeys:
        rows = _contingency_counts(rows, f"wet_fc_{k}", f"wet_obs_{k}", f"t{k}_")
    m = rows["_m"].to_numpy(dtype=float)
    ens = m > 1
    factor = np.where(ens, (m + 1.0) / np.where(ens, m, 1.0), np.nan)
    pe = rows["_pe"].to_numpy(dtype=float)
    paired = np.isfinite(pe) & np.isfinite(rows["_e"].to_numpy(dtype=float))
    paired_crps = np.isfinite(pe) & np.isfinite(rows["_crps"].to_numpy(dtype=float))
    rows = rows.assign(
        _fsp=factor * rows["_sp2"].to_numpy(dtype=float),
        _se_pp=rows["_se"].where(paired),
        _sep=pd.Series(pe**2, index=rows.index).where(paired),
        _crps_pp=rows["_crps"].where(paired_crps),
        _aep=pd.Series(np.abs(pe), index=rows.index).where(paired_crps),
    )

    spec = {
        "n": ("_e", "count"),
        "n_inits": ("init", "nunique"),
        "n_stations": ("station", "nunique"),
        "mode_applied": ("_applied", "all"),
        "se": ("_se", "mean"),
        "mae": ("_ae", "mean"),
        "bias": ("_e", "mean"),
        "crps": ("_crps", "mean"),
        "brier": ("_brier", "mean"),
        "se_bc": ("_se_bc", "mean"),
        "mae_bc": ("_ae_bc", "mean"),
        "fsp": ("_fsp", "mean"),
        "se_pp": ("_se_pp", "mean"),
        "sep": ("_sep", "mean"),
        "crps_pp": ("_crps_pp", "mean"),
        "aep": ("_aep", "mean"),
        "h": ("h", "sum"),
        "m": ("m", "sum"),
        "f": ("f", "sum"),
        "cn": ("cn", "sum"),
        "n_cat": ("h", "count"),
    }
    for k in tkeys:
        spec[f"brier_{k}"] = (f"_brier_{k}", "mean")
        for part in ("h", "m", "f", "cn"):
            spec[f"t{k}_{part}"] = (f"t{k}_{part}", "sum")
        spec[f"t{k}_n_cat"] = (f"t{k}_h", "count")
    # pandas' groupby mean/sum skip NaN and give NaN for an all-NaN group, no warning.
    out = rows.groupby(keys, sort=True).agg(**spec).reset_index()

    out["rmse"] = np.sqrt(out["se"])
    out["rmse_bc"] = np.sqrt(out["se_bc"])
    bias2 = out["bias"].to_numpy(dtype=float) ** 2
    out["error_sd"] = np.sqrt(np.maximum(out["se"].to_numpy(dtype=float) - bias2, 0.0))
    spread = np.sqrt(out["fsp"].to_numpy(dtype=float))
    out["spread_skill"] = _div(spread, out["rmse"])
    out["spread_error"] = _div(spread, out["error_sd"])
    out["rmse_ss_persist"] = 1.0 - _div(out["se_pp"], out["sep"])
    out["crpss_persist"] = 1.0 - _div(out["crps_pp"], out["aep"])

    cat = contingency_scores(out["h"], out["m"], out["f"], out["cn"])
    no_cat = out["n_cat"].to_numpy() == 0
    for name, values in cat.items():
        out[name] = np.where(no_cat, np.nan, values)
    for k in tkeys:
        cat = contingency_scores(*(out[f"t{k}_{p}"] for p in ("h", "m", "f", "cn")))
        no_cat = out[f"t{k}_n_cat"].to_numpy() == 0
        for name, short in (
            ("ets", "ets"),
            ("frequency_bias", "fbi"),
            ("pod", "pod"),
            ("far", "far"),
        ):
            out[f"{short}_{k}"] = np.where(no_cat, np.nan, cat[name])
        roc = _roc_table(rows, keys, k)
        area = roc.drop_duplicates(keys)[[*keys, "area"]]
        out = out.merge(
            area.rename(columns={"area": f"roc_area_{k}"}), on=keys, how="left"
        )

    offered = out["variable"].map(
        lambda v: {
            *SCORECARD_VARIABLES.get(v, {}).get("scores", ()),
            *SCORECARD_VARIABLES.get(v, {}).get("extra_scores", ()),
        }
    )
    for score in SCORE_COLUMNS:
        mask = offered.map(lambda s, score=score: score in s).to_numpy(dtype=bool)
        if score not in out:
            out[score] = np.nan
        out[score] = out[score].where(mask)

    out["low_n"] = (out["n"] < min_n) | (out["n_inits"] < SCORECARD_MIN_INITS)
    return out[out_cols]


# Per-pair parts the bootstrap can test, and whether the difference is
# reported on the root (RMSE) scale of a mean square (Ruling V3).
_SIGNIFICANCE_PARTS: dict[str, tuple[str, bool]] = {
    "rmse": ("_se", True),
    "rmse_bc": ("_se_bc", True),
    "mae": ("_ae", False),
    "mae_bc": ("_ae_bc", False),
    "crps": ("_crps", False),
    "brier": ("_brier", False),
    **{
        f"brier_{threshold_key(t)}": (f"_brier_{threshold_key(t)}", False)
        for t in _THRESHOLDS
    },
}
SIGNIFICANCE_SCORES: tuple[str, ...] = tuple(_SIGNIFICANCE_PARTS)


@functools.lru_cache(maxsize=64)
def _block_indices(n: int, resamples: int, block: int) -> np.ndarray:
    """Circular moving-block resamples of 0..n-1, shape (resamples, n).

    Seeded from `SCORECARD_RNG_SEED` and n only, so every store and lead with
    the same number of runs is resampled with the same blocks (common random
    numbers), whatever the row order.
    """
    rng = np.random.default_rng([SCORECARD_RNG_SEED, n, block])
    n_blocks = -(-n // block)
    starts = rng.integers(0, n, size=(resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n
    idx = idx.reshape(resamples, n_blocks * block)[:, :n]
    idx.setflags(write=False)
    return idx


def significance(
    rows: pd.DataFrame,
    baseline: str = "ifs_ens",
    score: str = "rmse",
    period_days: int = SCORECARD_DAYS,
    mode: str = "raw",
    *,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Paired moving-block bootstrap of each store's score difference to a baseline.

    Parameters
    ----------
    rows : pandas.DataFrame
        Pair rows as for `aggregate` (the whole window: the bias-corrected
        scores use its station offsets).
    baseline : str
        Reference store.
    score : str
        One of `SIGNIFICANCE_SCORES` (scores that are means of per-pair parts).
    period_days : int
        As in `aggregate`; the common-init filter is applied the same way.
    mode : str
        One of `SCORECARD_MODES`.
    end : pandas.Timestamp, optional
        Last init date of the period.

    Returns
    -------
    pandas.DataFrame
        `variable`, `store`, `lead_day`, `n_inits` (common runs), `diff`
        (store minus baseline, in the score's unit), `lo`, `hi` (percentile
        interval at `SCORECARD_BOOTSTRAP["level"]`) and `sig` (1.0 when
        `lo > 0` or `hi < 0`, else 0.0). With fewer than `SCORECARD_MIN_INITS`
        common runs `lo`, `hi` and `sig` are NaN ("not assessed", never "not
        significant"); the baseline's own row has `diff` 0 and NaN `sig`.

    Notes
    -----
    Pairs are matched on (init, station, lead_hours) and kept only where both
    parts are finite. Each init contributes its summed part and pair count;
    a resample draws circular blocks of `block_days` consecutive inits
    (sorted) and pools their sums, so each pair keeps equal weight as in
    `aggregate`. RMSE and bias-corrected RMSE test the paired MSE and report
    √mse_store − √mse_base of each resample's pooled means (Ruling V3).
    """
    if score not in _SIGNIFICANCE_PARTS:
        raise ValueError(f"No significance test for {score!r}")
    part, root = _SIGNIFICANCE_PARTS[score]
    cols = ["variable", "store", "lead_day", "n_inits", "diff", "lo", "hi", "sig"]
    rows = _scored_rows(rows, period_days, mode, end)
    if rows.empty or not (rows["store"] == baseline).any():
        return pd.DataFrame(columns=cols)
    on = ["variable", "init", "station", "lead_hours"]
    sub = rows[[*on, "store", "lead_day", part]]
    sub = sub[np.isfinite(sub[part].to_numpy(dtype=float))]
    base = sub.loc[sub["store"] == baseline, [*on, part]]
    pairs = sub.merge(base, on=on, suffixes=("", "_base"))
    per_init = (
        pairs.assign(_x=pairs[part], _b=pairs[f"{part}_base"])
        .groupby(["variable", "store", "lead_day", "init"], sort=True)
        .agg(x=("_x", "sum"), b=("_b", "sum"), n=("_x", "size"))
        .reset_index()
    )
    boot = SCORECARD_BOOTSTRAP
    tail = 100.0 * (1.0 - boot["level"]) / 2.0
    out = []
    for (variable, store, lead_day), g in per_init.groupby(
        ["variable", "store", "lead_day"], sort=True
    ):
        x, b, n = (g[c].to_numpy(dtype=float) for c in ("x", "b", "n"))
        n_inits = x.size
        stat = (lambda s: np.sqrt(s)) if root else (lambda s: s)
        diff = float(stat(x.sum() / n.sum()) - stat(b.sum() / n.sum()))
        lo = hi = sig = np.nan
        if store == baseline:
            diff = 0.0
        elif n_inits >= SCORECARD_MIN_INITS:
            idx = _block_indices(n_inits, boot["resamples"], boot["block_days"])
            nn = n[idx].sum(axis=1)
            d = stat(x[idx].sum(axis=1) / nn) - stat(b[idx].sum(axis=1) / nn)
            lo, hi = np.percentile(d, [tail, 100.0 - tail])
            sig = float(lo > 0 or hi < 0)
        out.append((variable, store, int(lead_day), n_inits, diff, lo, hi, sig))
    return pd.DataFrame(out, columns=cols)


def rank_bin_weights(m: int, bins: int = 10) -> np.ndarray:
    """Share of each integer rank 0..m falling in each of `bins` equal bins.

    The normalised rank of r is the interval [r, r + 1) / (m + 1), centred on
    (r + 0.5)/(m + 1); its unit mass is split over the bins it overlaps.
    Exchangeable ranks (each of the m + 1 values equally likely) then give
    exactly 1/bins per bin for any m, including m + 1 not a multiple of
    `bins` (51 ranks for 50 members), which whole-rank binning cannot.
    Shape (m + 1, bins); each row sums to 1.
    """
    lo = np.arange(m + 1)[:, None] / (m + 1)
    hi = lo + 1.0 / (m + 1)
    edges = np.arange(bins + 1) / bins
    overlap = np.clip(
        np.minimum(hi, edges[None, 1:]) - np.maximum(lo, edges[None, :-1]), 0.0, None
    )
    return overlap * (m + 1)


def rank_histogram(
    rows: pd.DataFrame,
    period_days: int,
    lead_days: tuple[int, ...] = STATION_LEAD_DAYS,
    mode: str = "raw",
    *,
    end: pd.Timestamp | None = None,
    bins: int = 10,
) -> pd.DataFrame:
    """Rank histogram per store and lead day, in `bins` bins of rank / (M + 1).

    Parameters
    ----------
    rows : pandas.DataFrame
        Pair rows as for `aggregate`.
    period_days : int
        As in `aggregate` (common inits).
    lead_days : tuple of int
        Lead days kept (`STATION_LEAD_DAYS` by default).
    mode : str
        `rank` / `n_members`, or `rank_pert` / `n_members_pert` when perturbed
        (`mode_columns`; a deterministic store has ranks only then).
    end : pandas.Timestamp, optional
        Last init date of the period.
    bins : int
        Number of bins.

    Returns
    -------
    pandas.DataFrame
        `variable`, `store`, `lead_day`, `bin` (0..bins-1), `count` (fractional,
        `rank_bin_weights`), `n` (ranks), `freq` (count / n, flat = 1/bins),
        `n_inits`, `mode_applied`.
    """
    cols = ["variable", "store", "lead_day", "bin", "count", "n", "freq"]
    cols += ["n_inits", "mode_applied"]
    rows = _scored_rows(rows, period_days, mode, end)
    if rows.empty:
        return pd.DataFrame(columns=cols)
    r = rows["_rank"].to_numpy(dtype=float)
    m = rows["_m"].to_numpy(dtype=float)
    ok = np.isfinite(r) & np.isfinite(m) & (m > 1)
    ok &= rows["lead_day"].isin(lead_days).to_numpy()
    rows, r, m = rows[ok], r[ok], m[ok].astype(int)
    if rows.empty:
        return pd.DataFrame(columns=cols)
    weights = np.zeros((len(rows), bins))
    for mm in np.unique(m):
        sel = m == mm
        rank = np.clip(np.rint(r[sel]).astype(int), 0, mm)
        weights[sel] = rank_bin_weights(int(mm), bins)[rank]
    keys = ["variable", "store", "lead_day"]
    w = pd.DataFrame(weights, index=rows.index, columns=range(bins))
    frame = pd.concat([rows[[*keys, "init", "_applied"]], w], axis=1)
    g = frame.groupby(keys, sort=True)
    counts = g[list(range(bins))].sum()
    info = g.agg(
        n=("init", "size"),
        n_inits=("init", "nunique"),
        mode_applied=("_applied", "all"),
    )
    long = counts.stack().rename("count").reset_index()
    long = long.rename(columns={f"level_{len(keys)}": "bin"})
    long = long.merge(info.reset_index(), on=keys)
    long["freq"] = long["count"] / long["n"]
    return long[cols]


def reliability(
    rows: pd.DataFrame,
    period_days: int,
    threshold: float,
    mode: str = "raw",
    *,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Reliability table per store and lead-day group at one event threshold.

    Parameters
    ----------
    rows : pandas.DataFrame
        Pair rows as for `aggregate`.
    period_days : int
        As in `aggregate` (common inits).
    threshold : float
        One of the variable's `thresholds` (columns `p_{t}` / `wet_obs_{t}`).
    mode : str
        `p_{t}`, or `p_pert_{t}` when perturbed. A deterministic store enters
        only as a perturbed ensemble.
    end : pandas.Timestamp, optional
        Last init date of the period.

    Returns
    -------
    pandas.DataFrame
        `variable`, `store`, `lead_group` (`SCORECARD_LEAD_GROUPS`), `bin`
        (0..9: [0, 0.1), …, [0.9, 1.0], the last closed), `bin_lo`, `bin_hi`,
        `p_mean` (mean forecast probability), `obs_freq` (observed relative
        frequency), `count`, `n_inits`, `mode_applied`. Empty bins have no row.
    """
    k = threshold_key(threshold)
    cols = ["variable", "store", "lead_group", "bin", "bin_lo", "bin_hi", "p_mean"]
    cols += ["obs_freq", "count", "n_inits", "mode_applied"]
    rows = _scored_rows(rows, period_days, mode, end)
    if rows.empty:
        return pd.DataFrame(columns=cols)
    p = rows[f"_p_{k}"].to_numpy(dtype=float)
    o = rows[f"wet_obs_{k}"].to_numpy(dtype=float)
    m = rows["_m"].to_numpy(dtype=float)
    group = _lead_group(rows["lead_day"])
    ok = np.isfinite(p) & np.isfinite(o) & (m > 1) & group.notna().to_numpy()
    if not ok.any():
        return pd.DataFrame(columns=cols)
    # 1e-9: p = j/M times 10 can land a hair under an integer (0.7 * 10).
    b = np.minimum(np.floor(p[ok] * 10.0 + 1e-9), 9).astype(int)
    frame = rows.loc[ok, ["variable", "store", "init", "_applied"]].assign(
        lead_group=group[ok], bin=b, _p=p[ok], _o=o[ok]
    )
    out = (
        frame.groupby(["variable", "store", "lead_group", "bin"], sort=True)
        .agg(
            p_mean=("_p", "mean"),
            obs_freq=("_o", "mean"),
            count=("_p", "size"),
            n_inits=("init", "nunique"),
            mode_applied=("_applied", "all"),
        )
        .reset_index()
    )
    out["bin_lo"] = out["bin"] / 10.0
    out["bin_hi"] = (out["bin"] + 1) / 10.0
    return out[cols]


def roc(
    rows: pd.DataFrame,
    period_days: int,
    threshold: float,
    mode: str = "raw",
    *,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """ROC points (POD, POFD) at p >= j/M per store and lead-day group.

    Same inputs as `reliability`. Returns `variable`, `store`, `lead_group`,
    `j`, `prob_threshold`, `pod`, `pofd`, `area` (trapezoid including (0, 0)
    and (1, 1)), `n`, `mode_applied`; empty for a deterministic store in raw
    mode.
    """
    k = threshold_key(threshold)
    keys = ["variable", "store", "lead_group"]
    cols = [*keys, "j", "prob_threshold", "pod", "pofd", "area", "n", "mode_applied"]
    rows = _scored_rows(rows, period_days, mode, end)
    if rows.empty:
        return pd.DataFrame(columns=cols)
    rows = rows.assign(lead_group=_lead_group(rows["lead_day"]))
    rows = rows[rows["lead_group"].notna()]
    table = _roc_table(rows, keys, k)
    if table.empty:
        return pd.DataFrame(columns=cols)
    applied = rows.groupby(keys)["_applied"].all().rename("mode_applied")
    return table.merge(applied.reset_index(), on=keys)[cols]


def diurnal(
    rows: pd.DataFrame,
    period_days: int,
    mode: str = "raw",
    *,
    end: pd.Timestamp | None = None,
    min_n: int = 50,
) -> pd.DataFrame:
    """Bias and error SD per store, valid hour and lead-day group.

    Parameters
    ----------
    rows : pandas.DataFrame
        Pair rows as for `aggregate`.
    period_days : int
        As in `aggregate` (common inits).
    mode : str
        One of `SCORECARD_MODES` (the error column read).
    end : pandas.Timestamp, optional
        Last init date of the period.
    min_n : int
        As in `aggregate`, for `low_n`.

    Returns
    -------
    pandas.DataFrame
        `variable`, `store`, `lead_group`, `valid_hour` (UTC hour of init +
        lead: 00/06/12/18 for 00 UTC runs), `n`, `n_inits`, `low_n`,
        `mode_applied`, `bias`, `rmse`, `error_sd`.
    """
    keys = ["variable", "store", "lead_group", "valid_hour"]
    cols = [*keys, *_COUNT_COLUMNS[:2], "low_n", "mode_applied", "bias", "rmse"]
    cols.append("error_sd")
    rows = _scored_rows(rows, period_days, mode, end)
    if rows.empty:
        return pd.DataFrame(columns=cols)
    valid = rows["init"] + pd.to_timedelta(rows["lead_hours"], unit="h")
    rows = rows.assign(
        lead_group=_lead_group(rows["lead_day"]), valid_hour=valid.dt.hour
    )
    rows = rows[rows["lead_group"].notna()]
    out = (
        rows.groupby(keys, sort=True)
        .agg(
            n=("_e", "count"),
            n_inits=("init", "nunique"),
            mode_applied=("_applied", "all"),
            bias=("_e", "mean"),
            se=("_se", "mean"),
        )
        .reset_index()
    )
    out["rmse"] = np.sqrt(out["se"])
    var = out["se"].to_numpy(dtype=float) - out["bias"].to_numpy(dtype=float) ** 2
    out["error_sd"] = np.sqrt(np.maximum(var, 0.0))
    out["low_n"] = (out["n"] < min_n) | (out["n_inits"] < SCORECARD_MIN_INITS)
    return out[cols]


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
    """Score when the file is missing, built on another network or row layout
    (`SCORECARD_ROW_VERSION`), or has leads still awaiting observations that
    have since arrived."""
    if meta is None or meta.get("network_hash") != net_hash:
        return True
    if meta.get("row_version") != SCORECARD_ROW_VERSION:
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


def _network_dz(net: pd.DataFrame) -> dict[str, float]:
    """Station id -> ERA5 cell minus station elevation (m), NaN where unknown."""
    ids = [str(i) for i in net["id"]]
    if "elev_m" not in net:
        return dict.fromkeys(ids, np.nan)
    elev = {i: (None if pd.isna(e) else float(e)) for i, e in zip(ids, net["elev_m"])}
    return dict(zip(ids, station_dz(ids, elev)))


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
    An init is scored when its file is missing, its network hash or row version
    (`SCORECARD_ROW_VERSION`) differs, or it is incomplete and newer
    observations exist than it was scored with.
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
        removed += scores_db.prune_unconfigured_variables(tuple(SCORECARD_VARIABLES))
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
        dz = _network_dz(net)

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
                    window_end_hour=_spec(variable).get("window_end_hour"),
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
            rows = pair_scores(matched, variable, dz=dz)
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
                row_version=SCORECARD_ROW_VERSION,
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
    # Task 4 columns, for the height-corrected / TM865 / rank / persistence views.
    "obs",
    *PAIR_COLUMNS[PAIR_COLUMNS.index("dz") :],
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


def _variable_scores(variable: str) -> set[str]:
    spec = SCORECARD_VARIABLES[variable]
    return {*spec["scores"], *spec.get("extra_scores", ())}


def _relative_with_significance(
    rows: pd.DataFrame, lead: pd.DataFrame, variable: str, period: int, mode: str
) -> pd.DataFrame:
    """`relative_to(lead)` plus `diff_`, `lo_`, `hi_`, `sig_` per testable score."""
    rel = relative_to(lead)
    keys = ["variable", "store", "lead_day"]
    for score in SIGNIFICANCE_SCORES:
        if score not in _variable_scores(variable):
            continue
        sig = significance(rows, score=score, period_days=period, mode=mode)
        sig = sig[[*keys, "diff", "lo", "hi", "sig"]].rename(
            columns={c: f"{c}_{score}" for c in ("diff", "lo", "hi", "sig")}
        )
        rel = rel.merge(sig, on=keys, how="left")
    return rel


def _aggregate_pieces(
    rows: pd.DataFrame, variable: str, mode: str, period: int
) -> list[pd.DataFrame]:
    """Every kind of one variable, mode and period (see `write_aggregates`)."""
    tag = {"mode": mode, "period_days": period}
    lead = aggregate(rows, period, by=("store", "lead_day"), mode=mode)
    pieces = [lead.assign(kind="lead", **tag)]
    rel = _relative_with_significance(rows, lead, variable, period, mode)
    pieces.append(rel.assign(kind="lead_relative", **tag))
    # Whole window, all leads: rmse_bc/mae_bc subtract each (store, station)
    # mean error over every lead, as on the other tabs.
    by_station = aggregate(rows, period, by=("store", "lead_day", "station"), mode=mode)
    keep = by_station[by_station["lead_day"].isin(STATION_LEAD_DAYS)]
    pieces.append(keep.assign(kind="station", **tag))
    pieces.append(diurnal(rows, period, mode).assign(kind="diurnal", **tag))
    hist = rank_histogram(rows, period, STATION_LEAD_DAYS, mode)
    pieces.append(hist.assign(kind="rank", **tag))
    for t in SCORECARD_VARIABLES[variable]["thresholds"] or ():
        rel_t = reliability(rows, period, t, mode)
        pieces.append(rel_t.assign(kind="reliability", threshold=float(t), **tag))
        roc_t = roc(rows, period, t, mode)
        pieces.append(roc_t.assign(kind="roc", threshold=float(t), **tag))
    return [p for p in pieces if not p.empty]


def write_aggregates(today: pd.Timestamp) -> Path:
    """Aggregate the stored window once, for every mode and period, into one Parquet.

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
    Long form; every row has `kind`, `mode` (`SCORECARD_MODES`), `period_days`
    (`SCORECARD_PERIODS`), `variable` and `store`, and `mode_applied` (False
    where the variable has no such mode and its raw scores stand in). Kinds:

    - `lead`: `aggregate` by store and lead day.
    - `lead_relative`: the same as percent change against ifs_ens
      (`relative_to`), plus `diff_{s}`, `lo_{s}`, `hi_{s}`, `sig_{s}` from
      `significance` for each testable score s the variable offers.
    - `station`: by store and station at lead days `STATION_LEAD_DAYS`, with
      lat/lon/name.
    - `diurnal`: `diurnal` (`lead_group`, `valid_hour`).
    - `rank`: `rank_histogram` at `STATION_LEAD_DAYS` (`bin`, `count`, `freq`).
    - `reliability` and `roc`: per `threshold` of a variable with thresholds
      (`lead_group`, `bin`… / `j`, `pod`, `pofd`, `area`).

    Rows are read one variable at a time (all stores together, so common
    inits hold). Key-value metadata (`scorecard`): n_stations, start, end,
    generated, stores, lacking, modes, mode_raw_scores (`MODE_RAW_SCORES`),
    orography_source, and
    height_corrected_excluded (2 m temperature stations with no height
    correction: unknown elevation or no orography cell). Written to a
    temporary name and renamed.
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
    hc_excluded = 0
    for variable in SCORECARD_VARIABLES:
        rows = scores_db.load_rows(SCORECARD_STORES, [variable], first, last)
        if rows.empty:
            continue
        # reindex: a file of an older row version lacks the new columns (NaN).
        rows = rows.reindex(columns=_AGG_PAIR_COLUMNS)
        stores_present |= set(rows["store"].unique())
        stations |= set(rows["station"].astype(str).unique())
        inits += [rows["init"].min(), rows["init"].max()]
        if variable == "temperature_2m":
            has_dz = rows.groupby(rows["station"].astype(str))["dz"].count()
            hc_excluded = int((has_dz == 0).sum())
        for mode in SCORECARD_MODES:
            for period in SCORECARD_PERIODS:
                pieces += _aggregate_pieces(rows, variable, mode, period)
        del rows
    path = aggregates_path()
    if not pieces:
        logger.warning("No scorecard rows to aggregate: %s not written", path)
        return path
    out = pd.concat(pieces, ignore_index=True)
    # One schema whichever kinds had rows.
    for col in ("lead_day", "bin", "valid_hour", "j", "station", "lead_group"):
        if col not in out:
            out[col] = pd.NA
    for col in ("lead_day", "bin", "valid_hour", "j"):
        out[col] = pd.to_numeric(out[col]).astype("Int64")
    for col in ("station", "lead_group"):
        out[col] = out[col].astype("string")
    if "threshold" not in out:
        out["threshold"] = np.nan
    out["mode_applied"] = out["mode_applied"].astype("boolean")
    out["low_n"] = out["low_n"].astype("boolean")
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
        "modes": list(SCORECARD_MODES),
        "orography_source": orography_source(),
        "height_corrected_excluded": hc_excluded,
        "mode_raw_scores": {k: list(v) for k, v in MODE_RAW_SCORES.items()},
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
    frame = table.to_pandas()
    # A file written before the display modes holds raw scores only.
    if "mode" not in frame:
        frame["mode"] = "raw"
    if "mode_applied" not in frame:
        frame["mode_applied"] = True
    return frame, meta


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


# ---------------------------------------------------------------------------
# ERA5 orography at the station cells (built once by
# scripts/build_station_orography.py; the app never fetches it)
# ---------------------------------------------------------------------------
OROGRAPHY_PATH: Path = (
    Path(__file__).resolve().parent.parent / "data" / "scorecard_orography.json"
)


@functools.lru_cache(maxsize=1)
def _read_orography(path: Path) -> dict[str, float]:
    """Station id -> `z_m`; a cell whose `z_m` is null is left out. Never raises."""
    try:
        cells = json.loads(path.read_text())["cells"]
        return {
            str(sid): float(c["z_m"])
            for sid, c in cells.items()
            if c.get("z_m") is not None
        }
    except FileNotFoundError:
        logger.warning("No station orography at %s: height corrections are NaN", path)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as err:
        logger.warning("Unreadable station orography %s (%s)", path, err)
    return {}


@functools.lru_cache(maxsize=1)
def _read_orography_source(path: Path) -> str | None:
    """The orography file's `source` text, or None. Never raises."""
    try:
        source = json.loads(path.read_text()).get("source")
    except (OSError, ValueError, AttributeError) as err:
        logger.warning("No orography source from %s (%s)", path, err)
        return None
    return str(source) if source is not None else None


def orography_source() -> str | None:
    """Where the station-cell orography came from (`source` of the JSON)."""
    return _read_orography_source(OROGRAPHY_PATH)


def station_orography() -> dict[str, float]:
    """Station id -> ERA5 cell orography `z_m` (m); `{}` if the file is missing."""
    return _read_orography(OROGRAPHY_PATH)


def station_dz(ids: list[str], elev: dict[str, float]) -> np.ndarray:
    """`z_m - elev_m` per station id (m); NaN where either is missing.

    A positive value is a cell higher than the station (model too high). A
    None or NaN elevation counts as missing, never as 0 m.
    """
    z = station_orography()
    out = np.full(len(ids), np.nan)
    for i, sid in enumerate(ids):
        zm, e = z.get(sid), elev.get(sid)
        if zm is not None and e is not None and np.isfinite(float(e)):
            out[i] = zm - float(e)
    return out
