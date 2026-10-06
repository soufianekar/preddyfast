"""Glue for the Evaluation page: archived forecasts at a station vs its obs.

Reads and matching only; the scores themselves live in `verification.py`.
Each model is read, matched and scored on its own — never pooled.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.earthmover_api import (
    GridDataError,
    available_inits,
    check_read_budget,
    eval_inits,
    imerg_halfhours,
    physical_stores,
    point_archive,
    point_run_cached,
    read_estimate,
    run_calendar,
    store_geometry,
    store_variables,
)
from utils.observations import (
    ObservationError,
    fetch_station_year,
    last_obs_time,
    precip_inventory,
    reported_totals,
    station_list,
)
from utils.scores_db import save_per_run
from utils.settings import (
    EVAL_COLD_READS_MAX,
    EVAL_INIT_HOUR,
    EVAL_LEAD_STEP_DEFAULT,
    EVAL_REFERENCES,
    EVAL_RUNS_DEFAULT,
    EVAL_VARIABLES,
    GRID_RESOLUTION_DEG,
    GRID_STORES,
)
from utils.text import in_sentence
from utils.verification import (
    aggregate_scores,
    common_leads,
    imerg_window_totals,
    match_obs,
    obs_window_total,
    per_run_scores,
    sample_leads,
)

# Longest lead any archived store carries; obs are read this far past the
# newest run so every verifiable lead has its observation.
_MAX_LEAD = pd.Timedelta(days=15)


def _grid_cell(lat: float, lon: float, res: float = GRID_RESOLUTION_DEG) -> tuple[float, float]:
    """The cell centre a point read lands on (as `fetch_point_ensemble`)."""
    lat_c = float(np.clip(np.round(lat / res) * res, -90.0, 90.0))
    lon_c = float(((np.round(lon / res) * res + 180.0) % 360.0) - 180.0)
    return lat_c, lon_c


def _run_chunks(store: str, lat: float, lon: float) -> int:
    """Chunks one past run's point read downloads: 1 for IFS, 3 for GEFS (lead blocks)."""
    lat_c, lon_c = _grid_cell(lat, lon, GRID_STORES[store]["res_deg"])
    return max(1, read_estimate((store,), (lat_c, lat_c, lon_c, lon_c), n_inits=1)[0])


def _station_frame(station_id: str, inits: list[pd.Timestamp]) -> pd.DataFrame:
    """Every observed column over every year the runs and their leads touch."""
    now_year = pd.Timestamp.now(tz="UTC").year
    first, last = min(inits), max(inits) + _MAX_LEAD
    years = [y for y in range(first.year, last.year + 1) if y <= now_year]
    frames = []
    for year in years:
        try:
            frames.append(fetch_station_year(station_id, year))
        except ObservationError:
            if year == now_year:
                raise
    if not frames:
        return pd.DataFrame(columns=["valid_time"])
    return pd.concat(frames, ignore_index=True)


def _observations(station_id: str, inits: list[pd.Timestamp], column: str) -> pd.DataFrame:
    """The station's `column` over every year the runs and their leads touch."""
    obs = _station_frame(station_id, inits)
    if column not in obs:
        return pd.DataFrame(columns=["valid_time", "value"])
    return obs[["valid_time", column]].rename(columns={column: "value"})


def store_leads(store: str) -> np.ndarray:
    """A store's lead hours (metadata only)."""
    return np.asarray(store_geometry(store)["lead_hours"], dtype=float)


def reference_inits(store: str) -> list[pd.Timestamp]:
    """A store's finished inits, oldest first: the T+0 states a reference can use."""
    unfinished = set(run_calendar(store)["unfinished"])
    return [t for t in available_inits(store) if f"{t:%Y-%m-%dT%H:%M}" not in unfinished]


def _reference_store(reference: str, store: str) -> str:
    """The store whose T+0 verifies `store`: a fixed one (IFS control) or its own."""
    return EVAL_REFERENCES[reference].get("store") or store


def _t0_reference(
    store: str, variable: str, lat: float, lon: float, inits: list[pd.Timestamp]
) -> pd.DataFrame:
    """`valid_time, value`: each run's initial state at the cell nearest the point.

    An ensemble's T+0 is its member mean (the perturbed initial states centre
    on the unperturbed analysis); the control and deterministic models have
    one member.
    """
    if not inits:
        return pd.DataFrame(columns=["valid_time", "value"])
    df = point_archive(store, variable, lat, lon, sorted(inits), None)
    t0 = df[df["lead_hours"] == 0]
    return t0.groupby("valid_time", as_index=False)["value"].mean()


def evaluate_point(
    station: dict,
    eval_variable: str,
    stores: tuple[str, ...],
    n_runs: int,
    hour: int | None = EVAL_INIT_HOUR,
    max_cold: int = EVAL_COLD_READS_MAX,
    reference: str = "stations",
    lead_step: int | None = EVAL_LEAD_STEP_DEFAULT,
) -> dict:
    """Score each model's recent runs at a station, per lead.

    Parameters
    ----------
    station : dict
        A row of `observations.available_stations` (`id, name, lat, lon,
        elev_m, dist_km`).
    eval_variable : str
        Key into `EVAL_VARIABLES`.
    stores : tuple[str, ...]
        `GRID_STORES` keys; only inits all of them hold are used.
    n_runs : int
        How many of the newest common finished runs to score. Against
        stations, only runs that start no later than the station's last
        report of the variable (GHCNh lags by days).
    hour : int or None
        Init hour kept (00 UTC by default); None keeps every cycle.
    max_cold : int
        Uncached chunks allowed in this call (an IFS point run is 1, GEFS 3),
        reference runs included. Runs are taken newest first and only while
        every read they need fits, so all models are scored on the same runs;
        the rest are reported in `pending` for the next call. The newest run
        is always taken, even alone over the cap, so a request always makes
        progress.
    reference : str
        Key into `EVAL_REFERENCES`: station reports, the IFS control's T+0
        (one reference for every model), or each model's own T+0.
    lead_step : int or None
        Keep leads on this grid (6, 12, 24 h) that every model has; None keeps
        every lead of each model.

    Returns
    -------
    dict
        `results`: `{store: {"scores", "per_run", "matched"}}`; `errors`:
        per-store messages; `inits` (the runs scored); `pending` (runs not
        yet read); `grid_point`; `obs` (the reference series as scored);
        `obs_end` (last station report, None for gridded references);
        `reference`; `spec`.

    Raises
    ------
    GridDataError
        No selected model holds the variable, no common run, an accumulated
        variable against a T+0 reference, or the request is over budget.
    ObservationError
        The station's file cannot be read, or it never reports the variable.
    """
    spec = EVAL_VARIABLES[eval_variable]
    lat, lon = round(float(station["lat"]), 2), round(float(station["lon"]), 2)
    thresholds = tuple(spec.get("thresholds", ()))
    accum = spec["accum_hours"]
    imerg = reference == "imerg"
    gridded = reference not in ("stations", "imerg")  # a model's T+0
    if imerg and not accum:
        raise GridDataError(
            "IMERG verifies precipitation only: choose 24 h precipitation, or another reference."
        )
    if reference == "stations" and spec["obs"] is None:
        raise GridDataError(
            f"No station observes {in_sentence(spec['label'])}: verify it against a T+0 reference."
        )
    if gridded and accum:
        raise GridDataError(
            f"{spec['label']} cannot be verified against T+0: a run's precipitation "
            "at T+0 is zero. Use stations."
        )

    # A model without the variable would only cost reads and a blank line.
    errors, held = [], []
    for store in stores:
        try:
            has = spec["grid"] in store_variables(store)
        except GridDataError as exc:
            errors.append(f"{GRID_STORES[store]['label']}: {exc}")
            continue
        if has:
            held.append(store)
        else:
            errors.append(f"{GRID_STORES[store]['label']} does not archive {in_sentence(spec['label'])}.")
    if not held:
        raise GridDataError(" ".join(errors) or "No selected model archives this variable.")
    stores = tuple(held)

    obs_end = None
    if reference != "stations":
        inits = eval_inits(stores, n_runs, hour, before=None)
    else:
        columns = [spec["obs"]] + ([f"precip_total_{accum}h"] if accum else [])
        ends = [t for t in (last_obs_time(station["id"], c) for c in columns) if t is not None]
        obs_end = max(ends) if ends else None
        if obs_end is None:
            raise ObservationError(
                f"{station['name']} has no {in_sentence(spec['label'])} reports to verify against."
            )
        inits = eval_inits(stores, n_runs, hour, before=obs_end)
    if not inits:
        raise GridDataError(
            f"The selected models have no finished {hour:02d} UTC run in common"
            + (f" up to the last observation ({obs_end:%Y-%m-%d %H:%M} UTC)." if obs_end else ".")
            if hour is not None
            else "The selected models have no finished run in common."
        )
    lat_c, lon_c = _grid_cell(lat, lon, GRID_STORES[stores[0]]["res_deg"])
    grid_points = {s: _grid_cell(lat, lon, GRID_STORES[s]["res_deg"]) for s in stores}

    # Reference runs a forecast run needs: one T+0 per sampled valid time.
    refs = {s: _reference_store(reference, s) for s in stores} if gridded else {}
    ref_available = {r: set(reference_inits(r)) for r in set(refs.values())}
    sample = lead_step or 1

    def ref_reads(init: pd.Timestamp) -> list[tuple[str, pd.Timestamp]]:
        out = []
        for store, ref in refs.items():
            for lead in store_leads(store):
                valid = init + pd.Timedelta(hours=float(lead))
                if lead > 0 and lead % sample == 0 and valid in ref_available[ref]:
                    out.append((ref, valid))
        return out

    # A view (IFS Control) is cut from its source's extract: one read for both.
    physical = physical_stores(stores + tuple(refs.values()))
    chunks = {s: _run_chunks(s, lat, lon) for s in physical}
    var = spec["grid"]
    # `seen`: (store, init) runs to read, as named; `read`: the physical runs
    # behind them, so a run two reads share is paid for once.
    loaded, seen, read, cold, cold_runs = [], set(), set(), 0, 0
    for init in inits:
        wanted = [(s, init) for s in stores] + ref_reads(init)
        new = {(physical_stores((s,))[0], t) for s, t in wanted} - read
        cost = sum(
            chunks[s]
            for s, t in new
            if not point_run_cached(lat, lon, var, f"{t:%Y-%m-%dT%H:%M}", accum, s)
        )
        if cold + cost > max_cold and loaded:
            break
        cold += cost
        cold_runs += cost > 0
        seen.update(wanted)
        read |= new
        loaded.append(init)
    pending = len(inits) - len(loaded)
    if not loaded:
        raise GridDataError("No run fits this request's read limit — try fewer models.")
    inits = loaded
    # Only this request's cold runs download anything; cached runs are free, and
    # the rest wait in `pending` for the next request.
    if cold_runs:
        check_read_budget(physical, (lat_c, lat_c, lon_c, lon_c), n_inits=cold_runs)

    precip_source, offset, sample_step = None, 0.0, lead_step
    if gridded:
        ref_series = {
            ref: _t0_reference(ref, var, lat, lon, [t for s, t in seen if s == ref])
            for ref in set(refs.values())
        }
        # One reference line only when every model shares it; several models'
        # own initial states are not one series.
        obs = next(iter(ref_series.values())) if len(ref_series) == 1 else None
    elif imerg:
        last_valid = max(inits) + _MAX_LEAD
        halfhours = imerg_halfhours(lat, lon, min(inits), last_valid)
        obs = None
    else:
        frame = _station_frame(station["id"], inits)
        obs = (
            frame[["valid_time", spec["obs"]]].rename(columns={spec["obs"]: "value"})
            if spec["obs"] in frame
            else pd.DataFrame(columns=["valid_time", "value"])
        )
        if accum:
            # Only windows the station can supply: complete hours, or its own totals.
            inventory = precip_inventory(frame, accum) if not frame.empty else []
            if not inventory:
                raise ObservationError(
                    f"{station['name']} has no complete {accum} h precipitation window "
                    f"(hourly reports have gaps and it reports no {accum} h totals). "
                    "Verify against IMERG, or pick a station marked as verifiable."
                )
            precip_source = inventory[0]
            if precip_source["source"] == "reported":
                # One total a day, ending at the station's hour: align the leads to it.
                totals = reported_totals(frame, accum)
                ends = pd.to_datetime(totals["valid_time"], utc=True)
                obs = totals[ends.dt.hour == precip_source["end_hour"]].reset_index(drop=True)
                sample_step = 24
                offset = float((precip_source["end_hour"] - inits[0].hour) % 24)

    results, windows = {}, []
    for store in stores:
        try:
            fc = point_archive(store, var, lat, lon, inits, accum)
        except GridDataError as exc:
            errors.append(f"{GRID_STORES[store]['label']}: {exc}")
            continue
        if gridded:
            matched = match_obs(fc, ref_series[refs[store]], tolerance="1min")
        elif imerg:
            ends = sorted(pd.to_datetime(fc["valid_time"]).unique())
            totals = imerg_window_totals(halfhours, ends, accum)
            window_obs = pd.DataFrame({"valid_time": totals.index, "value": totals.to_numpy()})
            windows.append(window_obs)
            matched = match_obs(fc, window_obs, tolerance="1min")
        elif accum and precip_source and precip_source["source"] == "reported":
            windows.append(obs)
            matched = match_obs(fc, obs, tolerance="1min")
        elif accum:
            ends = sorted(pd.to_datetime(fc["valid_time"]).unique())
            totals = obs_window_total(obs, ends, accum)
            window_obs = pd.DataFrame({"valid_time": totals.index, "value": totals.to_numpy()})
            windows.append(window_obs)
            matched = match_obs(fc, window_obs, tolerance="1min")
        else:
            matched = match_obs(fc, obs)
        per_run = per_run_scores(matched, thresholds)
        results[store] = {
            "scores": aggregate_scores(per_run, thresholds),
            "per_run": per_run,
            "matched": matched,
        }

    if sample_step:
        # Models step differently: compare them only at leads they all have
        # (shifted to a station's window end where it reports its own totals).
        leads = common_leads({s: r["scores"] for s, r in results.items()}, sample_step, offset)
        for r in results.values():
            r["scores"] = sample_leads(r["scores"], leads)
            r["per_run"] = r["per_run"][r["per_run"]["lead_hours"].isin(leads)].reset_index(drop=True)

    if accum and windows:
        # What is plotted must be the same quantity as what was scored: the
        # observed window totals, not the hourly amounts they were built from.
        obs = pd.concat(windows, ignore_index=True)
    obs = (
        obs.drop_duplicates("valid_time").sort_values("valid_time").reset_index(drop=True)
        if obs is not None and not obs.empty
        else pd.DataFrame(columns=["valid_time", "value"])
    )
    return {
        "results": results,
        "errors": errors,
        "inits": inits,
        "pending": pending,
        "grid_point": (lat_c, lon_c),
        "grid_points": grid_points,
        "obs": obs,
        "obs_end": obs_end,
        "precip_source": precip_source,
        "reference": reference,
        "spec": spec,
    }


def station_precip_window(station_id: str, hours: int, days: int = 45) -> dict | None:
    """The station's best `hours` precipitation window over the last `days`, or None."""
    now = pd.Timestamp.now(tz="UTC")
    frame = _station_frame(station_id, [(now - pd.Timedelta(days=days)).tz_localize(None)])
    if frame.empty:
        return None
    recent = frame[pd.to_datetime(frame["valid_time"], utc=True) >= now - pd.Timedelta(days=days)]
    inventory = precip_inventory(recent, hours)
    return inventory[0] if inventory else None


def persist_scores(out: dict, station: dict, eval_variable: str) -> None:
    """Keep each model's per-run rows from `evaluate_point` (see `scores_db`)."""
    for store, result in out["results"].items():
        save_per_run(out["reference"], store, eval_variable, station["id"], result["per_run"])


def refresh_scores(station_ids: tuple[str, ...], stores: tuple[str, ...]) -> int:
    """Score fixed stations against every reference that applies, and keep the rows.

    For the precompute job: no per-request read cap (the job has no gunicorn
    timeout), so each evaluation reads all it needs in one pass. A station,
    variable or reference that cannot be scored is skipped. Returns the
    number of evaluations persisted.
    """
    listing = station_list().set_index("id")
    done = 0
    for station_id in station_ids:
        if station_id not in listing.index:
            continue
        station = {"id": station_id, **listing.loc[station_id].to_dict()}
        for variable, spec in EVAL_VARIABLES.items():
            if spec["accum_hours"]:
                references = ["stations", "imerg"]
            elif spec["obs"] is None:
                references = [r for r in EVAL_REFERENCES if r != "stations"]
            else:
                references = list(EVAL_REFERENCES)
            for reference in references:
                try:
                    out = evaluate_point(
                        station, variable, stores, EVAL_RUNS_DEFAULT,
                        reference=reference, max_cold=10_000,
                    )
                except (GridDataError, ObservationError):
                    continue
                persist_scores(out, station, variable)
                done += 1
    return done
