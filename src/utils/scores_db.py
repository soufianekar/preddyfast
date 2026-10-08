"""Persisted per-run verification scores: the store any period is aggregated from.

One Parquet file per (reference, model, variable, site), rows from
`verification.per_run_scores`: one per (init, lead). Rescoring a run replaces
its rows. A site is a station id (or, later, a grid cell key). Files are
written whole to a temporary name and renamed, so a reader in another worker
never sees half a file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from utils import hub_sync
from utils.settings import SCORECARD_DIR, SCORES_DIR

_KEY = ["init_time", "lead_hours"]


def _safe(part: str) -> str:
    """A path component that cannot leave the scores directory."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", part).lstrip(".") or "_"


def _path(reference: str, store: str, variable: str, site: str) -> Path:
    parts = (_safe(reference), _safe(store), _safe(variable), f"{_safe(site)}.parquet")
    return Path(SCORES_DIR).joinpath(*parts)


def load_per_run(
    reference: str,
    store: str,
    variable: str,
    site: str,
    since: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Stored per-run rows, optionally from `since` (init time) on; empty if none."""
    hub_sync.sync()
    path = _path(reference, store, variable, site)
    if not path.exists():
        return pd.DataFrame(columns=_KEY)
    rows = pd.read_parquet(path)
    if since is not None:
        rows = rows[rows["init_time"] >= pd.Timestamp(since)]
    return rows.reset_index(drop=True)


def save_per_run(
    reference: str, store: str, variable: str, site: str, rows: pd.DataFrame
) -> None:
    """Merge per-run rows into the store; a rescored (init, lead) replaces the old row."""
    if rows.empty:
        return
    path = _path(reference, store, variable, site)
    old = load_per_run(reference, store, variable, site)
    merged = pd.concat([old, rows], ignore_index=True) if not old.empty else rows.copy()
    merged["init_time"] = pd.to_datetime(merged["init_time"])
    merged = (
        merged.drop_duplicates(_KEY, keep="last").sort_values(_KEY).reset_index(drop=True)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.part")
    merged.to_parquet(tmp, index=False)
    tmp.replace(path)


# --- Station scorecard: one file per (store, variable, init) ----------------------
#
# SCORECARD_DIR/<store>/<variable>/<YYYYMMDD>.parquet holds the rows of
# `scorecard.pair_scores` plus `store` and `variable` columns. The file's own
# bookkeeping lives in its Parquet key-value metadata (key `meteox_scorecard`,
# a JSON object), so it can never drift from the rows it describes and is read
# without touching the data pages:
#   obs_end        newest observation time used (naive UTC ISO), or null
#   network_hash   `station_hash` of the stations scored
#   complete       last lead's valid time <= obs_end (no lead awaits observations)
#   n_pairs        rows written
#   dropped_nan_fc pairs not scored for a non-finite forecast

_SCORECARD_KEY = b"meteox_scorecard"


def scorecard_root() -> Path:
    """The scheduled job's scorecard state: init files, network, window.

    Outside `PRECOMPUTE_DIR`, so it is never uploaded or mirrored; the workflow
    keeps it between runs with actions/cache. Only the aggregates are published.
    """
    return Path(SCORECARD_DIR)


def station_hash(stations: pd.DataFrame) -> str:
    """Short hash of station ids and coordinates (4 dp), in the order given.

    The same notion as `earthmover_api.station_cube`'s cache key (ids plus
    rounded coordinates), so a changed network changes both.
    """
    key = json.dumps(
        [
            [str(i), round(float(la), 4), round(float(lo), 4)]
            for i, la, lo in zip(
                stations["id"], stations["lat"], stations["lon"], strict=True
            )
        ]
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def scorecard_path(store: str, variable: str, init: pd.Timestamp) -> Path:
    """File holding one init's rows for a (store, variable)."""
    return (
        scorecard_root()
        / _safe(store)
        / _safe(variable)
        / f"{pd.Timestamp(init):%Y%m%d}.parquet"
    )


def save_scorecard_init(
    store: str,
    variable: str,
    init: pd.Timestamp,
    rows: pd.DataFrame,
    *,
    obs_end: pd.Timestamp | None,
    network_hash: str,
    complete: bool,
) -> Path:
    """Write one init's rows and metadata atomically, replacing any earlier file.

    `rows` is `pair_scores` output; `store` and `variable` columns are added and
    `rows.attrs["dropped_nan_fc"]` is recorded. The file is written to a
    temporary name and renamed, so a reader never sees half a file.
    """
    path = scorecard_path(store, variable, init)
    out = rows.copy()
    out["store"] = store
    out["variable"] = variable
    meta = {
        "obs_end": None if obs_end is None else pd.Timestamp(obs_end).isoformat(),
        "network_hash": network_hash,
        "complete": bool(complete),
        "n_pairs": len(out),
        "dropped_nan_fc": int(rows.attrs.get("dropped_nan_fc", 0)),
    }
    table = pa.Table.from_pandas(out, preserve_index=False)
    table = table.replace_schema_metadata(
        {**(table.schema.metadata or {}), _SCORECARD_KEY: json.dumps(meta).encode()}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.part")
    pq.write_table(table, tmp)
    tmp.replace(path)
    return path


def scorecard_meta(store: str, variable: str, init: pd.Timestamp) -> dict | None:
    """The file's metadata dict (see above), or None if there is no readable file."""
    path = scorecard_path(store, variable, init)
    if not path.exists():
        return None
    try:
        raw = (pq.read_schema(path).metadata or {}).get(_SCORECARD_KEY)
    except (OSError, pa.ArrowInvalid):
        return None
    return json.loads(raw) if raw else None


def load_rows(
    stores: list[str] | tuple[str, ...],
    variables: list[str] | tuple[str, ...],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    """Scorecard pair rows for several stores and variables over a range of inits.

    This is the interface the aggregation (Task 6) consumes.

    Parameters
    ----------
    stores, variables : sequence of str
        `SCORECARD_STORES` keys and `SCORECARD_VARIABLES` keys.
    start, end : pandas.Timestamp
        Inclusive range of init dates (the time of day is ignored).

    Returns
    -------
    pandas.DataFrame
        The union of every matching file: `scorecard.PAIR_COLUMNS` plus `store`
        and `variable`. Empty (columns only) when nothing matches.
    """
    first, last = pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
    frames = []
    for store in stores:
        for variable in variables:
            folder = scorecard_root() / _safe(store) / _safe(variable)
            for path in sorted(folder.glob("*.parquet")):
                try:
                    day = pd.Timestamp(path.stem)
                except ValueError:
                    continue
                if first <= day <= last:
                    frame = pd.read_parquet(path)
                    if not frame.empty:
                        frames.append(frame)
    if not frames:
        from utils.scorecard import PAIR_COLUMNS  # local: scorecard imports this module

        return pd.DataFrame(columns=[*PAIR_COLUMNS, "store", "variable"])
    return pd.concat(frames, ignore_index=True)


def prune_scorecard(before: pd.Timestamp) -> int:
    """Delete init files dated before `before`; returns how many were removed."""
    cutoff = pd.Timestamp(before).normalize()
    removed = 0
    for path in scorecard_root().glob("*/*/*.parquet"):
        try:
            day = pd.Timestamp(path.stem)
        except ValueError:
            continue
        if day < cutoff:
            path.unlink(missing_ok=True)
            removed += 1
    return removed
