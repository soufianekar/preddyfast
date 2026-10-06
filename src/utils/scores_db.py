"""Persisted per-run verification scores: the store any period is aggregated from.

One Parquet file per (reference, model, variable, site), rows from
`verification.per_run_scores`: one per (init, lead). Rescoring a run replaces
its rows. A site is a station id (or, later, a grid cell key). Files are
written whole to a temporary name and renamed, so a reader in another worker
never sees half a file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pandas as pd

from utils.settings import SCORES_DIR

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
