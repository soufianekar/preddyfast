"""Evaluation snapshots: finished score figures, drawn offline, served as JSON.

The full workbench scores any station on request, reading up to a few GB of
archived point runs. The public demo scores a fixed set of stations
(`PRECOMPUTE_STATIONS`) with fixed models (`EVAL_SNAPSHOT_STORES`) in the
scheduled job instead, and saves the two figures the page draws — scores by
lead for every score offered, and the newest runs — as Plotly JSON under
`PRECOMPUTE_DIR/eval/`, with an `index.json` naming what exists. The page only
loads them. JSON rather than pickles: nothing executable comes off the Dataset.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pandas as pd

from utils import hub_sync
from utils.settings import (
    EVAL_LEAD_STEP_DEFAULT,
    EVAL_REFERENCES,
    EVAL_RUNS_DEFAULT,
    EVAL_SNAPSHOT_STORES,
    EVAL_VARIABLES,
    GRID_STORES,
    GRID_VARIABLES,
    PRECOMPUTE_DIR,
)

_log = logging.getLogger(__name__)

_SUBDIR = "eval"
_INDEX = "index.json"

SCORE_OPTIONS = [
    {"value": "crps", "label": "CRPS"},
    {"value": "rmse", "label": "RMSE"},
    {"value": "mae", "label": "MAE"},
    {"value": "bias", "label": "Bias (mean error)"},
    {"value": "sd_error", "label": "SD of error"},
    {"value": "spread_skill", "label": "Spread / skill"},
]


def score_options(variable: str | None) -> list[dict]:
    """Scores offered for a variable: the WMO set, plus Brier per threshold where defined."""
    thresholds = EVAL_VARIABLES.get(variable or "", {}).get("thresholds", ())
    return SCORE_OPTIONS + [
        {"value": f"brier_{t:g}", "label": f"Brier ≥ {t:g} mm"} for t in thresholds
    ]


def usable_references(variable: str) -> list[str]:
    """References that apply to a variable.

    T+0 references are refused for accumulated variables (T+0 precipitation is
    zero), IMERG is precipitation only, and stations cannot verify a field no
    station observes (Z500).
    """
    spec = EVAL_VARIABLES[variable]
    accumulated = bool(spec.get("accum_hours"))
    out = []
    for key, meta in EVAL_REFERENCES.items():
        if key == "stations":
            ok = spec.get("obs") is not None
        elif meta.get("accumulated"):
            ok = accumulated
        else:
            ok = not accumulated
        if ok:
            out.append(key)
    return out


def _root() -> Path:
    return Path(PRECOMPUTE_DIR) / _SUBDIR


def figure_path(station: str, variable: str, reference: str, name: str) -> str:
    """Path of one figure, relative to `PRECOMPUTE_DIR`; `name` is a score or `runs`."""
    return f"{_SUBDIR}/{station}/{variable}/{reference}/{name}.json"


def _write(rel: str, text: str) -> None:
    path = Path(PRECOMPUTE_DIR) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.part")
    tmp.write_text(text)
    tmp.replace(path)


def load_index() -> dict:
    """What has been drawn: `{"stations": {id: {...}}, "generated": ...}`."""
    hub_sync.sync()
    path = _root() / _INDEX
    if not path.exists():
        return {"stations": {}}
    return json.loads(path.read_text())


def load_figure(station: str, variable: str, reference: str, name: str) -> dict | None:
    """A saved figure as a Plotly dict, or None if it was not drawn."""
    hub_sync.sync()
    path = Path(PRECOMPUTE_DIR) / figure_path(station, variable, reference, name)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _context(out: dict, station: dict, reference: str) -> tuple[dict, str]:
    """The figure context the page builds, and the unit."""
    spec = out["spec"]
    grid_meta = GRID_VARIABLES[spec["grid"]]
    unit = grid_meta.get("accum_unit", grid_meta["unit"]) if spec["accum_hours"] else grid_meta["unit"]
    context = {
        "station": f"{station['name']} ({station['id']}) · {station['elev_m']:.0f} m",
        "grid_point": out["grid_point"],
        "grid_points": out["grid_points"],
        "inits": out["inits"],
        "obs_end": out["obs_end"],
        "precip_source": out.get("precip_source"),
        "reference": reference,
        "unit": unit,
    }
    return context, unit


def draw(out: dict, station: dict, variable: str, reference: str) -> dict:
    """Write every figure for one evaluation; return its index entry."""
    from utils.figures import make_eval_runs_fig, make_scores_fig

    context, _ = _context(out, station, reference)
    spec = out["spec"]
    scores = {s: r["scores"] for s, r in out["results"].items()}
    drawn = []
    for option in score_options(variable):
        fig = make_scores_fig(scores, option["value"], spec, context)
        _write(figure_path(station["id"], variable, reference, option["value"]), fig.to_json())
        drawn.append(option["value"])
    runs = make_eval_runs_fig(
        {s: r["matched"] for s, r in out["results"].items()}, out["obs"], spec, context
    )
    _write(figure_path(station["id"], variable, reference, "runs"), runs.to_json())
    return {
        "scores": drawn,
        "runs": len(out["inits"]),
        "newest_init": f"{max(out['inits']):%Y-%m-%d %H} UTC" if out["inits"] else None,
        "errors": list(out["errors"]),
        "empty": all(df.empty for df in scores.values()),
    }


def refresh(station_ids: tuple[str, ...], stores: tuple[str, ...] = EVAL_SNAPSHOT_STORES) -> int:
    """Score each station on every variable and reference that applies, and draw it.

    For the precompute job: reads with no per-request cap. Per-run rows are
    persisted too (`scores_db`), so the archive is read once per run. Returns
    the number of evaluations drawn.
    """
    from utils.earthmover_api import GridDataError
    from utils.evaluation import evaluate_point, persist_scores
    from utils.observations import ObservationError, station_list

    listing = station_list().set_index("id")
    index = load_index()
    done = 0
    for station_id in station_ids:
        if station_id not in listing.index:
            _log.warning("evaluation: station %s is not in the GHCNh list", station_id)
            continue
        row = listing.loc[station_id]
        station = {"id": station_id, **row.to_dict(), "dist_km": 0.0}
        entry = {
            "name": row["name"],
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
            "elev_m": float(row["elev_m"]),
            "stores": [GRID_STORES[s]["label"] for s in stores],
            "store_keys": list(stores),
            "lead_step": EVAL_LEAD_STEP_DEFAULT,
            "variables": {},
        }
        for variable in EVAL_VARIABLES:
            for reference in usable_references(variable):
                try:
                    out = evaluate_point(
                        station, variable, stores, EVAL_RUNS_DEFAULT,
                        reference=reference, max_cold=10_000,
                    )
                except (GridDataError, ObservationError) as exc:
                    _log.warning("evaluation %s %s %s: %s", station_id, variable, reference, exc)
                    continue
                persist_scores(out, station, variable)
                entry["variables"].setdefault(variable, {})[reference] = draw(
                    out, station, variable, reference
                )
                done += 1
        index["stations"][station_id] = entry
    index["generated"] = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%dT%H:%M}Z"
    _write(f"{_SUBDIR}/{_INDEX}", json.dumps(index, indent=1))
    return done
