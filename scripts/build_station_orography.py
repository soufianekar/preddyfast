"""Build src/data/scorecard_orography.json: ERA5 orography at the station cells.

One CDS request (ERA5 single levels, geopotential, a single time: orography is
time-invariant) over the `SCORECARD_STATIONS` bounding box. Each station's
nearest 0.25 degree cell (the rounding rule of `earthmover_api._station_cells`)
is read, divided by g, and written with the station's own GHCNh elevation so
that `dz_m = z_m - elev_m` can be audited from the file alone.

    .venv/bin/python scripts/build_station_orography.py [--out PATH] [--dry-run]

Needs a CDS key in ~/.cdsapirc. The app never runs this.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from utils import observations  # noqa: E402
from utils.settings import SCORECARD_STATIONS  # noqa: E402

logger = logging.getLogger(__name__)

OUTPUT = (
    Path(__file__).resolve().parent.parent / "src" / "data" / "scorecard_orography.json"
)
DATASET = "reanalysis-era5-single-levels"
RES_DEG = 0.25
PAD_DEG = 0.5
G = 9.80665
SOURCE = "ERA5 invariant orography (geopotential/g), 0.25°"


def snap_cells(lat: np.ndarray, lon: np.ndarray, res: float = RES_DEG) -> tuple:
    """Nearest grid cell, by `earthmover_api._station_cells`' rule.

    Returns
    -------
    tuple
        `(lat_c, lon_c)` arrays: `round(x / res) * res`, longitude in [-180, 180).
    """
    lat_c = np.round(np.asarray(lat, dtype=float) / res) * res
    lon_c = (
        (np.round(np.asarray(lon, dtype=float) / res) * res + 180.0) % 360.0
    ) - 180.0
    return lat_c, lon_c


def request_area(
    lat: np.ndarray, lon: np.ndarray, pad: float = PAD_DEG, res: float = RES_DEG
) -> list[float]:
    """`[N, W, S, E]` of the stations, padded by `pad`, rounded outward to `res`."""
    north = math.ceil((float(np.max(lat)) + pad) / res) * res
    south = math.floor((float(np.min(lat)) - pad) / res) * res
    east = math.ceil((float(np.max(lon)) + pad) / res) * res
    west = math.floor((float(np.min(lon)) - pad) / res) * res
    return [north, west, south, east]


def build_request(area: list[float]) -> dict:
    """The CDS request body."""
    return {
        "product_type": ["reanalysis"],
        "variable": ["geopotential"],
        "year": ["2020"],
        "month": ["01"],
        "day": ["01"],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area,
    }


def extract_cells(ds: xr.Dataset, stations: pd.DataFrame, retrieved: str) -> dict:
    """The JSON payload from a retrieved dataset.

    Parameters
    ----------
    ds
        Holds `z` (m² s⁻²) on `latitude`/`longitude`, optionally with a
        `time`/`valid_time` dimension (dropped).
    stations
        `id, lat, lon, elev_m`.
    retrieved
        Retrieval date string.

    Raises
    ------
    ValueError
        A station's cell is not in the dataset.
    """
    z = ds["z"]
    for dim in ("valid_time", "time", "step", "number"):
        if dim in z.dims:
            z = z.isel({dim: 0}, drop=True)
    z = z / G
    lat_c, lon_c = snap_cells(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    cells = {}
    for sid, la, lo, elev in zip(stations["id"], lat_c, lon_c, stations["elev_m"]):
        try:
            value = float(
                z.sel(
                    latitude=la, longitude=lo, method="nearest", tolerance=RES_DEG / 2
                ).item()
            )
        except KeyError as err:
            raise ValueError(f"No ERA5 cell at {la}, {lo} for {sid}") from err
        cells[sid] = {
            "lat": float(la),
            "lon": float(lo),
            "z_m": round(value, 2),
            "elev_m": None if pd.isna(elev) else float(elev),
        }
    return {"source": SOURCE, "retrieved": retrieved, "cells": cells}


def station_frame() -> pd.DataFrame:
    """`SCORECARD_STATIONS` rows of the GHCNh station list (cached read)."""
    st = observations.station_list()
    st = st[st["id"].isin(SCORECARD_STATIONS)].set_index("id")
    missing = [s for s in SCORECARD_STATIONS if s not in st.index]
    if missing:
        raise SystemExit(f"Not in the GHCNh station list: {missing}")
    return st.loc[list(SCORECARD_STATIONS)].reset_index()[
        ["id", "lat", "lon", "elev_m"]
    ]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    stations = station_frame()
    lat_c, lon_c = snap_cells(stations["lat"].to_numpy(), stations["lon"].to_numpy())
    request = build_request(request_area(stations["lat"], stations["lon"]))
    print(json.dumps({"dataset": DATASET, "request": request}, indent=2))
    for sid, la, lo, el in zip(stations["id"], lat_c, lon_c, stations["elev_m"]):
        print(f"{sid}: cell {la:.2f}, {lo:.2f}  station elev {el} m")
    if args.dry_run:
        return

    import cdsapi  # function-local: only this script needs it

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "orography.nc"
        cdsapi.Client().retrieve(DATASET, request, str(target))
        with xr.open_dataset(target) as ds:
            payload = extract_cells(ds.load(), stations, str(pd.Timestamp.now().date()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("Wrote %s (%d stations)", args.out, len(payload["cells"]))


if __name__ == "__main__":
    main()
