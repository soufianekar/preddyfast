"""requirements.txt carries what a fresh install (CI, the Space) needs."""

from pathlib import Path


def test_dask_is_a_requirement():
    # Without dask, xarray reads the Zarr stores unchunked: a fresh install
    # took one ensemble preset from ~1 GB to past a 16 GB runner.
    names = [line.split(">")[0].split("=")[0].split("[")[0].strip().lower()
             for line in (Path(__file__).resolve().parents[1] / "requirements.txt").read_text().splitlines()]
    assert "dask" in names
