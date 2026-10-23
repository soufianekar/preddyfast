"""The globe page's palettes, themes and coastlines are exported, not copied."""

from pathlib import Path

import build_web_data

WEB_DATA = Path(__file__).resolve().parents[1] / "web" / "data"
FILES = ("palettes.json", "themes.json", "coastlines.geojson")


def test_committed_web_data_matches_the_settings(tmp_path):
    written = build_web_data.build(tmp_path)
    assert sorted(p.name for p in written) == sorted(FILES)
    for name in FILES:
        assert (tmp_path / name).read_bytes() == (
            WEB_DATA / name
        ).read_bytes(), f"{name} is stale: run python scripts/build_web_data.py"


def test_coastlines_are_small():
    assert (WEB_DATA / "coastlines.geojson").stat().st_size < 1_000_000
