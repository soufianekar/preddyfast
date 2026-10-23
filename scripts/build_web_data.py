"""Export the settings the static globe page needs into `web/data/`.

The page (`web/`) is plain ES modules with no Python behind it, so the colours,
theme tokens and coastlines it draws are written out here from their single
sources: `MAP_PALETTES`, `THEMES` and `natural_earth_lines.npz`. The output is
committed and `tests/test_web_data.py` fails when it drifts.

    python scripts/build_web_data.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from components.theme_toggle import THEMES
from utils.figures import palette_levels
from utils.settings import GLOBE_LAYERS, GRID_VARIABLES, MAP_PALETTES

OUTPUT = ROOT / "web" / "data"
LINES = ROOT / "src" / "data" / "natural_earth_lines.npz"


def _dump(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n"


def palettes() -> dict[str, dict]:
    """Band edges and colours of every palette a shaded globe layer uses."""
    out: dict[str, dict] = {}
    for layer in GLOBE_LAYERS.values():
        if layer["kind"] != "shaded":
            continue
        key = GRID_VARIABLES[layer["variable"]]["palette"]
        palette = MAP_PALETTES[key]
        window = float(layer.get("accum_hours", 0))
        levels = palette_levels(palette, window)
        out[key] = {
            "levels": [float(f"{v:.6g}") for v in levels],
            "colors": palette["colors"].split("/"),
        }
    return out


def _multiline(lon: np.ndarray, lat: np.ndarray) -> list[list[list[float]]]:
    """Split NaN-separated lon/lat into polylines of rounded [lon, lat]."""
    lines: list[list[list[float]]] = []
    current: list[list[float]] = []
    for x, y in zip(lon.tolist(), lat.tolist()):
        if math.isnan(x) or math.isnan(y):
            if len(current) >= 2:
                lines.append(current)
            current = []
        else:
            current.append([round(x, 2), round(y, 2)])
    if len(current) >= 2:
        lines.append(current)
    return lines


def coastlines() -> dict:
    """110 m coastlines and borders as a two-feature FeatureCollection."""
    with np.load(LINES) as data:
        features = [
            {
                "type": "Feature",
                "properties": {"kind": kind},
                "geometry": {
                    "type": "MultiLineString",
                    "coordinates": _multiline(
                        data[f"{layer}_110_lon"], data[f"{layer}_110_lat"]
                    ),
                },
            }
            for kind, layer in (("coastline", "coastline"), ("border", "borders"))
        ]
    return {"type": "FeatureCollection", "features": features}


def build(out_dir: Path) -> list[Path]:
    """Write the three web data files into `out_dir` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "palettes.json": palettes(),
        "themes.json": THEMES,
        "coastlines.geojson": coastlines(),
    }
    paths = []
    for name, obj in payloads.items():
        path = out_dir / name
        path.write_text(_dump(obj))
        paths.append(path)
    return paths


def main() -> None:
    for path in build(OUTPUT):
        print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
