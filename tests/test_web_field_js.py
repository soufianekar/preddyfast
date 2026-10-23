import itertools
import json
import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from utils.globe import encode16

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "web" / "js" / "field.js"

pytestmark = pytest.mark.skipif(not shutil.which("node"), reason="node not installed")


def _nan_to_null(x):
    if isinstance(x, float) and math.isnan(x):
        return None
    if isinstance(x, (list, tuple)):
        return [_nan_to_null(i) for i in x]
    return x


def _js(fn_name: str, *args):
    """Call an exported function of field.js with JSON args; NaN <-> null."""
    args = _nan_to_null(list(args))
    prog = f"""
    import * as f from "{JS.as_uri()}";
    const fix = (x) => Array.isArray(x) ? x.map(fix) : x === null ? NaN : x;
    const args = fix({json.dumps(args)});
    let r = f.{fn_name}(...args);
    if (ArrayBuffer.isView(r)) r = Array.from(r);
    const out = (x) => Array.isArray(x) ? x.map(out)
        : (typeof x === "number" && Number.isNaN(x)) ? null : x;
    console.log(JSON.stringify(out(r)));
    """
    res = subprocess.run(
        ["node", "--input-type=module", "-e", prog],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(res.stdout)


def test_decode_matches_python_encoding():
    v = np.array([[-80.0, 12.3, np.nan, 51.0]])
    rgba = encode16(v, -80.0, 0.002)
    out = _js("decode16", rgba.ravel().tolist(), -80.0, 0.002)
    expected = [None if np.isnan(x) else float(x) for x in v.ravel()]
    assert [None if o is None else pytest.approx(o, abs=1e-3) for o in out] == [
        None if e is None else pytest.approx(e, abs=1e-3) for e in expected
    ]
    assert out[2] is None


def test_band_index_matches_default_palette_semantics():
    p = json.loads((ROOT / "web/data/palettes.json").read_text())["default_2t"]
    for v in (-60, -48, -47.9, 0, 55.9, 56, 80, float("nan")):
        expected = (
            -1
            if math.isnan(v) or v < p["levels"][0]
            else min(
                int(np.searchsorted(p["levels"], v, side="right")) - 1,
                len(p["colors"]) - 1,
            )
        )
        assert _js("bandIndex", v, p["levels"], len(p["colors"])) == expected


def test_band_index_closed_top_is_unshaded():
    assert _js("bandIndex", 5, [0, 1, 2], 2) == -1
    assert _js("bandIndex", 1.5, [0, 1, 2], 2) == 1


def test_mercator_rows_are_monotonic_and_bounded():
    rows = _js("mercatorRows", 512)
    assert rows[0] < 85.0511 and rows[-1] > -85.0511
    assert all(a > b for a, b in itertools.pairwise(rows))
    assert rows[0] == pytest.approx(-rows[-1])


GRID = {"nx": 4, "ny": 3, "lat0": 90.0, "lon0": -180.0, "res_deg": 90.0}


def test_sample_at_wraps_longitude_across_the_seam():
    values = list(range(12))  # row-major: row r, col c -> 4r + c
    a = _js("sampleAt", values, GRID, 180.2, 0.0)
    b = _js("sampleAt", values, GRID, -179.8, 0.0)
    assert a == b == 4 * 1 + 0
    assert _js("sampleAt", values, GRID, 179.9, 90.0) == 0  # nearest wraps to col 0
    assert _js("sampleAt", values, GRID, 0.0, -200.0) == 4 * 2 + 2  # row clamped


def test_colourise_colours_bands_and_leaves_gaps_clear():
    grid = {"nx": 2, "ny": 2, "lat0": 90.0, "lon0": -180.0, "res_deg": 180.0}
    bands = {"levels": [0, 10], "colors": ["#ff0000", "#00ff00"]}
    # rows lat 90 and -90; outHeight 2 -> output rows lat ~ +/-85 -> source rows 0, 1
    values = [5.0, 20.0, -3.0, None]
    out = _js("colourise", values, grid, bands, 2)
    assert len(out) == 2 * 2 * 4
    assert out[0:4] == [255, 0, 0, 255]  # 5 -> first band
    assert out[4:8] == [0, 255, 0, 255]  # 20 -> open top
    assert out[8:12] == [0, 0, 0, 0]  # below lowest
    assert out[12:16] == [0, 0, 0, 0]  # NaN
