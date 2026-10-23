import io

import numpy as np
from PIL import Image

from utils.globe import decode16, encode16, isobars, png_bytes


def test_round_trip_is_within_half_a_step():
    v = np.array([[-80.0, -12.345, 0.0, 51.06]])
    out = decode16(encode16(v, -80.0, 0.002), -80.0, 0.002)
    assert np.allclose(out, v, atol=0.001)


def test_nan_is_the_missing_code_and_never_a_value():
    rgba = encode16(np.array([[np.nan, 0.0]]), 0.0, 0.005)
    assert tuple(rgba[0, 0]) == (255, 255, 0, 255)
    assert np.isnan(decode16(rgba, 0.0, 0.005)[0, 0])


def test_out_of_range_clamps_to_the_end_codes():
    rgba = encode16(np.array([[-500.0, 500.0]]), -80.0, 0.002)
    q = rgba[..., 0].astype(int) * 256 + rgba[..., 1]
    assert q.tolist() == [[0, 65534]]


def test_alpha_is_always_opaque():
    assert (encode16(np.random.rand(3, 4), 0.0, 0.01)[..., 3] == 255).all()


def test_png_is_lossless():
    rgba = encode16(np.random.rand(361, 720) * 100, 0.0, 0.005)
    assert (np.asarray(Image.open(io.BytesIO(png_bytes(rgba)))) == rgba).all()


def test_png_has_no_colour_profile_or_gamma():
    img = Image.open(io.BytesIO(png_bytes(encode16(np.zeros((2, 2)), 0.0, 1.0))))
    assert "icc_profile" not in img.info and "gamma" not in img.info


def _low_on_the_dateline():
    lat = np.arange(90, -90.5, -0.5)
    lon = np.arange(-180, 180, 0.5)
    lo, la = np.meshgrid(lon, lat)
    d = np.hypot((((lo - 180) + 180) % 360) - 180, la - 40)
    return 1010 - 30 * np.exp(-((d / 10) ** 2)), lat, lon


def test_isobars_close_across_the_antimeridian():
    field, lat, lon = _low_on_the_dateline()
    gj = isobars(field, lat, lon, 5.0)
    ring = [f for f in gj["features"] if f["properties"]["value"] == 990.0]
    # One -180...180 domain cuts the ring into pieces at its edges; the cyclic
    # column makes those pieces end exactly on +-180 at matching latitudes, so
    # on the globe they meet with no 0.5 degree gap.
    ends = [
        c
        for f in ring
        for c in (f["geometry"]["coordinates"][0], f["geometry"]["coordinates"][-1])
    ]
    east = sorted(round(y, 2) for x, y in ends if x == 180.0)
    west = sorted(round(y, 2) for x, y in ends if x == -180.0)
    assert east and east == west


def test_isobar_coordinates_are_rounded():
    field, lat, lon = _low_on_the_dateline()
    gj = isobars(field, lat, lon, 5.0)
    assert gj["features"]
    for f in gj["features"]:
        assert f["geometry"]["type"] == "LineString"
        assert f["properties"]["value"] % 5.0 == 0
        for c in f["geometry"]["coordinates"]:
            assert list(c) == [round(x, 2) for x in c]
