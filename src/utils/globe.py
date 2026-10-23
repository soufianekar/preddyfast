"""Global-field export for the static globe page.

Data texture encoding: one 16-bit code per cell in an RGBA PNG. R is the high
byte, G the low byte, B is 0, and alpha is always 255 (premultiplied alpha
would corrupt R and G). ``q = round((v - offset) / scale)`` is clipped to
0..65534; 65535 is the missing code (NaN). Decoding is ``offset + q * scale``.
"""

import io

import contourpy
import numpy as np
from PIL import Image

MISSING = 65535


def encode16(values: np.ndarray, offset: float, scale: float) -> np.ndarray:
    """Pack an ``(ny, nx)`` float field into ``(ny, nx, 4)`` uint8 RGBA."""
    v = np.asarray(values, dtype=np.float64)
    nan = np.isnan(v)
    q = np.rint((np.where(nan, 0.0, v) - offset) / scale)
    q = np.clip(q, 0, MISSING - 1).astype(np.uint32)
    q[nan] = MISSING
    rgba = np.zeros(v.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = q >> 8
    rgba[..., 1] = q & 255
    rgba[..., 3] = 255
    return rgba


def decode16(rgba: np.ndarray, offset: float, scale: float) -> np.ndarray:
    """Inverse of :func:`encode16`; the missing code becomes NaN (float32)."""
    q = rgba[..., 0].astype(np.uint32) * 256 + rgba[..., 1].astype(np.uint32)
    out = (offset + q * scale).astype(np.float32)
    out[q == MISSING] = np.nan
    return out


def png_bytes(rgba: np.ndarray) -> bytes:
    """Lossless RGBA PNG, with no colour profile and no gamma chunk."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgba, dtype=np.uint8), "RGBA").save(
        buf, format="PNG", optimize=True
    )
    return buf.getvalue()


def isobars(
    values: np.ndarray, lat: np.ndarray, lon: np.ndarray, interval: float
) -> dict:
    """Isolines every multiple of ``interval`` as a GeoJSON FeatureCollection.

    A cyclic column (the first column again at ``lon[0] + 360``) is appended
    before contouring, so lines that cross the seam run on to +-180 at the
    same latitude on both sides instead of stopping a cell short.
    """
    v = np.asarray(values, dtype=np.float64)
    lon_x = np.append(lon, lon[0] + 360.0)
    z = np.concatenate([v, v[:, :1]], axis=1)
    z = np.ma.masked_invalid(z)
    if z.count() == 0:
        return {"type": "FeatureCollection", "features": []}
    gen = contourpy.contour_generator(lon_x, np.asarray(lat), z)
    first = int(np.ceil(float(z.min()) / interval))
    last = int(np.floor(float(z.max()) / interval))
    features = []
    for k in range(first, last + 1):
        level = k * interval
        for line in gen.lines(level):
            if len(line) < 2:
                continue
            coords = [[round(float(x), 2), round(float(y), 2)] for x, y in line]
            features.append(
                {
                    "type": "Feature",
                    "properties": {"value": float(level)},
                    "geometry": {"type": "LineString", "coordinates": coords},
                }
            )
    return {"type": "FeatureCollection", "features": features}
