// Decode 16-bit data-texture PNG pixels and colour them with a stepped palette.
// Pure functions, no DOM: the encoding mirrors src/utils/globe.py (encode16).

const MISSING = 65535;

export function decode16(rgba, offset, scale) {
  const n = rgba.length >> 2;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const q = rgba[4 * i] * 256 + rgba[4 * i + 1];
    out[i] = q === MISSING ? NaN : offset + q * scale;
  }
  return out;
}

// levels[i] <= v < levels[i+1] -> i. Below levels[0] or NaN -> -1. At or above the
// last level: the last colour when there is one colour per level (open top), else -1.
export function bandIndex(value, levels, nColors) {
  if (!(value >= levels[0])) return -1;
  let lo = 0;
  let hi = levels.length;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (levels[mid] <= value) lo = mid + 1;
    else hi = mid;
  }
  const i = lo - 1;
  if (i === levels.length - 1) return nColors === levels.length ? nColors - 1 : -1;
  return Math.min(i, nColors - 1);
}

// Latitude of each output row's centre, evenly spaced in Web Mercator y.
export function mercatorRows(outHeight, latMax = 85.0511) {
  const rad = Math.PI / 180;
  const yMax = Math.log(Math.tan(Math.PI / 4 + (latMax * rad) / 2));
  const rows = new Float64Array(outHeight);
  for (let r = 0; r < outHeight; r++) {
    const y = yMax * (1 - (2 * (r + 0.5)) / outHeight);
    rows[r] = (2 * Math.atan(Math.exp(y)) - Math.PI / 2) / rad;
  }
  return rows;
}

function parseHex(hex) {
  return [
    parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16),
  ];
}

function sourceRow(grid, lat) {
  const r = Math.round((grid.lat0 - lat) / grid.res_deg);
  return Math.min(Math.max(r, 0), grid.ny - 1);
}

function sourceCol(grid, lon) {
  const wrapped = ((((lon + 180) % 360) + 360) % 360) - 180;
  const c = Math.round((wrapped - grid.lon0) / grid.res_deg);
  return ((c % grid.nx) + grid.nx) % grid.nx;
}

// RGBA outWidth x outHeight for an image spanning exactly -180..180: one Mercator
// row per output row, and each output column takes the cell nearest its centre
// longitude (cell 0, centred on -180, straddles the seam: half at each edge).
// Unshaded = alpha 0.
export function colourise(values, grid, bands, outHeight, outWidth = 2 * grid.nx) {
  const { nx } = grid;
  const { levels, colors } = bands;
  const rgb = colors.map(parseHex);
  const lats = mercatorRows(outHeight);
  const cols = new Int32Array(outWidth);
  for (let j = 0; j < outWidth; j++) {
    cols[j] = sourceCol(grid, -180 + ((j + 0.5) * 360) / outWidth);
  }
  const out = new Uint8ClampedArray(outWidth * outHeight * 4);
  for (let r = 0; r < outHeight; r++) {
    const base = sourceRow(grid, lats[r]) * nx;
    for (let j = 0; j < outWidth; j++) {
      const k = bandIndex(values[base + cols[j]], levels, rgb.length);
      if (k < 0) continue;
      const o = 4 * (r * outWidth + j);
      out[o] = rgb[k][0];
      out[o + 1] = rgb[k][1];
      out[o + 2] = rgb[k][2];
      out[o + 3] = 255;
    }
  }
  return out;
}

// Nearest cell value for the click readout; lon is wrapped into -180..180.
export function sampleAt(values, grid, lon, lat) {
  return values[sourceRow(grid, lat) * grid.nx + sourceCol(grid, lon)];
}
