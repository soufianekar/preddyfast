// The globe page: the newest global AIFS Single run on a MapLibre globe.
// Data: DATA_URL + "globe/manifest.json" and the frames it lists (16-bit PNG
// data textures for shaded fields, GeoJSON for isobars). ui.js holds the DOM.

/* global maplibregl */

import { DATA_URL } from "./config.js";
import { colourise, decode16, sampleAt } from "./field.js";
import * as ui from "./ui.js";

const LAT_MAX = 85.0511;
const IMAGE_COORDS = [
  [-180, LAT_MAX],
  [180, LAT_MAX],
  [180, -LAT_MAX],
  [-180, -LAT_MAX],
];
const OUT_HEIGHT = 720;
const FRAME_MS = 500;
// One full layer of frames plus a few: bounds decoded memory on a phone.
const CACHE_MAX = 48;
const LAYERS = ["t2m", "tp6", "none"];
const EMPTY = { type: "FeatureCollection", features: [] };

let manifest = null;
let palettes = null;
let themes = null;
let map = null;
let mapReady = false;

const state = { layer: "t2m", lead: 0, isobars: true, playing: false };
// What is actually drawn: a failed frame leaves the previous one (and its
// legend, provenance and click readout) in place.
const shown = { layer: null, frame: null, values: null, isoFrame: null };

// ---- Frame cache: decoded frames keyed by path, least recently used out ------

const cache = new Map();
const inflight = new Map();

function cached(path, load) {
  if (cache.has(path)) {
    const value = cache.get(path);
    cache.delete(path);
    cache.set(path, value);
    return Promise.resolve(value);
  }
  if (inflight.has(path)) return inflight.get(path);
  const p = load(DATA_URL + path)
    .then((value) => {
      cache.set(path, value);
      while (cache.size > CACHE_MAX) cache.delete(cache.keys().next().value);
      return value;
    })
    .finally(() => inflight.delete(path));
  inflight.set(path, p);
  return p;
}

function makeCanvas(width, height) {
  if (typeof OffscreenCanvas !== "undefined") return new OffscreenCanvas(width, height);
  return Object.assign(document.createElement("canvas"), { width, height });
}

async function decodeFrame(url, layer) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const bitmap = await createImageBitmap(await res.blob(), {
    premultiplyAlpha: "none",
    colorSpaceConversion: "none",
  });
  const { nx, ny } = manifest.grid;
  if (bitmap.width !== nx || bitmap.height !== ny) {
    bitmap.close();
    throw new Error(`${url}: ${bitmap.width}x${bitmap.height}, expected ${nx}x${ny}`);
  }
  const ctx = makeCanvas(nx, ny).getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  return decode16(ctx.getImageData(0, 0, nx, ny).data, layer.offset, layer.scale);
}

async function fetchJson(url, options) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

// ---- Frames and leads ----------------------------------------------------------

function sliderFrames() {
  const key = state.layer === "none" ? "msl" : state.layer;
  return manifest.layers[key]?.frames ?? [];
}

// The frame at the wanted lead, else the nearest (a layer switch keeps the lead
// when it exists: precipitation starts at T+6).
function indexFor(frames, lead) {
  let best = 0;
  for (let i = 0; i < frames.length; i++) {
    if (Math.abs(frames[i].lead_hours - lead) < Math.abs(frames[best].lead_hours - lead)) {
      best = i;
    }
  }
  return best;
}

function mslFrameAt(lead) {
  return manifest.layers.msl?.frames.find((f) => f.lead_hours === lead) ?? null;
}

function loadFrames(frame) {
  const layer = manifest.layers[state.layer];
  const shaded =
    layer && layer.kind === "shaded" ? cached(frame.path, (u) => decodeFrame(u, layer)) : null;
  const isoFrame = state.isobars ? mslFrameAt(frame.lead_hours) : null;
  const iso = isoFrame ? cached(isoFrame.path, (u) => fetchJson(u)) : null;
  return { shaded, iso, isoFrame };
}

function prefetch(frame) {
  const { shaded, iso } = loadFrames(frame);
  shaded?.catch(() => {});
  iso?.catch(() => {});
}

// ---- Drawing -------------------------------------------------------------------

const imageCanvas = makeCanvas(1, OUT_HEIGHT);
let imageUrl = null;

async function canvasUrl(canvas) {
  const blob = canvas.convertToBlob
    ? await canvas.convertToBlob({ type: "image/png" })
    : await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
  return URL.createObjectURL(blob);
}

// Colour the decoded values with the layer's palette; a blob URL of the PNG.
// Twice the grid's width, so each cell (centred on lon0 + c * res) spans its own
// half-columns either side of its centre and the image edges sit at +-180.
async function shadedUrl(values, layer) {
  const width = 2 * manifest.grid.nx;
  const rgba = colourise(values, manifest.grid, palettes[layer.palette], OUT_HEIGHT, width);
  imageCanvas.width = width;
  imageCanvas.height = OUT_HEIGHT;
  imageCanvas.getContext("2d").putImageData(new ImageData(rgba, width, OUT_HEIGHT), 0, 0);
  return canvasUrl(imageCanvas);
}

// Synchronous, so it runs only for the newest render (see renderFrame).
function showShaded(url) {
  const source = map.getSource("shaded");
  if (source) {
    source.updateImage({ url });
  } else {
    map.addSource("shaded", { type: "image", url, coordinates: IMAGE_COORDS });
    map.addLayer(
      {
        id: "shaded",
        type: "raster",
        source: "shaded",
        paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 },
      },
      "coastline",
    );
  }
  map.setLayoutProperty("shaded", "visibility", "visible");
  // The previous URL may still be loading if frames change fast; free it later.
  const old = imageUrl;
  imageUrl = url;
  if (old) setTimeout(() => URL.revokeObjectURL(old), 10000);
}

function hideShaded() {
  if (map.getLayer("shaded")) map.setLayoutProperty("shaded", "visibility", "none");
}

let renderSeq = 0;

// Never rejects: a failure keeps the previous frame and says so.
async function render() {
  const seq = ++renderSeq;
  try {
    await renderFrame(seq);
  } catch (err) {
    console.warn("globe frame", err);
    if (seq === renderSeq) ui.setStatus("frame unavailable");
  }
}

// All or nothing: the shaded field and the isobars are applied together, after
// every await, so what is drawn always matches the legend and provenance.
async function renderFrame(seq) {
  const frames = sliderFrames();
  if (!frames.length) return;
  const index = indexFor(frames, state.lead);
  const frame = frames[index];
  state.lead = frame.lead_hours;
  ui.setLeadSlider(frames, index);
  writeUrl();

  const layer = manifest.layers[state.layer];
  const { shaded, iso, isoFrame } = loadFrames(frame);
  const [s, i] = await Promise.allSettled([shaded, iso]);
  if (seq !== renderSeq) return;
  const shadedFailed = shaded && s.status !== "fulfilled";
  const isoFailed = state.isobars && !(iso && i.status === "fulfilled");
  if (shadedFailed || isoFailed) {
    ui.setStatus("frame unavailable");
    return;
  }

  const url = shaded ? await shadedUrl(s.value, layer) : null;
  if (seq !== renderSeq) {
    if (url) URL.revokeObjectURL(url);
    return;
  }

  if (url) showShaded(url);
  else hideShaded();
  map.getSource("isobars").setData(iso ? i.value : EMPTY);
  Object.assign(shown, {
    layer: shaded ? state.layer : null,
    values: shaded ? s.value : null,
    frame,
    isoFrame: iso ? isoFrame : null,
  });
  ui.setStatus("");
  refreshText();
}

function refreshText() {
  const layer = shown.layer ? manifest.layers[shown.layer] : null;
  ui.renderLegend(document.getElementById("legend"), layer, layer && palettes[layer.palette]);
  const msl = manifest.layers.msl;
  document.getElementById("iso-note").textContent =
    shown.isoFrame && msl ? `Isobars every ${msl.interval} ${msl.unit}` : "";
  document.getElementById("provenance").textContent = ui.provenanceText(
    manifest,
    shown.layer ?? "none",
    shown.frame,
    Boolean(shown.isoFrame),
  );
  if (popup.isOpen()) updatePopup();
}

// ---- Click readout -------------------------------------------------------------

const popup = new maplibregl.Popup({ closeOnClick: false, maxWidth: "260px" });
let clicked = null;

function updatePopup() {
  if (!shown.values || !clicked) {
    popup.remove();
    return;
  }
  const layer = manifest.layers[shown.layer];
  const value = sampleAt(shown.values, manifest.grid, clicked.lng, clicked.lat);
  const box = document.createElement("div");
  for (const line of ui.popupLines(value, layer, palettes[layer.palette], clicked)) {
    const div = document.createElement("div");
    div.textContent = line;
    box.append(div);
  }
  popup.setLngLat(clicked).setDOMContent(box);
}

function onClick(e) {
  if (!shown.values) return;
  clicked = e.lngLat;
  updatePopup();
  if (!popup.isOpen()) popup.addTo(map);
}

// ---- Playback ------------------------------------------------------------------

let timer = null;

async function tick() {
  if (!state.playing) return;
  const started = performance.now();
  const frames = sliderFrames();
  if (!frames.length) {
    stopPlaying();
    return;
  }
  const next = (indexFor(frames, state.lead) + 1) % frames.length;
  state.lead = frames[next].lead_hours;
  await render();
  if (!state.playing) return;
  prefetch(frames[(next + 1) % frames.length]);
  timer = setTimeout(tick, Math.max(0, FRAME_MS - (performance.now() - started)));
}

function stopPlaying() {
  state.playing = false;
  ui.setPlaying(false);
  clearTimeout(timer);
}

function togglePlay() {
  const frames = sliderFrames();
  if (state.playing || !frames.length) {
    stopPlaying();
    return;
  }
  state.playing = true;
  ui.setPlaying(true);
  prefetch(frames[(indexFor(frames, state.lead) + 1) % frames.length]);
  timer = setTimeout(tick, FRAME_MS);
}

// ---- URL query: ?layer=t2m&lead=72&isobars=1 ------------------------------------

function readUrl() {
  const q = new URLSearchParams(window.location.search);
  const layer = q.get("layer");
  if (LAYERS.includes(layer) && (layer === "none" || manifest.layers[layer])) state.layer = layer;
  else if (!manifest.layers[state.layer]) state.layer = "none";
  const lead = Number(q.get("lead"));
  if (q.has("lead") && Number.isFinite(lead)) state.lead = lead;
  else state.lead = sliderFrames()[0]?.lead_hours ?? 0;
  if (q.has("isobars")) state.isobars = q.get("isobars") !== "0";
  if (!manifest.layers.msl) state.isobars = false;
}

function writeUrl() {
  const q = new URLSearchParams(window.location.search);
  q.set("layer", state.layer);
  q.set("lead", String(state.lead));
  q.set("isobars", state.isobars ? "1" : "0");
  const url = `${window.location.pathname}?${q}${window.location.hash}`;
  window.history.replaceState(window.history.state, "", url);
}

// ---- Theme -----------------------------------------------------------------------

function setTheme(name) {
  ui.applyTheme(themes, name);
  if (!mapReady) return;
  const t = themes[name];
  map.setPaintProperty("background", "background-color", t.panel);
  map.setPaintProperty("coastline", "line-color", t.coast);
  map.setPaintProperty("border", "line-color", t.border);
  map.setPaintProperty("isobars", "line-color", t.ink);
}

// ---- Map -------------------------------------------------------------------------

function buildMap(theme) {
  const t = themes[theme];
  const kind = (k) => ["==", ["get", "kind"], k];
  return new maplibregl.Map({
    container: "map",
    hash: "map",
    center: [10, 35],
    zoom: 1.6,
    attributionControl: { compact: true },
    style: {
      version: 8,
      projection: { type: "globe" },
      sources: {
        coastlines: { type: "geojson", data: EMPTY, attribution: "Natural Earth" },
        isobars: { type: "geojson", data: EMPTY },
      },
      layers: [
        // The globe's surface; the space around it is the .map container's
        // --mx-rule (the canvas is transparent off the globe), so the disc's
        // edge reads in both themes.
        { id: "background", type: "background", paint: { "background-color": t.panel } },
        {
          id: "coastline",
          type: "line",
          source: "coastlines",
          filter: kind("coastline"),
          paint: { "line-color": t.coast, "line-width": 0.8 },
        },
        {
          id: "border",
          type: "line",
          source: "coastlines",
          filter: kind("border"),
          paint: { "line-color": t.border, "line-width": 0.5 },
        },
        // No on-line labels: symbol text needs a glyph server; the interval is
        // stated in the panel and the provenance line instead.
        {
          id: "isobars",
          type: "line",
          source: "isobars",
          paint: { "line-color": t.ink, "line-width": 0.9, "line-opacity": 0.85 },
        },
      ],
    },
  });
}

function validManifest(m) {
  const g = m.grid;
  return Boolean(
    g &&
      Number.isInteger(g.nx) &&
      Number.isInteger(g.ny) &&
      m.layers &&
      typeof m.layers === "object" &&
      Object.keys(m.layers).length,
  );
}

async function main() {
  [themes, palettes] = await Promise.all([
    fetchJson("data/themes.json"),
    fetchJson("data/palettes.json"),
  ]);
  const theme = ui.initialTheme();
  ui.applyTheme(themes, theme);
  ui.bindTheme(setTheme);

  // No WebGL makes the constructor throw; an error before "load" means the map
  // never draws, so it fails the start instead of leaving a blank page.
  map = buildMap(theme);
  const loaded = new Promise((resolve, reject) => {
    const failed = (e) => reject(e.error ?? new Error("the map failed to load"));
    map.on("error", failed);
    map.once("load", () => {
      map.off("error", failed);
      resolve();
    });
  });
  loaded.catch(() => {}); // awaited below; never an unhandled rejection meanwhile
  const coast = fetchJson("data/coastlines.geojson");

  try {
    manifest = await fetchJson(`${DATA_URL}globe/manifest.json`, { cache: "no-cache" });
  } catch (err) {
    console.warn("globe manifest", err);
    manifest = null;
  }
  if (manifest && !validManifest(manifest)) {
    throw new Error("the published manifest is malformed (no grid or layers)");
  }

  await loaded;
  mapReady = true;
  setTheme(document.documentElement.dataset.theme);
  coast.then((g) => map.getSource("coastlines").setData(g)).catch((err) => {
    console.warn("coastlines", err);
  });

  if (!manifest) {
    ui.showEmpty();
    return;
  }

  readUrl();
  ui.setLayerRadio(state.layer);
  document.getElementById("isobars").checked = state.isobars;
  document.getElementById("isobars").disabled = !manifest.layers.msl;
  ui.showControls();
  ui.bindControls({
    onLayer: (layer) => {
      state.layer = layer;
      render();
    },
    onIsobars: (on) => {
      state.isobars = on;
      render();
    },
    onLead: (index) => {
      state.lead = sliderFrames()[index].lead_hours;
      render();
    },
    onPlay: togglePlay,
  });
  map.on("click", onClick);
  popup.on("close", () => {
    clicked = null;
  });
  await render();
}

// One catch for every way the page can fail to start (themes or palettes not
// loaded, no WebGL, a malformed manifest): say so in the panel, never blank.
main().catch((err) => {
  console.error("globe page", err);
  const reason = String(err?.message ?? err).slice(0, 160);
  ui.showEmpty(`The globe could not start: ${reason}`);
});
