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
const CACHE_MAX = 90;
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

// Colour the decoded values with the layer's palette into the image source.
async function drawShaded(values, layer) {
  const { nx } = manifest.grid;
  const rgba = colourise(values, manifest.grid, palettes[layer.palette], OUT_HEIGHT);
  imageCanvas.width = nx;
  imageCanvas.height = OUT_HEIGHT;
  imageCanvas.getContext("2d").putImageData(new ImageData(rgba, nx, OUT_HEIGHT), 0, 0);
  const url = await canvasUrl(imageCanvas);
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

async function render() {
  const seq = ++renderSeq;
  const frames = sliderFrames();
  if (!frames.length) return;
  const index = indexFor(frames, state.lead);
  const frame = frames[index];
  state.lead = frame.lead_hours;
  ui.setLeadSlider(frames, index);
  writeUrl();

  const { shaded, iso, isoFrame } = loadFrames(frame);
  const [s, i] = await Promise.allSettled([shaded, iso]);
  if (seq !== renderSeq) return;

  let failed = false;
  const layer = manifest.layers[state.layer];
  if (!shaded) {
    hideShaded();
    Object.assign(shown, { layer: null, values: null, frame });
  } else if (s.status === "fulfilled") {
    await drawShaded(s.value, layer);
    if (seq !== renderSeq) return;
    Object.assign(shown, { layer: state.layer, values: s.value, frame });
  } else {
    failed = true;
  }

  const isoSource = map.getSource("isobars");
  if (!state.isobars) {
    isoSource.setData(EMPTY);
    shown.isoFrame = null;
  } else if (iso && i.status === "fulfilled") {
    isoSource.setData(i.value);
    shown.isoFrame = isoFrame;
  } else {
    failed = true;
  }

  ui.setStatus(failed ? "frame unavailable" : "");
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
  const next = (indexFor(frames, state.lead) + 1) % frames.length;
  state.lead = frames[next].lead_hours;
  await render();
  if (!state.playing) return;
  prefetch(frames[(next + 1) % frames.length]);
  timer = setTimeout(tick, Math.max(0, FRAME_MS - (performance.now() - started)));
}

function togglePlay() {
  state.playing = !state.playing;
  ui.setPlaying(state.playing);
  clearTimeout(timer);
  if (state.playing) {
    const frames = sliderFrames();
    prefetch(frames[(indexFor(frames, state.lead) + 1) % frames.length]);
    timer = setTimeout(tick, FRAME_MS);
  }
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
  map.setPaintProperty("background", "background-color", t.night);
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
        { id: "background", type: "background", paint: { "background-color": t.night } },
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

async function main() {
  [themes, palettes] = await Promise.all([
    fetchJson("data/themes.json"),
    fetchJson("data/palettes.json"),
  ]);
  const theme = ui.initialTheme();
  ui.applyTheme(themes, theme);
  ui.bindTheme(setTheme);

  map = buildMap(theme);
  const loaded = new Promise((resolve) => map.once("load", resolve));
  const coast = fetchJson("data/coastlines.geojson");

  try {
    manifest = await fetchJson(`${DATA_URL}globe/manifest.json`, { cache: "no-cache" });
  } catch (err) {
    console.warn("globe manifest", err);
    manifest = null;
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

main().catch((err) => console.error("globe page", err));
