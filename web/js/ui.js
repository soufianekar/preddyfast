// DOM side of the globe page: theme tokens, the legend, the provenance text and
// the controls. No map and no data fetching here; globe.js wires it together.

import { bandIndex } from "./field.js";

const THEME_KEY = "preddyfast-theme";
const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

export const $ = (id) => document.getElementById(id);

// ---- Theme ------------------------------------------------------------------

function storedTheme() {
  try {
    const t = window.localStorage.getItem(THEME_KEY);
    return t === "dark" || t === "light" ? t : null;
  } catch {
    return null;
  }
}

function storeTheme(name) {
  try {
    window.localStorage.setItem(THEME_KEY, name);
  } catch {
    // Private mode or blocked storage: the toggle still works for this visit.
  }
}

function systemTheme() {
  return window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
}

export function initialTheme() {
  return storedTheme() ?? systemTheme();
}

// --mx-<token> on :root from data/themes.json, and data-theme for the toggle.
export function applyTheme(themes, name) {
  const root = document.documentElement;
  for (const [token, value] of Object.entries(themes[name])) {
    root.style.setProperty(`--mx-${token}`, value);
  }
  root.dataset.theme = name;
  root.style.colorScheme = name;
  const toggle = $("theme-toggle");
  toggle.textContent = name === "dark" ? "Light" : "Dark";
  toggle.setAttribute("aria-label", `Switch to ${name === "dark" ? "light" : "dark"} theme`);
  toggle.setAttribute("aria-pressed", String(name === "light"));
}

// Toggle remembered in localStorage; until then the page follows the system.
export function bindTheme(onChange) {
  $("theme-toggle").addEventListener("click", () => {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    storeTheme(next);
    onChange(next);
  });
  window.matchMedia("(prefers-color-scheme: light)").addEventListener("change", () => {
    if (!storedTheme()) onChange(systemTheme());
  });
}

// ---- Time and provenance ------------------------------------------------------

// Manifest times are naive UTC ("YYYY-MM-DDTHH:MM:SS").
export function parseUtc(iso) {
  return new Date(`${iso}Z`);
}

export function formatUtc(iso) {
  const d = parseUtc(iso);
  const hh = String(d.getUTCHours()).padStart(2, "0");
  return `${DAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${hh} UTC`;
}

function formatDeg(x) {
  return `${Number(x)}°`;
}

// Built from the manifest only: model, run, lead, valid time, drawn resolution,
// precipitation window, latitude limit, isobar spacing.
export function provenanceText(manifest, layerKey, frame, isobars) {
  const parts = [manifest.label, `run ${formatUtc(manifest.init_time)}`];
  if (frame) parts.push(`T+${frame.lead_hours} h, valid ${formatUtc(frame.valid_time)}`);
  const layer = manifest.layers[layerKey];
  if (layer && layer.kind === "shaded") {
    if (layer.window_hours) parts.push(`${layer.window_hours} h total ending at the valid time`);
    const { res_deg: drawn, native_res_deg: native } = manifest.grid;
    parts.push(
      native && native !== drawn
        ? `drawn at ${formatDeg(drawn)} from ${formatDeg(native)}`
        : `drawn at ${formatDeg(drawn)}`,
    );
    parts.push("shown to 85°N/S");
  }
  const msl = manifest.layers.msl;
  if (isobars && msl) {
    parts.push(`isobars at ${formatDeg(msl.res_deg)}, every ${msl.interval} ${msl.unit}`);
  }
  return parts.join(" · ");
}

// ---- Legend -----------------------------------------------------------------

// One equal-width swatch per band, level labels at the band edges, as the
// Default legends do. With one colour per level the top band is open ("≥").
export function renderLegend(el, layer, bands) {
  el.replaceChildren();
  if (!layer || !bands) return;
  const { levels, colors } = bands;
  const n = colors.length;
  const openTop = n === levels.length;

  const title = document.createElement("div");
  title.className = "legend-title";
  title.textContent = `${layer.label} (${layer.unit})`;

  const bar = document.createElement("div");
  bar.className = "legend-bar";
  for (let i = 0; i < n; i++) {
    const sw = document.createElement("span");
    sw.style.background = colors[i];
    const range = openTop && i === n - 1 ? `≥ ${levels[i]}` : `${levels[i]}–${levels[i + 1]}`;
    sw.title = `${range} ${layer.unit}`;
    bar.append(sw);
  }

  // Edge labels: levels[i] at the left edge of swatch i, thinned so they do not
  // collide on a narrow panel. The top label is always shown, right-aligned:
  // "≥ level" over the open-top swatch, else the closing level at the right edge.
  const labels = document.createElement("div");
  labels.className = "legend-labels";
  const step = Math.max(1, Math.ceil(n / 9));
  const top = openTop ? n - 1 : n;
  const add = (text, pct, cls) => {
    const span = document.createElement("span");
    span.textContent = text;
    span.style.left = `${pct}%`;
    if (cls) span.className = cls;
    labels.append(span);
  };
  for (let i = 0; i < n && i <= top - step; i += step) {
    add(String(levels[i]), (100 * i) / n, i === 0 ? "first" : "");
  }
  add(openTop ? `≥${levels[top]}` : String(levels[top]), 100, "last");
  el.append(title, bar, labels);
}

// ---- Popup content ------------------------------------------------------------

function formatLat(lat) {
  return `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}`;
}

function formatLon(lon) {
  const w = ((((lon + 180) % 360) + 360) % 360) - 180;
  return `${Math.abs(w).toFixed(2)}°${w >= 0 ? "E" : "W"}`;
}

// "no value" where the cell is unshaded (missing, below the lowest level, or
// poleward of the drawn 85°).
export function popupLines(value, layer, bands, lngLat) {
  const where = `${formatLat(lngLat.lat)} ${formatLon(lngLat.lng)}`;
  const shaded =
    Math.abs(lngLat.lat) <= 85.0511 &&
    bandIndex(value, bands.levels, bands.colors.length) >= 0;
  const what = shaded ? `${value.toFixed(1)} ${layer.unit}` : "no value";
  return [`${layer.label}: ${what}`, where];
}

// ---- Controls -----------------------------------------------------------------

export function setLayerRadio(layerKey) {
  for (const r of document.querySelectorAll('input[name="layer"]')) {
    r.checked = r.value === layerKey;
  }
}

export function setLeadSlider(frames, index) {
  const slider = $("lead");
  slider.max = String(Math.max(frames.length - 1, 0));
  slider.value = String(index);
  const text = frames.length ? `T+${frames[index].lead_hours} h` : "";
  $("lead-label").textContent = text;
  slider.setAttribute("aria-valuetext", text);
}

export function setPlaying(playing) {
  const b = $("play");
  b.textContent = playing ? "Pause" : "Play";
  b.setAttribute("aria-pressed", String(playing));
}

export function setStatus(text) {
  $("status").textContent = text;
}

export function bindControls({ onLayer, onIsobars, onLead, onPlay }) {
  for (const r of document.querySelectorAll('input[name="layer"]')) {
    r.addEventListener("change", () => r.checked && onLayer(r.value));
  }
  $("isobars").addEventListener("change", (e) => onIsobars(e.target.checked));
  $("lead").addEventListener("input", (e) => onLead(Number(e.target.value)));
  $("play").addEventListener("click", onPlay);
}

export function showEmpty() {
  $("panel-body").hidden = true;
  $("empty").hidden = false;
}

export function showControls() {
  $("empty").hidden = true;
  $("panel-body").hidden = false;
}
