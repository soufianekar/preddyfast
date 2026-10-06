# asifast

A Plotly Dash web app for any-location climate context and probabilistic
forecast outlooks, built on [Open-Meteo](https://open-meteo.com) and on
cloud-native ensemble archives published as
[Icechunk](https://icechunk.io) stores on AWS Open Data, plus gridded maps
retrieved on demand from the Copernicus
[Climate Data Store](https://cds.climate.copernicus.eu) and
[Atmosphere Data Store](https://ads.atmosphere.copernicus.eu).

No proprietary data. Everything except the Copernicus maps page works with no
API key at all; that page needs a free Copernicus token (see
[Configuration](#configuration)).

---

## Pages

### Climatology — `/`

Pick a location, a climate variable, a year, and a reanalysis model. Returns
two plots:

- **YTD accumulation** — cumulative value from Jan 1 against historical
  P10/P25/P50/P75/P90 bands. For the current year a 10-day ECMWF-IFS forecast
  extension is appended as a dashed line.
- **Daily anomaly** — day-by-day value against the same bands, with days
  outside P10/P90 colored darker to flag extremes.

Supporting features: selectable climatology reference period (1961–1990,
1981–2010, 1991–2020), location comparison, shareable URLs encoding full
state, CSV export, plain-language auto-summary, and a linear trend overlay.

### Ensemble Meteogram — `/meteogram`

A short-range probabilistic outlook for a point. One or more ensemble systems
are overlaid on a single chart — members as box/whisker distributions per
forecast day, the control run bold on top, one color per system — so
inter-model and intra-model spread are legible together.

### Regional Ensemble — `/regional`

Area-averaged ensemble spread over a lat/lon box, drawn by dragging a
rectangle on the map. Space is reduced **per member** with `cos(lat)`
weighting first, then percentiles are taken across members — so every
spaghetti line is a physically self-consistent realisation of the regional
average, and the envelope describes the spread of *that* quantity.

This page reads a gridded archive rather than a point API, because a regional
average is not something a point endpoint can serve.

### Copernicus Maps — `/maps`

Gridded fields retrieved on demand from Copernicus, drawn as a heatmap at the
data's own grid spacing with bundled Natural Earth coastlines and borders. No
smoothing or interpolation: what you hover is the gridbox value.

| Dataset | Store | What it gives |
|---|---|---|
| ERA5 hourly, single levels | CDS | Any hour since 1940 — temperature, dew point, MSLP, precipitation, cloud, radiation, gusts |
| ERA5 monthly means | CDS | A calendar month, and optionally its anomaly against 1961–1990, 1981–2010 or 1991–2020 |
| CAMS global composition forecast | ADS | AOD at 550 nm (total, dust, organic, sea salt, sulphate) and surface PM, to T+120 h |
| CAMS European air quality forecast | ADS | 0.1° PM2.5, PM10, O₃, NO₂, SO₂ from the regional ensemble or any single model, to T+96 h |

Copernicus queues requests server-side, so a first retrieval takes anywhere
from seconds to minutes. The page submits the request on a worker thread and
polls; the download is cached on disk by request hash, so the same field is
never requested twice.

---

## Data sources

| Feature | Backend | Model |
|---|---|---|
| Point climatology, percentiles, trends | Open-Meteo archive API | ERA5 family, CERRA, IFS |
| Point forecast extension | Open-Meteo forecast API | ECMWF IFS |
| Point ensemble meteogram | Open-Meteo ensemble API | ICON, GFS, ECMWF ENS, others |
| Regional area averages | Icechunk store on AWS Open Data | ECMWF IFS ENS, 51 members, 0.25° |
| Reanalysis maps | Copernicus CDS (`cdsapi`) | ERA5, 0.25° |
| Atmospheric composition maps | Copernicus ADS (`cdsapi`) | CAMS global 0.4°, CAMS Europe 0.1° |

The regional store is pinned by version in `utils/settings.py` and read
anonymously — no AWS account is needed.

Note that the regional page and the point pages come from **different models
and different runs**, so their values will not match for the same location.
The page says so on screen.

---

## Quick start

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

REACT_VERSION=18.2.0 python src/app.py
```

Then open <http://localhost:8050/climate/>.

`REACT_VERSION=18.2.0` is required by dash-mantine-components.

The regional page's first read of a given box takes roughly 10–14 s — it is
fetching and decompressing chunks from S3. Repeat reads are served from cache
in about a millisecond.

For the maps page, put your Climate Data Store token in `~/.cdsapirc`:

```
url: https://cds.climate.copernicus.eu/api
key: <your CDS personal access token>
```

The Atmosphere Data Store needs a second token, and `cdsapi` reads only one rc
file — so supply it through the environment instead:

```bash
export ADS_KEY=<your ADS personal access token>
```

Both are free from <https://cds.climate.copernicus.eu/profile> and
<https://ads.atmosphere.copernicus.eu/profile>. Each dataset also has to have
its licence accepted once, on its page in the web interface; the app surfaces
the server's message if you have not. Without `ADS_KEY` the CDS datasets work
normally and the CAMS ones explain what is missing.

---

## Configuration

Everything is environment-driven; there are no secrets in the repository.

| Variable | Default | Purpose |
|---|---|---|
| `URL_BASE_PATHNAME` | `/climate/` | Mount path behind a reverse proxy |
| `APP_HOST` / `APP_PORT` | `0.0.0.0` / `8050` | Dev server bind |
| `CACHE_TYPE` / `CACHE_DIR` | `FileSystemCache` / `.cache` | flask-caching backend |
| `NOMINATIM_USER_AGENT` | `climate-dashboard/1.0` | Geocoding UA string |
| `MAPBOX_KEY` | *(unset)* | Optional geocoding fallback |
| `DYNAMICAL_STAC_URL` | `https://stac.dynamical.org` | Catalogue the gridded stores' addresses are looked up in (pinned fallbacks in `GRID_STORES`) |
| `CDS_KEY` / `CDS_URL` | *(from `~/.cdsapirc`)* | Climate Data Store token and endpoint |
| `ADS_KEY` / `ADS_URL` | *(unset)* / ADS API | Atmosphere Data Store token and endpoint |
| `CDS_CACHE_DIR` | `.cds-cache` | Where retrieved NetCDF is cached, by request hash |
| `OBS_CACHE_DIR` | `.obs-cache` | GHCNh station list and station-year files (Evaluation page) |
| `POINT_CACHE_DIR` | `.point-cache` | Past-run point extracts, kept for good (mount as a volume) |
| `GHCNH_BASE_URL` | NCEI GHCNh hourly | Observation source root |
| `CDS_JOB_TIMEOUT_S` | `1800` | Abandon a retrieval still queued after this |

Variable and model definitions live in `utils/settings.py`. Adding an entry to
`VARIABLES`, `MODELS`, `ENSEMBLE_MODELS`, `GRID_VARIABLES`, `CDS_DATASETS` or
`CDS_VARIABLES` propagates through the UI automatically — nothing else hardcodes variable names, model
strings, units, or colors.

---

## Architecture

| Layer | Module | Responsibility |
|---|---|---|
| App shell | `app.py` | Dash init, MantineProvider, cache, page routing |
| Pages | `pages/` | Layout and callbacks only — no business logic |
| Point data | `utils/openmeteo_api.py` | Every Open-Meteo call, all memoized |
| Gridded data | `utils/earthmover_api.py` | Every Icechunk/Zarr read, bbox only |
| Geocoding | `utils/geocoding.py` | Nominatim (+ optional Mapbox) |
| Computation | `utils/climatology.py` | Percentiles, accumulation, trends, validation |
| Presentation | `utils/figures.py` | `go.Figure` factories — no API calls |
| Configuration | `utils/settings.py` | Variables, models, periods, colors |
| Widgets | `components/` | Location search, bbox picker, export button |

Data flows one way:

```
User input  →  dcc.Store  →  page callback
                               ├─ *_api.fetch_*()        [cached]
                               ├─ climatology.compute_*()
                               └─ figures.make_*_fig()
```

Callbacks never compute. They guard their inputs, validate the model against
the location and year, call cached `utils/` functions, and return figures or
`no_update`. No exception is allowed to escape a callback.

---

## Invariants worth knowing

These are enforced deliberately and should not be "simplified" away.

**Ensembles**

- **Never average across ensemble systems.** Each system has its own bias,
  resolution, and physics; a cross-system mean is not meteorologically
  meaningful. Averaging is valid only *within* one system.
- **Member counts are derived from the response**, never hardcoded — they vary
  by system and change over time.
- **Reduce space before taking percentiles.** On the regional page the
  `cos(lat)`-weighted spatial mean is computed per member, and percentiles are
  taken across members afterwards. The reverse order produces a per-gridpoint
  patchwork that corresponds to no member and is systematically wider at the
  tails.
- **Init time is always annotated.** Ensemble spread is uninterpretable
  without knowing how old the run is.

**Labelling**

- **Label must match code.** If the UI says "1991–2020 climatology", the
  filter is `between(1991, 2020)`. The active period and data source appear in
  every plot subtitle.
- **Raw ensemble output is a gridbox average, not a point value.** Neither the
  point pages nor the regional page resolve anything finer than the native
  grid, and the regional page states its box extent and cell count explicitly.

**Data handling**

- **DOY 366 is forward-filled from DOY 365**, never dropped.
- **The forecast extension only appears when `year == current year`.**
- **`best_match` always shows a non-dismissable consistency warning** about the
  ERA5/IFS boundary.

---

## Gridded backend notes

Two constraints shape `utils/earthmover_api.py`:

- **The store handle is opened lazily, per process.** Icechunk's Rust/tokio
  runtime and its TLS connection pool do not survive `fork()`. The handle is
  tagged with the PID that opened it and reopened in a child, and the store is
  never opened at import time — which also matters under gunicorn `--preload`.
- **`background=True` is not usable for this path.** Dash's `DiskcacheManager`
  runs callbacks in a forked child, which dies outright on this stack
  (`objc_initializeAfterForkError`) with nothing in the server log. The
  regional page uses a plain callback and drives its own loading overlay.

The store is *time-optimized*: one chunk holds all 51 members and all 85 lead
times for a 32×32-cell tile. Member and lead selection are therefore
effectively free, while continental or global map frames are not viable from
it at all.

Design decisions, measurements, and the reasoning behind them are recorded in
[`docs/icechunk-handoff.md`](docs/icechunk-handoff.md).

---

## Notebooks

```bash
pip install -r requirements-dev.txt
jupyter lab notebooks/
```

- **`01_regional_map_smoke.ipynb`** — exercises the gridded path without Dash:
  store metadata, `snap_bbox` cases, one field read and map, every variable's
  units, and the failure modes that must surface as `GridDataError`.
- **`02_regional_map_interactive.ipynb`** — the app's area picker rebuilt on
  ipyleaflet (same OSM basemap, rectangle-only draw tool, snapped rectangle
  redrawn), with variable/lead/member/init controls and a reload cell for
  iterating on `figures.make_regional_map_fig` without refetching.

Both import the same `utils/` functions the pages call, so a figure settled on
in a notebook is already the app's figure. They bind a `SimpleCache` to
`utils.cache` themselves, which `app.py` normally does at startup.

---

## Status and known limitations

- **No automated test suite yet.** `notebooks/01_regional_map_smoke.ipynb`
  covers the gridded path by hand; the pure functions in `climatology.py` and
  `earthmover_api.py` are the obvious first target for real assertions.
- **Deployment is unverified against the current dependency set.** The
  scientific stack adds roughly 1 GB to a container image, and the fork
  constraints above apply to any multi-process server. A `Dockerfile` is kept
  out of the repository until it has actually been built and run.
- **No antimeridian wrapping.** A regional box centred near 180° is clipped at
  179.75° rather than wrapped. The true extent is stated on the plot, so
  nothing is silently misreported, but the box is not what was requested.
- **Regional boxes snap outward** to the 0.25° grid, so a 5° request becomes
  5.25°. Stated on the plot rather than hidden.
- **Forecast verification** against the multi-year init archive is possible but
  unscoped.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
