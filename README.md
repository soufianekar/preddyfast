---
title: preddyfast
emoji: 🌦️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
short_description: Ensemble forecasts, verification and ERA5 maps
---

# preddyfast

**A weather and climate workbench built around ensemble forecasts:
what the models say, how sure they are, and how well they have done.**

Live demo: `https://huggingface.co/spaces/<hf-user>/preddyfast`

This is the public, reduced version of a larger private workbench. Everything
here runs on free services: open data, a free app host, and a scheduled job
that does the expensive work offline.

---

## What it does

| View | What you get | Data |
|---|---|---|
| **Weather · At a location** | Meteogram for any point: several ensemble systems overlaid, members as box plots per window, deterministic models as dashed lines, the init time of every run stated. Latest runs or any archived past run. | Open-Meteo ensemble and forecast APIs; dynamical.org archives |
| **Weather · Area average** | Area-averaged ensemble for a box or a standard region (ECMWF, WMO, IPCC AR6): member spaghetti, percentile envelope, and a map animation of mean, percentile or spread. | ECMWF IFS ENS / AIFS ENS, NOAA GEFS / GFS archives (Icechunk/Zarr on AWS Open Data) |
| **Weather · Evaluation** | A station scorecard: seven archived models (IFS ENS, AIFS ENS, GEFS, IFS control, AIFS Single, GFS, ICON-EU) scored over the latest 14 days of 00 UTC runs at eight European city airports, by lead day — 2 m temperature, 6 h precipitation and 10 m wind against hourly reports (NOAA GHCNh), Z500 against IFS control T+0. RMSE, MAE, bias, fair CRPS, bias-corrected RMSE/MAE, spread/skill, ETS, HSS, frequency bias, Brier; absolute or relative to IFS ENS, and per station on a map. | Same archives, GHCNh |
| **Climate · Maps** | ERA5 monthly maps for the latest month: temperature and precipitation anomalies against 1991–2020, Europe and global. | Copernicus Climate Data Store |

## How it runs for free

| Piece | Service |
|---|---|
| App | Hugging Face Spaces (Docker, 2 vCPU, 16 GB), Dash under gunicorn |
| Precomputed products | A public Hugging Face Dataset: preset animations, the maps gallery, the scorecard's aggregates (`eval/scorecard/aggregates.parquet`) |
| Scheduled job | GitHub Actions running `scripts/precompute.py` after the 00/12 UTC runs |
| Data | Open-Meteo (free tier), dynamical.org Icechunk stores (anonymous AWS Open Data), Copernicus CDS (job only), NOAA GHCNh, NASA IMERG |

A public site on free tiers needs guards. They are all in
`src/utils/settings.py` and can be overridden by environment:

| Guard | Default | Why |
|---|---|---|
| `OPENMETEO_DAILY_BUDGET` | 8000 weighted calls/day | Open-Meteo allows 10,000/day per IP, and every visitor shares the server's. Requests are weighted as Open-Meteo counts them (per 14 days, per 10 series, ensemble members included) and refused until 00 UTC once the shared total is reached. |
| `READ_BUDGET_MB` | 400 MB per request | Gridded reads are costed before reading |
| `MAX_MODELS` | 2 | Models per comparison |
| `RATE_LIMIT` | 120/min, 1500/h per client | Dash callbacks; the client is taken `TRUSTED_PROXY_HOPS` from the right of `X-Forwarded-For` |
| Nominatim | 1 request/s for the site | Nominatim's usage policy |
| Copernicus | never from the app | Maps are retrieved and drawn by the scheduled job; the CDS key exists only as a GitHub secret |

## Architecture

```
GitHub Actions (cron) ── scripts/precompute.py ──► HF Dataset (PNG / WebP / JSON / npz)
                                                     │ images by URL      │ manifests, scorecard
                                                     ▼                    ▼
Browser ◄────────────── Dash pages ◄──── utils/ (pure, tested) ◄── hub_sync mirror
                                           ├─ openmeteo_api   points, latest runs (budgeted)
                                           └─ earthmover_api  Icechunk/Zarr archives (read-budgeted)
```

- `pages/` wire controls to `utils/` and compute nothing themselves.
- `utils/` holds the science: accumulation windows, step types, verification
  scores, grid geometry, regions.
- `figures.py` (Plotly) and `static_figures.py` (matplotlib + cartopy, object
  API only, bundled Natural Earth lines) draw. All colours come from one theme
  table (`settings.THEMES`), checked for ≥ 3:1 contrast in both themes.
- `utils/settings.py` declares every variable, model, store, region and palette;
  the UI derives from it.

## Run it locally

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-dev.txt
.venv/bin/python -m pytest -q
cd src && REACT_VERSION=18.2.0 ../.venv/bin/python app.py      # http://localhost:8050
```

Build the precomputed products into `.precomputed/` (the maps gallery needs a
free CDS key in `~/.cdsapirc` or `CDS_KEY`):

```bash
python scripts/precompute.py --scorecard --maps
```

The scorecard fills its 14-day window newest init first; to fill it from
scratch, run `--no-animations --scorecard --backfill --max-minutes 300` until a
run writes nothing new (the workflow's `backfill` input does the same).

## Deploying your own copy

1. Create a public HF **Dataset** (e.g. `<hf-user>/preddyfast-data`) and a
   **Space** with the Docker SDK.
2. GitHub repository: secrets `HF_TOKEN` (write) and `CDS_KEY`, variable
   `PRECOMPUTE_HF_REPO`. Run the `precompute` workflow once with *upload* ticked.
3. Space settings: variables `PRECOMPUTE_HF_REPO` and `NOMINATIM_USER_AGENT`
   (`preddyfast/1.0 (your contact)`). Push this repository to the Space.

## Data and credits

Open-Meteo (CC BY 4.0) · dynamical.org archives of ECMWF IFS ENS / AIFS
(ECMWF, CC BY 4.0) and NOAA GEFS / GFS · Copernicus Climate Change Service ERA5
(contains modified Copernicus Climate Change Service information 2026) · NOAA
GHCNh · NASA GPM IMERG · Natural Earth · IPCC AR6 reference regions ·
OpenStreetMap contributors / Nominatim.

### Bundled data

These files keep their own licences; the Apache License below covers the code.

| File | Source | Licence |
|---|---|---|
| `src/data/ar6_regions.json` | IPCC WGI Atlas reference regions v4 (Iturbide et al., 2020, *Earth Syst. Sci. Data* 12, 2959–2970) | CC BY 4.0 |
| `src/data/scorecard_orography.json` | ERA5 invariant orography, Copernicus Climate Change Service | Copernicus licence (contains modified Copernicus Climate Change Service information 2026) |
| `src/data/natural_earth_lines.npz` | Natural Earth coastlines and borders | Public domain |
| `tests/data/ghcnh_*` | NOAA GHCNh samples | US Government work, public domain |

## Licence

Apache License 2.0 — see [LICENSE](LICENSE).
