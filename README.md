# IgnisAI

IgnisAI is a map-first wildfire intelligence application for exploring active fire incidents, source health, fire-weather context, and advisory fire-spread risk forecasts. The app combines a React/Mapbox frontend, a Node/Express API, and a Python FastAPI tile service that serves the wildfire prediction model.

Live app: [https://ignisai-frontend.onrender.com](https://ignisai-frontend.onrender.com)

IgnisAI forecasts are advisory risk visualizations. They are not official evacuation guidance, official incident intelligence, or predicted official perimeters.

## What It Does

- Displays current wildfire activity from NASA FIRMS and incident/perimeter sources.
- Shows official perimeter, hotspot, weather alert, and prediction layers on an interactive map.
- Provides source-health indicators so degraded or partial data is visible in the UI.
- Runs fire-spread risk forecasts through the tile service and renders raster timeline overlays.
- Supports historical fire presets for model testing, including Camp/Paradise, Eaton, Palisades, Dixie, and Caldor.
- Exposes health, metrics, and prediction contract checks for local and deployed validation.

## Architecture

| Layer | Path | Role |
| --- | --- | --- |
| Frontend | `frontend/` | React app with Mapbox GL, dashboard controls, auth screens, forecast timeline, and source-health UI. |
| Backend API | `backend/` | Express API for map bootstrap data, fire data, weather, perimeters, auth, prediction proxying, health, and metrics. |
| Tile service | `services/tilesvc/` | FastAPI service for v3 model input assembly, prediction raster generation, multistep rollout, health checks, and model metadata. |
| ML package | `ignis_ml/` | Training/runtime model and feature code shared with the serving path. |
| Static pipeline | `services/static_pipeline/` | Builds static raster catalogs used by the production predictor. |
| Runtime cache | `services/runtime_cache/` | Tools for preparing runtime cache artifacts. |
| Deployment | `render.yaml`, `docker/`, `docker-compose.yml` | Render blueprint and container definitions for local and production-style runs. |

## Data And Model Inputs

IgnisAI uses live and cached data from several sources:

- NASA FIRMS for satellite fire detections.
- NOAA/NWS sources for weather and fire-weather alerts.
- WFIGS/FIRIS-style perimeter feeds where available.
- Static raster channels such as elevation, slope, aspect, NDVI, burn index, ERC, PDSI, CHILI, impervious surface, water, population, and fuel layers.
- A v3 ConvLSTM/UNet delta model configured by `services/tilesvc/model_config.default.json`.

The prediction service is designed to fail visibly when required model or data artifacts are missing. When predictions are unavailable, observed map layers should remain usable.

## Documentation

- [Test plan](docs/test-plan.md)
- [Operations runbook](docs/runbook.md)
- [Environment configuration](docs/environment.md)
- [Static raster pipeline](docs/static-pipeline.md)
- [Test results](docs/results.md)
- [Project notes](docs/project.md)

## Operational Notes

- Render free-tier services may cold start. First prediction requests can take longer while the backend and tile service wake up.
- The frontend timeout for multistep predictions is intentionally long to allow cold-start model loading and a full rollout.
- `PREDICTIONS_ENABLED=false` disables model calls while keeping observed map layers available.
- The forecast layer is a relative risk heatmap and should be labeled as advisory wherever it appears.

## Maintainer

- Christian Juarez

## Acknowledgments

IgnisAI began as a capstone project with early contributions from Dylan Nguyen, Travis Nguyen, and Emmanuel Montoya. It is now maintained and developed by Christian Juarez.
