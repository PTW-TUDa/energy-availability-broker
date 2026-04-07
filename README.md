# Energy Availability Broker

Implements the **Energy Availability Broker** for the ETA Factory.
It exposes a FastAPI application that serves Day-Ahead prices, extended by machine-learning based forecasts (e.g., a Day-Ahead-Market price forecast) and related utilities.


## Table of Contents

- [Features](#features)
- [Repository Layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration (secrets)](#configuration-secrets)
- [Run the API](#run-the-api)
  - [Dev mode (auto reload)](#dev-mode-auto-reload)
  - [Normal run](#normal-run)
  - [Docker](#docker)
- [API Documentation](#api-documentation)
- [Retraining the Price Model](#retraining-the-price-model)
- [Testing & Code Quality](#testing--code-quality)
- [Troubleshooting](#troubleshooting)
- [License](#license)


## Features

- **FastAPI** service (`energy_availability_broker/main.py`).
- **5-day Day-Ahead-Market (DAM) price forecast** at 15-minute resolution powered by an XGBoost model (`models/xgb_daily_model.pkl`).
- Forecast/data utilities in `forecast.py`, `forecast_utils.py`, `supply_forecast.py`, `services.py`.
- Ready-to-use **Dockerfile** and GitLab CI scripts in `.gitlab/docker/`.
- Auto-generated **OpenAPI** docs (Swagger UI & ReDoc).


## Repository Layout

```
.
├─ energy_availability_broker/
│  ├─ main.py                     # FastAPI application entrypoint
│  ├─ dayahead_forecast.py        # Day Ahead Price forecast provider class
│  ├─ dayahead_forecast_utils.py  # Day Ahead Price forecast feature engineering and data assembly
│  ├─ energy_availability.py      # Forecast provider class
│  ├─ supply_forecast.py          # Supply forecast provider class
│  ├─ models/xgb_daily_model.pkl
│  ├─ retrain_cli.py              # CLI for retraining the price model
│  └─ secret.py (local-only; do NOT commit real tokens)
├─ test/
│  └─ test_main.py
├─ pyproject.toml
└─ README.md
```

> You may also see runtime artifacts like `combined_data.csv`, `df_forecast.csv` and `.cache.sqlite` created during data pulls and retraining.


## Prerequisites

- **Python**: 3.10 (recommended)
- **Git**
- **Poetry** for dependency management (installed via **pipx** is recommended)
- **Docker** (optional, for containerized runs)

### Install pipx & Poetry

**Windows (PowerShell):**
```powershell
python -m pip install --user pipx
pipx install poetry
```

**macOS / Linux (bash):**
```bash
python3 -m pip install --user pipx
pipx install poetry
```

Verify:
```bash
poetry --version
```


## Quick Start

```bash
# 1) Clone
git clone <your-repo-url>
cd <repo-folder>

# 2) Install dependencies
poetry install --sync

# 3) Configure secrets (see next section)

# 4) Dev server
poetry run fastapi_dev
```

The service will start on `http://127.0.0.1:8000` by default.
Jump to [API Documentation](#api-documentation) to explore endpoints.


## Configuration (secrets)

Create a file named `.env` and keep your tokens there. Never commit this file to the repository.

**Example content for `.env`:**
```python
# ENTSO-E Transparency Platform token
ENTSOE_API_TOKEN = "INSERT-YOUR-API-TOKEN-HERE"

# Forecast.Solar API key
FORECAST_SOLAR_API_TOKEN = "YOURAPITOKENHERE"
```

## Run the API

### Dev mode (auto reload)

Uses FastAPI’s CLI with hot reload (ideal during development):

```bash
poetry run fastapi_dev
# default: http://127.0.0.1:8000
```

To change the port:
```bash
poetry run fastapi dev ./energy_availability_broker/main.py --port 8010
```

### Normal run

```bash
poetry run fastapi run ./energy_availability_broker/main.py
```

### Docker

Build and run locally:

```bash
# Build (from repo root)
docker build -f .gitlab/docker/Dockerfile -t energy_info_service_image:local .

# Run — make sure secret.py is available inside the container.
# If the container's app workdir is /app, mount your local secret.py into it:
docker run --rm -p 8000:8000 \
  -v "$(pwd)/energy_availability_broker/secret.py:/app/energy_availability_broker/secret.py:ro" \
  energy_info_service_image:local
```

## API Documentation

Once the server is running (dev or docker):

- **Swagger UI** (interactive):
  `http://127.0.0.1:8000/docs`

- **ReDoc**:
  `http://127.0.0.1:8000/redoc`

- **Raw OpenAPI JSON**:
  `http://127.0.0.1:8000/openapi.json`

These pages list all available endpoints exposed by `main.py` (for example, a `GET /dam-forecast` endpoint for day-ahead price predictions, if enabled in your build).


## Retraining the Price Model

The repository includes a CLI to (re)train the XGBoost model with fresh data and save it under `energy_availability_broker/models/`.

Run:

```bash
# From repo root
poetry run retrain_model
```

Notes:

- Requires valid tokens in `energy_availability_broker/secret.py`.
- By default it pulls historical data, engineers features, trains, and writes the model (e.g., `xgb_daily_model.pkl`).
- CSV artifacts such as `combined_data.csv` may be generated or updated.


## Testing & Code Quality

Run tests:

```bash
poetry run pytest -q
```

Install pre-commit hooks (recommended for contributors):

```bash
poetry run pre-commit install
```

Run linters/formatters on demand:

```bash
poetry run ruff check .
poetry run ruff format .
```


## Troubleshooting

- **Missing or NaN values from ENTSO-E**
  The transparency platform may publish data with slight delays. If the latest quarter hours are `NaN`, try again later or confirm your timezone settings.

- **Auth errors**
  Ensure `ENTSOE_API_TOKEN` and `FORECAST_SOLAR_API_KEY` are set in `.env`.

- **Port already in use**
  Start on another port: `--port 8010`.

- **Windows tip**
  Prefer running commands in **PowerShell** (or WSL) rather than Git Bash when using Poetry and FastAPI CLI.


## License

This project is licensed under the terms of the **LICENSE** file in the repository root.
