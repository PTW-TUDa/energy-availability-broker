# Energy Information Service

Implements the **Energy Information Service** for the ETA Factory.
It exposes a FastAPI application that serves Day-Ahead prices, extended by machine-learning based forecasts (e.g., a Day-Ahead-Market price forecast) and related utilities.

---

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

---

## Features

- **FastAPI** service (`energy_information_service/main.py`).
- **5-day Day-Ahead-Market (DAM) price forecast** at 15-minute resolution powered by an XGBoost model (`models/xgb_daily_model.pkl`).
- Forecast/data utilities in `forecast.py`, `forecast_utils.py`, `supply_forecast.py`, `services.py`.
- Ready-to-use **Dockerfile** and GitLab CI scripts in `.gitlab/docker/`.
- Auto-generated **OpenAPI** docs (Swagger UI & ReDoc).

---

## Repository Layout

```
.
├─ energy_information_service/
│  ├─ main.py                 # FastAPI application entrypoint
│  ├─ forecast.py             # Forecast provider runtime
│  ├─ forecast_utils.py       # Feature engineering, data assembly
│  ├─ retrain.py              # CLI for retraining the price model
│  ├─ supply_forecast.py      # Supply forecast helpers
│  ├─ services.py             # Data provider/services
│  ├─ models/xgb_daily_model.pkl
│  └─ secret.py (local-only; do NOT commit real tokens)
├─ test/
│  └─ test_main.py
├─ .gitlab/docker/Dockerfile
├─ poetry.lock / pyproject.toml
└─ README.md
```

> You may also see runtime artifacts like `combined_data.csv`, `df_forecast.csv` and `.cache.sqlite` created during data pulls and retraining.

---

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

---

## Quick Start

```bash
# 1) Clone
git clone <your-repo-url>
cd <repo-folder>

# 2) Install dependencies
poetry install --sync

# 3) Configure secrets (see next section)

# 4) Dev server
poetry run fastapi dev ./energy_information_service/main.py
```

The service will start on `http://127.0.0.1:8000` by default.
Jump to [API Documentation](#api-documentation) to explore endpoints.

---

## Configuration (secrets)

**Create a file named `energy_information_service/secret.py` and keep your tokens there. Never commit this file to the repository.**

- Add `energy_information_service/secret.py` to `.gitignore`.
- If it’s already tracked, remove it from Git history with:
  ```bash
  git rm --cached energy_information_service/secret.py
  ```
- Optionally, commit a safe template as `energy_information_service/secret.example.py` (no real keys).

**Example content for `energy_information_service/secret.py`:**
```python
# energy_information_service/secret.py
# NEVER commit real secrets. This file is for local/dev use only.

# ENTSO-E Transparency Platform token
ENTSOE_API_TOKEN = "INSERT-YOUR-API-TOKEN-HERE"

# Forecast.Solar API key
FORECAST_SOLAR_API_KEY = "YOURAPIKEYHERE"

```

> The application imports from `energy_information_service.secret`
> (e.g., `from energy_information_service import secret`
> or `from .secret import ENTSOE_API_TOKEN, FORECAST_SOLAR_API_KEY`).

---

## Run the API

### Dev mode (auto reload)

Uses FastAPI’s CLI with hot reload (ideal during development):

```bash
poetry run fastapi dev ./energy_information_service/main.py
# default: http://127.0.0.1:8000
```

To change the port:
```bash
poetry run fastapi dev ./energy_information_service/main.py --port 8010
```

### Normal run

```bash
poetry run fastapi run ./energy_information_service/main.py
```

### Docker

Build and run locally:

```bash
# Build (from repo root)
docker build -f .gitlab/docker/Dockerfile -t energy_info_service_image:local .

# Run — make sure secret.py is available inside the container.
# If the container's app workdir is /app, mount your local secret.py into it:
docker run --rm -p 8000:8000 \
  -v "$(pwd)/energy_information_service/secret.py:/app/energy_information_service/secret.py:ro" \
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

---

## Retraining the Price Model

The repository includes a CLI to (re)train the XGBoost model with fresh data and save it under `energy_information_service/models/`.

Run:

```bash
# From repo root
poetry run retrain_model
```

Notes:

- Requires valid tokens in `energy_information_service/secret.py`.
- By default it pulls historical data, engineers features, trains, and writes the model (e.g., `xgb_daily_model.pkl`).
- CSV artifacts such as `combined_data.csv` may be generated or updated.

---

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

---

## Troubleshooting

- **Missing or NaN values from ENTSO-E**
  The transparency platform may publish data with slight delays. If the latest quarter hours are `NaN`, try again later or confirm your timezone settings.

- **Auth errors**
  Ensure `ENTSOE_API_TOKEN` (and `ENTSOE_API_KEY` if required) and `FORECAST_SOLAR_API_KEY` are set in `secret.py`.

- **Port already in use**
  Start on another port: `--port 8010`.

- **Windows tip**
  Prefer running commands in **PowerShell** (or WSL) rather than Git Bash when using Poetry and FastAPI CLI.

---

## License

This project is licensed under the terms of the **LICENSE** file in the repository root.
