from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from enum import Enum

from apscheduler import AsyncScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends, FastAPI, Query
from fastapi.responses import RedirectResponse

from energy_information_service.config import SERVICE_CONFIG
from energy_information_service.dayahead_forecast import DamForecastProvider
from energy_information_service.demand_forecast import DemandForecast, DemandForecastModel
from energy_information_service.energy_availability import EnergyAvailabilityProvider
from energy_information_service.energy_broker import EnergyBrokerProvider
from energy_information_service.env_utils import load_service_env
from energy_information_service.flmp_demand_forecast import demand_forecast_model_from_flmp_file
from energy_information_service.non_production_forecast import NonProductionPowerForecastProvider
from energy_information_service.supply_forecast import SupplyForecastProvider

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)

load_service_env(override=False)

data_provider = EnergyAvailabilityProvider()
forecast_provider = DamForecastProvider()
supply_forecast_provider = SupplyForecastProvider(forecast_provider)
demand_forecast_provider = DemandForecast()
non_production_forecast_provider = NonProductionPowerForecastProvider()


class EnergySource(str, Enum):
    PV = "PV forecast"
    GRID = "Grid"
    FORECAST = "Grid forecast"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the app lifecycle using BackgroundTask as a context manager."""
    # async with data_provider:  # Start & stop automatically
    async with AsyncScheduler() as scheduler:
        # Add periodic task
        await scheduler.add_schedule(data_provider.refresh, IntervalTrigger(minutes=5))
        await scheduler.add_schedule(supply_forecast_provider.refresh, IntervalTrigger(minutes=15))

        # refresh forecast at server start + every 24 h
        await scheduler.add_schedule(forecast_provider.get_forecast, IntervalTrigger(hours=24))

        # Start the scheduler in the background
        await scheduler.start_in_background()
        log.info("Started periodic background task to fetch data every 15 minutes.")

        log.debug("yielding")
        yield  # Run the app
    log.debug("App shutdown complete.")


app = FastAPI(title="Energy Availability Service", lifespan=lifespan)


def get_data_provider():
    return data_provider


def get_forecast_provider():
    return forecast_provider


def get_supply_forecast_provider():
    return supply_forecast_provider


def get_demand_forecast_provider():
    return demand_forecast_provider


def get_production_demand_forecast() -> DemandForecastModel:
    return demand_forecast_model_from_flmp_file(source="production")


def get_non_production_forecast_provider():
    return non_production_forecast_provider


def get_energy_broker_provider(
    data_provider: EnergyAvailabilityProvider = Depends(get_data_provider),
    supply_provider: SupplyForecastProvider = Depends(get_supply_forecast_provider),
    demand_provider: DemandForecast = Depends(get_demand_forecast_provider),
    non_production_provider: NonProductionPowerForecastProvider = Depends(get_non_production_forecast_provider),
):
    return EnergyBrokerProvider(data_provider, supply_provider, demand_provider, non_production_provider)


def _resolve_time_range(
    from_time: datetime | None,
    to_time: datetime | None,
) -> tuple[datetime, datetime]:
    if from_time is not None and to_time is not None:
        return from_time, to_time

    horizon = timedelta(hours=SERVICE_CONFIG.prediction_horizon_hours)
    resolved_from = from_time or datetime.now().astimezone()
    resolved_to = to_time or (resolved_from + horizon)
    return resolved_from, resolved_to


# @app.get("/csv")
# async def get_data_csv(provider: EnergyAvailabilityProvider = Depends(get_data_provider)):
#     """Serve the latest DataFrame as a CSV file."""
#     price_matrix = await provider.get_data()
#     return Response(price_matrix.to_csv(index=False), media_type="text/csv")


@app.get("/energy", tags=["energy availability"])
async def energy_availability(
    from_time: datetime | None = Query(
        None,
        description="ISO-8601 start datetime, e.g. 2025-05-05T06:00:00+02:00",
    ),
    to_time: datetime | None = Query(
        None,
        description="ISO-8601 end datetime, e.g. 2025-05-05T12:00:00+02:00",
    ),
    source: EnergySource | None = Query(
        None,
        description="Optional energy source to filter by (e.g. 'PV' or 'Grid').",
    ),
    provider: EnergyBrokerProvider = Depends(get_energy_broker_provider),
    production_forecast: DemandForecastModel = Depends(get_production_demand_forecast),
):
    """
    Returns all information on energy availability filtered by the specified times and energy source.
    All parameters are optional.
    """
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    # --- validate source first (case-insensitive) ---
    from_time, to_time = _resolve_time_range(from_time, to_time)

    # --- fetch data ---
    filtered = await provider.get_rows(
        from_time,
        to_time,
        source,
        production_forecast,
    )

    # If we reach here, the source is valid (or absent). An empty dataframe means simply “no data in that time slice”.
    if not filtered:
        return {"error": "No data found for the given time range"}

    return filtered


@app.get("/dam-forecast", tags=["day-ahead price forecast"])
async def day_ahead_price_forecast(
    from_time: datetime | None = Query(
        None,
        description="ISO-8601 start, e.g. 2025-08-18T06:00:00+02:00",
    ),
    to_time: datetime | None = Query(
        None,
        description="ISO-8601 end, e.g. 2025-08-18T12:00:00+02:00",
    ),
    provider: DamForecastProvider = Depends(get_forecast_provider),
):
    """
    Returns a forecast for 5 x 24 h of 15 Minute Day-Ahead prices as:
        [{'Time': '...', 'Cost (EUR/MWh)': ...}, …]
    Optionally filtered by a time range.
    """
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    from_time, to_time = _resolve_time_range(from_time, to_time)
    rows = await provider.get_data_by_time_range(from_time, to_time)
    if not rows:
        return {"error": "No data found for the given time range"}
    return rows


@app.get("/supply-forecast", tags=["energy supply forecast"])
async def supply_forecast(
    from_time: datetime | None = Query(
        None,
        description="ISO-8601 start, e.g. 2025-08-18T06:00:00+02:00",
    ),
    to_time: datetime | None = Query(
        None,
        description="ISO-8601 end, e.g. 2025-08-18T12:00:00+02:00",
    ),
    source: EnergySource | None = Query(
        None,
        description="Optional source filter: 'PV', 'Grid', or 'Forecast'.",
    ),
    provider: SupplyForecastProvider = Depends(get_supply_forecast_provider),
):
    """
    Returns the energy supply forecast filtered by the specified times and energy source.
    All parameters are optional.
    """
    # same validation style you already use on /data/time-range
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    if source is not None:
        valid_sources = await provider.get_sources()
        if source.value.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. Available sources: {valid_sources}"}

    from_time, to_time = _resolve_time_range(from_time, to_time)
    rows = await provider.get_data_by_time_range(from_time, to_time, source)
    if not rows:
        return {"error": "No data found for the given time range"}

    return rows


@app.get("/energy/maximum-horizon", tags=["energy availability"])
async def energy_availability_horizon_available(
    source: EnergySource | None = Query(
        None,
        description="Optional energy source to filter by (e.g. 'PV' or 'Grid').",
    ),
    provider: EnergyBrokerProvider = Depends(get_energy_broker_provider),
    production_forecast: DemandForecastModel = Depends(get_production_demand_forecast),
):
    """
    Returns the time horizon (earliest & latest timestamp as ISO-8601) currently available
    in the cached data for the given source (or for all, if left blank) as a dictionary:
        {'from_time': '...', 'to_time': '...'}
    """

    # --- validate source if provided (case-insensitive) ---
    default_from, default_to = _resolve_time_range(None, None)
    horizon = await provider.get_horizon(source, default_from, default_to, production_forecast)
    if horizon["from_time"] is None:
        return {"error": "No data found for the requested source"}
    return horizon


@app.get("/dam-forecast/maximum-horizon", tags=["day-ahead price forecast"])
async def day_ahead_price_forecast_horizon_available(
    provider: DamForecastProvider = Depends(get_forecast_provider),
):
    """
    Returns the time horizon (earliest & latest timestamp as ISO-8601) currently available
    in the prediction data as a dictionary:
        {'from_time': '...', 'to_time': '...'}
    """
    horizon = await provider.get_horizon()
    if horizon["from_time"] is None:
        return {"error": "No forecast available"}
    return horizon


@app.get("/supply-forecast/maximum-horizon", tags=["energy supply forecast"])
async def supply_forecast_horizon_available(
    source: EnergySource | None = Query(
        None,
        description="Optional energy source to filter by: 'PV', 'Grid', or 'Forecast'.",
    ),
    provider: SupplyForecastProvider = Depends(get_supply_forecast_provider),
):
    """
    Returns the time horizon (earliest & latest timestamp) available in the
    cached supply forecast:
        {"from_time": "...", "to_time": "..."}
    """
    # --- validate source if provided (case-insensitive), same style as /data/horizon ---
    if source is not None:
        valid_sources = await provider.get_sources()
        if source.value.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. Available sources: {valid_sources}"}

    horizon = await provider.get_horizon(source)

    if horizon["from_time"] is None:
        return {"error": "No data found for the requested source"}

    return horizon


@app.get("/energy/sources", tags=["energy availability"])
async def energy_availability_sources(
    provider: EnergyBrokerProvider = Depends(get_energy_broker_provider),
    production_forecast: DemandForecastModel = Depends(get_production_demand_forecast),
):
    """
    Returns the list of energy sources in energy availability.
    """
    from_time, to_time = _resolve_time_range(None, None)
    sources = await provider.get_sources(
        from_time,
        to_time,
        production_forecast,
    )

    return {"sources": sources}


@app.get("/supply-forecast/sources", tags=["energy supply forecast"])
async def supply_forecast_sources(provider: SupplyForecastProvider = Depends(get_supply_forecast_provider)):
    """
    Returns the list of energy sources present in the supply forecast.
    """
    sources = await provider.get_sources()
    return {"sources": sources}


@app.put("/demand-forecast", tags=["demand forecast"])
async def update_demand_forecast(
    forecast: DemandForecastModel,
    provider: DemandForecast = Depends(get_demand_forecast_provider),
):
    """Receive a production planning system's demand forecast and persist it.

    The incoming payload must follow the `DemandForecastModel` schema: a `source`
    identifier and a list of timestamped `time`/`energy_kwh` points (ISO-8601).

    The incoming forecast will replace any existing values for the same source in
    the timeframe of the incoming forecast, and will leave other timeframes
    untouched.
    """
    # validation is handled by Pydantic model
    await provider.store_forecast(forecast)
    return {"status": "ok", "message": "Forecast stored"}


@app.get("/demand-forecast", tags=["demand forecast"])
async def get_demand_forecast(
    from_time: datetime | None = Query(None, description="ISO-8601 start datetime, e.g. 2025-05-05T06:00:00+02:00"),
    to_time: datetime | None = Query(None, description="ISO-8601 end datetime, e.g. 2025-05-05T12:00:00+02:00"),
    source: str | None = Query(
        None,
        description="Optional source filter; omit to return combined demand rows from all available sources.",
    ),
    provider: DemandForecast = Depends(get_demand_forecast_provider),
    production_forecast: DemandForecastModel = Depends(get_production_demand_forecast),
    non_production_provider: NonProductionPowerForecastProvider = Depends(get_non_production_forecast_provider),
):
    """Return demand rows from the site/non-production forecast, FLMP production, and any extra stored sources."""
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    from_time, to_time = _resolve_time_range(from_time, to_time)
    site_forecast = non_production_provider.get_demand_forecast_model(from_time, to_time)

    if source is None:
        stored_rows = await provider.get_data(from_time, to_time)
        supplemental_rows = [row for row in stored_rows if row["Source"].lower() not in {"site", "production"}]
        site_rows = await provider.get_model_data(site_forecast, from_time, to_time)
        production_rows = await provider.get_model_data(production_forecast, from_time, to_time)
        rows = sorted(
            [*supplemental_rows, *site_rows, *production_rows],
            key=lambda row: (row["Time"], row["Source"].lower()),
        )
    elif source.lower() == "site":
        rows = await provider.get_model_data(site_forecast, from_time, to_time)
    elif source.lower() == "production":
        rows = await provider.get_model_data(production_forecast, from_time, to_time)
    else:
        rows = await provider.get_data(from_time, to_time, source)

    if not rows:
        return {"error": "No demand forecast available for the given filters"}
    return rows


@app.get("/non-production-demand-forecast", tags=["non-production demand forecast"])
async def non_production_demand_forecast(
    from_time: datetime | None = Query(None, description="ISO-8601 start datetime, e.g. 2025-05-05T06:00:00+02:00"),
    to_time: datetime | None = Query(None, description="ISO-8601 end datetime, e.g. 2025-05-05T06:00:00+02:00"),
    provider: NonProductionPowerForecastProvider = Depends(get_non_production_forecast_provider),
):
    """
    Returns a 48-hour hourly forecast:
      [{'Time': '...', 'TimeLocal': '...', 'horizon_h': 1..48,
      'non_production_energy_kwh': ..., 'ts_issue_utc': '...'}, ...]
    """
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    try:
        from_time, to_time = _resolve_time_range(from_time, to_time)
        return provider.get_forecast(from_time, to_time)
    except Exception as e:
        return {"error": str(e)}


@app.get("/", include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")
