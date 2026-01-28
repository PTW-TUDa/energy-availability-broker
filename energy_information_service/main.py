from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler import AsyncScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends, FastAPI, Query
from fastapi.responses import RedirectResponse

from energy_information_service.dam_forecast import DamForecastProvider
from energy_information_service.services import DataProvider
from energy_information_service.supply_forecast import SupplyForecastProvider
from energy_information_service.type_annotations import EnergySource

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
data_provider = DataProvider()
forecast_provider = DamForecastProvider()
supply_forecast_provider = SupplyForecastProvider(forecast_provider)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the app lifecycle using BackgroundTask as a context manager."""
    # async with data_provider:  # Start & stop automatically
    async with AsyncScheduler() as scheduler:
        # Add periodic task
        await scheduler.add_schedule(data_provider.run, IntervalTrigger(minutes=5))

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


# @app.get("/csv")
# async def get_data_csv(provider: DataProvider = Depends(get_data_provider)):
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
    provider: DataProvider = Depends(get_data_provider),
):
    """
    Returns all information on energy availability filtered by the specified times and energy source.
    All parameters are optional.
    """
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    # --- validate source first (case-insensitive) ---
    if source is not None:
        valid_sources = await provider.get_sources()
        if source.value.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. " f"Available sources:  {valid_sources}"}
    # --- fetch data ---
    filtered = await provider.get_data_by_time_range(from_time, to_time, source)

    # If we reach here, the source is valid (or absent). An empty dataframe means simply “no data in that time slice”.
    if filtered.empty:
        return {"error": "No data found for the given time range"}

    return filtered.to_dict(orient="records")


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
    Returns a forecast for 5 x 24 h of 15 Minute Day-Ahead-Market prices as:
        [{'Time': '...', 'Cost (EUR/MWh)': ...}, …]
    Optionally filtered by a time range.
    """
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

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
    provider: DataProvider = Depends(get_data_provider),
):
    """
    Returns the time horizon (earliest & latest timestamp as ISO-8601) currently available
    in the cached data for the given source (or for all, if left blank) as a dictionary:
        {'from_time': '...', 'to_time': '...'}
    """

    # --- validate source if provided (case-insensitive) ---
    if source is not None:
        valid_sources = await provider.get_sources()
        if source.value.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. " f"Available sources: {valid_sources}"}

    horizon = await provider.get_horizon(source)

    if horizon["from_time"] is None:
        return {"error": "No data found for the requested source"}

    return horizon


@app.get("/dam-forecast/maximum-horizon", tags=["day-ahead price forecast"])
async def day_ahead_market_horizon_available(
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
async def energy_availability_sources(provider: DataProvider = Depends(get_data_provider)):
    """
    Returns the list of energy sources in energy availability.
    """
    sources = await provider.get_sources()

    return {"sources": sources}


@app.get("/supply-forecast/sources", tags=["energy supply forecast"])
async def supply_forecast_sources(provider: SupplyForecastProvider = Depends(get_supply_forecast_provider)):
    """
    Returns the list of energy sources present in the supply forecast.
    """
    sources = await provider.get_sources()
    return {"sources": sources}


@app.get("/", include_in_schema=False)
async def docs():
    return RedirectResponse(url="/docs")
