from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler import AsyncScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends, FastAPI, Query, Response

from energy_information_service.forecast import DamForecastProvider
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


app = FastAPI(title="Energy Information Service", lifespan=lifespan)


def get_data_provider():
    return data_provider


def get_forecast_provider():
    return forecast_provider


def get_supply_forecast_provider():
    return supply_forecast_provider


@app.get("/")
async def home():
    return {}


@app.get("/data")
async def get_price_matrix(task: DataProvider = Depends(get_data_provider)):
    price_matrix = await task.get_data()
    return price_matrix.to_dict(orient="records")  # Convert DataFrame to JSON


@app.get("/csv")
async def get_data_csv(task: DataProvider = Depends(get_data_provider)):
    """Serve the latest DataFrame as a CSV file."""
    price_matrix = await task.get_data()
    return Response(price_matrix.to_csv(index=False), media_type="text/csv")


@app.get("/source/{energy_source}")
async def get_data_by_source(energy_source: EnergySource, task: DataProvider = Depends(get_data_provider)):
    """
    Returns data filtered by the specified energy source (e.g. 'PV' or 'Grid').
    Case-insensitive (i.e., 'pv' or 'PV' will work).
    """
    # use the underlying string
    src = energy_source.value
    filtered = await task.get_data_by_source(src)

    if filtered.empty:
        # Show the list of valid options so the client can recover
        valid_sources = await task.get_sources()
        return {"error": f"Invalid energy source '{energy_source}'. " f"Available sources: {valid_sources}"}

    return filtered.to_dict(orient="records")


@app.get("/sources")
async def get_sources(task: DataProvider = Depends(get_data_provider)):
    """
    Returns a list of available energy sources.
    """
    sources = await task.get_sources()

    return {"sources": sources}


@app.get("/data/time-range")
async def get_data_time_range(
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
    task: DataProvider = Depends(get_data_provider),
):
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    # --- validate source first (case-insensitive) ---
    if source is not None:
        valid_sources = await task.get_sources()
        if source.value.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. " f"Available sources:  {valid_sources}"}
    # --- fetch data ---
    filtered = await task.get_data_by_time_range(from_time, to_time, source)

    # If we reach here, the source is valid (or absent). An empty dataframe means simply “no data in that time slice”.
    if filtered.empty:
        return {"error": "No data found for the given time range"}

    return filtered.to_dict(orient="records")


@app.get("/data/horizon")
async def get_data_horizon(
    source: EnergySource | None = Query(
        None,
        description="Optional energy source to filter by (e.g. 'PV' or 'Grid').",
    ),
    task: DataProvider = Depends(get_data_provider),
):
    """
    Returns the time horizon (earliest & latest timestamp) currently available
    in the cached data as a dictionary:
        {"start_time": "...", "end_time": "..."}
    """

    # --- validate source if provided (case-insensitive) ---
    if source is not None:
        valid_sources = await task.get_sources()
        if source.value.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. " f"Available sources: {valid_sources}"}

    horizon = await task.get_horizon(source)

    if horizon["start_time"] is None:
        return {"error": "No data found for the requested source"}

    return horizon


@app.get("/dam-forecast")
async def get_dam_forecast(task: DamForecastProvider = Depends(get_forecast_provider)):
    """
    Returns 5 x 24 h of 15 Minute Day-Ahead-Market prices as:
        [{"Time": "...", "Cost (EUR/MWh)": ...}, …]
    """
    return await task.get_forecast()


@app.get("/dam-forecast/horizon")
async def get_dam_forecast_horizon(
    task: DamForecastProvider = Depends(get_forecast_provider),
):
    """
    Returns the time horizon (earliest & latest timestamp) available in the
    cached DAM forecast:
        {"start_time": "...", "end_time": "..."}
    """
    horizon = await task.get_horizon()
    if horizon["start_time"] is None:
        return {"error": "No forecast available"}
    return horizon


@app.get("/dam-forecast/time-range")
async def get_dam_forecast_time_range(
    from_time: datetime | None = Query(
        None,
        description="ISO-8601 start, e.g. 2025-08-18T06:00:00+02:00",
    ),
    to_time: datetime | None = Query(
        None,
        description="ISO-8601 end, e.g. 2025-08-18T12:00:00+02:00",
    ),
    task: DamForecastProvider = Depends(get_forecast_provider),
):
    """
    Returns a time-sliced subset of the cached 5-day DAM forecast as:
        [{"Time": "...", "Cost (EUR/MWh)": ...}, …]
    """
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    rows = await task.get_data_by_time_range(from_time, to_time)
    if not rows:
        return {"error": "No data found for the given time range"}
    return rows


@app.get("/supply-forecast")
async def get_supply_forecast(task: SupplyForecastProvider = Depends(get_supply_forecast_provider)):
    """
    Returns 5 x 24 h of 15 Minute rows with the following columns

    * Time (ISO-8601 string)
    * Energy (kWh)
    * Cost (EUR/kWh)
    * Source  → "PV", "Grid", or "Forecast"

    The endpoint re-uses *one* shared `ForecastProvider` instance, so it
    does **not** increase the number of calls to the ENTSO-E API and keeps
    us safely under the token's rate limit.
    """
    return await task.get_supply_forecast()


@app.get("/supply-forecast/sources")
async def get_supply_forecast_sources(task: SupplyForecastProvider = Depends(get_supply_forecast_provider)):
    """
    Returns the list of energy sources present in the 5-day supply-forecast matrix,
    e.g. ["PV", "Grid", "Forecast"].
    """
    sources = await task.get_sources()
    return {"sources": sources}


@app.get("/supply-forecast/source/{energy_source}")
async def get_supply_forecast_by_source(
    energy_source: EnergySource,
    task: SupplyForecastProvider = Depends(get_supply_forecast_provider),
):
    """
    Returns rows for a single supply-forecast source (case-insensitive),
    e.g. 'PV', 'Grid', or 'Forecast'.
    """
    valid_sources = await task.get_sources()
    if energy_source.value.lower() not in (s.lower() for s in valid_sources):
        return {"error": f"Invalid energy source '{energy_source}'. Available sources: {valid_sources}"}

    return await task.get_data_by_source(energy_source)


@app.get("/supply-forecast/time-range")
async def get_supply_forecast_time_range(
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
    task: SupplyForecastProvider = Depends(get_supply_forecast_provider),
):
    # same validation style you already use on /data/time-range
    if from_time and to_time and from_time > to_time:
        return {"error": "from_time must not be after to_time"}

    if source is not None:
        valid_sources = await task.get_sources()
        if source.value.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. Available sources: {valid_sources}"}

    rows = await task.get_data_by_time_range(from_time, to_time, source)
    if not rows:
        return {"error": "No data found for the given time range"}

    return rows


@app.get("/supply-forecast/horizon")
async def get_supply_forecast_horizon(
    source: EnergySource | None = Query(
        None,
        description="Optional energy source to filter by: 'PV', 'Grid', or 'Forecast'.",
    ),
    task: SupplyForecastProvider = Depends(get_supply_forecast_provider),
):
    """
    Returns the time horizon (earliest & latest timestamp) available in the
    cached supply-forecast matrix:
        {"start_time": "...", "end_time": "..."}
    """
    # --- validate source if provided (case-insensitive), same style as /data/horizon ---
    if source is not None:
        valid_sources = await task.get_sources()
        if source.value.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. Available sources: {valid_sources}"}

    horizon = await task.get_horizon(source)

    if horizon["start_time"] is None:
        return {"error": "No data found for the requested source"}

    return horizon
