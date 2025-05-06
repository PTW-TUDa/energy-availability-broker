from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime

from apscheduler import AsyncScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends, FastAPI, Query, Response

from energy_information_service.services import DataProvider

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)
data_provider = DataProvider()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages the app lifecycle using BackgroundTask as a context manager."""
    # async with data_provider:  # Start & stop automatically
    async with AsyncScheduler() as scheduler:
        # Add periodic task
        await scheduler.add_schedule(data_provider.run, IntervalTrigger(minutes=5))

        # Start the scheduler in the background
        await scheduler.start_in_background()
        log.info("Started periodic background task to fetch data every 15 minutes.")

        log.debug("yielding")
        yield  # Run the app
    log.debug("App shutdown complete.")


app = FastAPI(title="Energy Information Service", lifespan=lifespan)


def get_data_provider():
    return data_provider


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
async def get_data_by_source(energy_source: str, task: DataProvider = Depends(get_data_provider)):
    """
    Returns data filtered by the specified energy source (e.g. 'PV' or 'Grid').
    Case-insensitive (i.e., 'pv' or 'PV' will work).
    """
    filtered = await task.get_data_by_source(energy_source)

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
    source: str | None = Query(
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
        if source.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. " f"Available sources:  {valid_sources}"}
    # --- fetch data ---
    filtered = await task.get_data_by_time_range(from_time, to_time, source)

    # If we reach here, the source is valid (or absent). An empty dataframe means simply “no data in that time slice”.
    if filtered.empty:
        return {"error": "No data found for the given time range"}

    return filtered.to_dict(orient="records")


@app.get("/data/horizon")
async def get_data_horizon(
    source: str | None = Query(
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
        if source.lower() not in (s.lower() for s in valid_sources):
            return {"error": f"Invalid energy source '{source}'. " f"Available sources: {valid_sources}"}

    horizon = await task.get_horizon(source)

    if horizon["start_time"] is None:
        return {"error": "No data found for the requested source"}

    return horizon
