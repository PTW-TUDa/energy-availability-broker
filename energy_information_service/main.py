import logging
from contextlib import asynccontextmanager

from apscheduler import AsyncScheduler
from apscheduler.triggers.interval import IntervalTrigger
from fastapi import Depends, FastAPI, Response

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
        return {"error": "No rows found for source '{energy_source}'"}

    return filtered.to_dict(orient="records")


@app.get("/sources")
async def get_sources(task: DataProvider = Depends(get_data_provider)):
    """
    Returns a list of available energy sources.
    """
    sources = await task.get_sources()

    return {"sources": sources}
