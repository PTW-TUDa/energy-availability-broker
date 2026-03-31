from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import anyio
import pandas as pd
from pydantic import BaseModel, Field

from energy_information_service.config import SERVICE_CONFIG

log = logging.getLogger(__name__)


class DemandPoint(BaseModel):
    time: datetime
    energy_kwh: float = Field(..., ge=0.0)


class DemandForecastModel(BaseModel):
    """Pydantic model for a demand forecast payload.

    - `source`: identifies the origin of the forecast (string, e.g. 'production').
    - `values`: list of timestamped demand points (ISO-8601 timestamps).
    """

    forecast_id: str | None = None
    source: str = Field(..., description="Forecast source identifier")
    values: list[DemandPoint] = Field(..., min_length=1)


class DemandForecast:
    """Holds and persists demand forecasts.

    Storage: JSON file containing a list of records with Time, energy_kwh and source.
    When a new forecast is stored it replaces any existing values for the same source
    in the timeframe covered by the new forecast, but leaves other timeframes intact.
    """

    DEFAULT_PATH = SERVICE_CONFIG.demand_forecast.path
    OUTPUT_TIMEZONE = SERVICE_CONFIG.non_production.tz_local

    def __init__(self, path: str | None = None) -> None:
        self._path = Path(path or DemandForecast.DEFAULT_PATH)
        self._path.parent.mkdir(parents=True, exist_ok=True)

        self._lock = anyio.Lock()
        self._dataframe: pd.DataFrame | None = None
        # load persisted if present
        try:
            self._load_from_disk()
        except Exception:
            log.debug("No existing demand forecast found or failed to load.")

    def _load_from_disk(self) -> None:
        if not self._path.exists():
            self._dataframe = pd.DataFrame()
            return

        with self._path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        if not data:
            self._df = pd.DataFrame()
            return

        loaded_df = pd.DataFrame(data)
        loaded_df["Time"] = pd.to_datetime(loaded_df["Time"], utc=True)
        if "Energy (kWh)" in loaded_df.columns:
            loaded_df["energy_kwh"] = loaded_df["Energy (kWh)"]
        if "Source" in loaded_df.columns:
            loaded_df["source"] = loaded_df["Source"]
        loaded_df = loaded_df[[column for column in loaded_df.columns if column in {"Time", "energy_kwh", "source"}]]
        loaded_df = loaded_df.set_index("Time").sort_index()
        self._dataframe = loaded_df

    def _persist(self) -> None:
        # write atomically
        tmp = self._path.with_name(self._path.name + ".tmp")
        records = self._records_from_dataframe(self._dataframe)
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(records, fh, indent=2, default=str)
        tmp.replace(self._path)

    @staticmethod
    def _dataframe_from_forecast_model(forecast: DemandForecastModel) -> pd.DataFrame:
        dataframe = pd.DataFrame(
            [
                {"Time": point.time, "energy_kwh": point.energy_kwh, "source": forecast.source}
                for point in forecast.values  # noqa: PD011
            ]
        )
        dataframe["Time"] = pd.to_datetime(dataframe["Time"], utc=True)
        return dataframe.set_index("Time").sort_index()

    @staticmethod
    def _records_from_dataframe(dataframe: pd.DataFrame) -> list[dict]:
        if dataframe.empty:
            return []

        records = dataframe.reset_index(names="Time")
        if "source" in records.columns:
            records = records.sort_values(by=["Time", "source"]).reset_index(drop=True)
        else:
            records = records.sort_values(by=["Time"]).reset_index(drop=True)
        records["Time"] = (
            records["Time"]
            .dt.tz_convert(DemandForecast.OUTPUT_TIMEZONE)
            .map(lambda timestamp: timestamp.isoformat(timespec="seconds"))
        )
        records = records.rename(columns={"energy_kwh": "Energy (kWh)", "source": "Source"})
        return records.to_dict(orient="records")

    @staticmethod
    def _filter_dataframe(
        dataframe: pd.DataFrame,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        source: str | None = None,
    ) -> pd.DataFrame:
        working = dataframe
        if source is not None:
            working = working[working["source"].str.lower() == source.lower()]
        if from_time is not None:
            working = working.loc[working.index >= pd.to_datetime(from_time, utc=True)]
        if to_time is not None:
            working = working.loc[working.index <= pd.to_datetime(to_time, utc=True)]
        return working

    async def store_forecast(self, forecast: DemandForecastModel) -> None:
        """Store a forecast: replace values for the forecast's source in the timeframe covered.

        This method is safe to call concurrently.
        """
        async with self._lock:
            new_dataframe = self._dataframe_from_forecast_model(forecast)

            if self._dataframe is None or self._dataframe.empty:
                self._dataframe = new_dataframe.copy()
            else:
                # remove overlapping times for same source
                mask = ~(
                    (self._dataframe.index >= new_dataframe.index.min())
                    & (self._dataframe.index <= new_dataframe.index.max())
                    & (self._dataframe["source"] == forecast.source)
                )
                kept = self._dataframe.loc[mask]
                combined = pd.concat([kept, new_dataframe])
                self._dataframe = combined.sort_index()

            # persist synchronously in worker thread
            await anyio.to_thread.run_sync(self._persist)

    async def get_data(
        self, from_time: datetime | None = None, to_time: datetime | None = None, source: str | None = None
    ) -> list[dict]:
        async with self._lock:
            if self._dataframe is None or self._dataframe.empty:
                return []

            working = self._filter_dataframe(self._dataframe, from_time, to_time, source)

            if working.empty:
                return []

            return self._records_from_dataframe(working)

    async def get_merged_data(
        self,
        production_forecast: DemandForecastModel,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[dict]:
        async with self._lock:
            non_production = pd.DataFrame() if self._dataframe is None else self._dataframe.copy()
            if not non_production.empty:
                non_production = non_production[non_production["source"].str.lower() != "production"]

            production_dataframe = self._dataframe_from_forecast_model(production_forecast)
            combined = pd.concat([non_production, production_dataframe]).sort_index()

            if combined.empty:
                return []

            combined = self._filter_dataframe(combined, from_time, to_time)

            if combined.empty:
                return []

            return self._records_from_dataframe(combined)

    async def get_model_data(
        self,
        forecast: DemandForecastModel,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[dict]:
        async with self._lock:
            working = self._dataframe_from_forecast_model(forecast)
            working = self._filter_dataframe(working, from_time, to_time)
            return self._records_from_dataframe(working)

    async def get_horizon(self, source: str | None = None) -> dict:
        async with self._lock:
            if self._dataframe is None or self._dataframe.empty:
                return {"from_time": None, "to_time": None}
            working = self._dataframe
            if source is not None:
                working = working[working["source"].str.lower() == source.lower()]
            if working.empty:
                return {"from_time": None, "to_time": None}
            return {"from_time": working.index.min().isoformat(), "to_time": working.index.max().isoformat()}


__all__ = ["DemandForecast", "DemandForecastModel", "DemandPoint"]
