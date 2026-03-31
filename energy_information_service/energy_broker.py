from __future__ import annotations

from datetime import datetime
from enum import Enum

import pandas as pd

from energy_information_service.demand_forecast import DemandForecast, DemandForecastModel
from energy_information_service.energy_availability import EnergyAvailabilityProvider
from energy_information_service.non_production_forecast import NonProductionPowerForecastProvider
from energy_information_service.supply_forecast import SupplyForecastProvider


class EnergyBrokerProvider:
    """Composes the broker-facing `/energy` view from cached provider snapshots."""

    def __init__(
        self,
        data_provider: EnergyAvailabilityProvider,
        supply_provider: SupplyForecastProvider,
        demand_provider: DemandForecast,
        non_production_provider: NonProductionPowerForecastProvider,
    ) -> None:
        self._data_provider = data_provider
        self._supply_provider = supply_provider
        self._demand_provider = demand_provider
        self._non_production_provider = non_production_provider

    @staticmethod
    def _source_value(source: Enum | str | None) -> str | None:
        if source is None:
            return None
        return source.value if hasattr(source, "value") else str(source)

    @staticmethod
    def _normalize_rows(rows: list[dict]) -> list[dict]:
        normalized: list[dict] = []
        for row in rows:
            normalized_row = dict(row)
            normalized_row["Time"] = pd.Timestamp(row["Time"])
            normalized.append(normalized_row)

        normalized.sort(key=lambda row: (row["Time"], str(row["Source"]).lower()))

        serialized: list[dict] = []
        for row in normalized:
            serialized_row = dict(row)
            serialized_row["Time"] = row["Time"].isoformat(timespec="seconds")
            serialized.append(serialized_row)

        return serialized

    async def _get_combined_demand_rows(
        self,
        from_time: datetime,
        to_time: datetime,
        production_forecast: DemandForecastModel,
    ) -> list[dict]:
        site_forecast = self._non_production_provider.get_demand_forecast_model(from_time, to_time)
        stored_rows = await self._demand_provider.get_data(from_time, to_time)
        supplemental_rows = [row for row in stored_rows if row["Source"].lower() not in {"site", "production"}]
        site_rows = await self._demand_provider.get_model_data(site_forecast, from_time, to_time)
        production_rows = await self._demand_provider.get_model_data(production_forecast, from_time, to_time)
        return [*supplemental_rows, *site_rows, *production_rows]

    async def get_rows(
        self,
        from_time: datetime,
        to_time: datetime,
        source: Enum | str | None,
        production_forecast: DemandForecastModel,
    ) -> list[dict]:
        source_value = self._source_value(source)
        source_name = source_value.lower() if source_value is not None else None
        rows: list[dict] = []

        if source_name in {None, "pv forecast", "grid"}:
            base_rows = await self._data_provider.get_data_by_time_range(
                from_time,
                to_time,
                source_value if source_name in {"pv forecast", "grid"} else None,
            )
            rows.extend(base_rows.to_dict(orient="records"))

        if source_name in {None, "grid forecast"}:
            rows.extend(await self._supply_provider.get_data_by_time_range(from_time, to_time, "Grid forecast"))

        if source_name is None:
            rows.extend(await self._get_combined_demand_rows(from_time, to_time, production_forecast))

        return self._normalize_rows(rows)

    async def get_sources(
        self,
        from_time: datetime,
        to_time: datetime,
        production_forecast: DemandForecastModel,
    ) -> list[str]:
        rows = await self.get_rows(from_time, to_time, None, production_forecast)
        seen: set[str] = set()
        sources: list[str] = []
        for row in rows:
            source = str(row["Source"])
            if source not in seen:
                seen.add(source)
                sources.append(source)
        return sources

    async def get_horizon(
        self,
        source: Enum | str | None,
        from_time: datetime,
        to_time: datetime,
        production_forecast: DemandForecastModel,
    ) -> dict:
        source_value = self._source_value(source)
        source_name = source_value.lower() if source_value is not None else None

        if source_name == "grid forecast":
            return await self._supply_provider.get_horizon("Grid forecast")
        if source_name in {"pv forecast", "grid"}:
            return await self._data_provider.get_horizon(source_value)

        rows = await self.get_rows(from_time, to_time, None, production_forecast)
        if not rows:
            return {"from_time": None, "to_time": None}

        times = [pd.Timestamp(row["Time"]) for row in rows]
        return {"from_time": min(times).isoformat(), "to_time": max(times).isoformat()}
