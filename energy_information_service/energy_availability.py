from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import anyio
import pandas as pd
from eta_nexus.connections import EntsoeConnection, ForecastsolarConnection
from eta_nexus.nodes import EntsoeNode, ForecastsolarNode

from energy_information_service.config import SERVICE_CONFIG
from energy_information_service.forecastsolar_utils import forecastsolar_to_energy_frame

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class EnergyAvailabilityProvider:
    def __init__(self):
        self._running = False
        self._task_group = None
        self._lock = anyio.Lock()  # Ensures safe access to data
        self._refresh_lock = anyio.Lock()
        self._data = pd.DataFrame()

        key = os.getenv("FORECAST_SOLAR_API_TOKEN") or None
        solar_cfg = SERVICE_CONFIG.forecast_solar
        if not (len(solar_cfg.declinations) == len(solar_cfg.azimuths) == len(solar_cfg.kwps)):
            msg = "Forecast.Solar plane configuration lengths must match."
            raise ValueError(msg)

        self.forecast_nodes = [
            ForecastsolarNode(
                name=f"plane_{index + 1}",
                url=solar_cfg.url,
                protocol="forecast_solar",
                api_key=key,
                data=solar_cfg.data,
                latitude=solar_cfg.latitude,
                longitude=solar_cfg.longitude,
                declination=abs(declination),
                azimuth=azimuth,
                kwp=kwp,
            )
            for index, (declination, azimuth, kwp) in enumerate(
                zip(solar_cfg.declinations, solar_cfg.azimuths, solar_cfg.kwps, strict=True)
            )
        ]
        self.forecast_connections = [ForecastsolarConnection.from_node([node]) for node in self.forecast_nodes]

        entsoe_cfg = SERVICE_CONFIG.entsoe
        self.entsoe_node = EntsoeNode(
            name="entsoe_node",
            url=entsoe_cfg.url,
            protocol="entsoe",
            bidding_zone=entsoe_cfg.bidding_zone,
            endpoint=entsoe_cfg.endpoint,
        )
        self.entsoe_connection = EntsoeConnection.from_node(self.entsoe_node)

    async def get_data(self):
        """Returns the latest DataFrame safely."""
        async with self._lock:
            return self._data.copy()

    async def run(self, task_status=anyio.TASK_STATUS_IGNORED):
        """Runs the background task continuously."""
        task_status.started()  # Signal that the task has started

        log.info("Background task is running...")
        price_matrix = await self.refresh(force=True)
        log.info("Updated DataFrame rows %s", len(price_matrix))

    async def get_data_by_source(self, energy_source: str):
        """
        Returns data filtered by the specified energy source (e.g. 'PV' or 'Grid').
        Case-insensitive (i.e., 'pv' or 'PV' will work).
        """
        async with self._lock:
            price_matrix = self._data.copy()
        return price_matrix[price_matrix["Source"].str.lower() == energy_source.lower()]

    async def get_sources(self):
        """
        Returns a list of available energy sources.
        """
        async with self._lock:
            price_matrix = self._data.copy()
        return price_matrix["Source"].unique().tolist()

    async def get_data_by_time_range(
        self,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        energy_source: str | None = None,
    ):
        """Return a slice of the matrix between from_time and to_time (inclusive)."""
        async with self._lock:
            price_matrix = self._data.copy()

        if self._needs_fetch(price_matrix, from_time, to_time):
            price_matrix = await self.refresh(from_time, to_time, energy_source)

        if from_time is not None:
            price_matrix = price_matrix[price_matrix["Time"] >= from_time]
        if to_time is not None:
            price_matrix = price_matrix[price_matrix["Time"] <= to_time]
        if energy_source is not None:
            price_matrix = price_matrix[price_matrix["Source"].str.lower() == energy_source.lower()]

        return price_matrix

    @staticmethod
    def _merge_frames(existing: pd.DataFrame, requested: pd.DataFrame) -> pd.DataFrame:
        if existing.empty:
            return requested.reset_index(drop=True)
        if requested.empty:
            return existing.reset_index(drop=True)
        return (
            pd.concat([existing, requested], ignore_index=True)
            .drop_duplicates(subset=["Time", "Source"], keep="last")
            .sort_values(by=["Time", "Source"])
            .reset_index(drop=True)
        )

    async def refresh(
        self,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        energy_source: str | None = None,
        *,
        force: bool = False,
    ) -> pd.DataFrame:
        if not hasattr(self, "_refresh_lock"):
            self._refresh_lock = anyio.Lock()

        async with self._lock:
            snapshot = self._data.copy()

        if not force and not self._needs_fetch(snapshot, from_time, to_time):
            return snapshot

        async with self._refresh_lock:
            async with self._lock:
                snapshot = self._data.copy()

            if not force and not self._needs_fetch(snapshot, from_time, to_time):
                return snapshot

            requested = await anyio.to_thread.run_sync(self.fetch_data, from_time, to_time, energy_source)
            merged = self._merge_frames(snapshot, requested)

            async with self._lock:
                self._data = merged
                return self._data.copy()

    async def get_horizon(self, energy_source: str | None = None) -> dict:
        """
        Return the earliest and latest timestamps that are presently cached.
        If *energy_source* is specified, return the timestamps for that source only.
        Result format: {"from_time": "<ISO-8601>", "to_time": "<ISO-8601>"}
        """
        async with self._lock:
            if self._data.empty:
                return {"from_time": None, "to_time": None}

            price_matrix = self._data.copy()
            if energy_source is not None:
                price_matrix = price_matrix[price_matrix["Source"].str.lower() == energy_source.lower()]

            if price_matrix.empty:
                return {"from_time": None, "to_time": None}

            start_ts: datetime = price_matrix["Time"].min()
            end_ts: datetime = price_matrix["Time"].max()

        return {
            "from_time": start_ts.isoformat(),
            "to_time": end_ts.isoformat(),
        }

    @staticmethod
    def _empty_frame() -> pd.DataFrame:
        return pd.DataFrame(columns=["Time", "Energy (kWh)", "Cost (EUR/kWh)", "Source"])

    @staticmethod
    def _needs_fetch(price_matrix: pd.DataFrame, from_time: datetime | None, to_time: datetime | None) -> bool:
        if from_time is None and to_time is None:
            return price_matrix.empty
        if price_matrix.empty:
            return True

        cache_start = price_matrix["Time"].min()
        cache_end = price_matrix["Time"].max()

        return bool(
            (from_time is not None and from_time < cache_start) or (to_time is not None and to_time > cache_end)
        )

    def _fetch_grid_data(self, from_time: datetime, to_time: datetime, interval: timedelta) -> pd.DataFrame:
        day_ahead_prices = self.entsoe_connection.read_series(
            from_time, to_time, self.entsoe_node, interval
        ).reset_index()
        num_columns = day_ahead_prices.shape[1]

        if num_columns == 2:
            day_ahead_prices.columns = ["Time", "Grid Price 1h (EUR/MWh)"]
            day_ahead_prices["Time"] = pd.to_datetime(day_ahead_prices["Time"])
            day_ahead_prices["Energy (kWh)"] = 149 * 0.25
            day_ahead_prices["Cost (EUR/kWh)"] = day_ahead_prices["Grid Price 1h (EUR/MWh)"] / 1000
            day_ahead_prices["Source"] = 2

            return day_ahead_prices.drop(columns=["Grid Price 1h (EUR/MWh)"])

        if num_columns == 3:
            day_ahead_prices.columns = ["Time", "Grid Price 0.25h (EUR/MWh)", "Grid Price 1h (EUR/MWh)"]
            day_ahead_prices["Time"] = pd.to_datetime(day_ahead_prices["Time"])
            day_ahead_prices["Energy (kWh)"] = 149 * 0.25
            day_ahead_prices["Cost (EUR/kWh)"] = day_ahead_prices["Grid Price 1h (EUR/MWh)"] / 1000
            day_ahead_prices["Source"] = 2

            return day_ahead_prices.drop(columns=["Grid Price 1h (EUR/MWh)", "Grid Price 0.25h (EUR/MWh)"])

        return self._empty_frame()

    def _fetch_pv_data(self, from_time: datetime, to_time: datetime, interval: timedelta) -> pd.DataFrame:
        plane_frames: list[pd.DataFrame] = []

        for node, connection in zip(self.forecast_nodes, self.forecast_connections, strict=True):
            try:
                pv_raw = connection.read_series(from_time, to_time, [node], interval)
            except Exception:
                log.warning(
                    "Forecast.Solar data unavailable for plane %s in range %s - %s",
                    node.name,
                    from_time,
                    to_time,
                    exc_info=True,
                )
                continue

            plane_frame = forecastsolar_to_energy_frame(pv_raw, source=1).drop(columns=["Source"])
            plane_frames.append(plane_frame)

        if not plane_frames:
            return self._empty_frame()

        combined = pd.concat(plane_frames, ignore_index=True)
        aggregated = (
            combined.groupby("Time", as_index=False)
            .agg({"Energy (kWh)": "sum", "Cost (EUR/kWh)": "first"})
            .assign(Source=1)
        )
        return aggregated[["Time", "Energy (kWh)", "Cost (EUR/kWh)", "Source"]]

    @staticmethod
    def _prediction_horizon() -> timedelta:
        return timedelta(hours=SERVICE_CONFIG.prediction_horizon_hours)

    def fetch_data(
        self,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        energy_source: str | None = None,
    ) -> pd.DataFrame:
        horizon = self._prediction_horizon()
        from_time = from_time or datetime.now().astimezone()
        to_time = to_time or (from_time + horizon)
        interval = timedelta(minutes=15)

        frames: list[pd.DataFrame] = []
        source_name = energy_source.lower() if energy_source is not None else None

        if source_name in {None, "pv", "pv forecast"}:
            pv_frame = self._fetch_pv_data(from_time, to_time, interval)
            if not pv_frame.empty:
                frames.append(pv_frame)

        if source_name in {None, "grid"}:
            frames.append(self._fetch_grid_data(from_time, to_time, interval))

        if not frames:
            return self._empty_frame()

        price_matrix = pd.concat(frames, ignore_index=True)
        return (
            price_matrix.sort_values(by=["Time", "Source"])
            .reset_index(drop=True)
            .replace({1: "PV forecast", 2: "Grid"})
            .dropna()
        )


if __name__ == "__main__":
    provider = EnergyAvailabilityProvider()
    data = provider.fetch_data()
