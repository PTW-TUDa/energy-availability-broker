from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import anyio
import pandas as pd
from eta_nexus.connections import EntsoeConnection, ForecastsolarConnection
from eta_nexus.nodes import EntsoeNode, ForecastsolarNode

from energy_information_service.dayahead_forecast import DamForecastProvider

log = logging.getLogger(__name__)


class SupplyForecastProvider:
    """Builds and caches the 5-day supply matrix."""

    REFRESH_MIN = 15  # rebuild cadence
    TIME_FMT = "%Y-%m-%d %H:%M:%S%z"

    def __init__(self, forecast_provider: DamForecastProvider) -> None:
        # shared DAM forecast instance (no extra API load)
        self._fp = forecast_provider

        # thread-safety
        self._lock = anyio.Lock()
        self._df: pd.DataFrame | None = None
        self._last_refresh: datetime | None = None

        # ENTSO-E set-up
        self._entsoe_node = EntsoeNode(
            name="entsoe_node",
            url="https://web-api.tp.entsoe.eu/",
            protocol="entsoe",
            bidding_zone="DEU-LUX",
            endpoint="Price",
        )
        self._entsoe_connection = EntsoeConnection.from_node(self._entsoe_node)

        # Forecast.Solar set-up
        key = os.getenv("FORECAST_SOLAR_API_TOKEN")
        self._pv_nodes = [
            ForecastsolarNode(
                name="east",
                url="https://api.forecast.solar",
                protocol="forecast_solar",
                api_key=key,
                data="watts",
                latitude=49.86381,
                longitude=8.68105,
                declination=[14, 10],
                azimuth=[90, -90],
                kwp=[23.31, 23.31],
            ),
            ForecastsolarNode(
                name="west",
                url="https://api.forecast.solar",
                protocol="forecast_solar",
                api_key=key,
                data="watts",
                latitude=49.86381,
                longitude=8.68105,
                declination=[10, 14],
                azimuth=[90, -90],
                kwp=[23.31, 23.31],
            ),
        ]
        self._pv_connection = ForecastsolarConnection.from_node(self._pv_nodes)

    # helper : round a timestamp down to the current 15-minute slice
    @staticmethod
    def _quarter_hour_floor(ts: datetime) -> datetime:
        return ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)

    def _need_refresh(self, now: datetime) -> bool:
        return (
            self._df is None
            or self._last_refresh is None
            or (now - self._last_refresh) > timedelta(minutes=self.REFRESH_MIN)
            or self._quarter_hour_floor(now) != self._quarter_hour_floor(self._last_refresh)
        )

    async def _ensure_ready(self) -> None:
        now = datetime.now()
        if self._need_refresh(now):
            # run the sync _refresh() in an AnyIO worker thread
            await anyio.to_thread.run_sync(self._refresh)

    async def _snapshot(self) -> pd.DataFrame:
        """Thread-safe snapshot of the cached 5-day matrix."""
        async with self._lock:
            await self._ensure_ready()
            # _df is populated by _refresh(); assert to satisfy type checkers
            assert self._df is not None
            return self._df.copy()

    @staticmethod
    def _to_records(dataframe: pd.DataFrame) -> list[dict]:
        """Common 'Time' formatting + to_dict conversion used by the routes."""
        return (
            dataframe.reset_index(names="Time")
            .assign(Time=lambda d: d["Time"].dt.strftime(SupplyForecastProvider.TIME_FMT))
            .to_dict(orient="records")
        )

    # Internal: heavy work done in a worker-thread
    def _refresh(self) -> None:
        """Blocking refresh: fetch / merge / tidy the three data sources."""
        log.info("Fetching Supply-Forecast Data …")

        # 1) 5-day price forecast (shared instance, cached)
        forecast_records = anyio.from_thread.run(self._fp.get_forecast)
        forecast_df = pd.DataFrame(forecast_records)
        forecast_df["Time"] = pd.to_datetime(forecast_df["Time"])
        forecast_df["Energy (kWh)"] = 149 * 0.25
        forecast_df["Cost (EUR/kWh)"] = forecast_df["Cost (EUR/MWh)"] / 1000
        forecast_df["Source"] = 3  # 3 → Forecast
        forecast_df = forecast_df.drop(columns=["Cost (EUR/MWh)"]).set_index("Time")

        # 2) 48 h Day-Ahead prices from ENTSO-E
        from_time = self._quarter_hour_floor(datetime.now())
        to_time = from_time + timedelta(days=2)
        interval = timedelta(minutes=15)

        day_ahead_prices = (
            self._entsoe_connection.read_series(from_time, to_time, self._entsoe_node, interval).reset_index().dropna()
        )

        if day_ahead_prices.shape[1] == 2:
            day_ahead_prices.columns = ["Time", "Grid Price 1h (EUR/MWh)"]
        else:
            day_ahead_prices.columns = ["Time", "Grid Price 0.25h (EUR/MWh)", "Grid Price 1h (EUR/MWh)"]
            day_ahead_prices = day_ahead_prices.drop(columns=["Grid Price 0.25h (EUR/MWh)"])

        day_ahead_prices["Time"] = pd.to_datetime(day_ahead_prices["Time"])
        day_ahead_prices["Energy (kWh)"] = 149 * 0.25
        day_ahead_prices["Cost (EUR/kWh)"] = day_ahead_prices["Grid Price 1h (EUR/MWh)"] / 1000
        day_ahead_prices["Source"] = 2  # 2 → Grid
        day_ahead_prices = day_ahead_prices.drop(columns=["Grid Price 1h (EUR/MWh)"]).set_index("Time")

        # Overwrite the first 48 h of the forecast with real Day-Ahead prices
        forecast_df.update(day_ahead_prices)
        forecast_plus_day_ahead_prices = forecast_df.reset_index()

        # 3) 5-day PV production from Forecast.Solar
        pv_from = from_time
        pv_to = from_time + timedelta(days=5)

        pv_raw = self._pv_connection.read_series(pv_from, pv_to, self._pv_nodes, interval).reset_index()
        pv_raw.columns = ["Time", "PV1", "PV2"]
        pv_raw["Time"] = pd.to_datetime(pv_raw["Time"])
        pv_raw["Energy (kWh)"] = ((pv_raw["PV1"] + pv_raw["PV2"]) * 0.25) / 1000
        pv_raw["Cost (EUR/kWh)"] = 0.1
        pv_raw["Source"] = 1  # 1 → PV
        pv_production = pv_raw.drop(columns=["PV1", "PV2"])

        # 4) Final assembly
        combined = pd.concat([pv_production, forecast_plus_day_ahead_prices], ignore_index=True)

        tidy = (
            combined.sort_values(by=["Time", "Source"])
            .reset_index(drop=True)
            .replace({1: "PV", 2: "Grid", 3: "Forecast"})
            .dropna()
        )

        tidy["Time"] = pd.to_datetime(tidy["Time"])
        self._df = tidy.set_index("Time")
        self._last_refresh = datetime.now()
        log.info("Supply-Forecast data ready: %s rows", len(self._df))

    # Public API (used by FastAPI route)
    async def get_supply_forecast(self) -> list[dict]:
        """Return a 5-day quarter-hourly supply matrix as list[dict]."""
        snapshot = await self._snapshot()
        return self._to_records(snapshot)

    async def get_sources(self) -> list[str]:
        """Return a list of available energy sources in the 5-day supply matrix."""
        snapshot = await self._snapshot()
        return snapshot["Source"].drop_duplicates().tolist()

    async def get_data_by_source(self, energy_source: str) -> list[dict]:
        """
        Case-insensitive filter of the 5-day matrix by 'Source' (e.g. 'PV', 'Grid', 'Forecast').
        Returns list[dict] with the same 'Time' formatting as /supply-forecast.
        """
        snapshot = await self._snapshot()

        mask = snapshot["Source"].str.lower() == energy_source.lower()
        filtered = snapshot.loc[mask]

        return self._to_records(filtered)

    async def get_data_by_time_range(
        self,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
        energy_source: str | None = None,
    ) -> list[dict]:
        """
        Slice the 5-day matrix between from_time/to_time (inclusive), optionally filtered by Source.
        Returns list[dict] with the same 'Time' formatting as /supply-forecast.
        """
        snapshot = await self._snapshot()

        if from_time is not None:
            snapshot = snapshot.loc[snapshot.index >= pd.to_datetime(from_time)]
        if to_time is not None:
            snapshot = snapshot.loc[snapshot.index <= pd.to_datetime(to_time)]
        if energy_source is not None:
            snapshot = snapshot.loc[snapshot["Source"].str.lower() == energy_source.lower()]

        return self._to_records(snapshot)

    async def get_horizon(self, energy_source: str | None = None) -> dict:
        """
        Return earliest/latest timestamps currently cached in the 5-day supply matrix.
        If *energy_source* is given (e.g. 'PV' | 'Grid' | 'Forecast'), compute the
        horizon for that subset only.
        Result: {"from_time": "<ISO-8601>|None", "to_time": "<ISO-8601>|None"}
        """
        # make sure cache is warm & current quarter-hour
        snapshot = await self._snapshot()

        if energy_source is not None:
            snapshot = snapshot[snapshot["Source"].str.lower() == energy_source.lower()]

        if snapshot.empty:
            return {"from_time": None, "to_time": None}

        start_ts = snapshot.index.min()
        end_ts = snapshot.index.max()
        return {"from_time": start_ts.isoformat(), "to_time": end_ts.isoformat()}
