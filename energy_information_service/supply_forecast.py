from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta

import anyio
import pandas as pd
from eta_nexus.connections import EntsoeConnection, ForecastsolarConnection
from eta_nexus.nodes import EntsoeNode, ForecastsolarNode

from energy_information_service.config import SERVICE_CONFIG
from energy_information_service.dayahead_forecast import DamForecastProvider
from energy_information_service.forecastsolar_utils import forecastsolar_to_energy_frame

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
        self._refresh_lock = anyio.Lock()
        self._df: pd.DataFrame | None = None
        self._last_refresh: datetime | None = None

        # ENTSO-E set-up
        entsoe_cfg = SERVICE_CONFIG.entsoe
        self._entsoe_node = EntsoeNode(
            name="entsoe_node",
            url=entsoe_cfg.url,
            protocol="entsoe",
            bidding_zone=entsoe_cfg.bidding_zone,
            endpoint=entsoe_cfg.endpoint,
        )
        self._entsoe_connection = EntsoeConnection.from_node(self._entsoe_node)

        # Forecast.Solar set-up
        key = os.getenv("FORECAST_SOLAR_API_TOKEN") or None
        solar_cfg = SERVICE_CONFIG.forecast_solar
        if not (len(solar_cfg.declinations) == len(solar_cfg.azimuths) == len(solar_cfg.kwps)):
            msg = "Forecast.Solar plane configuration lengths must match."
            raise ValueError(msg)

        self._pv_nodes = [
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
        self._pv_connections = [ForecastsolarConnection.from_node([node]) for node in self._pv_nodes]

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

    @staticmethod
    def _prediction_horizon() -> timedelta:
        return timedelta(hours=SERVICE_CONFIG.prediction_horizon_hours)

    async def _ensure_ready(self, *, require_fresh: bool) -> None:
        async with self._lock:
            has_snapshot = self._df is not None

        if has_snapshot and not require_fresh:
            return

        await self.refresh(force=not has_snapshot if not require_fresh else False)

    async def _snapshot(self, *, require_fresh: bool = False) -> pd.DataFrame:
        """Thread-safe snapshot of the cached 5-day matrix."""
        await self._ensure_ready(require_fresh=require_fresh)
        async with self._lock:
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

    def _fetch_pv_production(self, from_time: datetime, to_time: datetime, interval: timedelta) -> pd.DataFrame:
        plane_frames: list[pd.DataFrame] = []

        for node, connection in zip(self._pv_nodes, self._pv_connections, strict=True):
            pv_raw = connection.read_series(from_time, to_time, [node], interval)
            plane_frame = forecastsolar_to_energy_frame(pv_raw, source=1).drop(columns=["Source"])
            plane_frames.append(plane_frame)

        combined = pd.concat(plane_frames, ignore_index=True)
        aggregated = (
            combined.groupby("Time", as_index=False)
            .agg({"Energy (kWh)": "sum", "Cost (EUR/kWh)": "first"})
            .assign(Source=1)
        )
        return aggregated[["Time", "Energy (kWh)", "Cost (EUR/kWh)", "Source"]]

    # Internal: heavy work done in a worker-thread
    def _build_snapshot(self) -> pd.DataFrame:
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

        # 2) Day-Ahead prices from ENTSO-E covering at least the configured horizon
        from_time = self._quarter_hour_floor(datetime.now())
        to_time = from_time + self._prediction_horizon()
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
        pv_to = max(to_time, from_time + timedelta(days=5))

        pv_production = self._fetch_pv_production(pv_from, pv_to, interval)

        # 4) Final assembly
        combined = pd.concat([pv_production, forecast_plus_day_ahead_prices], ignore_index=True)

        tidy = (
            combined.sort_values(by=["Time", "Source"])
            .reset_index(drop=True)
            .replace({1: "PV forecast", 2: "Grid", 3: "Grid forecast"})
            .dropna()
        )

        tidy["Time"] = pd.to_datetime(tidy["Time"])
        snapshot = tidy.set_index("Time")
        log.info("Supply-Forecast data ready: %s rows", len(snapshot))
        return snapshot

    async def refresh(self, *, force: bool = False) -> None:
        now = datetime.now()
        async with self._lock:
            if not force and not self._need_refresh(now):
                return

        async with self._refresh_lock:
            async with self._lock:
                if not force and not self._need_refresh(datetime.now()):
                    return

            snapshot = await anyio.to_thread.run_sync(self._build_snapshot)
            async with self._lock:
                self._df = snapshot
                self._last_refresh = datetime.now()

    # Public API (used by FastAPI route)
    async def get_supply_forecast(self) -> list[dict]:
        """Return the configured-horizon quarter-hourly supply matrix as list[dict]."""
        snapshot = await self._snapshot(require_fresh=True)
        window_start = pd.Timestamp.now(tz="Europe/Berlin").floor("15min")
        window_end = window_start + self._prediction_horizon()
        filtered = snapshot.loc[(snapshot.index >= window_start) & (snapshot.index <= window_end)]
        return self._to_records(filtered)

    async def get_sources(self) -> list[str]:
        """Return a list of available energy sources in the 5-day supply matrix."""
        snapshot = await self._snapshot()
        return snapshot["Source"].drop_duplicates().tolist()

    async def get_data_by_source(self, energy_source: str) -> list[dict]:
        """
        Case-insensitive filter of the 5-day matrix by 'Source' (e.g. 'PV forecast', 'Grid', 'Grid forecast').
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
        Slice the cached supply matrix between from_time/to_time (inclusive), optionally filtered by Source.
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
        If *energy_source* is given (e.g. 'PV forecast' | 'Grid' | 'Grid forecast'), compute the
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


if __name__ == "__main__":
    # Quick test: run the provider and print the next 5-day supply forecast window
    dam_provider = DamForecastProvider()
    supply_provider = SupplyForecastProvider(dam_provider)
    forecast = supply_provider._build_snapshot()
