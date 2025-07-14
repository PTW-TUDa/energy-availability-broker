from __future__ import annotations

import logging
from datetime import datetime, timedelta

import anyio
import pandas as pd
from eta_utility.connectors.entso_e import ENTSOEConnection
from eta_utility.connectors.forecast_solar import ForecastSolarConnection
from eta_utility.connectors.node import NodeEntsoE, NodeForecastSolar

from energy_information_service.forecast import ForecastProvider  # just the *type*

from .secret import ENTSOE_API_TOKEN, FORECAST_SOLAR_API_KEY

log = logging.getLogger(__name__)


class SupplyForecastProvider:
    """Builds and caches the 5-day supply matrix."""

    REFRESH_MIN = 15  # rebuild cadence

    def __init__(self, forecast_provider: ForecastProvider) -> None:
        # --- shared DAM forecast instance (no extra API load) -------------
        self._fp = forecast_provider

        # --- thread-safety -------------------------------------------------
        self._lock = anyio.Lock()
        self._df: pd.DataFrame | None = None
        self._last_refresh: datetime | None = None

        # --- ENTSO-E set-up -----------------------------------------------
        self._entsoe_node = NodeEntsoE(
            name="entsoe",
            url="https://web-api.tp.entsoe.eu/",
            protocol="entsoe",
            bidding_zone="DEU-LUX",
            endpoint="Price",
        )
        self._entsoe_connection = ENTSOEConnection.from_node(self._entsoe_node, api_token=ENTSOE_API_TOKEN)

        # --- Forecast.Solar set-up ----------------------------------------
        self._pv_nodes = [
            NodeForecastSolar(
                name="east",
                url="https://api.forecast.solar",
                protocol="forecast_solar",
                api_key=FORECAST_SOLAR_API_KEY,
                data="watts",
                latitude=49.86381,
                longitude=8.68105,
                declination=[14, 10],
                azimuth=[90, -90],
                kwp=[23.31, 23.31],
            ),
            NodeForecastSolar(
                name="west",
                url="https://api.forecast.solar",
                protocol="forecast_solar",
                api_key=FORECAST_SOLAR_API_KEY,
                data="watts",
                latitude=49.86381,
                longitude=8.68105,
                declination=[10, 14],
                azimuth=[90, -90],
                kwp=[23.31, 23.31],
            ),
        ]
        self._pv_connection = ForecastSolarConnection.from_node(self._pv_nodes)

    # ------------------------------------------------------------------
    # helper : round a timestamp down to the current 15-minute slice
    # ------------------------------------------------------------------
    @staticmethod
    def _qh_floor(ts: datetime) -> datetime:
        return ts.replace(minute=(ts.minute // 15) * 15, second=0, microsecond=0)

    # ------------------------------------------------------------------ #
    # Public API (used by FastAPI route)
    # ------------------------------------------------------------------ #
    async def get_supply_forecast(self) -> list[dict]:
        """Return a 5-day quarter-hourly supply matrix as list[dict]."""
        async with self._lock:
            now = datetime.now()
            # refresh if …
            need_refresh = (
                self._df is None  # … no cache yet
                or self._last_refresh is None  # … never refreshed
                or (now - self._last_refresh) > timedelta(minutes=self.REFRESH_MIN)  # … too old
                or self._qh_floor(now)  # … crossed into a *new* 15-min slice
                != self._qh_floor(self._last_refresh)
            )

            if need_refresh:
                await anyio.to_thread.run_sync(self._refresh)

            return (
                self._df.reset_index(names="Time")
                .assign(Time=lambda d: d["Time"].dt.strftime("%Y-%m-%d %H:%M:%S%z"))
                .to_dict(orient="records")
            )

    # ------------------------------------------------------------------ #
    # Internal: heavy work done in a worker-thread
    # ------------------------------------------------------------------ #
    def _refresh(self) -> None:
        """Blocking refresh: fetch / merge / tidy the three data sources."""
        log.info("Fetching Supply-Forecast Data …")

        # ────────────────────────────────────────────────────────────────
        # 1) 5-day price forecast (shared instance, cached)
        # ────────────────────────────────────────────────────────────────
        forecast_records = anyio.from_thread.run(self._fp.get_forecast)
        forecast_df = pd.DataFrame(forecast_records)
        forecast_df["Time"] = pd.to_datetime(forecast_df["Time"])
        forecast_df["Energy (kWh)"] = 149 * 0.25
        forecast_df["Cost (EUR/kWh)"] = forecast_df["Cost (EUR/MWh)"] / 1000
        forecast_df["Source"] = 3  # 3 → Forecast
        forecast_df = forecast_df.drop(columns=["Cost (EUR/MWh)"]).set_index("Time")

        # ────────────────────────────────────────────────────────────────
        # 2) 48 h Day-Ahead prices from ENTSO-E
        # ────────────────────────────────────────────────────────────────
        from_time = self._qh_floor(datetime.now())
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

        # ────────────────────────────────────────────────────────────────
        # 3) 5-day PV production from Forecast.Solar
        # ────────────────────────────────────────────────────────────────
        pv_from = from_time
        pv_to = from_time + timedelta(days=5)

        pv_raw = self._pv_connection.read_series(pv_from, pv_to, self._pv_nodes, interval).reset_index()
        pv_raw.columns = ["Time", "PV1", "PV2"]
        pv_raw["Time"] = pd.to_datetime(pv_raw["Time"])
        pv_raw["Energy (kWh)"] = ((pv_raw["PV1"] + pv_raw["PV2"]) * 0.25) / 1000
        pv_raw["Cost (EUR/kWh)"] = 0.1
        pv_raw["Source"] = 1  # 1 → PV
        pv_production = pv_raw.drop(columns=["PV1", "PV2"])

        # ────────────────────────────────────────────────────────────────
        # 4) Final assembly
        # ────────────────────────────────────────────────────────────────
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
