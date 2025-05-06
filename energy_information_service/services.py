from __future__ import annotations

import logging
from datetime import datetime, timedelta

import anyio
import pandas as pd
from eta_utility.connectors.entso_e import ENTSOEConnection
from eta_utility.connectors.forecast_solar import ForecastSolarConnection
from eta_utility.connectors.node import NodeEntsoE, NodeForecastSolar

from .secret import ENTSOE_API_TOKEN, FORECAST_SOLAR_API_KEY

logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger(__name__)


class DataProvider:
    def __init__(self):
        self._running = False
        self._task_group = None
        self._lock = anyio.Lock()  # Ensures safe access to data
        self._data = pd.DataFrame()

        self.forecast_nodes = [
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
                name="east",
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
        self.forecast_connection = ForecastSolarConnection.from_node(self.forecast_nodes)

        self.entsoe_node = NodeEntsoE(
            name="entsoe",
            url="https://web-api.tp.entsoe.eu/",
            protocol="entsoe",
            bidding_zone="DEU-LUX",
            endpoint="Price",
        )
        self.entsoe_connection = ENTSOEConnection.from_node(self.entsoe_node, api_token=ENTSOE_API_TOKEN)

    async def get_data(self):
        """Returns the latest DataFrame safely."""
        async with self._lock:
            return self._data.copy()

    async def run(self, task_status=anyio.TASK_STATUS_IGNORED):
        """Runs the background task continuously."""
        task_status.started()  # Signal that the task has started

        log.info("Background task is running...")

        price_matrix = self.fetch_data()
        async with self._lock:
            self._data = price_matrix
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

        if from_time is not None:
            price_matrix = price_matrix[price_matrix["Time"] >= from_time]
        if to_time is not None:
            price_matrix = price_matrix[price_matrix["Time"] <= to_time]
        if energy_source is not None:
            price_matrix = price_matrix[price_matrix["Source"].str.lower() == energy_source.lower()]

        return price_matrix

    async def get_horizon(self) -> dict:
        """
        Return the earliest and latest timestamps that are presently cached.
        Result format: {"start_time": "<ISO-8601>", "end_time": "<ISO-8601>"}
        """
        async with self._lock:
            if self._data.empty:
                return {"start_time": None, "end_time": None}

            start_ts: datetime = self._data["Time"].min()
            end_ts: datetime = self._data["Time"].max()

        return {
            "start_time": start_ts.isoformat(),
            "end_time": end_ts.isoformat(),
        }

    def fetch_data(self):
        from_time = datetime.now()
        to_time = from_time + timedelta(days=1)
        interval = timedelta(minutes=15)

        pv_production = self.forecast_connection.read_series(from_time, to_time, self.forecast_nodes, interval)
        day_ahead_prices = self.entsoe_connection.read_series(from_time, to_time, self.entsoe_node, interval)

        pv_production = pv_production.reset_index()
        pv_production.columns = ["Time", "PV1", "PV2"]
        pv_production["Time"] = pd.to_datetime(pv_production["Time"])
        pv_production["Energy (kWh)"] = ((pv_production["PV1"] + pv_production["PV2"]) * 0.25) / 1000
        pv_production["Cost (EUR/kWh)"] = 0.1
        pv_production["Source"] = 1
        pv_production = pv_production.drop(columns=["PV1", "PV2"])

        day_ahead_prices = day_ahead_prices.reset_index()
        num_columns = day_ahead_prices.shape[1]

        if num_columns == 2:
            day_ahead_prices.columns = ["Time", "Grid Price 1h (EUR/MWh)"]
            day_ahead_prices["Time"] = pd.to_datetime(day_ahead_prices["Time"])
            day_ahead_prices["Energy (kWh)"] = 149 * 0.25
            day_ahead_prices["Cost (EUR/kWh)"] = day_ahead_prices["Grid Price 1h (EUR/MWh)"] / 1000
            day_ahead_prices["Source"] = 2

            day_ahead_prices = day_ahead_prices.drop(columns=["Grid Price 1h (EUR/MWh)"])

        if num_columns == 3:
            day_ahead_prices.columns = ["Time", "Grid Price 1h (EUR/MWh)", "Grid Price 0.25h (EUR/MWh)"]
            day_ahead_prices["Time"] = pd.to_datetime(day_ahead_prices["Time"])
            day_ahead_prices["Energy (kWh)"] = 149 * 0.25
            day_ahead_prices["Cost (EUR/kWh)"] = day_ahead_prices["Grid Price 1h (EUR/MWh)"] / 1000
            day_ahead_prices["Source"] = 2

            day_ahead_prices = day_ahead_prices.drop(columns=["Grid Price 1h (EUR/MWh)", "Grid Price 0.25h (EUR/MWh)"])

        price_matrix = pd.concat([pv_production, day_ahead_prices], ignore_index=True)
        return (
            price_matrix.sort_values(by=["Time", "Source"])
            .reset_index(drop=True)
            .replace({1: "PV", 2: "Grid"})
            .dropna()
        )

        # price_matrix["Time"] = price_matrix["Time"].dt.strftime("%H:%M")
        # return price_matrix
