import logging
import time
from datetime import datetime, timedelta

import pandas as pd
from eta_utility.connectors.entso_e import ENTSOEConnection
from eta_utility.connectors.forecast_solar import ForecastSolarConnection
from eta_utility.connectors.node import NodeEntsoE, NodeForecastSolar
from secret import ENTSOE_API_TOKEN, FORECAST_SOLAR_API_KEY


def process_forecastsolar_data(solar_df):
    solar_df = solar_df.reset_index()
    solar_df.columns = ["Time", "PV1", "PV2"]
    solar_df["Time"] = pd.to_datetime(solar_df["Time"])
    solar_df["Energy (kWh)"] = ((solar_df["PV1"] + solar_df["PV2"]) * 0.25) / 1000
    solar_df["Cost (EUR/kWh)"] = 0.1
    solar_df["Source"] = 1

    return solar_df.drop(columns=["PV1", "PV2"])


def process_entsoe_data(entsoe_df):
    entsoe_df = entsoe_df.reset_index()
    num_columns = entsoe_df.shape[1]

    if num_columns == 2:
        entsoe_df.columns = ["Time", "Grid Price 1hP (EUR/MWh)"]
        entsoe_df["Time"] = pd.to_datetime(entsoe_df["Time"])
        entsoe_df["Energy (kWh)"] = 149
        entsoe_df["Cost (EUR/kWh)"] = entsoe_df["Grid Price 1hP (EUR/MWh)"] / 1000
        entsoe_df["Source"] = 2

        return entsoe_df.drop(columns=["Grid Price 1hP (EUR/MWh)"])

    if num_columns == 3:
        entsoe_df.columns = ["Time", "Grid Price 1hP (EUR/MWh)", "Grid Price 0.25hP (EUR/MWh)"]
        entsoe_df["Time"] = pd.to_datetime(entsoe_df["Time"])
        entsoe_df["Energy (kWh)"] = 149
        entsoe_df["Cost (EUR/kWh)"] = entsoe_df["Grid Price 1hP (EUR/MWh)"] / 1000
        entsoe_df["Source"] = 2

        return entsoe_df.drop(columns=["Grid Price 1hP (EUR/MWh)", "Grid Price 0.25hP (EUR/MWh)"])

    return None


def generate_price_matrix(forecast_solar_data, entsoe_data):
    concatenated_data = pd.concat([forecast_solar_data, entsoe_data], ignore_index=True)
    price_matrix = (
        concatenated_data.sort_values(by=["Time", "Source"])
        .reset_index(drop=True)
        .replace({1: "PV", 2: "Grid"})
        .dropna()
    )
    price_matrix["Time"] = price_matrix["Time"].dt.strftime("%H:%M")

    return price_matrix


def main() -> None:
    try:
        nodes = [
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

        connector = ForecastSolarConnection(url="https://api.forecast.solar", api_key=FORECAST_SOLAR_API_KEY)

        node_e = NodeEntsoE(
            name="entsoe",
            url="https://web-api.tp.entsoe.eu/",
            protocol="entsoe",
            bidding_zone="DEU-LUX",
            endpoint="Price",
        )

        connector_e = ENTSOEConnection(url="https://web-api.tp.entsoe.eu/", api_token=ENTSOE_API_TOKEN)

        while True:
            from_time = datetime.now()
            to_time = from_time + timedelta(days=1)
            interval = timedelta(minutes=15)

            res = connector.read_series(from_time, to_time, nodes, interval)
            res_e = connector_e.read_series(from_time, to_time, node_e, interval)

            res = process_forecastsolar_data(res)
            res_e = process_entsoe_data(res_e)

            price_matrix = generate_price_matrix(res, res_e)
            data = price_matrix.to_json()

            logging.info(data)
            # Wait for 15 minutes before the next iteration
            time.sleep(15 * 60)

    except KeyboardInterrupt:
        logging.error("Received KeyboardInterrupt: Shutting down agent")


if __name__ == "__main__":
    main()
