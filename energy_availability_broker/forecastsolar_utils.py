from __future__ import annotations

import pandas as pd


def forecastsolar_to_energy_frame(raw_dataframe: pd.DataFrame, source: int | str) -> pd.DataFrame:
    """Convert a Forecast.Solar timeseries dataframe to the service's common energy format."""

    dataframe = raw_dataframe.reset_index()
    if dataframe.shape[1] < 2:
        msg = "Forecast.Solar dataframe must contain a time column and at least one power column."
        raise ValueError(msg)

    time_column = dataframe.columns[0]
    power_columns = list(dataframe.columns[1:])

    dataframe = dataframe.rename(columns={time_column: "Time"})
    dataframe["Time"] = pd.to_datetime(dataframe["Time"])
    dataframe[power_columns] = dataframe[power_columns].apply(pd.to_numeric, errors="coerce")

    total_power_watts = dataframe[power_columns].sum(axis=1, min_count=1)
    dataframe["Energy (kWh)"] = (total_power_watts * 0.25) / 1000
    dataframe["Cost (EUR/kWh)"] = 0
    dataframe["Source"] = source

    return dataframe[["Time", "Energy (kWh)", "Cost (EUR/kWh)", "Source"]]


__all__ = ["forecastsolar_to_energy_frame"]
