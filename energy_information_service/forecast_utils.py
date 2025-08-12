from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
log = logging.getLogger(__name__)


FEATURES = [
    "wind_speed_100m_avg",
    "wind_direction_100m_avg",
    "gti_avg",
    "temperature_2m_avg",
    "price_lag1",
    "wind_speed_100m_avg_lag1",
    "wind_speed_100m_avg_lag12",
    "wind_speed_100m_avg_lag24",
    "wind_direction_100m_avg_lag1",
    "wind_direction_100m_avg_lag12",
    "wind_direction_100m_avg_lag24",
    "hour",
    "minute",
    "dayofweek",
    "is_weekend",
]

# FETCH WEATHER DATA FROM OPEN-METEO


def fetch_openmeteo_data(start_date, end_date, latitude, longitude, tilt, azimuth, timezone="Europe/Berlin"):
    """
    Fetch historical weather data (2020-2025) from Open-Meteo's archive API.
    Returns a DataFrame with wind_speed_100m, wind_direction_100m, global_tilted_irradiance, temperature_2m.
    """
    # Setup caching + retry
    cache_session = requests_cache.CachedSession(".cache", expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ["wind_speed_100m", "wind_direction_100m", "global_tilted_irradiance", "temperature_2m"],
        "timezone": timezone,
        "tilt": tilt,
        "azimuth": azimuth,
    }

    responses = openmeteo.weather_api(url, params=params)
    if not responses:
        log.warning("No response from Open-Meteo.")
        return pd.DataFrame()

    # requested one location, so just use responses[0].
    response = responses[0]
    log.info(f"[Open-Meteo] Fetched data for lat={response.Latitude()}, lon={response.Longitude()}")

    # Extract hourly data
    hourly = response.Hourly()
    wind_speed_100m = hourly.Variables(0).ValuesAsNumpy()
    wind_dir_100m = hourly.Variables(1).ValuesAsNumpy()
    gti = hourly.Variables(2).ValuesAsNumpy()
    temp_2m = hourly.Variables(3).ValuesAsNumpy()

    times = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )

    df_weather = pd.DataFrame(
        index=times,
        data={
            "wind_speed_100m": wind_speed_100m,
            "wind_direction_100m": wind_dir_100m,
            "global_tilted_irradiance": gti,
            "temperature_2m": temp_2m,
        },
    )
    df_weather.index.name = "datetime"
    return df_weather


# FETCH MULTI LOCATION WEATHER DATA


def fetch_multi_location_weather(start_date, end_date):
    locations = [
        {"lat": 54.3233, "lon": 10.1228},  # North
        {"lat": 49.86381, "lon": 8.68105},  # Central
        {"lat": 48.1371, "lon": 11.5754},  # South
    ]
    dfs = []
    for loc in locations:
        df_loc = fetch_openmeteo_data(
            start_date,
            end_date,
            latitude=loc["lat"],
            longitude=loc["lon"],
            tilt=49.86381,
            azimuth=90,
            timezone="Europe/Berlin",
        )
        if not df_loc.empty:
            df_loc = df_loc.rename(
                columns={
                    "wind_speed_100m": f"wind_speed_100m_{loc['lat']:.2f}",
                    "wind_direction_100m": f"wind_direction_100m_{loc['lat']:.2f}",
                    "global_tilted_irradiance": f"gti_{loc['lat']:.2f}",
                    "temperature_2m": f"temperature_2m_{loc['lat']:.2f}",
                }
            )
            dfs.append(df_loc)

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, axis=1, join="outer").sort_index()


# FETCH DAY-AHEAD PRICE FROM ENTSO-E


def fetch_entsoe_day_ahead_prices(client, start_date="2020-01-01", end_date="2025-03-07"):
    """
    Fetch day-ahead electricity prices from ENTSO-E for the given date range
    (e.g., DE_LU). Returns a DataFrame with a 'Day-Ahead Price [€/MWh]' column.
    """
    # Convert date strings to Timestamps
    start_ts = pd.Timestamp(start_date, tz="Europe/Brussels")
    end_ts = pd.Timestamp(end_date, tz="Europe/Brussels")

    try:
        df_prices = client.query_day_ahead_prices(country_code="DE_LU", start=start_ts, end=end_ts)
        # Convert to a DataFrame with a named column
        df_prices = df_prices.to_frame(name="Day-Ahead Price [€/MWh]")
        # Make sure index is in UTC or a consistent timezone
        df_prices.index = pd.to_datetime(df_prices.index, utc=True).tz_convert("Europe/Brussels")
        return df_prices.sort_index()
    except Exception:
        log.exception("Error fetching day-ahead prices")
        return pd.DataFrame()


# COMBINE INTO ONE DATASET


def get_combined_data(client, start_date, end_date, combined_csv="combined_data.csv", force_fetch=False):
    """
    Fetches weather data and ENTSO-E day-ahead prices for the specified date range,
    then merges them into one DataFrame and saves to CSV.
    """
    csv_path = Path(combined_csv)

    if csv_path.exists() and not force_fetch:
        log.info(f"Loading combined data from {combined_csv}...")
        return pd.read_csv(combined_csv, parse_dates=True, index_col=0)

    # 1) Fetch weather data for the given date range
    log.info("Fetching historical weather data")
    df_weather = fetch_multi_location_weather(start_date, end_date)
    df_weather["wind_speed_100m_avg"] = df_weather.filter(like="wind_speed_100m_").mean(axis=1)
    df_weather["wind_direction_100m_avg"] = df_weather.filter(like="wind_direction_100m_").mean(axis=1)
    df_weather["gti_avg"] = df_weather.filter(like="gti_").mean(axis=1)
    df_weather["temperature_2m_avg"] = df_weather.filter(like="temperature_2m_").mean(axis=1)

    # 2) Fetch ENTSO-E day-ahead prices for the specified date range
    log.info("Fetching historical day ahead price")
    df_prices = fetch_entsoe_day_ahead_prices(client, start_date=start_date, end_date=end_date)

    # 3) Merge the datasets (using an outer join to keep all timestamps)
    df_combined = pd.concat([df_prices, df_weather], axis=1, join="outer").sort_index()

    # Optional: trim the DataFrame if needed (example slicing based on end_date)
    df_combined = df_combined.loc[: f"{end_date} 23:00:00"]
    df_combined.to_csv(combined_csv)
    log.info(f"Saved combined dataset to {combined_csv}. Rows: {len(df_combined)}")
    return df_combined


def create_features(df):
    df_features = df.copy()

    # Time-based features
    df_features["hour"] = df_features.index.hour
    df_features["minute"] = df_features.index.minute
    df_features["dayofweek"] = df_features.index.dayofweek
    df_features["is_weekend"] = (df_features["dayofweek"] >= 5).astype(int)  # 1 if Sat/Sun, else 0

    return df_features.dropna()


def smape(y_true, y_pred):
    """
    Computes Symmetric Mean Absolute Percentage Error (SMAPE) in percent.
    SMAPE = (100% / n) * Σ(2|y_i - ŷ_i| / (|y_i| + |ŷ_i|))
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Avoid division by zero by ignoring pairs where both y_true and y_pred are zero
    denominator = np.abs(y_true) + np.abs(y_pred)
    non_zero_mask = denominator != 0

    numerator = np.abs(y_true - y_pred)
    return (2.0 * numerator[non_zero_mask] / denominator[non_zero_mask]).mean() * 100


def create_forecast_features(df_forecast_ft: pd.DataFrame):
    """
    A function to average multi-location columns, then create time-based features.
    Modify as needed to match your original 'create_features' approach.
    """
    # 2A) Average columns
    df_forecast_ft["wind_speed_100m_avg"] = df_forecast_ft.filter(like="wind_speed_100m_").mean(axis=1)
    df_forecast_ft["wind_direction_100m_avg"] = df_forecast_ft.filter(like="wind_direction_100m_").mean(axis=1)
    df_forecast_ft["gti_avg"] = df_forecast_ft.filter(like="gti_").mean(axis=1)
    df_forecast_ft["temperature_2m_avg"] = df_forecast_ft.filter(like="temperature_2m_").mean(axis=1)

    # 2B) Time-based features
    # localize to UTC if naive, then convert to Berlin
    if df_forecast_ft.index.tz is None:
        df_forecast_ft = df_forecast_ft.tz_localize("UTC")
    df_forecast_ft = df_forecast_ft.tz_convert("Europe/Berlin")

    df_forecast_ft["hour"] = df_forecast_ft.index.hour
    df_forecast_ft["minute"] = df_forecast_ft.index.minute
    df_forecast_ft["dayofweek"] = df_forecast_ft.index.dayofweek
    df_forecast_ft["is_weekend"] = df_forecast_ft["dayofweek"].isin([5, 6]).astype(int)
    return df_forecast_ft


def generate_lag_features(df_lag_ft: pd.DataFrame, lags: list[int] | None = None):
    """
    Generate lag features for specific columns.

    This function computes lag features for the following columns:
      - 'Day-Ahead Price [€/MWh]'
      - 'wind_speed_100m_avg'
      - 'wind_direction_100m_avg'
      - 'gti_avg'
      - 'temperature_2m_avg'

    Parameters:
      df_Lag_ft : pd.DataFrame
          The input DataFrame, indexed by datetime.
      lags : list of int, optional
          List of lags (in time steps) to generate features for. Default is [1, 12, 24].

    Returns:
      pd.DataFrame
          The DataFrame with additional lag feature columns.
    """
    if lags is None:  # create a fresh list each call
        lags = [1, 12, 24]

    # Generate lag features for target variable
    price_column = "Day-Ahead Price [€/MWh]"
    for lag in lags:
        df_lag_ft[f"price_lag{lag}"] = df_lag_ft[price_column].shift(lag)

    # List of weather feature columns to generate lag features for
    weather_features = ["wind_speed_100m_avg", "wind_direction_100m_avg", "gti_avg", "temperature_2m_avg"]

    for feature in weather_features:
        for lag in lags:
            df_lag_ft[f"{feature}_lag{lag}"] = df_lag_ft[feature].shift(lag)

    return df_lag_ft


def fetch_multi_location_forecast(start_date, end_date):
    """
    Loops over multiple lat/lon for a future forecast
    (not historical data). Use the /forecast endpoint.
    """
    locations = [
        {"lat": 54.3233, "lon": 10.1228},  # North
        {"lat": 49.86381, "lon": 8.68105},  # Central
        {"lat": 48.1371, "lon": 11.5754},  # South
    ]
    dfs = []
    for loc in locations:
        # For forecast, switch from archive to forecast endpoint
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        forecast_params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ["wind_speed_100m", "wind_direction_100m", "global_tilted_irradiance", "temperature_2m"],
            "timezone": "Europe/Berlin",
        }
        responses_forecast = openmeteo.weather_api(forecast_url, params=forecast_params)
        if not responses_forecast:
            log.warning(f"No forecast data for lat={loc['lat']}, lon={loc['lon']}")
            continue
        response_forecast = responses_forecast[0]
        hourly_forecast = response_forecast.Hourly()

        # Extract arrays
        ws = hourly_forecast.Variables(0).ValuesAsNumpy()
        wd = hourly_forecast.Variables(1).ValuesAsNumpy()
        gti = hourly_forecast.Variables(2).ValuesAsNumpy()
        temp = hourly_forecast.Variables(3).ValuesAsNumpy()

        # Build a DateTime index
        times_fc = pd.date_range(
            start=pd.to_datetime(hourly_forecast.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly_forecast.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly_forecast.Interval()),
            inclusive="left",
        )

        df_loc = pd.DataFrame(
            {
                f"wind_speed_100m_{loc['lat']:.2f}": ws,
                f"wind_direction_100m_{loc['lat']:.2f}": wd,
                f"gti_{loc['lat']:.2f}": gti,
                f"temperature_2m_{loc['lat']:.2f}": temp,
            },
            index=times_fc,
        )
        df_loc.index.name = "datetime"
        dfs.append(df_loc)

    if not dfs:
        return pd.DataFrame()

    # Merge them
    return pd.concat(dfs, axis=1, join="outer").sort_index()


def build_feature_history(client, earliest="2022-01-01"):
    """
    Downloads raw price+weather back to 'earliest', engineers every feature
    the model expects, and returns a ready-to-use DataFrame.
    """
    yesterday = (pd.Timestamp.now(tz="Europe/Berlin") - timedelta(days=1)).strftime("%Y-%m-%d")

    raw = get_combined_data(
        client=client,
        start_date=earliest,
        end_date=yesterday,
        # combined_csv=None,  # <- don't create/require a file
        force_fetch=True,
    )

    with_lags = generate_lag_features(raw)
    return create_features(with_lags)


def predict_future_prices(reg, start_date, end_date, entsoe_client, df_history):
    """
    Hybrid / Recursive forecast for a multi-day horizon:
      - For the first 24 hours, use ENTSO-E day-ahead forecast prices for the "price_lag1" feature.
      - For hours beyond 24, update "price_lag1" recursively with the model's own predictions.

    Other lag features are computed from forecast data by shifting.

    Parameters:
      reg            : Trained forecasting model.
      start_date     : Start date (string, e.g., "2025-04-11") for the forecast period.
      end_date       : End date (string) for the forecast period.
      entsoe_client  : Instance for querying ENTSO-E day-ahead prices.
      df_history     : Historical DataFrame with a datetime index and at least the column
                       "Day-Ahead Price [€/MWh]". This is used to seed the forecast.

    Returns:
      df_pred        : DataFrame for the forecast period with a "prediction" column and the
                       required lag features.
    """

    start_ts = pd.Timestamp(start_date, tz="Europe/Brussels")
    end_ts = pd.Timestamp(end_date, tz="Europe/Brussels")

    # Fetch forecast weather data.
    df_forecast_raw = fetch_multi_location_forecast(start_date, end_date)
    if df_forecast_raw.empty:
        log.warning("No forecast weather data returned.")
        return None

    # Create base weather forecast freatures
    df_forecast = create_forecast_features(df_forecast_raw)

    # Compute weather lag features on the forecast data.
    df_forecast["wind_speed_100m_avg_lag1"] = df_forecast["wind_speed_100m_avg"].shift(1)
    df_forecast["wind_speed_100m_avg_lag12"] = df_forecast["wind_speed_100m_avg"].shift(12)
    df_forecast["wind_speed_100m_avg_lag24"] = df_forecast["wind_speed_100m_avg"].shift(24)

    df_forecast["wind_direction_100m_avg_lag1"] = df_forecast["wind_direction_100m_avg"].shift(1)
    df_forecast["wind_direction_100m_avg_lag12"] = df_forecast["wind_direction_100m_avg"].shift(12)
    df_forecast["wind_direction_100m_avg_lag24"] = df_forecast["wind_direction_100m_avg"].shift(24)

    df_forecast["gti_avg_lag1"] = df_forecast["gti_avg"].shift(1)
    df_forecast["gti_avg_lag12"] = df_forecast["gti_avg"].shift(12)
    df_forecast["gti_avg_lag24"] = df_forecast["gti_avg"].shift(24)

    df_forecast["temperature_2m_avg_lag1"] = df_forecast["temperature_2m_avg"].shift(1)
    df_forecast["temperature_2m_avg_lag12"] = df_forecast["temperature_2m_avg"].shift(12)
    df_forecast["temperature_2m_avg_lag24"] = df_forecast["temperature_2m_avg"].shift(24)

    # Initialize the target lag column ("price_lag1") to NaN.
    df_forecast["price_lag1"] = np.nan
    df_forecast.to_csv("df_forecast.csv")

    day_ahead_df = entsoe_client.query_day_ahead_prices(country_code="DE_LU", start=start_ts, end=end_ts).to_frame(
        name="Day-Ahead Price [€/MWh]"
    )

    # Sinitial seed price
    current_price = df_history.iloc[-1]["Day-Ahead Price [€/MWh]"]

    forecasts = []
    df_forecast = df_forecast.sort_index()
    forecast_start = df_forecast.index[0]

    for t in df_forecast.index:
        elapsed_seconds = (t - forecast_start).total_seconds()
        # For the first 24 hours, use ENTSO-E prices if available.
        if elapsed_seconds < 24 * 3600 and t in day_ahead_df.index:
            current_price = day_ahead_df.loc[t, "Day-Ahead Price [€/MWh]"]
        df_forecast.loc[t, "price_lag1"] = current_price

        # Prepare the feature vector
        row_features = df_forecast.loc[t, FEATURES].to_numpy().reshape(1, -1)
        predicted_price = reg.predict(row_features)[0]
        df_forecast.loc[t, "prediction"] = predicted_price

        # Update recursively.
        current_price = predicted_price

        forecasts.append(df_forecast.loc[t])

    df_pred = pd.DataFrame(forecasts, index=df_forecast.index)
    df_forecast.to_csv("df_forecast.csv")
    return df_pred
