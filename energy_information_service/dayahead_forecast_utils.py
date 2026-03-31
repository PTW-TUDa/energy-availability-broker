from __future__ import annotations

import logging
from datetime import timedelta
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests
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
PRICE_COLUMN = "Day-Ahead Price [€/MWh]"


def _fetch_entsoe_prices(entsoe, from_time, to_time) -> pd.DataFrame:
    """Returns a DataFrame [Time, Grid Price 0.25h (EUR/MWh)] from the 15 min ENTSOE Day-Ahead Market prices
    for the given time frame. The DataFrame is indexed by Time and sorted in ascending order."""

    log.info(f"Fetching ENTSO-E data from {from_time} to {to_time}...")
    price_frames = []
    current_start = from_time
    try:
        while current_start < to_time:
            # if the requested timeframe is above 1 year, split into multiple queries to avoid errors from ENTSO-E API
            current_end = min(current_start + pd.Timedelta(days=300), to_time)
            log.info(f"Fetching ENTSO-E data from {current_start} to {current_end}...")
            for attempt in range(5):
                try:
                    price_data = entsoe.read_series(
                        from_time=current_start.to_pydatetime(), to_time=current_end.to_pydatetime(), interval=900
                    )
                except requests.exceptions.HTTPError:
                    log.warning(f"HTTP error when fetching ENTSO-E data (attempt {attempt + 1}/5). Retrying...")
                else:
                    break
            price_frames.append(_normalize_price_frame(price_data, source="_fetch_entsoe_prices"))
            current_start = current_end

        if price_frames == []:
            raise ValueError("No ENTSO-E price data fetched for the given time range.")

        combined = pd.concat(price_frames)
        combined = combined.loc[~combined.index.duplicated(), :].copy()
        if combined.index.has_duplicates:
            raise ValueError("Duplicate timestamps found in combined ENTSO-E price data after concatenation.")
        return combined
    except Exception:
        log.exception("Error fetching day-ahead prices")
    return pd.DataFrame()


def _normalize_price_frame(price_data: pd.Series | pd.DataFrame, source: str = "") -> pd.DataFrame:
    """
    Normalize ENTSO-E day-ahead price payloads to a DataFrame with a canonical
    `PRICE_COLUMN` name and timezone-aware DatetimeIndex.
    """
    if isinstance(price_data, pd.Series):
        df_prices = price_data.to_frame(name=PRICE_COLUMN)
    elif isinstance(price_data, pd.DataFrame):
        df_prices = price_data.copy()
        if isinstance(price_data.columns, pd.MultiIndex):
            # if entsoe connection is used, the output should behave like this
            df_prices = df_prices[[("entsoe_node_Price", "15")]]
            df_prices.columns = [PRICE_COLUMN]
        if PRICE_COLUMN not in df_prices.columns:
            if "value" in df_prices.columns:
                df_prices = df_prices.rename(columns={"value": PRICE_COLUMN})
            elif len(df_prices.columns) == 1:
                df_prices = df_prices.rename(columns={df_prices.columns[0]: PRICE_COLUMN})
            else:
                numeric_cols = [col for col in df_prices.columns if pd.api.types.is_numeric_dtype(df_prices[col])]
                if len(numeric_cols) == 1:
                    df_prices = df_prices.rename(columns={numeric_cols[0]: PRICE_COLUMN})
                else:
                    msg = (
                        f"{source}: could not identify day-ahead price column in ENTSO-E response. "
                        f"Columns: {list(df_prices.columns)}"
                    )
                    raise ValueError(msg)
    else:
        msg = f"{source}: unsupported ENTSO-E response type '{type(price_data).__name__}'"
        raise TypeError(msg)

    if PRICE_COLUMN not in df_prices.columns:
        msg = f"{source}: normalized ENTSO-E data is missing required column '{PRICE_COLUMN}'"
        raise ValueError(msg)

    return df_prices


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


def get_combined_data(entsoe, start_date, end_date, combined_csv="combined_data.csv", force_fetch=False):
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

    start_datetime = pd.to_datetime(start_date, utc=True)
    end_datetime = pd.to_datetime(end_date, utc=True)
    df_prices = _fetch_entsoe_prices(entsoe=entsoe, from_time=start_datetime, to_time=end_datetime)

    # 3) Merge the datasets (using an outer join to keep all timestamps)
    df_combined = pd.concat([df_prices, df_weather], axis=1, join="outer").sort_index()

    # Optional: trim the DataFrame if needed (example slicing based on end_date)
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
    df_lag_ft = df_lag_ft.copy()
    if lags is None:  # create a fresh list each call
        lags = [1, 12, 24]

    # Generate lag features for target variable
    price_column = PRICE_COLUMN
    if price_column not in df_lag_ft.columns:
        msg = (
            f"Missing required price column '{price_column}' for lag feature generation. "
            f"Available columns: {list(df_lag_ft.columns)}"
        )
        raise ValueError(msg)
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


def build_feature_history(entsoe, earliest="2023-01-01"):
    """
    Downloads raw price+weather back to 'earliest', engineers every feature
    the model expects, and returns a ready-to-use DataFrame.
    """
    yesterday = (pd.Timestamp.now(tz="Europe/Berlin") - timedelta(days=1)).strftime("%Y-%m-%d")

    raw = get_combined_data(
        entsoe=entsoe,
        start_date=earliest,
        end_date=yesterday,
        # combined_csv=None,  # <- don't create/require a file
        force_fetch=True,
    )
    raw = raw.interpolate(method="time")
    with_lags = generate_lag_features(raw)
    return create_features(with_lags)


def predict_future_prices(reg, start_date, end_date, entsoe, df_history):
    """
    Hybrid / Recursive forecast for a multi-day horizon:
      - For the first 24 hours, use ENTSO-E day-ahead forecast prices for the "price_lag1" feature.
      - For hours beyond 24, update "price_lag1" recursively with the model's own predictions.

    Other lag features are computed from forecast data by shifting.

    Parameters:
      reg            : Trained forecasting model.
      start_date     : Start date (string, e.g., "2025-04-11") for the forecast period.
      end_date       : End date (string) for the forecast period.
      entsoe         : Instance for querying ENTSO-E day-ahead prices.
      df_history     : Historical DataFrame with a datetime index and at least the column
                       "Day-Ahead Price [€/MWh]". This is used to seed the forecast.

    Returns:
      df_pred        : DataFrame for the forecast period with a "prediction" column and the
                       required lag features.
    """

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if df_history.empty:
        raise ValueError("No history data available.")

    # Fetch forecast weather data.
    df_forecast_raw = fetch_multi_location_forecast(start_date, end_date)
    if df_forecast_raw.empty:
        raise ValueError("No forecast weather data returned.")

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

    day_ahead_end = min(end_ts, start_ts + pd.Timedelta(days=1))
    day_ahead_df = _fetch_entsoe_prices(entsoe, from_time=start_ts, to_time=day_ahead_end)
    if day_ahead_df.empty:
        log.warning(
            "No ENTSO-E day-ahead data available yet for %s-%s; " "falling back to model-only seeding.",
            start_ts,
            day_ahead_end,
        )
        day_ahead_df = pd.DataFrame()

    # initial seed price
    current_price = df_history.iloc[-1][PRICE_COLUMN]

    forecasts = []
    df_forecast = df_forecast.sort_index()
    forecast_start = df_forecast.index[0]

    for t in df_forecast.index:
        elapsed_seconds = (t - forecast_start).total_seconds()
        # For the first 24 hours, use ENTSO-E prices if available.
        if elapsed_seconds < 24 * 3600 and t in day_ahead_df.index:
            current_price = day_ahead_df.loc[t, PRICE_COLUMN]
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
