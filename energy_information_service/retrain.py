"""
CLI entry-point for (re)training the day-ahead price model.

Run with:
    poetry run retrain_model
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from entsoe import EntsoePandasClient

from energy_information_service.forecast_utils import (  # (project import)
    create_features,
    get_combined_data,
)

logger = logging.getLogger(__name__)
logging.basicConfig(  # one-time global config
    level=logging.INFO,  # INFO prints to terminal
    format="%(asctime)s | %(levelname)s | %(message)s",
    stream=sys.stdout,
)

# Feature / target schema
FEATURES: list[str] = [
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

TARGET = "Day-Ahead Price [€/MWh]"


# Helper functions
def smape(y_true: np.ndarray | pd.Series, y_pred: np.ndarray | pd.Series) -> float:
    """Symmetric MAPE in %."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0
    return (2.0 * np.abs(y_true - y_pred)[mask] / denom[mask]).mean() * 100.0


def generate_lag_features(df: pd.DataFrame, lags: list[int] = (1, 12, 24)) -> pd.DataFrame:
    """Add lag columns for price and selected weather variables."""
    # target lags
    for lag in lags:
        df[f"price_lag{lag}"] = df[TARGET].shift(lag)

    weather_cols = [
        "wind_speed_100m_avg",
        "wind_direction_100m_avg",
        "gti_avg",
        "temperature_2m_avg",
    ]
    for col in weather_cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


# Core pipeline
def retrain_daily(
    entsoe_client: EntsoePandasClient,
    *,
    model_dir: Path | None = None,
    earliest_data: str = "2022-01-01",
) -> None:
    """Fetch data, (re)train the model, and persist it to *model_dir*."""
    logger.info("▶️  Starting model retraining")

    # 1. fetch / refresh raw data
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    data_df = get_combined_data(
        client=entsoe_client,
        start_date=earliest_data,
        end_date=yesterday,
        combined_csv="combined_data.csv",
        force_fetch=True,
    )
    logger.info("✅  Raw data frame shape: %s", data_df.shape)

    # 2. feature engineering
    logger.info("🛠️  Engineering lag & calendar features")
    data_df = generate_lag_features(data_df)
    data_df = create_features(data_df)

    # 3. restrict to the most-recent ~2 years
    cutoff = pd.Timestamp.now(tz="Europe/Berlin") - pd.Timedelta(days=800)
    data_df = data_df.loc[data_df.index >= cutoff].copy().sort_index()
    logger.info("🗂️  Windowed data frame shape: %s", data_df.shape)

    # 4. time-series train/validation split
    split_idx = int(0.8 * len(data_df))
    train_df, valid_df = data_df.iloc[:split_idx], data_df.iloc[split_idx:]

    x_train, y_train = train_df[FEATURES], train_df[TARGET]
    x_valid, y_valid = valid_df[FEATURES], valid_df[TARGET]

    # 5. fit first pass & report validation SMAPE
    logger.info("🏃  Fitting temporary model for validation")
    reg_tmp = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.1,
        early_stopping_rounds=50,
        objective="reg:squarederror",
    )
    reg_tmp.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=False)

    smape_val = smape(y_valid, reg_tmp.predict(x_valid))
    logger.info("📊  Validation SMAPE (last 20 %%): %.2f %%", smape_val)

    # 6. refit on full window & save
    logger.info("🔄  Re-fitting on full window and saving model")
    x_full, y_full = data_df[FEATURES], data_df[TARGET]
    reg_final = xgb.XGBRegressor(
        n_estimators=1200,
        learning_rate=0.1,
        early_stopping_rounds=50,
        objective="reg:squarederror",
    )
    reg_final.fit(x_full, y_full, eval_set=[(x_full, y_full)], verbose=False)

    if model_dir is None:
        model_dir = Path(__file__).resolve().parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = model_dir / f"xgb_daily_model_{ts}.pkl"

    joblib.dump(reg_final, model_path)
    logger.info("💾  Trained model written to %s", model_path)
    logger.info("✅  Retraining completed successfully")


# CLI wrapper
def main() -> None:  # entry-point wired in pyproject.toml
    client = EntsoePandasClient(api_key=os.environ["ENTSOE_API_KEY"])
    retrain_daily(client)


if __name__ == "__main__":
    main()
