import pandas as pd
import pytest

from energy_information_service.dayahead_forecast_utils import (
    PRICE_COLUMN,
    _normalize_price_frame,
    fetch_entsoe_day_ahead_prices,
    generate_lag_features,
)


def test_normalize_price_frame_accepts_series():
    idx = pd.date_range("2025-01-01", periods=3, freq="h")
    series = pd.Series([10.0, 11.0, 12.0], index=idx)

    normalized = _normalize_price_frame(series, source="test")

    assert list(normalized.columns) == [PRICE_COLUMN]
    assert normalized[PRICE_COLUMN].tolist() == [10.0, 11.0, 12.0]


def test_normalize_price_frame_accepts_single_column_dataframe():
    idx = pd.date_range("2025-01-01", periods=2, freq="h")
    frame = pd.DataFrame({"value": [20.0, 21.0]}, index=idx)

    normalized = _normalize_price_frame(frame, source="test")

    assert list(normalized.columns) == [PRICE_COLUMN]
    assert normalized[PRICE_COLUMN].tolist() == [20.0, 21.0]


def test_generate_lag_features_raises_clear_error_without_price_column():
    idx = pd.date_range("2025-01-01", periods=4, freq="h", tz="UTC")
    frame = pd.DataFrame(
        {
            "wind_speed_100m_avg": [1.0, 2.0, 3.0, 4.0],
            "wind_direction_100m_avg": [1.0, 2.0, 3.0, 4.0],
            "gti_avg": [0.0, 0.0, 0.0, 0.0],
            "temperature_2m_avg": [5.0, 6.0, 7.0, 8.0],
        },
        index=idx,
    )

    with pytest.raises(ValueError, match="Missing required price column"):
        generate_lag_features(frame)


def test_fetch_entsoe_day_ahead_prices_accepts_callable_price_source():
    idx = pd.date_range("2025-01-01", periods=2, freq="h")

    def dummy_price_fetch(from_time, to_time):
        return pd.DataFrame({"Grid Price 1h (EUR/MWh)": [30.0, 31.0]}, index=idx)

    prices = fetch_entsoe_day_ahead_prices(dummy_price_fetch, start_date="2025-01-01", end_date="2025-01-01")

    assert list(prices.columns) == [PRICE_COLUMN]
    assert prices[PRICE_COLUMN].tolist() == [30.0, 31.0]
