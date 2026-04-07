from datetime import datetime

import pandas as pd

from energy_availability_broker.non_production_forecast_utils import (
    build_electrical_power_features_for_inference,
    filter_non_production_forecast_window,
    resample_non_production_power_to_quarter_hour_energy,
    resolve_issue_time_local,
)


def test_filter_non_production_forecast_window_respects_bounds():
    forecast_df = pd.DataFrame(
        {
            "ts_forecast_local": pd.to_datetime(
                [
                    "2025-11-17T08:00:00+01:00",
                    "2025-11-17T09:00:00+01:00",
                    "2025-11-17T10:00:00+01:00",
                ]
            ),
            "y_pred_kw": [1.0, 2.0, 3.0],
        }
    )

    filtered = filter_non_production_forecast_window(
        forecast_df,
        from_time=datetime.fromisoformat("2025-11-17T08:30:00+01:00"),
        to_time=datetime.fromisoformat("2025-11-17T09:30:00+01:00"),
        tz_local="Europe/Berlin",
    )

    assert filtered["y_pred_kw"].tolist() == [2.0]


def test_filter_non_production_forecast_window_localizes_naive_bounds():
    forecast_df = pd.DataFrame(
        {
            "ts_forecast_local": pd.to_datetime(
                [
                    "2025-11-17T08:00:00+01:00",
                    "2025-11-17T09:00:00+01:00",
                ]
            ),
            "y_pred_kw": [1.0, 2.0],
        }
    )

    filtered = filter_non_production_forecast_window(
        forecast_df,
        from_time="2025-11-17T09:00:00",
        to_time=None,
        tz_local="Europe/Berlin",
    )

    assert filtered["y_pred_kw"].tolist() == [2.0]


def test_resolve_issue_time_local_uses_from_time_when_issue_time_missing():
    resolved = resolve_issue_time_local(from_time="2025-11-17T09:37:00", tz_local="Europe/Berlin")

    assert resolved == pd.Timestamp("2025-11-17T09:00:00+01:00")


def test_resolve_issue_time_local_clamps_future_from_time_to_now(mocker):
    mock_now = pd.Timestamp("2025-11-17T08:15:00+01:00")
    mocker.patch("energy_availability_broker.non_production_forecast_utils.pd.Timestamp.now", return_value=mock_now)

    resolved = resolve_issue_time_local(from_time="2025-11-17T12:00:00+01:00", tz_local="Europe/Berlin")

    assert resolved == pd.Timestamp("2025-11-17T08:00:00+01:00")


def test_build_electrical_power_features_interpolates_sparse_hourly_gaps():
    ts_issue = pd.Timestamp("2025-11-17T08:00:00Z")
    hourly_index = pd.date_range(end=ts_issue, periods=170, freq="1H", tz="UTC")
    points = pd.Series(range(len(hourly_index)), index=hourly_index, dtype="float64") * 1000.0
    points = points.drop([hourly_index[20], hourly_index[90]])

    class DummyPointsApi:
        def get(self, parent_id, start, end):
            return points.loc[(points.index >= pd.Timestamp(start)) & (points.index < pd.Timestamp(end))]

    class DummySeriesApi:
        points = DummyPointsApi()

    class DummyPlatform:
        series = DummySeriesApi()

    features_df = build_electrical_power_features_for_inference(
        DummyPlatform(),
        series_id="dummy-series",
        ts_issue_utc=ts_issue,
        apply_tag_patch=False,
    )

    assert not features_df.isna().any(axis=None)


def test_resample_non_production_power_to_quarter_hour_energy():
    forecast_df = pd.DataFrame(
        {
            "ts_issue_utc": pd.to_datetime(["2025-11-17T08:00:00Z"]),
            "horizon_h": [1],
            "ts_forecast_utc": pd.to_datetime(["2025-11-17T09:00:00Z"]),
            "ts_forecast_local": pd.to_datetime(["2025-11-17T10:00:00+01:00"]),
            "y_pred_kw": [4.0],
        }
    )

    resampled = resample_non_production_power_to_quarter_hour_energy(forecast_df)

    assert resampled["ts_forecast_local"].dt.strftime("%Y-%m-%dT%H:%M:%S%z").tolist() == [
        "2025-11-17T09:00:00+0100",
        "2025-11-17T09:15:00+0100",
        "2025-11-17T09:30:00+0100",
        "2025-11-17T09:45:00+0100",
    ]
    assert resampled["non_production_energy_kwh"].tolist() == [1.0, 1.0, 1.0, 1.0]
