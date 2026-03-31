from __future__ import annotations

from datetime import datetime, timedelta

import anyio
import pandas as pd

from energy_information_service.energy_availability import EnergyAvailabilityProvider


def _provider_with_cache(records: list[dict]) -> EnergyAvailabilityProvider:
    provider = EnergyAvailabilityProvider.__new__(EnergyAvailabilityProvider)
    provider._lock = anyio.Lock()
    provider._data = pd.DataFrame(records)
    return provider


def test_get_data_by_time_range_fetches_historical_range_when_not_cached():
    provider = _provider_with_cache(
        [
            {
                "Time": pd.Timestamp("2026-03-26T12:00:00+01:00"),
                "Energy (kWh)": 37.25,
                "Cost (EUR/kWh)": 0.02004,
                "Source": "Grid",
            }
        ]
    )

    requested_rows = pd.DataFrame(
        [
            {
                "Time": pd.Timestamp("2025-11-13T00:00:00+01:00"),
                "Energy (kWh)": 37.25,
                "Cost (EUR/kWh)": 0.01834,
                "Source": "Grid",
            }
        ]
    )
    calls: list[tuple[datetime | None, datetime | None, str | None]] = []

    def _fetch_data(from_time: datetime | None, to_time: datetime | None, energy_source: str | None = None):
        calls.append((from_time, to_time, energy_source))
        return requested_rows.copy()

    provider.fetch_data = _fetch_data  # type: ignore[method-assign]

    result = anyio.run(
        provider.get_data_by_time_range,
        datetime.fromisoformat("2025-11-13T00:00:00+01:00"),
        datetime.fromisoformat("2025-11-13T00:00:00+01:00"),
        None,
    )

    assert calls == [
        (
            datetime.fromisoformat("2025-11-13T00:00:00+01:00"),
            datetime.fromisoformat("2025-11-13T00:00:00+01:00"),
            None,
        )
    ]
    assert result.to_dict(orient="records") == requested_rows.to_dict(orient="records")


class _FakeForecastConnection:
    def __init__(self, values: list[float]) -> None:
        self._values = values

    def read_series(self, from_time, to_time, nodes, interval):
        _ = (from_time, to_time, nodes, interval)
        return pd.DataFrame(
            {"power": self._values},
            index=pd.to_datetime(
                [
                    "2025-11-13T00:00:00+01:00",
                    "2025-11-13T00:15:00+01:00",
                ]
            ),
        )


def test_fetch_pv_data_sums_all_configured_planes():
    provider = EnergyAvailabilityProvider.__new__(EnergyAvailabilityProvider)
    provider.forecast_nodes = [type("Node", (), {"name": f"plane_{index}"})() for index in range(1, 5)]
    provider.forecast_connections = [
        _FakeForecastConnection([1000.0, 2000.0]),
        _FakeForecastConnection([1000.0, 2000.0]),
        _FakeForecastConnection([1000.0, 2000.0]),
        _FakeForecastConnection([1000.0, 2000.0]),
    ]

    result = provider._fetch_pv_data(
        datetime.fromisoformat("2025-11-13T00:00:00+01:00"),
        datetime.fromisoformat("2025-11-13T00:15:00+01:00"),
        timedelta(minutes=15),
    )

    assert result.to_dict(orient="records") == [
        {
            "Time": pd.Timestamp("2025-11-13T00:00:00+01:00"),
            "Energy (kWh)": 1.0,
            "Cost (EUR/kWh)": 0,
            "Source": 1,
        },
        {
            "Time": pd.Timestamp("2025-11-13T00:15:00+01:00"),
            "Energy (kWh)": 2.0,
            "Cost (EUR/kWh)": 0,
            "Source": 1,
        },
    ]
