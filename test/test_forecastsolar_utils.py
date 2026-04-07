from datetime import UTC, datetime

import pandas as pd

from energy_availability_broker.forecastsolar_utils import forecastsolar_to_energy_frame


def test_forecastsolar_to_energy_frame_supports_single_power_column():
    raw = pd.DataFrame(
        {"power": [1000.0, 2000.0]},
        index=pd.to_datetime(
            [
                datetime(2025, 8, 27, 8, 0, tzinfo=UTC),
                datetime(2025, 8, 27, 8, 15, tzinfo=UTC),
            ]
        ),
    )

    result = forecastsolar_to_energy_frame(raw, source=1)

    assert result.to_dict(orient="records") == [
        {
            "Time": pd.Timestamp("2025-08-27T08:00:00+0000", tz=UTC),
            "Energy (kWh)": 0.25,
            "Cost (EUR/kWh)": 0,
            "Source": 1,
        },
        {
            "Time": pd.Timestamp("2025-08-27T08:15:00+0000", tz=UTC),
            "Energy (kWh)": 0.5,
            "Cost (EUR/kWh)": 0,
            "Source": 1,
        },
    ]


def test_forecastsolar_to_energy_frame_sums_multiple_power_columns():
    raw = pd.DataFrame(
        {"east": [1000.0], "west": [500.0]},
        index=pd.to_datetime([datetime(2025, 8, 27, 8, 0, tzinfo=UTC)]),
    )

    result = forecastsolar_to_energy_frame(raw, source="PV")

    assert result.to_dict(orient="records") == [
        {
            "Time": pd.Timestamp("2025-08-27T08:00:00+0000", tz=UTC),
            "Energy (kWh)": 0.375,
            "Cost (EUR/kWh)": 0,
            "Source": "PV",
        }
    ]
