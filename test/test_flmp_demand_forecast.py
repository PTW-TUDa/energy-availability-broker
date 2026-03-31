from datetime import datetime

import anyio
import pytest

from energy_information_service.demand_forecast import DemandForecast
from energy_information_service.flmp_demand_forecast import (
    DEFAULT_FLMP_PATH,
    demand_forecast_model_from_flmp_file,
    flmp_response_from_file,
    flmp_response_from_payload,
    load_flmp_file_into_demand_forecast,
    parse_flmp_load_curve,
)


def test_parse_flmp_load_curve_from_sample_file():
    forecast = demand_forecast_model_from_flmp_file()
    points = forecast.values  # noqa: PD011

    assert forecast.forecast_id == "33435488-40d4-4b27-be2c-569c4f418c23"
    assert forecast.source == "flmp"
    assert len(points) == 31

    assert points[0].time == datetime.fromisoformat("2025-11-13T09:00:00+02:00")
    assert points[0].energy_kwh == 0.0

    assert points[3].time == datetime.fromisoformat("2025-11-13T09:45:00+02:00")
    assert points[3].energy_kwh == pytest.approx(0.036667)

    assert points[4].time == datetime.fromisoformat("2025-11-13T10:00:00+02:00")
    assert points[4].energy_kwh == pytest.approx(0.518283)

    assert points[-1].time == datetime.fromisoformat("2025-11-13T16:30:00+02:00")
    assert points[-1].energy_kwh == pytest.approx(1.51585)


def test_flmp_response_from_payload_resamples_to_quarter_hour_energy():
    payload = {
        "metadata": {"instanceId": "demo"},
        "flexibleLoadMeasures": [
            {
                "flexibleLoadId": {"uuid": "load-1"},
                "loadChangeProfile": [
                    {"timestamp": "2025-08-27T10:07:00+02:00", "power": {"unit": "W", "value": 500}},
                    {"timestamp": "2025-08-27T10:22:00+02:00", "power": {"unit": "MW", "value": 0.001}},
                    {"timestamp": "2025-08-27T10:41:00+02:00", "power": {"unit": "kW", "value": 0}},
                ],
            }
        ],
    }

    rows = flmp_response_from_payload(payload)
    load_curve = parse_flmp_load_curve(payload)

    assert rows == [
        {"Time": "2025-08-27T10:00:00+02:00", "Energy (kWh)": 0.066667, "Source": "production"},
        {"Time": "2025-08-27T10:15:00+02:00", "Energy (kWh)": 0.191667, "Source": "production"},
        {"Time": "2025-08-27T10:30:00+02:00", "Energy (kWh)": 0.183333, "Source": "production"},
    ]
    assert [point.energy_kwh for point in load_curve] == [0.066667, 0.191667, 0.183333]


def test_load_flmp_file_into_demand_forecast_store():
    store = DemandForecast(path="data/test_demand_forecast.json")
    store._persist = lambda: None

    parsed = anyio.run(load_flmp_file_into_demand_forecast, store, DEFAULT_FLMP_PATH)
    parsed_points = parsed.values  # noqa: PD011
    persisted = anyio.run(lambda: store.get_data(source="flmp"))

    assert len(parsed_points) == len(persisted)
    assert persisted[0]["Time"] == "2025-11-13T08:00:00+01:00"
    assert persisted[0]["Energy (kWh)"] == 0.0
    assert persisted[-1]["Time"] == "2025-11-13T15:30:00+01:00"
    assert persisted[-1]["Energy (kWh)"] == pytest.approx(1.51585)


def test_flmp_response_from_file_uses_response_field_names():
    rows = flmp_response_from_file(DEFAULT_FLMP_PATH)

    assert rows[0] == {
        "Time": "2025-11-13T09:00:00+02:00",
        "Energy (kWh)": 0.0,
        "Source": "production",
    }
    assert set(rows[0]) == {"Time", "Energy (kWh)", "Source"}
