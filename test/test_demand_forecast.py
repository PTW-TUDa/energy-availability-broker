import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import anyio
from fastapi.testclient import TestClient

os.environ.setdefault("FORECAST_SOLAR_API_TOKEN", "A1B2C3D4E5F6G7H8")

from energy_availability_broker import main
from energy_availability_broker.demand_forecast import DemandForecast, DemandForecastModel, DemandPoint


def _make_payload(start: datetime, values: list[float], source: str = "site") -> dict:
    return {
        "source": source,
        "values": [
            {"time": (start + timedelta(minutes=15 * index)).isoformat(), "energy_kwh": energy_kwh}
            for index, energy_kwh in enumerate(values)
        ],
    }


class StubNonProductionForecastProvider:
    def __init__(self, payload: DemandForecastModel):
        self.payload = payload

    def get_demand_forecast_model(self, from_time=None, to_time=None) -> DemandForecastModel:
        return self.payload

    def get_forecast(self, from_time=None, to_time=None):
        payload = self.payload.model_dump(mode="json")
        return [
            {
                "Time": point["time"].replace("+00:00", "Z"),
                "non_production_energy_kwh": point["energy_kwh"],
            }
            for point in payload["values"]
        ]


def test_store_and_get_persistence():
    path = Path("data/test_store_and_get_persistence.json")
    try:
        if path.exists():
            path.unlink()
        forecast_store = DemandForecast(path=str(path))

        start = datetime.now(UTC).replace(microsecond=0)
        payload = DemandForecastModel(source="site", values=[DemandPoint(time=start, energy_kwh=1.0)])

        anyio.run(forecast_store.store_forecast, payload)

        reloaded_store = DemandForecast(path=str(path))
        records = anyio.run(reloaded_store.get_data)
        assert len(records) == 1
        assert records[0]["Energy (kWh)"] == 1.0
        assert records[0]["Source"] == "site"
    finally:
        if path.exists():
            path.unlink()


def test_get_endpoint_returns_merged_non_production_and_production(monkeypatch):
    provider = DemandForecast(path="data/test_merged_endpoint.json")
    provider._persist = lambda: None
    monkeypatch.setattr(main, "demand_forecast_provider", provider)

    start = datetime(2025, 8, 27, 8, 0, tzinfo=UTC)
    auxiliary_payload = DemandForecastModel(
        source="auxiliary",
        values=[
            DemandPoint(time=start, energy_kwh=10.0),
            DemandPoint(time=start + timedelta(minutes=30), energy_kwh=0.0),
        ],
    )
    anyio.run(provider.store_forecast, auxiliary_payload)

    production_payload = DemandForecastModel(
        source="production",
        values=[
            DemandPoint(time=start + timedelta(minutes=5), energy_kwh=3.0),
            DemandPoint(time=start + timedelta(minutes=20), energy_kwh=0.0),
        ],
    )
    site_payload = DemandForecastModel(
        source="site",
        values=[
            DemandPoint(time=start, energy_kwh=7.0),
            DemandPoint(time=start + timedelta(minutes=15), energy_kwh=9.0),
            DemandPoint(time=start + timedelta(minutes=30), energy_kwh=0.0),
        ],
    )
    main.app.dependency_overrides[main.get_production_demand_forecast] = lambda: production_payload
    main.app.dependency_overrides[main.get_non_production_forecast_provider] = lambda: (
        StubNonProductionForecastProvider(site_payload)
    )

    client = TestClient(main.app)
    try:
        response = client.get(
            "/demand-forecast",
            params={
                "from_time": "2025-08-27T08:00:00+00:00",
                "to_time": "2025-08-27T08:30:00+00:00",
            },
        )

        assert response.status_code == 200
        assert response.json() == [
            {"Time": "2025-08-27T10:00:00+02:00", "Energy (kWh)": 10.0, "Source": "auxiliary"},
            {"Time": "2025-08-27T10:00:00+02:00", "Energy (kWh)": 7.0, "Source": "site"},
            {"Time": "2025-08-27T10:05:00+02:00", "Energy (kWh)": 3.0, "Source": "production"},
            {"Time": "2025-08-27T10:15:00+02:00", "Energy (kWh)": 9.0, "Source": "site"},
            {"Time": "2025-08-27T10:20:00+02:00", "Energy (kWh)": 0.0, "Source": "production"},
            {"Time": "2025-08-27T10:30:00+02:00", "Energy (kWh)": 0.0, "Source": "auxiliary"},
            {"Time": "2025-08-27T10:30:00+02:00", "Energy (kWh)": 0.0, "Source": "site"},
        ]
    finally:
        main.app.dependency_overrides.clear()


def test_get_endpoint_source_filter_still_returns_raw_non_production_data(monkeypatch):
    provider = DemandForecast(path="data/test_source_filter_endpoint.json")
    provider._persist = lambda: None
    monkeypatch.setattr(main, "demand_forecast_provider", provider)
    main.app.dependency_overrides[main.get_production_demand_forecast] = lambda: DemandForecastModel(
        source="production",
        values=[DemandPoint(time=datetime(2025, 8, 27, 8, 0, tzinfo=UTC), energy_kwh=5.0)],
    )
    main.app.dependency_overrides[main.get_non_production_forecast_provider] = lambda: (
        StubNonProductionForecastProvider(
            DemandForecastModel(
                source="site",
                values=[
                    DemandPoint(time=datetime(2025, 8, 27, 8, 0, tzinfo=UTC), energy_kwh=10.0),
                    DemandPoint(time=datetime(2025, 8, 27, 8, 15, tzinfo=UTC), energy_kwh=0.0),
                ],
            )
        )
    )

    start = datetime(2025, 8, 27, 8, 0, tzinfo=UTC)
    payload = DemandForecastModel(
        source="auxiliary",
        values=[
            DemandPoint(time=start, energy_kwh=4.0),
            DemandPoint(time=start + timedelta(minutes=15), energy_kwh=0.0),
        ],
    )
    anyio.run(provider.store_forecast, payload)

    client = TestClient(main.app)
    try:
        response = client.get(
            "/demand-forecast",
            params={
                "source": "site",
                "from_time": "2025-08-27T08:00:00+00:00",
                "to_time": "2025-08-27T08:15:00+00:00",
            },
        )

        assert response.status_code == 200
        assert response.json() == [
            {"Time": "2025-08-27T10:00:00+02:00", "Energy (kWh)": 10.0, "Source": "site"},
            {"Time": "2025-08-27T10:15:00+02:00", "Energy (kWh)": 0.0, "Source": "site"},
        ]
    finally:
        main.app.dependency_overrides.clear()
