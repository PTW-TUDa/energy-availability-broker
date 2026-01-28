# test/test_main.py
# Tests for energy_information_service.main with sample data blocks.

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import energy_information_service.main as api_main
from energy_information_service.main import (
    app,
    get_data_provider,
    get_forecast_provider,
    get_supply_forecast_provider,
)

# Records used by /data, /csv, /source/{...}, /sources, /data/time-range, /data/horizon
DATA_RECORDS: list[dict[str, Any]] = [
    {"Time": "2025-08-18T12:00:00+02:00", "Energy (kWh)": 37.25, "Cost (EUR/kWh)": 0.02004, "Source": "Grid"},
    {"Time": "2025-08-18T12:15:00+02:00", "Energy (kWh)": 16.269, "Cost (EUR/kWh)": 0.1, "Source": "PV"},
    {"Time": "2025-08-18T14:30:00+02:00", "Energy (kWh)": 37.25, "Cost (EUR/kWh)": 0.01834, "Source": "Grid"},
    {"Time": "2025-08-18T14:45:00+02:00", "Energy (kWh)": 15.3425, "Cost (EUR/kWh)": 0.1, "Source": "PV"},
]

# Response rows returned by /dam-forecast
FORECAST_ROWS: list[dict[str, Any]] = [
    {"Time": "2025-08-18 12:00:00+0200", "Cost (EUR/MWh)": 30.4739532470703},
    {"Time": "2025-08-18 12:15:00+0200", "Cost (EUR/MWh)": 30.4739532470703},
    {"Time": "2025-08-18 12:30:00+0200", "Cost (EUR/MWh)": 30.4739532470703},
    {"Time": "2025-08-18 12:45:00+0200", "Cost (EUR/MWh)": 30.4739532470703},
    {"Time": "2025-08-18 13:00:00+0200", "Cost (EUR/MWh)": 39.0820846557617},
    {"Time": "2025-08-18 13:15:00+0200", "Cost (EUR/MWh)": 39.0820846557617},
    {"Time": "2025-08-18 13:30:00+0200", "Cost (EUR/MWh)": 39.0820846557617},
    {"Time": "2025-08-18 13:45:00+0200", "Cost (EUR/MWh)": 39.0820846557617},
    {"Time": "2025-08-18 14:00:00+0200", "Cost (EUR/MWh)": 63.9502983093262},
    {"Time": "2025-08-18 14:15:00+0200", "Cost (EUR/MWh)": 63.9502983093262},
    {"Time": "2025-08-18 14:30:00+0200", "Cost (EUR/MWh)": 63.9502983093262},
    {"Time": "2025-08-18 14:45:00+0200", "Cost (EUR/MWh)": 63.9502983093262},
]

# Response rows returned by /supply-forecast
SUPPLY_FORECAST_ROWS: list[dict[str, Any]] = [
    {"Time": "2025-08-19 12:00:00+0200", "Energy (kWh)": 13.426, "Cost (EUR/kWh)": 0.1, "Source": "PV"},
    {
        "Time": "2025-08-19 12:00:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.395589630126953,
        "Source": "Forecast",
    },
    {"Time": "2025-08-19 12:15:00+0200", "Energy (kWh)": 13.6015, "Cost (EUR/kWh)": 0.1, "Source": "PV"},
    {
        "Time": "2025-08-19 12:15:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.395589630126953,
        "Source": "Forecast",
    },
    {"Time": "2025-08-19 12:30:00+0200", "Energy (kWh)": 13.70075, "Cost (EUR/kWh)": 0.1, "Source": "PV"},
    {
        "Time": "2025-08-19 12:30:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.395589630126953,
        "Source": "Forecast",
    },
    {"Time": "2025-08-19 12:45:00+0200", "Energy (kWh)": 13.682, "Cost (EUR/kWh)": 0.1, "Source": "PV"},
    {
        "Time": "2025-08-19 12:45:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.395589630126953,
        "Source": "Forecast",
    },
    {"Time": "2025-08-19 13:00:00+0200", "Energy (kWh)": 13.7635, "Cost (EUR/kWh)": 0.1, "Source": "PV"},
    {
        "Time": "2025-08-19 13:00:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.409508483886719,
        "Source": "Forecast",
    },
]


# Helpers
def _csv_from_records(records: list[dict[str, Any]]) -> str:
    if not records:
        return ""
    keys = list(records[0].keys())
    header = ",".join(keys)
    rows = [",".join(str(r[k]) for k in keys) for r in records]
    return header + "\n" + "\n".join(rows)


def _parse_iso(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


# Minimal DataFrame stand-in used by mocks
class MockDF:
    def __init__(self, records: list[dict[str, Any]]):
        self._records = list(records)
        self.empty = len(self._records) == 0

    def to_dict(self, orient: str):
        assert orient == "records"
        return list(self._records)

    def to_csv(self, index: bool = False) -> str:
        assert index is False
        return _csv_from_records(self._records)


# Dummy scheduler to avoid real background jobs during tests
class DummyAsyncScheduler:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_):
        return False

    async def add_schedule(self, *_a, **_k): ...
    async def start_in_background(self): ...


# Fixtures
@pytest.fixture()
def data_provider_mock():
    m = AsyncMock(name="DataProviderMock")

    # /data
    async def get_data():
        return MockDF(DATA_RECORDS)

    # /source/{energy_source}
    async def get_data_by_source(src: str):
        filtered = [r for r in DATA_RECORDS if str(r.get("Source", "")).lower() == str(src).lower()]
        return MockDF(filtered)

    # /sources
    async def get_sources():
        seen, result = set(), []
        for r in DATA_RECORDS:
            s = r.get("Source")
            if s is not None and (s not in seen):
                seen.add(s)
                result.append(s)
        return result

    # /data/time-range
    async def get_data_by_time_range(from_time, to_time, source: str | None):
        rows = DATA_RECORDS
        if source is not None:
            rows = [r for r in rows if str(r.get("Source", "")).lower() == source.lower()]
        if from_time is not None:
            rows = [r for r in rows if _parse_iso(r["Time"]) >= from_time]
        if to_time is not None:
            rows = [r for r in rows if _parse_iso(r["Time"]) <= to_time]
        return MockDF(rows)

    # /data/horizon
    async def get_horizon(source: str | None):
        rows = DATA_RECORDS
        if source is not None:
            rows = [r for r in rows if str(r.get("Source", "")).lower() == source.lower()]
        if not rows:
            return {"from_time": None, "to_time": None}
        times = [_parse_iso(r["Time"]) for r in rows]
        return {"from_time": min(times).isoformat(), "to_time": max(times).isoformat()}

    m.get_data.side_effect = get_data
    m.get_data_by_source.side_effect = get_data_by_source
    m.get_sources.side_effect = get_sources
    m.get_data_by_time_range.side_effect = get_data_by_time_range
    m.get_horizon.side_effect = get_horizon
    return m


@pytest.fixture()
def dam_forecast_provider_mock():
    m = AsyncMock(name="DamForecastProviderMock")
    m.get_forecast.return_value = FORECAST_ROWS
    return m


@pytest.fixture()
def supply_forecast_provider_mock():
    m = AsyncMock(name="SupplyForecastProviderMock")
    m.get_supply_forecast.return_value = SUPPLY_FORECAST_ROWS
    return m


@pytest.fixture()
def client(monkeypatch, data_provider_mock, dam_forecast_provider_mock, supply_forecast_provider_mock):
    monkeypatch.setattr(api_main, "AsyncScheduler", DummyAsyncScheduler, raising=True)

    # dependency overrides
    app.dependency_overrides[get_data_provider] = lambda: data_provider_mock
    app.dependency_overrides[get_forecast_provider] = lambda: dam_forecast_provider_mock
    app.dependency_overrides[get_supply_forecast_provider] = lambda: supply_forecast_provider_mock

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# Tests: basic endpoints
def test_home_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {}


def test_get_data_json_ok(client):
    r = client.get("/data")
    assert r.status_code == 200
    assert r.json() == DATA_RECORDS


def test_get_data_csv_ok(client):
    r = client.get("/csv")
    assert r.status_code == 200
    assert r.text == _csv_from_records(DATA_RECORDS)
    assert r.headers["content-type"].startswith("text/csv")


# # Tests: /source/{energy_source} & /sources
# @pytest.mark.parametrize("label", ["PV", "pv", "Pv"])
# def test_get_data_by_source_valid_case_insensitive(client, label):
#     expected = [r for r in DATA_RECORDS if r["Source"].lower() == "pv"]
#     r = client.get(f"/source/{label}")
#     assert r.status_code == 200
#     assert r.json() == expected


def test_get_data_by_source_invalid_returns_error(client):
    r = client.get("/source/wind")
    assert r.status_code == 200
    body = r.json()
    assert "error" in body
    for s in {r["Source"] for r in DATA_RECORDS}:
        assert s in body["error"]


def test_get_sources_ok(client):
    r = client.get("/sources")
    assert r.status_code == 200
    expected_order = []
    seen = set()
    for item in DATA_RECORDS:
        s = item["Source"]
        if s not in seen:
            seen.add(s)
            expected_order.append(s)
    assert r.json() == {"sources": expected_order}


# Tests: /data/time-range
def test_time_range_from_after_to_validation_error(client):
    r = client.get(
        "/data/time-range",
        params={
            "from_time": "2025-08-18T20:00:00+02:00",
            "to_time": "2025-08-18T15:00:00+02:00",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"error": "from_time must not be after to_time"}


def test_time_range_invalid_source_error(client):
    r = client.get(
        "/data/time-range",
        params={
            "from_time": "2025-08-18T15:00:00+02:00",
            "to_time": "2025-08-18T20:00:00+02:00",
            "source": "wind",
        },
    )
    assert r.status_code == 422


def test_time_range_no_data_error(client):
    r = client.get(
        "/data/time-range",
        params={
            "from_time": "2025-08-18T15:00:00+02:00",
            "to_time": "2025-08-18T20:00:00+02:00",
            "source": "PV",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"error": "No data found for the given time range"}


def test_time_range_success_pv(client):
    r = client.get(
        "/data/time-range",
        params={
            "from_time": "2025-08-18T12:15:00+02:00",
            "to_time": "2025-08-18T14:45:00+02:00",
            "source": "PV",
        },
    )
    assert r.status_code == 200
    expected = [DATA_RECORDS[1], DATA_RECORDS[3]]
    assert r.json() == expected


def test_time_range_success_grid(client):
    r = client.get(
        "/data/time-range",
        params={
            "from_time": "2025-08-18T12:00:00+02:00",
            "to_time": "2025-08-18T14:45:00+02:00",
            "source": "Grid",
        },
    )
    assert r.status_code == 200
    expected = [DATA_RECORDS[0], DATA_RECORDS[2]]
    assert r.json() == expected


# Tests: /data/horizon
def test_horizon_invalid_source_error(client):
    r = client.get("/data/horizon", params={"source": "wind"})
    assert r.status_code == 422


def test_horizon_no_data_error_when_start_none(client, data_provider_mock):
    async def _no_horizon(_source):
        return {"from_time": None, "to_time": None}

    data_provider_mock.get_horizon.side_effect = _no_horizon

    r = client.get("/data/horizon", params={"source": "PV"})
    assert r.status_code == 200
    assert r.json() == {"error": "No data found for the requested source"}


def test_horizon_success(client):
    pv_times = [_parse_iso(r["Time"]) for r in DATA_RECORDS if r["Source"].lower() == "pv"]
    expected = {"from_time": min(pv_times).isoformat(), "to_time": max(pv_times).isoformat()}
    r = client.get("/data/horizon", params={"source": "PV"})
    assert r.status_code == 200
    assert r.json() == expected


# Tests: /dam-forecast & /supply-forecast
def test_dam_forecast_ok(client):
    r = client.get("/dam-forecast")
    assert r.status_code == 200
    assert r.json() == FORECAST_ROWS


def test_supply_forecast_ok(client):
    r = client.get("/supply-forecast")
    assert r.status_code == 200
    assert r.json() == SUPPLY_FORECAST_ROWS
