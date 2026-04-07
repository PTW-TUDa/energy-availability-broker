# test/test_main.py
# Tests for energy_availability_broker.main with sample data blocks.

from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

import energy_availability_broker.main as api_main
from energy_availability_broker.main import (
    app,
    get_data_provider,
    get_demand_forecast_provider,
    get_forecast_provider,
    get_non_production_forecast_provider,
    get_production_demand_forecast,
    get_supply_forecast_provider,
)

# Records used by /data, /csv, /source/{...}, /sources, /data/time-range, /data/horizon
DATA_RECORDS: list[dict[str, Any]] = [
    {"Time": "2025-08-18T12:00:00+02:00", "Energy (kWh)": 37.25, "Cost (EUR/kWh)": 0.02004, "Source": "Grid"},
    {"Time": "2025-08-18T12:15:00+02:00", "Energy (kWh)": 16.269, "Cost (EUR/kWh)": 0, "Source": "PV forecast"},
    {"Time": "2025-08-18T14:30:00+02:00", "Energy (kWh)": 37.25, "Cost (EUR/kWh)": 0.01834, "Source": "Grid"},
    {"Time": "2025-08-18T14:45:00+02:00", "Energy (kWh)": 15.3425, "Cost (EUR/kWh)": 0, "Source": "PV forecast"},
]

DEMAND_ROWS: list[dict[str, Any]] = [
    {"Time": "2025-08-18T12:00:00+02:00", "Energy (kWh)": 5.0, "Source": "production"},
    {"Time": "2025-08-18T12:15:00+02:00", "Energy (kWh)": 3.0, "Source": "site"},
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
    {"Time": "2025-08-19 12:00:00+0200", "Energy (kWh)": 13.426, "Cost (EUR/kWh)": 0, "Source": "PV forecast"},
    {
        "Time": "2025-08-19 12:00:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.395589630126953,
        "Source": "Grid forecast",
    },
    {"Time": "2025-08-19 12:15:00+0200", "Energy (kWh)": 13.6015, "Cost (EUR/kWh)": 0, "Source": "PV forecast"},
    {
        "Time": "2025-08-19 12:15:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.395589630126953,
        "Source": "Grid forecast",
    },
    {"Time": "2025-08-19 12:30:00+0200", "Energy (kWh)": 13.70075, "Cost (EUR/kWh)": 0, "Source": "PV forecast"},
    {
        "Time": "2025-08-19 12:30:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.395589630126953,
        "Source": "Grid forecast",
    },
    {"Time": "2025-08-19 12:45:00+0200", "Energy (kWh)": 13.682, "Cost (EUR/kWh)": 0, "Source": "PV forecast"},
    {
        "Time": "2025-08-19 12:45:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.395589630126953,
        "Source": "Grid forecast",
    },
    {"Time": "2025-08-19 13:00:00+0200", "Energy (kWh)": 13.7635, "Cost (EUR/kWh)": 0, "Source": "PV forecast"},
    {
        "Time": "2025-08-19 13:00:00+0200",
        "Energy (kWh)": 37.25,
        "Cost (EUR/kWh)": 0.409508483886719,
        "Source": "Grid forecast",
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
@pytest.fixture
def data_provider_mock():
    m = AsyncMock(name="EnergyAvailabilityProviderMock")

    # /data
    async def get_data():
        return MockDF(DATA_RECORDS)

    # /source/{energy_source}
    async def get_data_by_source(src: str):
        # support enum input (FastAPI may pass an Enum instance)
        src_val = src.value if hasattr(src, "value") else src
        # map API source labels back to record labels when needed
        if src_val and src_val.lower().startswith("pv"):
            record_label = "PV forecast"
        elif src_val and src_val.lower().startswith("grid"):
            record_label = "Grid"
        else:
            record_label = src_val
        filtered = [r for r in DATA_RECORDS if str(r.get("Source", "")).lower() == str(record_label).lower()]
        return MockDF(filtered)

    # /sources
    async def get_sources():
        # return labels as expected by the API's EnergySource enum
        seen, result = set(), []
        for r in DATA_RECORDS:
            s = r.get("Source")
            if s is None or s in seen:
                continue
            seen.add(s)
            if "pv" in str(s).lower():
                result.append("PV forecast")
            else:
                result.append(s)
        return result

    # /data/time-range
    async def get_data_by_time_range(from_time, to_time, source: str | None):
        rows = DATA_RECORDS
        if source is not None:
            src_val = source.value if hasattr(source, "value") else source
            # map API label to record label
            if src_val and src_val.lower().startswith("pv"):
                record_label = "PV forecast"
            elif src_val and src_val.lower().startswith("grid"):
                record_label = "Grid"
            else:
                record_label = src_val
            rows = [r for r in rows if str(r.get("Source", "")).lower() == str(record_label).lower()]
        if from_time is not None:
            rows = [r for r in rows if _parse_iso(r["Time"]) >= from_time]
        if to_time is not None:
            rows = [r for r in rows if _parse_iso(r["Time"]) <= to_time]
        return MockDF(rows)

    # /data/horizon
    async def get_horizon(source: str | None):
        rows = DATA_RECORDS
        if source is not None:
            src_val = source.value if hasattr(source, "value") else source
            if src_val and src_val.lower().startswith("pv"):
                record_label = "PV forecast"
            elif src_val and src_val.lower().startswith("grid"):
                record_label = "Grid"
            else:
                record_label = src_val
            rows = [r for r in rows if str(r.get("Source", "")).lower() == str(record_label).lower()]
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


@pytest.fixture
def dam_forecast_provider_mock():
    m = AsyncMock(name="DamForecastProviderMock")
    # ensure methods used by main.py are available
    m.get_forecast.return_value = FORECAST_ROWS
    m.get_data_by_time_range.return_value = FORECAST_ROWS
    return m


@pytest.fixture
def supply_forecast_provider_mock():
    m = AsyncMock(name="SupplyForecastProviderMock")
    m.get_supply_forecast.return_value = SUPPLY_FORECAST_ROWS

    async def get_data_by_time_range(from_time, to_time, source=None):
        rows = SUPPLY_FORECAST_ROWS
        if source is not None:
            src_val = source.value if hasattr(source, "value") else source
            rows = [r for r in rows if str(r["Source"]).lower() == str(src_val).lower()]
        if from_time is not None:
            rows = [r for r in rows if _parse_iso(r["Time"]) >= from_time]
        if to_time is not None:
            rows = [r for r in rows if _parse_iso(r["Time"]) <= to_time]
        return rows

    m.get_data_by_time_range.side_effect = get_data_by_time_range
    return m


@pytest.fixture
def demand_forecast_provider_mock():
    m = AsyncMock(name="DemandForecastProviderMock")
    m.get_data.return_value = []

    async def get_model_data(forecast, from_time=None, to_time=None):
        source = getattr(forecast, "source", "")
        rows = [r for r in DEMAND_ROWS if r["Source"].lower() == str(source).lower()]
        if from_time is not None:
            rows = [r for r in rows if _parse_iso(r["Time"]) >= from_time]
        if to_time is not None:
            rows = [r for r in rows if _parse_iso(r["Time"]) <= to_time]
        return rows

    m.get_model_data.side_effect = get_model_data
    return m


@pytest.fixture
def client(
    monkeypatch,
    data_provider_mock,
    dam_forecast_provider_mock,
    supply_forecast_provider_mock,
    demand_forecast_provider_mock,
):
    monkeypatch.setattr(api_main, "AsyncScheduler", DummyAsyncScheduler, raising=True)
    monkeypatch.setattr(
        api_main,
        "_resolve_time_range",
        lambda from_time, to_time: (
            from_time or datetime.fromisoformat("2025-08-18T12:00:00+02:00"),
            to_time or datetime.fromisoformat("2025-08-20T00:00:00+02:00"),
        ),
        raising=True,
    )

    # dependency overrides
    app.dependency_overrides[get_data_provider] = lambda: data_provider_mock
    app.dependency_overrides[get_forecast_provider] = lambda: dam_forecast_provider_mock
    app.dependency_overrides[get_supply_forecast_provider] = lambda: supply_forecast_provider_mock
    app.dependency_overrides[get_demand_forecast_provider] = lambda: demand_forecast_provider_mock
    app.dependency_overrides[get_production_demand_forecast] = lambda: SimpleNamespace(source="production")
    app.dependency_overrides[get_non_production_forecast_provider] = lambda: SimpleNamespace(
        get_demand_forecast_model=lambda *_args, **_kwargs: SimpleNamespace(source="site")
    )

    with TestClient(app) as c:
        yield c

    app.dependency_overrides.clear()


# Tests: basic endpoints
def test_home_ok(client):
    r = client.get("/")
    assert r.status_code == 200
    # root redirects to Swagger UI (HTML)
    assert r.headers["content-type"].startswith("text/html")


def test_get_data_json_ok(client):
    r = client.get("/energy")
    assert r.status_code == 200
    expected = sorted(
        [*DATA_RECORDS, *DEMAND_ROWS, *SUPPLY_FORECAST_ROWS[1::2]],
        key=lambda row: (_parse_iso(row["Time"]), str(row["Source"]).lower()),
    )
    expected = [{**row, "Time": _parse_iso(row["Time"]).isoformat(timespec="seconds")} for row in expected]
    assert r.json() == expected


# # Tests: /source/{energy_source} & /sources
# @pytest.mark.parametrize("label", ["PV", "pv", "Pv"])
# def test_get_data_by_source_valid_case_insensitive(client, label):
#     expected = [r for r in DATA_RECORDS if r["Source"].lower() == "pv"]
#     r = client.get(f"/source/{label}")
#     assert r.status_code == 200
#     assert r.json() == expected


def test_get_data_by_source_invalid_returns_error(client):
    r = client.get("/energy", params={"source": "wind"})
    assert r.status_code == 422


def test_get_sources_ok(client):
    r = client.get("/energy/sources")
    assert r.status_code == 200
    expected_order = []
    seen = set()
    for item in sorted(
        [*DATA_RECORDS, *DEMAND_ROWS, *SUPPLY_FORECAST_ROWS[1::2]],
        key=lambda row: (_parse_iso(row["Time"]), str(row["Source"]).lower()),
    ):
        s = item["Source"]
        if s not in seen:
            seen.add(s)
            expected_order.append(s)
    assert r.json() == {"sources": expected_order}


# Tests: /data/time-range
def test_time_range_from_after_to_validation_error(client):
    r = client.get(
        "/energy",
        params={
            "from_time": "2025-08-18T20:00:00+02:00",
            "to_time": "2025-08-18T15:00:00+02:00",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"error": "from_time must not be after to_time"}


def test_time_range_invalid_source_error(client):
    r = client.get(
        "/energy",
        params={
            "from_time": "2025-08-18T15:00:00+02:00",
            "to_time": "2025-08-18T20:00:00+02:00",
            "source": "wind",
        },
    )
    assert r.status_code == 422


def test_time_range_no_data_error(client):
    r = client.get(
        "/energy",
        params={
            "from_time": "2025-08-18T15:00:00+02:00",
            "to_time": "2025-08-18T20:00:00+02:00",
            "source": "PV forecast",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"error": "No data found for the given time range"}


def test_time_range_success_pv(client):
    r = client.get(
        "/energy",
        params={
            "from_time": "2025-08-18T12:15:00+02:00",
            "to_time": "2025-08-18T14:45:00+02:00",
            "source": "PV forecast",
        },
    )
    assert r.status_code == 200
    expected = [DATA_RECORDS[1], DATA_RECORDS[3]]
    assert r.json() == expected


def test_time_range_success_grid(client):
    r = client.get(
        "/energy",
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
    r = client.get("/energy/maximum-horizon", params={"source": "wind"})
    assert r.status_code == 422


def test_horizon_no_data_error_when_start_none(client, data_provider_mock):
    async def _no_horizon(_source):
        return {"from_time": None, "to_time": None}

    data_provider_mock.get_horizon.side_effect = _no_horizon

    r = client.get("/energy/maximum-horizon", params={"source": "PV forecast"})
    assert r.status_code == 200
    assert r.json() == {"error": "No data found for the requested source"}


def test_horizon_success(client):
    pv_times = [_parse_iso(r["Time"]) for r in DATA_RECORDS if "pv" in r["Source"].lower()]
    expected = {"from_time": min(pv_times).isoformat(), "to_time": max(pv_times).isoformat()}
    r = client.get("/energy/maximum-horizon", params={"source": "PV forecast"})
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
