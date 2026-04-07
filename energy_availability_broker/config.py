from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForecastSolarConfig:
    url: str = "https://api.forecast.solar"
    latitude: float = 49.86381
    longitude: float = 8.68105
    declinations: Sequence[int] = (14, 14, 10, 10)
    azimuths: Sequence[int] = (90, -90, 90, -90)
    kwps: Sequence[float] = (23.31, 23.31, 23.31, 23.31)
    data: str = "watts"


@dataclass(frozen=True)
class EntsoeConfig:
    url: str = "https://web-api.tp.entsoe.eu/"
    bidding_zone: str = "DEU-LUX"
    endpoint: str = "Price"


@dataclass(frozen=True)
class NonProductionConfig:
    tz_local: str
    lvmdb_series_id: str
    latitude: float
    longitude: float
    feature_columns_json: Path
    models_dir: Path


@dataclass(frozen=True)
class DemandForecastConfig:
    path: str = "data/demand_forecast.json"


@dataclass(frozen=True)
class ServiceConfig:
    demand_forecast: DemandForecastConfig
    forecast_solar: ForecastSolarConfig
    entsoe: EntsoeConfig
    non_production: NonProductionConfig
    prediction_horizon_hours: int = 60


MODELS_DIR = Path(__file__).with_name("models")

SERVICE_CONFIG = ServiceConfig(
    demand_forecast=DemandForecastConfig(),
    forecast_solar=ForecastSolarConfig(),
    entsoe=EntsoeConfig(),
    non_production=NonProductionConfig(
        tz_local="Europe/Berlin",
        lvmdb_series_id="0199dd98-b752-73bd-87e2-6c565287bf41",
        latitude=49.86367404764316,
        longitude=8.681025665360762,
        feature_columns_json=MODELS_DIR / "non_production_forecast_48" / "feature_columns.json",
        models_dir=MODELS_DIR / "non_production_forecast_48",
    ),
)


__all__ = ["SERVICE_CONFIG", "ServiceConfig"]
