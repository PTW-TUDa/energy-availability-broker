from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from energy_availability_broker.demand_forecast import DemandForecast, DemandForecastModel, DemandPoint

DEFAULT_FLMP_PATH = Path(__file__).parent / "data" / "TimeReconstructedOptimizedEfdmFlmp_2025-11-13.json"
QUARTER_HOUR = timedelta(minutes=15)


@dataclass(frozen=True)
class _LoadSegment:
    start: datetime
    end: datetime
    power_kw: float


def _parse_timestamp(value: str) -> datetime:
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None:
        msg = f"FLMP timestamp must include timezone information: {value}"
        raise ValueError(msg)
    return timestamp


def _power_to_kw(power: dict[str, Any]) -> float:
    unit = power["unit"].lower()
    value = float(power["value"])

    if unit == "kw":
        return value
    if unit == "w":
        return value / 1000
    if unit == "mw":
        return value * 1000

    msg = f"Unsupported FLMP power unit: {power['unit']}"
    raise ValueError(msg)


def _floor_to_full_hour(timestamp: datetime) -> datetime:
    return timestamp.replace(minute=0, second=0, microsecond=0)


def _build_flmp_segments(flmp_payload: dict[str, Any]) -> list[_LoadSegment]:
    segments: list[_LoadSegment] = []

    for measure in flmp_payload.get("flexibleLoadMeasures", []):
        profile = sorted(measure.get("loadChangeProfile", []), key=lambda point: _parse_timestamp(point["timestamp"]))

        for current, following in zip(profile, profile[1:]):
            start = _parse_timestamp(current["timestamp"])
            end = _parse_timestamp(following["timestamp"])

            if end < start:
                msg = f"FLMP loadChangeProfile is not time-ordered for flexibleLoadId={measure.get('flexibleLoadId')}"
                raise ValueError(msg)
            if end == start:
                continue

            segments.append(_LoadSegment(start=start, end=end, power_kw=_power_to_kw(current["power"])))

    if not segments:
        raise ValueError("FLMP payload does not contain any load segments.")

    return segments


def _resample_segments_to_quarter_hour_energy(segments: list[_LoadSegment]) -> list[tuple[datetime, float]]:
    start = _floor_to_full_hour(min(segment.start for segment in segments))
    end = max(segment.end for segment in segments)

    buckets: list[tuple[datetime, float]] = []
    bucket_start = start
    while bucket_start < end:
        bucket_end = bucket_start + QUARTER_HOUR
        energy_kwh = 0.0

        for segment in segments:
            overlap_start = max(bucket_start, segment.start)
            overlap_end = min(bucket_end, segment.end)
            if overlap_end <= overlap_start:
                continue

            overlap_hours = (overlap_end - overlap_start).total_seconds() / 3600
            energy_kwh += segment.power_kw * overlap_hours

        buckets.append((bucket_start, round(energy_kwh, 6)))
        bucket_start = bucket_end

    return buckets


def parse_flmp_load_curve(flmp_payload: dict[str, Any]) -> list[DemandPoint]:
    """Convert an FLMP payload into full-hour aligned quarter-hour demand points."""

    energy_buckets = _resample_segments_to_quarter_hour_energy(_build_flmp_segments(flmp_payload))
    return [DemandPoint(time=bucket_start, energy_kwh=energy_kwh) for bucket_start, energy_kwh in energy_buckets]


def flmp_response_from_payload(
    flmp_payload: dict[str, Any],
    *,
    source: str = "production",
) -> list[dict[str, Any]]:
    """Convert an FLMP payload into quarter-hour response rows with energy values."""

    return [
        {
            "Time": bucket_start.isoformat(timespec="seconds"),
            "Energy (kWh)": energy_kwh,
            "Source": source,
        }
        for bucket_start, energy_kwh in _resample_segments_to_quarter_hour_energy(_build_flmp_segments(flmp_payload))
    ]


def demand_forecast_model_from_flmp(
    flmp_payload: dict[str, Any],
    *,
    source: str = "flmp",
) -> DemandForecastModel:
    """Create a DemandForecastModel from an FLMP payload."""

    metadata = flmp_payload.get("metadata", {})
    return DemandForecastModel(
        forecast_id=metadata.get("instanceId"),
        source=source,
        values=parse_flmp_load_curve(flmp_payload),
    )


def demand_forecast_model_from_flmp_file(
    path: str | Path = DEFAULT_FLMP_PATH,
    *,
    source: str = "flmp",
) -> DemandForecastModel:
    """Read an FLMP JSON file and convert it into a DemandForecastModel."""

    flmp_path = Path(path)
    with flmp_path.open(encoding="utf-8") as file_handle:
        flmp_payload = json.load(file_handle)

    return demand_forecast_model_from_flmp(flmp_payload, source=source)


def flmp_response_from_file(
    path: str | Path = DEFAULT_FLMP_PATH,
    *,
    source: str = "production",
) -> list[dict[str, Any]]:
    """Read an FLMP JSON file and convert it into quarter-hour response rows."""

    flmp_path = Path(path)
    with flmp_path.open(encoding="utf-8") as file_handle:
        flmp_payload = json.load(file_handle)

    return flmp_response_from_payload(flmp_payload, source=source)


async def load_flmp_file_into_demand_forecast(
    demand_forecast: DemandForecast,
    path: str | Path = DEFAULT_FLMP_PATH,
    *,
    source: str = "flmp",
) -> DemandForecastModel:
    """Parse an FLMP file, store it in DemandForecast, and return the parsed model."""

    forecast_model = demand_forecast_model_from_flmp_file(path, source=source)
    await demand_forecast.store_forecast(forecast_model)
    return forecast_model


__all__ = [
    "DEFAULT_FLMP_PATH",
    "demand_forecast_model_from_flmp",
    "demand_forecast_model_from_flmp_file",
    "flmp_response_from_file",
    "flmp_response_from_payload",
    "load_flmp_file_into_demand_forecast",
    "parse_flmp_load_curve",
]
