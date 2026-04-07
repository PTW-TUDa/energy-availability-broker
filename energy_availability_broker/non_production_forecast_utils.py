from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import numpy as np
import openmeteo_requests
import pandas as pd
import requests_cache
import xgboost as xgb
from etaone_py_sdk import EtaOne
from etaone_py_sdk.models.series import SeriesModel
from etaone_py_sdk.models.tags import TagModel
from openmeteo_sdk.Variable import Variable
from retry_requests import retry

from energy_availability_broker.config import SERVICE_CONFIG

log = logging.getLogger(__name__)


WEATHER_COLS = {
    "temperature_2m": "temperature_air_mean_C",
    "relative_humidity_2m": "humidity_percent",
    "wind_speed_10m": "wind_speed_m_per_s",
    "precipitation": "precipitation_height_mm",
}

_TIME_FMT = "%Y-%m-%d %H:%M:%S%z"
QUARTER_HOUR = pd.Timedelta(minutes=15)


def make_calendar_features_one_df(ts_issue_utc: pd.Timestamp, tz_local: str) -> pd.DataFrame:
    """
    Calendar features at issue time, returned as a 1-row DataFrame indexed by ts_issue_utc (UTC floored to hour).
    """

    ts_issue_utc = _as_utc_hour(ts_issue_utc)
    local = ts_issue_utc.tz_convert(tz_local)

    hour = int(local.hour)
    dow = int(local.dayofweek)
    month = int(local.month)
    is_weekend = int(dow >= 5)

    row = {
        "hour": hour,
        "dow": dow,
        "month": month,
        "is_weekend": is_weekend,
        "hour_sin": float(np.sin(2 * np.pi * hour / 24.0)),
        "hour_cos": float(np.cos(2 * np.pi * hour / 24.0)),
        "dow_sin": float(np.sin(2 * np.pi * dow / 7.0)),
        "dow_cos": float(np.cos(2 * np.pi * dow / 7.0)),
    }
    return pd.DataFrame([row], index=pd.DatetimeIndex([ts_issue_utc], name="timestamp_utc"))


def make_openmeteo_client(
    *,
    cache_path: str = ".openmeteo_cache",
    cache_expire_s: int = 3600,
    retries: int = 5,
    backoff_factor: float = 0.2,
) -> openmeteo_requests.Client:
    """
    Creates an Open-Meteo client. If requests-cache + retry-requests are available,
    it enables caching + retry.
    """
    if requests_cache is None or retry is None:
        return openmeteo_requests.Client()

    cache_session = requests_cache.CachedSession(cache_path, expire_after=cache_expire_s)
    retry_session = retry(cache_session, retries=retries, backoff_factor=backoff_factor)
    return openmeteo_requests.Client(session=retry_session)


def _call_openmeteo(client: openmeteo_requests.Client, url: str, params: dict):
    """
    openmeteo_requests has had examples using `.weather_api()` and also `.get()`.
    This wrapper supports both.
    """
    if hasattr(client, "weather_api"):
        return client.weather_api(url, params=params)
    if hasattr(client, "get"):
        return client.get(url, params=params)
    raise AttributeError("openmeteo_requests client has neither weather_api() nor get().")


def _hourly_time_index_utc(hourly) -> pd.DatetimeIndex:
    """
    Build a tz-aware UTC DatetimeIndex from Open-Meteo Hourly() flatbuffer object.
    """
    start = pd.to_datetime(hourly.Time(), unit="s", utc=True)
    end = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True)
    step = pd.Timedelta(seconds=int(hourly.Interval()))
    # inclusive="left" -> [start, end)
    return pd.date_range(start=start, end=end, freq=step, inclusive="left")


def _find_hourly_values(hourly, *, var: Variable, altitude: int | None = None) -> np.ndarray:
    """
    Robustly locate the correct hourly variable inside the flatbuffer response.
    """
    vars_list = [hourly.Variables(i) for i in range(hourly.VariablesLength())]
    for v in vars_list:
        if v.Variable() != var:
            continue
        if altitude is not None and int(v.Altitude()) != int(altitude):
            continue
        return v.ValuesAsNumpy()
    raise KeyError(f"Hourly variable not found: {var} altitude={altitude}")


def fetch_open_meteo_hourly_utc(
    *,
    lat: float,
    lon: float,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    client: openmeteo_requests.Client,
    archive_if_older_than_hours: int = 3,
) -> pd.DataFrame:
    start_utc = _as_utc_hour(start_utc)
    end_utc = _as_utc_hour(end_utc)

    now_utc = pd.Timestamp.now(tz="UTC").floor("h")
    use_archive = end_utc <= (now_utc - pd.Timedelta(hours=archive_if_older_than_hours))

    if use_archive:
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "UTC",
            "hourly": list(WEATHER_COLS.keys()),
            "start_date": start_utc.date().isoformat(),
            "end_date": end_utc.date().isoformat(),
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }
    else:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "timezone": "UTC",
            "hourly": list(WEATHER_COLS.keys()),
            "start_hour": start_utc.strftime("%Y-%m-%dT%H:%M"),
            "end_hour": end_utc.strftime("%Y-%m-%dT%H:%M"),
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }

    responses = _call_openmeteo(client, url, params)
    if not responses:
        raise RuntimeError("Open-Meteo returned no responses.")
    response = responses[0]

    hourly = response.Hourly()
    idx = _hourly_time_index_utc(hourly)

    temp = _find_hourly_values(hourly, var=Variable.temperature, altitude=2)
    rh = _find_hourly_values(hourly, var=Variable.relative_humidity, altitude=2)
    wind = _find_hourly_values(hourly, var=Variable.wind_speed, altitude=10)
    prcp = _find_hourly_values(hourly, var=Variable.precipitation, altitude=None)

    weather_df = pd.DataFrame(index=idx)
    weather_df[WEATHER_COLS["temperature_2m"]] = temp
    weather_df[WEATHER_COLS["relative_humidity_2m"]] = rh
    weather_df[WEATHER_COLS["wind_speed_10m"]] = wind
    weather_df[WEATHER_COLS["precipitation"]] = prcp

    return weather_df.loc[start_utc:end_utc].sort_index()


def _relax_tag_fields_for_series_embeds() -> None:
    """
    Workaround for backend returning embedded tags without readOnly/numberAssigned.
    Safe to call multiple times.
    """

    # Provide defaults if those fields are missing in embedded tag dicts
    if "read_only" in TagModel.model_fields:
        TagModel.model_fields["read_only"].default = False
    if "number_assigned" in TagModel.model_fields:
        TagModel.model_fields["number_assigned"].default = 0

    TagModel.model_rebuild(force=True)
    SeriesModel.model_rebuild(force=True)


def _as_utc_hour(ts: pd.Timestamp | str | datetime) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.floor("h")


def _as_local_timestamp(ts: pd.Timestamp | str | datetime, tz_local: str) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize(tz_local) if ts.tzinfo is None else ts.tz_convert(tz_local)


def resolve_issue_time_local(
    *,
    from_time: pd.Timestamp | str | datetime | None,
    tz_local: str,
) -> pd.Timestamp:
    """Resolve the local issue time used as the forecast inference anchor."""

    now_local = pd.Timestamp.now(tz=tz_local).floor("h")
    if from_time not in (None, ""):
        requested_local = _as_local_timestamp(from_time, tz_local).floor("h")
        return min(requested_local, now_local)
    return now_local


def filter_non_production_forecast_window(
    forecast_df: pd.DataFrame,
    *,
    from_time: pd.Timestamp | str | datetime | None,
    to_time: pd.Timestamp | str | datetime | None,
    tz_local: str,
) -> pd.DataFrame:
    """Filter a non-production forecast dataframe on local forecast timestamps."""

    filtered = forecast_df.copy()

    if from_time is not None:
        from_ts = _as_local_timestamp(from_time, tz_local)
        filtered = filtered.loc[filtered["ts_forecast_local"] >= from_ts]
    if to_time is not None:
        to_ts = _as_local_timestamp(to_time, tz_local)
        filtered = filtered.loc[filtered["ts_forecast_local"] <= to_ts]

    return filtered


def resample_non_production_power_to_quarter_hour_energy(
    forecast_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert hourly power predictions into quarter-hour energy rows."""

    rows: list[dict] = []
    for forecast in forecast_df.itertuples(index=False):
        energy_kwh = float(forecast.y_pred_kw) * 0.25
        start_local = pd.Timestamp(forecast.ts_forecast_local) - pd.Timedelta(hours=1)
        start_utc = pd.Timestamp(forecast.ts_forecast_utc) - pd.Timedelta(hours=1)

        for offset in range(4):
            quarter_start_local = start_local + (offset * QUARTER_HOUR)
            quarter_start_utc = start_utc + (offset * QUARTER_HOUR)
            rows.append(
                {
                    "ts_issue_utc": forecast.ts_issue_utc,
                    "horizon_h": forecast.horizon_h,
                    "ts_forecast_utc": quarter_start_utc,
                    "ts_forecast_local": quarter_start_local,
                    "non_production_energy_kwh": round(energy_kwh, 6),
                }
            )

    return pd.DataFrame(rows)


def _fetch_points_in_chunks(
    platform,
    *,
    series_id: str,
    start_utc: pd.Timestamp,
    end_utc: pd.Timestamp,
    chunk_hours: int = 48,
) -> pd.Series:
    """
    Fetch points in smaller windows to avoid backend limits.
    """
    out = []
    t0 = start_utc
    step = pd.Timedelta(hours=chunk_hours)

    while t0 < end_utc:
        t1 = min(t0 + step, end_utc)
        s = platform.series.points.get(parent_id=series_id, start=_to_dt_utc(t0), end=_to_dt_utc(t1))
        if s is not None and len(s) > 0:
            out.append(s)
        t0 = t1

    if not out:
        return pd.Series(dtype="float64")

    s_all = pd.concat(out).sort_index()
    # drop duplicate timestamps if any
    return s_all[~s_all.index.duplicated(keep="last")]


def _to_dt_utc(x: pd.Timestamp) -> datetime:
    x = pd.Timestamp(x)
    x = x.tz_localize("UTC") if x.tzinfo is None else x.tz_convert("UTC")
    return x.to_pydatetime()


def build_electrical_power_features_for_inference(
    platform,
    *,
    series_id: str,
    ts_issue_utc: pd.Timestamp | str,
    net_col: str = "Elec_P_LVMDB_ETA_kW",
    lags: Iterable[int] = (*range(1, 49), 168),
    roll_windows: Iterable[int] = (6, 24, 168),
    strict: bool = False,
    apply_tag_patch: bool = True,
) -> pd.DataFrame:
    """
    Build the electrical (net power) lag/rolling features used by the trained model.

    - Queries etaONE points for [ts_issue - history, ts_issue]
    - Converts W -> kW
    - Aggregates to hourly mean if needed
    - Produces ONE row at ts_issue_utc (floored to hour) with:
        net_col,
        f"{net_col}_lag{L}" for L in lags,
        f"{net_col}_rollmean{W}" and f"{net_col}_rollstd{W}" for W in roll_windows
    """
    if apply_tag_patch:
        _relax_tag_fields_for_series_embeds()

    ts_issue = _as_utc_hour(ts_issue_utc)

    lags = sorted({int(x) for x in lags})
    roll_windows = sorted({int(x) for x in roll_windows})

    max_lag = max(lags) if lags else 0
    max_roll = max(roll_windows) if roll_windows else 1

    # Need enough hours to compute lag168 and roll168 at ts_issue
    # (roll window W needs W values including current)
    history_hours = max(max_lag, max_roll) + 2

    start_utc = ts_issue - pd.Timedelta(hours=history_hours)
    end_utc = ts_issue + pd.Timedelta(hours=1)

    # 1) Fetch points
    try:
        points = platform.series.points.get(parent_id=series_id, start=_to_dt_utc(start_utc), end=_to_dt_utc(end_utc))
    except Exception:
        # fallback: chunk fetch
        points = _fetch_points_in_chunks(platform, series_id=series_id, start_utc=start_utc, end_utc=end_utc)

    if points is None or len(points) == 0:
        raise ValueError(f"No points returned for series_id={series_id} in [{start_utc}, {end_utc}].")

    # Ensure tz-aware UTC index
    s = pd.Series(points).copy()
    s.index = pd.to_datetime(s.index, utc=True)

    # 2) W -> kW
    s_kw = pd.to_numeric(s, errors="coerce") / 1000.0

    # 3) Aggregate to hourly mean
    hourly = s_kw.resample("1H").mean().sort_index()

    # 4) Restrict to the exact window we need and ensure the ts_issue row exists
    hourly = hourly.loc[start_utc.floor("h") : ts_issue]

    # Create an expected hourly grid to detect gaps
    expected_idx = pd.date_range(start=hourly.index.min(), end=ts_issue, freq="1H", tz="UTC")
    hourly = hourly.reindex(expected_idx)

    if strict:
        # We need these timestamps to be present for features at ts_issue:
        need_times = (
            [ts_issue]
            + [ts_issue - pd.Timedelta(hours=lag_hours) for lag_hours in lags]
            + [ts_issue - pd.Timedelta(hours=(max_roll - 1))]
        )
        missing = [t for t in need_times if (t not in hourly.index) or pd.isna(hourly.loc[t])]
        if missing:
            raise ValueError(
                "Not enough hourly data to compute all lag/rolling features at ts_issue.\n"
                f"ts_issue={ts_issue}\n"
                f"Example missing timestamps (UTC): {missing[:10]}"
            )
    elif hourly.notna().any():
        # Small gaps in the measured series should not invalidate long rolling windows.
        # Interpolate on the hourly grid so sparse missing points do not propagate into
        # the lag/rolling features, while still failing later if the history is absent.
        hourly = hourly.interpolate(method="time", limit_direction="both")

    # 5) Build features exactly like training logic (shift + rolling)
    features_df = pd.DataFrame({net_col: hourly})
    for lag_hours in lags:
        features_df[f"{net_col}_lag{lag_hours}"] = features_df[net_col].shift(lag_hours)

    for window_hours in roll_windows:
        features_df[f"{net_col}_rollmean{window_hours}"] = (
            features_df[net_col].rolling(window_hours, min_periods=window_hours).mean()
        )
        features_df[f"{net_col}_rollstd{window_hours}"] = (
            features_df[net_col].rolling(window_hours, min_periods=window_hours).std()
        )

    return features_df.loc[[ts_issue]].copy()


def build_inference_feature_vector(
    *,
    platform,
    ts_issue_utc: pd.Timestamp,
    tz_local: str,
    lvmdb_series_id: str,
    lat: float,
    lon: float,
    feature_columns_json_path: str,
    openmeteo_client: openmeteo_requests.Client,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns:
      - x_series: ordered feature vector (index = feature names) exactly as feature_columns.json expects
      - x_row_df: 1-row DataFrame (same columns) indexed by timestamp_utc
    """
    ts_issue_utc = _as_utc_hour(ts_issue_utc)

    payload = json.loads(Path(feature_columns_json_path).read_text(encoding="utf-8"))
    feats = payload["features"]

    # electrical (net + lags + rolls) features at issue time
    elec_df = build_electrical_power_features_for_inference(
        platform,
        series_id=lvmdb_series_id,
        ts_issue_utc=ts_issue_utc,
        net_col="Elec_P_LVMDB_ETA_kW",
    )

    # calendar features at issue time
    cal_df = make_calendar_features_one_df(ts_issue_utc, tz_local=tz_local)

    # weather features at issue time
    weather_df = fetch_open_meteo_hourly_utc(
        lat=lat,
        lon=lon,
        start_utc=ts_issue_utc,
        end_utc=ts_issue_utc,
        client=openmeteo_client,
    )
    # keep only the issue hour row
    weather_row = weather_df.loc[[ts_issue_utc]].copy()

    # combine
    x = pd.concat([elec_df, cal_df, weather_row], axis=1)

    # reorder and validate
    missing = [c for c in feats if c not in x.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    x = x[feats].astype(float)

    if x.isna().any(axis=None):
        bad = x.columns[x.isna().iloc[0]].tolist()
        raise ValueError(f"NaNs present in inference features: {bad}")

    x_series = x.iloc[0].copy()
    return x_series, x


def _predict_with_best(booster: xgb.Booster, dmat: xgb.DMatrix) -> np.ndarray:
    try:
        if getattr(booster, "best_iteration", None) is not None:
            return booster.predict(dmat, iteration_range=(0, booster.best_iteration + 1))
    except Exception:
        pass
    try:
        best_ntree_limit = getattr(booster, "best_ntree_limit", None)
        if best_ntree_limit is not None and best_ntree_limit > 0:
            return booster.predict(dmat, ntree_limit=best_ntree_limit)
    except Exception:
        pass
    return booster.predict(dmat)


def predict_out_of_production_power_48h(
    *,
    platform,
    ts_issue_utc: pd.Timestamp,
    tz_local: str,
    lvmdb_series_id: str,
    lat: float,
    lon: float,
    feature_columns_json_path: str,
    models_dir: str,
    openmeteo_client: openmeteo_requests.Client,
) -> pd.DataFrame:
    """
    Builds features at ts_issue_utc and predicts out-of-production power (kW)
    for horizons 1..48 using xgb_h{hh}.json models.
    Returns a DataFrame with forecast timestamps.
    """

    ts_issue_utc = _as_utc_hour(ts_issue_utc)

    payload = json.loads(Path(feature_columns_json_path).read_text(encoding="utf-8"))
    feats = payload["features"]

    x_series, _ = build_inference_feature_vector(
        platform=platform,
        ts_issue_utc=ts_issue_utc,
        tz_local=tz_local,
        lvmdb_series_id=lvmdb_series_id,
        lat=lat,
        lon=lon,
        feature_columns_json_path=feature_columns_json_path,
        openmeteo_client=openmeteo_client,
    )

    x_np = x_series.to_numpy(dtype=float).reshape(1, -1)
    dmat = xgb.DMatrix(x_np, feature_names=feats)

    models_path = Path(models_dir)
    preds = []
    for h in range(1, 49):
        model_file = models_path / f"xgb_h{h:02d}.json"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing model file: {model_file}")

        booster = xgb.Booster()
        booster.load_model(model_file.as_posix())

        yhat = float(_predict_with_best(booster, dmat)[0])
        preds.append({"horizon_h": h, "y_pred_kw": yhat})

    out = pd.DataFrame(preds)
    out["ts_issue_utc"] = ts_issue_utc
    out["ts_forecast_utc"] = ts_issue_utc + pd.to_timedelta(out["horizon_h"], unit="h")
    out["ts_forecast_local"] = out["ts_forecast_utc"].dt.tz_convert(tz_local)
    return out[["ts_issue_utc", "horizon_h", "ts_forecast_utc", "ts_forecast_local", "y_pred_kw"]]


def run_non_prod_forecast_from_env(
    from_time: str | None = None,
    to_time: str | None = None,
) -> list[dict]:
    """
    Runs non_production forecast using ONLY environment variables.
    Intended to be called from a subprocess to avoid Trio/Uvicorn collisions.
    """
    non_production_cfg = SERVICE_CONFIG.non_production
    tz_local = non_production_cfg.tz_local
    series_id = non_production_cfg.lvmdb_series_id
    lat = non_production_cfg.latitude
    lon = non_production_cfg.longitude
    feature_columns_json = str(non_production_cfg.feature_columns_json)
    models_dir = str(non_production_cfg.models_dir)

    ts_local = resolve_issue_time_local(from_time=from_time, tz_local=tz_local)
    ts_issue_utc = ts_local.tz_convert("UTC")

    openmeteo_client = make_openmeteo_client()

    log.info("Running non-production forecast with ts_issue_utc=%s, tz_local=%s", ts_issue_utc, tz_local)

    with EtaOne(_env_file=".env.etaone") as platform:
        forecast_df = predict_out_of_production_power_48h(
            platform=platform,
            ts_issue_utc=ts_issue_utc,
            tz_local=tz_local,
            lvmdb_series_id=series_id,
            lat=lat,
            lon=lon,
            feature_columns_json_path=feature_columns_json,
            models_dir=models_dir,
            openmeteo_client=openmeteo_client,
        )

    forecast_df = resample_non_production_power_to_quarter_hour_energy(forecast_df)
    forecast_df = filter_non_production_forecast_window(
        forecast_df,
        from_time=from_time,
        to_time=to_time,
        tz_local=tz_local,
    ).copy()
    forecast_df["Time"] = forecast_df["ts_forecast_local"].dt.strftime(_TIME_FMT)

    return forecast_df[["Time", "non_production_energy_kwh"]].to_dict(orient="records")
