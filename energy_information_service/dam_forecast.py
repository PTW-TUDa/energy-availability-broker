# energy_information_service/forecast.py
"""
Provides a background-friendly **ForecastProvider** that supplies five-day,
15-minute Day-Ahead-Market (DAM) price forecasts.

Key responsibilities
--------------------
* Loads a pre-trained XGBoost regression model once at application start-up.
* Builds & caches a historical feature matrix (weather + price lags).
* Generates and stores a rolling five-day forecast (refreshed every 6 h).
* Exposes an async `get_forecast()` helper used by the `/dam-forecast` route.

The heavy work (network I/O, model inference) is executed in a worker
thread via `anyio.to_thread.run_sync`, so the async event-loop stays
responsive.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path

import anyio
import joblib
import pandas as pd
from entsoe import EntsoePandasClient
from eta_utility.connectors.entso_e import ENTSOEConnection
from eta_utility.connectors.node import NodeEntsoE

from .forecast_utils import (
    build_feature_history,
    predict_future_prices,
)
from .secret import ENTSOE_API_TOKEN

log = logging.getLogger(__name__)


class DamForecastProvider:
    """
    Thread-safe, in-memory DAM-forecast cache.

    Life-cycle
    ----------
    * **Initialisation** - loads model & prepares the ENTSO-E client.
    * **First request** - builds historical feature data and the forecast.
    * **Subsequent requests** - serves cached DataFrame as `list[dict]`.
    * **Refresh** - `_refresh_forecast()` is triggered by APScheduler
      every 6h (configurable) or on demand if the cache is stale.

    Notes
    -----
    * Uses `anyio.Lock` so async readers don't trip over concurrent refreshes.
    * All blocking CPU / I/O is pushed to a worker thread with
      `anyio.to_thread.run_sync` to keep the FastAPI event-loop happy.
    """

    MODEL_DIR = Path(__file__).with_name("models")

    VALIDATE_INTERVAL_MIN = 15  # how often to compare with latest ENTSOE data
    REFRESH_INTERVAL_H = 6  # hours after which the cache is refreshed

    def __init__(self):
        self._lock = anyio.Lock()

        # — expensive, do only once —
        self._model_path = self._select_latest_model()
        self._model = joblib.load(self._model_path)
        log.info("Loaded DAM model from %s", self._model_path)

        self._entsoe = EntsoePandasClient(api_key=ENTSOE_API_TOKEN)

        self.entsoe_node = NodeEntsoE(
            name="entsoe",
            url="https://web-api.tp.entsoe.eu/",
            protocol="entsoe",
            bidding_zone="DEU-LUX",
            endpoint="Price",
        )
        self.entsoe_connection = ENTSOEConnection.from_node(self.entsoe_node, api_token=ENTSOE_API_TOKEN)

        # Lazily populated state (at first request or a scheduled refresh)
        self._feature_history_df: pd.DataFrame | None = None
        # ^ Historical **feature matrix** (weather + lagged prices), reused
        #   on every refresh. Renamed from _history.

        self._dam_price_forecast_df: pd.DataFrame | None = None
        # ^ Cached **quarter-hourly DAM price forecast** covering five days.
        #   Renamed from _forecast.

        self.entsoe_reference_df: pd.DataFrame | None = None

        self._last_refresh: datetime | None = None
        self._last_validation: datetime | None = None

        self._validator_started = False
        self._validator_task: asyncio.Task | None = None

    @staticmethod
    def _parse_timestamp(path: Path) -> datetime | None:
        """
        Return a datetime extracted from filenames like:
            xgb_daily_model_20250721T130459Z.pkl
        If no timestamp is present, return ``None``.
        """
        import re

        m = re.search(r"_([0-9]{8}T[0-9]{4,6})Z?\.pkl$", path.name)
        if not m:
            return None
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")

    def _select_latest_model(self) -> Path:
        """
        Pick the newest model in *models/*.

        Priority
        --------
        1. Highest timestamp embedded in filename
        2. If none have a timestamp, most recently *modified* file
        3. If exactly one .pkl exists, return it
        4. Otherwise raise ``FileNotFoundError``
        """
        if not self.MODEL_DIR.exists():
            raise FileNotFoundError(f"Model directory '{self.MODEL_DIR}' does not exist")

        pkl_files = list(self.MODEL_DIR.glob("*.pkl"))
        if not pkl_files:
            raise FileNotFoundError(f"No model checkpoints found in '{self.MODEL_DIR}'")
        if len(pkl_files) == 1:
            return pkl_files[0]

        # 1) try timestamp parsing
        dated: list[tuple[Path, datetime]] = [
            (p, ts) for p in pkl_files if (ts := self._parse_timestamp(p)) is not None
        ]
        if dated:
            return max(dated, key=lambda t: t[1])[0]

        # 2) fallback: newest modification time
        return max(pkl_files, key=lambda p: p.stat().st_mtime)

    async def get_forecast(self) -> list[dict]:
        """
        Return a 5-day, 15-minute forecast window that **always begins at the
        current quarter-hour**.

        * If the cached data are older than `REFRESH_INTERVAL_H` hours, or the
          requested window extends beyond the cached horizon, a background
          refresh is triggered.
        """
        async with self._lock:
            await self._ensure_validator_started()
            tz_now = pd.Timestamp.now(tz="Europe/Berlin")

            # 1) Determine whether the cache is stale or too short
            cache_stale = (
                self._dam_price_forecast_df is None
                or self._last_refresh is None
                or (tz_now - self._last_refresh) > timedelta(hours=self.REFRESH_INTERVAL_H)
            )

            # If cache empty or stale, refresh in a worker-thread
            if cache_stale:
                await anyio.to_thread.run_sync(self._refresh_forecast)

            # 2) Slice the cached DataFrame to the *current* 5-day window
            window_start = tz_now.floor("15min")
            window_end = window_start + timedelta(days=5)

            # If the requested end extends beyond the cached horizon, refresh
            if window_end > self._dam_price_forecast_df.index.max():
                await anyio.to_thread.run_sync(self._refresh_forecast)

            slice_df = self._dam_price_forecast_df.loc[window_start:window_end]

            # Serialise to list-of-dicts with ISO-formatted strings
            return (
                slice_df.reset_index(names="Time")
                .assign(Time=lambda d: d["Time"].dt.strftime("%Y-%m-%d %H:%M:%S%z"))
                .to_dict(orient="records")
            )

    def _refresh_forecast(self):
        """Executed in a worker thread - heavy, blocking code is OK here."""
        log.info("Building / refreshing DAM forecast …")

        if self._feature_history_df is None:
            self._feature_history_df = build_feature_history(self._entsoe, earliest="2022-01-01")

        start_date = datetime.now(tz=pd.Timestamp.now().tz).strftime("%Y-%m-%d")
        end_date = (datetime.now(tz=pd.Timestamp.now().tz) + timedelta(days=5)).strftime("%Y-%m-%d")

        df_pred = predict_future_prices(
            reg=self._model,
            start_date=start_date,
            end_date=end_date,
            entsoe_client=self._entsoe,
            df_history=self._feature_history_df,
        )

        # Convert daily predictions to a continuous 15-minute series.
        price_forecast_15min = (
            df_pred[["prediction"]].resample("15min").ffill().rename(columns={"prediction": "Cost (EUR/MWh)"})
        )
        # ^ 15-minute-granularity DAM price forecast (DateTimeIndex kept).

        self._dam_price_forecast_df = price_forecast_15min
        self._last_refresh = pd.Timestamp.now(tz="Europe/Berlin")

        today = pd.Timestamp.now("Europe/Berlin").normalize()
        day_after_tomorrow = today + timedelta(days=2)
        self.entsoe_reference_df = self._fetch_entsoe_prices(today, day_after_tomorrow)

        log.info(
            "Forecast cache refreshed: %s rows (5-day) | reference updated: %s rows (48 h)",
            len(price_forecast_15min),
            len(self.entsoe_reference_df),
        )

    def _fetch_entsoe_prices(self, from_time, to_time) -> pd.DataFrame:
        """Returns a DataFrame [Time, Grid Price 1h (EUR/MWh)] from the hourly ENTSOE Day-Ahead Market prices
        for the given time frame. The DataFrame is indexed by Time and sorted in ascending order."""
        interval = timedelta(minutes=15)

        day_ahead_prices = self.entsoe_connection.read_series(from_time, to_time, self.entsoe_node, interval)

        day_ahead_prices = day_ahead_prices.reset_index()

        if day_ahead_prices.shape[1] == 2:
            day_ahead_prices.columns = ["Time", "Grid Price 1h (EUR/MWh)"]
            day_ahead_prices["Time"] = pd.to_datetime(day_ahead_prices["Time"])
        else:
            day_ahead_prices.columns = ["Time", "Grid Price 0.25h (EUR/MWh)", "Grid Price 1h (EUR/MWh)"]
            day_ahead_prices["Time"] = pd.to_datetime(day_ahead_prices["Time"])
            day_ahead_prices = day_ahead_prices.drop(columns=["Grid Price 0.25h (EUR/MWh)"])

        return day_ahead_prices.set_index("Time").sort_index().dropna()

    def _entsoe_checksum(self, entsoe_df: pd.DataFrame) -> str:
        """Return SHA-256 of ENTSOE Grid Price column for quick equality check."""
        return hashlib.sha256(entsoe_df["Grid Price 1h (EUR/MWh)"].to_numpy().tobytes()).hexdigest()

    async def _validator_loop(self):
        """Runs in the background: every VALIDATE_INTERVAL_MIN minutes it
        re-fetches the 48 h ENTSO-E prices and refreshes the cache on change."""
        while True:
            await anyio.sleep(self.VALIDATE_INTERVAL_MIN * 60)

            async with self._lock:
                if self.entsoe_reference_df is None:
                    await anyio.to_thread.run_sync(self._refresh_forecast)
                    continue

                today = pd.Timestamp.now("Europe/Berlin").normalize()
                day_after_tomorrow = today + timedelta(days=2)

                latest_entsoe = await anyio.to_thread.run_sync(self._fetch_entsoe_prices, today, day_after_tomorrow)

                cached_hash = self._entsoe_checksum(self.entsoe_reference_df)
                live_hash = self._entsoe_checksum(latest_entsoe)

                if cached_hash != live_hash:
                    log.warning("ENTSOE data changed — refreshing forecast …")
                    await anyio.to_thread.run_sync(self._refresh_forecast)
                else:
                    log.debug("ENTSOE data unchanged [%s]", cached_hash)

                self._last_validation = pd.Timestamp.now("UTC")

    async def _ensure_validator_started(self):
        """Launch the validator exactly once in the current event-loop."""
        if not self._validator_started:
            self._validator_started = True
            self._validator_task = asyncio.create_task(self._validator_loop())  # keep ref

    async def get_horizon(self) -> dict:
        """
        Return earliest/latest timestamps in the cached 5-day DAM forecast:
            {"start_time": "<ISO-8601>|None", "end_time": "<ISO-8601>|None"}.
        Refreshes cache if it's missing or older than REFRESH_INTERVAL_H hours.
        """
        async with self._lock:
            await self._ensure_validator_started()

            cache_missing = self._dam_price_forecast_df is None
            cache_stale = self._last_refresh is None or (
                pd.Timestamp.now("Europe/Berlin") - self._last_refresh
            ) > pd.Timedelta(hours=self.REFRESH_INTERVAL_H)

            if cache_missing or cache_stale:
                await anyio.to_thread.run_sync(self._refresh_forecast)

            df_dam_price = self._dam_price_forecast_df

            if df_dam_price is None or df_dam_price.empty:
                return {"start_time": None, "end_time": None}

            start_ts = df_dam_price.index.min()
            end_ts = df_dam_price.index.max()
            return {"start_time": start_ts.isoformat(), "end_time": end_ts.isoformat()}

    async def get_data_by_time_range(
        self,
        from_time: datetime | None = None,
        to_time: datetime | None = None,
    ) -> list[dict]:
        """
        Slice the cached 5-day, 15-minute DAM price forecast by time only.
        When the requested window extends the cached horizon (to_time),
        trigger a refresh to extend the cache.
        Result is a list of dicts with 'Time' as ISO string and 'Cost (EUR/MWh)'.
        """
        async with self._lock:
            await self._ensure_validator_started()

            # Ensure cache exists / is fresh enough for a typical query
            cache_missing = self._dam_price_forecast_df is None
            cache_stale = self._last_refresh is None or (
                pd.Timestamp.now("Europe/Berlin") - self._last_refresh
            ) > pd.Timedelta(hours=self.REFRESH_INTERVAL_H)
            if cache_missing or cache_stale:
                await anyio.to_thread.run_sync(self._refresh_forecast)

            df_dam_price = self._dam_price_forecast_df

            # Default to full cached horizon if bounds are omitted
            if from_time is None:
                from_time = df_dam_price.index.min().to_pydatetime()
            if to_time is None:
                to_time = df_dam_price.index.max().to_pydatetime()

            # If the requested window extends beyond the cached horizon, refresh
            if to_time > df_dam_price.index.max().to_pydatetime():
                await anyio.to_thread.run_sync(self._refresh_forecast)
                df_dam_price = self._dam_price_forecast_df

            sliced = df_dam_price.loc[pd.to_datetime(from_time) : pd.to_datetime(to_time)]

            # Serialize to list-of-dicts with consistent Time formatting
            return (
                sliced.reset_index(names="Time")
                .assign(Time=lambda d: d["Time"].dt.strftime("%Y-%m-%d %H:%M:%S%z"))
                .to_dict(orient="records")
            )
