from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar

from .config import SERVICE_CONFIG
from .demand_forecast import DemandForecastModel, DemandPoint
from .env_utils import load_service_env
from .non_production_forecast_utils import (
    make_openmeteo_client,
)

log = logging.getLogger(__name__)

load_service_env(override=False)


@dataclass(frozen=True)
class NonProductionConfig:
    feature_columns_json: Path
    models_dir: Path


def load_non_production_config_from_env() -> NonProductionConfig:
    feature_columns_json = SERVICE_CONFIG.non_production.feature_columns_json.resolve()
    models_dir = SERVICE_CONFIG.non_production.models_dir.resolve()

    return NonProductionConfig(
        feature_columns_json=feature_columns_json,
        models_dir=models_dir,
    )


class NonProductionPowerForecastProvider:
    """
    On-demand (no caching of predictions).
    Loads config at startup; does inference at request time.
    """

    TIME_FMT = "%Y-%m-%d %H:%M:%S%z"
    _SUBPROCESS_ENV_BLACKLIST: ClassVar[set[str]] = {
        "entsoe_api_token",
        "ENTSOE_API_TOKEN",
        "deploy_token",
        "DEPLOY_TOKEN",
    }

    def __init__(self) -> None:
        self.cfg = load_non_production_config_from_env()

        self._openmeteo_client = make_openmeteo_client()

        if not self.cfg.feature_columns_json.exists():
            raise FileNotFoundError(f"feature_columns.json not found: {self.cfg.feature_columns_json}")
        if not self.cfg.models_dir.exists():
            raise FileNotFoundError(f"models dir not found: {self.cfg.models_dir}")

    @staticmethod
    def _time_to_str(value: datetime | str | None) -> str | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        return str(value)

    def get_forecast(
        self,
        from_time: datetime | str | None = None,
        to_time: datetime | str | None = None,
    ) -> list[dict]:
        from_str = self._time_to_str(from_time)
        to_str = self._time_to_str(to_time)

        # Pass request timestamps via env vars to avoid quoting issues.
        env = os.environ.copy()
        for variable_name in self._SUBPROCESS_ENV_BLACKLIST:
            env.pop(variable_name, None)
        if from_str:
            env["NONPRODUCTION_FROM_TIME"] = from_str
        else:
            env.pop("NONPRODUCTION_FROM_TIME", None)
        if to_str:
            env["NONPRODUCTION_TO_TIME"] = to_str
        else:
            env.pop("NONPRODUCTION_TO_TIME", None)

        code = (
            "import os, json; "
            "from energy_information_service.non_production_forecast_utils import run_non_prod_forecast_from_env;"
            "ft = os.environ.get('NONPRODUCTION_FROM_TIME'); "
            "tt = os.environ.get('NONPRODUCTION_TO_TIME'); "
            "print(json.dumps(run_non_prod_forecast_from_env(ft, tt)))"
        )

        cmd = [sys.executable, "-c", code]

        log.info(
            "non_production: launching subprocess (from_time=%s, to_time=%s)",
            from_str,
            to_str,
        )
        p = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )

        if p.returncode != 0:
            # stderr contains the traceback
            err = (p.stderr or p.stdout or "").strip()
            raise RuntimeError(f"non_production subprocess failed: {err}")

        out = p.stdout.strip()
        if not out:
            raise RuntimeError("non_production subprocess returned empty output")

        return json.loads(out)

    def get_demand_forecast_model(
        self,
        from_time: datetime | str | None = None,
        to_time: datetime | str | None = None,
    ) -> DemandForecastModel:
        records = self.get_forecast(from_time, to_time)
        return DemandForecastModel(
            source="site",
            values=[
                DemandPoint(
                    time=datetime.strptime(record["Time"], self.TIME_FMT),
                    energy_kwh=float(record["non_production_energy_kwh"]),
                )
                for record in records
            ],
        )
