from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .non_production_forecast_utils import (
    make_openmeteo_client,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NonProductionConfig:
    feature_columns_json: Path
    models_dir: Path


def load_non_production_config_from_env() -> NonProductionConfig:
    feature_columns_json = Path(
        os.getenv(
            "NONPRODUCTION_FEATURES_JSON",
            "energy_information_service/models/non_production_forecast_48/feature_columns.json",
        )
    ).resolve()

    models_dir = Path(
        os.getenv(
            "NONPRODUCTION_MODELS_DIR",
            "energy_information_service/models/non_production_forecast_48",
        )
    ).resolve()

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

    def __init__(self) -> None:
        self.cfg = load_non_production_config_from_env()

        self._openmeteo_client = make_openmeteo_client()

        if not self.cfg.feature_columns_json.exists():
            raise FileNotFoundError(f"feature_columns.json not found: {self.cfg.feature_columns_json}")
        if not self.cfg.models_dir.exists():
            raise FileNotFoundError(f"models dir not found: {self.cfg.models_dir}")

    @staticmethod
    def _issue_time_to_str(issue_time: datetime | str | None) -> str | None:
        if issue_time is None:
            return None
        if isinstance(issue_time, datetime):
            # preserve timezone if provided
            return issue_time.isoformat()
        return str(issue_time)

    def get_forecast(self, issue_time: datetime | str | None) -> list[dict]:
        issue_str = self._issue_time_to_str(issue_time)

        # Pass issue time via env var to avoid quoting issues
        env = os.environ.copy()
        if issue_str:
            env["NONPRODUCTION_ISSUE_TIME"] = issue_str
        else:
            env.pop("NONPRODUCTION_ISSUE_TIME", None)

        code = (
            "import os, json; "
            "from energy_information_service.non_production_forecast_utils import run_non_prod_forecast_from_env;"
            "it = os.environ.get('NONPRODUCTION_ISSUE_TIME'); "
            "print(json.dumps(run_non_prod_forecast_from_env(it)))"
        )

        cmd = [sys.executable, "-c", code]

        log.info("non_production: launching subprocess (issue_time=%s)", issue_str or "now")
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
