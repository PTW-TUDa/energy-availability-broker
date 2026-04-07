from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def fastapi_dev() -> None:
    """Run the FastAPI development server for the main app module."""
    app_path = Path(__file__).with_name("main.py")
    subprocess.run(
        [sys.executable, "-m", "fastapi", "dev", str(app_path)],
        check=True,
    )
