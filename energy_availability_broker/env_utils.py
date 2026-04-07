from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def load_service_env(*, override: bool = False) -> bool:
    """Load the service env file from stable repo/package-relative locations."""

    package_dir = Path(__file__).resolve().parent
    candidates = [
        package_dir / ".env.energy-availability-broker",
        package_dir / ".env.energy_availability_broker",
        Path.cwd() / ".env.energy-availability-broker",
        Path.cwd() / ".env.energy_availability_broker",
    ]

    loaded = False
    for candidate in candidates:
        if candidate.exists():
            loaded = load_dotenv(candidate, override=override) or loaded

    return loaded


__all__ = ["load_service_env"]
