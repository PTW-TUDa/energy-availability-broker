from enum import Enum


class EnergySource(str, Enum):
    PV = "PV forecast"
    GRID = "Grid"
    FORECAST = "Grid forecast"
