from enum import Enum


class EnergySource(str, Enum):
    PV = "PV"
    GRID = "Grid"
    FORECAST = "Forecast"
