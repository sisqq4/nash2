from __future__ import annotations

import math
from typing import Optional

import numpy as np

G0 = 9.80665
STANDARD_GRAVITY_MPS2 = G0
AIR_GAS_CONSTANT_JPKGK = 287.05287
AIR_HEAT_CAPACITY_RATIO = 1.4
SEA_LEVEL_TEMPERATURE_K = 288.15
SEA_LEVEL_PRESSURE_PA = 101325.0
EPS = 1.0e-9
NORTH_AXIS = 0
UP_AXIS = 1
EAST_AXIS = 2


def norm(x: np.ndarray) -> float:
    return float(np.linalg.norm(x))


def unit(x: np.ndarray, fallback: Optional[np.ndarray] = None) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    n = norm(value)
    if n > EPS:
        return value / n
    if fallback is None:
        return np.zeros_like(value)
    return np.asarray(fallback, dtype=np.float64).copy()


def clip_norm(x: np.ndarray, max_norm: float) -> np.ndarray:
    value = np.asarray(x, dtype=np.float64)
    n = norm(value)
    if n <= max_norm or n <= EPS:
        return value
    return value * (max_norm / n)


_ATMOSPHERE_BASE_ALTITUDES_M = (0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0)
_ATMOSPHERE_LAPSE_RATES_KPM = (-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020)


def standard_atmosphere(altitude_m: float) -> tuple[float, float, float, float]:
    """Return temperature, pressure, density and sound speed for ISA layers."""
    altitude = float(np.clip(altitude_m, 0.0, _ATMOSPHERE_BASE_ALTITUDES_M[-1]))
    base_temperature = SEA_LEVEL_TEMPERATURE_K
    base_pressure = SEA_LEVEL_PRESSURE_PA
    for index, lapse_rate in enumerate(_ATMOSPHERE_LAPSE_RATES_KPM):
        base_altitude = _ATMOSPHERE_BASE_ALTITUDES_M[index]
        layer_top = _ATMOSPHERE_BASE_ALTITUDES_M[index + 1]
        height = min(altitude, layer_top) - base_altitude
        if lapse_rate == 0.0:
            temperature = base_temperature
            pressure = base_pressure * math.exp(
                -STANDARD_GRAVITY_MPS2 * height / (AIR_GAS_CONSTANT_JPKGK * base_temperature)
            )
        else:
            temperature = base_temperature + lapse_rate * height
            pressure = base_pressure * (temperature / base_temperature) ** (
                -STANDARD_GRAVITY_MPS2 / (AIR_GAS_CONSTANT_JPKGK * lapse_rate)
            )
        if altitude <= layer_top:
            density = pressure / (AIR_GAS_CONSTANT_JPKGK * temperature)
            sound_speed = math.sqrt(AIR_HEAT_CAPACITY_RATIO * AIR_GAS_CONSTANT_JPKGK * temperature)
            return temperature, pressure, density, sound_speed
        base_temperature = temperature
        base_pressure = pressure
    density = base_pressure / (AIR_GAS_CONSTANT_JPKGK * base_temperature)
    sound_speed = math.sqrt(AIR_HEAT_CAPACITY_RATIO * AIR_GAS_CONSTANT_JPKGK * base_temperature)
    return base_temperature, base_pressure, density, sound_speed


def air_density(altitude_m: float) -> float:
    return standard_atmosphere(altitude_m)[2]


def speed_of_sound(altitude_m: float) -> float:
    return standard_atmosphere(altitude_m)[3]


def velocity_local_frame(velocity: np.ndarray) -> np.ndarray:
    forward = unit(velocity, np.array([1.0, 0.0, 0.0], dtype=np.float64))
    inertial_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    local_up = unit(inertial_up - np.dot(inertial_up, forward) * forward)
    if norm(local_up) <= EPS:
        east = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        local_up = unit(east - np.dot(east, forward) * forward, np.array([0.0, 1.0, 0.0]))
    local_right = unit(np.cross(forward, local_up), np.array([0.0, 0.0, 1.0]))
    return np.column_stack((forward, local_up, local_right))
