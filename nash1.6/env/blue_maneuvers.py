
"""Blue-aircraft maneuver library.

This module defines a small library of high-level blue maneuvers
(e.g., barrel roll, snake, high-G break). The current environment
does *not* directly call these maneuvers yet, but they are provided
as a reusable interface for future policy training.

Typical usage in the future:
    - Treat each maneuver as a high-level action choice;
    - For a chosen maneuver, call `apply_maneuver_step` every time step,
      passing the current `step_index` within the maneuver window.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class BlueManeuver:
    """Definition of a blue-aircraft maneuver.

    Attributes:
        name: Short name, e.g. "barrel_roll_left".
        duration_steps: Number of discrete time steps for this maneuver.
        description: Text explanation of the maneuver style.
        accel_pattern: Array of shape (duration_steps, 3), each row is a
            *direction* (unit or sub-unit) of acceleration in (x, y, z),
            to be scaled by a scalar acceleration magnitude.
    """

    name: str
    duration_steps: int
    description: str
    accel_pattern: np.ndarray


def _normalize_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms < 1e-8, 1.0, norms)
    return arr / norms


def build_default_maneuvers(dt: float) -> Dict[str, BlueManeuver]:
    """Construct a dictionary of default maneuvers.

    The generated patterns are heuristic but capture typical shapes:
    - barrel roll: helical path in the lateral plane;
    - snake: alternating left-right lateral acceleration;
    - high-G break: strong one-side pull for a short burst.
    """
    maneuvers: Dict[str, BlueManeuver] = {}

    # ---------------------- Barrel roll (left / right) ----------------------
    # 4-second roll
    T_roll = max(1, int(round(4.0 / dt)))
    angles = np.linspace(0.0, 2.0 * np.pi, T_roll, endpoint=False)
    # In this simplified model, we use primarily lateral (y-z) accelerations
    # while keeping x acceleration near zero.
    y = np.cos(angles)
    z = np.sin(angles)
    pattern = np.stack([np.zeros_like(y), y, z], axis=1)
    pattern = _normalize_rows(pattern)

    maneuvers["barrel_roll_left"] = BlueManeuver(
        name="barrel_roll_left",
        duration_steps=T_roll,
        description="4s left-handed barrel roll (lateral circular acceleration).",
        accel_pattern=pattern.copy(),
    )

    maneuvers["barrel_roll_right"] = BlueManeuver(
        name="barrel_roll_right",
        duration_steps=T_roll,
        description="4s right-handed barrel roll (mirror in lateral plane).",
        accel_pattern=np.stack([pattern[:, 0], -pattern[:, 1], pattern[:, 2]], axis=1),
    )

    # --------------------------- Snake (horizontal) -------------------------
    # 6-second horizontal snake, alternating left/right every ~1s
    T_snake = max(1, int(round(6.0 / dt)))
    t_idx = np.arange(T_snake)
    # Use a square wave in y-direction
    period_steps = max(1, int(round(1.0 / dt)))
    sign = np.sign(np.sin(2.0 * np.pi * t_idx / period_steps))
    sign[sign == 0.0] = 1.0
    pattern_snake = np.stack([np.zeros_like(sign), sign, np.zeros_like(sign)], axis=1)

    maneuvers["snake_horizontal"] = BlueManeuver(
        name="snake_horizontal",
        duration_steps=T_snake,
        description="6s horizontal snake (alternating left/right y-acceleration).",
        accel_pattern=pattern_snake,
    )

    # --------------------------- Snake (vertical) ---------------------------
    pattern_snake_v = np.stack([np.zeros_like(sign), np.zeros_like(sign), sign], axis=1)
    maneuvers["snake_vertical"] = BlueManeuver(
        name="snake_vertical",
        duration_steps=T_snake,
        description="6s vertical snake (alternating up/down z-acceleration).",
        accel_pattern=pattern_snake_v,
    )

    # -------------------------- High-G break left/right ---------------------
    # 3-second strong lateral pull
    T_break = max(1, int(round(3.0 / dt)))
    pat_break_left = np.tile(np.array([[0.0, 1.0, 0.0]], dtype=float), (T_break, 1))
    pat_break_right = np.tile(np.array([[0.0, -1.0, 0.0]], dtype=float), (T_break, 1))

    maneuvers["high_g_break_left"] = BlueManeuver(
        name="high_g_break_left",
        duration_steps=T_break,
        description="3s high-G break to the left (strong +y acceleration).",
        accel_pattern=pat_break_left,
    )

    maneuvers["high_g_break_right"] = BlueManeuver(
        name="high_g_break_right",
        duration_steps=T_break,
        description="3s high-G break to the right (strong -y acceleration).",
        accel_pattern=pat_break_right,
    )

    return maneuvers


def list_maneuver_names(dt: float) -> List[str]:
    """Return the names of all default maneuvers for a given time step."""
    return list(build_default_maneuvers(dt).keys())


def apply_maneuver_step(
    pos: np.ndarray,
    vel: np.ndarray,
    maneuver: BlueManeuver,
    step_index: int,
    dt: float,
    accel_mag: float,
    v_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a single time step of the given maneuver.

    Args:
        pos: current blue position (3,)
        vel: current blue velocity (3,)
        maneuver: maneuver definition
        step_index: which step of the maneuver we are in [0, duration_steps)
        dt: time step
        accel_mag: scalar acceleration magnitude (e.g., EnvConfig.blue_accel)
        v_max: maximum speed for the aircraft

    Returns:
        (new_pos, new_vel)
    """
    pos = pos.astype(float)
    vel = vel.astype(float)

    if step_index < 0 or step_index >= maneuver.duration_steps:
        # Outside the maneuver window: keep straight flight
        new_pos = pos + vel * dt
        return new_pos, vel

    direction = maneuver.accel_pattern[step_index]
    a = accel_mag * direction

    vel = vel + a * dt
    speed = np.linalg.norm(vel)
    if speed > v_max and speed > 1e-8:
        vel = vel / speed * v_max

    pos = pos + vel * dt
    return pos, vel
