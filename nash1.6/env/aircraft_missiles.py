
"""Aircraft and radar-missile classes for the 3D escape game.

These classes wrap the existing kinematic update functions so that
the environment can treat the blue aircraft and red missiles as
objects, while keeping all physical parameters and behaviour
identical to the earlier implementation.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np

from .missile_dynamics import update_blue_state, update_missiles_pn
from .blue_action_library import BlueStrategy, build_escape_strategies


class Aircraft:
    """Blue aircraft model (evasive target)."""

    def __init__(
        self,
        dt: float,
        accel_mag: float,
        v_max: float,
        v_min: float,
        max_sustained_pitch: float | None = None,
    ) -> None:
        self.dt = float(dt)
        self.accel_mag = float(accel_mag)
        self.v_max = float(v_max)
        self.v_min = float(v_min)
        self.max_sustained_pitch = max_sustained_pitch
        self._strategies: List[BlueStrategy] = build_escape_strategies()
        self._pending_actions: List[np.ndarray] = []
        self._forced_actions: List[np.ndarray] = []
        self._roll_rad: float | None = None

    @property
    def num_strategies(self) -> int:
        return len(self._strategies)

    @property
    def roll_rad(self) -> float | None:
        return self._roll_rad

    def step(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        action: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Update aircraft state according to the chosen action."""
        if self._forced_actions:
            next_action = self._forced_actions.pop(0)
        else:
            if not self._pending_actions:
                if not (0 <= action < len(self._strategies)):
                    raise ValueError(f"Invalid blue strategy {action}")
                strategy = self._strategies[action]
                self._pending_actions = [a.copy() for a in strategy.actions]
            next_action = self._pending_actions.pop(0)
        pos, vel, self._roll_rad = update_blue_state(
            pos,
            vel,
            next_action,
            dt=self.dt,
            accel_mag=self.accel_mag,
            v_max=self.v_max,
            v_min=self.v_min,
            max_sustained_pitch=self.max_sustained_pitch,
            roll_state=self._roll_rad,
        )
        return pos, vel

    def force_actions(self, actions: List[np.ndarray]) -> None:
        self._forced_actions = [a.copy() for a in actions]

    def has_forced_actions(self) -> bool:
        return bool(self._forced_actions)

    def reset(self) -> None:
        self._roll_rad = None


class Missiles:
    """Radar missile model (pursuit attacker).

    This class updates a *batch* of missiles in one call. Per-missile
    position, velocity and navigation gains are still stored and
    managed by the environment for compatibility with the original
    vectorised implementation.
    """

    def __init__(self, dt: float, speed: float, max_overload_g: float | None = None) -> None:
        self.dt = float(dt)
        self.speed = float(speed)
        self.max_overload_g = None if max_overload_g is None else float(max_overload_g)

    def step(
        self,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
        missile_speed: np.ndarray | float,
        blue_pos: np.ndarray,
        blue_vel: np.ndarray,
        nav_gains: np.ndarray,
        max_overload_g: float | np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return update_missiles_pn(
            missile_pos,
            missile_vel,
            blue_pos,
            blue_vel,
            missile_speed,
            self.dt,
            nav_gains,
            max_overload_g=self.max_overload_g if max_overload_g is None else max_overload_g,
        )
