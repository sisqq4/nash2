from __future__ import annotations

import numpy as np

from .math_utils import EPS, norm, unit
from .types import MissileConfig, ThreeDoFState, los_kinematics


GUIDANCE_CONTRACT = "pure_proportional_navigation_vm_v1"


class ProportionalNavigationGuidance:
    contract = GUIDANCE_CONTRACT

    def __init__(self, missile: MissileConfig) -> None:
        self.missile = missile

    def command(self, missile_state: ThreeDoFState, target_state: ThreeDoFState | None) -> np.ndarray:
        if target_state is None or not missile_state.alive or not target_state.alive:
            return np.zeros(3, dtype=np.float64)
        los = los_kinematics(missile_state, target_state)
        return self.command_from_kinematics(
            missile_velocity_mps=missile_state.velocity_mps,
            los_unit=los.los_unit,
            los_rate_radps=los.los_rate_radps,
        )

    def command_from_kinematics(
        self,
        *,
        missile_velocity_mps: np.ndarray,
        los_unit: np.ndarray,
        los_rate_radps: np.ndarray,
    ) -> np.ndarray:
        """Compute strict 3-D pure-PN acceleration normal to missile velocity.

        ``los_rate_radps`` is the derivative of the LOS unit vector, so
        ``cross(los_unit, los_rate_radps)`` is the LOS angular-velocity
        vector. Pure PN rotates the missile velocity direction at ``N`` times
        that angular velocity and therefore scales the command with missile
        speed rather than closing speed.
        """
        missile_velocity = np.asarray(missile_velocity_mps, dtype=np.float64)
        los = np.asarray(los_unit, dtype=np.float64)
        los_rate = np.asarray(los_rate_radps, dtype=np.float64)
        for name, value in (
            ("missile_velocity_mps", missile_velocity),
            ("los_unit", los),
            ("los_rate_radps", los_rate),
        ):
            if value.shape != (3,):
                raise ValueError(f"{name} shape {value.shape} must be (3,)")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must contain only finite values")

        speed = norm(missile_velocity)
        if speed <= EPS or norm(los) <= EPS:
            return np.zeros(3, dtype=np.float64)
        los_angular_velocity = np.cross(unit(los), los_rate)
        return (
            self.missile.proportional_navigation_gain
            * speed
            * np.cross(los_angular_velocity, unit(missile_velocity))
        )
