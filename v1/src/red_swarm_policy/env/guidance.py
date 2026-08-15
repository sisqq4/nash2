from __future__ import annotations

import numpy as np

from .types import MissileConfig, ThreeDoFState, los_kinematics


class ProportionalNavigationGuidance:
    def __init__(self, missile: MissileConfig) -> None:
        self.missile = missile

    def command(self, missile_state: ThreeDoFState, target_state: ThreeDoFState | None) -> np.ndarray:
        if target_state is None or not missile_state.alive or not target_state.alive:
            return np.zeros(3, dtype=np.float64)
        los = los_kinematics(missile_state, target_state)
        closing = max(0.0, los.closing_speed_mps)
        return self.missile.proportional_navigation_gain * closing * los.los_rate_radps
