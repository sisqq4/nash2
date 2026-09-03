from __future__ import annotations

import numpy as np

from ..env.types import EngagementState, EnvironmentConfig
from .environment import BlueEscapeEnvConfig
from .policy import DiscreteBluePolicy


class BlueRLController:
    """Drop-in blue policy beside ``BlueEvasionController`` for normal v1 runs."""

    def __init__(self, policy: DiscreteBluePolicy, environment: EnvironmentConfig,
                 config: BlueEscapeEnvConfig = BlueEscapeEnvConfig()) -> None:
        config.validate(environment); self.policy, self.environment, self.config = policy, environment, config
        self.decision_steps = int(round(config.decision_interval_s / environment.time_step_s)); self._action = 0
        self._learning_active = False
        self._last_decision_step: int | None = None

    def reset(self) -> None:
        self._action = 0
        self._learning_active = False
        self._last_decision_step = None

    def __call__(self, state: EngagementState) -> dict[str, np.ndarray]:
        if not self._learning_active:
            self._learning_active = any(
                red.alive and np.linalg.norm(red.position_m - blue.position_m) < self.config.threat_detection_range_m
                for blue in state.blue
                for red in state.red
            )
        decision_due = (
            self._learning_active
            and (
                self._last_decision_step is None
                or state.step_count - self._last_decision_step >= self.decision_steps
            )
        )
        if decision_due:
            self._action = self.policy.select_action(self.encode(state), evaluation=True)
            self._last_decision_step = int(state.step_count)
        elif not self._learning_active:
            self._action = 0
        return {"action_indices": np.full(len(state.blue), self._action, dtype=np.int64)}

    def action_for(self, state: EngagementState) -> tuple[dict[str, np.ndarray], None]:
        """Compatibility with the existing blue-evasion episode runner."""
        return self(state), None

    def encode(self, state: EngagementState) -> np.ndarray:
        if len(state.blue) != 1: raise ValueError("the trained blue policy supports exactly one blue aircraft")
        if len(state.red) != self.config.missile_count:
            raise ValueError(
                f"checkpoint expects {self.config.missile_count} missiles, got {len(state.red)}"
            )
        blue = state.blue[0]
        if self.config.observation_schema == "normalized_v2":
            values = [0.0, blue.position_m[1] / 20000.0, 0.0,
                      *(blue.velocity_mps / 2000.0)]
            for red in state.red:
                values.extend((red.position_m - blue.position_m) / 200000.0)
                values.append(1.0)
            return np.asarray(values, dtype=np.float32)
        values = [*(blue.position_m / 1000.0), *(blue.velocity_mps / 1000.0)]
        for red in state.red:
            values.extend((red.position_m - blue.position_m) / 1000.0)
        return np.asarray(values, dtype=np.float32)
