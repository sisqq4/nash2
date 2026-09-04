from __future__ import annotations

import math
import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from ..env.types import EngagementState, EnvironmentConfig
from .environment import BlueEscapeEnvConfig, blue_action_context
from .flight_envelope import FlightEnvelopeConfig, FlightEnvelopeConstraintLayer
from .policy import DiscreteBluePolicy


class BlueRLController:
    """Drop-in blue policy beside ``BlueEvasionController`` for normal v1 runs."""

    def __init__(self, policy: DiscreteBluePolicy, environment: EnvironmentConfig,
                 config: BlueEscapeEnvConfig = BlueEscapeEnvConfig()) -> None:
        config.validate(environment); self.policy, self.environment, self.config = policy, environment, config
        self.decision_steps = int(round(config.decision_interval_s / environment.time_step_s)); self._action = 0
        self._learning_active = False
        self._last_decision_step: int | None = None
        saved = getattr(getattr(policy, "config", None), "flight_envelope_config", None)
        envelope_config = (FlightEnvelopeConfig(**saved) if saved else
                           FlightEnvelopeConfig(action_prediction_s=config.decision_interval_s))
        if saved and not math.isclose(envelope_config.action_prediction_s,
                                      config.decision_interval_s,
                                      rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError("controller decision interval must match the checkpoint flight envelope")
        self.constraint_layer = FlightEnvelopeConstraintLayer(envelope_config)
        self.last_action_diagnostic: dict[str, object] = {}

    def reset(self) -> None:
        self._action = 0
        self._learning_active = False
        self._last_decision_step = None
        self.constraint_layer.reset()
        self.last_action_diagnostic = {}

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
            observation = self.encode(state)
            action_values = getattr(self.policy, "expected_action_values", None)
            if callable(action_values):
                q_values = np.asarray(action_values(observation, evaluation=True), dtype=np.float64).reshape(-1)
            else:
                raw_action = int(self.policy.select_action(observation, evaluation=True))
                q_values = np.zeros(len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G), dtype=np.float64)
                q_values[raw_action] = 1.0
            self._action, self.last_action_diagnostic = self.constraint_layer.select(
                q_values, self._snapshot(state)
            )
            self._last_decision_step = int(state.step_count)
        elif not self._learning_active:
            self._action = 0
        return {"action_indices": np.full(len(state.blue), self._action, dtype=np.int64)}

    def _snapshot(self, state: EngagementState) -> dict[str, object]:
        blue = state.blue[0]
        return {
            "blue_position_m": blue.position_m.tolist(),
            "blue_velocity_mps": blue.velocity_mps.tolist(),
            "time_s": float(state.time_s),
            "red_positions_m": [red.position_m.tolist() for red in state.red],
            "red_velocities_mps": [red.velocity_mps.tolist() for red in state.red],
            "red_alive": [bool(red.alive) for red in state.red],
            "min_altitude_m": self.environment.aircraft.min_altitude_m,
            "max_altitude_m": self.environment.aircraft.max_altitude_m,
            "min_speed_mps": self.environment.aircraft.min_speed_mps,
            "max_speed_mps": self.environment.aircraft.max_speed_mps,
            "max_load_factor_g": self.environment.aircraft.max_load_factor_g,
            "previous_executed_action_index": self._action,
            "actual_load_command_body_g": BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[self._action].tolist(),
        }

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
        if self.config.observation_schema in {"normalized_v2", "normalized_v3"}:
            values = [0.0, blue.position_m[1] / 20000.0, 0.0,
                      *(blue.velocity_mps / 2000.0)]
            for red in state.red:
                values.extend((red.position_m - blue.position_m) / 200000.0)
                values.append(1.0)
            if self.config.observation_schema == "normalized_v3":
                values.extend(blue_action_context(
                    self._action, BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[self._action]
                ))
            return np.asarray(values, dtype=np.float32)
        values = [*(blue.position_m / 1000.0), *(blue.velocity_mps / 1000.0)]
        for red in state.red:
            values.extend((red.position_m - blue.position_m) / 1000.0)
        return np.asarray(values, dtype=np.float32)
