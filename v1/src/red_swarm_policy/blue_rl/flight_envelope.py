"""Shared predictive flight-envelope constraint layer for blue-aircraft RL."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from ..env.math_utils import G0


_EPS = 1.0e-9


@dataclass(frozen=True)
class FlightEnvelopeConfig:
    """Prediction, envelope, and actuator limits used in training and evaluation.

    The soft limits are deliberately inside the physical limits.  A soft-limit
    violation subtracts a bounded cost from an action value, while a physical
    or unrecoverable envelope violation masks the action completely.
    """

    enabled: bool = True
    action_prediction_s: float = 0.1
    altitude_safety_margin_m: float = 500.0
    altitude_extrapolation_s: float = 2.0
    envelope_prediction_s: float = 1.0
    heading_recovery_window_s: float = 2.0
    minimum_horizontal_speed_mps: float = 150.0
    hard_minimum_horizontal_speed_mps: float = 100.0
    minimum_horizontal_speed_ratio: float = 0.70
    hard_minimum_horizontal_speed_ratio: float = 0.35
    maximum_flight_path_angle_deg: float = 45.0
    hard_maximum_flight_path_angle_deg: float = 70.0
    horizontal_decay_tolerance_mps: float = 10.0
    soft_load_command_rate_gps: float = 30.0
    hard_load_command_rate_gps: float = 100.0
    soft_roll_command_rate_deg_s: float = 240.0
    hard_roll_command_rate_deg_s: float = 1200.0
    altitude_penalty_weight: float = 4.0
    envelope_penalty_weight: float = 1.0
    command_rate_penalty_weight: float = 0.35
    integration_step_s: float = 0.02

    def validate(self) -> None:
        values = (
            self.action_prediction_s, self.altitude_safety_margin_m,
            self.altitude_extrapolation_s, self.envelope_prediction_s,
            self.heading_recovery_window_s, self.minimum_horizontal_speed_mps,
            self.hard_minimum_horizontal_speed_mps, self.minimum_horizontal_speed_ratio,
            self.hard_minimum_horizontal_speed_ratio, self.maximum_flight_path_angle_deg,
            self.hard_maximum_flight_path_angle_deg, self.horizontal_decay_tolerance_mps,
            self.soft_load_command_rate_gps, self.hard_load_command_rate_gps,
            self.soft_roll_command_rate_deg_s, self.hard_roll_command_rate_deg_s,
            self.altitude_penalty_weight, self.envelope_penalty_weight,
            self.command_rate_penalty_weight, self.integration_step_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("flight-envelope parameters must be finite")
        if min(values[:7]) <= 0.0 or self.integration_step_s <= 0.0:
            raise ValueError("flight-envelope horizons, margins, and speeds must be positive")
        if not 0.0 < self.hard_minimum_horizontal_speed_ratio <= self.minimum_horizontal_speed_ratio <= 1.0:
            raise ValueError("horizontal-speed ratios must satisfy 0 < hard <= soft <= 1")
        if not 0.0 < self.maximum_flight_path_angle_deg <= self.hard_maximum_flight_path_angle_deg < 90.0:
            raise ValueError("flight-path angle limits must satisfy 0 < soft <= hard < 90")
        if self.hard_minimum_horizontal_speed_mps > self.minimum_horizontal_speed_mps:
            raise ValueError("hard horizontal-speed minimum cannot exceed the soft minimum")
        if self.envelope_prediction_s < self.action_prediction_s:
            raise ValueError("envelope prediction must include the candidate-action interval")
        if not 0.0 < self.soft_load_command_rate_gps <= self.hard_load_command_rate_gps:
            raise ValueError("load command rates must satisfy 0 < soft <= hard")
        if not 0.0 < self.soft_roll_command_rate_deg_s <= self.hard_roll_command_rate_deg_s:
            raise ValueError("roll command rates must satisfy 0 < soft <= hard")
        if min(self.altitude_penalty_weight, self.envelope_penalty_weight,
               self.command_rate_penalty_weight, self.horizontal_decay_tolerance_mps) < 0.0:
            raise ValueError("flight-envelope weights and tolerances must be non-negative")


class FlightEnvelopeConstraintLayer:
    """Predict every discrete action, mask hard violations, then rank by RL value.

    New observation schemas expose the previous executed action and continuous
    actuator command in the state snapshot.  ``previous_action`` remains only
    as a compatibility fallback for older callers which do not expose them.
    """

    def __init__(self, config: FlightEnvelopeConfig = FlightEnvelopeConfig()) -> None:
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.previous_action = 0

    @staticmethod
    def _horizontal_metrics(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(velocity, dtype=np.float64)
        speed = np.linalg.norm(values, axis=-1)
        horizontal = np.linalg.norm(values[..., [0, 2]], axis=-1)
        ratio = horizontal / np.maximum(speed, _EPS)
        fpa = np.degrees(np.arctan2(values[..., 1], horizontal))
        return speed, horizontal, ratio, fpa

    @staticmethod
    def _accelerations_for_commands(velocities: np.ndarray, commands: np.ndarray) -> np.ndarray:
        values = np.asarray(velocities, dtype=np.float64)
        command_values = np.asarray(commands, dtype=np.float64)
        if values.shape != command_values.shape or values.shape[-1:] != (3,):
            raise ValueError("velocities and commands must have matching [..., 3] shapes")
        original_shape = values.shape
        values = values.reshape(-1, 3)
        command_values = command_values.reshape(-1, 3)
        speed = np.linalg.norm(values, axis=1, keepdims=True)
        forward = np.divide(values, speed, out=np.zeros_like(values), where=speed > _EPS)
        forward[speed[:, 0] <= _EPS] = np.array([1.0, 0.0, 0.0])
        inertial_up = np.array([0.0, 1.0, 0.0])
        local_up = inertial_up - np.sum(inertial_up * forward, axis=1, keepdims=True) * forward
        up_norm = np.linalg.norm(local_up, axis=1, keepdims=True)
        degenerate = up_norm[:, 0] <= _EPS
        if np.any(degenerate):
            east = np.array([0.0, 0.0, 1.0])
            replacement = east - np.sum(east * forward[degenerate], axis=1, keepdims=True) * forward[degenerate]
            local_up[degenerate] = replacement
            up_norm = np.linalg.norm(local_up, axis=1, keepdims=True)
        local_up = np.divide(local_up, up_norm, out=np.zeros_like(local_up), where=up_norm > _EPS)
        local_right = np.cross(forward, local_up)
        right_norm = np.linalg.norm(local_right, axis=1, keepdims=True)
        local_right = np.divide(local_right, right_norm, out=np.zeros_like(local_right), where=right_norm > _EPS)
        normal = (np.cos(command_values[:, 2, None]) * local_up
                  - np.sin(command_values[:, 2, None]) * local_right)
        gravity = np.array([0.0, -G0, 0.0], dtype=np.float64)
        acceleration = (command_values[:, 0, None] * G0 * forward
                        + command_values[:, 1, None] * G0 * normal + gravity)
        return acceleration.reshape(original_shape)

    @classmethod
    def _accelerations(cls, velocities: np.ndarray) -> np.ndarray:
        values = np.asarray(velocities, dtype=np.float64)
        if values.shape != (len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G), 3):
            raise ValueError("candidate velocities must have shape [29, 3]")
        return cls._accelerations_for_commands(values, BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)

    @staticmethod
    def _interpolated_commands(start: np.ndarray, target: np.ndarray,
                               fraction: float) -> np.ndarray:
        start_values = np.asarray(start, dtype=np.float64)
        target_values = np.asarray(target, dtype=np.float64)
        result = start_values + float(fraction) * (target_values - start_values)
        bank_delta = np.arctan2(np.sin(target_values[..., 2] - start_values[..., 2]),
                                np.cos(target_values[..., 2] - start_values[..., 2]))
        result[..., 2] = start_values[..., 2] + float(fraction) * bank_delta
        return result

    def _action_context(self, snapshot: dict[str, object]) -> tuple[int, np.ndarray]:
        previous_action = int(snapshot.get("previous_executed_action_index", self.previous_action))
        if not 0 <= previous_action < len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G):
            raise ValueError("previous executed action index is out of range")
        previous_command = np.asarray(
            snapshot.get("actual_load_command_body_g",
                         BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[previous_action]),
            dtype=np.float64,
        )
        if previous_command.shape != (3,) or not np.all(np.isfinite(previous_command)):
            raise ValueError("actual load command must be a finite three-vector")
        return previous_action, previous_command

    def _predict_repeated_action(self, position: np.ndarray, velocity: np.ndarray,
                                 duration_s: float,
                                 starting_command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        count = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)
        positions = np.repeat(np.asarray(position, dtype=np.float64)[None, :], count, axis=0)
        velocities = np.repeat(np.asarray(velocity, dtype=np.float64)[None, :], count, axis=0)
        starts = np.repeat(np.asarray(starting_command, dtype=np.float64)[None, :], count, axis=0)
        remaining = float(duration_s)
        elapsed = 0.0
        while remaining > 1.0e-12:
            dt = min(self.config.integration_step_s, remaining)
            fraction = min(1.0, (elapsed + dt) / self.config.action_prediction_s)
            commands = self._interpolated_commands(
                starts, BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G, fraction
            )
            acceleration = self._accelerations_for_commands(velocities, commands)
            positions += velocities * dt + 0.5 * acceleration * dt * dt
            velocities += acceleration * dt
            remaining -= dt
            elapsed += dt
        return positions, velocities

    def _command_rates_between(self, previous: np.ndarray,
                               commands: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        commands = np.asarray(commands, dtype=np.float64)
        previous = np.broadcast_to(np.asarray(previous, dtype=np.float64), commands.shape)
        # Express the normal command in its banked plane before differencing;
        # this treats a left/right reversal as a large command change.
        vectors = np.stack((commands[..., 0],
                            commands[..., 1] * np.cos(commands[..., 2]),
                            -commands[..., 1] * np.sin(commands[..., 2])), axis=-1)
        previous_vectors = np.stack((previous[..., 0],
                                     previous[..., 1] * np.cos(previous[..., 2]),
                                     -previous[..., 1] * np.sin(previous[..., 2])), axis=-1)
        load_rate = np.linalg.norm(vectors - previous_vectors, axis=-1) / self.config.action_prediction_s
        bank_delta = np.arctan2(np.sin(commands[..., 2] - previous[..., 2]),
                                np.cos(commands[..., 2] - previous[..., 2]))
        # Bank is irrelevant for a zero-normal-load command.
        active_bank = (np.abs(commands[..., 1]) > _EPS) & (np.abs(previous[..., 1]) > _EPS)
        roll_rate = np.where(active_bank, np.degrees(np.abs(bank_delta)) /
                             self.config.action_prediction_s, 0.0)
        return load_rate, roll_rate

    def _command_rates(self, previous_command: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._command_rates_between(previous_command, BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)

    def _heading_recovery(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        *,
        min_altitude_m: float,
        max_altitude_m: float,
        min_speed_mps: float,
        max_speed_mps: float,
        available_load_g: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Dynamically test all candidate/recovery-action pairs over the window.

        Acceleration is recomputed from the changing velocity frame at every
        integration step.  A recovery action is ramped in over one decision
        interval and remains eligible only while load/rate and physical
        altitude/speed limits are respected.
        """
        _, horizontal, ratio, fpa = self._horizontal_metrics(velocities)
        valid_now = ((horizontal >= self.config.minimum_horizontal_speed_mps)
                     & (ratio >= self.config.minimum_horizontal_speed_ratio)
                     & (np.abs(fpa) <= self.config.maximum_flight_path_angle_deg))
        recoverable = valid_now.copy()
        count = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)
        recovery_time = np.full(count, np.inf, dtype=np.float64)
        recovery_action = np.full(count, -1, dtype=np.int64)
        recovery_time[valid_now] = 0.0
        recovery_action[valid_now] = np.flatnonzero(valid_now)
        if np.all(valid_now):
            return recoverable, recovery_time, recovery_action

        candidate_commands = np.broadcast_to(
            BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[:, None, :], (count, count, 3)
        )
        target_commands = np.broadcast_to(
            BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[None, :, :], (count, count, 3)
        )
        predicted_p = np.broadcast_to(
            np.asarray(positions, dtype=np.float64)[:, None, :], (count, count, 3)
        ).copy()
        predicted_v = np.broadcast_to(
            np.asarray(velocities, dtype=np.float64)[:, None, :], (count, count, 3)
        ).copy()
        load_rate, roll_rate = self._command_rates_between(candidate_commands, target_commands)
        target_load = np.linalg.norm(target_commands[..., :2], axis=-1)
        path_feasible = (
            (target_load <= available_load_g + 1.0e-9)
            & (load_rate <= self.config.hard_load_command_rate_gps)
            & (roll_rate <= self.config.hard_roll_command_rate_deg_s)
        )
        path_feasible[valid_now, :] = False

        remaining = self.config.heading_recovery_window_s
        elapsed = 0.0
        while remaining > 1.0e-12 and np.any(path_feasible):
            dt = min(self.config.integration_step_s, remaining)
            fraction = min(1.0, (elapsed + dt) / self.config.action_prediction_s)
            applied_commands = self._interpolated_commands(
                candidate_commands, target_commands, fraction
            )
            acceleration = self._accelerations_for_commands(predicted_v, applied_commands)
            predicted_p += predicted_v * dt + 0.5 * acceleration * dt * dt
            predicted_v += acceleration * dt
            speed, recovery_horizontal, recovery_ratio, recovery_fpa = self._horizontal_metrics(
                predicted_v
            )
            path_feasible &= (
                (predicted_p[..., 1] >= min_altitude_m)
                & (predicted_p[..., 1] <= max_altitude_m)
                & (speed >= min_speed_mps)
                & (speed <= max_speed_mps)
            )
            heading_valid = (
                (recovery_horizontal >= self.config.minimum_horizontal_speed_mps)
                & (recovery_ratio >= self.config.minimum_horizontal_speed_ratio)
                & (np.abs(recovery_fpa) <= self.config.maximum_flight_path_angle_deg)
            )
            newly_recovered = path_feasible & heading_valid & ~recoverable[:, None]
            recovered_candidates = np.flatnonzero(np.any(newly_recovered, axis=1))
            for candidate in recovered_candidates:
                recovery = int(np.flatnonzero(newly_recovered[candidate])[0])
                recoverable[candidate] = True
                recovery_time[candidate] = elapsed + dt
                recovery_action[candidate] = recovery
                path_feasible[candidate, :] = False
            elapsed += dt
            remaining -= dt
        return recoverable, recovery_time, recovery_action

    def constraints(self, snapshot: dict[str, object]) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Return effective hard mask, additive soft cost, and prediction details."""
        position = np.asarray(snapshot["blue_position_m"], dtype=np.float64)
        velocity = np.asarray(snapshot["blue_velocity_mps"], dtype=np.float64)
        if position.shape != (3,) or velocity.shape != (3,) or not np.all(np.isfinite(position)) or not np.all(np.isfinite(velocity)):
            raise ValueError("blue position and velocity must be finite three-vectors")
        if not self.config.enabled:
            count = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)
            return np.ones(count, dtype=bool), np.zeros(count), {"hard_mask_reasons": []}

        previous_action, previous_command = self._action_context(snapshot)
        candidate_p, candidate_v = self._predict_repeated_action(
            position, velocity, self.config.action_prediction_s, previous_command
        )
        remaining = max(0.0, self.config.envelope_prediction_s - self.config.action_prediction_s)
        future_p, future_v = self._predict_from_candidates(candidate_p, candidate_v, remaining)
        speed, horizontal, ratio, fpa = self._horizontal_metrics(future_v)
        current_horizontal = float(np.hypot(velocity[0], velocity[2]))
        horizontal_decay = current_horizontal - horizontal
        extrapolated_altitude = candidate_p[:, 1] + candidate_v[:, 1] * self.config.altitude_extrapolation_s
        min_altitude = float(snapshot.get("min_altitude_m", 8000.0))
        max_altitude = float(snapshot.get("max_altitude_m", 12000.0))
        soft_min = min_altitude + self.config.altitude_safety_margin_m
        soft_max = max_altitude - self.config.altitude_safety_margin_m
        if soft_min >= soft_max:
            raise ValueError("altitude safety margin leaves no soft protection band")
        min_speed = float(snapshot.get("min_speed_mps", 100.0))
        max_speed = float(snapshot.get("max_speed_mps", 600.0))
        available_load = float(snapshot.get("max_load_factor_g", 9.0))
        loads = np.linalg.norm(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[:, :2], axis=1)
        load_rate, roll_rate = self._command_rates(previous_command)
        recoverable, recovery_time, recovery_action = self._heading_recovery(
            future_p, future_v,
            min_altitude_m=min_altitude,
            max_altitude_m=max_altitude,
            min_speed_mps=min_speed,
            max_speed_mps=max_speed,
            available_load_g=available_load,
        )

        checks = {
            "maximum_load": loads <= available_load + 1.0e-9,
            "physical_speed": (speed >= min_speed) & (speed <= max_speed),
            "physical_altitude": ((extrapolated_altitude >= min_altitude)
                                  & (extrapolated_altitude <= max_altitude)),
            "horizontal_speed": horizontal >= self.config.hard_minimum_horizontal_speed_mps,
            "horizontal_speed_ratio": ratio >= self.config.hard_minimum_horizontal_speed_ratio,
            "flight_path_angle": np.abs(fpa) <= self.config.hard_maximum_flight_path_angle_deg,
            "horizontal_speed_decay": ~((horizontal < self.config.minimum_horizontal_speed_mps)
                                          & (horizontal_decay > self.config.horizontal_decay_tolerance_mps)),
            "load_command_rate": load_rate <= self.config.hard_load_command_rate_gps,
            "roll_command_rate": roll_rate <= self.config.hard_roll_command_rate_deg_s,
            "heading_recovery": recoverable,
        }
        hard_mask = np.logical_and.reduce(tuple(checks.values()))
        reasons = [name for name, condition in checks.items() if np.any(~condition)]

        margin = self.config.altitude_safety_margin_m
        altitude_intrusion = (np.maximum(0.0, soft_min - extrapolated_altitude)
                              + np.maximum(0.0, extrapolated_altitude - soft_max)) / margin
        altitude_cost = self.config.altitude_penalty_weight * np.minimum(altitude_intrusion, 2.0) ** 2
        envelope_cost = self.config.envelope_penalty_weight * (
            np.maximum(0.0, self.config.minimum_horizontal_speed_mps - horizontal)
            / self.config.minimum_horizontal_speed_mps
            + np.maximum(0.0, self.config.minimum_horizontal_speed_ratio - ratio)
            / self.config.minimum_horizontal_speed_ratio
            + np.maximum(0.0, np.abs(fpa) - self.config.maximum_flight_path_angle_deg)
            / self.config.maximum_flight_path_angle_deg
            + np.maximum(0.0, horizontal_decay - self.config.horizontal_decay_tolerance_mps)
            / max(current_horizontal, self.config.minimum_horizontal_speed_mps)
            + (~recoverable).astype(np.float64)
        )
        command_cost = self.config.command_rate_penalty_weight * (
            np.maximum(0.0, load_rate - self.config.soft_load_command_rate_gps)
            / self.config.hard_load_command_rate_gps
            + np.maximum(0.0, roll_rate - self.config.soft_roll_command_rate_deg_s)
            / self.config.hard_roll_command_rate_deg_s
        )
        # Emergency maneuvering may temporarily spend soft envelope and
        # actuator margins, but it never relaxes a physical hard mask.  The
        # gate is estimated from physical threat state and exposed to the
        # learner, so selection remains Markov and independent of Q-values.
        emergency_gate = float(np.clip(snapshot.get("mechanism_emergency_gate", 0.0), 0.0, 1.0))
        envelope_gate = 1.0 - 0.50 * emergency_gate
        command_gate = 1.0 - 0.80 * emergency_gate
        soft_cost = altitude_cost + envelope_gate * envelope_cost + command_gate * command_cost

        # A deterministic least-risk action keeps Double-DQN bootstrapping and
        # behavior selection well-defined even in an already-invalid state.
        fallback_risk = (
            np.maximum(0.0, min_altitude - extrapolated_altitude) / margin
            + np.maximum(0.0, extrapolated_altitude - max_altitude) / margin
            + np.maximum(0.0, self.config.hard_minimum_horizontal_speed_mps - horizontal)
            / self.config.hard_minimum_horizontal_speed_mps
            + np.maximum(0.0, np.abs(fpa) - self.config.hard_maximum_flight_path_angle_deg)
            / self.config.hard_maximum_flight_path_angle_deg
            + np.maximum(0.0, load_rate - self.config.hard_load_command_rate_gps)
            / self.config.hard_load_command_rate_gps
            + (~recoverable).astype(np.float64)
        )
        fallback_action = int(np.argmin(fallback_risk + 1.0e-6 * np.arange(len(fallback_risk))))
        effective_mask = hard_mask.copy()
        if not effective_mask.any():
            effective_mask[fallback_action] = True
        details = {
            "hard_mask_reasons": reasons,
            "hard_checks": checks,
            "hard_mask": hard_mask,
            "hard_safe_action_count": int(hard_mask.sum()),
            "fallback_action": fallback_action,
            "fallback_required": not bool(hard_mask.any()),
            "extrapolated_altitude_m": extrapolated_altitude,
            "future_horizontal_speed_mps": horizontal,
            "future_horizontal_speed_ratio": ratio,
            "future_flight_path_angle_deg": fpa,
            "horizontal_speed_decay_mps": horizontal_decay,
            "heading_recoverable": recoverable,
            "heading_recovery_time_s": recovery_time,
            "heading_recovery_action": recovery_action,
            "previous_executed_action_index": previous_action,
            "starting_load_command_body_g": previous_command,
            "load_command_rate_gps": load_rate,
            "roll_command_rate_deg_s": roll_rate,
            "altitude_cost": altitude_cost,
            "envelope_cost": envelope_cost,
            "command_cost": command_cost,
            "emergency_gate": emergency_gate,
            "envelope_cost_gate": envelope_gate,
            "command_cost_gate": command_gate,
        }
        return effective_mask, soft_cost, details

    def _predict_from_candidates(self, positions: np.ndarray, velocities: np.ndarray,
                                 duration_s: float) -> tuple[np.ndarray, np.ndarray]:
        predicted_p = np.asarray(positions, dtype=np.float64).copy()
        predicted_v = np.asarray(velocities, dtype=np.float64).copy()
        remaining = float(duration_s)
        while remaining > 1.0e-12:
            dt = min(self.config.integration_step_s, remaining)
            acceleration = self._accelerations(predicted_v)
            predicted_p += predicted_v * dt + 0.5 * acceleration * dt * dt
            predicted_v += acceleration * dt
            remaining -= dt
        return predicted_p, predicted_v

    def select(self, q_values: np.ndarray, snapshot: dict[str, object]) -> tuple[int, dict[str, object]]:
        values = np.asarray(q_values, dtype=np.float64)
        count = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)
        if values.shape != (count,):
            raise ValueError(f"q_values must contain all {count} actions")
        finite = np.isfinite(values)
        invalid_network = not bool(finite.all())
        if finite.any():
            floor = float(np.min(values[finite]) - max(1.0, np.ptp(values[finite])))
            clean = np.where(finite, values, floor)
        else:
            clean = np.zeros(count, dtype=np.float64)
        raw_action = int(np.argmax(clean))
        if not self.config.enabled:
            self.previous_action = raw_action
            return raw_action, {
                "raw_action": raw_action, "executed_action": raw_action,
                "intervened": False, "safety_filter_intervened": False,
                "fallback_reason": "network_nan" if invalid_network else None,
                "hard_mask_reasons": [], "raw_action_hard_violation_reasons": [],
                "safe_action_count": count, "altitude_soft_penalty": 0.0,
                "envelope_soft_penalty": 0.0, "command_rate_soft_penalty": 0.0,
                "total_soft_penalty": 0.0,
                "effective_action_mask": [True] * count,
                "fallback_required": False,
                "emergency_gate": float(np.clip(
                    snapshot.get("mechanism_emergency_gate", 0.0), 0.0, 1.0
                )),
            }
        mask, soft_cost, details = self.constraints(snapshot)
        constrained = clean - soft_cost
        constrained[~mask] = -math.inf
        action = int(np.argmax(constrained))
        selected = lambda name: float(np.asarray(details[name])[action])
        recovery_time_s = selected("heading_recovery_time_s")
        raw_violation_reasons = [
            name for name, condition in details["hard_checks"].items()
            if not bool(condition[raw_action])
        ]
        self.previous_action = action
        diagnostic = {
            "raw_action": raw_action,
            "executed_action": action,
            "intervened": action != raw_action,
            "safety_filter_intervened": not bool(details["hard_mask"][raw_action]),
            "fallback_reason": "network_nan" if invalid_network else (
                "empty_hard_safe_set" if details["fallback_required"] else None
            ),
            "hard_mask_reasons": list(details["hard_mask_reasons"]),
            "raw_action_hard_violation_reasons": raw_violation_reasons,
            "safe_action_count": int(np.sum(mask)),
            "soft_altitude_min_m": float(snapshot.get("min_altitude_m", 8000.0))
            + self.config.altitude_safety_margin_m,
            "soft_altitude_max_m": float(snapshot.get("max_altitude_m", 12000.0))
            - self.config.altitude_safety_margin_m,
            "predicted_altitude_m": selected("extrapolated_altitude_m"),
            "predicted_horizontal_speed_mps": selected("future_horizontal_speed_mps"),
            "predicted_horizontal_speed_ratio": selected("future_horizontal_speed_ratio"),
            "predicted_flight_path_angle_deg": selected("future_flight_path_angle_deg"),
            "predicted_horizontal_speed_decay_mps": selected("horizontal_speed_decay_mps"),
            "heading_recoverable": bool(np.asarray(details["heading_recoverable"])[action]),
            "heading_recovery_time_s": (
                recovery_time_s if math.isfinite(recovery_time_s) else None
            ),
            "heading_recovery_action": int(
                np.asarray(details["heading_recovery_action"])[action]
            ),
            "load_command_rate_gps": selected("load_command_rate_gps"),
            "roll_command_rate_deg_s": selected("roll_command_rate_deg_s"),
            "altitude_soft_penalty": selected("altitude_cost"),
            "envelope_soft_penalty": selected("envelope_cost"),
            "command_rate_soft_penalty": selected("command_cost"),
            "total_soft_penalty": float(soft_cost[action]),
            "effective_action_mask": np.asarray(mask, dtype=bool).tolist(),
            "fallback_required": bool(details["fallback_required"]),
            "emergency_gate": float(details["emergency_gate"]),
            "envelope_cost_gate": float(details["envelope_cost_gate"]),
            "command_cost_gate": float(details["command_cost_gate"]),
        }
        return action, diagnostic
