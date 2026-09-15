from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np

from .actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from .blue_evasion_kernel import (
    propagate_aircraft_candidates,
    score_aircraft_candidates,
)
from .math_utils import EPS, G0, UP_AXIS, norm
from .physics import ThreeDoFPhysicsLayer
from .types import EngagementState, EnvironmentConfig, ThreeDoFState


BlueEvasionExecutionBackend = Literal[
    "reference",
    "vectorized_guarded",
    "shadow",
]
BLUE_EVASION_EXECUTION_BACKENDS = (
    "reference",
    "vectorized_guarded",
    "shadow",
)


@dataclass
class BlueEvasionRuntimeStats:
    decision_count: int = 0
    threatened_blue_count: int = 0
    reference_fallback_count: int = 0
    near_tie_fallback_count: int = 0
    nonfinite_fallback_count: int = 0
    shadow_action_mismatch_count: int = 0
    max_score_abs_error: float = 0.0
    min_top2_margin: float = math.inf
    candidate_kernel_seconds: float = 0.0
    reference_seconds: float = 0.0

    def output_record(self, backend: str) -> dict[str, object]:
        return {
            "execution_backend": backend,
            "decision_count": self.decision_count,
            "threatened_blue_count": self.threatened_blue_count,
            "reference_fallback_count": self.reference_fallback_count,
            "near_tie_fallback_count": self.near_tie_fallback_count,
            "nonfinite_fallback_count": self.nonfinite_fallback_count,
            "shadow_action_mismatch_count": self.shadow_action_mismatch_count,
            "max_score_abs_error": self.max_score_abs_error,
            "min_top2_margin": (
                self.min_top2_margin
                if math.isfinite(self.min_top2_margin)
                else None
            ),
            "candidate_kernel_seconds": self.candidate_kernel_seconds,
            "reference_seconds": self.reference_seconds,
        }


@dataclass(frozen=True)
class BlueEvasionConfig:
    """Configuration for the deterministic blue-side evasion rule machine."""

    decision_interval_s: float = 0.1
    detection_range_m: float = 60000.0
    critical_range_m: float = 30000.0
    lookahead_s: float = 6.0
    altitude_margin_m: float = 500.0
    altitude_prediction_s: float = 2.0
    speed_margin_mps: float = 25.0
    effort_penalty: float = 0.04
    switch_penalty: float = 0.02
    targeted_multiplier: float = 1.35
    seeker_lock_multiplier: float = 1.50

    def validate(self, environment: EnvironmentConfig) -> None:
        values = (
            self.decision_interval_s,
            self.detection_range_m,
            self.critical_range_m,
            self.lookahead_s,
            self.altitude_margin_m,
            self.altitude_prediction_s,
            self.speed_margin_mps,
            self.effort_penalty,
            self.switch_penalty,
            self.targeted_multiplier,
            self.seeker_lock_multiplier,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("blue evasion configuration values must be finite")
        if self.decision_interval_s <= 0.0 or self.lookahead_s <= 0.0:
            raise ValueError("decision_interval_s and lookahead_s must be positive")
        if not 0.0 < self.critical_range_m < self.detection_range_m:
            raise ValueError("critical_range_m must be in (0, detection_range_m)")
        if self.altitude_margin_m < 0.0 or self.altitude_prediction_s <= 0.0:
            raise ValueError("altitude safety values are invalid")
        if self.speed_margin_mps < 0.0 or self.effort_penalty < 0.0 or self.switch_penalty < 0.0:
            raise ValueError("speed margin and penalties must be non-negative")
        if self.targeted_multiplier < 1.0 or self.seeker_lock_multiplier < 1.0:
            raise ValueError("threat multipliers must be at least 1")
        ratio = self.decision_interval_s / environment.time_step_s
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("decision_interval_s must be an integer multiple of environment time_step_s")
        altitude_span = environment.aircraft.max_altitude_m - environment.aircraft.min_altitude_m
        if 2.0 * self.altitude_margin_m >= altitude_span:
            raise ValueError("altitude_margin_m must be less than half the aircraft altitude span")
        speed_span = environment.aircraft.max_speed_mps - environment.aircraft.min_speed_mps
        if 2.0 * self.speed_margin_mps >= speed_span:
            raise ValueError("speed_margin_mps must be less than half the aircraft speed span")


@dataclass(frozen=True)
class BlueThreatAssessment:
    red_index: int
    range_m: float
    closing_speed_mps: float
    time_to_closest_s: float
    predicted_miss_m: float
    targeted: bool
    seeker_locked: bool
    severity: float


@dataclass(frozen=True)
class BlueEvasionDecision:
    time_s: float
    step_count: int
    action_indices: np.ndarray
    modes: tuple[str, ...]
    primary_threat_indices: np.ndarray
    primary_threat_ranges_m: np.ndarray
    primary_closing_speeds_mps: np.ndarray
    selected_scores: np.ndarray

    def action(self) -> dict[str, np.ndarray]:
        return {"action_indices": np.asarray(self.action_indices, dtype=np.int64).copy()}

    def output_record(self) -> dict[str, object]:
        indices = np.asarray(self.action_indices, dtype=np.int64)
        threat_ranges = [
            float(value) if math.isfinite(float(value)) else None
            for value in np.asarray(self.primary_threat_ranges_m, dtype=np.float64)
        ]
        return {
            "event": "blue_evasion_decision",
            "time_s": float(self.time_s),
            "step_count": int(self.step_count),
            "blue_action_api_indices": indices.tolist(),
            "blue_action_library_entries": (indices + 1).tolist(),
            "modes": list(self.modes),
            "primary_threat_indices": np.asarray(self.primary_threat_indices, dtype=np.int64).tolist(),
            "primary_threat_ranges_m": threat_ranges,
            "primary_closing_speeds_mps": np.asarray(
                self.primary_closing_speeds_mps,
                dtype=np.float64,
            ).tolist(),
        }


class BlueEvasionRuleMachine:
    """Score all 29 library actions from the current engagement state."""

    def __init__(
        self,
        environment: EnvironmentConfig,
        config: BlueEvasionConfig = BlueEvasionConfig(),
        *,
        execution_backend: BlueEvasionExecutionBackend = "vectorized_guarded",
    ) -> None:
        environment.validate()
        config.validate(environment)
        if execution_backend not in BLUE_EVASION_EXECUTION_BACKENDS:
            raise ValueError(
                "execution_backend must be reference, vectorized_guarded, or shadow"
            )
        self.environment = environment
        self.config = config
        self.execution_backend = execution_backend
        self.decision_steps = int(round(config.decision_interval_s / environment.time_step_s))
        self._physics = ThreeDoFPhysicsLayer(environment)
        self._previous_indices: np.ndarray | None = None
        self._runtime_stats = BlueEvasionRuntimeStats()
        commands = np.asarray(
            BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G,
            dtype=np.float64,
        ).copy()
        axial_acceleration = commands[:, 0] * G0
        normal_acceleration = commands[:, 1] * G0
        cos_bank = np.array(
            [math.cos(float(command[2])) for command in commands],
            dtype=np.float64,
        )
        sin_bank = np.array(
            [math.sin(float(command[2])) for command in commands],
            dtype=np.float64,
        )
        max_load_factor_g = environment.aircraft.max_load_factor_g
        action_effort = np.array(
            [
                (
                    abs(float(command[0])) / max_load_factor_g
                    + abs(float(command[1]) - 1.0) / max(max_load_factor_g - 1.0, 1.0)
                    + abs(float(command[2])) / math.pi
                )
                / 3.0
                for command in commands
            ],
            dtype=np.float64,
        )
        for values in (
            commands,
            axial_acceleration,
            normal_acceleration,
            cos_bank,
            sin_bank,
            action_effort,
        ):
            values.setflags(write=False)
        self._commands = commands
        self._axial_acceleration_mps2 = axial_acceleration
        self._normal_acceleration_mps2 = normal_acceleration
        self._cos_bank = cos_bank
        self._sin_bank = sin_bank
        self._action_effort = action_effort

    def reset(self) -> None:
        self._previous_indices = None
        self._runtime_stats = BlueEvasionRuntimeStats()

    def runtime_statistics(self) -> dict[str, object]:
        return self._runtime_stats.output_record(self.execution_backend)

    def decide(self, state: EngagementState) -> BlueEvasionDecision:
        self._runtime_stats.decision_count += 1
        n_blue = len(state.blue)
        previous = self._previous_indices
        if previous is None or previous.shape != (n_blue,):
            previous = np.zeros(n_blue, dtype=np.int64)

        indices = np.zeros(n_blue, dtype=np.int64)
        modes: list[str] = []
        primary_indices = np.full(n_blue, -1, dtype=np.int64)
        primary_ranges = np.full(n_blue, math.inf, dtype=np.float64)
        primary_closing = np.zeros(n_blue, dtype=np.float64)
        selected_scores = np.zeros(n_blue, dtype=np.float64)

        for blue_index, blue in enumerate(state.blue):
            if not blue.alive:
                modes.append("inactive")
                continue
            threats = self._assess_threats(state, blue_index)
            if not threats:
                action_index, mode = self._recovery_or_cruise(blue)
                indices[blue_index] = action_index
                modes.append(mode)
                continue

            self._runtime_stats.threatened_blue_count += 1
            primary = max(threats, key=lambda item: (item.severity, -item.range_m, -item.red_index))
            primary_indices[blue_index] = primary.red_index
            primary_ranges[blue_index] = primary.range_m
            primary_closing[blue_index] = primary.closing_speed_mps
            scores, action_index = self._score_and_select_actions(
                state,
                blue_index,
                threats,
                int(previous[blue_index]),
            )
            indices[blue_index] = action_index
            selected_scores[blue_index] = scores[action_index]
            critical = (
                primary.seeker_locked
                or primary.range_m <= self.config.critical_range_m
                or (
                    primary.time_to_closest_s <= self.config.lookahead_s
                    and primary.predicted_miss_m <= self.config.critical_range_m
                )
            )
            modes.append("critical" if critical else "evade")

        self._previous_indices = indices.copy()
        return BlueEvasionDecision(
            time_s=float(state.time_s),
            step_count=int(state.step_count),
            action_indices=indices,
            modes=tuple(modes),
            primary_threat_indices=primary_indices,
            primary_threat_ranges_m=primary_ranges,
            primary_closing_speeds_mps=primary_closing,
            selected_scores=selected_scores,
        )

    def _score_and_select_actions(
        self,
        state: EngagementState,
        blue_index: int,
        threats: list[BlueThreatAssessment],
        previous_index: int,
    ) -> tuple[np.ndarray, int]:
        if self.execution_backend == "reference":
            scores = self._timed_reference_scores(
                state,
                blue_index,
                threats,
                previous_index,
            )
            return scores, int(np.argmax(scores))

        started = time.perf_counter()
        vectorized_scores = self._score_all_actions_vectorized(
            state,
            blue_index,
            threats,
            previous_index,
        )
        self._runtime_stats.candidate_kernel_seconds += time.perf_counter() - started
        vectorized_index, near_tie, nonfinite = self._guarded_argmax(vectorized_scores)

        if self.execution_backend == "shadow":
            reference_scores = self._timed_reference_scores(
                state,
                blue_index,
                threats,
                previous_index,
            )
            reference_index = int(np.argmax(reference_scores))
            finite = np.isfinite(reference_scores) & np.isfinite(vectorized_scores)
            if np.any(finite):
                self._runtime_stats.max_score_abs_error = max(
                    self._runtime_stats.max_score_abs_error,
                    float(np.max(np.abs(reference_scores[finite] - vectorized_scores[finite]))),
                )
            if vectorized_index != reference_index:
                self._runtime_stats.shadow_action_mismatch_count += 1
            if near_tie or nonfinite:
                self._record_reference_fallback(near_tie=near_tie, nonfinite=nonfinite)
            return reference_scores, reference_index

        if near_tie or nonfinite:
            self._record_reference_fallback(near_tie=near_tie, nonfinite=nonfinite)
            reference_scores = self._timed_reference_scores(
                state,
                blue_index,
                threats,
                previous_index,
            )
            finite = np.isfinite(reference_scores) & np.isfinite(vectorized_scores)
            if np.any(finite):
                self._runtime_stats.max_score_abs_error = max(
                    self._runtime_stats.max_score_abs_error,
                    float(np.max(np.abs(reference_scores[finite] - vectorized_scores[finite]))),
                )
            return reference_scores, int(np.argmax(reference_scores))
        return vectorized_scores, vectorized_index

    def _timed_reference_scores(
        self,
        state: EngagementState,
        blue_index: int,
        threats: list[BlueThreatAssessment],
        previous_index: int,
    ) -> np.ndarray:
        started = time.perf_counter()
        scores = self._score_all_actions_reference(
            state,
            blue_index,
            threats,
            previous_index,
        )
        self._runtime_stats.reference_seconds += time.perf_counter() - started
        return scores

    def _score_all_actions_reference(
        self,
        state: EngagementState,
        blue_index: int,
        threats: list[BlueThreatAssessment],
        previous_index: int,
    ) -> np.ndarray:
        return np.array(
            [
                self._score_action(
                    state,
                    blue_index,
                    threats,
                    action_index,
                    previous_index,
                )
                for action_index in range(len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G))
            ],
            dtype=np.float64,
        )

    def _score_all_actions_vectorized(
        self,
        state: EngagementState,
        blue_index: int,
        threats: list[BlueThreatAssessment],
        previous_index: int,
    ) -> np.ndarray:
        blue = state.blue[blue_index]
        aircraft = self.environment.aircraft
        predicted_position, predicted_velocity = propagate_aircraft_candidates(
            blue.position_m,
            blue.velocity_mps,
            axial_acceleration_mps2=self._axial_acceleration_mps2,
            normal_acceleration_mps2=self._normal_acceleration_mps2,
            cos_bank=self._cos_bank,
            sin_bank=self._sin_bank,
            time_step_s=self.environment.time_step_s,
            decision_steps=self.decision_steps,
            min_speed_mps=aircraft.min_speed_mps,
            max_speed_mps=aircraft.max_speed_mps,
            min_altitude_m=aircraft.min_altitude_m,
            max_altitude_m=aircraft.max_altitude_m,
        )
        red_position = np.array(
            [state.red[threat.red_index].position_m for threat in threats],
            dtype=np.float64,
        )
        red_velocity = np.array(
            [state.red[threat.red_index].velocity_mps for threat in threats],
            dtype=np.float64,
        )
        severity = np.array(
            [threat.severity for threat in threats],
            dtype=np.float64,
        )
        return score_aircraft_candidates(
            original_velocity_mps=blue.velocity_mps,
            predicted_position_m=predicted_position,
            predicted_velocity_mps=predicted_velocity,
            red_position_m=red_position,
            red_velocity_mps=red_velocity,
            threat_severity=severity,
            action_effort=self._action_effort,
            previous_index=previous_index,
            decision_interval_s=self.config.decision_interval_s,
            lookahead_s=self.config.lookahead_s,
            critical_range_m=self.config.critical_range_m,
            detection_range_m=self.config.detection_range_m,
            max_load_factor_g=aircraft.max_load_factor_g,
            effort_penalty=self.config.effort_penalty,
            switch_penalty=self.config.switch_penalty,
            min_altitude_m=aircraft.min_altitude_m,
            max_altitude_m=aircraft.max_altitude_m,
            altitude_margin_m=self.config.altitude_margin_m,
            altitude_prediction_s=self.config.altitude_prediction_s,
            min_speed_mps=aircraft.min_speed_mps,
            max_speed_mps=aircraft.max_speed_mps,
            speed_margin_mps=self.config.speed_margin_mps,
        )

    def _guarded_argmax(self, scores: np.ndarray) -> tuple[int, bool, bool]:
        values = np.asarray(scores, dtype=np.float64)
        finite = bool(np.all(np.isfinite(values)))
        if not finite:
            return 0, False, True
        best_index = int(np.argmax(values))
        if values.size < 2:
            return best_index, False, False
        second_score = float(np.max(np.delete(values, best_index)))
        best_score = float(values[best_index])
        margin = best_score - second_score
        self._runtime_stats.min_top2_margin = min(
            self._runtime_stats.min_top2_margin,
            margin,
        )
        tolerance = max(
            1.0e-12,
            1.0e-10 * max(abs(best_score), abs(second_score), 1.0),
        )
        return best_index, margin <= 2.0 * tolerance, False

    def _record_reference_fallback(
        self,
        *,
        near_tie: bool,
        nonfinite: bool,
    ) -> None:
        self._runtime_stats.reference_fallback_count += 1
        if near_tie:
            self._runtime_stats.near_tie_fallback_count += 1
        if nonfinite:
            self._runtime_stats.nonfinite_fallback_count += 1

    def _assess_threats(
        self,
        state: EngagementState,
        blue_index: int,
    ) -> list[BlueThreatAssessment]:
        blue = state.blue[blue_index]
        threats: list[BlueThreatAssessment] = []
        range_span = self.config.detection_range_m - self.config.critical_range_m
        missile_speed_scale = max(self.environment.missile.max_speed_mps, 1.0)
        for red_index, red in enumerate(state.red):
            if not red.alive:
                continue
            relative_position = red.position_m - blue.position_m
            distance = max(norm(relative_position), EPS)
            relative_velocity = red.velocity_mps - blue.velocity_mps
            closing_speed = max(0.0, -float(np.dot(relative_velocity, relative_position / distance)))
            targeted = red.current_target_index == blue_index
            seeker_locked = bool(red.seeker_locked and targeted)
            if closing_speed <= 0.0 and not seeker_locked:
                continue
            if distance > self.config.detection_range_m:
                continue
            relative_speed_squared = float(np.dot(relative_velocity, relative_velocity))
            time_to_closest = 0.0
            if relative_speed_squared > EPS:
                time_to_closest = float(
                    np.clip(
                        -np.dot(relative_position, relative_velocity) / relative_speed_squared,
                        0.0,
                        self.config.lookahead_s,
                    )
                )
            miss_vector = relative_position + relative_velocity * time_to_closest
            predicted_miss = norm(miss_vector)
            range_factor = float(
                np.clip(
                    (self.config.detection_range_m - distance) / range_span,
                    0.0,
                    1.0,
                )
            )
            closing_factor = float(np.clip(closing_speed / missile_speed_scale, 0.0, 1.0))
            miss_factor = float(
                np.clip(
                    (self.config.critical_range_m - predicted_miss) / self.config.critical_range_m,
                    0.0,
                    1.0,
                )
            )
            severity = range_factor * (0.40 + 0.35 * closing_factor + 0.25 * miss_factor)
            if targeted:
                severity = max(severity, 0.05) * self.config.targeted_multiplier
            if seeker_locked:
                severity = max(severity, 0.15) * self.config.seeker_lock_multiplier
            if severity <= 0.0:
                continue
            threats.append(
                BlueThreatAssessment(
                    red_index=red_index,
                    range_m=distance,
                    closing_speed_mps=closing_speed,
                    time_to_closest_s=time_to_closest,
                    predicted_miss_m=predicted_miss,
                    targeted=targeted,
                    seeker_locked=seeker_locked,
                    severity=float(severity),
                )
            )
        return threats

    def _score_action(
        self,
        state: EngagementState,
        blue_index: int,
        threats: list[BlueThreatAssessment],
        action_index: int,
        previous_index: int,
    ) -> float:
        blue = state.blue[blue_index]
        command = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[action_index]
        predicted_blue = self._propagate_blue(blue, command)
        decision_s = self.config.decision_interval_s
        acceleration = (predicted_blue.velocity_mps - blue.velocity_mps) / decision_s
        max_acceleration = self.environment.aircraft.max_load_factor_g * G0
        threat_weight = sum(item.severity for item in threats)
        evasion_utility = 0.0

        for threat in threats:
            red = state.red[threat.red_index]
            predicted_red_position = red.position_m + red.velocity_mps * decision_s
            relative_position = predicted_red_position - predicted_blue.position_m
            distance = max(norm(relative_position), EPS)
            los_unit = relative_position / distance
            relative_velocity = red.velocity_mps - predicted_blue.velocity_mps
            relative_speed_squared = float(np.dot(relative_velocity, relative_velocity))
            time_to_closest = 0.0
            if relative_speed_squared > EPS:
                time_to_closest = float(
                    np.clip(
                        -np.dot(relative_position, relative_velocity) / relative_speed_squared,
                        0.0,
                        self.config.lookahead_s,
                    )
                )
            closest_vector = relative_position + relative_velocity * time_to_closest
            predicted_miss = norm(closest_vector)
            end_range = norm(relative_position + relative_velocity * self.config.lookahead_s)
            transverse_velocity = relative_velocity - np.dot(relative_velocity, los_unit) * los_unit
            transverse_acceleration = acceleration - np.dot(acceleration, los_unit) * los_unit
            lateral_score = norm(transverse_acceleration) / max(max_acceleration, EPS)
            away_score = float(np.dot(acceleration, -los_unit) / max(max_acceleration, EPS))
            miss_score = float(np.clip(predicted_miss / self.config.critical_range_m, 0.0, 2.0))
            range_score = float(np.clip(end_range / self.config.detection_range_m, 0.0, 2.0))
            transverse_score = float(np.clip(norm(transverse_velocity) / 600.0, 0.0, 2.0))
            local_utility = (
                0.35 * miss_score
                + 0.15 * range_score
                + 0.40 * lateral_score
                + 0.20 * away_score
                + 0.10 * transverse_score
            )
            evasion_utility += threat.severity * local_utility

        evasion_utility /= max(threat_weight, EPS)
        urgency = float(np.clip(max(item.severity for item in threats), 0.0, 1.0))
        axial_load_g, normal_load_g, bank_rad = command
        effort = (
            abs(axial_load_g) / self.environment.aircraft.max_load_factor_g
            + abs(normal_load_g - 1.0) / max(self.environment.aircraft.max_load_factor_g - 1.0, 1.0)
            + abs(bank_rad) / math.pi
        ) / 3.0
        vertical_acceleration = abs(float(acceleration[UP_AXIS])) / max(max_acceleration, EPS)
        boundary_penalty = self._boundary_penalty(predicted_blue)
        change_penalty = self.config.switch_penalty if action_index != previous_index else 0.0
        return float(
            urgency * evasion_utility
            - self.config.effort_penalty * effort
            - 0.20 * vertical_acceleration
            - boundary_penalty
            - change_penalty
        )

    def _propagate_blue(self, blue: ThreeDoFState, command: np.ndarray) -> ThreeDoFState:
        predicted = blue.copy()
        for _ in range(self.decision_steps):
            predicted = self._physics._step_aircraft(predicted, command)
        return predicted

    def _boundary_penalty(self, blue: ThreeDoFState) -> float:
        aircraft = self.environment.aircraft
        altitude_forecast = (
            float(blue.position_m[UP_AXIS])
            + float(blue.velocity_mps[UP_AXIS]) * self.config.altitude_prediction_s
        )
        lower_guard = aircraft.min_altitude_m + self.config.altitude_margin_m
        upper_guard = aircraft.max_altitude_m - self.config.altitude_margin_m
        altitude_scale = max(self.config.altitude_margin_m, 1.0)
        altitude_penalty = (
            max(0.0, lower_guard - altitude_forecast)
            + max(0.0, altitude_forecast - upper_guard)
        ) / altitude_scale

        speed = norm(blue.velocity_mps)
        lower_speed = aircraft.min_speed_mps + self.config.speed_margin_mps
        upper_speed = aircraft.max_speed_mps - self.config.speed_margin_mps
        speed_scale = max(self.config.speed_margin_mps, 1.0)
        speed_penalty = (
            max(0.0, lower_speed - speed)
            + max(0.0, speed - upper_speed)
        ) / speed_scale
        return float(4.0 * altitude_penalty + 2.0 * speed_penalty)

    def _recovery_or_cruise(self, blue: ThreeDoFState) -> tuple[int, str]:
        aircraft = self.environment.aircraft
        altitude = float(blue.position_m[UP_AXIS])
        vertical_speed = float(blue.velocity_mps[UP_AXIS])
        speed = norm(blue.velocity_mps)
        if altitude <= aircraft.min_altitude_m + self.config.altitude_margin_m and vertical_speed <= 0.0:
            return 11, "recover"
        if altitude >= aircraft.max_altitude_m - self.config.altitude_margin_m and vertical_speed >= 0.0:
            return 14, "recover"
        if speed <= aircraft.min_speed_mps + self.config.speed_margin_mps:
            return 1, "recover"
        if speed >= aircraft.max_speed_mps - self.config.speed_margin_mps:
            return 3, "recover"
        return 0, "cruise"


class BlueEvasionController:
    """Cache one rule-machine decision for exactly one blue decision period."""

    def __init__(self, rule_machine: BlueEvasionRuleMachine) -> None:
        self.rule_machine = rule_machine
        self.decision_steps = rule_machine.decision_steps
        self.last_decision: BlueEvasionDecision | None = None
        self._next_decision_step: int | None = None
        self._last_seen_step: int | None = None

    def reset(self) -> None:
        self.rule_machine.reset()
        self.last_decision = None
        self._next_decision_step = None
        self._last_seen_step = None

    def action_for(
        self,
        state: EngagementState,
    ) -> tuple[dict[str, np.ndarray], BlueEvasionDecision | None]:
        if self._last_seen_step is not None and state.step_count < self._last_seen_step:
            self.reset()
        self._last_seen_step = int(state.step_count)
        updated: BlueEvasionDecision | None = None
        if self.last_decision is None or (
            self._next_decision_step is not None and state.step_count >= self._next_decision_step
        ):
            updated = self.rule_machine.decide(state)
            self.last_decision = updated
            self._next_decision_step = state.step_count + self.decision_steps
        assert self.last_decision is not None
        return self.last_decision.action(), updated

    def __call__(self, state: EngagementState) -> dict[str, np.ndarray]:
        action, _ = self.action_for(state)
        return action
