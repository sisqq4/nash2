from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from .math_utils import EPS, G0, UP_AXIS, norm
from .physics import ThreeDoFPhysicsLayer
from .types import EngagementState, EnvironmentConfig, ThreeDoFState


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
    ) -> None:
        environment.validate()
        config.validate(environment)
        self.environment = environment
        self.config = config
        self.decision_steps = int(round(config.decision_interval_s / environment.time_step_s))
        self._physics = ThreeDoFPhysicsLayer(environment)
        self._previous_indices: np.ndarray | None = None

    def reset(self) -> None:
        self._previous_indices = None

    def decide(self, state: EngagementState) -> BlueEvasionDecision:
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

            primary = max(threats, key=lambda item: (item.severity, -item.range_m, -item.red_index))
            primary_indices[blue_index] = primary.red_index
            primary_ranges[blue_index] = primary.range_m
            primary_closing[blue_index] = primary.closing_speed_mps
            scores = np.array(
                [
                    self._score_action(
                        state,
                        blue_index,
                        threats,
                        action_index,
                        int(previous[blue_index]),
                    )
                    for action_index in range(len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G))
                ],
                dtype=np.float64,
            )
            action_index = int(np.argmax(scores))
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
