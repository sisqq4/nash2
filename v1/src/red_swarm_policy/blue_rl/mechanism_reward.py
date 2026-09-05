"""State estimator and bounded mechanism rewards for blue-aircraft RL.

This module deliberately contains no policy values.  It derives the four
mechanisms only from physical state and the action that was actually applied:
threat outcome, maneuver timing, maneuver direction, and overload matching.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from ..env.math_utils import G0
from .flight_envelope import FlightEnvelopeConstraintLayer


_EPS = 1.0e-9
_UP = np.array([0.0, 1.0, 0.0], dtype=np.float64)


@dataclass(frozen=True)
class MechanismRewardConfig:
    """Numerical contract for the four training-time mechanisms."""

    enabled: bool = True
    threat_potential_scale: float = 1.0
    threat_filter_time_constant_s: float = 0.2
    threat_range_scale_m: float = 20_000.0
    threat_closing_scale_mps: float = 1_200.0
    threat_tgo_scale_s: float = 8.0
    threat_los_rate_scale_rad_s: float = 0.02
    threat_range_weight: float = 0.15
    threat_closing_weight: float = 0.20
    threat_tgo_weight: float = 0.30
    threat_collision_weight: float = 0.20
    threat_energy_weight: float = 0.10
    threat_guidance_weight: float = 0.05
    threat_mean_weight: float = 0.25
    threat_encirclement_weight: float = 0.35
    phase_on: float = 1.25
    phase_off: float = 0.75
    phase_on_confirmations: int = 3
    phase_off_confirmations: int = 6
    minimum_phase_residence_s: float = 0.5
    emergency_tgo_s: float = 5.0
    release_time_constant_s: float = 0.5
    threat_rate_scale_per_s: float = 0.5
    phase_threat_rate_weight: float = 0.20
    main_threat_margin: float = 0.08
    main_threat_confirmations: int = 3
    legacy_emergency_suppression: float = 0.75
    timing_penalty_budget: float = 0.60
    direction_penalty_budget: float = 0.45
    overload_penalty_budget: float = 0.55
    reward_horizon_s: float = 200.0
    minimum_remaining_horizon_s: float = 60.0
    direction_dead_zone_deg: float = 20.0
    direction_activation_low_g: float = 0.30
    direction_activation_high_g: float = 1.00
    overload_dead_zone_g: float = 0.50
    overload_full_error_g: float = 3.00
    overload_direction_tolerance: float = 0.15
    soft_horizontal_ratio: float = 0.70
    hard_horizontal_ratio: float = 0.35
    soft_flight_path_angle_deg: float = 45.0
    hard_flight_path_angle_deg: float = 70.0
    speed_margin_width_mps: float = 100.0

    def validate(self) -> None:
        scalar = tuple(
            float(value)
            for name, value in self.__dict__.items()
            if name not in {"enabled", "phase_on_confirmations", "phase_off_confirmations",
                            "main_threat_confirmations"}
        )
        if not all(math.isfinite(value) for value in scalar):
            raise ValueError("mechanism reward parameters must be finite")
        if min(
            self.threat_potential_scale,
            self.threat_filter_time_constant_s,
            self.threat_range_scale_m,
            self.threat_closing_scale_mps,
            self.threat_tgo_scale_s,
            self.threat_los_rate_scale_rad_s,
            self.emergency_tgo_s,
            self.reward_horizon_s,
            self.minimum_remaining_horizon_s,
            self.release_time_constant_s,
            self.threat_rate_scale_per_s,
            self.overload_full_error_g,
            self.speed_margin_width_mps,
        ) <= 0.0:
            raise ValueError("mechanism scales and horizons must be positive")
        weights = (
            self.threat_range_weight, self.threat_closing_weight,
            self.threat_tgo_weight, self.threat_collision_weight,
            self.threat_energy_weight, self.threat_guidance_weight,
        )
        if min(weights) < 0.0 or not math.isclose(sum(weights), 1.0, abs_tol=1.0e-9):
            raise ValueError("local threat weights must be non-negative and sum to one")
        if min(self.threat_mean_weight, self.threat_encirclement_weight,
               self.timing_penalty_budget, self.direction_penalty_budget,
               self.overload_penalty_budget) < 0.0:
            raise ValueError("mechanism weights and penalty budgets must be non-negative")
        if not 0.0 <= self.legacy_emergency_suppression <= 1.0:
            raise ValueError("legacy emergency suppression must be in [0, 1]")
        if self.phase_on <= self.phase_off:
            raise ValueError("phase_on must exceed phase_off")
        if self.phase_off < 0.0 or self.minimum_phase_residence_s < 0.0:
            raise ValueError("phase thresholds and residence time must be non-negative")
        if self.main_threat_margin < 0.0:
            raise ValueError("main-threat switch margin must be non-negative")
        if min(self.phase_on_confirmations, self.phase_off_confirmations,
               self.main_threat_confirmations) < 1:
            raise ValueError("mechanism confirmation counts must be positive")
        if not 0.0 <= self.direction_activation_low_g < self.direction_activation_high_g:
            raise ValueError("direction activation thresholds must satisfy 0 <= low < high")
        if not 0.0 <= self.direction_dead_zone_deg < 90.0:
            raise ValueError("direction dead zone must be in [0, 90) degrees")
        if not 0.0 < self.hard_horizontal_ratio <= self.soft_horizontal_ratio <= 1.0:
            raise ValueError("horizontal ratios must satisfy 0 < hard <= soft <= 1")
        if not 0.0 < self.soft_flight_path_angle_deg <= self.hard_flight_path_angle_deg < 90.0:
            raise ValueError("flight-path limits must satisfy 0 < soft <= hard < 90")
        if self.overload_full_error_g <= self.overload_dead_zone_g:
            raise ValueError("overload full error must exceed its dead zone")
        if self.overload_dead_zone_g < 0.0 or not 0.0 <= self.overload_direction_tolerance <= 2.0:
            raise ValueError("overload dead zone or direction tolerance is invalid")
        if self.minimum_remaining_horizon_s > self.reward_horizon_s:
            raise ValueError("minimum remaining horizon cannot exceed reward horizon")


def mechanism_observation_dim(missile_slots: int) -> int:
    """Dimension of the normalized-v4 mechanism portion (without action context)."""
    slots = int(missile_slots)
    if slots < 1:
        raise ValueError("missile_slots must be positive")
    # 13 values per missile, then 21 fixed global values and two slot one-hots.
    return 13 * slots + 21 + 2 * slots


def _smoothstep(value: float, low: float, high: float) -> float:
    if high <= low:
        return float(value >= high)
    x = float(np.clip((value - low) / (high - low), 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _guidance_strength(mode: str) -> float:
    return {"locked": 1.0, "lock_hold": 0.70, "inertial": 0.25,
            "boost": 0.25}.get(str(mode), 0.0)


def encode_normalized_v4(
    snapshot: dict[str, object],
    mechanism: dict[str, object],
    missile_slots: int,
) -> list[float]:
    """Encode physical and estimator state needed to keep normalized-v4 Markov."""
    blue_p = np.asarray(snapshot["blue_position_m"], dtype=np.float64)
    blue_v = np.asarray(snapshot["blue_velocity_mps"], dtype=np.float64)
    red_p = np.asarray(snapshot["red_positions_m"], dtype=np.float64)
    red_v = np.asarray(snapshot["red_velocities_mps"], dtype=np.float64)
    alive = np.asarray(snapshot["red_alive"], dtype=bool)
    energy = np.asarray(snapshot.get("red_energy", np.zeros(len(red_p))), dtype=np.float64)
    modes = list(snapshot.get("red_guidance_modes", ["inertial"] * len(red_p)))
    values: list[float] = []
    for slot in range(int(missile_slots)):
        if slot >= len(red_p) or not alive[slot]:
            values.extend([0.0] * 13)
            continue
        rel = red_p[slot] - blue_p
        rel_v = red_v[slot] - blue_v
        distance = max(float(np.linalg.norm(rel)), 1.0)
        closing = max(0.0, -float(np.dot(rel, rel_v)) / distance)
        tgo = distance / max(closing, 1.0)
        los_rate = float(np.linalg.norm(np.cross(rel, rel_v)) / distance ** 2)
        values.extend([
            *(rel / 200_000.0).tolist(),
            *(rel_v / 2_000.0).tolist(),
            float(np.clip(distance / 60_000.0, 0.0, 2.0)),
            float(np.clip(closing / 1_200.0, 0.0, 2.0)),
            float(np.clip(tgo / 30.0, 0.0, 2.0)),
            float(np.clip(los_rate / 0.05, 0.0, 2.0)),
            float(np.clip(energy[slot], 0.0, 1.5)),
            _guidance_strength(modes[slot]),
            1.0,
        ])

    phase = str(mechanism.get("phase", "P0"))
    phase_one_hot = [float(phase == name) for name in ("P0", "P1", "P2")]
    primary = int(mechanism.get("primary_slot", -1))
    candidate = int(mechanism.get("primary_candidate_slot", -1))
    primary_one_hot = [float(slot == primary) for slot in range(int(missile_slots))]
    candidate_one_hot = [float(slot == candidate) for slot in range(int(missile_slots))]
    desired_body = np.asarray(mechanism.get("desired_direction_body", [0.0, 0.0, 0.0]),
                              dtype=np.float64)
    values.extend([
        float(mechanism.get("total_threat", 0.0)) / 2.0,
        float(np.clip(float(mechanism.get("threat_rate_per_s", 0.0)) / 0.5, -2.0, 2.0)),
        float(mechanism.get("C_ang", 0.0)),
        float(mechanism.get("C_syn", 0.0)),
        float(mechanism.get("C_cor", 0.0)),
        float(mechanism.get("C_enc", 0.0)),
        float(mechanism.get("W_safe", 1.0)),
        *phase_one_hot,
        float(np.clip(float(mechanism.get("phase_age_s", 0.0)) / 2.0, 0.0, 2.0)),
        float(np.clip(float(mechanism.get("phase_on_count", 0)) / 3.0, 0.0, 2.0)),
        float(np.clip(float(mechanism.get("phase_off_count", 0)) / 6.0, 0.0, 2.0)),
        *primary_one_hot,
        *candidate_one_hot,
        float(np.clip(float(mechanism.get("primary_candidate_count", 0)) / 3.0, 0.0, 1.0)),
        *desired_body.tolist(),
        float(mechanism.get("reference_load_g", 1.0)) / 9.0,
        float(mechanism.get("emergency_gate", 0.0)),
        float(mechanism.get("envelope_margin", 0.0)),
        float(mechanism.get("evasion_target", 0.0)),
    ])
    expected = mechanism_observation_dim(missile_slots)
    if len(values) != expected:
        raise RuntimeError(f"normalized-v4 mechanism encoder produced {len(values)}, expected {expected}")
    return values


class BlueMechanismStateEstimator:
    """Stateful physical estimator shared by environment and deployment controller."""

    def __init__(self, config: MechanismRewardConfig = MechanismRewardConfig()) -> None:
        config.validate()
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = "P0"
        self.phase_since_s = 0.0
        self.on_count = 0
        self.off_count = 0
        self.primary_slot: int | None = None
        self.primary_candidate: int | None = None
        self.primary_candidate_count = 0
        self.previous_direction: np.ndarray | None = None
        self.previous_time_s: float | None = None
        self.filtered_threat: float | None = None

    @staticmethod
    def _body_axes(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        speed = float(np.linalg.norm(velocity))
        forward = velocity / speed if speed > _EPS else np.array([1.0, 0.0, 0.0])
        local_up = _UP - float(np.dot(_UP, forward)) * forward
        if np.linalg.norm(local_up) <= _EPS:
            local_up = np.array([0.0, 0.0, 1.0])
            local_up -= float(np.dot(local_up, forward)) * forward
        local_up /= max(float(np.linalg.norm(local_up)), _EPS)
        right = np.cross(forward, local_up)
        right /= max(float(np.linalg.norm(right)), _EPS)
        return forward, right, local_up

    @staticmethod
    def _action_accelerations(velocity: np.ndarray) -> np.ndarray:
        velocities = np.repeat(np.asarray(velocity, dtype=np.float64)[None, :],
                               len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G), axis=0)
        return FlightEnvelopeConstraintLayer._accelerations_for_commands(
            velocities, BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
        )

    def _primary_threat(self, priorities: np.ndarray, slots: np.ndarray) -> int:
        best_local = int(np.argmax(priorities))
        best_slot = int(slots[best_local])
        if self.primary_slot is None or self.primary_slot not in slots:
            self.primary_slot = best_slot
            self.primary_candidate = None
            self.primary_candidate_count = 0
            return best_local
        current_local = int(np.flatnonzero(slots == self.primary_slot)[0])
        if (best_slot != self.primary_slot
                and priorities[best_local] > priorities[current_local] + self.config.main_threat_margin):
            if self.primary_candidate == best_slot:
                self.primary_candidate_count += 1
            else:
                self.primary_candidate = best_slot
                self.primary_candidate_count = 1
            if self.primary_candidate_count >= self.config.main_threat_confirmations:
                self.primary_slot = best_slot
                self.primary_candidate = None
                self.primary_candidate_count = 0
                return best_local
        else:
            self.primary_candidate = None
            self.primary_candidate_count = 0
        return current_local

    def _assess(self, snapshot: dict[str, object]) -> dict[str, Any]:
        blue_p = np.asarray(snapshot["blue_position_m"], dtype=np.float64)
        blue_v = np.asarray(snapshot["blue_velocity_mps"], dtype=np.float64)
        red_p_all = np.asarray(snapshot["red_positions_m"], dtype=np.float64)
        red_v_all = np.asarray(snapshot["red_velocities_mps"], dtype=np.float64)
        alive = np.asarray(snapshot["red_alive"], dtype=bool)
        slots = np.flatnonzero(alive)
        if not len(slots):
            return {
                "blue_p": blue_p, "blue_v": blue_v, "slots": slots,
                "rel": np.empty((0, 3)), "rel_v": np.empty((0, 3)),
                "ranges": np.empty(0), "closing": np.empty(0), "tgo": np.empty(0),
                "los_rate": np.empty(0), "local": np.empty(0), "raw_total": 0.0,
                "C_ang": 0.0, "C_syn": 0.0, "C_cor": 0.0,
                "C_enc": 0.0, "W_safe": 1.0, "primary": None,
            }
        red_p = red_p_all[alive]
        red_v = red_v_all[alive]
        rel = red_p - blue_p
        rel_v = red_v - blue_v
        ranges = np.linalg.norm(rel, axis=1).clip(1.0)
        closing = np.maximum(0.0, -np.sum(rel * rel_v, axis=1) / ranges)
        tgo = ranges / np.maximum(closing, 1.0)
        los_rate = np.linalg.norm(np.cross(rel, rel_v), axis=1) / ranges ** 2
        energy_all = np.asarray(snapshot.get("red_energy", np.ones(len(red_p_all))),
                                dtype=np.float64)
        energy = np.clip(energy_all[alive], 0.0, 1.0)
        modes = list(snapshot.get("red_guidance_modes", ["locked"] * len(red_p_all)))
        guidance = np.asarray([_guidance_strength(modes[index]) for index in slots])
        c = self.config
        local = np.clip(
            c.threat_range_weight * np.exp(-ranges / c.threat_range_scale_m)
            + c.threat_closing_weight * np.clip(closing / c.threat_closing_scale_mps, 0.0, 1.0)
            + c.threat_tgo_weight * np.exp(-tgo / c.threat_tgo_scale_s)
            + c.threat_collision_weight
            * np.exp(-(los_rate / c.threat_los_rate_scale_rad_s) ** 2)
            + c.threat_energy_weight * energy
            + c.threat_guidance_weight * guidance,
            0.0, 1.0,
        )
        bearings = np.mod(np.arctan2(rel[:, 2], rel[:, 0]), 2.0 * math.pi)
        if len(bearings) > 1:
            ordered = np.sort(bearings)
            gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * math.pi])
            angular_coverage = 1.0 - float(np.max(gaps)) / (2.0 * math.pi)
            synchronization = math.exp(-float(np.var(np.clip(tgo, 0.0, 120.0))) / 25.0)
        else:
            angular_coverage = synchronization = 0.0
        candidate_angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        candidate_dirs = np.stack((np.cos(candidate_angles), np.zeros(24),
                                   np.sin(candidate_angles)), axis=1)
        los = rel / ranges[:, None]
        clearance = np.min(1.0 - np.abs(candidate_dirs @ los.T), axis=1)
        safe_width = float(np.max(clearance))
        corridor_compression = 1.0 - safe_width
        encirclement = float(np.clip(
            (angular_coverage + synchronization + corridor_compression) / 3.0, 0.0, 1.0
        ))
        raw_total = float(np.clip(
            np.max(local) + c.threat_mean_weight * np.mean(local)
            + c.threat_encirclement_weight * encirclement,
            0.0, 2.0,
        ))
        priority = local + 0.15 / np.maximum(tgo, 0.2) + 0.10 * guidance
        primary = self._primary_threat(priority, slots)
        return {
            "blue_p": blue_p, "blue_v": blue_v, "slots": slots,
            "rel": rel, "rel_v": rel_v, "ranges": ranges, "closing": closing,
            "tgo": tgo, "los_rate": los_rate, "local": local,
            "raw_total": raw_total, "C_ang": angular_coverage,
            "C_syn": synchronization, "C_cor": corridor_compression,
            "C_enc": encirclement, "W_safe": safe_width, "primary": primary,
        }

    def _update_phase(self, threat: dict[str, Any], now_s: float) -> tuple[float, float, bool]:
        dt = 0.0 if self.previous_time_s is None else max(0.0, now_s - self.previous_time_s)
        previous = threat["raw_total"] if self.filtered_threat is None else self.filtered_threat
        if self.filtered_threat is None:
            filtered = float(threat["raw_total"])
        else:
            alpha = 1.0 - math.exp(-dt / self.config.threat_filter_time_constant_s) if dt else 0.0
            filtered = float(self.filtered_threat + alpha * (threat["raw_total"] - self.filtered_threat))
        threat_rate = (filtered - previous) / max(dt, 1.0e-3) if dt else 0.0
        self.phase_since_s += dt
        normalized_rise = float(np.clip(threat_rate / self.config.threat_rate_scale_per_s, 0.0, 1.0))
        criterion = filtered + self.config.phase_threat_rate_weight * normalized_rise
        emergency = bool(len(threat["tgo"]) and np.min(threat["tgo"]) < self.config.emergency_tgo_s)
        self.on_count = self.on_count + 1 if criterion >= self.config.phase_on else 0
        self.off_count = self.off_count + 1 if criterion <= self.config.phase_off else 0
        resident = self.phase_since_s >= self.config.minimum_phase_residence_s
        old = self.phase
        if emergency or (self.phase != "P1" and resident
                         and self.on_count >= self.config.phase_on_confirmations):
            self.phase = "P1"
        elif (self.phase == "P1" and resident
              and self.off_count >= self.config.phase_off_confirmations):
            self.phase = "P2"
        elif (self.phase == "P2" and resident
              and self.off_count >= 2 * self.config.phase_off_confirmations):
            self.phase = "P0"
        if self.phase != old:
            self.phase_since_s = 0.0
            self.on_count = self.off_count = 0
        self.previous_time_s = now_s
        self.filtered_threat = filtered
        return criterion, threat_rate, self.phase != old

    def _desired_direction(self, threat: dict[str, Any], snapshot: dict[str, object],
                           action_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        velocity = threat["blue_v"]
        accelerations = self._action_accelerations(velocity)
        if threat["primary"] is None:
            return np.zeros(3), accelerations
        primary = int(threat["primary"])
        rel = threat["rel"][primary]
        rel_v = threat["rel_v"][primary]
        radial = rel / max(float(np.linalg.norm(rel)), _EPS)
        angular_momentum = np.cross(rel, rel_v)
        if np.linalg.norm(angular_momentum) <= 1.0e-6:
            angular_momentum = np.cross(radial, _UP)
        if np.linalg.norm(angular_momentum) <= 1.0e-6:
            angular_momentum = np.array([0.0, 0.0, 1.0])
        normal = angular_momentum / np.linalg.norm(angular_momentum)
        tangent = np.cross(normal, radial)
        tangent /= max(float(np.linalg.norm(tangent)), _EPS)
        candidates = [math.cos(phi) * tangent + math.sin(phi) * normal
                      for phi in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)]
        los = threat["rel"] / threat["ranges"][:, None]
        altitude = float(threat["blue_p"][1])
        min_alt = float(snapshot.get("min_altitude_m", 0.0))
        max_alt = float(snapshot.get("max_altitude_m", 1.0e9))
        best_score = -math.inf
        desired = candidates[0]
        for direction in candidates:
            angular_gain = min(float(np.linalg.norm(np.cross(item, direction))) for item in los)
            closing_gain = float(np.mean(-los @ direction))
            synchronization_gain = float(np.var(
                np.clip(threat["tgo"], 0.0, 120.0) + 2.0 * (los @ direction)
            ))
            boundary_risk = max(
                0.0,
                direction[1] * (altitude - max_alt + 500.0) / 500.0,
                -direction[1] * (min_alt + 500.0 - altitude) / 500.0,
            )
            continuity = (0.0 if self.previous_direction is None
                          else float(np.dot(direction, self.previous_direction)))
            score = (angular_gain + 0.25 * closing_gain + 0.05 * synchronization_gain
                     + 0.15 * continuity - boundary_risk)
            if score > best_score:
                best_score, desired = score, direction

        feasible = np.ones(len(accelerations), dtype=bool) if action_mask is None else np.asarray(
            action_mask, dtype=bool
        ).copy()
        if feasible.shape != (len(accelerations),):
            raise ValueError("action_mask must contain all blue actions")
        feasible &= np.linalg.norm(accelerations, axis=1) > 0.05 * G0
        if feasible.any():
            unit = accelerations / np.maximum(np.linalg.norm(accelerations, axis=1)[:, None], _EPS)
            projected_action = int(np.flatnonzero(feasible)[np.argmax(unit[feasible] @ desired)])
            desired = unit[projected_action]
        self.previous_direction = desired.copy()
        return desired, accelerations

    def _envelope_margin(self, snapshot: dict[str, object]) -> float:
        velocity = np.asarray(snapshot["blue_velocity_mps"], dtype=np.float64)
        speed = float(np.linalg.norm(velocity))
        horizontal = float(np.hypot(velocity[0], velocity[2]))
        ratio = horizontal / max(speed, _EPS)
        fpa = abs(math.degrees(math.atan2(float(velocity[1]), horizontal)))
        min_speed = float(snapshot.get("min_speed_mps", 100.0))
        max_speed = float(snapshot.get("max_speed_mps", 600.0))
        width = min(self.config.speed_margin_width_mps, max((max_speed - min_speed) / 2.0, 1.0))
        speed_margin = (_smoothstep(speed, min_speed, min_speed + width)
                        * (1.0 - _smoothstep(speed, max_speed - width, max_speed)))
        horizontal_margin = _smoothstep(
            ratio, self.config.hard_horizontal_ratio, self.config.soft_horizontal_ratio
        )
        fpa_margin = 1.0 - _smoothstep(
            fpa, self.config.soft_flight_path_angle_deg, self.config.hard_flight_path_angle_deg
        )
        return float(np.clip(speed_margin * horizontal_margin * fpa_margin, 0.0, 1.0))

    def _reference_load(self, threat: dict[str, Any], total: float, envelope_margin: float,
                        desired: np.ndarray, accelerations: np.ndarray,
                        action_mask: np.ndarray | None) -> tuple[float, float]:
        minimum_tgo = float(np.min(threat["tgo"])) if len(threat["tgo"]) else math.inf
        urgency = 1.0 / (1.0 + math.exp(-float(np.clip(
            4.0 * (total + 2.0 / max(minimum_tgo, 0.2) - 1.0), -60.0, 60.0
        ))))
        available = float(np.clip(float(np.asarray(
            [np.linalg.norm(command[:2]) for command in BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G]
        ).max()), 1.0, float("inf")))
        available = min(available, float(threat.get("available_load_g", available)))
        continuous = 1.0 + (available - 1.0) * envelope_margin * urgency
        mask = np.ones(len(accelerations), dtype=bool) if action_mask is None else np.asarray(
            action_mask, dtype=bool
        ).copy()
        loads = np.linalg.norm(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[:, :2], axis=1)
        mask &= loads <= available + 1.0e-9
        active = np.linalg.norm(accelerations, axis=1) > 0.05 * G0
        if np.linalg.norm(desired) > _EPS and np.any(mask & active):
            unit = accelerations / np.maximum(np.linalg.norm(accelerations, axis=1)[:, None], _EPS)
            alignment = unit @ desired
            best = float(np.max(alignment[mask & active]))
            mask &= active & (alignment >= best - self.config.overload_direction_tolerance)
        if not mask.any():
            return float(continuous), float(continuous)
        candidates = np.flatnonzero(mask)
        selected = int(candidates[np.argmin(np.abs(loads[candidates] - continuous))])
        return float(continuous), float(loads[selected])

    def observe(self, snapshot: dict[str, object],
                action_mask: np.ndarray | None = None) -> dict[str, object]:
        threat = self._assess(snapshot)
        threat["available_load_g"] = float(snapshot.get("max_load_factor_g", 9.0))
        now_s = float(snapshot.get("time_s", 0.0))
        criterion, threat_rate, changed = self._update_phase(threat, now_s)
        desired, accelerations = self._desired_direction(threat, snapshot, action_mask)
        envelope_margin = self._envelope_margin(snapshot)
        total = float(self.filtered_threat or 0.0)
        reference, reference_discrete = self._reference_load(
            threat, total, envelope_margin, desired, accelerations, action_mask
        )
        if self.phase == "P1":
            emergency_gate = evasion_target = 1.0
        elif self.phase == "P2":
            emergency_gate = evasion_target = math.exp(
                -self.phase_since_s / self.config.release_time_constant_s
            )
        else:
            emergency_gate = evasion_target = 0.0
        forward, right, local_up = self._body_axes(threat["blue_v"])
        desired_body = np.array([
            float(np.dot(desired, forward)),
            float(np.dot(desired, local_up)),
            float(np.dot(desired, right)),
        ])
        minimum_tgo = float(np.min(threat["tgo"])) if len(threat["tgo"]) else math.inf
        return {
            "phase": self.phase,
            "phase_changed": changed,
            "phase_age_s": float(self.phase_since_s),
            "phase_on_count": int(self.on_count),
            "phase_off_count": int(self.off_count),
            "phase_criterion": float(criterion),
            "raw_total_threat": float(threat["raw_total"]),
            "total_threat": total,
            "threat_rate_per_s": float(threat_rate),
            "local_threats": np.asarray(threat["local"]).tolist(),
            "minimum_tgo_s": None if not math.isfinite(minimum_tgo) else minimum_tgo,
            "C_ang": float(threat["C_ang"]),
            "C_syn": float(threat["C_syn"]),
            "C_cor": float(threat["C_cor"]),
            "C_enc": float(threat["C_enc"]),
            "W_safe": float(threat["W_safe"]),
            "primary_slot": -1 if self.primary_slot is None else int(self.primary_slot),
            "primary_candidate_slot": (-1 if self.primary_candidate is None
                                       else int(self.primary_candidate)),
            "primary_candidate_count": int(self.primary_candidate_count),
            "desired_direction_inertial": desired.tolist(),
            "desired_direction_body": desired_body.tolist(),
            "reference_load_continuous_g": reference,
            "reference_load_g": reference_discrete,
            "envelope_margin": envelope_margin,
            "emergency_gate": float(emergency_gate),
            "evasion_target": float(evasion_target),
        }

    def restrict_targets_to_action_mask(
        self,
        snapshot: dict[str, object],
        mechanism: dict[str, object],
        action_mask: np.ndarray,
    ) -> dict[str, object]:
        """Project stored direction/load targets onto the exact hard-safe set.

        This is intentionally stateless.  Behavior collection computes the
        expensive predictive mask outside the environment and passes it with
        the action; retargeting must not advance threat filters or phase
        confirmation counters a second time.
        """
        mask = np.asarray(action_mask, dtype=bool)
        count = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)
        if mask.shape != (count,):
            raise ValueError("action_mask must contain all blue actions")
        if not mask.any():
            return dict(mechanism)
        result = dict(mechanism)
        velocity = np.asarray(snapshot["blue_velocity_mps"], dtype=np.float64)
        accelerations = self._action_accelerations(velocity)
        norms = np.linalg.norm(accelerations, axis=1)
        active = mask & (norms > 0.05 * G0)
        desired = np.asarray(
            mechanism.get("desired_direction_inertial", [0.0, 0.0, 0.0]),
            dtype=np.float64,
        )
        unit = accelerations / np.maximum(norms[:, None], _EPS)
        if active.any() and np.linalg.norm(desired) > _EPS:
            selected = int(np.flatnonzero(active)[np.argmax(unit[active] @ desired)])
            desired = unit[selected]
            alignment = unit @ desired
            load_candidates = active & (
                alignment >= float(np.max(alignment[active]))
                - self.config.overload_direction_tolerance
            )
        else:
            load_candidates = mask.copy()
        loads = np.linalg.norm(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[:, :2], axis=1)
        available = float(snapshot.get("max_load_factor_g", 9.0))
        load_candidates &= loads <= available + 1.0e-9
        if load_candidates.any():
            candidates = np.flatnonzero(load_candidates)
            continuous = float(mechanism.get("reference_load_continuous_g", 1.0))
            selected = int(candidates[np.argmin(np.abs(loads[candidates] - continuous))])
            result["reference_load_g"] = float(loads[selected])
        forward, right, local_up = self._body_axes(velocity)
        result["desired_direction_inertial"] = desired.tolist()
        result["desired_direction_body"] = [
            float(np.dot(desired, forward)),
            float(np.dot(desired, local_up)),
            float(np.dot(desired, right)),
        ]
        return result

    def penalties(
        self,
        mechanism: dict[str, object],
        average_net_acceleration_mps2: np.ndarray,
        average_load_g: float,
        transition_dt_s: float,
        elapsed_learning_s: float,
        *,
        action_mask: np.ndarray | None = None,
        fallback_required: bool = False,
    ) -> dict[str, float]:
        """Return bounded positive costs for the actual transition."""
        if not self.config.enabled:
            return {"timing": 0.0, "direction": 0.0, "overload": 0.0,
                    "total": 0.0, "choice_gate": 0.0, "evasion_activation": 0.0,
                    "timing_error": 0.0, "direction_error": 0.0,
                    "overload_error": 0.0}
        if action_mask is None:
            choice_gate = 1.0
        else:
            mask = np.asarray(action_mask, dtype=bool)
            if mask.shape != (len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G),):
                raise ValueError("action_mask must contain all blue actions")
            choice_gate = float(np.count_nonzero(mask) > 1 and not fallback_required)
        acceleration = np.asarray(average_net_acceleration_mps2, dtype=np.float64)
        acceleration_g = float(np.linalg.norm(acceleration) / G0)
        activation = _smoothstep(
            acceleration_g, self.config.direction_activation_low_g,
            self.config.direction_activation_high_g,
        )
        target = float(mechanism.get("evasion_target", 0.0))
        timing_error = abs(activation - target) * (1.0 if activation < target else 0.5)
        desired = np.asarray(mechanism.get("desired_direction_inertial", [0.0, 0.0, 0.0]),
                             dtype=np.float64)
        if acceleration_g <= _EPS or np.linalg.norm(desired) <= _EPS:
            cosine = 1.0 if target <= _EPS else -1.0
        else:
            cosine = float(np.clip(
                np.dot(acceleration, desired) / (np.linalg.norm(acceleration) * np.linalg.norm(desired)),
                -1.0, 1.0,
            ))
        dead_cosine = math.cos(math.radians(self.config.direction_dead_zone_deg))
        direction_error = max(0.0, (dead_cosine - cosine) / (1.0 + dead_cosine)) ** 2
        load_error_g = abs(float(average_load_g) - float(mechanism.get("reference_load_g", 1.0)))
        overload_error = float(np.clip(
            (load_error_g - self.config.overload_dead_zone_g)
            / (self.config.overload_full_error_g - self.config.overload_dead_zone_g),
            0.0, 1.0,
        ) ** 2)
        horizon = max(
            self.config.minimum_remaining_horizon_s,
            self.config.reward_horizon_s - max(0.0, float(elapsed_learning_s)),
        )
        scale = max(0.0, float(transition_dt_s)) / horizon * choice_gate
        timing = self.config.timing_penalty_budget * scale * timing_error
        direction = (self.config.direction_penalty_budget * scale * activation * target
                     * direction_error)
        overload = (self.config.overload_penalty_budget * scale * activation * target
                    * overload_error)
        return {
            "timing": float(timing), "direction": float(direction),
            "overload": float(overload), "total": float(timing + direction + overload),
            "choice_gate": choice_gate, "evasion_activation": activation,
            "timing_error": float(timing_error), "direction_error": float(direction_error),
            "overload_error": overload_error,
        }
