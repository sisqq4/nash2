"""Deterministic, evaluation-only mechanism shaping for the blue Rainbow policy."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G

_EPS = 1.0e-9
_UP = np.array([0.0, 1.0, 0.0])


@dataclass(frozen=True)
class EvaluationShapingConfig:
    """Ablation switches and bounded arbitration parameters.

    These options are intentionally not part of ``BlueEscapeEnvConfig`` so they
    cannot silently enter training or replay collection.
    """

    threat: bool = False
    timing: bool = False
    direction: bool = False
    overload: bool = False
    weight: float = 0.35
    threat_weight: float = 1.0
    timing_weight: float = 1.0
    direction_weight: float = 1.0
    overload_weight: float = 1.0
    prediction_s: float = 0.5
    main_threat_margin: float = 0.08
    main_threat_confirmations: int = 3
    phase_on: float = 1.25
    phase_off: float = 0.75
    phase_confirmations: int = 3
    minimum_phase_residence_s: float = 0.5
    emergency_tgo_s: float = 5.0
    ground_prediction_s: float = 2.0
    load_change_penalty: float = 0.15
    switch_penalty: float = 0.08

    @property
    def enabled(self) -> bool:
        return self.threat or self.timing or self.direction or self.overload


class EvaluationActionShaper:
    """Score and arbitrate all 29 actions without altering the training path."""

    def __init__(self, config: EvaluationShapingConfig) -> None:
        scalar = (config.weight, config.threat_weight, config.timing_weight,
                  config.direction_weight, config.overload_weight,
                  config.prediction_s, config.main_threat_margin,
                  config.phase_on, config.phase_off, config.minimum_phase_residence_s,
                  config.emergency_tgo_s, config.ground_prediction_s,
                  config.load_change_penalty, config.switch_penalty)
        if not all(math.isfinite(value) for value in scalar):
            raise ValueError("mechanism configuration must be finite")
        if min(config.weight, config.threat_weight, config.timing_weight,
               config.direction_weight, config.overload_weight) < 0.0 or config.prediction_s <= 0.0 or config.phase_on <= config.phase_off:
            raise ValueError("mechanism weight, horizon, or phase thresholds are invalid")
        if config.main_threat_confirmations < 1 or config.phase_confirmations < 1:
            raise ValueError("mechanism confirmation counts must be positive")
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = "P0"
        self.previous_action = 0
        self.previous_load_g = 1.0
        self.previous_direction: np.ndarray | None = None
        self.primary_slot: int | None = None
        self._primary_candidate: int | None = None
        self._primary_count = 0
        self._on_count = self._off_count = 0
        self._phase_since_s = 0.0
        self._previous_time_s: float | None = None
        self._previous_threat = 0.0

    @staticmethod
    def _bounded(values: np.ndarray) -> np.ndarray:
        """Robustly map a score channel to [-1, 1], including tied actions."""
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values)
        if not finite.any():
            return np.zeros_like(values)
        replacement = float(np.median(values[finite]))
        clean = np.where(finite, values, replacement)
        low, high = np.percentile(clean, [10.0, 90.0])
        scale = max(float(high - low), float(np.ptp(clean)), 1.0e-9)
        return np.clip(2.0 * (clean - 0.5 * (high + low)) / scale, -1.0, 1.0)

    @staticmethod
    def _sigmoid(value: np.ndarray | float) -> np.ndarray | float:
        return 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))

    @staticmethod
    def _body_axes(velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        forward = velocity / max(float(np.linalg.norm(velocity)), _EPS)
        right = np.cross(_UP, forward)
        if np.linalg.norm(right) < 1.0e-7:
            right = np.array([0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        return forward, right, up

    def _accelerations(self, velocity: np.ndarray) -> np.ndarray:
        forward, right, up = self._body_axes(velocity)
        command = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
        normal = np.cos(command[:, 2, None]) * up + np.sin(command[:, 2, None]) * right
        return 9.80665 * (command[:, 0, None] * forward + command[:, 1, None] * normal)

    def _main_threat(self, scores: np.ndarray, slots: np.ndarray) -> int:
        best_local = int(np.argmax(scores)); best_slot = int(slots[best_local])
        if self.primary_slot is None or self.primary_slot not in slots:
            self.primary_slot = best_slot; self._primary_candidate = None; self._primary_count = 0
            return best_local
        current_local = int(np.flatnonzero(slots == self.primary_slot)[0])
        if best_slot != self.primary_slot and scores[best_local] > scores[current_local] + self.config.main_threat_margin:
            if self._primary_candidate == best_slot:
                self._primary_count += 1
            else:
                self._primary_candidate, self._primary_count = best_slot, 1
            if self._primary_count >= self.config.main_threat_confirmations:
                self.primary_slot = best_slot; self._primary_candidate = None; self._primary_count = 0
                return best_local
        else:
            self._primary_candidate = None; self._primary_count = 0
        return current_local

    def _threat_assessment(self, snapshot: dict[str, object]) -> dict[str, Any]:
        blue_p = np.asarray(snapshot["blue_position_m"], dtype=np.float64)
        blue_v = np.asarray(snapshot["blue_velocity_mps"], dtype=np.float64)
        red_p_all = np.asarray(snapshot["red_positions_m"], dtype=np.float64)
        red_v_all = np.asarray(snapshot["red_velocities_mps"], dtype=np.float64)
        alive = np.asarray(snapshot["red_alive"], dtype=bool)
        slots = np.flatnonzero(alive); red_p, red_v = red_p_all[alive], red_v_all[alive]
        rel, rel_v = red_p - blue_p, red_v - blue_v
        ranges = np.linalg.norm(rel, axis=1).clip(1.0)
        range_rate = np.sum(rel * rel_v, axis=1) / ranges
        closing = np.maximum(0.0, -range_rate)
        tgo = ranges / np.maximum(closing, 1.0)
        los_rate = np.linalg.norm(np.cross(rel, rel_v), axis=1) / ranges ** 2
        energy = np.asarray(snapshot.get("red_energy", np.ones(len(red_p_all))), dtype=np.float64)[alive]
        modes = list(snapshot.get("red_guidance_modes", ["locked"] * len(red_p_all)))
        guidance = np.asarray([1.0 if modes[index] in {"locked", "lock_hold"} else 0.25
                               for index in slots])
        local = self._sigmoid(3.0 * (20000.0 / ranges + closing / 1200.0 + 8.0 / np.maximum(tgo, 0.2)
                                     + los_rate / 0.02 + 0.25 * energy + 0.25 * guidance - 2.0))
        bearings = np.mod(np.arctan2(rel[:, 2], rel[:, 0]), 2.0 * math.pi)
        if len(bearings) > 1:
            ordered = np.sort(bearings)
            gaps = np.diff(np.r_[ordered, ordered[0] + 2.0 * math.pi])
            angular_coverage = 1.0 - float(np.max(gaps)) / (2.0 * math.pi)
            synchronization = math.exp(-float(np.var(tgo)) / 25.0)
        else:
            angular_coverage = synchronization = 0.0
        candidate_angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        candidate_dirs = np.stack((np.cos(candidate_angles), np.zeros(24), np.sin(candidate_angles)), axis=1)
        clearance = np.min(1.0 - np.abs(candidate_dirs @ (rel / ranges[:, None]).T), axis=1)
        safe_width = float(np.max(clearance))
        corridor_compression = 1.0 - safe_width
        encirclement = float(np.clip((angular_coverage + synchronization + corridor_compression) / 3.0, 0.0, 1.0))
        total = float(np.clip(np.max(local) + 0.25 * np.mean(local) + 0.35 * encirclement, 0.0, 2.0))
        priority = local + 0.15 / np.maximum(tgo, 0.2) + 0.1 * guidance
        primary = self._main_threat(priority, slots)
        return {"blue_p": blue_p, "blue_v": blue_v, "red_p": red_p, "red_v": red_v,
                "slots": slots, "rel": rel, "rel_v": rel_v, "ranges": ranges,
                "closing": closing, "tgo": tgo, "los_rate": los_rate, "local": local,
                "total": total, "angular_coverage": angular_coverage,
                "synchronization": synchronization, "corridor_compression": corridor_compression,
                "encirclement": encirclement, "safe_width": safe_width, "primary": primary}

    def _update_phase(self, threat: dict[str, Any], now_s: float) -> tuple[float, bool]:
        dt = 0.0 if self._previous_time_s is None else max(0.0, now_s - self._previous_time_s)
        self._phase_since_s += dt
        threat_rate = (threat["total"] - self._previous_threat) / max(dt, 1.0e-3) if dt else 0.0
        criterion = threat["total"] + 0.2 * max(0.0, threat_rate) + 0.35 * threat["encirclement"]
        emergency = float(np.min(threat["tgo"])) < self.config.emergency_tgo_s
        self._on_count = self._on_count + 1 if criterion >= self.config.phase_on else 0
        self._off_count = self._off_count + 1 if criterion <= self.config.phase_off else 0
        resident = self._phase_since_s >= self.config.minimum_phase_residence_s
        old = self.phase
        if emergency or (self.phase != "P1" and resident and self._on_count >= self.config.phase_confirmations):
            self.phase = "P1"
        elif self.phase == "P1" and resident and self._off_count >= self.config.phase_confirmations:
            self.phase = "P2"
        elif self.phase == "P2" and resident and self._off_count >= 2 * self.config.phase_confirmations:
            self.phase = "P0"
        if self.phase != old:
            self._phase_since_s = 0.0; self._on_count = self._off_count = 0
        self._previous_time_s, self._previous_threat = now_s, threat["total"]
        return criterion, self.phase != old

    def _desired_direction(self, threat: dict[str, Any], snapshot: dict[str, object]) -> np.ndarray:
        primary = threat["primary"]
        rel, rel_v = threat["rel"][primary], threat["rel_v"][primary]
        er = rel / max(float(np.linalg.norm(rel)), _EPS)
        cross = np.cross(rel, rel_v)
        if np.linalg.norm(cross) < 1.0e-6:
            if self.previous_direction is not None:
                return self.previous_direction
            cross = np.cross(er, _UP)
            if np.linalg.norm(cross) < 1.0e-6: cross = np.array([0.0, 0.0, 1.0])
        eh = cross / np.linalg.norm(cross); etheta = np.cross(eh, er); etheta /= np.linalg.norm(etheta)
        candidates = [math.cos(phi) * etheta + math.sin(phi) * eh
                      for phi in np.linspace(0.0, 2.0 * math.pi, 16, endpoint=False)]
        best_score, best = -math.inf, candidates[0]
        min_alt = float(snapshot.get("min_altitude_m", 0.0)); max_alt = float(snapshot.get("max_altitude_m", 1.0e9))
        altitude = float(threat["blue_p"][1])
        for direction in candidates:
            angular_gain = min(float(np.linalg.norm(np.cross(r, direction))) for r in threat["rel"] / threat["ranges"][:, None])
            closing_gain = float(np.mean(-(threat["rel"] / threat["ranges"][:, None]) @ direction))
            sync_gain = float(np.var(threat["tgo"] + 2.0 * ((threat["rel"] / threat["ranges"][:, None]) @ direction)))
            risk = max(0.0, direction[1] * (altitude - max_alt + 500.0) / 500.0,
                       -direction[1] * (min_alt + 500.0 - altitude) / 500.0)
            continuity = 0.0 if self.previous_direction is None else float(np.dot(direction, self.previous_direction))
            score = angular_gain + 0.25 * closing_gain + 0.05 * sync_gain + 0.15 * continuity - risk
            if score > best_score: best_score, best = score, direction
        self.previous_direction = best
        return best

    def _safe_actions(self, snapshot: dict[str, object], accelerations: np.ndarray) -> tuple[np.ndarray, list[str]]:
        blue_p = np.asarray(snapshot["blue_position_m"], dtype=np.float64)
        blue_v = np.asarray(snapshot["blue_velocity_mps"], dtype=np.float64)
        horizon = self.config.ground_prediction_s
        predicted_p = blue_p + blue_v * horizon + 0.5 * accelerations * horizon ** 2
        predicted_v = blue_v + accelerations * self.config.prediction_s
        speeds = np.linalg.norm(predicted_v, axis=1)
        horizontal_speeds = np.hypot(predicted_v[:, 0], predicted_v[:, 2])
        flight_path_angles = np.degrees(np.arctan2(predicted_v[:, 1], np.maximum(horizontal_speeds, _EPS)))
        altitude_margin = float(snapshot.get("altitude_recovery_margin_m", 0.0))
        loads = np.linalg.norm(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[:, :2], axis=1)
        safe = loads <= float(snapshot.get("max_load_factor_g", 9.0)) + 1.0e-9
        reasons = []
        checks = ((speeds >= float(snapshot.get("min_speed_mps", 100.0)), "minimum_speed"),
                  (speeds <= float(snapshot.get("max_speed_mps", 600.0)), "maximum_speed"),
                  (horizontal_speeds >= float(snapshot.get("horizontal_speed_hard_mps", 0.0)), "horizontal_speed"),
                  (flight_path_angles >= float(snapshot.get("flight_path_hard_down_deg", -90.0)), "flight_path_down"),
                  (flight_path_angles <= float(snapshot.get("flight_path_hard_up_deg", 90.0)), "flight_path_up"),
                  (predicted_p[:, 1] >= float(snapshot.get("min_altitude_m", 0.0)) + altitude_margin, "lower_altitude_recovery"),
                  (predicted_p[:, 1] <= float(snapshot.get("max_altitude_m", 1.0e9)) - altitude_margin, "upper_altitude_recovery"))
        if not np.all(safe): reasons.append("maximum_load")
        for condition, name in checks:
            if np.any(safe & ~condition): reasons.append(name)
            safe &= condition
        return safe, reasons

    def select(self, q_values: np.ndarray, snapshot: dict[str, object]) -> tuple[int, dict[str, object]]:
        q_values = np.asarray(q_values, dtype=np.float64)
        if q_values.shape != (29,): raise ValueError("q_values must contain all 29 actions")
        invalid_network = not np.isfinite(q_values).all()
        q = self._bounded(q_values) if not invalid_network else np.zeros(29)
        alive = np.asarray(snapshot["red_alive"], dtype=bool)
        if not alive.any():
            action = int(np.argmax(q)); self.previous_action = action
            return action, {"phase": self.phase, "raw_action": action, "executed_action": action,
                            "intervened": False, "active_scores": [], "fallback_reason": None}
        threat = self._threat_assessment(snapshot)
        criterion, phase_changed = self._update_phase(threat, float(snapshot.get("time_s", 0.0)))
        accelerations = self._accelerations(threat["blue_v"])
        horizon = self.config.prediction_s
        predicted_blue = threat["blue_p"] + threat["blue_v"] * horizon + 0.5 * accelerations * horizon ** 2
        predicted_red = threat["red_p"][:, None, :] + threat["red_v"][:, None, :] * horizon
        predicted_range = np.linalg.norm(predicted_red - predicted_blue[None, :, :], axis=2)
        scores = {name: np.zeros(29) for name in ("relief", "phase", "direction", "overload", "switch")}
        scores["relief"] = self._bounded(np.min(predicted_range, axis=0) - np.min(threat["ranges"]))
        desired = self._desired_direction(threat, snapshot)
        accel_norm = np.linalg.norm(accelerations, axis=1)
        scores["direction"] = np.divide(accelerations @ desired, accel_norm, out=np.zeros(29), where=accel_norm > _EPS)
        loads = np.linalg.norm(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[:, :2], axis=1)
        available = float(snapshot.get("max_load_factor_g", 9.0))
        speed = float(np.linalg.norm(threat["blue_v"]))
        min_speed = float(snapshot.get("min_speed_mps", 100.0))
        max_speed = float(snapshot.get("max_speed_mps", 600.0))
        energy_margin = np.clip((speed - min_speed) / max(max_speed - min_speed, _EPS), 0.0, 1.0)
        target_load = 1.0 + (available - 1.0) * energy_margin * self._sigmoid(
            4.0 * (threat["total"] + 0.5 * threat["encirclement"] + 2.0 / max(float(np.min(threat["tgo"])), 0.2) - 1.0))
        if self.phase == "P0": target_load = min(target_load, 0.5 * available)
        elif self.phase == "P2": target_load = min(target_load, max(1.0, 0.35 * available))
        scores["overload"] = np.clip(-np.abs(loads - target_load) / max(available - 1.0, 1.0)
                                             - self.config.load_change_penalty * np.abs(loads - self.previous_load_g) / available,
                                             -1.0, 0.0)
        phase_target = {"P0": 0.35, "P1": 1.0, "P2": 0.2}[self.phase] * available
        scores["phase"] = np.clip(-np.abs(loads - phase_target) / max(available, 1.0), -1.0, 0.0)
        scores["switch"] = np.asarray([0.0 if index == self.previous_action else 1.0 for index in range(29)])
        phase_weights = {"P0": (0.6, 0.7, 0.5, 1.0), "P1": (1.0, 1.0, 1.0, 0.8),
                         "P2": (0.5, 0.6, 0.5, 1.0)}[self.phase]
        enabled = ((self.config.threat, "relief", phase_weights[0] * self.config.threat_weight),
                   (self.config.timing, "phase", phase_weights[1] * self.config.timing_weight),
                   (self.config.direction, "direction", phase_weights[2] * self.config.direction_weight),
                   (self.config.overload, "overload", phase_weights[3] * self.config.overload_weight))
        fused = q.copy(); active = []
        for on, name, phase_weight in enabled:
            if on: fused += self.config.weight * phase_weight * scores[name]; active.append(name)
        if self.config.enabled:
            fused -= self.config.weight * self.config.switch_penalty * scores["switch"]
        safe, mask_reasons = self._safe_actions(snapshot, accelerations)
        raw_action = int(np.argmax(q)); fallback_reason = "network_nan" if invalid_network else None
        if not safe.any():
            # Deterministic minimum-risk recovery when constraints conflict.
            recovery_v = threat["blue_v"] + accelerations * self.config.prediction_s
            recovery_speed = np.linalg.norm(recovery_v, axis=1)
            recovery_p = threat["blue_p"] + threat["blue_v"] * self.config.ground_prediction_s + \
                0.5 * accelerations * self.config.ground_prediction_s ** 2
            min_speed = float(snapshot.get("min_speed_mps", 100.0))
            max_speed = float(snapshot.get("max_speed_mps", 600.0))
            min_alt = float(snapshot.get("min_altitude_m", 0.0))
            max_alt = float(snapshot.get("max_altitude_m", 1.0e9))
            risk = (np.maximum(0.0, min_speed - recovery_speed) / max(min_speed, 1.0)
                    + np.maximum(0.0, recovery_speed - max_speed) / max(max_speed, 1.0)
                    + np.maximum(0.0, min_alt - recovery_p[:, 1]) / 500.0
                    + np.maximum(0.0, recovery_p[:, 1] - max_alt) / 500.0
                    + np.maximum(0.0, loads - available) / max(available, 1.0))
            action = int(np.argmin(risk)); fallback_reason = "empty_safe_set"
        else:
            fused[~safe] = -np.inf; action = int(np.argmax(fused))
        self.previous_action, self.previous_load_g = action, float(loads[action])
        diagnostic = {"phase": self.phase, "phase_changed": phase_changed,
                      "phase_criterion": float(criterion), "main_threat_slot": int(self.primary_slot),
                      "raw_action": raw_action, "executed_action": action,
                      "intervened": action != raw_action, "active_scores": active,
                      "fallback_reason": fallback_reason, "hard_mask_reasons": mask_reasons,
                      "safe_action_count": int(np.sum(safe)), "total_threat": threat["total"],
                      "local_threats": np.asarray(threat["local"]).tolist(),
                      "minimum_tgo_s": float(np.min(threat["tgo"])),
                      "C_ang": threat["angular_coverage"], "C_syn": threat["synchronization"],
                      "C_cor": threat["corridor_compression"], "C_enc": threat["encirclement"],
                      "W_safe": threat["safe_width"], "n_ref_g": float(target_load),
                      "scores": {name: values.tolist() for name, values in scores.items()},
                      "q_rl": [float(value) if math.isfinite(float(value)) else None for value in q_values],
                      "q_fuse": [float(value) if math.isfinite(float(value)) else None for value in fused]}
        return action, diagnostic
