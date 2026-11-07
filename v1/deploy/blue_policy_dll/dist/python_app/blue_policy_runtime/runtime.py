from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G as ACTIONS
from .flight_envelope import FlightEnvelopeConfig, FlightEnvelopeConstraintLayer
from .mechanism_reward import (
    BlueMechanismStateEstimator,
    MechanismRewardConfig,
    encode_normalized_v4,
)
from .model import FrozenRainbowNumpy


_GUIDANCE_MODES = {"boost", "locked", "lock_hold", "inertial"}


@dataclass(frozen=True)
class BluePolicyRuntimeConfig:
    time_step_s: float = 0.005
    decision_interval_s: float = 0.1
    detection_range_m: float = 60_000.0
    missile_slots: int = 3
    aircraft_min_altitude_m: float = 8_000.0
    aircraft_max_altitude_m: float = 12_000.0
    aircraft_min_speed_mps: float = 100.0
    aircraft_max_speed_mps: float = 600.0
    aircraft_max_load_factor_g: float = 10.0
    missile_max_speed_mps: float = 1_770.0
    missile_boost_duration_s: float = 7.0
    seeker_acquisition_fov_deg: float = 35.0
    seeker_tracking_fov_deg: float = 60.0
    fov_break_hold_s: float = 0.75

    @classmethod
    def from_manifest(cls, manifest: dict[str, Any]) -> "BluePolicyRuntimeConfig":
        runtime = manifest.get("runtime_contract")
        if not isinstance(runtime, dict):
            raise ValueError("model manifest is missing runtime_contract")
        return cls(**runtime)

    def validate(self) -> None:
        ratio = self.decision_interval_s / self.time_step_s
        if (
            not math.isfinite(ratio)
            or ratio < 1.0
            or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9)
        ):
            raise ValueError(
                "decision_interval_s must be a positive multiple of time_step_s"
            )
        if self.missile_slots != 3:
            raise ValueError("the bundled checkpoint requires exactly three slots")


@dataclass
class _TrackMemory:
    order: int
    last_seen_time_s: float
    guidance_mode: str = "inertial"
    seeker_locked: bool = False
    fov_out_time_s: float = 0.0


class BluePolicySession:
    """One stateful blue-aircraft session sharing immutable model weights."""

    def __init__(
        self,
        policy: FrozenRainbowNumpy,
        *,
        input_mode: str = "exact",
    ) -> None:
        if input_mode not in {"exact", "kinematics_only"}:
            raise ValueError("input_mode must be exact or kinematics_only")
        self.policy = policy
        self.input_mode = input_mode
        self.config = BluePolicyRuntimeConfig.from_manifest(policy.manifest)
        self.config.validate()
        self.decision_steps = int(
            round(self.config.decision_interval_s / self.config.time_step_s)
        )
        self.envelope_config = FlightEnvelopeConfig(
            **policy.manifest["flight_envelope_config"]
        )
        self.mechanism_config = MechanismRewardConfig(
            **policy.manifest["mechanism_reward_config"]
        )
        self.reset(episode_id=0)

    def reset(self, episode_id: int) -> None:
        if isinstance(episode_id, bool) or not isinstance(episode_id, int):
            raise ValueError("episode_id must be an integer")
        self.episode_id = episode_id
        self.constraint = FlightEnvelopeConstraintLayer(self.envelope_config)
        self.estimator = BlueMechanismStateEstimator(self.mechanism_config)
        self.active = False
        self.last_decision_step: int | None = None
        self.last_seen_step: int | None = None
        self.last_seen_time_s: float | None = None
        self.action = 0
        self.raw_action = 0
        self.action_intervened = False
        self.safety_intervened = False
        self.start_command = ACTIONS[0].copy()
        self.last_applied_command = ACTIONS[0].copy()
        self.visible_track_ids: tuple[int, ...] = ()
        self.mechanism: dict[str, object] = {}
        self.track_memory: dict[int, _TrackMemory] = {}
        self.next_track_order = 0
        self.decision_count = 0

    @staticmethod
    def _vector(name: str, value: object) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite three-vector")
        return vector

    @staticmethod
    def _track_id(value: object) -> int:
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            raise ValueError("track_id must be an integer")
        result = int(value)
        if result < -(2**63) or result >= 2**63:
            raise ValueError("track_id must fit in int64")
        return result

    @staticmethod
    def _boresight_angle_deg(
        missile_position_m: np.ndarray,
        missile_velocity_mps: np.ndarray,
        blue_position_m: np.ndarray,
    ) -> float:
        los = blue_position_m - missile_position_m
        los_norm = float(np.linalg.norm(los))
        speed = float(np.linalg.norm(missile_velocity_mps))
        if los_norm <= 1.0e-9 or speed <= 1.0e-9:
            return 180.0
        cosine = float(
            np.clip(
                np.dot(missile_velocity_mps / speed, los / los_norm), -1.0, 1.0
            )
        )
        return math.degrees(math.acos(cosine))

    def _update_estimated_guidance(
        self,
        memory: _TrackMemory,
        time_s: float,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        blue_position_m: np.ndarray,
    ) -> None:
        dt = max(0.0, time_s - memory.last_seen_time_s)
        if time_s <= self.config.missile_boost_duration_s + 1.0e-9:
            memory.guidance_mode = "boost"
            memory.last_seen_time_s = time_s
            return
        angle_deg = self._boresight_angle_deg(
            position_m, velocity_mps, blue_position_m
        )
        limit = (
            self.config.seeker_tracking_fov_deg
            if memory.seeker_locked
            else self.config.seeker_acquisition_fov_deg
        )
        if angle_deg <= limit:
            memory.fov_out_time_s = 0.0
            memory.seeker_locked = True
            memory.guidance_mode = "locked"
        else:
            was_held = memory.seeker_locked or memory.guidance_mode == "lock_hold"
            if was_held:
                memory.fov_out_time_s += dt
            held = was_held and (
                memory.fov_out_time_s < self.config.fov_break_hold_s
            )
            memory.seeker_locked = held
            memory.guidance_mode = "lock_hold" if held else "inertial"
        memory.last_seen_time_s = time_s

    def _tracks(
        self,
        raw_tracks: object,
        *,
        time_s: float,
        blue_position_m: np.ndarray,
    ) -> list[dict[str, object]]:
        if not isinstance(raw_tracks, list):
            raise ValueError("tracks must be an array")
        validated: list[dict[str, object]] = []
        seen: set[int] = set()
        for index, raw in enumerate(raw_tracks):
            if not isinstance(raw, dict):
                raise ValueError(f"tracks[{index}] must be an object")
            track_id = self._track_id(raw.get("track_id"))
            if track_id in seen:
                raise ValueError(f"duplicate track_id {track_id}")
            seen.add(track_id)
            position = self._vector(
                f"tracks[{index}].position_m", raw.get("position_m")
            )
            velocity = self._vector(
                f"tracks[{index}].velocity_mps", raw.get("velocity_mps")
            )
            alive = raw.get("alive", True)
            if not isinstance(alive, bool):
                raise ValueError(f"tracks[{index}].alive must be boolean")

            memory = self.track_memory.get(track_id)
            if memory is None:
                memory = _TrackMemory(
                    order=self.next_track_order,
                    last_seen_time_s=time_s,
                    guidance_mode=(
                        "boost"
                        if time_s <= self.config.missile_boost_duration_s + 1.0e-9
                        else "inertial"
                    ),
                )
                self.next_track_order += 1
                self.track_memory[track_id] = memory

            if self.input_mode == "exact":
                if "energy" not in raw or "guidance_mode" not in raw:
                    raise ValueError(
                        "exact input mode requires energy and guidance_mode for every track"
                    )
                energy = float(raw["energy"])
                guidance_mode = str(raw["guidance_mode"])
                if not math.isfinite(energy) or not 0.0 <= energy <= 1.0:
                    raise ValueError(f"tracks[{index}].energy must be in [0, 1]")
                if guidance_mode not in _GUIDANCE_MODES:
                    raise ValueError(
                        f"tracks[{index}].guidance_mode is unsupported"
                    )
                memory.guidance_mode = guidance_mode
                memory.last_seen_time_s = time_s
            else:
                self._update_estimated_guidance(
                    memory, time_s, position, velocity, blue_position_m
                )
                energy = float(
                    np.clip(
                        np.linalg.norm(velocity)
                        / self.config.missile_max_speed_mps,
                        0.0,
                        1.0,
                    )
                )
                guidance_mode = memory.guidance_mode

            validated.append(
                {
                    "track_id": track_id,
                    "order": memory.order,
                    "position_m": position,
                    "velocity_mps": velocity,
                    "alive": alive,
                    "energy": energy,
                    "guidance_mode": guidance_mode,
                }
            )
        return validated

    def _visible(
        self, tracks: list[dict[str, object]], blue_position_m: np.ndarray
    ) -> list[dict[str, object]]:
        detected = [
            track
            for track in tracks
            if bool(track["alive"])
            and float(np.linalg.norm(track["position_m"] - blue_position_m))
            < self.config.detection_range_m
        ]
        nearest = sorted(
            detected,
            key=lambda track: (
                float(np.linalg.norm(track["position_m"] - blue_position_m)),
                int(track["order"]),
            ),
        )[: self.config.missile_slots]
        return sorted(nearest, key=lambda track: int(track["order"]))

    def _remap_slots(self, track_ids: tuple[int, ...]) -> None:
        for name in ("primary_slot", "primary_candidate"):
            old_slot = getattr(self.estimator, name)
            entity = (
                self.visible_track_ids[old_slot]
                if old_slot is not None and old_slot < len(self.visible_track_ids)
                else None
            )
            new_slot = track_ids.index(entity) if entity in track_ids else None
            setattr(self.estimator, name, new_slot)
            if name == "primary_candidate" and new_slot is None:
                self.estimator.primary_candidate_count = 0
        self.visible_track_ids = track_ids

    def _command(self, step_count: int, *, next_frame: bool) -> np.ndarray:
        if self.last_decision_step is None:
            return ACTIONS[0].copy()
        fraction = np.clip(
            (
                step_count
                - self.last_decision_step
                + int(next_frame)
            )
            / self.decision_steps,
            0.0,
            1.0,
        )
        return FlightEnvelopeConstraintLayer._interpolated_commands(
            self.start_command, ACTIONS[self.action], float(fraction)
        )

    def _snapshot(
        self,
        *,
        time_s: float,
        blue_position_m: np.ndarray,
        blue_velocity_mps: np.ndarray,
        actual_command: np.ndarray,
        visible: list[dict[str, object]],
    ) -> dict[str, object]:
        return {
            "blue_position_m": blue_position_m.tolist(),
            "blue_velocity_mps": blue_velocity_mps.tolist(),
            "time_s": time_s,
            "red_positions_m": np.asarray(
                [track["position_m"] for track in visible], dtype=np.float64
            ).reshape(-1, 3),
            "red_velocities_mps": np.asarray(
                [track["velocity_mps"] for track in visible], dtype=np.float64
            ).reshape(-1, 3),
            "red_alive": [True] * len(visible),
            "red_energy": [float(track["energy"]) for track in visible],
            "red_guidance_modes": [
                str(track["guidance_mode"]) for track in visible
            ],
            "min_altitude_m": self.config.aircraft_min_altitude_m,
            "max_altitude_m": self.config.aircraft_max_altitude_m,
            "min_speed_mps": self.config.aircraft_min_speed_mps,
            "max_speed_mps": self.config.aircraft_max_speed_mps,
            "max_load_factor_g": self.config.aircraft_max_load_factor_g,
            "previous_executed_action_index": self.action,
            "actual_load_command_body_g": actual_command.tolist(),
            "mechanism_emergency_gate": float(
                self.mechanism.get("emergency_gate", 0.0)
            ),
        }

    def _observation(
        self,
        snapshot: dict[str, object],
        blue_position_m: np.ndarray,
        blue_velocity_mps: np.ndarray,
        actual_command: np.ndarray,
    ) -> np.ndarray:
        values = [
            0.0,
            blue_position_m[1] / 20_000.0,
            0.0,
            *(blue_velocity_mps / 2_000.0),
        ]
        values.extend(
            encode_normalized_v4(
                snapshot, self.mechanism, self.config.missile_slots
            )
        )
        one_hot = np.zeros(len(ACTIONS), dtype=np.float64)
        one_hot[self.action] = 1.0
        values.extend(one_hot.tolist())
        values.extend(
            (
                actual_command
                / np.asarray([9.0, 9.0, math.pi], dtype=np.float64)
            ).tolist()
        )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.policy.observation_dim,):
            raise RuntimeError(
                f"observation has shape {observation.shape}, "
                f"expected ({self.policy.observation_dim},)"
            )
        return observation

    def step(
        self, frame: dict[str, object], *, include_diagnostics: bool = False
    ) -> dict[str, object]:
        if not isinstance(frame, dict):
            raise ValueError("frame must be an object")
        step_count = frame.get("step_count")
        if isinstance(step_count, bool) or not isinstance(
            step_count, (int, np.integer)
        ):
            raise ValueError("step_count must be an integer")
        step_count = int(step_count)
        if step_count < 0:
            raise ValueError("step_count must be non-negative")
        time_s = float(frame.get("time_s"))
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        if self.last_seen_step is not None and step_count <= self.last_seen_step:
            raise ValueError("step_count must increase strictly within an episode")
        if self.last_seen_time_s is not None and time_s <= self.last_seen_time_s:
            raise ValueError("time_s must increase strictly within an episode")

        blue = frame.get("blue")
        if not isinstance(blue, dict):
            raise ValueError("blue must be an object")
        blue_alive = blue.get("alive", True)
        if not isinstance(blue_alive, bool):
            raise ValueError("blue.alive must be boolean")
        blue_position = self._vector("blue.position_m", blue.get("position_m"))
        blue_velocity = self._vector("blue.velocity_mps", blue.get("velocity_mps"))
        actual_raw = frame.get("actual_load_command_body_g")
        actual_command = (
            self.last_applied_command.copy()
            if actual_raw is None
            else self._vector("actual_load_command_body_g", actual_raw)
        )
        tracks = self._tracks(
            frame.get("tracks", []),
            time_s=time_s,
            blue_position_m=blue_position,
        )
        visible = self._visible(tracks, blue_position)
        visible_ids = tuple(int(track["track_id"]) for track in visible)
        self._remap_slots(visible_ids)
        self.active = self.active or bool(visible)

        decision_updated = False
        diagnostic: dict[str, object] = {}
        q_values: np.ndarray | None = None
        observation: np.ndarray | None = None
        if blue_alive and self.active and (
            self.last_decision_step is None
            or step_count - self.last_decision_step >= self.decision_steps
        ):
            snapshot = self._snapshot(
                time_s=time_s,
                blue_position_m=blue_position,
                blue_velocity_mps=blue_velocity,
                actual_command=actual_command,
                visible=visible,
            )
            if self.policy.manifest["model_contract"]["observation_schema"] == "normalized_v4":
                mask, _, _ = self.constraint.constraints(snapshot)
                self.mechanism = self.estimator.observe(snapshot, mask)
                snapshot["mechanism_emergency_gate"] = float(
                    self.mechanism["emergency_gate"]
                )
            observation = self._observation(
                snapshot, blue_position, blue_velocity, actual_command
            )
            q_values = self.policy.expected_action_values(observation)[0]
            self.action, diagnostic = self.constraint.select(q_values, snapshot)
            self.raw_action = int(diagnostic["raw_action"])
            self.action_intervened = bool(diagnostic.get("intervened", False))
            self.safety_intervened = bool(
                diagnostic.get("safety_filter_intervened", False)
            )
            self.last_decision_step = step_count
            self.start_command = actual_command.copy()
            self.decision_count += 1
            decision_updated = True

        applied = (
            self._command(step_count, next_frame=True)
            if blue_alive and self.active
            else ACTIONS[0].copy()
        )
        self.last_applied_command = applied.copy()
        self.last_seen_step = step_count
        self.last_seen_time_s = time_s

        result: dict[str, object] = {
            "episode_id": self.episode_id,
            "step_count": step_count,
            "time_s": time_s,
            "active": bool(self.active and blue_alive),
            "mode": (
                "inactive"
                if not blue_alive
                else "rainbow"
                if self.active
                else "cruise"
            ),
            "decision_updated": decision_updated,
            "raw_action_index": int(self.raw_action),
            "action_index": int(self.action),
            "target_command_body_g": ACTIONS[self.action].tolist(),
            "applied_command_body_g": applied.tolist(),
            "valid_until_s": time_s + self.config.time_step_s,
            "visible_track_ids": list(self.visible_track_ids),
            "input_mode": self.input_mode,
            "decision_count": self.decision_count,
            "action_intervened": self.action_intervened,
            "safety_intervened": self.safety_intervened,
            "fallback_required": bool(
                diagnostic.get("fallback_required", False)
            ),
            "safe_action_count": int(diagnostic.get("safe_action_count", 0)),
        }
        if include_diagnostics and decision_updated:
            result["diagnostics"] = {
                "observation": observation.tolist() if observation is not None else None,
                "q_values": q_values.tolist() if q_values is not None else None,
                "intervened": bool(diagnostic.get("intervened", False)),
                "fallback_reason": diagnostic.get("fallback_reason"),
                "hard_mask_reasons": list(
                    diagnostic.get("hard_mask_reasons", [])
                ),
                "raw_action_hard_violation_reasons": list(
                    diagnostic.get("raw_action_hard_violation_reasons", [])
                ),
                "total_soft_penalty": float(
                    diagnostic.get("total_soft_penalty", 0.0)
                ),
                "effective_action_mask": list(
                    diagnostic.get("effective_action_mask", [])
                ),
            }
        return result
