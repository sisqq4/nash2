"""Stateful compatibility adapter for the blue policy's minimal external input.

The adapter deliberately keeps the checkpoint contract unchanged.  Callers
provide only time/reset metadata, ownship kinematics, stable missile tracks,
and the previously applied continuous maneuver.  The adapter reconstructs the
fixed normalized-v4 observation and keeps the discrete action history inside
the blue module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Hashable, Sequence

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from ..env.types import EnvironmentConfig
from .environment import (
    BlueEscapeEnvConfig,
    blue_action_context,
    blue_observation_dim,
)
from .mechanism_reward import BlueMechanismStateEstimator, encode_normalized_v4


@dataclass(frozen=True)
class BlueMissileTrack:
    """One externally supplied attacking-missile track."""

    track_id: Hashable
    position_m: Sequence[float]
    velocity_mps: Sequence[float]


@dataclass(frozen=True)
class BlueMinimalInput:
    """Minimal per-step data contract consumed by the blue-side adapter."""

    time_s: float
    reset: bool
    blue_position_m: Sequence[float]
    blue_velocity_mps: Sequence[float]
    missile_tracks: Sequence[BlueMissileTrack]
    previous_actual_load_command_body_g: Sequence[float]


@dataclass(frozen=True)
class BlueMinimalAdapterOutput:
    """Reconstructed policy input plus state used by test-time arbiters."""

    observation: np.ndarray
    mechanism_snapshot: dict[str, object]
    mechanism_state: dict[str, object]
    active: bool
    visible_track_ids: tuple[Hashable, ...]


@dataclass
class _TrackMemory:
    order: int
    last_seen_time_s: float
    guidance_mode: str = "inertial"
    seeker_locked: bool = False
    fov_out_time_s: float = 0.0


def _finite_vector(name: str, value: Sequence[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    return vector


class BlueMinimalInputAdapter:
    """Expand a minimal external stream into an existing normalized-v4 input.

    ``record_executed_action`` is called by the blue module after its final
    discrete action has passed arbitration.  The action index is therefore
    module-owned state and is not part of :class:`BlueMinimalInput`.
    """

    def __init__(
        self,
        environment: EnvironmentConfig,
        config: BlueEscapeEnvConfig,
    ) -> None:
        if config.observation_schema != "normalized_v4":
            raise ValueError(
                "the minimal input adapter currently supports normalized_v4 checkpoints only"
            )
        self.environment = environment
        self.config = config
        self.missile_slots = (
            config.max_missiles
            if config.pad_observation_to_max_missiles
            else config.missile_count
        )
        self.observation_dim = blue_observation_dim("normalized_v4", self.missile_slots)
        loads = np.linalg.norm(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[:, :2], axis=1)
        self._structural_action_mask = (
            loads <= environment.aircraft.max_load_factor_g + 1.0e-9
        )
        self.reset_state()

    def reset_state(self) -> None:
        """Clear every episode-scoped estimate and action-history value."""
        self._initialized = False
        self._last_time_s: float | None = None
        self._active = False
        self._previous_executed_action_index = 0
        self._track_memory: dict[Hashable, _TrackMemory] = {}
        self._next_track_order = 0
        self._visible_track_ids: tuple[Hashable, ...] = ()
        self._mechanism_estimator = BlueMechanismStateEstimator(
            self.config.mechanism_reward
        )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def previous_executed_action_index(self) -> int:
        return self._previous_executed_action_index

    def record_executed_action(self, action_index: int) -> None:
        """Remember the final discrete action selected by the blue module."""
        action = int(action_index)
        if not 0 <= action < len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G):
            raise ValueError("executed action index is out of range")
        self._previous_executed_action_index = action

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
        cosine = float(np.clip(
            np.dot(missile_velocity_mps / speed, los / los_norm), -1.0, 1.0
        ))
        return math.degrees(math.acos(cosine))

    def _new_track_memory(
        self,
        time_s: float,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        blue_position_m: np.ndarray,
    ) -> _TrackMemory:
        memory = _TrackMemory(
            order=self._next_track_order,
            last_seen_time_s=time_s,
        )
        self._next_track_order += 1
        # At the v1 post-boost entry instant the simulator still exposes the
        # final boost mode.  A later first sighting is treated as a fresh seeker
        # acquisition based only on the supplied kinematics.
        if time_s <= self.environment.missile.boost_duration_s + 1.0e-9:
            memory.guidance_mode = "boost"
            return memory
        self._update_guidance(
            memory, time_s, position_m, velocity_mps, blue_position_m
        )
        return memory

    def _update_guidance(
        self,
        memory: _TrackMemory,
        time_s: float,
        position_m: np.ndarray,
        velocity_mps: np.ndarray,
        blue_position_m: np.ndarray,
    ) -> None:
        dt = max(0.0, time_s - memory.last_seen_time_s)
        missile = self.environment.missile
        if (
            memory.guidance_mode == "boost"
            and time_s <= missile.boost_duration_s + 1.0e-9
        ):
            memory.last_seen_time_s = time_s
            return
        angle_deg = self._boresight_angle_deg(
            position_m, velocity_mps, blue_position_m
        )
        fov_limit_deg = (
            missile.seeker_tracking_fov_deg
            if memory.seeker_locked
            else missile.seeker_acquisition_fov_deg
        )
        if angle_deg <= fov_limit_deg:
            memory.fov_out_time_s = 0.0
            memory.seeker_locked = True
            memory.guidance_mode = "locked"
        else:
            was_held = memory.seeker_locked or memory.guidance_mode == "lock_hold"
            if was_held:
                memory.fov_out_time_s += dt
            held = was_held and memory.fov_out_time_s < missile.fov_break_hold_s
            memory.seeker_locked = held
            memory.guidance_mode = "lock_hold" if held else "inertial"
        memory.last_seen_time_s = time_s

    def _validated_tracks(
        self,
        frame: BlueMinimalInput,
        blue_position_m: np.ndarray,
    ) -> list[tuple[BlueMissileTrack, np.ndarray, np.ndarray, _TrackMemory]]:
        validated: list[tuple[BlueMissileTrack, np.ndarray, np.ndarray]] = []
        seen_ids: set[Hashable] = set()
        for index, track in enumerate(frame.missile_tracks):
            track_id = track.track_id
            if isinstance(track_id, (bool, np.bool_)):
                raise ValueError("track_id must not be boolean")
            try:
                hash(track_id)
            except TypeError as error:
                raise ValueError("track_id must be hashable") from error
            if track_id in seen_ids:
                raise ValueError(f"duplicate track_id {track_id!r}")
            seen_ids.add(track_id)
            position = _finite_vector(
                f"missile_tracks[{index}].position_m", track.position_m
            )
            velocity = _finite_vector(
                f"missile_tracks[{index}].velocity_mps", track.velocity_mps
            )
            validated.append((track, position, velocity))

        tracks: list[tuple[BlueMissileTrack, np.ndarray, np.ndarray, _TrackMemory]] = []
        for track, position, velocity in validated:
            track_id = track.track_id
            memory = self._track_memory.get(track_id)
            if memory is None:
                memory = self._new_track_memory(
                    float(frame.time_s), position, velocity, blue_position_m
                )
                self._track_memory[track_id] = memory
            else:
                self._update_guidance(
                    memory, float(frame.time_s), position, velocity, blue_position_m
                )
            tracks.append((track, position, velocity, memory))
        return tracks

    def _select_visible(
        self,
        tracks: list[tuple[BlueMissileTrack, np.ndarray, np.ndarray, _TrackMemory]],
        blue_position_m: np.ndarray,
    ) -> list[tuple[BlueMissileTrack, np.ndarray, np.ndarray, _TrackMemory]]:
        detected = [
            item
            for item in tracks
            if float(np.linalg.norm(item[1] - blue_position_m))
            < self.config.threat_detection_range_m
        ]
        nearest = sorted(
            detected,
            key=lambda item: (
                float(np.linalg.norm(item[1] - blue_position_m)),
                item[3].order,
            ),
        )[: self.missile_slots]
        # Match the existing contract: select nearest threats, then preserve
        # their stable entity order in policy slots.
        return sorted(nearest, key=lambda item: item[3].order)

    def _remap_mechanism_slots(self, track_ids: tuple[Hashable, ...]) -> None:
        for name in ("primary_slot", "primary_candidate"):
            old_local = getattr(self._mechanism_estimator, name)
            entity = (
                self._visible_track_ids[old_local]
                if old_local is not None and old_local < len(self._visible_track_ids)
                else None
            )
            new_local = track_ids.index(entity) if entity in track_ids else None
            setattr(self._mechanism_estimator, name, new_local)
            if name == "primary_candidate" and new_local is None:
                self._mechanism_estimator.primary_candidate_count = 0
        self._visible_track_ids = track_ids

    def _snapshot(
        self,
        time_s: float,
        blue_position_m: np.ndarray,
        blue_velocity_mps: np.ndarray,
        actual_command: np.ndarray,
        visible: list[
            tuple[BlueMissileTrack, np.ndarray, np.ndarray, _TrackMemory]
        ],
    ) -> dict[str, object]:
        missile_max_speed = self.environment.missile.max_speed_mps
        aircraft = self.environment.aircraft
        return {
            "blue_position_m": blue_position_m.tolist(),
            "blue_velocity_mps": blue_velocity_mps.tolist(),
            "time_s": time_s,
            "red_positions_m": [item[1].tolist() for item in visible],
            "red_velocities_mps": [item[2].tolist() for item in visible],
            "red_alive": [True] * len(visible),
            "red_energy": [
                float(np.clip(np.linalg.norm(item[2]) / missile_max_speed, 0.0, 1.0))
                for item in visible
            ],
            "red_guidance_modes": [item[3].guidance_mode for item in visible],
            "min_altitude_m": aircraft.min_altitude_m,
            "max_altitude_m": aircraft.max_altitude_m,
            "min_speed_mps": aircraft.min_speed_mps,
            "max_speed_mps": aircraft.max_speed_mps,
            "max_load_factor_g": aircraft.max_load_factor_g,
            "previous_executed_action_index": self._previous_executed_action_index,
            "actual_load_command_body_g": actual_command.tolist(),
            "mechanism_emergency_gate": 0.0,
        }

    def observe(self, frame: BlueMinimalInput) -> BlueMinimalAdapterOutput:
        """Validate one external frame and reconstruct the checkpoint input."""
        time_s = float(frame.time_s)
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError("time_s must be finite and non-negative")
        if not isinstance(frame.reset, (bool, np.bool_)):
            raise ValueError("reset must be boolean")
        if frame.reset:
            self.reset_state()
        elif not self._initialized:
            raise RuntimeError("the first minimal input frame must set reset=True")
        elif self._last_time_s is not None and time_s <= self._last_time_s:
            raise ValueError("time_s must increase strictly within an episode")

        blue_position = _finite_vector("blue_position_m", frame.blue_position_m)
        blue_velocity = _finite_vector("blue_velocity_mps", frame.blue_velocity_mps)
        actual_command = _finite_vector(
            "previous_actual_load_command_body_g",
            frame.previous_actual_load_command_body_g,
        )
        tracks = self._validated_tracks(frame, blue_position)
        visible = self._select_visible(tracks, blue_position)
        if visible:
            self._active = True
        visible_ids = tuple(item[0].track_id for item in visible)
        self._remap_mechanism_slots(visible_ids)
        snapshot = self._snapshot(
            time_s, blue_position, blue_velocity, actual_command, visible
        )
        mechanism: dict[str, object] = {}
        if self._active:
            mechanism = self._mechanism_estimator.observe(
                snapshot, self._structural_action_mask
            )
            snapshot["mechanism_emergency_gate"] = float(
                mechanism["emergency_gate"]
            )

        values = [
            0.0,
            blue_position[1] / 20_000.0,
            0.0,
            *(blue_velocity / 2_000.0),
        ]
        values.extend(
            encode_normalized_v4(snapshot, mechanism, self.missile_slots)
        )
        values.extend(
            blue_action_context(
                self._previous_executed_action_index,
                actual_command,
            )
        )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                f"minimal adapter produced {observation.shape}, "
                f"expected ({self.observation_dim},)"
            )
        self._initialized = True
        self._last_time_s = time_s
        return BlueMinimalAdapterOutput(
            observation=observation,
            mechanism_snapshot=snapshot,
            mechanism_state=dict(mechanism),
            active=self._active,
            visible_track_ids=visible_ids,
        )
