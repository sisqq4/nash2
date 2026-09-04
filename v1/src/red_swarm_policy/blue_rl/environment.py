from __future__ import annotations

import math
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from ..env.environment import RedBlueEngagementEnv
from ..env.types import EnvironmentConfig, RedAction
from .acmi import AcmiRecorder


BLUE_ACTION_CONTEXT_DIM = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G) + 3


def blue_observation_dim(schema: str, missile_slots: int) -> int:
    if schema == "legacy_v1":
        return 6 + 3 * int(missile_slots)
    if schema == "normalized_v2":
        return 6 + 4 * int(missile_slots)
    if schema == "normalized_v3":
        return 6 + 4 * int(missile_slots) + BLUE_ACTION_CONTEXT_DIM
    raise ValueError(f"unsupported blue observation schema: {schema}")


def blue_action_context(action_index: int, applied_command: np.ndarray) -> list[float]:
    """Encode all state used by command-rate constraints into normalized inputs."""
    action = int(action_index)
    if not 0 <= action < len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G):
        raise ValueError("previous executed action index is out of range")
    command = np.asarray(applied_command, dtype=np.float64)
    if command.shape != (3,) or not np.all(np.isfinite(command)):
        raise ValueError("applied load command must be a finite three-vector")
    one_hot = np.zeros(len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G), dtype=np.float64)
    one_hot[action] = 1.0
    normalized_command = command / np.array([9.0, 9.0, math.pi], dtype=np.float64)
    return [*one_hot.tolist(), *normalized_command.tolist()]


@dataclass(frozen=True)
class BlueEscapeEnvConfig:
    """Blue-RL adapter settings; physical/scenario values remain in EnvironmentConfig."""

    missile_count: int = 1
    max_missiles: int = 4
    pad_observation_to_max_missiles: bool = False
    observation_schema: str = "legacy_v1"
    decision_interval_s: float = 0.1
    threat_detection_range_m: float = 60000.0
    initial_altitude_range_m: tuple[float, float] = (9000.0, 11000.0)
    record_acmi: bool = True
    acmi_episode_interval: int = 1
    acmi_directory: str = "outputs/blue_rl/acmi"
    expose_evaluation_mechanism_state: bool = False
    terminal_success_reward: float = 10.0
    terminal_killed_reward: float = -10.0
    terminal_timeout_reward: float = 2.0
    survival_progress_bonus: float = 1.0
    fast_success_bonus: float = 1.0
    shaping_scale: float = 2.0
    shaping_discount: float = 0.999
    near_range_m: float = 30000.0
    range_transition_m: float = 8000.0
    threat_softmin_temperature_m: float = 12000.0
    far_away_weight: float = 1.0
    near_tangent_weight: float = 0.65
    near_dive_weight: float = 0.35

    def validate(self, environment: EnvironmentConfig) -> None:
        if not 1 <= self.missile_count <= self.max_missiles <= 4:
            raise ValueError("blue training supports one to four missiles against one aircraft")
        if self.observation_schema not in {"legacy_v1", "normalized_v2", "normalized_v3"}:
            raise ValueError(
                "observation_schema must be 'legacy_v1', 'normalized_v2', or 'normalized_v3'"
            )
        if (
            isinstance(self.acmi_episode_interval, bool)
            or not isinstance(self.acmi_episode_interval, (int, np.integer))
            or self.acmi_episode_interval < 0
        ):
            raise ValueError("acmi_episode_interval must be a non-negative integer")
        ratio = self.decision_interval_s / environment.time_step_s
        if self.decision_interval_s <= 0 or not np.isclose(ratio, round(ratio)):
            raise ValueError("decision_interval_s must be a positive multiple of the physics time step")
        scalar_values = (
            self.terminal_success_reward, self.terminal_killed_reward,
            self.terminal_timeout_reward, self.survival_progress_bonus,
            self.fast_success_bonus, self.shaping_scale, self.shaping_discount, self.near_range_m,
            self.range_transition_m, self.threat_softmin_temperature_m,
            self.far_away_weight, self.near_tangent_weight, self.near_dive_weight,
            self.threat_detection_range_m, *self.initial_altitude_range_m,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("blue reward configuration values must be finite")
        if (
            self.terminal_success_reward <= self.terminal_timeout_reward
            or self.terminal_timeout_reward <= self.terminal_killed_reward
            or min(self.survival_progress_bonus, self.fast_success_bonus, self.shaping_scale) < 0.0
            or not 0.0 < self.shaping_discount <= 1.0
            or min(self.near_range_m, self.range_transition_m, self.threat_softmin_temperature_m) <= 0.0
            or self.threat_detection_range_m <= 0.0
            or min(self.far_away_weight, self.near_tangent_weight, self.near_dive_weight) < 0.0
            or not np.isclose(self.near_tangent_weight + self.near_dive_weight, 1.0)
            or len(self.initial_altitude_range_m) != 2
            or self.initial_altitude_range_m[0] > self.initial_altitude_range_m[1]
            or self.initial_altitude_range_m[0] < environment.aircraft.min_altitude_m
            or self.initial_altitude_range_m[1] > environment.aircraft.max_altitude_m
        ):
            raise ValueError("blue reward scales, ranges, or tactical weights are invalid")


class BlueEscapeEnv:
    """Separated Gym-like blue training env backed by the unchanged v1 simulation.

    Red missiles are always assigned to the sole blue aircraft with exactly zero
    residual bias. Consequently the existing physics layer supplies pure PN and
    no red actor, critic, high-level assignment policy, or low-level policy runs.
    """

    def __init__(self, environment_config: EnvironmentConfig = EnvironmentConfig(),
                 config: BlueEscapeEnvConfig = BlueEscapeEnvConfig()) -> None:
        # Blue-RL scenarios have a deliberately narrower launch-altitude
        # contract than the general engagement environment.
        environment_config = replace(
            environment_config,
            scenario=replace(
                environment_config.scenario,
                blue_altitude_range_m=tuple(config.initial_altitude_range_m),
            ),
        )
        environment_config.validate()
        config.validate(environment_config)
        self.environment_config, self.config = environment_config, config
        self.inner = RedBlueEngagementEnv(environment_config, record_replay=False)
        self.frames_per_action = int(round(config.decision_interval_s / environment_config.time_step_s))
        # Single-scenario runs retain the nash1.6 observation contract.  A
        # multi-scenario run pads missing missile slots to max_missiles so one
        # policy can consume every selected scenario.
        slots = config.max_missiles if config.pad_observation_to_max_missiles else config.missile_count
        self.observation_dim = blue_observation_dim(config.observation_schema, slots)
        self.action_dim = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)
        self.recorder = AcmiRecorder(); self.episode = 0
        self._previous_potential: dict[str, float] = {}
        self._record_current_episode = False
        self._learning_active = False
        self._activation_time_s: float | None = None
        self._activation_range_m: float | None = None
        self._previous_executed_action_index = 0
        self._applied_load_command_body_g = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[0].copy()

    def reset(self, seed: int | None = None, *, episode_index: int | None = None,
              missile_count: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        if missile_count is not None:
            if not 1 <= int(missile_count) <= self.config.max_missiles:
                raise ValueError(f"missile_count must be in [1, {self.config.max_missiles}]")
            self._missile_count = int(missile_count)
        else:
            self._missile_count = self.config.missile_count
        self.episode = self.episode + 1 if episode_index is None else int(episode_index)
        if self.episode < 1:
            raise ValueError("episode_index must be positive")
        self.recorder = AcmiRecorder()
        self._record_current_episode = bool(
            self.config.record_acmi
            and self.config.acmi_episode_interval > 0
            and self.episode % self.config.acmi_episode_interval == 0
        )
        self.inner.reset(seed=seed, style="many_to_one", red_count=self._missile_count,
                         blue_count=1, start_mode="post_boost")
        assert self.inner.state is not None
        if self._record_current_episode:
            self.recorder.record(self.inner.state)
        self._learning_active = False
        self._activation_time_s = None
        self._activation_range_m = None
        self._previous_executed_action_index = 0
        self._applied_load_command_body_g = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[0].copy()
        self._activate_if_threat_observed()
        self._previous_potential = self._threat_potential() if self._learning_active else self._zero_potential()
        info = {
            "time_s": self.inner.state.time_s,
            "pure_pn": True,
            "missile_slot_mask": [index < self._missile_count for index in range(self.config.max_missiles)],
            "initialization": self._initialization_snapshot(),
            "flight_quality_state": self._mechanism_snapshot(),
            **self._learning_status(learning_transition=False, requested_action=0,
                                    constrained_action=0, executed_action=0),
        }
        if self.config.expose_evaluation_mechanism_state:
            info["mechanism_state"] = self._mechanism_snapshot()
        return self._observation(), info

    def _mechanism_snapshot(self) -> dict[str, object]:
        """Expose physical state to the optional evaluation-only action shaper."""
        assert self.inner.state is not None
        blue = self.inner.state.blue[0]
        return {"blue_position_m": blue.position_m.tolist(),
                "blue_velocity_mps": blue.velocity_mps.tolist(),
                "time_s": float(self.inner.state.time_s),
                "red_positions_m": [red.position_m.tolist() for red in self.inner.state.red],
                "red_velocities_mps": [red.velocity_mps.tolist() for red in self.inner.state.red],
                "red_alive": [bool(red.alive) for red in self.inner.state.red],
                "red_energy": [float(red.energy) for red in self.inner.state.red],
                "red_guidance_modes": [red.guidance_mode for red in self.inner.state.red],
                "min_altitude_m": self.environment_config.aircraft.min_altitude_m,
                "max_altitude_m": self.environment_config.aircraft.max_altitude_m,
                "min_speed_mps": self.environment_config.aircraft.min_speed_mps,
                "max_speed_mps": self.environment_config.aircraft.max_speed_mps,
                "max_load_factor_g": self.environment_config.aircraft.max_load_factor_g,
                "previous_executed_action_index": self._previous_executed_action_index,
                "actual_load_command_body_g": self._applied_load_command_body_g.tolist()}

    def _initialization_snapshot(self) -> dict[str, object]:
        """Return a JSON-safe, immutable description of the sampled scenario."""
        assert self.inner.state is not None

        def describe(entity: object) -> dict[str, object]:
            position = np.asarray(entity.position_m, dtype=np.float64)
            velocity = np.asarray(entity.velocity_mps, dtype=np.float64)
            horizontal_speed = float(np.hypot(velocity[0], velocity[2]))
            return {
                "position_m": position.tolist(),
                "altitude_m": float(position[1]),
                "heading_deg": float(math.degrees(math.atan2(velocity[2], velocity[0]))),
                "flight_path_angle_deg": float(math.degrees(math.atan2(velocity[1], horizontal_speed))),
                "bank_angle_deg": float(math.degrees(entity.bank_angle_rad)),
                "speed_mps": float(np.linalg.norm(velocity)),
            }

        blue = self.inner.state.blue[0]
        missile_center = np.mean([red.position_m for red in self.inner.state.red], axis=0)
        reference = missile_center[[0, 2]] - blue.position_m[[0, 2]]
        blue_velocity = blue.velocity_mps[[0, 2]]
        relative_heading_deg = float(math.degrees(math.atan2(
            reference[0] * blue_velocity[1] - reference[1] * blue_velocity[0],
            float(np.dot(reference, blue_velocity)),
        )))
        if -45.0 <= relative_heading_deg < 45.0:
            orientation = "toward_missile_swarm"
        elif 45.0 <= relative_heading_deg < 135.0:
            orientation = "positive_90_deg"
        elif -135.0 <= relative_heading_deg < -45.0:
            orientation = "negative_90_deg"
        else:
            orientation = "away_from_missile_swarm"
        return {
            "blue_aircraft": [describe(blue)],
            "red_missiles": [describe(red) for red in self.inner.state.red],
            "missile_swarm_center_position_m": missile_center.tolist(),
            "blue_relative_heading_deg": relative_heading_deg,
            "blue_orientation": orientation,
        }

    def step(self, action: int, *, policy_action: int | None = None) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if not 0 <= int(action) < self.action_dim: raise ValueError(f"action must be in [0, {self.action_dim})")
        if policy_action is not None and not 0 <= int(policy_action) < self.action_dim:
            raise ValueError(f"policy_action must be in [0, {self.action_dim})")
        constrained_action = int(action)
        requested_action = constrained_action if policy_action is None else int(policy_action)
        learning_transition = self._learning_active
        # Action zero is [0 axial g, 1 normal g, 0 bank], exactly cancelling
        # gravity for unchanged straight-and-level flight before detection.
        executed_action = constrained_action if learning_transition else 0
        target_command = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[executed_action]
        starting_command = self._applied_load_command_body_g.copy()
        bank_delta = math.atan2(math.sin(float(target_command[2] - starting_command[2])),
                                math.cos(float(target_command[2] - starting_command[2])))
        result = None
        for frame_index in range(self.frames_per_action):
            assert self.inner.state is not None
            red = RedAction(np.zeros(len(self.inner.state.red), np.int64), np.zeros((len(self.inner.state.red), 2)))
            # Linearly interpolate the post-constraint command over the 0.1 s
            # decision interval.  This keeps the physical actuator input
            # continuous while the learner's action remains the discrete target.
            fraction = (frame_index + 1) / self.frames_per_action
            applied = starting_command + fraction * (target_command - starting_command)
            applied[2] = starting_command[2] + fraction * bank_delta
            self._applied_load_command_body_g = applied.copy()
            result = self.inner.step(
                red_action=red,
                blue_action={"load_command_body_g": [applied.tolist()]},
            )
            self._activate_if_threat_observed()
            if self._record_current_episode:
                self.recorder.record(self.inner.state)
            if result.done: break
        assert result is not None and self.inner.state is not None
        self._previous_executed_action_index = executed_action
        blue_alive = self.inner.state.blue[0].alive
        measured_potential = self._threat_potential()
        potential_before = self._previous_potential["total"]
        if learning_transition:
            # A terminal MDP state has Phi=0.  Using the learner's discount
            # gives policy-invariant shaping: gamma*Phi(s')-Phi(s).
            current_potential = self._zero_potential() if result.done else measured_potential
            shaping = {
                name: self.config.shaping_discount * current_potential[name] - self._previous_potential[name]
                for name in ("far_away", "near_tangent", "near_dive")
            }
            terminal_reward = self._terminal_reward(result.info) if result.done else 0.0
        else:
            # The transition that first crosses the detection boundary belongs
            # to straight-flight warmup.  It establishes the first RL state but
            # is never rewarded or inserted into replay.
            current_potential = (
                measured_potential if self._learning_active and not result.done else self._zero_potential()
            )
            shaping = {"far_away": 0.0, "near_tangent": 0.0, "near_dive": 0.0}
            terminal_reward = 0.0
        shaping_reward = sum(shaping.values())
        potential_after = current_potential["total"]
        self._previous_potential = current_potential
        reward = shaping_reward + terminal_reward
        terminated, truncated = bool(result.terminated), bool(result.truncated)
        info = dict(result.info); info.update({
            "pure_pn": True,
            "missile_slot_mask": [index < self._missile_count for index in range(self.config.max_missiles)],
            "blue_survived": blue_alive,
            "reward_components": {
                "tactical_shaping": float(shaping_reward),
                "far_away_shaping": float(shaping["far_away"]),
                "near_tangent_shaping": float(shaping["near_tangent"]),
                "near_dive_shaping": float(shaping["near_dive"]),
                "terminal": float(terminal_reward),
            },
            "reward_diagnostics": {
                "range_blend_weight": float(measured_potential["range_blend_weight"]),
                "softmin_threat_distance": float(measured_potential["softmin_threat_distance"]),
                "potential_before": float(potential_before),
                "potential_after": float(potential_after),
                "measured_potential_after": float(measured_potential["total"]),
            },
            # Kept separate from the normalized observation so diagnostics can
            # evolve without changing a checkpoint's observation schema.
            "flight_quality_state": self._mechanism_snapshot(),
            **self._learning_status(
                learning_transition=learning_transition,
                requested_action=requested_action,
                constrained_action=constrained_action,
                executed_action=executed_action,
            ),
            "target_load_command_body_g": target_command.tolist(),
            "actual_load_command_body_g": self._applied_load_command_body_g.tolist(),
        })
        if self.config.expose_evaluation_mechanism_state:
            info["mechanism_state"] = self._mechanism_snapshot()
        if result.done:
            info["red_loss_reasons"] = [item.loss_reason or "unknown" for item in self.inner.state.red]
        if result.done and self._record_current_episode:
            path = Path(self.config.acmi_directory) / f"episode_{self.episode:06d}.acmi"
            info["acmi_path"] = str(self.recorder.save(path))
        return self._observation(), float(reward), terminated, truncated, info

    def _minimum_live_missile_range_m(self) -> float | None:
        assert self.inner.state is not None
        blue = self.inner.state.blue[0]
        distances = [
            float(np.linalg.norm(red.position_m - blue.position_m))
            for red in self.inner.state.red
            if red.alive
        ]
        return min(distances) if distances else None

    def _activate_if_threat_observed(self) -> bool:
        if self._learning_active:
            return False
        minimum_range_m = self._minimum_live_missile_range_m()
        if minimum_range_m is None or minimum_range_m >= self.config.threat_detection_range_m:
            return False
        assert self.inner.state is not None
        self._learning_active = True
        self._activation_time_s = float(self.inner.state.time_s)
        self._activation_range_m = minimum_range_m
        return True

    def _learning_status(self, *, learning_transition: bool, requested_action: int,
                         constrained_action: int, executed_action: int) -> dict[str, object]:
        return {
            "learning_active": bool(self._learning_active),
            "learning_transition": bool(learning_transition),
            "threat_detection_range_m": float(self.config.threat_detection_range_m),
            "minimum_live_missile_range_m": self._minimum_live_missile_range_m(),
            "learning_activation_time_s": self._activation_time_s,
            "learning_activation_range_m": self._activation_range_m,
            "requested_action_index": int(requested_action),
            "constrained_action_index": int(constrained_action),
            "executed_action_index": int(executed_action),
        }

    @staticmethod
    def _zero_potential() -> dict[str, float]:
        return {
            "far_away": 0.0, "near_tangent": 0.0, "near_dive": 0.0,
            "total": 0.0, "range_blend_weight": 0.0, "softmin_threat_distance": 0.0,
        }

    def _threat_potential(self) -> dict[str, float]:
        """Bounded multi-missile potential encoding the desired two-stage tactic.

        At long range it rewards flying directly away from incoming missiles.
        As range falls below ``near_range_m`` it smoothly changes to rewarding
        tangential velocity and a downward flight-path component.  A soft-min
        weighting keeps every live missile relevant without hard identity jumps.
        """
        assert self.inner.state is not None
        blue = self.inner.state.blue[0]
        speed = float(np.linalg.norm(blue.velocity_mps))
        if not blue.alive or speed <= 1.0e-9:
            return self._zero_potential()
        velocity_hat = blue.velocity_mps / speed
        far_scores: list[float] = []; tangent_scores: list[float] = []
        dive_scores: list[float] = []; near_gates: list[float] = []
        distances: list[float] = []; logits: list[float] = []
        for red in self.inner.state.red:
            if not red.alive:
                continue
            missile_to_blue = blue.position_m - red.position_m
            distance = float(np.linalg.norm(missile_to_blue))
            if distance <= 1.0e-9:
                continue
            away_hat = missile_to_blue / distance
            away_score = 0.5 * (1.0 + float(np.dot(velocity_hat, away_hat)))
            tangent_score = 1.0 - abs(float(np.dot(velocity_hat, away_hat)))
            dive_score = max(0.0, -float(velocity_hat[1]))
            near_gate = 1.0 / (1.0 + math.exp(np.clip(
                (distance - self.config.near_range_m) / self.config.range_transition_m, -60.0, 60.0
            )))
            far_scores.append((1.0 - near_gate) * self.config.far_away_weight * away_score)
            tangent_scores.append(near_gate * self.config.near_tangent_weight * tangent_score)
            dive_scores.append(near_gate * self.config.near_dive_weight * dive_score)
            near_gates.append(near_gate); distances.append(distance)
            logits.append(-distance / self.config.threat_softmin_temperature_m)
        if not far_scores:
            return self._zero_potential()
        shifted = np.asarray(logits) - max(logits)
        weights = np.exp(shifted); weights /= weights.sum()
        components = {
            "far_away": self.config.shaping_scale * float(np.dot(weights, far_scores)),
            "near_tangent": self.config.shaping_scale * float(np.dot(weights, tangent_scores)),
            "near_dive": self.config.shaping_scale * float(np.dot(weights, dive_scores)),
        }
        minimum_distance = min(distances)
        softmin_distance = minimum_distance - self.config.threat_softmin_temperature_m * math.log(sum(
            math.exp(-(distance - minimum_distance) / self.config.threat_softmin_temperature_m)
            for distance in distances
        ))
        return {
            **components,
            "total": sum(components.values()),
            "range_blend_weight": float(np.dot(weights, near_gates)),
            "softmin_threat_distance": float(softmin_distance),
        }

    def _terminal_reward(self, info: dict[str, object]) -> float:
        reason = str(info.get("termination_reason", "none"))
        horizon = max(self.environment_config.policy_horizon_s, self.config.decision_interval_s)
        elapsed = max(0.0, float(info.get("time_s", 0.0)) - self.environment_config.policy_entry_time_s)
        progress = float(np.clip(elapsed / horizon, 0.0, 1.0))
        if reason == "success":  # adjudication names this from the red side
            return self.config.terminal_killed_reward + self.config.survival_progress_bonus * progress
        if reason == "red_failure":
            return self.config.terminal_success_reward + self.config.fast_success_bonus * (1.0 - progress)
        if reason == "timeout":
            return self.config.terminal_timeout_reward
        raise RuntimeError(f"unexpected terminal reason: {reason}")

    def _observation(self) -> np.ndarray:
        assert self.inner.state is not None
        state = self.inner.state
        blue = state.blue[0]
        if self.config.observation_schema in {"normalized_v2", "normalized_v3"}:
            # Versioned, dimensionless input.  Horizontal position is relative
            # to the blue aircraft (altitude remains absolute because ground
            # clearance matters).  A per-slot validity bit removes zero-padding
            # ambiguity in mixed 1v1--1v4 training.
            values = [0.0, blue.position_m[1] / 20000.0, 0.0,
                      *(blue.velocity_mps / 2000.0)]
            for red in state.red:
                values.extend((red.position_m - blue.position_m) / 200000.0)
                values.append(1.0)
            if self.config.pad_observation_to_max_missiles:
                values.extend([0.0] * (4 * (self.config.max_missiles - len(state.red))))
            if self.config.observation_schema == "normalized_v3":
                values.extend(blue_action_context(
                    self._previous_executed_action_index,
                    self._applied_load_command_body_g,
                ))
            return np.asarray(values, dtype=np.float32)
        # Legacy checkpoint contract: kilometres and kilometres/second.
        values = [*(blue.position_m / 1000.0), *(blue.velocity_mps / 1000.0)]
        for red in state.red:
            values.extend((red.position_m - blue.position_m) / 1000.0)
        if self.config.pad_observation_to_max_missiles:
            values.extend([0.0] * (3 * (self.config.max_missiles - len(state.red))))
        return np.asarray(values, dtype=np.float32)
