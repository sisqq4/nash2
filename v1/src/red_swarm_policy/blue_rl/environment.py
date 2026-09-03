from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from ..env.environment import RedBlueEngagementEnv
from ..env.math_utils import G0
from ..env.types import EnvironmentConfig, RedAction
from .acmi import AcmiRecorder


@dataclass(frozen=True)
class BlueEscapeEnvConfig:
    """Blue-RL adapter settings; physical/scenario values remain in EnvironmentConfig."""

    missile_count: int = 1
    max_missiles: int = 4
    pad_observation_to_max_missiles: bool = False
    observation_schema: str = "legacy_v1"
    decision_interval_s: float = 0.1
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
    action_switch_penalty: float = 0.50
    opposite_maneuver_penalty: float = 1.00
    climb_rate_penalty: float = 0.50
    descent_rate_penalty: float = 0.25
    overload_penalty: float = 0.50
    lateral_speed_penalty: float = 0.50
    vertical_speed_deadband_mps: float = 20.0
    overload_soft_limit_g: float = 6.0
    lateral_speed_limit_mps: float = 5.0
    opposite_maneuver_cosine: float = -0.25

    def validate(self, environment: EnvironmentConfig) -> None:
        if not 1 <= self.missile_count <= self.max_missiles <= 4:
            raise ValueError("blue training supports one to four missiles against one aircraft")
        if self.observation_schema not in {"legacy_v1", "normalized_v2", "normalized_v3"}:
            raise ValueError("observation_schema must be 'legacy_v1', 'normalized_v2', or 'normalized_v3'")
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
            self.action_switch_penalty, self.opposite_maneuver_penalty,
            self.climb_rate_penalty, self.descent_rate_penalty,
            self.overload_penalty, self.lateral_speed_penalty,
            self.vertical_speed_deadband_mps, self.overload_soft_limit_g,
            self.lateral_speed_limit_mps, self.opposite_maneuver_cosine,
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("blue reward configuration values must be finite")
        if (
            self.terminal_success_reward <= self.terminal_timeout_reward
            or self.terminal_timeout_reward <= self.terminal_killed_reward
            or min(self.survival_progress_bonus, self.fast_success_bonus, self.shaping_scale) < 0.0
            or not 0.0 < self.shaping_discount <= 1.0
            or min(self.near_range_m, self.range_transition_m, self.threat_softmin_temperature_m) <= 0.0
            or min(self.far_away_weight, self.near_tangent_weight, self.near_dive_weight) < 0.0
            or not np.isclose(self.near_tangent_weight + self.near_dive_weight, 1.0)
            or min(self.action_switch_penalty, self.opposite_maneuver_penalty,
                   self.climb_rate_penalty, self.descent_rate_penalty,
                   self.overload_penalty, self.lateral_speed_penalty) < 0.0
            or not 0.0 <= self.vertical_speed_deadband_mps < environment.aircraft.max_speed_mps
            or not 0.0 < self.overload_soft_limit_g <= environment.aircraft.max_load_factor_g
            or not 0.0 <= self.lateral_speed_limit_mps < environment.aircraft.max_speed_mps
            or not -1.0 <= self.opposite_maneuver_cosine < 0.0
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
        config.validate(environment_config)
        self.environment_config, self.config = environment_config, config
        self.inner = RedBlueEngagementEnv(environment_config, record_replay=False)
        self.frames_per_action = int(round(config.decision_interval_s / environment_config.time_step_s))
        # Single-scenario runs retain the nash1.6 observation contract.  A
        # multi-scenario run pads missing missile slots to max_missiles so one
        # policy can consume every selected scenario.
        slots = config.max_missiles if config.pad_observation_to_max_missiles else config.missile_count
        normalized = config.observation_schema in {"normalized_v2", "normalized_v3"}
        self.observation_dim = 6 + slots * (4 if normalized else 3)
        if config.observation_schema == "normalized_v3":
            self.observation_dim += 3
        self.action_dim = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)
        self.recorder = AcmiRecorder(); self.episode = 0
        self._previous_potential: dict[str, float] = {}
        self._previous_action: int | None = 0
        self._record_current_episode = False

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
        self._previous_potential = self._threat_potential()
        # Level flight is the explicit pre-decision command represented in the
        # normalized_v3 observation and used for the first switch comparison.
        self._previous_action = 0
        info = {
            "time_s": self.inner.state.time_s,
            "pure_pn": True,
            "missile_slot_mask": [index < self._missile_count for index in range(self.config.max_missiles)],
            "initialization": self._initialization_snapshot(),
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
                "max_load_factor_g": self.environment_config.aircraft.max_load_factor_g}

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

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if not 0 <= int(action) < self.action_dim: raise ValueError(f"action must be in [0, {self.action_dim})")
        assert self.inner.state is not None
        velocity_before = self.inner.state.blue[0].velocity_mps.copy()
        result = None
        for _ in range(self.frames_per_action):
            assert self.inner.state is not None
            red = RedAction(np.zeros(len(self.inner.state.red), np.int64), np.zeros((len(self.inner.state.red), 2)))
            result = self.inner.step(red_action=red, blue_action={"action_indices": [int(action)]})
            if self._record_current_episode:
                self.recorder.record(self.inner.state)
            if result.done: break
        assert result is not None and self.inner.state is not None
        blue_alive = self.inner.state.blue[0].alive
        measured_potential = self._threat_potential()
        # A terminal MDP state has Phi=0.  Using the learner's discount here is
        # required for policy-invariant potential shaping: gamma*Phi(s')-Phi(s).
        current_potential = self._zero_potential() if result.done else measured_potential
        shaping = {
            name: self.config.shaping_discount * current_potential[name] - self._previous_potential[name]
            for name in ("far_away", "near_tangent", "near_dive")
        }
        shaping_reward = sum(shaping.values())
        potential_before = self._previous_potential["total"]
        potential_after = current_potential["total"]
        self._previous_potential = current_potential
        terminal_reward = self._terminal_reward(result.info) if result.done else 0.0
        maneuver_penalties, maneuver_diagnostics = self._maneuver_penalties(int(action), velocity_before)
        maneuver_penalty = sum(maneuver_penalties.values())
        self._previous_action = int(action)
        reward = shaping_reward + terminal_reward - maneuver_penalty
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
                **{name: -float(value) for name, value in maneuver_penalties.items()},
            },
            "reward_diagnostics": {
                "range_blend_weight": float(measured_potential["range_blend_weight"]),
                "softmin_threat_distance": float(measured_potential["softmin_threat_distance"]),
                "potential_before": float(potential_before),
                "potential_after": float(potential_after),
                "measured_potential_after": float(measured_potential["total"]),
                **maneuver_diagnostics,
            },
        })
        if self.config.expose_evaluation_mechanism_state:
            info["mechanism_state"] = self._mechanism_snapshot()
        if result.done:
            info["red_loss_reasons"] = [item.loss_reason or "unknown" for item in self.inner.state.red]
        if result.done and self._record_current_episode:
            path = Path(self.config.acmi_directory) / f"episode_{self.episode:06d}.acmi"
            info["acmi_path"] = str(self.recorder.save(path))
        return self._observation(), float(reward), terminated, truncated, info

    def _maneuver_penalties(
        self, action: int, velocity_before: np.ndarray | None = None
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return bounded, decision-rate regularizers for physically plausible flight.

        Switching and reversal use maneuver load (with level-flight gravity support
        removed), while the state terms constrain the resulting trajectory.  This
        keeps the shaping independent of the number of 0.005 s physics substeps.
        """
        assert self.inner.state is not None
        command = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[action]
        previous = None if self._previous_action is None else BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[self._previous_action]

        def maneuver_vector(value: np.ndarray) -> np.ndarray:
            axial, normal, bank = value
            return np.array([axial, (normal * math.cos(bank)) - 1.0,
                             -normal * math.sin(bank)], dtype=np.float64)

        current_vector = maneuver_vector(command)
        switch_severity = 0.0
        reversal_severity = 0.0
        maneuver_cosine = 1.0
        if previous is not None:
            previous_vector = maneuver_vector(previous)
            load_delta = float(np.linalg.norm(current_vector - previous_vector))
            switch_severity = min(1.0, load_delta / (2.0 * self.environment_config.aircraft.max_load_factor_g))
            norms = float(np.linalg.norm(current_vector) * np.linalg.norm(previous_vector))
            if norms > 1.0e-9:
                maneuver_cosine = float(np.clip(np.dot(current_vector, previous_vector) / norms, -1.0, 1.0))
                reversal_severity = float(np.clip(
                    (self.config.opposite_maneuver_cosine - maneuver_cosine)
                    / (1.0 + self.config.opposite_maneuver_cosine), 0.0, 1.0
                ))

        blue = self.inner.state.blue[0]
        before = blue.velocity_mps if velocity_before is None else np.asarray(velocity_before, dtype=np.float64)
        if before.shape != (3,) or not np.all(np.isfinite(before)):
            raise ValueError("velocity_before must be a finite three-vector")
        vertical_speed = float(blue.velocity_mps[1])
        vertical_excess = max(0.0, abs(vertical_speed) - self.config.vertical_speed_deadband_mps)
        vertical_severity = min(1.0, vertical_excess / max(
            self.environment_config.aircraft.max_speed_mps - self.config.vertical_speed_deadband_mps, 1.0
        ))
        load_g = float(np.hypot(command[0], command[1]))
        overload_severity = float(np.clip(
            (load_g - self.config.overload_soft_limit_g)
            / max(self.environment_config.aircraft.max_load_factor_g - self.config.overload_soft_limit_g, 1.0e-9),
            0.0, 1.0,
        ))
        horizontal_before = before[[0, 2]]
        horizontal_speed_before = float(np.linalg.norm(horizontal_before))
        forward = (horizontal_before / horizontal_speed_before
                   if horizontal_speed_before > 1.0e-9 else np.array([1.0, 0.0]))
        lateral_direction = np.array([-forward[1], forward[0]])
        lateral_speed = abs(float(np.dot(
            blue.velocity_mps[[0, 2]] - horizontal_before, lateral_direction
        )))
        maximum_lateral_delta = (
            self.environment_config.aircraft.max_load_factor_g * G0 * self.config.decision_interval_s
        )
        lateral_severity = float(np.clip(
            (lateral_speed - self.config.lateral_speed_limit_mps)
            / max(maximum_lateral_delta - self.config.lateral_speed_limit_mps, 1.0e-9),
            0.0, 1.0,
        ))
        # Configured weights are full-horizon budgets.  Scaling by the fraction
        # of one decision in the mission prevents dense penalties from growing
        # large enough to reverse the terminal survival preference.
        budget_scale = self.config.decision_interval_s / max(
            self.environment_config.policy_horizon_s, self.config.decision_interval_s
        )
        penalties = {
            "action_switch_penalty": budget_scale * self.config.action_switch_penalty * switch_severity,
            "opposite_maneuver_penalty": budget_scale * self.config.opposite_maneuver_penalty * reversal_severity,
            "climb_rate_penalty": budget_scale * self.config.climb_rate_penalty * vertical_severity if vertical_speed > 0.0 else 0.0,
            "descent_rate_penalty": budget_scale * self.config.descent_rate_penalty * vertical_severity if vertical_speed < 0.0 else 0.0,
            "overload_penalty": budget_scale * self.config.overload_penalty * overload_severity ** 2,
            "lateral_speed_penalty": budget_scale * self.config.lateral_speed_penalty * lateral_severity ** 2,
        }
        diagnostics = {
            "maneuver_switch_severity": switch_severity,
            "maneuver_cosine": maneuver_cosine,
            "vertical_speed_mps": vertical_speed,
            "commanded_load_g": load_g,
            "lateral_velocity_change_mps": lateral_speed,
            "maneuver_penalty_budget_scale": budget_scale,
        }
        return penalties, diagnostics

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
                previous = 0 if self._previous_action is None else self._previous_action
                command = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[previous]
                values.extend((command[:2] / self.environment_config.aircraft.max_load_factor_g).tolist())
                values.append(float(command[2] / math.pi))
            return np.asarray(values, dtype=np.float32)
        # Legacy checkpoint contract: kilometres and kilometres/second.
        values = [*(blue.position_m / 1000.0), *(blue.velocity_mps / 1000.0)]
        for red in state.red:
            values.extend((red.position_m - blue.position_m) / 1000.0)
        if self.config.pad_observation_to_max_missiles:
            values.extend([0.0] * (3 * (self.config.max_missiles - len(state.red))))
        return np.asarray(values, dtype=np.float32)
