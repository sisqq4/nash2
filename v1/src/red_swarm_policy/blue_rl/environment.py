from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from ..env.environment import RedBlueEngagementEnv
from ..env.types import EnvironmentConfig, RedAction
from .acmi import AcmiRecorder


@dataclass(frozen=True)
class BlueEscapeEnvConfig:
    """Blue-RL adapter settings; physical/scenario values remain in EnvironmentConfig."""

    missile_count: int = 1
    max_missiles: int = 4
    decision_interval_s: float = 0.1
    record_acmi: bool = True
    acmi_episode_interval: int = 1
    acmi_directory: str = "outputs/blue_rl/acmi"
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
        # Preserve nash1.6's observation contract exactly: blue absolute
        # position/velocity followed by every missile's relative position.
        # A checkpoint is therefore tied to its configured missile count.
        self.observation_dim = 6 + config.missile_count * 3
        self.action_dim = len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)
        self.recorder = AcmiRecorder(); self.episode = 0
        self._previous_potential: dict[str, float] = {}
        self._record_current_episode = False

    def reset(self, seed: int | None = None, *, episode_index: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        self.episode = self.episode + 1 if episode_index is None else int(episode_index)
        if self.episode < 1:
            raise ValueError("episode_index must be positive")
        self.recorder = AcmiRecorder()
        self._record_current_episode = bool(
            self.config.record_acmi
            and self.config.acmi_episode_interval > 0
            and self.episode % self.config.acmi_episode_interval == 0
        )
        self.inner.reset(seed=seed, style="many_to_one", red_count=self.config.missile_count,
                         blue_count=1, start_mode="post_boost")
        assert self.inner.state is not None
        if self._record_current_episode:
            self.recorder.record(self.inner.state)
        self._previous_potential = self._threat_potential()
        return self._observation(), {"time_s": self.inner.state.time_s, "pure_pn": True}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if not 0 <= int(action) < self.action_dim: raise ValueError(f"action must be in [0, {self.action_dim})")
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
        reward = shaping_reward + terminal_reward
        terminated, truncated = bool(result.terminated), bool(result.truncated)
        info = dict(result.info); info.update({
            "pure_pn": True,
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
        })
        if result.done:
            info["red_loss_reasons"] = [item.loss_reason or "unknown" for item in self.inner.state.red]
        if result.done and self._record_current_episode:
            path = Path(self.config.acmi_directory) / f"episode_{self.episode:06d}.acmi"
            info["acmi_path"] = str(self.recorder.save(path))
        return self._observation(), float(reward), terminated, truncated, info

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
        # nash1.6 exposes kilometres and kilometres/second to the network.
        values = [*(blue.position_m / 1000.0), *(blue.velocity_mps / 1000.0)]
        for red in state.red:
            values.extend((red.position_m - blue.position_m) / 1000.0)
        return np.asarray(values, dtype=np.float32)
