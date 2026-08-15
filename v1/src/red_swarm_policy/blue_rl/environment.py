from __future__ import annotations

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
        self.recorder = AcmiRecorder(); self.episode = 0; self._previous_min_range = 0.0
        self._record_current_episode = False

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict[str, object]]:
        self.episode += 1; self.recorder = AcmiRecorder()
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
        self._previous_min_range = self._minimum_range()
        return self._observation(), {"time_s": self.inner.state.time_s, "pure_pn": True}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        if not 0 <= int(action) < self.action_dim: raise ValueError(f"action must be in [0, {self.action_dim})")
        reward = 0.0; result = None
        for _ in range(self.frames_per_action):
            assert self.inner.state is not None
            red = RedAction(np.zeros(len(self.inner.state.red), np.int64), np.zeros((len(self.inner.state.red), 2)))
            result = self.inner.step(red_action=red, blue_action={"action_indices": [int(action)]})
            current = self._minimum_range()
            reward += np.clip((current - self._previous_min_range) / 1000.0, -1.0, 1.0) * 0.05
            self._previous_min_range = current
            if self._record_current_episode:
                self.recorder.record(self.inner.state)
            if result.done: break
        assert result is not None and self.inner.state is not None
        blue_alive = self.inner.state.blue[0].alive
        red_alive = any(item.alive for item in self.inner.state.red)
        if result.done: reward += 10.0 if blue_alive and not red_alive else -10.0 if not blue_alive else 3.0
        terminated, truncated = bool(result.terminated), bool(result.truncated)
        info = dict(result.info); info.update({"pure_pn": True, "blue_survived": blue_alive})
        if result.done and self._record_current_episode:
            path = Path(self.config.acmi_directory) / f"episode_{self.episode:06d}.acmi"
            info["acmi_path"] = str(self.recorder.save(path))
        return self._observation(), float(reward), terminated, truncated, info

    def _minimum_range(self) -> float:
        assert self.inner.state is not None
        blue = self.inner.state.blue[0]
        distances = [np.linalg.norm(red.position_m - blue.position_m) for red in self.inner.state.red if red.alive]
        return float(min(distances, default=self.environment_config.missile.escape_range_m))

    def _observation(self) -> np.ndarray:
        assert self.inner.state is not None
        state = self.inner.state
        blue = state.blue[0]
        # nash1.6 exposes kilometres and kilometres/second to the network.
        values = [*(blue.position_m / 1000.0), *(blue.velocity_mps / 1000.0)]
        for red in state.red:
            values.extend((red.position_m - blue.position_m) / 1000.0)
        return np.asarray(values, dtype=np.float32)
