from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from red_swarm_policy.blue_rl import (
    BlueEscapeEnv,
    BlueEscapeEnvConfig,
    BlueProcessEnvironmentPool,
    BlueRLController,
    RainbowDQNAgent,
    RainbowDQNConfig,
)
from red_swarm_policy.env import EnvironmentConfig


class FixedPolicy:
    def select_action(self, observation: np.ndarray, *, evaluation: bool = False) -> int:
        assert observation.ndim == 1
        return 0

    def save(self, path: str) -> None:
        pass


def short_config() -> EnvironmentConfig:
    base = EnvironmentConfig()
    return replace(base, max_steps=base.policy_entry_steps + 2)


def test_blue_env_is_fixed_shape_and_uses_pure_pn(tmp_path) -> None:
    cfg = short_config()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(missile_count=2, decision_interval_s=cfg.time_step_s,
                                                  acmi_directory=str(tmp_path)))
    observation, info = env.reset(seed=3)
    assert observation.shape == (env.observation_dim,) == (12,)
    assert info["pure_pn"] is True
    _, _, terminated, truncated, info = env.step(0)
    assert terminated or truncated
    assert info["pure_pn"] is True
    assert np.allclose(env.inner.previous_action.red.guidance_bias, 0.0)
    assert (tmp_path / "episode_000001.acmi").is_file()


def test_controller_is_drop_in_discrete_policy() -> None:
    cfg = EnvironmentConfig(); env = BlueEscapeEnv(cfg)
    env.reset(seed=2)
    controller = BlueRLController(FixedPolicy(), cfg, BlueEscapeEnvConfig(missile_count=1))
    action = controller(env.inner.state)
    assert action["action_indices"].tolist() == [0]


def test_acmi_interval_skips_unscheduled_episodes(tmp_path) -> None:
    cfg = short_config()
    env = BlueEscapeEnv(
        cfg,
        BlueEscapeEnvConfig(
            missile_count=1,
            decision_interval_s=cfg.time_step_s,
            acmi_episode_interval=2,
            acmi_directory=str(tmp_path),
        ),
    )
    for episode in (1, 2):
        env.reset(seed=episode)
        _, _, terminated, truncated, info = env.step(0)
        assert terminated or truncated
        assert ("acmi_path" in info) is (episode == 2)
    assert not (tmp_path / "episode_000001.acmi").exists()
    assert (tmp_path / "episode_000002.acmi").is_file()


def test_rainbow_select_actions_batches_observations() -> None:
    agent = RainbowDQNAgent(RainbowDQNConfig(9, 29, hidden_dim=16))
    observations = np.zeros((4, 9), dtype=np.float32)

    actions = agent.select_actions(observations, evaluation=True)

    assert actions.shape == (4,)
    assert actions.dtype == np.int64
    assert np.all((0 <= actions) & (actions < 29))


def test_parallel_observations_keep_independent_n_step_sequences() -> None:
    agent = RainbowDQNAgent(RainbowDQNConfig(2, 2, n_step=2, learning_starts=100))
    zero = np.zeros(2, dtype=np.float32)

    agent.observe_for_env(0, zero, 0, 1.0, zero, False)
    agent.observe_for_env(1, zero, 1, 10.0, zero, False)
    agent.observe_for_env(0, zero, 0, 2.0, zero, True)

    assert agent.replay.size == 2
    assert agent.replay.actions[:2].tolist() == [0, 0]
    assert agent.replay.rewards[0] == pytest.approx(1.0 + agent.config.gamma * 2.0)


def test_parallel_updates_sync_target_only_once_per_step_threshold() -> None:
    agent = RainbowDQNAgent(
        RainbowDQNConfig(2, 2, batch_size=1, learning_starts=1, n_step=1,
                         target_update_interval=2, hidden_dim=16)
    )
    zero = np.zeros(2, dtype=np.float32)
    agent.observe_for_env(0, zero, 0, 0.0, zero, False)
    agent.observe_for_env(1, zero, 1, 0.0, zero, False)

    assert agent.update() is not None
    assert agent.update() is not None
    assert agent.optimizer_updates == 2
    assert agent.target_updates == 1
    assert agent.last_update_metrics["replay_size"] == 2.0


def test_process_blue_pool_uses_global_episode_numbers(tmp_path) -> None:
    cfg = short_config()
    blue = BlueEscapeEnvConfig(missile_count=1, decision_interval_s=cfg.time_step_s,
                               acmi_episode_interval=2, acmi_directory=str(tmp_path))
    with BlueProcessEnvironmentPool(cfg, blue, 2, timeout_s=30.0) as pool:
        reset = pool.reset({0: (11, 1), 1: (12, 2)})
        assert sorted(reset) == [0, 1]
        results = pool.step({0: 0, 1: 0})
        assert all(result.terminated or result.truncated for result in results.values())
    assert not (tmp_path / "episode_000001.acmi").exists()
    assert (tmp_path / "episode_000002.acmi").is_file()
