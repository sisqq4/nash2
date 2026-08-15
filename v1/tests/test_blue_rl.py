from __future__ import annotations

from dataclasses import replace

import numpy as np

from red_swarm_policy.blue_rl import BlueEscapeEnv, BlueEscapeEnvConfig, BlueRLController
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
