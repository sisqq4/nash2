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
from red_swarm_policy.blue_rl.config_io import configure_blue_mission_duration
from red_swarm_policy.env import EnvironmentConfig
from red_swarm_policy.cli_utils import parse_missile_scenarios


class FixedPolicy:
    def select_action(self, observation: np.ndarray, *, evaluation: bool = False) -> int:
        assert observation.ndim == 1
        return 0

    def save(self, path: str) -> None:
        pass


def short_config() -> EnvironmentConfig:
    base = EnvironmentConfig()
    return replace(base, max_steps=base.policy_entry_steps + 2)


def test_blue_cli_duration_sets_mission_and_guidance_to_200_seconds() -> None:
    config = configure_blue_mission_duration(EnvironmentConfig())
    assert config.max_steps * config.time_step_s == pytest.approx(200.0)
    assert config.missile.max_guidance_time_s == pytest.approx(200.0)


def test_rainbow_defaults_match_long_decision_horizon() -> None:
    config = RainbowDQNConfig(9, 29)
    assert config.gamma == pytest.approx(0.999)
    assert config.n_step == 20


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


def test_multi_scenario_observations_are_padded_to_shared_shape() -> None:
    cfg = short_config()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(
        missile_count=1, max_missiles=4, pad_observation_to_max_missiles=True,
        decision_interval_s=cfg.time_step_s, record_acmi=False,
    ))
    one, _ = env.reset(seed=1, missile_count=1)
    four, _ = env.reset(seed=2, missile_count=4)
    assert one.shape == four.shape == (18,)
    assert np.allclose(one[9:], 0.0)


@pytest.mark.parametrize(("value", "expected"), [("4", (4,)), ("1,2,3,4", (1, 2, 3, 4)),
                                                   ("1,3,3", (1, 3))])
def test_parse_missile_scenarios(value: str, expected: tuple[int, ...]) -> None:
    assert parse_missile_scenarios(value) == expected


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


def test_blue_reward_prefers_away_far_and_tangent_dive_near() -> None:
    cfg = EnvironmentConfig()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(missile_count=1, record_acmi=False))
    env.reset(seed=7)
    assert env.inner.state is not None
    blue, missile = env.inner.state.blue[0], env.inner.state.red[0]
    missile.position_m = blue.position_m + np.array([100000.0, 0.0, 0.0])
    blue.velocity_mps = np.array([-350.0, 0.0, 0.0])
    far_away = env._threat_potential()["total"]
    blue.velocity_mps = np.array([350.0, 0.0, 0.0])
    assert far_away > env._threat_potential()["total"]

    missile.position_m = blue.position_m + np.array([10000.0, 0.0, 0.0])
    blue.velocity_mps = np.array([0.0, -247.5, 247.5])
    tangent_dive = env._threat_potential()["total"]
    blue.velocity_mps = np.array([-350.0, 0.0, 0.0])
    assert tangent_dive > env._threat_potential()["total"]


def test_potential_components_sum_and_include_multi_threat_diagnostics() -> None:
    env = BlueEscapeEnv(EnvironmentConfig(), BlueEscapeEnvConfig(missile_count=2, record_acmi=False))
    env.reset(seed=9)
    potential = env._threat_potential()
    assert potential["total"] == pytest.approx(
        potential["far_away"] + potential["near_tangent"] + potential["near_dive"]
    )
    assert 0.0 <= potential["range_blend_weight"] <= 1.0
    assert potential["softmin_threat_distance"] > 0.0


def test_terminal_reward_distinguishes_miss_timeout_and_red_success() -> None:
    env = BlueEscapeEnv(EnvironmentConfig(), BlueEscapeEnvConfig(record_acmi=False))
    assert env._terminal_reward({"termination_reason": "red_failure", "time_s": 20.0}) > 10.0
    assert env._terminal_reward({"termination_reason": "timeout", "time_s": 180.0}) == 2.0
    killed = env._terminal_reward({"termination_reason": "success", "time_s": 20.0})
    assert -10.0 < killed < -9.0


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
