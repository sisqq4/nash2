from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from red_swarm_policy.blue_rl import (
    BlueEscapeEnv,
    BlueEscapeEnvConfig,
    BlueProcessEnvironmentPool,
    BlueRLController,
    EvaluationActionShaper,
    EvaluationShapingConfig,
    RainbowDQNAgent,
    RainbowDQNConfig,
)
from red_swarm_policy.blue_rl.config_io import configure_blue_mission_duration
from red_swarm_policy.env import EnvironmentConfig
from red_swarm_policy.cli_utils import parse_missile_scenarios
from red_swarm_policy.blue_rl.curriculum import CurriculumSchedule, balanced_score, within_forgetting_limit
from red_swarm_policy.evaluate_blue_rl import _emit as emit_evaluation_event
from red_swarm_policy.evaluate_blue_rl import _aggregate_results
from red_swarm_policy.evaluate_blue_rl import _numeric_distribution
from red_swarm_policy.evaluate_blue_rl import _one_meter_probability_histogram
from red_swarm_policy.evaluate_blue_rl import build_parser as build_evaluation_parser


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
    one, info = env.reset(seed=1, missile_count=1)
    four, _ = env.reset(seed=2, missile_count=4)
    assert one.shape == four.shape == (18,)
    assert np.allclose(one[9:], 0.0)
    assert info["missile_slot_mask"] == [True, False, False, False]


def test_normalized_v2_observation_is_dimensionless_and_masks_padding() -> None:
    cfg = short_config()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(
        missile_count=1, max_missiles=4, pad_observation_to_max_missiles=True,
        observation_schema="normalized_v2", decision_interval_s=cfg.time_step_s,
        record_acmi=False,
    ))
    observation, _ = env.reset(seed=1, missile_count=1)
    assert observation.shape == (22,)
    assert observation[0] == observation[2] == 0.0
    assert 0.0 < observation[1] < 1.0
    assert np.linalg.norm(observation[3:6]) < 1.0
    assert observation[9] == 1.0
    assert np.allclose(observation[10:], 0.0)

    assert env.inner.state is not None
    translated = observation.copy()
    for entity in [*env.inner.state.red, *env.inner.state.blue]:
        entity.position_m += np.array([50000.0, 0.0, 30000.0])
    assert env._observation() == pytest.approx(translated)


def test_normalized_v3_observation_exposes_previous_action_state() -> None:
    cfg = short_config()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(
        observation_schema="normalized_v3", decision_interval_s=cfg.time_step_s,
        record_acmi=False,
    ))
    initial, _ = env.reset(seed=1)
    assert initial.shape == (13,)
    assert initial[-3:] == pytest.approx([0.0, 1.0 / 9.0, 0.0])

    following, _, _, _, _ = env.step(1)
    assert following[-3:] == pytest.approx([0.5, 1.0 / 9.0, 0.0])


def test_curriculum_rehearses_old_scenarios_and_ramps_probabilities() -> None:
    schedule = CurriculumSchedule()
    assert schedule.total_episodes == 7500
    assert schedule.stage_at(7500)[1].name == "E_balanced"
    assert schedule.stage_at(7500)[2] == 2000
    assert schedule.probabilities_at(1000) == (1.0, 0.0, 0.0, 0.0)
    start = schedule.probabilities_at(1001)
    assert start[0] > .99 and 0.0 < start[1] < .01
    assert schedule.probabilities_at(1500) == (.70, .30, 0.0, 0.0)
    assert all(stage.probabilities[0] > 0 for stage in schedule.stages)
    assert balanced_score({1: .2, 2: .1, 3: 0., 4: .1}, (.1, .2, .3, .4)) == pytest.approx(.08)
    assert within_forgetting_limit({1: .16}, {1: .20})
    assert not within_forgetting_limit({1: .14}, {1: .20})


@pytest.mark.parametrize(("value", "expected"), [("4", (4,)), ("1,2,3,4", (1, 2, 3, 4)),
                                                   ("1,3,3", (1, 3))])
def test_parse_missile_scenarios(value: str, expected: tuple[int, ...]) -> None:
    assert parse_missile_scenarios(value) == expected


def test_evaluation_progress_logging_defaults_and_jsonl(tmp_path, capsys) -> None:
    args = build_evaluation_parser().parse_args(["checkpoint.pt"])
    assert args.log_interval == 10
    assert args.jsonl_path is None
    assert not any((args.mechanism_threat, args.mechanism_timing,
                    args.mechanism_direction, args.mechanism_overload))

    path = tmp_path / "evaluation.jsonl"
    emit_evaluation_event({"event": "evaluation_progress", "completed_episodes": 10}, path)

    expected = '{"event": "evaluation_progress", "completed_episodes": 10}'
    assert capsys.readouterr().out.strip() == expected
    assert path.read_text(encoding="utf-8") == expected + "\n"


def test_evaluation_mechanisms_are_independent_and_deterministic() -> None:
    snapshot = {"blue_position_m": [0.0, 9000.0, 0.0],
                "blue_velocity_mps": [300.0, 0.0, 0.0],
                "red_positions_m": [[10000.0, 9000.0, 0.0]],
                "red_velocities_mps": [[-500.0, 0.0, 0.0]], "red_alive": [True],
                "min_altitude_m": 8000.0}
    q_values = np.zeros(29)
    assert EvaluationActionShaper(EvaluationShapingConfig()).select(q_values, snapshot)[0] == 0
    shaper = EvaluationActionShaper(EvaluationShapingConfig(direction=True))
    first, diagnostic = shaper.select(q_values, snapshot)
    shaper.reset(); second, _ = shaper.select(q_values, snapshot)
    assert first == second and first != 0
    assert diagnostic["active_scores"] == ["direction"]


def test_evaluation_safety_mask_prevents_low_altitude_descent() -> None:
    snapshot = {"blue_position_m": [0.0, 8100.0, 0.0],
                "blue_velocity_mps": [300.0, 0.0, 0.0],
                "red_positions_m": [[10000.0, 8100.0, 0.0]],
                "red_velocities_mps": [[-500.0, 0.0, 0.0]], "red_alive": [True],
                "min_altitude_m": 8000.0}
    q_values = np.zeros(29); q_values[16] = 100.0
    action, _ = EvaluationActionShaper(EvaluationShapingConfig(threat=True)).select(q_values, snapshot)
    assert action != 16


def test_evaluation_shield_respects_low_mobility_platform() -> None:
    snapshot = {"blue_position_m": [0.0, 9000.0, 0.0],
                "blue_velocity_mps": [300.0, 0.0, 0.0],
                "red_positions_m": [[10000.0, 9000.0, 0.0]],
                "red_velocities_mps": [[-500.0, 0.0, 0.0]], "red_alive": [True],
                "min_altitude_m": 8000.0, "max_altitude_m": 12000.0,
                "min_speed_mps": 80.0, "max_speed_mps": 200.0,
                "max_load_factor_g": 3.0, "time_s": 1.0}
    q_values = np.zeros(29); q_values[13] = 100.0
    action, diagnostic = EvaluationActionShaper(
        EvaluationShapingConfig(overload=True)
    ).select(q_values, snapshot)
    assert action != 13
    assert diagnostic["safe_action_count"] < 29
    assert "maximum_load" in diagnostic["hard_mask_reasons"]


def test_evaluation_nan_uses_deterministic_fallback() -> None:
    snapshot = {"blue_position_m": [0.0, 9000.0, 0.0],
                "blue_velocity_mps": [300.0, 0.0, 0.0],
                "red_positions_m": [[10000.0, 9000.0, 0.0]],
                "red_velocities_mps": [[-500.0, 0.0, 0.0]], "red_alive": [True],
                "min_altitude_m": 8000.0, "max_altitude_m": 12000.0,
                "min_speed_mps": 100.0, "max_speed_mps": 600.0,
                "max_load_factor_g": 9.0, "time_s": 1.0}
    action, diagnostic = EvaluationActionShaper(
        EvaluationShapingConfig(threat=True)
    ).select(np.full(29, np.nan), snapshot)
    assert 0 <= action < 29
    assert diagnostic["fallback_reason"] == "network_nan"
    assert all(value is None or np.isfinite(value) for value in diagnostic["q_fuse"])


def test_evaluation_final_statistics_include_distributions() -> None:
    rows = [
        {"blue_survived": True, "termination_reason": "red_failure", "red_loss_reasons": ["miss"],
         "hit_count": 0, "action_histogram": {"2": 3}, "reward": 8.0,
         "miss_distance_m": 100.0, "simulation_time_s": 20.0, "decision_steps": 3},
        {"blue_survived": False, "termination_reason": "success", "red_loss_reasons": [],
         "hit_count": 1, "action_histogram": {"2": 1, "4": 2}, "reward": -10.0,
         "miss_distance_m": 0.0, "simulation_time_s": 10.0, "decision_steps": 3},
    ]

    statistics = _aggregate_results(rows)

    assert statistics["episodes"] == 2
    assert statistics["survival_rate"] == pytest.approx(0.5)
    assert statistics["termination_counts"] == {"red_failure": 1, "success": 1}
    assert statistics["action_distribution"] == {"2": 4, "4": 2}
    assert statistics["reward"]["median"] == pytest.approx(-1.0)
    assert sum(statistics["reward"]["histogram"]["counts"]) == 2
    assert statistics["miss_distance_probability_histogram_1m"] == {
        "bin_width_m": 1.0, "sample_count": 2,
        "bins": [
            {"lower_m": 0, "upper_m": 1, "count": 1, "probability": 0.5},
            {"lower_m": 100, "upper_m": 101, "count": 1, "probability": 0.5},
        ],
    }
    assert _numeric_distribution([])["histogram"] == {"bin_edges": [], "counts": []}
    assert _one_meter_probability_histogram([])["bins"] == []


def test_blue_reset_records_initial_geometry_and_orientation() -> None:
    env = BlueEscapeEnv(EnvironmentConfig(), BlueEscapeEnvConfig(missile_count=2, record_acmi=False))

    _, info = env.reset(seed=19)

    initialization = info["initialization"]
    assert len(initialization["blue_aircraft"]) == 1
    assert len(initialization["red_missiles"]) == 2
    assert initialization["blue_orientation"] in {
        "toward_missile_swarm", "positive_90_deg", "negative_90_deg",
        "away_from_missile_swarm",
    }
    for entity in initialization["blue_aircraft"] + initialization["red_missiles"]:
        assert len(entity["position_m"]) == 3
        assert entity["altitude_m"] == pytest.approx(entity["position_m"][1])
        assert -180.0 <= entity["heading_deg"] <= 180.0
        assert -90.0 <= entity["flight_path_angle_deg"] <= 90.0


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


def test_blue_threat_potential_is_bounded_and_normalized_across_missile_counts() -> None:
    """Duplicating an identical threat must not multiply the shaping reward."""
    totals: list[float] = []
    for missile_count in (1, 2, 3, 4):
        env = BlueEscapeEnv(
            EnvironmentConfig(),
            BlueEscapeEnvConfig(missile_count=missile_count, record_acmi=False),
        )
        env.reset(seed=100 + missile_count)
        assert env.inner.state is not None
        blue = env.inner.state.blue[0]
        blue.velocity_mps = np.array([-350.0, -50.0, 0.0])
        for missile in env.inner.state.red:
            missile.position_m = blue.position_m + np.array([20000.0, 0.0, 0.0])
        potential = env._threat_potential()
        assert 0.0 <= potential["total"] <= env.config.shaping_scale
        totals.append(potential["total"])

    assert totals == pytest.approx([totals[0]] * 4)


def test_maneuver_regularizers_penalize_switch_reversal_and_state_limits() -> None:
    env = BlueEscapeEnv(EnvironmentConfig(), BlueEscapeEnvConfig(record_acmi=False))
    env.reset(seed=4)
    assert env.inner.state is not None

    first, _ = env._maneuver_penalties(0)
    assert first["action_switch_penalty"] == 0.0
    assert first["opposite_maneuver_penalty"] == 0.0

    env._previous_action = 1
    reversed_penalties, diagnostics = env._maneuver_penalties(3)
    assert reversed_penalties["action_switch_penalty"] > 0.0
    assert reversed_penalties["opposite_maneuver_penalty"] > 0.0
    assert diagnostics["maneuver_cosine"] < env.config.opposite_maneuver_cosine

    blue = env.inner.state.blue[0]
    velocity_before = np.array([400.0, 0.0, 0.0])
    blue.velocity_mps[[0, 2]] = [400.0, 400.0]
    blue.velocity_mps[1] = 300.0
    constrained, state_diagnostics = env._maneuver_penalties(7, velocity_before)
    assert constrained["climb_rate_penalty"] > 0.0
    assert constrained["descent_rate_penalty"] == 0.0
    assert constrained["overload_penalty"] > 0.0
    assert constrained["lateral_speed_penalty"] > 0.0
    assert state_diagnostics["lateral_velocity_change_mps"] == pytest.approx(400.0)

    blue.velocity_mps[1] = -300.0
    descending, _ = env._maneuver_penalties(0, velocity_before)
    assert descending["climb_rate_penalty"] == 0.0
    assert descending["descent_rate_penalty"] > 0.0


def test_maneuver_regularizer_configuration_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="reward scales, ranges, or tactical weights"):
        BlueEscapeEnv(EnvironmentConfig(), BlueEscapeEnvConfig(overload_soft_limit_g=10.0))
    with pytest.raises(ValueError, match="reward scales, ranges, or tactical weights"):
        BlueEscapeEnv(EnvironmentConfig(), BlueEscapeEnvConfig(opposite_maneuver_cosine=0.0))


def test_discounted_potential_shaping_telescopes() -> None:
    gamma = 0.999
    potentials = np.asarray([0.4, 1.7, 0.8, 0.0])
    shaping = gamma * potentials[1:] - potentials[:-1]
    discounted_sum = sum(gamma ** index * reward for index, reward in enumerate(shaping))
    assert discounted_sum == pytest.approx(-potentials[0] + gamma ** 3 * potentials[-1])


def test_terminal_reward_distinguishes_miss_timeout_and_red_success() -> None:
    env = BlueEscapeEnv(EnvironmentConfig(), BlueEscapeEnvConfig(record_acmi=False))
    assert env._terminal_reward({"termination_reason": "red_failure", "time_s": 20.0}) > 10.0
    assert env._terminal_reward({"termination_reason": "timeout", "time_s": 180.0}) == 2.0
    killed = env._terminal_reward({"termination_reason": "success", "time_s": 20.0})
    assert -10.0 < killed < -9.0


def test_rainbow_select_actions_batches_observations() -> None:
    agent = RainbowDQNAgent(RainbowDQNConfig(9, 29, hidden_dim=16))
    observations = np.zeros((4, 9), dtype=np.float32)
    parameters_before = {name: value.detach().clone() for name, value in agent.online.state_dict().items()}

    actions = agent.select_actions(observations, evaluation=True)

    assert actions.shape == (4,)
    assert actions.dtype == np.int64
    assert np.all((0 <= actions) & (actions < 29))
    assert agent.total_steps == agent.optimizer_updates == agent.target_updates == 0
    assert agent.replay.size == 0
    assert all(np.array_equal(parameters_before[name].cpu().numpy(), value.cpu().numpy())
               for name, value in agent.online.state_dict().items())


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
