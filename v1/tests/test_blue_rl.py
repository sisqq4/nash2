from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from red_swarm_policy.blue_rl import (
    BLUE_ACTION_CONTEXT_DIM,
    BlueEscapeEnv,
    BlueEscapeEnvConfig,
    BlueProcessEnvironmentPool,
    BlueRLController,
    EvaluationActionShaper,
    EvaluationShapingConfig,
    FlightEnvelopeConfig,
    FlightEnvelopeConstraintLayer,
    BlueMechanismStateEstimator,
    MechanismRewardConfig,
    RainbowDQNAgent,
    RainbowDQNConfig,
    blue_observation_dim,
)
from red_swarm_policy.blue_rl.config_io import configure_blue_mission_duration
from red_swarm_policy.env import EnvironmentConfig, ScenarioConfig
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


class CountingPolicy(FixedPolicy):
    def __init__(self, action: int = 7) -> None:
        self.action = action
        self.calls = 0

    def select_action(self, observation: np.ndarray, *, evaluation: bool = False) -> int:
        assert observation.ndim == 1
        self.calls += 1
        return self.action


def short_config() -> EnvironmentConfig:
    base = EnvironmentConfig()
    return replace(base, max_steps=base.policy_entry_steps + 1)


def test_blue_cli_duration_sets_mission_and_guidance_to_200_seconds() -> None:
    config = configure_blue_mission_duration(EnvironmentConfig(
        scenario=ScenarioConfig(blue_altitude_range_m=(8000.0, 12000.0)),
    ))
    assert config.max_steps * config.time_step_s == pytest.approx(200.0)
    assert config.missile.max_guidance_time_s == pytest.approx(200.0)
    assert config.scenario.blue_altitude_range_m == (9000.0, 11000.0)


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


def test_normalized_v3_exposes_previous_executed_action_and_actuator_command() -> None:
    cfg = EnvironmentConfig()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(
        missile_count=1, observation_schema="normalized_v3",
        decision_interval_s=cfg.time_step_s, record_acmi=False,
    ))
    observation, reset_info = env.reset(seed=1)
    context_offset = 6 + 4

    assert observation.shape == (10 + BLUE_ACTION_CONTEXT_DIM,)
    assert observation[context_offset] == 1.0
    assert np.count_nonzero(observation[context_offset:context_offset + 29]) == 1
    np.testing.assert_allclose(observation[-3:], [0.0, 1.0 / 9.0, 0.0])
    assert reset_info["flight_quality_state"]["previous_executed_action_index"] == 0

    env._learning_active = True
    env._previous_potential = env._threat_potential()
    next_observation, _, _, _, info = env.step(7, policy_action=2)

    assert next_observation[context_offset + 7] == 1.0
    assert np.count_nonzero(next_observation[context_offset:context_offset + 29]) == 1
    np.testing.assert_allclose(
        next_observation[-3:], np.asarray(info["actual_load_command_body_g"]) / [9.0, 9.0, np.pi]
    )
    assert info["flight_quality_state"]["previous_executed_action_index"] == 7
    np.testing.assert_allclose(
        info["flight_quality_state"]["actual_load_command_body_g"],
        info["actual_load_command_body_g"],
    )


def test_normalized_v4_exposes_physical_mechanism_state_and_padding() -> None:
    cfg = EnvironmentConfig()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(
        missile_count=1, max_missiles=4, pad_observation_to_max_missiles=True,
        observation_schema="normalized_v4", record_acmi=False,
    ))

    observation, info = env.reset(seed=5)

    assert observation.shape == (blue_observation_dim("normalized_v4", 4),) == (119,)
    assert np.all(np.isfinite(observation))
    mechanism = info["reward_mechanism_state"]
    assert mechanism["phase"] in {"P0", "P1", "P2"}
    assert 0.0 <= mechanism["total_threat"] <= 2.0
    assert len(mechanism["desired_direction_body"]) == 3
    # Three unavailable rich missile slots contain exactly 13 zeroes each.
    per_missile_offset = 6
    assert np.allclose(observation[per_missile_offset + 13:per_missile_offset + 52], 0.0)
    assert observation.shape[0] > blue_observation_dim("normalized_v3", 4)


def _mechanism_snapshot(*, los_velocity_z_mps: float = 0.0,
                        missile_position_x_m: float = 10_000.0,
                        missile_velocity_x_mps: float = -900.0,
                        time_s: float = 0.0) -> dict[str, object]:
    return {
        "blue_position_m": [0.0, 10_000.0, 0.0],
        "blue_velocity_mps": [300.0, 0.0, 0.0],
        "time_s": time_s,
        "red_positions_m": [[missile_position_x_m, 10_000.0, 0.0]],
        "red_velocities_mps": [[missile_velocity_x_mps, 0.0, los_velocity_z_mps]],
        "red_alive": [True],
        "red_energy": [1.0],
        "red_guidance_modes": ["locked"],
        "min_altitude_m": 8_000.0,
        "max_altitude_m": 12_000.0,
        "min_speed_mps": 100.0,
        "max_speed_mps": 600.0,
        "max_load_factor_g": 9.0,
        "previous_executed_action_index": 0,
        "actual_load_command_body_g": [0.0, 1.0, 0.0],
    }


def test_mechanism_threat_treats_low_los_rate_as_collision_risk() -> None:
    direct = BlueMechanismStateEstimator().observe(
        _mechanism_snapshot(los_velocity_z_mps=0.0, missile_position_x_m=5_000.0)
    )
    crossing = BlueMechanismStateEstimator().observe(
        _mechanism_snapshot(los_velocity_z_mps=1000.0, missile_position_x_m=5_000.0)
    )

    assert direct["total_threat"] > crossing["total_threat"]
    assert direct["minimum_tgo_s"] < 10.0
    assert direct["phase"] == "P1"  # emergency t_go bypasses confirmation delay


def test_mechanism_penalties_separate_timing_direction_and_load() -> None:
    estimator = BlueMechanismStateEstimator(MechanismRewardConfig())
    mechanism = {
        "evasion_target": 1.0,
        "desired_direction_inertial": [1.0, 0.0, 0.0],
        "reference_load_g": 5.0,
    }
    mask = np.ones(29, dtype=bool)

    absent = estimator.penalties(mechanism, np.zeros(3), 1.0, 0.1, 0.0,
                                 action_mask=mask)
    aligned = estimator.penalties(mechanism, np.array([9.80665, 0.0, 0.0]), 5.0,
                                  0.1, 0.0, action_mask=mask)
    wrong = estimator.penalties(mechanism, np.array([-9.80665, 0.0, 0.0]), 9.0,
                                0.1, 0.0, action_mask=mask)

    assert absent["timing"] > 0.0
    assert absent["direction"] == absent["overload"] == 0.0
    assert aligned["timing"] == aligned["direction"] == aligned["overload"] == 0.0
    assert wrong["direction"] > 0.0
    assert wrong["overload"] > 0.0
    assert max(wrong["timing"], wrong["direction"], wrong["overload"]) < 0.001


def test_mechanism_penalties_turn_off_when_no_real_action_choice_exists() -> None:
    estimator = BlueMechanismStateEstimator()
    mechanism = {
        "evasion_target": 1.0,
        "desired_direction_inertial": [1.0, 0.0, 0.0],
        "reference_load_g": 9.0,
    }
    mask = np.zeros(29, dtype=bool); mask[4] = True

    penalties = estimator.penalties(
        mechanism, np.zeros(3), 1.0, 0.1, 0.0, action_mask=mask
    )

    assert penalties["choice_gate"] == 0.0
    assert penalties["total"] == 0.0


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


def test_training_parser_accepts_replay_capacity_override() -> None:
    from red_swarm_policy.train_blue_rl import build_parser

    args = build_parser().parse_args(["--replay-size", "500000"])
    assert args.replay_size == 500_000


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


def _envelope_snapshot(*, altitude: float = 10_000.0,
                       velocity: tuple[float, float, float] = (300.0, 0.0, 0.0)) -> dict[str, object]:
    return {
        "blue_position_m": [0.0, altitude, 0.0],
        "blue_velocity_mps": list(velocity),
        "min_altitude_m": 8_000.0, "max_altitude_m": 12_000.0,
        "min_speed_mps": 100.0, "max_speed_mps": 600.0,
        "max_load_factor_g": 9.0,
    }


def test_shared_envelope_uses_8500_to_11500_soft_altitude_band() -> None:
    layer = FlightEnvelopeConstraintLayer()
    mask, penalty, details = layer.constraints(
        _envelope_snapshot(altitude=11_450.0, velocity=(300.0, 40.0, 0.0))
    )

    assert mask.any()
    assert details["extrapolated_altitude_m"][13] > 11_500.0
    assert details["altitude_cost"][13] > 0.0
    assert penalty[13] >= details["altitude_cost"][13]
    assert FlightEnvelopeConfig().altitude_safety_margin_m == 500.0
    assert FlightEnvelopeConfig().altitude_extrapolation_s == 2.0


@pytest.mark.parametrize(("altitude_m", "expected_penalty"), [
    (8_500.0, 0.0),
    (8_250.0, 1.0),
    (8_000.0, 4.0),
    (11_500.0, 0.0),
    (11_750.0, 1.0),
    (12_000.0, 4.0),
])
def test_shared_envelope_doubles_altitude_penalty_at_reference_points(
    altitude_m: float, expected_penalty: float,
) -> None:
    layer = FlightEnvelopeConstraintLayer()

    _, _, details = layer.constraints(_envelope_snapshot(altitude=altitude_m))

    assert details["extrapolated_altitude_m"][0] == pytest.approx(altitude_m)
    assert details["altitude_cost"][0] == pytest.approx(expected_penalty)
    assert layer.config.altitude_penalty_weight == pytest.approx(4.0)


def test_shared_envelope_masks_load_and_roll_rate_reversal() -> None:
    layer = FlightEnvelopeConstraintLayer()
    layer.previous_action = 7
    mask, _, details = layer.constraints(_envelope_snapshot())

    assert not mask[10]
    assert details["load_command_rate_gps"][10] > layer.config.hard_load_command_rate_gps
    assert details["roll_command_rate_deg_s"][10] > layer.config.hard_roll_command_rate_deg_s
    assert {"load_command_rate", "roll_command_rate"} <= set(details["hard_mask_reasons"])


def test_shared_envelope_uses_visible_action_context_instead_of_private_history() -> None:
    snapshot = _envelope_snapshot()
    snapshot.update({
        "previous_executed_action_index": 7,
        "actual_load_command_body_g": [0.0, 9.0, np.arccos(1.0 / 9.0)],
    })
    first = FlightEnvelopeConstraintLayer(); first.previous_action = 0
    second = FlightEnvelopeConstraintLayer(); second.previous_action = 10

    first_mask, first_penalty, first_details = first.constraints(snapshot)
    second_mask, second_penalty, second_details = second.constraints(snapshot)

    np.testing.assert_array_equal(first_mask, second_mask)
    np.testing.assert_allclose(first_penalty, second_penalty)
    np.testing.assert_allclose(
        first_details["starting_load_command_body_g"],
        snapshot["actual_load_command_body_g"],
    )
    assert first_details["previous_executed_action_index"] == 7
    np.testing.assert_allclose(
        first_details["load_command_rate_gps"], second_details["load_command_rate_gps"]
    )


def test_emergency_gate_relaxes_only_soft_envelope_costs() -> None:
    layer = FlightEnvelopeConstraintLayer()
    normal = _envelope_snapshot(altitude=11_400.0, velocity=(300.0, 35.0, 0.0))
    emergency = dict(normal); emergency["mechanism_emergency_gate"] = 1.0

    normal_mask, normal_cost, normal_details = layer.constraints(normal)
    emergency_mask, emergency_cost, emergency_details = layer.constraints(emergency)

    np.testing.assert_array_equal(normal_mask, emergency_mask)
    np.testing.assert_allclose(normal_details["altitude_cost"], emergency_details["altitude_cost"])
    assert np.all(emergency_cost <= normal_cost + 1.0e-12)
    assert emergency_details["envelope_cost_gate"] == pytest.approx(0.5)
    assert emergency_details["command_cost_gate"] == pytest.approx(0.2)


def test_heading_recovery_rejects_frozen_acceleration_false_positive() -> None:
    layer = FlightEnvelopeConstraintLayer()
    velocity = np.array([-12.751093610092154, 328.06811387774457, 78.32180460216358])
    snapshot = _envelope_snapshot(velocity=tuple(velocity))
    snapshot.update({
        "previous_executed_action_index": 0,
        "actual_load_command_body_g": [0.0, 1.0, 0.0],
    })

    _, _, details = layer.constraints(snapshot)

    # The old v+a*t approximation classified actions 4 and 15 as recoverable.
    # Reintegrating the rotating velocity frame over the full two-second
    # window shows that neither has a physically feasible recovery path.
    assert not details["heading_recoverable"][4]
    assert not details["heading_recoverable"][15]
    assert details["heading_recovery_action"][15] == -1
    assert np.isinf(details["heading_recovery_time_s"][15])
    # The dynamic checker still finds genuine recoveries and reports the first
    # feasible recovery action/time rather than making every steep state fail.
    assert details["heading_recoverable"][16]
    assert 0.0 < details["heading_recovery_time_s"][16] <= 2.0
    assert 0 <= details["heading_recovery_action"][16] < 29


def test_replay_attributes_transition_to_post_constraint_action() -> None:
    layer = FlightEnvelopeConstraintLayer()
    q_values = np.zeros(29); q_values[2] = 100.0  # sqrt(9^2 + 1^2) exceeds the 9-g hard limit.
    executed, diagnostic = layer.select(q_values, _envelope_snapshot())
    assert diagnostic["raw_action"] == 2
    assert executed != 2

    agent = RainbowDQNAgent(RainbowDQNConfig(2, 29, n_step=1, learning_starts=100))
    observation = np.zeros(2, dtype=np.float32)
    next_mask, next_penalty, _ = layer.constraints(_envelope_snapshot())
    agent.observe_for_env(0, observation, executed, 1.0, observation, False,
                          next_action_mask=next_mask, next_action_penalty=next_penalty)

    assert agent.replay.actions[0] == executed
    assert agent.replay.actions[0] != diagnostic["raw_action"]
    np.testing.assert_array_equal(agent.replay.next_action_masks[0], next_mask)
    np.testing.assert_allclose(agent.replay.next_action_penalties[0], next_penalty)


def test_checkpoint_round_trips_shared_envelope_configuration(tmp_path) -> None:
    envelope = FlightEnvelopeConfig(altitude_safety_margin_m=600.0)
    agent = RainbowDQNAgent(RainbowDQNConfig(
        2, 29, hidden_dim=16,
        flight_envelope_config=envelope.__dict__.copy(),
    ))
    path = tmp_path / "blue.pt"
    agent.save(str(path))

    restored = RainbowDQNAgent.load(str(path))

    assert restored.config.flight_envelope_config == envelope.__dict__
    FlightEnvelopeConfig(**restored.config.flight_envelope_config).validate()


def test_blue_env_records_raw_constrained_and_continuously_applied_commands() -> None:
    cfg = EnvironmentConfig()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(missile_count=1, record_acmi=False))
    env.reset(seed=2)
    env._learning_active = True
    env._previous_potential = env._threat_potential()
    applied: list[np.ndarray] = []
    original_step = env.inner.step

    def recording_step(*args: object, **kwargs: object):
        applied.append(np.asarray(kwargs["blue_action"]["load_command_body_g"][0]))
        return original_step(*args, **kwargs)

    env.inner.step = recording_step  # type: ignore[method-assign]
    _, _, _, _, info = env.step(7, policy_action=2)

    assert info["requested_action_index"] == 2
    assert info["constrained_action_index"] == info["executed_action_index"] == 7
    assert len(applied) == env.frames_per_action
    assert 0.0 < applied[0][2] < applied[-1][2]
    np.testing.assert_allclose(applied[-1], info["target_load_command_body_g"])


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
    config = EnvironmentConfig(scenario=ScenarioConfig(
        blue_altitude_range_m=(8000.0, 12000.0),
        velocity_perturb_mps=75.0,
    ))
    env = BlueEscapeEnv(config, BlueEscapeEnvConfig(missile_count=2, record_acmi=False))

    _, info = env.reset(seed=19)

    initialization = info["initialization"]
    assert len(initialization["blue_aircraft"]) == 1
    assert len(initialization["red_missiles"]) == 2
    assert initialization["blue_orientation"] in {
        "toward_missile_swarm", "positive_90_deg", "negative_90_deg",
        "away_from_missile_swarm",
    }
    blue = initialization["blue_aircraft"][0]
    assert 9000.0 <= blue["altitude_m"] <= 11000.0
    assert blue["flight_path_angle_deg"] == pytest.approx(0.0, abs=1.0e-12)
    assert blue["bank_angle_deg"] == pytest.approx(0.0, abs=1.0e-12)
    assert env.inner.state is not None
    assert env.inner.state.blue[0].velocity_mps[1] == pytest.approx(0.0, abs=1.0e-12)
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


def test_blue_rl_waits_for_strict_60km_detection_before_maneuver_and_reward() -> None:
    cfg = EnvironmentConfig()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(
        missile_count=1,
        decision_interval_s=cfg.time_step_s,
        record_acmi=False,
    ))
    _, reset_info = env.reset(seed=2)
    assert env.inner.state is not None
    blue, missile = env.inner.state.blue[0], env.inner.state.red[0]
    initial_velocity = blue.velocity_mps.copy()

    _, reward, terminated, truncated, info = env.step(7)
    assert not (terminated or truncated)
    assert reset_info["learning_active"] is False
    assert info["learning_active"] is False
    assert info["learning_transition"] is False
    assert info["requested_action_index"] == 7
    assert info["executed_action_index"] == 0
    assert reward == 0.0
    blue, missile = env.inner.state.blue[0], env.inner.state.red[0]
    np.testing.assert_allclose(blue.velocity_mps, initial_velocity, rtol=0.0, atol=1.0e-12)
    assert blue.bank_angle_rad == pytest.approx(0.0)

    missile.position_m = blue.position_m + np.array([59000.0, 0.0, 0.0])
    missile.velocity_mps = blue.velocity_mps.copy()
    _, reward, _, _, activation_info = env.step(7)
    assert activation_info["learning_active"] is True
    assert activation_info["learning_transition"] is False
    assert activation_info["learning_activation_range_m"] < 60000.0
    assert activation_info["executed_action_index"] == 0
    assert reward == 0.0

    _, _, _, _, learning_info = env.step(7)
    assert learning_info["learning_active"] is True
    assert learning_info["learning_transition"] is True
    assert learning_info["executed_action_index"] == 7
    assert env.inner.state.blue[0].bank_angle_rad == pytest.approx(
        np.arccos(1.0 / 9.0)
    )


def test_drop_in_controller_does_not_call_policy_before_detection() -> None:
    cfg = EnvironmentConfig()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(missile_count=1, record_acmi=False))
    env.reset(seed=2)
    assert env.inner.state is not None
    policy = CountingPolicy(action=7)
    controller = BlueRLController(policy, cfg, BlueEscapeEnvConfig(missile_count=1))

    assert controller(env.inner.state)["action_indices"].tolist() == [0]
    assert policy.calls == 0

    blue, missile = env.inner.state.blue[0], env.inner.state.red[0]
    missile.position_m = blue.position_m + np.array([60000.0, 0.0, 0.0])
    assert controller(env.inner.state)["action_indices"].tolist() == [0]
    assert policy.calls == 0

    missile.position_m = blue.position_m + np.array([59999.0, 0.0, 0.0])
    assert controller(env.inner.state)["action_indices"].tolist() == [7]
    assert policy.calls == 1


def test_acmi_writes_explicit_upright_initial_aircraft_attitude(tmp_path) -> None:
    cfg = short_config()
    env = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(
        missile_count=1,
        decision_interval_s=cfg.time_step_s,
        acmi_directory=str(tmp_path),
    ))
    env.reset(seed=3)
    _, _, terminated, truncated, _ = env.step(0)
    assert terminated or truncated

    lines = (tmp_path / "episode_000001.acmi").read_text(encoding="utf-8").splitlines()
    blue_line = next(line for line in lines if line.startswith("100,T="))
    transform = blue_line.split("T=", 1)[1].split(",", 1)[0].split("|")
    assert len(transform) == 6
    assert float(transform[3]) == pytest.approx(0.0)
    assert float(transform[4]) == pytest.approx(0.0)


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


def test_joint_reward_components_reconstruct_actual_transition_reward() -> None:
    env = BlueEscapeEnv(EnvironmentConfig(), BlueEscapeEnvConfig(
        missile_count=1, observation_schema="normalized_v4", record_acmi=False,
    ))
    env.reset(seed=12)
    assert env.inner.state is not None
    blue, missile = env.inner.state.blue[0], env.inner.state.red[0]
    missile.position_m = blue.position_m + np.array([20_000.0, 0.0, 0.0])
    missile.velocity_mps = blue.velocity_mps + np.array([-1_000.0, 0.0, 0.0])
    env._learning_active = True
    env._activation_time_s = env.inner.state.time_s
    env.mechanism_estimator.reset()
    env._mechanism_reward_state = env.mechanism_estimator.observe(
        env._mechanism_snapshot(), env._structural_action_mask()
    )
    env._previous_potential = env._joint_potential(env._mechanism_reward_state)

    _, reward, terminated, truncated, info = env.step(
        7, action_mask=np.ones(29, dtype=bool)
    )

    assert not (terminated or truncated)
    components = info["reward_components"]
    reconstructed = sum(components[name] for name in (
        "far_away_shaping", "near_tangent_shaping", "near_dive_shaping",
        "threat_outcome_shaping", "timing_penalty", "direction_penalty",
        "overload_penalty", "terminal",
    ))
    assert reward == pytest.approx(reconstructed)
    assert components["joint_potential_shaping"] == pytest.approx(
        components["tactical_shaping"] + components["threat_outcome_shaping"]
    )
    assert info["reward_diagnostics"]["choice_gate"] == 1.0


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
