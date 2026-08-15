from __future__ import annotations

import copy
import json
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from red_swarm_policy import (
    AssignmentActorInputs,
    AssignmentCriticInputs,
    EnvironmentConfig,
    EngagementState,
    MAPPOTrainer,
    MissileConfig,
    ObservationLayer,
    OverloadBiasActor,
    OverloadBiasActorInputs,
    OverloadBiasCritic,
    OverloadBiasCriticInputs,
    PPOConfig,
    RedAction,
    RedBlueEngagementEnv,
    RewardConfig,
    ScenarioConfig,
    SensorConfig,
    SwarmModelConfig,
    TargetAssignmentActor,
    TargetAssignmentCritic,
    ThreeDoFState,
    collect_parallel_rollout,
    collect_rollout,
    generalized_advantage_estimation,
)
from red_swarm_policy.env import ThreeDoFPhysicsLayer
from red_swarm_policy.env import (
    assignment_feasibility_potential,
    ineffective_loss_rate,
    low_intercept_potential,
    mission_completion,
)
from red_swarm_policy.env import physics as physics_module
from red_swarm_policy.env.math_utils import G0, clip_norm, speed_of_sound, standard_atmosphere
from red_swarm_policy.policy.actor import (
    _radial_tanh_forward,
    _radial_tanh_inverse,
    _radial_tanh_log_prob,
)
from red_swarm_policy.run_cluster_scenario import (
    _build_environment_config as build_cluster_environment_config,
    _load_policy_actors,
    build_parser as build_cluster_parser,
    run_cluster_scenario,
)
from red_swarm_policy.train_env import (
    CHECKPOINT_SCHEMA_VERSION,
    _configs_from_checkpoint,
    _load_torch_checkpoint,
    _restore_checkpoint,
    _save_checkpoint,
)
from red_swarm_policy.validate_checkpoint import build_parser as build_validation_parser
from red_swarm_policy.validate_checkpoint import validate


@pytest.fixture(scope="session")
def network_device() -> torch.device:
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _model_config() -> SwarmModelConfig:
    return SwarmModelConfig(d_model=16, num_heads=4)


def _environment_config(
    *,
    red_count: int = 2,
    blue_count: int = 1,
    max_steps: int = 3,
) -> EnvironmentConfig:
    return EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.05,
        assignment_update_interval_s=0.1,
        max_steps=max_steps,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(
            red_count=red_count,
            blue_count=blue_count,
            position_perturb_m=0.0,
            velocity_perturb_mps=0.0,
        ),
        sensor=SensorConfig(detection_range_m=200000.0),
    )


def test_environment_config_exposes_exact_three_level_time_scales() -> None:
    config = EnvironmentConfig()

    assert config.time_step_s == 0.005
    assert config.bias_update_interval_s == 0.1
    assert config.assignment_update_interval_s == 5.0
    assert config.bias_update_steps == 20
    assert config.assignment_update_steps == 1000
    assert config.bias_updates_per_assignment == 50
    assert config.policy_entry_steps == 1400
    assert config.policy_entry_time_s == 7.0
    assert config.max_steps == 36000
    assert config.missile.max_guidance_time_s == 180.0
    assert config.policy_horizon_steps == 34600
    assert config.policy_horizon_s == 173.0
    assert config.remaining_guidance_horizon_s == 173.0
    assert config.policy_entry_speed_tolerance_mps == pytest.approx(0.001770)
    assert config.policy_entry_flight_path_tolerance_deg == 0.5
    assert config.missile.boost_climb_angle_deg == 20.0
    assert config.missile.boost_pitch_transition_s == 2.0
    assert config.missile.drag_coefficient is None
    assert config.missile.induced_drag_factor == 0.08
    assert config.missile.max_guidance_bias_g == 5.0
    ppo = PPOConfig()
    assert ppo.gamma_high == 1.0
    assert ppo.gamma_low == 1.0
    assert ppo.assignment_entropy_coef == pytest.approx(0.001)
    assert config.reward.high_potential_weight == pytest.approx(512.0)
    assert config.reward.high_potential_gamma == 1.0
    assert config.reward.low_potential_gamma == 1.0

    with pytest.raises(ValueError, match="assignment_update_interval_s must be an integer multiple"):
        replace(config, assignment_update_interval_s=0.953).validate()

    with pytest.raises(ValueError, match="assignment_update_interval_s must be an integer multiple of bias_update_interval_s"):
        replace(config, assignment_update_interval_s=0.9, bias_update_interval_s=0.2).validate()
    with pytest.raises(ValueError, match="missile guidance parameters are invalid"):
        replace(config, missile=replace(config.missile, max_guidance_bias_g=35.1)).validate()


def test_standard_atmosphere_and_mach_drag_table_are_physical() -> None:
    sea_level = standard_atmosphere(0.0)
    ten_km = standard_atmosphere(10000.0)

    assert sea_level == pytest.approx((288.15, 101325.0, 1.2250000181, 340.2939880))
    assert ten_km == pytest.approx((223.15, 26436.242593, 0.4127061532, 299.4631649))
    assert speed_of_sound(10000.0) == pytest.approx(299.4631649)

    config = EnvironmentConfig(scenario=ScenarioConfig(red_count=1, blue_count=1))
    physics = ThreeDoFPhysicsLayer(config)
    assert physics._zero_lift_drag_coefficient(0.0) == pytest.approx(0.10)
    assert physics._zero_lift_drag_coefficient(1.05) == pytest.approx(0.34)
    assert physics._zero_lift_drag_coefficient(6.0) == pytest.approx(0.15)
    missile = _state(
        (0.0, 10000.0, 0.0),
        (6.0 * speed_of_sound(10000.0), 0.0, 0.0),
        config.missile.dry_mass_kg,
        age_s=config.missile.boost_duration_s,
    )
    drag = physics._missile_drag_acceleration(missile, np.zeros(3, dtype=np.float64))
    dynamic_pressure = 0.5 * ten_km[2] * np.dot(missile.velocity_mps, missile.velocity_mps)
    expected_drag = dynamic_pressure * 0.15 * config.missile.reference_area_m2 / missile.mass_kg
    assert np.linalg.norm(drag) == pytest.approx(expected_drag)


def test_boost_climb_reaches_20_degrees_at_2_seconds_and_holds_to_entry() -> None:
    config = EnvironmentConfig(scenario=ScenarioConfig(red_count=1, blue_count=1))
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=2, start_mode="launch", red_count=1, blue_count=1)

    for _ in range(int(round(2.0 / config.time_step_s))):
        env.step()
    assert env.state is not None
    assert env._flight_path_angle_deg(env.state.red[0].velocity_mps) == pytest.approx(20.0, abs=0.5)
    two_second_altitude = float(env.state.red[0].position_m[1])

    for _ in range(config.policy_entry_steps - env.state.step_count):
        env.step()
    assert env.state.time_s == 7.0
    assert env._flight_path_angle_deg(env.state.red[0].velocity_mps) == pytest.approx(20.0, abs=0.5)
    assert np.linalg.norm(env.state.red[0].velocity_mps) == pytest.approx(1770.0, abs=1.0)
    assert env.state.red[0].position_m[1] > two_second_altitude


def test_cluster_scenario_defaults_and_baseline_follow_schema10_constraints() -> None:
    args = build_cluster_parser().parse_args([])
    config = build_cluster_environment_config(args)

    assert args.duration_s == 180.0
    assert config.max_steps == 36000
    assert config.missile.max_guidance_time_s == 180.0
    assert config.scenario.max_missiles_per_target == 4

    short_config = _environment_config(red_count=24, blue_count=3, max_steps=3)
    trajectory = run_cluster_scenario(
        short_config,
        seed=20260716,
        duration_s=0.15,
        trajectory_sample_interval_s=0.05,
    )
    targets = trajectory.initial_target_indices
    assert int((targets < 0).sum()) == 12
    assert [int((targets == index).sum()) for index in range(3)] == [4, 4, 4]


def test_post_boost_entry_is_exact_and_warmup_isolated() -> None:
    config = replace(
        EnvironmentConfig(),
        scenario=ScenarioConfig(red_count=2, blue_count=1),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=True)
    observation = env.reset(seed=5, start_mode="launch")
    with pytest.raises(RuntimeError, match="before post-boost"):
        env.record_network_call("assignment_actor")
    warmup_action = {
        "target_indices": np.zeros(2, dtype=np.int64),
        "guidance_bias": np.ones((2, 2), dtype=np.float64),
    }
    for _ in range(1399):
        step = env.step(red_action=warmup_action)
        assert step.reward_high == 0.0
        assert np.count_nonzero(step.reward_low) == 0
    assert env.state is not None
    assert env.state.step_count == 1399
    assert not env.policy_ready
    assert len(env.replay_layer) == 0
    assert observation.assignment_actor_inputs.target_mask[..., 1:].sum().item() == 0

    entry = env.step(red_action=warmup_action)
    assert entry.info["network_entry_reached"] is True
    assert env.policy_ready
    assert env.state.step_count == 1400
    assert env.state.time_s == 7.0
    assert env.policy_remaining_fraction == 1.0
    assert len(env.replay_layer) == 0
    for missile in env.state.red:
        assert abs(np.linalg.norm(missile.velocity_mps) - 1770.0) <= 0.001770
        assert missile.fuel_mass_kg == 0.0
        assert missile.mass_kg == 120.0
        assert missile.current_target_index == -1
        assert np.count_nonzero(missile.pn_load_body_g) == 0
        assert np.count_nonzero(missile.bias_load_body_g) == 0
        assert np.linalg.norm(missile.final_load_body_g[1:]) <= 35.0
        assert RedBlueEngagementEnv._flight_path_angle_deg(missile.velocity_mps) == pytest.approx(
            20.0,
            abs=0.5,
        )

    env.record_network_call("assignment_actor")
    status = env.policy_status()
    assert status["first_network_call"]["assignment_actor"] == {
        "time_s": 7.0,
        "step_count": 1400,
    }


def test_reward_windows_and_public_potentials() -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.1,
        assignment_update_interval_s=0.2,
        max_steps=20,
        scenario=ScenarioConfig(red_count=2, blue_count=1),
        missile=MissileConfig(boost_duration_s=0.2),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=True)
    env.reset(seed=6, style="many_to_one")
    assert env.state is not None
    action = {
        "target_indices": np.zeros(2, dtype=np.int64),
        "guidance_bias": np.array([[0.1, -0.1], [-0.2, 0.2]], dtype=np.float64),
    }
    steps = [env.step(red_action=action) for _ in range(4)]
    assert [step.info["low_reward_settled"] for step in steps] == [False, True, False, True]
    assert [step.info["high_reward_settled"] for step in steps] == [False, False, False, True]
    assert steps[0].reward_high == 0.0
    assert np.count_nonzero(steps[0].reward_low) == 0
    high_components = steps[-1].info["reward_components"]
    expected_high = sum(
        high_components[key]
        for key in ("damage_reward", "waste_penalty", "potential_reward", "terminal_reward")
    )
    assert steps[-1].reward_high == pytest.approx(expected_high)
    low_components = steps[1].info["reward_low_components"]
    expected_low_0 = (
        config.reward.low_potential_weight * low_components["potential_delta"][0]
        - (
            config.reward.low_load_penalty + config.reward.low_smooth_penalty
        ) * low_components["control_effort_increment"][0]
    )
    assert steps[1].reward_low[0] == pytest.approx(expected_low_0, abs=1.0e-7)
    assert len(env.replay_layer) == 4
    assert 0.0 <= env.assignment_potential() <= 1.0
    low_phi, zem, lock, tau = env.execution_potential(0, 0)
    assert 0.0 <= low_phi <= 1.0
    assert 0.0 <= zem <= 1.0
    assert 0.0 <= lock <= 1.0
    assert math.isfinite(tau)
    assert mission_completion(env.state) == 0.0
    assert ineffective_loss_rate(env.state) == 0.0
    assert 0.0 <= assignment_feasibility_potential(config, env.state) <= 1.0
    assert 0.0 <= low_intercept_potential(config, env.state.red[0], env.state.blue[0])[0] <= 1.0

    monotonic = env.state.copy()
    monotonic.red[0].alive = False
    monotonic.red[0].loss_reason = "ground_impact"
    assert ineffective_loss_rate(monotonic, 2) == 0.5
    monotonic.blue[0].alive = False
    assert ineffective_loss_rate(monotonic, 2) == 0.5
    monotonic.red[1].alive = False
    monotonic.red[1].loss_reason = "valid_hit"
    assert ineffective_loss_rate(monotonic, 2) == 0.5


def test_state_only_high_potential_rewards_replacing_no_target_with_reachable_assignment() -> None:
    config = _environment_config(red_count=1, blue_count=1, max_steps=20)
    red = _state(
        (0.0, 9000.0, 0.0),
        (1200.0, 0.0, 0.0),
        config.missile.dry_mass_kg,
        age_s=config.policy_entry_time_s,
    )
    blue = _state((20000.0, 9000.0, 0.0), (300.0, 0.0, 0.0), 1.0)
    unassigned = EngagementState(red=[red], blue=[blue])
    assigned = unassigned.copy()
    assigned.red[0].current_target_index = 0

    phi_unassigned = assignment_feasibility_potential(config, unassigned)
    phi_assigned = assignment_feasibility_potential(config, assigned)
    shaped_reward = (
        phi_assigned
        - phi_unassigned
    )
    assert phi_unassigned == 0.0
    assert phi_assigned > 0.0
    assert shaped_reward > 0.0


def test_terminal_success_is_mutually_exclusive_and_settled_once() -> None:
    config = _environment_config(red_count=1, blue_count=1, max_steps=10)
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=8, style="one_to_one")
    assert env.state is not None
    red = env.state.red[0]
    blue = env.state.blue[0]
    red.position_m = np.array([0.0, 9000.0, 0.0])
    red.velocity_mps = np.array([config.missile.max_speed_mps, 0.0, 0.0])
    blue.position_m = np.array([1.0, 9000.0, 0.0])
    blue.velocity_mps = np.zeros(3, dtype=np.float64)
    env.last_observation = env.observation_layer.observe(env.state)
    step = env.step(
        red_action={
            "target_indices": np.array([0], dtype=np.int64),
            "guidance_bias": np.zeros((1, 2), dtype=np.float64),
        }
    )
    assert step.done
    assert step.info["reward_components"]["terminal_reason"] == "success"
    assert step.info["reward_components"]["terminal_reward"] < 0.0
    assert step.info["timeout"] is False
    assert step.info["reward_low_components"]["miss_event"] == [0.0]
    assert step.info["reward_low_components"]["hit_event"] == [1.0]
    with pytest.raises(RuntimeError, match="episode is done"):
        env.step()


def test_nonterminal_hit_triggers_immediate_assignment_redecision() -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.1,
        assignment_update_interval_s=5.0,
        max_steps=200,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(red_count=2, blue_count=2),
        sensor=SensorConfig(detection_range_m=200000.0),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=84, style="one_to_one")
    assert env.state is not None
    env.state.red[0].position_m = np.array([0.0, 9000.0, 0.0])
    env.state.red[0].velocity_mps = np.array([1000.0, 0.0, 0.0])
    env.state.blue[0].position_m = np.array([1.0, 9000.0, 0.0])
    env.state.blue[0].velocity_mps = np.zeros(3)
    env.last_observation = env.observation_layer.observe(env.state)
    first = env.step(
        red_action={
            "target_indices": np.array([0, 1], dtype=np.int64),
            "guidance_bias": np.zeros((2, 2), dtype=np.float64),
        }
    )
    assert not first.done
    assert first.info["assignment_event_boundary"]
    assert first.info["high_reward_settled"]
    assert first.info["low_reward_settled"]
    assert first.info["assignment_redecision_required"]
    assert first.info["reward_components"]["elapsed_s"] == pytest.approx(0.05)
    assert first.info["reward_low_components"]["elapsed_s"] == pytest.approx(0.05)
    assert first.info["reward_low_components"]["potential_discount"] == pytest.approx(
        1.0
    )

    second = env.step(
        red_action={
            "target_indices": np.array([-1, 1], dtype=np.int64),
            "guidance_bias": np.zeros((2, 2), dtype=np.float64),
        }
    )
    assert second.info["assignment_updated"]
    assert second.info["current_target_indices"] == [-1, 1]


def test_timeout_records_invalid_loss_zeroes_low_phi_and_ends_low_gae(
    network_device: torch.device,
) -> None:
    config = _environment_config(red_count=1, blue_count=1, max_steps=3)
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=44, style="one_to_one")
    step = env.step(
        red_action={
            "target_indices": np.array([0], dtype=np.int64),
            "guidance_bias": np.zeros((1, 2), dtype=np.float64),
        }
    )
    assert step.done
    assert step.info["termination_reason"] == "timeout"
    assert step.info["red_alive"] == [False]
    assert step.info["red_loss_events"][0]["loss_reason"] == "mission_timeout"
    assert ineffective_loss_rate(env.state) == 1.0
    assert step.info["reward_low_components"]["potential_next"] == [0.0]

    rollout_env = RedBlueEngagementEnv(config, device=network_device, record_replay=False)
    networks = _network_bundle(_model_config(), network_device)
    batch, _ = collect_rollout(
        rollout_env,
        *networks,
        steps=1,
        seed=45,
        style="one_to_one",
        red_count=1,
        blue_count=1,
        deterministic=True,
    )
    assert batch.dones_low[-1, 0, 0] == 1.0


def test_teammate_target_destruction_closes_low_option_without_negative_phi() -> None:
    config = _environment_config(red_count=2, blue_count=1, max_steps=10)
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=46, style="many_to_one")
    assert env.state is not None
    env.state.red[0].position_m = np.array([0.0, 9000.0, 0.0])
    env.state.red[0].velocity_mps = np.array([1000.0, 0.0, 0.0])
    env.state.red[1].position_m = np.array([-1000.0, 9000.0, 1000.0])
    env.state.red[1].velocity_mps = np.array([1000.0, 0.0, 0.0])
    env.state.blue[0].position_m = np.array([1.0, 9000.0, 0.0])
    env.state.blue[0].velocity_mps = np.zeros(3, dtype=np.float64)
    env.last_observation = env.observation_layer.observe(env.state)

    step = env.step(
        red_action={
            "target_indices": np.array([0, 0], dtype=np.int64),
            "guidance_bias": np.zeros((2, 2), dtype=np.float64),
        }
    )
    assert step.done
    low = step.info["reward_low_components"]
    assert low["potential_current"][1] > 0.0
    assert low["potential_delta"][1] == pytest.approx(0.0)
    assert low["option_boundary_reason"][1] == "target_destroyed"


def test_terminal_zero_option_boundary_settles_negative_start_potential() -> None:
    base = _environment_config(red_count=2, blue_count=1, max_steps=10)
    config = replace(
        base,
        reward=replace(
            base.reward,
            low_option_boundary_potential="terminal_zero",
        ),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=146, style="many_to_one")
    assert env.state is not None
    env.state.red[0].position_m = np.array([0.0, 9000.0, 0.0])
    env.state.red[0].velocity_mps = np.array([1000.0, 0.0, 0.0])
    env.state.red[1].position_m = np.array([-1000.0, 9000.0, 1000.0])
    env.state.red[1].velocity_mps = np.array([1000.0, 0.0, 0.0])
    env.state.blue[0].position_m = np.array([1.0, 9000.0, 0.0])
    env.state.blue[0].velocity_mps = np.zeros(3, dtype=np.float64)
    env.last_observation = env.observation_layer.observe(env.state)

    step = env.step(
        red_action={
            "target_indices": np.array([0, 0], dtype=np.int64),
            "guidance_bias": np.zeros((2, 2), dtype=np.float64),
        }
    )
    low = step.info["reward_low_components"]
    assert low["option_boundary_reason"][1] == "target_destroyed"
    assert low["potential_current"][1] > 0.0
    assert low["potential_next"][1] == 0.0
    assert low["potential_delta"][1] == pytest.approx(
        -low["potential_current"][1]
    )


@pytest.mark.parametrize("success", [True, False])
def test_terminal_active_share_time_credit_matches_mission_time_term(
    success: bool,
) -> None:
    base = _environment_config(red_count=2, blue_count=1, max_steps=3 if not success else 10)
    config = replace(
        base,
        reward=replace(
            base.reward,
            low_time_credit_mode="terminal_active_share",
            low_time_weight=2.0,
            low_option_boundary_potential="terminal_zero",
        ),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=147, style="many_to_one")
    assert env.state is not None
    if success:
        env.state.red[0].position_m = np.array([0.0, 9000.0, 0.0])
        env.state.red[0].velocity_mps = np.array([1000.0, 0.0, 0.0])
        env.state.red[1].position_m = np.array([-1000.0, 9000.0, 1000.0])
        env.state.red[1].velocity_mps = np.array([1000.0, 0.0, 0.0])
        env.state.blue[0].position_m = np.array([1.0, 9000.0, 0.0])
        env.state.blue[0].velocity_mps = np.zeros(3, dtype=np.float64)
        env.last_observation = env.observation_layer.observe(env.state)
    step = env.step(
        red_action={
            "target_indices": np.array([0, 0], dtype=np.int64),
            "guidance_bias": np.zeros((2, 2), dtype=np.float64),
        }
    )
    assert step.done
    low = step.info["reward_low_components"]
    high = step.info["reward_components"]
    assert low["time_credit_unassigned"] is False
    assert low["time_credit_total"] == pytest.approx(high["time_penalty"])
    assert sum(low["time_credit"]) == pytest.approx(
        -2.0 * low["normalized_terminal_time"]
    )
    assert low["time_credit"][0] == pytest.approx(low["time_credit"][1])


def test_terminal_time_credit_handles_no_active_low_agent() -> None:
    base = _environment_config(red_count=1, blue_count=1, max_steps=3)
    config = replace(
        base,
        reward=replace(
            base.reward,
            low_time_credit_mode="terminal_active_share",
            low_time_weight=2.0,
        ),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=148, style="one_to_one")
    step = env.step(
        red_action={
            "target_indices": np.array([-1], dtype=np.int64),
            "guidance_bias": np.zeros((1, 2), dtype=np.float64),
        }
    )
    assert step.done
    low = step.info["reward_low_components"]
    assert low["time_credit"] == [0.0]
    assert low["time_credit_total"] == 0.0
    assert low["time_credit_unassigned"] is True


def test_terminal_reward_configuration_is_applied() -> None:
    success_config = replace(
        _environment_config(red_count=1, blue_count=1, max_steps=10),
        reward=replace(RewardConfig(), terminal_success_reward=0.25),
    )
    success_env = RedBlueEngagementEnv(success_config, device="cpu", record_replay=False)
    success_env.reset(seed=47, style="one_to_one")
    assert success_env.state is not None
    success_env.state.red[0].position_m = np.array([0.0, 9000.0, 0.0])
    success_env.state.red[0].velocity_mps = np.array([1000.0, 0.0, 0.0])
    success_env.state.blue[0].position_m = np.array([1.0, 9000.0, 0.0])
    success_env.state.blue[0].velocity_mps = np.zeros(3, dtype=np.float64)
    success_env.last_observation = success_env.observation_layer.observe(success_env.state)
    success = success_env.step(
        red_action={
            "target_indices": np.array([0], dtype=np.int64),
            "guidance_bias": np.zeros((1, 2), dtype=np.float64),
        }
    )
    assert success.info["reward_components"]["terminal_outcome_adjustment"] == pytest.approx(0.25)

    timeout_config = replace(
        _environment_config(red_count=1, blue_count=1, max_steps=3),
        reward=replace(RewardConfig(), terminal_timeout_penalty=0.25),
    )
    timeout_env = RedBlueEngagementEnv(timeout_config, device="cpu", record_replay=False)
    timeout_env.reset(seed=48, style="one_to_one")
    timeout = timeout_env.step(
        red_action={
            "target_indices": np.array([0], dtype=np.int64),
            "guidance_bias": np.zeros((1, 2), dtype=np.float64),
        }
    )
    assert timeout.info["reward_components"]["terminal_outcome_adjustment"] == pytest.approx(-0.25)


def test_duration_aware_lambda_and_continuous_zem_gate() -> None:
    rewards = torch.ones(2, 1)
    values = torch.zeros(3, 1)
    dones = torch.zeros(2, 1)
    advantages, _ = generalized_advantage_estimation(
        rewards,
        values,
        dones,
        torch.ones(2, 1),
        torch.tensor([[0.9], [0.8]]),
    )
    assert advantages[:, 0].tolist() == pytest.approx([1.9, 1.0])

    config = _environment_config(red_count=1, blue_count=1, max_steps=10)
    missile = _state((0.0, 9000.0, 0.0), (0.0, 0.0, 0.0), 1.0, age_s=1.0)
    approaching = _state((1.0, 9000.0, 0.0), (-100.0, 0.0, 0.0), 1.0)
    receding = _state((1.0, 9000.0, 0.0), (100.0, 0.0, 0.0), 1.0)
    zem_approaching = low_intercept_potential(config, missile, approaching)[1]
    zem_receding = low_intercept_potential(config, missile, receding)[1]
    assert 0.0 < zem_approaching < 1.0
    assert 0.0 < zem_receding < 1.0
    assert abs(zem_approaching - zem_receding) < 0.02


def test_execution_observation_does_not_reuse_another_target_estimate() -> None:
    config = replace(
        _environment_config(red_count=1, blue_count=2),
        sensor=SensorConfig(detection_range_m=5000.0),
    )
    red = _state((0.0, 9000.0, 0.0), (1000.0, 0.0, 0.0), 1.0, age_s=1.0)
    red.current_target_index = 0
    red.guidance_mode = "locked"
    red.target_estimate_valid = True
    red.target_estimate_target_index = 0
    red.target_estimate_position_m = np.array([1000.0, 9000.0, 0.0])
    red.target_estimate_velocity_mps = np.zeros(3, dtype=np.float64)
    blue = [
        _state((1000.0, 9000.0, 0.0), (0.0, 0.0, 0.0), 1.0),
        _state((10000.0, 9000.0, 0.0), (0.0, 0.0, 0.0), 1.0),
    ]
    layer = ObservationLayer(config, device="cpu")
    layer.reset(seed=49)
    execution = layer.execution_inputs(EngagementState(red=[red], blue=blue), np.array([2]))
    target_features = execution.assigned_target[0, 0, 0].numpy()
    assert target_features[15] == 0.0
    assert np.count_nonzero(target_features[:15]) == 0


def test_reward_priority_rejects_unsupported_scale() -> None:
    with pytest.raises(ValueError, match="ineffective-loss priority"):
        RewardConfig().validate_lexicographic_priority(red_count=32, blue_count=6)


def test_full_success_reward_is_independent_highest_priority_event() -> None:
    reward = replace(RewardConfig(), terminal_success_reward=512.0)
    for blue_count in (4, 5, 6):
        reward.validate_lexicographic_priority(red_count=24, blue_count=blue_count)

    always_five_of_six = reward.high_damage_weight * (5.0 / 6.0)
    eighty_percent_full_success = 0.8 * (
        reward.high_damage_weight + reward.terminal_success_reward
    ) + 0.2 * (-reward.terminal_failure_penalty)
    assert always_five_of_six < eighty_percent_full_success


def _assignment_inputs(
    config: SwarmModelConfig,
    device: torch.device,
    *,
    batch: int = 2,
    agents: int = 3,
    friends: int = 3,
    targets: int = 4,
) -> AssignmentActorInputs:
    target_mask = torch.ones(batch, agents, targets, dtype=torch.bool, device=device)
    agent_mask = torch.ones(batch, agents, dtype=torch.bool, device=device)
    agent_mask[0, 1] = False
    return AssignmentActorInputs(
        self_state=torch.randn(batch, agents, config.d_self, device=device),
        friend_entities=torch.randn(batch, agents, friends, config.d_friend, device=device),
        friend_mask=torch.ones(batch, agents, friends, dtype=torch.bool, device=device),
        target_entities=torch.randn(batch, agents, targets, config.d_target, device=device),
        pair_state=torch.randn(batch, agents, targets, config.d_pair, device=device),
        current_assignment=torch.zeros(batch, agents, targets, device=device),
        target_mask=target_mask,
        environment_context=torch.randn(
            batch,
            agents,
            config.d_actor_context,
            device=device,
        ),
        target_assignment_counts=torch.rand(batch, agents, targets, device=device),
        target_entity_mask=target_mask.clone(),
        agent_mask=agent_mask,
        hidden=torch.zeros(batch, agents, config.d_model, device=device),
    )


def _execution_inputs(
    config: SwarmModelConfig,
    device: torch.device,
    *,
    batch: int = 2,
    agents: int = 3,
    friends: int = 3,
) -> OverloadBiasActorInputs:
    agent_mask = torch.ones(batch, agents, dtype=torch.bool, device=device)
    agent_mask[0, 1] = False
    target_mask = torch.ones(batch, agents, 1, dtype=torch.bool, device=device)
    target_mask[0, 2] = False
    return OverloadBiasActorInputs(
        self_state=torch.randn(batch, agents, config.d_execution_self, device=device),
        same_target_friends=torch.randn(
            batch,
            agents,
            friends,
            config.d_execution_friend,
            device=device,
        ),
        friend_mask=torch.zeros(batch, agents, friends, dtype=torch.bool, device=device),
        assigned_target=torch.randn(
            batch,
            agents,
            1,
            config.d_execution_target,
            device=device,
        ),
        target_mask=target_mask,
        environment_context=torch.randn(
            batch,
            agents,
            config.d_execution_context,
            device=device,
        ),
        agent_mask=agent_mask,
        hidden=torch.zeros(batch, agents, config.d_model, device=device),
    )


def _state(
    position_m: tuple[float, float, float],
    velocity_mps: tuple[float, float, float],
    mass_kg: float,
    *,
    fuel_mass_kg: float = 0.0,
    age_s: float = 0.0,
) -> ThreeDoFState:
    return ThreeDoFState(
        position_m=np.asarray(position_m, dtype=np.float64),
        velocity_mps=np.asarray(velocity_mps, dtype=np.float64),
        mass_kg=mass_kg,
        fuel_mass_kg=fuel_mass_kg,
        age_s=age_s,
    )


def _network_bundle(
    config: SwarmModelConfig,
    device: torch.device,
) -> tuple[
    TargetAssignmentActor,
    OverloadBiasActor,
    TargetAssignmentCritic,
    OverloadBiasCritic,
]:
    return (
        TargetAssignmentActor(config).to(device),
        OverloadBiasActor(config).to(device),
        TargetAssignmentCritic(config).to(device),
        OverloadBiasCritic(config).to(device),
    )


def test_schema14_config_and_assignment_network_contract(network_device: torch.device) -> None:
    torch.manual_seed(7)
    config = _model_config()
    assert CHECKPOINT_SCHEMA_VERSION == 14
    assert config.d_bias == 2
    assert config.d_execution_context == 4
    assert config.d_global_context == 8
    with pytest.raises(ValueError, match="exactly 2"):
        replace(config, d_bias=3).validate()
    with pytest.raises(ValueError, match="observation feature dimensions"):
        replace(config, d_self=11).validate()
    with pytest.raises(ValueError, match="assignment_stickiness_logit_bonus"):
        replace(config, assignment_stickiness_logit_bonus=-0.1).validate()

    actor = TargetAssignmentActor(config).to(network_device)
    critic = TargetAssignmentCritic(config).to(network_device)
    inputs = _assignment_inputs(config, network_device)
    output = actor(inputs)
    active = inputs.agent_mask.bool()

    assert output.actions.target.shape == (2, 3)
    assert output.next_hidden.shape == (2, 3, config.d_model)
    assert output.log_prob.shape == (2, 3)
    assert output.entropy.shape == (2, 3)
    assert output.target_probabilities.shape == (2, 3, 4)
    assert output.assignment_matrix.shape == (2, 3, 3)
    torch.testing.assert_close(output.joint_log_prob, output.log_prob.sum(dim=-1))
    torch.testing.assert_close(output.joint_entropy, output.entropy.sum(dim=-1))
    assert torch.count_nonzero(output.assignment_matrix[~active]) == 0
    assert torch.count_nonzero(output.next_hidden[~active]) == 0
    evaluation = actor.evaluate_actions(inputs, output.actions)
    assert torch.allclose(
        evaluation.log_prob[active],
        output.log_prob.detach()[active],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(evaluation.joint_log_prob, evaluation.log_prob.sum(dim=-1))

    critic_inputs = AssignmentCriticInputs(
        global_red=torch.randn(2, 3, config.d_global_red, device=network_device),
        red_mask=torch.ones(2, 3, dtype=torch.bool, device=network_device),
        global_blue=torch.randn(2, 2, config.d_global_blue, device=network_device),
        blue_mask=torch.ones(2, 2, dtype=torch.bool, device=network_device),
        global_context=torch.randn(2, config.d_global_context, device=network_device),
        target_assignment_counts=torch.rand(2, 2, device=network_device),
        pair_state=torch.randn(2, 3, 2, config.d_pair, device=network_device),
        current_assignment=torch.zeros(2, 3, 2, device=network_device),
    )
    critic_output = critic(critic_inputs)
    assert critic_output.value.shape == (2,)
    assert critic_output.value_components.shape == (2, config.d_value_components)
    assert torch.isfinite(critic_output.value).all()
    torch.testing.assert_close(
        critic_output.value,
        critic_output.value_components.sum(dim=-1),
    )
    (-evaluation.log_prob[active].mean() + critic_output.value.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in actor.parameters())
    assert any(parameter.grad is not None for parameter in critic.parameters())


def test_assignment_hysteresis_increases_current_target_probability_without_extra_head_pass(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260811)
    base_config = _model_config()
    sticky_config = replace(
        base_config,
        assignment_stickiness_logit_bonus=1.0,
    )
    base_actor = TargetAssignmentActor(base_config).to(network_device)
    sticky_actor = TargetAssignmentActor(sticky_config).to(network_device)
    sticky_actor.load_state_dict(base_actor.state_dict())
    inputs = _assignment_inputs(base_config, network_device)
    inputs.current_assignment.zero_()
    inputs.current_assignment[..., 1] = 1.0

    head_calls = 0

    def count_head_calls(*_args) -> None:
        nonlocal head_calls
        head_calls += 1

    handle = sticky_actor.target_head.register_forward_hook(count_head_calls)
    with torch.no_grad():
        base_output = base_actor(inputs, deterministic=True)
        sticky_output = sticky_actor(inputs, deterministic=True)
    handle.remove()

    active = inputs.agent_mask.bool()
    assert head_calls == inputs.self_state.shape[1]
    assert torch.all(
        sticky_output.target_probabilities[..., 1][active]
        > base_output.target_probabilities[..., 1][active]
    )


def test_assignment_hysteresis_does_not_reward_reserved_no_target_slot(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260812)
    base_config = _model_config()
    sticky_config = replace(
        base_config,
        assignment_stickiness_logit_bonus=1.0,
    )
    base_actor = TargetAssignmentActor(base_config).to(network_device)
    sticky_actor = TargetAssignmentActor(sticky_config).to(network_device)
    sticky_actor.load_state_dict(base_actor.state_dict())
    inputs = _assignment_inputs(base_config, network_device)
    inputs.current_assignment.zero_()
    inputs.current_assignment[..., base_config.no_target_index] = 1.0

    with torch.no_grad():
        base_output = base_actor(inputs, deterministic=True)
        sticky_output = sticky_actor(inputs, deterministic=True)

    torch.testing.assert_close(
        sticky_output.target_probabilities,
        base_output.target_probabilities,
    )
    torch.testing.assert_close(
        sticky_output.joint_log_prob,
        base_output.joint_log_prob,
    )
    assert torch.equal(sticky_output.actions.target, base_output.actions.target)


def test_assignment_hysteresis_ignores_unavailable_current_physical_target(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260812)
    base_config = _model_config()
    sticky_config = replace(
        base_config,
        assignment_stickiness_logit_bonus=1.0,
    )
    base_actor = TargetAssignmentActor(base_config).to(network_device)
    sticky_actor = TargetAssignmentActor(sticky_config).to(network_device)
    sticky_actor.load_state_dict(base_actor.state_dict())
    inputs = _assignment_inputs(base_config, network_device)
    inputs.current_assignment.zero_()
    inputs.current_assignment[..., 1] = 1.0
    inputs.target_mask[0, 0, 1] = False
    inputs.target_entity_mask[0, 0, 1] = False

    with torch.no_grad():
        base_output = base_actor(inputs, deterministic=True)
        sticky_output = sticky_actor(inputs, deterministic=True)

    torch.testing.assert_close(
        sticky_output.target_probabilities[0, 0],
        base_output.target_probabilities[0, 0],
    )
    assert sticky_output.target_probabilities[0, 0, 1].item() == 0.0


def test_execution_actor_bias_is_bounded_and_inactive_rows_are_zero(
    network_device: torch.device,
) -> None:
    torch.manual_seed(11)
    config = _model_config()
    actor = OverloadBiasActor(config).to(network_device)
    inputs = _execution_inputs(config, network_device)
    output = actor(inputs)
    deterministic = actor(inputs, deterministic=True)
    active = inputs.agent_mask.bool() & inputs.target_mask.any(dim=-1)

    assert torch.all(actor.bias_log_std == -2.5)
    assert torch.count_nonzero(deterministic.bias_matrix) == 0
    assert output.bias_matrix.shape == (2, 3, 2)
    assert torch.isfinite(output.bias_matrix).all()
    assert torch.all(output.bias_matrix >= -1.0)
    assert torch.all(output.bias_matrix <= 1.0)
    for value in (
        output.bias_matrix,
        output.next_hidden,
        output.log_prob,
        output.entropy,
    ):
        mask = ~active.unsqueeze(-1).expand_as(value) if value.dim() == 3 else ~active
        assert torch.count_nonzero(value.masked_select(mask)) == 0

    evaluation = actor.evaluate_actions(inputs, output.bias_matrix.detach())
    assert torch.allclose(
        evaluation.log_prob[active],
        output.log_prob.detach()[active],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.count_nonzero(evaluation.log_prob[~active]) == 0
    loss = -evaluation.log_prob[active].mean() - 0.01 * evaluation.entropy[active].mean()
    loss.backward()
    assert any(parameter.grad is not None for parameter in actor.parameters())


def test_radial_tanh_disk_round_trip_jacobian_and_log_prob() -> None:
    pre_tanh = torch.tensor(
        [[0.0, 0.0], [0.3, -0.4], [2.0, 1.0]],
        dtype=torch.float64,
    )
    action, log_abs_det = _radial_tanh_forward(pre_tanh)
    restored, inverse_log_abs_det = _radial_tanh_inverse(action)
    torch.testing.assert_close(restored, pre_tanh, atol=1.0e-9, rtol=1.0e-9)
    torch.testing.assert_close(
        inverse_log_abs_det,
        log_abs_det,
        atol=1.0e-9,
        rtol=1.0e-9,
    )
    assert (torch.linalg.vector_norm(action, dim=-1) < 1.0).all()
    assert torch.isfinite(log_abs_det).all()

    point = torch.tensor([0.3, -0.4], dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        lambda value: _radial_tanh_forward(value.unsqueeze(0))[0].squeeze(0),
        point,
    )
    numerical_log_abs_det = torch.logdet(jacobian)
    analytic_log_abs_det = _radial_tanh_forward(point.unsqueeze(0))[1].squeeze(0)
    torch.testing.assert_close(
        analytic_log_abs_det,
        numerical_log_abs_det,
        atol=1.0e-9,
        rtol=1.0e-9,
    )

    mu = torch.zeros_like(action)
    log_std = torch.full_like(action, -1.5)
    assert torch.isfinite(_radial_tanh_log_prob(action, mu, log_std)).all()
    with pytest.raises(ValueError, match="smaller than one"):
        _radial_tanh_inverse(torch.tensor([[1.0, 0.0]], dtype=torch.float64))


def test_execution_actor_radial_disk_actions_match_evaluation(
    network_device: torch.device,
) -> None:
    torch.manual_seed(111)
    config = replace(
        _model_config(),
        execution_action_distribution="radial_tanh_disk",
    )
    actor = OverloadBiasActor(config).to(network_device)
    inputs = _execution_inputs(config, network_device)
    output = actor(inputs)
    active = inputs.agent_mask.bool() & inputs.target_mask.any(dim=-1)

    assert output.action_distribution == "radial_tanh_disk"
    assert (
        torch.linalg.vector_norm(output.bias_matrix[active], dim=-1) < 1.0
    ).all()
    evaluation = actor.evaluate_actions(inputs, output.bias_matrix.detach())
    torch.testing.assert_close(
        evaluation.log_prob[active],
        output.log_prob.detach()[active],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.isfinite(evaluation.log_prob).all()


def test_assignment_actor_is_equivariant_to_target_permutation(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260717)
    config = _model_config()
    actor = TargetAssignmentActor(config).to(network_device).eval()
    inputs = _assignment_inputs(
        config,
        network_device,
        batch=1,
        agents=2,
        friends=2,
        targets=4,
    )
    inputs.agent_mask[:] = True
    permutation = torch.tensor([0, 3, 1, 2], device=network_device)
    permuted = replace(
        inputs,
        target_entities=inputs.target_entities[:, :, permutation],
        pair_state=inputs.pair_state[:, :, permutation],
        current_assignment=inputs.current_assignment[:, :, permutation],
        target_mask=inputs.target_mask[:, :, permutation],
        environment_context=inputs.environment_context,
        target_assignment_counts=inputs.target_assignment_counts[:, :, permutation],
        target_entity_mask=inputs.target_entity_mask[:, :, permutation],
    )

    torch.manual_seed(91)
    original_output = actor(inputs, deterministic=True)
    torch.manual_seed(91)
    permuted_output = actor(permuted, deterministic=True)
    torch.testing.assert_close(
        permuted_output.target_probabilities,
        original_output.target_probabilities[:, :, permutation],
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.equal(
        permutation[permuted_output.actions.target],
        original_output.actions.target,
    )


def test_execution_observation_filters_same_target_and_updates_immediately() -> None:
    config = _environment_config(red_count=3, blue_count=2)
    config = replace(
        config,
        sensor=replace(
            config.sensor,
            position_noise_m=100.0,
            velocity_noise_mps=5.0,
        ),
    )
    red = [
        _state((0.0, 9000.0, 0.0), (300.0, 0.0, 0.0), config.missile.dry_mass_kg, age_s=config.policy_entry_time_s),
        _state((1000.0, 9100.0, 200.0), (310.0, 5.0, 0.0), config.missile.dry_mass_kg, age_s=config.policy_entry_time_s),
        _state((-500.0, 8900.0, -300.0), (290.0, 0.0, 5.0), config.missile.dry_mass_kg, age_s=config.policy_entry_time_s),
    ]
    blue = [
        _state((20000.0, 9200.0, 1000.0), (350.0, 0.0, 0.0), 1.0),
        _state((18000.0, 10000.0, -4000.0), (330.0, 0.0, 10.0), 1.0),
    ]
    engagement = EngagementState(red=red, blue=blue)
    layer = ObservationLayer(config, device="cpu")
    layer.reset(seed=13)
    layer.observe(engagement)

    initial = layer.execution_inputs(engagement, np.array([1, 1, 2], dtype=np.int64))
    assert initial.same_target_friends.shape == (1, 3, 3, 14)
    assert initial.assigned_target.shape == (1, 3, 1, 17)
    assert initial.environment_context.shape == (1, 3, 4)
    assert initial.friend_mask[0, 0].tolist() == [False, True, False]
    assert initial.friend_mask[0, 1].tolist() == [True, False, False]
    assert initial.friend_mask[0, 2].tolist() == [False, False, False]

    changed = layer.execution_inputs(engagement, np.array([2, 1, 2], dtype=np.int64))
    assert changed.friend_mask[0, 0].tolist() == [False, False, True]
    assert changed.friend_mask[0, 2].tolist() == [True, False, False]
    assert not torch.allclose(initial.assigned_target[0, 0], changed.assigned_target[0, 0])
    assert changed.assigned_target[0, 0, 0, -2] == pytest.approx(1.0)
    assert changed.assigned_target[0, 0, 0, -1] == pytest.approx(1.0)
    assert not torch.allclose(
        changed.assigned_target[0, 0, 0],
        layer.execution_inputs(engagement, np.array([2, 1, 2], dtype=np.int64)).assigned_target[0, 0, 0],
    )
    assert engagement.red[0].current_target_index == -1
    assert changed.target_mask.all()


def test_target_identity_is_relational_not_a_continuous_entity_feature() -> None:
    config = _environment_config(red_count=1, blue_count=2)
    red = _state(
        (0.0, 9000.0, 0.0),
        (800.0, 0.0, 0.0),
        config.missile.dry_mass_kg,
        age_s=config.policy_entry_time_s,
    )
    blue = [
        _state((30000.0, 9000.0, 0.0), (300.0, 0.0, 0.0), 1.0),
        _state((32000.0, 9500.0, 1000.0), (320.0, 0.0, 0.0), 1.0),
    ]
    unassigned_state = EngagementState(red=[red], blue=blue)
    assigned_state = unassigned_state.copy()
    assigned_state.red[0].current_target_index = 1
    first_layer = ObservationLayer(config, device="cpu")
    second_layer = ObservationLayer(config, device="cpu")
    first_layer.reset(seed=4)
    second_layer.reset(seed=4)
    unassigned = first_layer.observe(unassigned_state)
    assigned = second_layer.observe(assigned_state)

    torch.testing.assert_close(
        unassigned.assignment_actor_inputs.self_state,
        assigned.assignment_actor_inputs.self_state,
    )
    torch.testing.assert_close(
        unassigned.assignment_critic_inputs.global_red,
        assigned.assignment_critic_inputs.global_red,
    )
    assert unassigned.assignment_actor_inputs.current_assignment[0, 0].tolist() == [
        1.0,
        0.0,
        0.0,
    ]
    assert assigned.assignment_actor_inputs.current_assignment[0, 0].tolist() == [
        0.0,
        0.0,
        1.0,
    ]
    assert not torch.allclose(
        unassigned.assignment_actor_inputs.pair_state,
        assigned.assignment_actor_inputs.pair_state,
    )


def test_temporary_track_loss_keeps_low_actor_active_and_preserves_hidden(
    network_device: torch.device,
) -> None:
    config = replace(
        _environment_config(red_count=1, blue_count=1),
        sensor=SensorConfig(detection_range_m=1000.0),
    )
    red = _state(
        (0.0, 9000.0, 0.0),
        (800.0, 0.0, 0.0),
        config.missile.dry_mass_kg,
        age_s=config.policy_entry_time_s,
    )
    red.current_target_index = 0
    red.guidance_mode = "lock_hold"
    red.seeker_locked = True
    red.target_estimate_valid = True
    red.target_estimate_position_m = np.array([20000.0, 9000.0, 0.0])
    red.target_estimate_velocity_mps = np.array([300.0, 0.0, 0.0])
    red.target_estimate_age_s = 0.25
    blue = _state((20000.0, 9000.0, 0.0), (300.0, 0.0, 0.0), 1.0)
    layer = ObservationLayer(config, device=network_device)
    inputs = layer.execution_inputs(
        EngagementState(red=[red], blue=[blue]),
        np.array([1], dtype=np.int64),
        hidden=torch.ones(1, 1, _model_config().d_model, device=network_device),
    )
    assert inputs.target_mask.item()
    assert 0.0 < inputs.assigned_target[0, 0, 0, -2].item() < 1.0

    torch.manual_seed(73)
    output = OverloadBiasActor(_model_config()).to(network_device)(inputs)
    assert torch.count_nonzero(output.next_hidden) > 0
    assert torch.count_nonzero(output.bias_matrix) > 0
    assert torch.isfinite(output.log_prob).all()
    assert torch.isfinite(output.entropy).all()
    assert output.entropy.item() != 0.0


def test_temporary_track_loss_keeps_current_target_available_to_high_actor() -> None:
    config = replace(
        _environment_config(red_count=1, blue_count=2),
        sensor=SensorConfig(detection_range_m=1000.0),
    )
    red = _state(
        (0.0, 9000.0, 0.0),
        (800.0, 0.0, 0.0),
        config.missile.dry_mass_kg,
        age_s=config.policy_entry_time_s,
    )
    red.current_target_index = 0
    red.guidance_mode = "lock_hold"
    red.seeker_locked = True
    red.target_estimate_valid = True
    red.target_estimate_target_index = 0
    red.target_estimate_position_m = np.array([20000.0, 9000.0, 0.0])
    red.target_estimate_velocity_mps = np.array([300.0, 0.0, 0.0])
    red.target_estimate_age_s = 0.25
    targets = [
        _state((30000.0, 9000.0, 0.0), (300.0, 0.0, 0.0), 1.0),
        _state((25000.0, 9000.0, 1000.0), (300.0, 0.0, 0.0), 1.0),
    ]

    observation = ObservationLayer(config).observe(
        EngagementState(red=[red], blue=targets)
    )
    inputs = observation.assignment_actor_inputs

    assert inputs.target_mask[0, 0].tolist() == [True, True, False]
    assert inputs.current_assignment[0, 0].tolist() == [0.0, 1.0, 0.0]
    assert inputs.target_entities[0, 0, 1, 0].item() == pytest.approx(0.1)
    assert 0.0 < inputs.target_entities[0, 0, 1, -1].item() < 1.0


def test_scalar_critic_mode_outputs_one_semantic_value(
    network_device: torch.device,
) -> None:
    config = replace(
        _model_config(),
        d_value_components=1,
        critic_value_head_mode="scalar",
    )
    assignment_critic = TargetAssignmentCritic(config).to(network_device)
    execution_critic = OverloadBiasCritic(config).to(network_device)
    assignment_inputs = AssignmentCriticInputs(
        global_red=torch.randn(2, 3, config.d_global_red, device=network_device),
        red_mask=torch.ones(2, 3, dtype=torch.bool, device=network_device),
        global_blue=torch.randn(2, 2, config.d_global_blue, device=network_device),
        blue_mask=torch.ones(2, 2, dtype=torch.bool, device=network_device),
        global_context=torch.randn(2, config.d_global_context, device=network_device),
        target_assignment_counts=torch.zeros(2, 2, device=network_device),
        pair_state=torch.randn(2, 3, 2, config.d_pair, device=network_device),
        current_assignment=torch.zeros(2, 3, 2, device=network_device),
    )
    execution_inputs = OverloadBiasCriticInputs(
        global_red=assignment_inputs.global_red,
        red_mask=assignment_inputs.red_mask,
        global_blue=assignment_inputs.global_blue,
        blue_mask=assignment_inputs.blue_mask,
        applied_bias=torch.zeros(2, 3, 2, device=network_device),
        global_context=assignment_inputs.global_context,
        pair_state=assignment_inputs.pair_state,
        current_assignment=assignment_inputs.current_assignment,
        hidden=torch.zeros(2, 3, config.d_model, device=network_device),
    )

    assignment_output = assignment_critic(assignment_inputs)
    execution_output = execution_critic(execution_inputs)
    assert assignment_output.value_components.shape == (2, 1)
    assert execution_output.value_components.shape == (2, 3, 1)
    torch.testing.assert_close(
        assignment_output.value,
        assignment_output.value_components[..., 0],
    )
    torch.testing.assert_close(
        execution_output.value,
        execution_output.value_components[..., 0],
    )
    with pytest.raises(ValueError, match="scalar critic requires"):
        replace(config, d_value_components=5).validate()


def test_assignment_critic_can_use_five_latent_components_with_scalar_low_critic(
    network_device: torch.device,
) -> None:
    config = replace(
        _model_config(),
        d_value_components=1,
        critic_value_head_mode="scalar",
        assignment_critic_value_head_mode="latent_sum",
    )
    config.validate()
    assignment_critic = TargetAssignmentCritic(config).to(network_device)
    execution_critic = OverloadBiasCritic(config).to(network_device)
    assignment_inputs = AssignmentCriticInputs(
        global_red=torch.randn(2, 3, config.d_global_red, device=network_device),
        red_mask=torch.ones(2, 3, dtype=torch.bool, device=network_device),
        global_blue=torch.randn(2, 2, config.d_global_blue, device=network_device),
        blue_mask=torch.ones(2, 2, dtype=torch.bool, device=network_device),
        global_context=torch.randn(2, config.d_global_context, device=network_device),
        target_assignment_counts=torch.zeros(2, 2, device=network_device),
        pair_state=torch.randn(2, 3, 2, config.d_pair, device=network_device),
        current_assignment=torch.zeros(2, 3, 2, device=network_device),
    )
    execution_inputs = OverloadBiasCriticInputs(
        global_red=assignment_inputs.global_red,
        red_mask=assignment_inputs.red_mask,
        global_blue=assignment_inputs.global_blue,
        blue_mask=assignment_inputs.blue_mask,
        applied_bias=torch.zeros(2, 3, 2, device=network_device),
        global_context=assignment_inputs.global_context,
        pair_state=assignment_inputs.pair_state,
        current_assignment=assignment_inputs.current_assignment,
        hidden=torch.zeros(2, 3, config.d_model, device=network_device),
    )

    assignment_output = assignment_critic(assignment_inputs)
    execution_output = execution_critic(execution_inputs)

    assert assignment_output.value_components.shape == (2, 5)
    assert execution_output.value_components.shape == (2, 3, 1)
    torch.testing.assert_close(
        assignment_output.value,
        assignment_output.value_components.sum(dim=-1),
    )


def test_execution_critic_encodes_sets_bias_and_gru(network_device: torch.device) -> None:
    torch.manual_seed(17)
    config = _model_config()
    critic = OverloadBiasCritic(config).to(network_device).eval()
    batch, red_count, blue_count = 2, 3, 2
    inputs = OverloadBiasCriticInputs(
        global_red=torch.randn(batch, red_count, config.d_global_red, device=network_device),
        red_mask=torch.ones(batch, red_count, dtype=torch.bool, device=network_device),
        global_blue=torch.randn(batch, blue_count, config.d_global_blue, device=network_device),
        blue_mask=torch.ones(batch, blue_count, dtype=torch.bool, device=network_device),
        applied_bias=torch.randn(batch, red_count, 2, device=network_device).tanh(),
        global_context=torch.randn(batch, config.d_global_context, device=network_device),
        pair_state=torch.randn(
            batch,
            red_count,
            blue_count,
            config.d_pair,
            device=network_device,
        ),
        current_assignment=torch.zeros(
            batch,
            red_count,
            blue_count,
            device=network_device,
        ),
        hidden=torch.zeros(batch, red_count, config.d_model, device=network_device),
    )
    output = critic(inputs)
    assert output.value.shape == (batch, red_count)
    assert output.value_components.shape == (
        batch,
        red_count,
        config.d_value_components,
    )
    assert output.next_hidden.shape == (batch, red_count, config.d_model)
    assert torch.isfinite(output.value).all()
    assert torch.isfinite(output.next_hidden).all()

    red_permutation = torch.tensor([2, 0, 1], device=network_device)
    blue_permutation = torch.tensor([1, 0], device=network_device)
    permuted = OverloadBiasCriticInputs(
        global_red=inputs.global_red[:, red_permutation],
        red_mask=inputs.red_mask[:, red_permutation],
        global_blue=inputs.global_blue[:, blue_permutation],
        blue_mask=inputs.blue_mask[:, blue_permutation],
        applied_bias=inputs.applied_bias[:, red_permutation],
        global_context=inputs.global_context,
        pair_state=inputs.pair_state[:, red_permutation][:, :, blue_permutation],
        current_assignment=inputs.current_assignment[:, red_permutation][
            :, :, blue_permutation
        ],
        hidden=inputs.hidden[:, red_permutation],
    )
    permuted_output = critic(permuted)
    assert torch.allclose(
        output.value[:, red_permutation],
        permuted_output.value,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert torch.allclose(
        output.next_hidden[:, red_permutation],
        permuted_output.next_hidden,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    mispaired_output = critic(
        replace(inputs, applied_bias=inputs.applied_bias[:, red_permutation])
    )
    assert not torch.allclose(
        output.next_hidden,
        mispaired_output.next_hidden,
        atol=1.0e-7,
        rtol=1.0e-7,
    )

    recurrent = critic(replace(inputs, hidden=output.next_hidden))
    assert not torch.allclose(output.next_hidden, recurrent.next_hidden)
    all_masked = critic(
        replace(
            inputs,
            red_mask=torch.zeros_like(inputs.red_mask),
            blue_mask=torch.zeros_like(inputs.blue_mask),
        )
    )
    assert torch.isfinite(all_masked.value).all()
    assert torch.isfinite(all_masked.next_hidden).all()
    (output.value.square().mean() + output.next_hidden.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in critic.parameters())


def test_red_action_requires_exact_two_column_finite_bounded_bias() -> None:
    valid = RedAction(
        target_indices=np.array([0, 1], dtype=np.int64),
        guidance_bias=np.array([[0.2, -0.4], [1.0, -1.0]], dtype=np.float64),
    )
    assert valid.guidance_bias.shape == (2, 2)

    for invalid_shape in ((2, 1), (2, 3), (4,)):
        with pytest.raises(ValueError, match="guidance_bias shape"):
            RedAction(
                target_indices=np.array([0, 1], dtype=np.int64),
                guidance_bias=np.zeros(invalid_shape, dtype=np.float64),
            )
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        RedAction(
            target_indices=np.array([0], dtype=np.int64),
            guidance_bias=np.array([[1.01, 0.0]], dtype=np.float64),
        )
    with pytest.raises(ValueError, match="finite"):
        RedAction(
            target_indices=np.array([0], dtype=np.int64),
            guidance_bias=np.array([[math.nan, 0.0]], dtype=np.float64),
        )


def test_pn_and_auxiliary_bias_use_independent_bias_and_final_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EnvironmentConfig(scenario=ScenarioConfig(red_count=1, blue_count=1))
    physics = ThreeDoFPhysicsLayer(config)
    missile = _state(
        (0.0, 9000.0, 0.0),
        (1000.0, 0.0, 0.0),
        config.missile.dry_mass_kg,
        age_s=config.missile.boost_duration_s,
    )
    missile.seeker_locked = True
    target = _state(
        (10000.0, 9000.0, 0.0),
        (300.0, 0.0, 0.0),
        1.0,
    )
    monkeypatch.setattr(
        physics.guidance,
        "command",
        lambda *_: np.zeros(3, dtype=np.float64),
    )
    baseline = physics._step_missile(missile.copy(), target.copy(), np.zeros(2))
    first_column = physics._step_missile(
        missile.copy(),
        target.copy(),
        np.array([0.5, 0.0]),
    )
    second_column = physics._step_missile(
        missile.copy(),
        target.copy(),
        np.array([0.0, 0.5]),
    )
    assert np.allclose(baseline.gravity_load_body_g[1:], [1.0, 0.0])
    assert np.allclose(baseline.final_load_body_g[1:], [1.0, 0.0])
    assert np.allclose(first_column.bias_load_body_g[1:], [2.5, 0.0])
    assert np.allclose(second_column.bias_load_body_g[1:], [0.0, 2.5])
    assert not np.isclose(first_column.velocity_mps[1], baseline.velocity_mps[1])
    assert not np.isclose(second_column.velocity_mps[2], baseline.velocity_mps[2])

    monkeypatch.setattr(
        physics.guidance,
        "command",
        lambda *_: np.array([0.0, 34.0 * G0, 0.0], dtype=np.float64),
    )
    real_clip_norm = clip_norm
    calls: list[tuple[np.ndarray, float]] = []

    def counted_clip_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
        calls.append((np.asarray(vector).copy(), float(maximum)))
        return real_clip_norm(vector, maximum)

    monkeypatch.setattr(physics_module, "clip_norm", counted_clip_norm)
    saturated = physics._step_missile(
        missile.copy(),
        target.copy(),
        np.ones(2, dtype=np.float64),
    )
    expected_bias = np.ones(2) / math.sqrt(2.0) * 5.0
    expected_input = np.array([35.0, 0.0]) + expected_bias
    expected = real_clip_norm(expected_input, 35.0)
    assert len(calls) == 1
    assert np.allclose(saturated.bias_load_body_g[1:], expected_bias)
    assert np.linalg.norm(saturated.bias_load_body_g[1:]) == pytest.approx(5.0)
    assert np.allclose(calls[0][0], expected_input)
    assert calls[0][1] == 35.0
    assert np.allclose(saturated.final_load_body_g[1:], expected)
    assert np.linalg.norm(saturated.final_load_body_g[1:]) <= 35.0 + 1.0e-12


def test_target_switch_resets_seeker_to_acquisition_fov() -> None:
    config = _environment_config()
    physics = ThreeDoFPhysicsLayer(config)
    missile = _state(
        (0.0, 9000.0, 0.0),
        (500.0, 0.0, 0.0),
        config.missile.dry_mass_kg,
        age_s=config.missile.boost_duration_s,
    )
    missile.current_target_index = 0
    missile.seeker_locked = True
    target = _state(
        (10000.0, 9000.0, 10000.0),
        (0.0, 0.0, 0.0),
        1.0,
    )
    same_target = physics._step_missile(
        missile.copy(),
        target,
        np.zeros(2),
        target_index=0,
    )
    switched_target = physics._step_missile(
        missile.copy(),
        target,
        np.zeros(2),
        target_index=1,
    )
    assert same_target.seeker_locked
    assert not switched_target.seeker_locked


def test_seeker_loss_predicts_for_hold_then_enters_inertial_and_reacquires() -> None:
    config = EnvironmentConfig(scenario=ScenarioConfig(red_count=1, blue_count=1))
    physics = ThreeDoFPhysicsLayer(config)
    missile = _state(
        (0.0, 9000.0, 0.0),
        (1000.0, 0.0, 0.0),
        config.missile.dry_mass_kg,
        age_s=config.missile.boost_duration_s,
    )
    missile.current_target_index = 0
    target = _state((10000.0, 9000.0, 0.0), (100.0, 0.0, 0.0), 1.0)
    tracked = physics._step_missile(missile, target, np.zeros(2), target_index=0)
    assert tracked.seeker_locked
    assert tracked.guidance_mode == "locked"
    last_visible_position = tracked.target_estimate_position_m.copy()

    target.position_m = np.array([0.0, 9000.0, 10000.0], dtype=np.float64)
    held = physics._step_missile(tracked, target, np.zeros(2), target_index=0)
    assert held.seeker_locked
    assert held.guidance_mode == "lock_hold"
    np.testing.assert_allclose(
        held.target_estimate_position_m,
        last_visible_position + target.velocity_mps * config.time_step_s,
    )

    lost = held
    loss_steps = int(round(config.missile.fov_break_hold_s / config.time_step_s))
    for _ in range(loss_steps - 1):
        lost = physics._step_missile(lost, target, np.zeros(2), target_index=0)
    assert not lost.seeker_locked
    assert lost.guidance_mode == "inertial"
    assert lost.fov_out_time_s == pytest.approx(0.75)
    assert lost.target_estimate_age_s == pytest.approx(0.75)

    forward = lost.velocity_mps / np.linalg.norm(lost.velocity_mps)
    target.position_m = lost.position_m + 10000.0 * forward
    reacquired = physics._step_missile(lost, target, np.zeros(2), target_index=0)
    assert reacquired.seeker_locked
    assert reacquired.guidance_mode == "locked"
    assert reacquired.target_estimate_age_s == 0.0


def test_environment_emits_four_guidance_telemetry_matrices() -> None:
    config = _environment_config(red_count=2, blue_count=1, max_steps=4)
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=True)
    env.reset(seed=23, style="many_to_one")
    assert env.state is not None
    guidance_bias = np.array([[0.2, -0.3], [-0.4, 0.5]], dtype=np.float64)
    step = env.step(
        red_action={
            "target_indices": np.zeros(2, dtype=np.int64),
            "guidance_bias": guidance_bias,
        }
    )

    telemetry_shapes = {
        "guidance_bias_matrix": (2, 2),
        "pn_load_body_g": (2, 3),
        "bias_load_body_g": (2, 3),
        "gravity_load_body_g": (2, 3),
        "final_load_body_g": (2, 3),
    }
    for key, expected_shape in telemetry_shapes.items():
        values = np.asarray(step.info[key], dtype=np.float64)
        assert values.shape == expected_shape
        assert np.isfinite(values).all()
    assert np.allclose(step.info["guidance_bias_matrix"], guidance_bias)
    assert torch.allclose(
        step.observation.execution_critic_inputs.applied_bias[0],
        torch.as_tensor(guidance_bias, dtype=torch.float32),
    )
    assert env.replay_layer[-1].info["guidance_bias_matrix"] == step.info["guidance_bias_matrix"]


def test_dead_target_is_not_repenalized_inside_held_control_period() -> None:
    config = replace(
        _environment_config(red_count=1, blue_count=1, max_steps=5),
        bias_update_interval_s=0.1,
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=27, style="one_to_one")
    action = {
        "target_indices": np.zeros(1, dtype=np.int64),
        "guidance_bias": np.zeros((1, 2), dtype=np.float64),
    }
    env.step(red_action=action)
    assert env.state is not None
    env.state.blue[0].alive = False
    env.last_observation = env.observation_layer.observe(env.state)
    held_step = env.step(red_action=action)
    assert held_step.info["reward_components"]["terminal_reason"] == "success"


def test_environment_holds_assignment_and_bias_on_independent_periods() -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.1,
        assignment_update_interval_s=0.2,
        max_steps=10,
        scenario=ScenarioConfig(red_count=1, blue_count=2),
        missile=MissileConfig(boost_duration_s=0.2),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=True)
    env.reset(seed=28, style="many_to_many")
    first = {
        "target_indices": np.array([0], dtype=np.int64),
        "guidance_bias": np.array([[0.1, -0.1]], dtype=np.float64),
    }
    second = {
        "target_indices": np.array([1], dtype=np.int64),
        "guidance_bias": np.array([[0.4, -0.4]], dtype=np.float64),
    }

    step0 = env.step(red_action=first)
    step1 = env.step(red_action=second)
    step2 = env.step(red_action=second)
    env.step(red_action=second)
    step4 = env.step(red_action=second)

    assert step0.info["assignment_updated"] is True
    assert step0.info["bias_updated"] is True
    assert step1.info["assignment_updated"] is False
    assert step1.info["bias_updated"] is False
    assert step2.info["assignment_updated"] is False
    assert step2.info["bias_updated"] is True
    assert env.replay_layer[2].action.red.target_indices.tolist() == [0]
    assert np.allclose(env.replay_layer[2].action.red.guidance_bias, second["guidance_bias"])
    assert step4.info["assignment_updated"] is True
    assert step4.info["bias_updated"] is True
    assert env.replay_layer[4].action.red.target_indices.tolist() == [1]


def test_rollout_and_mappo_minimum_update_on_accelerator(
    network_device: torch.device,
) -> None:
    torch.manual_seed(29)
    config = _model_config()
    environment = RedBlueEngagementEnv(
        _environment_config(max_steps=6),
        device=network_device,
        record_replay=False,
    )
    assignment_actor, execution_actor, assignment_critic, execution_critic = _network_bundle(
        config,
        network_device,
    )
    batch, stats = collect_rollout(
        environment,
        assignment_actor,
        execution_actor,
        assignment_critic,
        execution_critic,
        steps=2,
        seed=31,
        style="many_to_one",
        red_count=2,
        blue_count=1,
    )

    assert stats.steps == 2
    assert batch.assignment_actor_inputs.self_state.shape == (2, 1, 2, 13)
    assert batch.execution_actor_inputs.assigned_target.shape == (4, 1, 2, 1, 17)
    assert batch.bias_matrices.shape == (4, 1, 2, 2)
    assert batch.old_assignment_log_prob.shape == (2, 1)
    assert batch.old_execution_log_prob.shape == (4, 1, 2)
    assert batch.rewards_high.shape == (2, 1)
    assert batch.rewards_low.shape == (4, 1, 2)
    assert batch.dones_high.shape == (2, 1)
    assert batch.dones_low.shape == (4, 1, 2)
    assert batch.assignment_critic_inputs.global_red.shape == (3, 1, 2, 15)
    assert batch.execution_critic_inputs.applied_bias.shape == (5, 1, 2, 2)
    assert batch.rewards_high.device.type == "cpu"
    assert batch.rewards_high.is_pinned() == (network_device.type == "cuda")
    assert batch.assignment_actor_inputs.self_state.device.type == "cpu"

    trainer = MAPPOTrainer(
        assignment_actor,
        execution_actor,
        assignment_critic,
        execution_critic,
        PPOConfig(
            epochs=1,
            critic_updates_per_actor=1,
            learning_rate=1.0e-3,
            sequence_length=1,
        ),
    )
    metrics = trainer.update(batch)
    assert metrics["actor_updates"] == 1.0
    assert metrics["critic_updates"] == 1.0
    assert all(math.isfinite(value) for value in metrics.values())
    assert trainer.state_dict()["update_step"] == 1


def test_eight_cpu_envs_use_one_cuda_batch_and_update_all_networks(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260716)
    env_config = _environment_config(red_count=2, blue_count=1, max_steps=4)
    envs = [
        RedBlueEngagementEnv(env_config, device="cpu", record_replay=False)
        for _ in range(8)
    ]
    networks = _network_bundle(_model_config(), network_device)
    batch, stats = collect_parallel_rollout(
        envs,
        *networks,
        steps=1,
        seed=101,
        style="many_to_one",
        red_count=2,
        blue_count=1,
    )
    assert stats.final_info["parallel_env_count"] == 8
    assert batch.rewards_high.shape == (1, 8)
    assert batch.rewards_low.shape == (2, 8, 2)
    assert batch.assignment_actor_inputs.self_state.shape == (1, 8, 2, 13)
    assert batch.execution_actor_inputs.self_state.shape == (2, 8, 2, 20)
    assert batch.rewards_high.device.type == "cpu"
    assert batch.rewards_high.is_pinned() == (network_device.type == "cuda")
    assert batch.execution_actor_inputs.self_state.device.type == "cpu"
    before = [
        [parameter.detach().clone() for parameter in network.parameters()]
        for network in networks
    ]
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(epochs=1, critic_updates_per_actor=1, learning_rate=1.0e-3),
    )
    metrics = trainer.update(batch)
    assert all(math.isfinite(value) for value in metrics.values())
    for network, original_parameters in zip(networks, before):
        assert any(
            not torch.equal(parameter.detach(), original)
            for parameter, original in zip(network.parameters(), original_parameters)
        )


def test_staged_updates_freeze_the_opposite_policy_level(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260718)
    environment = RedBlueEngagementEnv(
        _environment_config(max_steps=6),
        device=network_device,
        record_replay=False,
    )
    networks = _network_bundle(_model_config(), network_device)
    batch, _ = collect_rollout(
        environment,
        *networks,
        steps=2,
        seed=83,
        style="many_to_one",
        red_count=2,
        blue_count=1,
        assignment_mode="capacity_aware",
    )
    assert environment.policy_status()["network_call_counts"].get("assignment_actor", 0) == 0
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(
            epochs=1,
            critic_updates_per_actor=1,
            actor_learning_rate=1.0e-3,
            critic_learning_rate=1.0e-3,
            sequence_length=1,
        ),
    )
    before_low = [
        [parameter.detach().clone() for parameter in network.parameters()]
        for network in networks
    ]
    low_metrics = trainer.update(batch, mode="low_only")
    for network_index in (0, 2):
        assert all(
            torch.equal(parameter.detach(), original)
            for parameter, original in zip(networks[network_index].parameters(), before_low[network_index])
        )
    for network_index in (1, 3):
        assert any(
            not torch.equal(parameter.detach(), original)
            for parameter, original in zip(networks[network_index].parameters(), before_low[network_index])
        )
    assert low_metrics["assignment_actor_updates"] == 0.0
    assert low_metrics["execution_actor_updates"] == 1.0

    before_high = [
        [parameter.detach().clone() for parameter in network.parameters()]
        for network in networks
    ]
    execution_forward_calls = 0

    def count_execution_forward(*_args) -> None:
        nonlocal execution_forward_calls
        execution_forward_calls += 1

    execution_actor_hook = networks[1].register_forward_hook(count_execution_forward)
    execution_critic_hook = networks[3].register_forward_hook(count_execution_forward)
    high_metrics = trainer.update(batch, mode="high_only")
    execution_actor_hook.remove()
    execution_critic_hook.remove()
    for network_index in (1, 3):
        assert all(
            torch.equal(parameter.detach(), original)
            for parameter, original in zip(networks[network_index].parameters(), before_high[network_index])
        )
    for network_index in (0, 2):
        assert any(
            not torch.equal(parameter.detach(), original)
            for parameter, original in zip(networks[network_index].parameters(), before_high[network_index])
        )
    assert high_metrics["assignment_actor_updates"] == 1.0
    assert high_metrics["execution_actor_updates"] == 0.0
    assert high_metrics["execution_critic_updates"] == 0.0
    assert execution_forward_calls == 0

    before_effort = [
        [parameter.detach().clone() for parameter in network.parameters()]
        for network in networks
    ]
    effort_metrics = trainer.update(batch, mode="effort_finetune")
    for network_index in (0, 2, 3):
        assert all(
            torch.equal(parameter.detach(), original)
            for parameter, original in zip(
                networks[network_index].parameters(),
                before_effort[network_index],
            )
        )
    assert any(
        not torch.equal(parameter.detach(), original)
        for parameter, original in zip(networks[1].parameters(), before_effort[1])
    )
    assert effort_metrics["execution_actor_updates"] == 1.0
    assert effort_metrics["execution_critic_updates"] == 0.0


def test_execution_learning_scale_preserves_normalized_advantage_and_scales_return(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260723)
    environment = RedBlueEngagementEnv(
        _environment_config(max_steps=6),
        device=network_device,
        record_replay=False,
    )
    networks = _network_bundle(_model_config(), network_device)
    batch, _ = collect_rollout(
        environment,
        *networks,
        steps=2,
        seed=20260723,
        style="many_to_one",
        red_count=2,
        blue_count=1,
        deterministic=True,
        assignment_mode="capacity_aware",
    )
    for parameter in networks[3].parameters():
        parameter.data.zero_()
    active = (
        batch.episode_active_low.bool().unsqueeze(-1)
        & batch.execution_actor_inputs.agent_mask.bool()
        & batch.execution_actor_inputs.target_mask.any(dim=-1)
    )
    batch.rewards_low.zero_()
    batch.rewards_low[active] = torch.linspace(
        64.0,
        512.0,
        int(active.sum()),
        device=batch.rewards_low.device,
    )
    training_batch = batch.training_view(
        network_device,
        assignment=True,
        execution=True,
    )
    unscaled = MAPPOTrainer(
        *networks,
        PPOConfig(epochs=1, critic_updates_per_actor=1),
    )._advantages_and_returns(training_batch)
    scaled = MAPPOTrainer(
        *networks,
        PPOConfig(
            epochs=1,
            critic_updates_per_actor=1,
            execution_reward_learning_scale=1.0 / 512.0,
        ),
    )._advantages_and_returns(training_batch)

    torch.testing.assert_close(
        scaled.execution_return,
        unscaled.execution_return / 512.0,
    )
    torch.testing.assert_close(
        scaled.execution_raw_advantage,
        unscaled.execution_raw_advantage / 512.0,
    )
    torch.testing.assert_close(
        scaled.execution_advantage,
        unscaled.execution_advantage,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert batch.rewards_low[active].max() == 512.0


def test_assignment_learning_scale_preserves_normalized_advantage_and_scales_return(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260811)
    environment = RedBlueEngagementEnv(
        _environment_config(max_steps=6),
        device=network_device,
        record_replay=False,
    )
    networks = _network_bundle(_model_config(), network_device)
    batch, _ = collect_rollout(
        environment,
        *networks,
        steps=2,
        seed=20260811,
        style="many_to_one",
        red_count=2,
        blue_count=1,
        deterministic=True,
    )
    for parameter in networks[2].parameters():
        parameter.data.zero_()
    active = batch.episode_active_high.bool()
    batch.rewards_high.zero_()
    batch.rewards_high[active] = torch.linspace(
        64.0,
        512.0,
        int(active.sum()),
        device=batch.rewards_high.device,
    )
    training_batch = batch.training_view(
        network_device,
        assignment=True,
        execution=True,
    )
    unscaled = MAPPOTrainer(
        *networks,
        PPOConfig(epochs=1, critic_updates_per_actor=1),
    )._advantages_and_returns(training_batch)
    scaled = MAPPOTrainer(
        *networks,
        PPOConfig(
            epochs=1,
            critic_updates_per_actor=1,
            assignment_reward_learning_scale=1.0 / 512.0,
        ),
    )._advantages_and_returns(training_batch)

    torch.testing.assert_close(
        scaled.assignment_return,
        unscaled.assignment_return / 512.0,
    )
    torch.testing.assert_close(
        scaled.assignment_raw_advantage,
        unscaled.assignment_raw_advantage / 512.0,
    )
    torch.testing.assert_close(
        scaled.assignment_advantage,
        unscaled.assignment_advantage,
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    assert batch.rewards_high[active].max() == 512.0


def test_assignment_learning_scale_reduces_critic_preclip_gradient() -> None:
    torch.manual_seed(20260812)
    environment = RedBlueEngagementEnv(
        _environment_config(max_steps=6),
        device="cpu",
        record_replay=False,
    )
    networks = _network_bundle(_model_config(), torch.device("cpu"))
    batch, _ = collect_rollout(
        environment,
        *networks,
        steps=2,
        seed=20260812,
        style="many_to_one",
        red_count=2,
        blue_count=1,
        deterministic=True,
    )
    active = batch.episode_active_high.bool()
    batch.rewards_high.zero_()
    batch.rewards_high[active] = 512.0
    for parameter in networks[2].parameters():
        parameter.data.zero_()
    unscaled_networks = copy.deepcopy(networks)
    scaled_networks = copy.deepcopy(networks)

    unscaled_metrics = MAPPOTrainer(
        *unscaled_networks,
        PPOConfig(epochs=1, critic_updates_per_actor=1),
    ).update(batch, mode="high_only")
    scaled_metrics = MAPPOTrainer(
        *scaled_networks,
        PPOConfig(
            epochs=1,
            critic_updates_per_actor=1,
            assignment_reward_learning_scale=1.0 / 512.0,
        ),
    ).update(batch, mode="high_only")

    unscaled_grad = unscaled_metrics["assignment_critic_grad_norm_preclip"]
    scaled_grad = scaled_metrics["assignment_critic_grad_norm_preclip"]
    assert unscaled_grad > 100.0
    assert scaled_grad == pytest.approx(unscaled_grad / 512.0, rel=1.0e-5)
    assert math.isfinite(scaled_metrics["assignment_explained_variance"])


def test_low_critic_only_updates_only_execution_critic(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260724)
    environment = RedBlueEngagementEnv(
        _environment_config(max_steps=6),
        device=network_device,
        record_replay=False,
    )
    networks = _network_bundle(_model_config(), network_device)
    batch, _ = collect_rollout(
        environment,
        *networks,
        steps=2,
        seed=20260724,
        style="many_to_one",
        red_count=2,
        blue_count=1,
        assignment_mode="capacity_aware",
    )
    before = [copy.deepcopy(network.state_dict()) for network in networks]
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(
            epochs=1,
            critic_updates_per_actor=1,
            execution_reward_learning_scale=1.0 / 512.0,
            execution_value_loss="huber",
        ),
    )
    metrics = trainer.update(batch, mode="low_critic_only")

    for network_index in (0, 1, 2):
        assert all(
            torch.equal(value, before[network_index][name])
            for name, value in networks[network_index].state_dict().items()
        )
    assert any(
        not torch.equal(value, before[3][name])
        for name, value in networks[3].state_dict().items()
    )
    assert metrics["execution_actor_steps_attempted"] == 0.0
    assert metrics["execution_actor_steps_accepted"] == 0.0
    assert metrics["execution_critic_updates"] == 1.0


def _assert_nested_state_equal(actual, expected) -> None:
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_state_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_nested_state_equal(actual_item, expected_item)
    else:
        assert actual == expected


def test_execution_post_step_kl_rollback_restores_actor_and_adam(
    network_device: torch.device,
) -> None:
    torch.manual_seed(20260725)
    environment = RedBlueEngagementEnv(
        _environment_config(max_steps=6),
        device=network_device,
        record_replay=False,
    )
    networks = _network_bundle(_model_config(), network_device)
    batch, _ = collect_rollout(
        environment,
        *networks,
        steps=2,
        seed=20260725,
        style="many_to_one",
        red_count=2,
        blue_count=1,
        assignment_mode="capacity_aware",
    )
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(
            epochs=1,
            critic_updates_per_actor=1,
            execution_actor_learning_rate=0.5,
            execution_reward_learning_scale=1.0 / 512.0,
            execution_value_loss="huber",
            execution_post_step_kl_rollback=True,
            execution_post_step_kl_limit=1.0e-12,
        ),
    )
    actor_before = copy.deepcopy(networks[1].state_dict())
    optimizer_before = copy.deepcopy(trainer.execution_actor_optimizer.state_dict())
    critic_before = copy.deepcopy(networks[3].state_dict())

    metrics = trainer.update(batch, mode="low_only")

    _assert_nested_state_equal(networks[1].state_dict(), actor_before)
    _assert_nested_state_equal(
        trainer.execution_actor_optimizer.state_dict(), optimizer_before
    )
    assert any(
        not torch.equal(value, critic_before[name])
        for name, value in networks[3].state_dict().items()
    )
    assert metrics["execution_actor_steps_attempted"] == 1.0
    assert metrics["execution_actor_steps_accepted"] == 0.0
    assert metrics["execution_actor_steps_rejected"] == 1.0
    assert metrics["execution_kl_rollback_triggered"] == 1.0
    assert metrics["execution_post_step_kl_last"] > 1.0e-12


def test_rollout_uses_one_to_ten_to_two_hundred_schedule(
    network_device: torch.device,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(20260715)
    environment = RedBlueEngagementEnv(
        EnvironmentConfig(
            time_step_s=0.005,
            bias_update_interval_s=0.1,
            assignment_update_interval_s=1.0,
            max_steps=400,
            scenario=ScenarioConfig(red_count=4, blue_count=2),
            missile=MissileConfig(boost_duration_s=1.0),
        ),
        device=network_device,
        record_replay=True,
    )
    networks = _network_bundle(_model_config(), network_device)
    calls = {"assignment": 0, "execution": 0}
    observation_calls = {"full": 0, "execution_only": 0}
    original_full_observe = environment.observation_layer.observe
    original_execution_observe = environment.observation_layer.observe_execution

    def counted_full_observe(*args, **kwargs):
        observation_calls["full"] += 1
        return original_full_observe(*args, **kwargs)

    def counted_execution_observe(*args, **kwargs):
        observation_calls["execution_only"] += 1
        return original_execution_observe(*args, **kwargs)

    monkeypatch.setattr(environment.observation_layer, "observe", counted_full_observe)
    monkeypatch.setattr(environment.observation_layer, "observe_execution", counted_execution_observe)
    assignment_hook = networks[0].register_forward_hook(
        lambda *_: calls.__setitem__("assignment", calls["assignment"] + 1)
    )
    execution_hook = networks[1].register_forward_hook(
        lambda *_: calls.__setitem__("execution", calls["execution"] + 1)
    )

    batch, stats = collect_rollout(
        environment,
        *networks,
        steps=1,
        seed=71,
        style="many_to_many",
        red_count=4,
        blue_count=2,
    )
    assignment_hook.remove()
    execution_hook.remove()

    assert stats.steps == 1
    assert calls == {"assignment": 1, "execution": 10}
    assert observation_calls == {"full": 2, "execution_only": 9}
    assert environment.state is not None
    assert environment.state.step_count == 400
    assert environment.state.time_s == 2.0
    assert environment.policy_time_s == 1.0
    assert len(environment.replay_layer) == 200
    assert torch.count_nonzero(batch.assignment_actor_inputs.hidden[0]) == 0
    assert torch.count_nonzero(batch.execution_actor_inputs.hidden[0]) == 0
    assert torch.count_nonzero(batch.execution_critic_inputs.hidden[0]) == 0
    status = environment.policy_status()
    assert status["first_network_call"]["assignment_actor"] == {
        "time_s": 1.0,
        "step_count": 200,
    }
    assert status["first_network_call"]["execution_actor"] == {
        "time_s": 1.0,
        "step_count": 200,
    }
    assert batch.assignment_actions.target.shape[0] == 1
    assert batch.bias_matrices.shape[0] == 10
    assert batch.assignment_critic_inputs.global_red.shape[0] == 2
    assert batch.execution_critic_inputs.global_red.shape[0] == 11
    expected_targets = (
        batch.assignment_actions.target[0, 0].detach().cpu().numpy().astype(np.int64) - 1
    )
    for low_step in range(10):
        expected_bias = batch.bias_matrices[low_step, 0].detach().cpu().numpy()
        for physics_step in range(20):
            transition = environment.replay_layer[low_step * 20 + physics_step]
            assert np.array_equal(transition.action.red.target_indices, expected_targets)
            assert np.allclose(transition.action.red.guidance_bias, expected_bias)

    parameters_before = [
        [parameter.detach().clone() for parameter in network.parameters()]
        for network in networks
    ]
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(epochs=1, critic_updates_per_actor=1, learning_rate=1.0e-3),
    )
    trainer.update(batch)
    for network, before in zip(networks, parameters_before):
        assert any(
            not torch.equal(parameter.detach(), original)
            for parameter, original in zip(network.parameters(), before)
        )


def test_schema14_checkpoint_save_restore_and_validation(
    tmp_path,
    network_device: torch.device,
) -> None:
    torch.manual_seed(37)
    model_config = _model_config()
    ppo_config = PPOConfig(epochs=1, critic_updates_per_actor=1)
    env_config = _environment_config(max_steps=4)
    assignment_actor, execution_actor, assignment_critic, execution_critic = _network_bundle(
        model_config,
        network_device,
    )
    trainer = MAPPOTrainer(
        assignment_actor,
        execution_actor,
        assignment_critic,
        execution_critic,
        ppo_config,
    )
    checkpoint_path = tmp_path / "schema14.pt"
    _save_checkpoint(
        checkpoint_path,
        assignment_actor,
        execution_actor,
        assignment_critic,
        execution_critic,
        trainer,
        model_config,
        ppo_config,
        env_config,
        {"completed_iterations": 3},
    )
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    assert checkpoint["schema_version"] == 14
    assert checkpoint["blue_policy"] == "rule"
    assert checkpoint["blue_evasion_config"]["detection_range_m"] == 60000.0
    assert checkpoint["model_config"]["max_missiles_per_target"] == 4
    assert checkpoint["env_config"]["scenario"]["max_missiles_per_target"] == 4
    assert checkpoint["env_config"]["missile"]["max_guidance_time_s"] == 180.0
    assert {
        "assignment_actor",
        "execution_actor",
        "assignment_critic",
        "execution_critic",
    } <= checkpoint.keys()

    restored_networks = _network_bundle(model_config, network_device)
    restored_trainer = MAPPOTrainer(*restored_networks, ppo_config)
    restore_info = _restore_checkpoint(
        checkpoint,
        *restored_networks,
        restored_trainer,
        network_device,
    )
    assert restore_info["mode"] == "full"
    assert restore_info["completed_iterations"] == 3
    for original, restored in zip(
        (assignment_actor, execution_actor, assignment_critic, execution_critic),
        restored_networks,
    ):
        for name, expected in original.state_dict().items():
            assert torch.equal(expected, restored.state_dict()[name])

    loaded_assignment, loaded_execution = _load_policy_actors(
        checkpoint_path,
        network_device,
    )
    trajectory = run_cluster_scenario(
        env_config,
        seed=41,
        duration_s=3 * env_config.time_step_s,
        trajectory_sample_interval_s=env_config.time_step_s,
        assignment_actor=loaded_assignment,
        execution_actor=loaded_execution,
    )
    assert trajectory.assignment_source == "schema14_actor"
    assert trajectory.initial_target_indices.shape == (2,)
    assert trajectory.guidance_bias_matrices.shape == (2, 2, 2)
    assert trajectory.pn_load_body_g.shape == (2, 2, 3)
    assert trajectory.bias_load_body_g.shape == (2, 2, 3)
    assert trajectory.gravity_load_body_g.shape == (2, 2, 3)
    assert trajectory.final_load_body_g.shape == (2, 2, 3)

    schema12_checkpoint = copy.deepcopy(checkpoint)
    schema12_checkpoint["schema_version"] = 12
    schema12_checkpoint["ppo_config"].pop("execution_advantage_normalization")
    schema12_checkpoint["ppo_config"].pop("execution_actor_loss_weighting")
    schema12_path = tmp_path / "schema12_legacy.pt"
    torch.save(schema12_checkpoint, schema12_path)
    schema12_assignment, _ = _load_policy_actors(schema12_path, network_device)
    assert schema12_assignment.checkpoint_schema_version == 12
    _, schema12_ppo, _ = _configs_from_checkpoint(
        schema12_checkpoint,
        SwarmModelConfig(),
        PPOConfig(),
        EnvironmentConfig(),
    )
    assert schema12_ppo.execution_advantage_normalization == "global"
    assert schema12_ppo.execution_actor_loss_weighting == "active_step"

    legacy_checkpoint = copy.deepcopy(checkpoint)
    legacy_checkpoint["schema_version"] = 11
    legacy_checkpoint["model_config"].pop("execution_action_distribution")
    legacy_checkpoint["model_config"].pop("critic_value_head_mode")
    legacy_checkpoint["env_config"]["reward"].pop("low_time_credit_mode")
    legacy_checkpoint["env_config"]["reward"].pop("low_time_weight")
    legacy_checkpoint["env_config"]["reward"].pop("low_option_boundary_potential")
    legacy_path = tmp_path / "schema11_legacy.pt"
    torch.save(legacy_checkpoint, legacy_path)
    legacy_assignment, legacy_execution = _load_policy_actors(
        legacy_path,
        network_device,
    )
    assert legacy_assignment.config.execution_action_distribution == "tanh_box"
    assert legacy_execution.config.critic_value_head_mode == "latent_sum"
    assert legacy_assignment.checkpoint_schema_version == 11

    with pytest.raises(ValueError, match="incompatible with supported schemas"):
        _configs_from_checkpoint(
            {"schema_version": 9},
            SwarmModelConfig(),
            PPOConfig(),
            EnvironmentConfig(),
        )

    metrics_path = tmp_path / "validation.json"
    csv_path = tmp_path / "validation.csv"
    validation_args = build_validation_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            str(network_device),
            "--trials",
            "1",
            "--rollout-steps",
            "1",
            "--red-counts",
            "2",
            "--blue-counts",
            "1",
            "--styles",
            "many_to_one",
            "--metrics-path",
            str(metrics_path),
            "--trials-csv",
            str(csv_path),
        ]
    )
    assert validate(validation_args) == 0
    validation_data = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert validation_data["model_config"]["d_bias"] == 2
    assert validation_data["schedule"]["max_physics_steps"] == 2
    assert len(validation_data["trials"]) == 1

    full_horizon_metrics = tmp_path / "validation_full_horizon.json"
    full_horizon_args = build_validation_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            str(network_device),
            "--trials",
            "1",
            "--red-counts",
            "2",
            "--blue-counts",
            "1",
            "--metrics-path",
            str(full_horizon_metrics),
            "--trials-csv",
            "",
        ]
    )
    assert validate(full_horizon_args) == 0
    full_horizon_data = json.loads(full_horizon_metrics.read_text(encoding="utf-8"))
    assert full_horizon_data["schedule"]["max_physics_steps"] == env_config.policy_horizon_steps

    mismatched_capacity_args = build_validation_parser().parse_args(
        [
            "--checkpoint",
            str(checkpoint_path),
            "--device",
            str(network_device),
            "--trials",
            "1",
            "--max-missiles-per-target",
            "3",
        ]
    )
    with pytest.raises(ValueError, match="model and environment max_missiles_per_target must match"):
        validate(mismatched_capacity_args)
    assert "ineffective_loss_rate_mean" in validation_data["summary"]
    assert "control_effort_mean" in validation_data["summary"]
    assert validation_data["blue_policy"] == "rule"
    assert csv_path.exists()
