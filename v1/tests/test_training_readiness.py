from __future__ import annotations

import copy
import hashlib
import json
import math

import numpy as np
import pytest
import torch
import red_swarm_policy.train_env as train_env_module

from red_swarm_policy import (
    AircraftConfig,
    AssignmentActorInputs,
    EnvironmentConfig,
    MAPPOTrainer,
    MissileConfig,
    PPOConfig,
    RedBlueEngagementEnv,
    ScenarioConfig,
    SwarmModelConfig,
    TargetAssignmentActor,
    ThreeDoFState,
    collect_parallel_rollout,
)
from red_swarm_policy.env import seeker_target_visible
from red_swarm_policy.policy.critic import OverloadBiasCritic, TargetAssignmentCritic
from red_swarm_policy.policy.actor import OverloadBiasActor
from red_swarm_policy.training.env_pool import EnvironmentWorkerError, ProcessEnvironmentPool
from red_swarm_policy.training.rollout import HierarchicalPolicyRuntime
from red_swarm_policy.train_env import (
    _configure_modules_for_update,
    _load_torch_checkpoint,
    _restore_assignment_policy_from_checkpoint,
    _restore_execution_policy_from_checkpoint,
    _restored_best_state,
    _seed_validation_policy_rng,
    _step_assignment_validation_scheduler,
    _step_execution_validation_scheduler,
    _stratified_red_counts,
    _validate_stage1_quality_gate,
    _validation_plan,
    build_parser as build_training_parser,
    train,
)
from red_swarm_policy.validate_stage1_low_checkpoint import (
    _build_stage1_quality_gate,
    _compare_to_pn_baseline,
)


def _state(angle_deg: float, *, locked: bool, target_index: int = 0) -> tuple[ThreeDoFState, ThreeDoFState]:
    angle = math.radians(angle_deg)
    red = ThreeDoFState(
        position_m=np.array([0.0, 10000.0, 0.0]),
        velocity_mps=np.array([1000.0, 0.0, 0.0]),
        mass_kg=120.0,
        age_s=7.0,
        current_target_index=target_index,
        seeker_locked=locked,
    )
    blue = ThreeDoFState(
        position_m=np.array([10000.0 * math.cos(angle), 10000.0, 10000.0 * math.sin(angle)]),
        velocity_mps=np.array([300.0, 0.0, 0.0]),
        mass_kg=1.0,
    )
    return red, blue


def _assignment_inputs(red_count: int, target_count: int) -> AssignmentActorInputs:
    return AssignmentActorInputs(
        self_state=torch.zeros(1, red_count, 13),
        friend_entities=torch.zeros(1, red_count, red_count, 11),
        friend_mask=torch.zeros(1, red_count, red_count, dtype=torch.bool),
        target_entities=torch.zeros(1, red_count, target_count, 8),
        pair_state=torch.zeros(1, red_count, target_count, 11),
        current_assignment=torch.zeros(1, red_count, target_count),
        target_mask=torch.ones(1, red_count, target_count, dtype=torch.bool),
        environment_context=torch.zeros(1, red_count, 5),
        target_assignment_counts=torch.zeros(1, red_count, target_count),
        target_entity_mask=torch.ones(1, red_count, target_count, dtype=torch.bool),
        agent_mask=torch.ones(1, red_count, dtype=torch.bool),
    )


def _network_bundle(config: SwarmModelConfig):
    return (
        TargetAssignmentActor(config),
        OverloadBiasActor(config),
        TargetAssignmentCritic(config),
        OverloadBiasCritic(config),
    )


def _assert_json_numbers_finite(value) -> None:
    if isinstance(value, dict):
        for item in value.values():
            _assert_json_numbers_finite(item)
    elif isinstance(value, list):
        for item in value:
            _assert_json_numbers_finite(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        assert math.isfinite(value)


def test_validation_runtime_can_sample_assignment_with_deterministic_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.05,
        assignment_update_interval_s=0.1,
        max_steps=4,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(red_count=1, blue_count=1),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    observation = env.reset(
        seed=20260814,
        style="one_to_one",
        start_mode="post_boost",
    )
    assignment_actor, execution_actor, _, _ = _network_bundle(
        SwarmModelConfig(d_model=16, num_heads=4)
    )
    assignment_flags: list[bool] = []
    execution_flags: list[bool] = []
    original_assignment_forward = assignment_actor.forward
    original_execution_forward = execution_actor.forward

    def assignment_forward(inputs, deterministic: bool = False):
        assignment_flags.append(deterministic)
        return original_assignment_forward(inputs, deterministic=deterministic)

    def execution_forward(inputs, deterministic: bool = False):
        execution_flags.append(deterministic)
        return original_execution_forward(inputs, deterministic=deterministic)

    monkeypatch.setattr(assignment_actor, "forward", assignment_forward)
    monkeypatch.setattr(execution_actor, "forward", execution_forward)
    runtime = HierarchicalPolicyRuntime(
        env,
        assignment_actor,
        execution_actor,
        deterministic=True,
        assignment_mode="actor",
        assignment_deterministic=False,
        execution_deterministic=True,
    )
    runtime.reset(observation)
    runtime.action(observation)

    assert assignment_flags == [False]
    assert execution_flags == [True]


def test_validation_policy_seed_is_reproducible_after_rng_changes() -> None:
    _seed_validation_policy_rng(20265001, torch.device("cpu"))
    first = torch.rand(8)
    torch.manual_seed(7)
    _seed_validation_policy_rng(20265001, torch.device("cpu"))
    repeated = torch.rand(8)
    _seed_validation_policy_rng(20265002, torch.device("cpu"))
    different = torch.rand(8)

    assert torch.equal(first, repeated)
    assert not torch.equal(first, different)


def test_high_only_explicitly_freezes_low_level_modules() -> None:
    networks = _network_bundle(SwarmModelConfig(d_model=16, num_heads=4))

    enabled = _configure_modules_for_update("high_only", *networks)

    assert enabled == {
        "assignment_actor": True,
        "execution_actor": False,
        "assignment_critic": True,
        "execution_critic": False,
    }
    assert networks[0].training and networks[2].training
    assert not networks[1].training and not networks[3].training
    assert all(parameter.requires_grad for parameter in networks[0].parameters())
    assert all(parameter.requires_grad for parameter in networks[2].parameters())
    assert not any(parameter.requires_grad for parameter in networks[1].parameters())
    assert not any(parameter.requires_grad for parameter in networks[3].parameters())


def test_parallel_rollout_supports_independent_blue_counts() -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.05,
        assignment_update_interval_s=0.1,
        max_steps=4,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(red_count=3, blue_count=3, max_missiles_per_target=4),
    )
    envs = [RedBlueEngagementEnv(config, device="cpu", record_replay=False) for _ in range(3)]

    batch, _ = collect_parallel_rollout(
        envs,
        *_network_bundle(SwarmModelConfig(d_model=16, num_heads=4)),
        steps=1,
        seed=53,
        red_count=3,
        blue_counts=[1, 2, 3],
        deterministic=True,
    )

    assert [len(env.state.blue) for env in envs if env.state is not None] == [1, 2, 3]
    assert batch.assignment_actor_inputs.target_mask.shape[-1] == 4
    assert batch.assignment_critic_inputs.blue_mask.shape[-1] == 3
    assert not batch.assignment_actor_inputs.target_mask[0, 0, :, 2:].any()
    assert not batch.assignment_critic_inputs.blue_mask[0, 0, 1:].any()


def test_process_rollout_supports_independent_blue_counts() -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.05,
        assignment_update_interval_s=0.1,
        max_steps=4,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(red_count=3, blue_count=3, max_missiles_per_target=4),
    )
    envs = [RedBlueEngagementEnv(config, device="cpu", record_replay=False) for _ in range(3)]
    pool = ProcessEnvironmentPool(envs, native_threads=1, timeout_s=30.0)
    try:
        batch, _ = collect_parallel_rollout(
            envs,
            *_network_bundle(SwarmModelConfig(d_model=16, num_heads=4)),
            steps=1,
            seed=54,
            red_count=3,
            blue_counts=[1, 2, 3],
            deterministic=True,
            env_pool=pool,
        )
    finally:
        pool.close()

    assert [len(env.state.blue) for env in envs if env.state is not None] == [1, 2, 3]
    assert batch.assignment_actor_inputs.target_mask.shape[-1] == 4
    assert batch.assignment_critic_inputs.blue_mask.shape[-1] == 3


@pytest.mark.parametrize("backend", ["thread", "process"])
def test_parallel_rollout_supports_mixed_red_counts_with_inactive_padding(
    backend: str,
) -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.05,
        assignment_update_interval_s=0.1,
        max_steps=4,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(red_count=4, blue_count=2, max_missiles_per_target=4),
    )
    envs = [RedBlueEngagementEnv(config, device="cpu", record_replay=False) for _ in range(4)]
    pool = (
        ProcessEnvironmentPool(envs, native_threads=1, timeout_s=30.0)
        if backend == "process"
        else None
    )
    try:
        batch, stats = collect_parallel_rollout(
            envs,
            *_network_bundle(SwarmModelConfig(d_model=16, num_heads=4)),
            steps=1,
            seed=55,
            red_counts=[1, 2, 3, 4],
            blue_counts=[1, 2, 1, 2],
            deterministic=True,
            assignment_mode="capacity_aware",
            env_pool=pool,
        )
    finally:
        if pool is not None:
            pool.close()

    assert batch.scenario_red_counts.tolist() == [1, 2, 3, 4]
    assert batch.scenario_blue_counts.tolist() == [1, 2, 1, 2]
    assert batch.rewards_low.shape[1:] == (4, 4)
    assert batch.assignment_actor_inputs.self_state.shape[2] == 4
    assert batch.execution_actor_inputs.same_target_friends.shape[2:4] == (4, 4)
    for env_index, red_count in enumerate((1, 2, 3, 4)):
        padded = slice(red_count, 4)
        assert not batch.assignment_actor_inputs.agent_mask[:, env_index, padded].any()
        assert not batch.execution_actor_inputs.agent_mask[:, env_index, padded].any()
        assert torch.count_nonzero(batch.assignment_actions.target[:, env_index, padded]) == 0
        assert torch.count_nonzero(batch.bias_matrices[:, env_index, padded]) == 0
        assert torch.count_nonzero(batch.old_execution_log_prob[:, env_index, padded]) == 0
        assert torch.count_nonzero(batch.rewards_low[:, env_index, padded]) == 0
        assert (batch.dones_low[:, env_index, padded] == 1).all()
    assert stats.active_execution_agent_steps > 0
    assert stats.final_info["parallel_env_count"] == 4


def test_stage_best_state_can_be_reset_without_discarding_checkpoint_resume() -> None:
    state = {
        "best_checkpoint_score": [1.0, 0.9, -0.1, -80.0, -0.02],
        "best_checkpoint_metrics": {
            "full_success_rate": 1.0,
            "average_damage_rate": 0.9,
            "ineffective_loss_rate": 0.1,
            "successful_completion_time_s": 80.0,
            "control_effort": 0.02,
        },
        "best_checkpoint_stage": "low_only",
        "completed_iterations": 600,
    }
    score, metrics, stage = _restored_best_state(state, reset=False)
    assert score == (1.0, 0.9, -0.1, -80.0, -0.02)
    assert metrics is not None and metrics["average_damage_rate"] == 0.9
    assert stage == "low_only"
    assert _restored_best_state(state, reset=True) == (None, None, None)
    assert state["completed_iterations"] == 600


def test_blue_aircraft_configuration_is_action_library_only() -> None:
    aircraft = AircraftConfig()

    assert not hasattr(aircraft, "mass_kg")
    assert not hasattr(aircraft, "reference_area_m2")
    assert not hasattr(aircraft, "drag_coefficient_sea_level")
    assert not hasattr(aircraft, "drag_coefficient_altitude_gradient_per_m")


def test_network_visibility_uses_seeker_35_and_60_degree_limits() -> None:
    missile = MissileConfig()

    red, blue = _state(35.0, locked=False)
    assert seeker_target_visible(red, blue, 0, missile, detection_range_m=20000.0)
    red, blue = _state(35.001, locked=False)
    assert not seeker_target_visible(red, blue, 0, missile, detection_range_m=20000.0)

    red, blue = _state(60.0, locked=True)
    assert seeker_target_visible(red, blue, 0, missile, detection_range_m=20000.0)
    red, blue = _state(60.001, locked=True)
    assert not seeker_target_visible(red, blue, 0, missile, detection_range_m=20000.0)

    red, blue = _state(35.001, locked=True, target_index=0)
    assert not seeker_target_visible(red, blue, 1, missile, detection_range_m=20000.0)


def test_assignment_actor_and_environment_enforce_four_missiles_per_target() -> None:
    torch.manual_seed(20260716)
    model = SwarmModelConfig(d_model=16, num_heads=4, max_missiles_per_target=4)
    actor = TargetAssignmentActor(model)
    inputs = _assignment_inputs(red_count=12, target_count=3)

    for _ in range(20):
        output = actor(inputs)
        targets = output.actions.target[0]
        for target_slot in (1, 2):
            assert int((targets == target_slot).sum()) <= 4

    config = EnvironmentConfig(
        scenario=ScenarioConfig(red_count=5, blue_count=1, max_missiles_per_target=4),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    env.reset(seed=20260716)
    with pytest.raises(ValueError, match="at most 4"):
        env.step(
            red_action={
                "target_indices": np.zeros(5, dtype=np.int64),
                "guidance_bias": np.zeros((5, 2), dtype=np.float64),
            }
        )


def test_parallel_rollout_marks_terminated_environment_inactive(monkeypatch: pytest.MonkeyPatch) -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.05,
        assignment_update_interval_s=0.1,
        max_steps=6,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(red_count=2, blue_count=1, max_missiles_per_target=4),
    )
    envs = [RedBlueEngagementEnv(config, device="cpu", record_replay=False) for _ in range(2)]
    original_step = envs[0].step
    calls = 0

    def terminate_first(red_action=None, blue_action=None):
        nonlocal calls
        result = original_step(red_action, blue_action)
        calls += 1
        if calls == 1:
            result.done = True
        return result

    monkeypatch.setattr(envs[0], "step", terminate_first)
    model = SwarmModelConfig(d_model=16, num_heads=4, max_missiles_per_target=4)
    batch, _ = collect_parallel_rollout(
        envs,
        *_network_bundle(model),
        steps=2,
        seed=19,
        red_count=2,
        blue_count=1,
    )

    assert batch.episode_active_high[:, 0].tolist() == [True, False]
    assert batch.episode_active_high[:, 1].tolist() == [True, True]
    assert batch.episode_active_low[0].tolist() == [True, True]
    assert not batch.episode_active_low[1:, 0].any()
    assert torch.count_nonzero(batch.execution_actor_inputs.hidden[1:, 0]) == 0


def test_process_environment_pool_matches_thread_rollout_and_closes() -> None:
    torch.manual_seed(20260716)
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.05,
        assignment_update_interval_s=0.1,
        max_steps=6,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(red_count=2, blue_count=1, max_missiles_per_target=4),
    )
    direct_envs = [
        RedBlueEngagementEnv(config, device="cpu", record_replay=False) for _ in range(2)
    ]
    process_envs = [
        RedBlueEngagementEnv(config, device="cpu", record_replay=False) for _ in range(2)
    ]
    networks = _network_bundle(SwarmModelConfig(d_model=16, num_heads=4))

    direct_batch, direct_stats = collect_parallel_rollout(
        direct_envs,
        *networks,
        steps=1,
        seed=31,
        red_count=2,
        blue_count=1,
        deterministic=True,
    )
    pool = ProcessEnvironmentPool(process_envs, native_threads=1, timeout_s=30.0)
    assert pool.alive_worker_count == 2
    assert [worker.torch_threads for worker in pool.worker_info] == [1, 1]
    try:
        process_batch, process_stats = collect_parallel_rollout(
            process_envs,
            *networks,
            steps=1,
            seed=31,
            red_count=2,
            blue_count=1,
            deterministic=True,
            env_pool=pool,
        )
    finally:
        pool.close()

    assert pool.alive_worker_count == 0
    torch.testing.assert_close(process_batch.rewards_high, direct_batch.rewards_high)
    torch.testing.assert_close(process_batch.rewards_low, direct_batch.rewards_low)
    torch.testing.assert_close(process_batch.dones_high, direct_batch.dones_high)
    torch.testing.assert_close(process_batch.dones_low, direct_batch.dones_low)
    torch.testing.assert_close(
        process_batch.assignment_actions.target,
        direct_batch.assignment_actions.target,
    )
    torch.testing.assert_close(process_batch.bias_matrices, direct_batch.bias_matrices)
    assert process_stats.steps == direct_stats.steps
    assert process_stats.execution_steps == direct_stats.execution_steps
    assert process_stats.done == direct_stats.done
    for process_env, direct_env in zip(process_envs, direct_envs):
        assert process_env.state is not None and direct_env.state is not None
        assert process_env.state.step_count == direct_env.state.step_count
        for process_red, direct_red in zip(process_env.state.red, direct_env.state.red):
            np.testing.assert_allclose(process_red.position_m, direct_red.position_m)
            np.testing.assert_allclose(process_red.velocity_mps, direct_red.velocity_mps)


def test_process_environment_pool_propagates_worker_error() -> None:
    config = EnvironmentConfig(
        time_step_s=0.05,
        bias_update_interval_s=0.05,
        assignment_update_interval_s=0.1,
        max_steps=4,
        missile=MissileConfig(boost_duration_s=0.1),
        scenario=ScenarioConfig(red_count=2, blue_count=1),
    )
    env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    pool = ProcessEnvironmentPool([env], native_threads=1, timeout_s=30.0)
    try:
        with pytest.raises(EnvironmentWorkerError, match="before post-boost"):
            pool.advance(
                [0],
                np.zeros((1, 2), dtype=np.int64),
                np.zeros((1, 2, 2), dtype=np.float64),
                1,
            )
    finally:
        pool.close()
    assert pool.alive_worker_count == 0


def test_training_writes_latest_fixed_validation_best_and_periodic(tmp_path) -> None:
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    metrics = tmp_path / "metrics.json"
    args = build_training_parser().parse_args(
        [
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--parallel-envs",
            "2",
            "--rollout-steps",
            "1",
            "--time-step-s",
            "0.05",
            "--bias-update-interval-s",
            "0.05",
            "--assignment-update-interval-s",
            "0.1",
            "--max-steps",
            "4",
            "--missile-boost-time-s",
            "0.1",
            "--d-model",
            "16",
            "--num-heads",
            "4",
            "--ppo-epochs",
            "1",
            "--critic-updates-per-actor",
            "1",
            "--validation-interval",
            "1",
            "--validation-trials-per-blue-count",
            "1",
            "--checkpoint-interval",
            "1",
            "--latest-checkpoint",
            str(latest),
            "--best-checkpoint",
            str(best),
            "--metrics-path",
            str(metrics),
        ]
    )

    assert train(args) == 0
    periodic = tmp_path / "iteration_000001.pt"
    assert latest.exists() and best.exists() and periodic.exists() and metrics.exists()
    latest_checkpoint = _load_torch_checkpoint(latest)
    best_checkpoint = _load_torch_checkpoint(best)
    assert latest_checkpoint["schema_version"] == 14
    assert latest_checkpoint["training_state"]["completed_iterations"] == 1
    assert len(best_checkpoint["training_state"]["best_checkpoint_score"]) == 5
    assert best_checkpoint["training_state"]["best_checkpoint_metrics"]["trial_count"] == 1.0
    assert best_checkpoint["training_state"]["best_checkpoint_metrics"]["assignment_mode"] == "capacity_aware"
    assert best_checkpoint["training_state"]["best_checkpoint_stage"] == "alternating"
    metrics_data = json.loads(metrics.read_text(encoding="utf-8"))
    assert metrics_data["blue_policy"] == "rule"
    assert metrics_data["blue_evasion_config"]["decision_interval_s"] == 0.1
    assert metrics_data["validation_config"]["trials_per_blue_count"] == 1
    fixed_validation = metrics_data["iterations"][0]["fixed_validation"]
    assert fixed_validation["by_scenario"][0]["red_count"] == 24
    assert fixed_validation["by_scenario"][0]["blue_count"] == 4

    for incompatible_flag in (
        ("--execution-action-distribution", "radial_tanh_disk"),
        ("--critic-value-head-mode", "scalar"),
        ("--low-time-credit-mode", "terminal_active_share"),
        ("--low-option-boundary-potential", "terminal_zero"),
        ("--execution-advantage-normalization", "per_scenario"),
        ("--execution-actor-loss-weighting", "per_scenario"),
    ):
        incompatible_args = build_training_parser().parse_args(
            [
                "--device",
                "cpu",
                "--iterations",
                "0",
                "--resume-checkpoint",
                str(latest),
                incompatible_flag[0],
                incompatible_flag[1],
                "--latest-checkpoint",
                str(tmp_path / "incompatible.pt"),
                "--metrics-path",
                "",
            ]
        )
        with pytest.raises(ValueError, match="full resume cannot override"):
            train(incompatible_args)

    resumed_latest = tmp_path / "resumed_latest.pt"
    resumed_metrics = tmp_path / "resumed_metrics.json"
    resumed_args = build_training_parser().parse_args(
        [
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--parallel-envs",
            "2",
            "--rollout-steps",
            "1",
            "--resume-checkpoint",
            str(latest),
            "--validation-interval",
            "0",
            "--validation-seed-start",
            "7",
            "--validation-trials-per-blue-count",
            "2",
            "--latest-checkpoint",
            str(resumed_latest),
            "--metrics-path",
            str(resumed_metrics),
        ]
    )
    assert train(resumed_args) == 0
    resumed_checkpoint = _load_torch_checkpoint(resumed_latest)
    assert resumed_checkpoint["training_state"]["completed_iterations"] == 2
    assert resumed_checkpoint["training_state"]["best_checkpoint_stage"] == "alternating"
    assert resumed_checkpoint["validation_config"] == latest_checkpoint["validation_config"]
    resumed_metrics_data = json.loads(resumed_metrics.read_text(encoding="utf-8"))
    assert resumed_metrics_data["validation_config"] == latest_checkpoint["validation_config"]

    transitioned_metrics = tmp_path / "transitioned_metrics.json"
    transitioned_args = build_training_parser().parse_args(
        [
            "--device",
            "cpu",
            "--training-mode",
            "alternating",
            "--reset-best-on-resume",
            "--iterations",
            "0",
            "--parallel-envs",
            "2",
            "--parallel-backend",
            "thread",
            "--resume-checkpoint",
            str(latest),
            "--validation-interval",
            "7",
            "--validation-seed-start",
            "8000",
            "--validation-trials-per-blue-count",
            "3",
            "--latest-checkpoint",
            "",
            "--best-checkpoint",
            "",
            "--metrics-path",
            str(transitioned_metrics),
        ]
    )
    assert train(transitioned_args) == 0
    transitioned = json.loads(transitioned_metrics.read_text(encoding="utf-8"))
    assert transitioned["validation_config"]["interval"] == 7
    assert transitioned["validation_config"]["seed_start"] == 8000
    assert transitioned["validation_config"]["trials_per_blue_count"] == 3
    assert transitioned["training_state"]["best_iteration"] is None
    assert transitioned["training_state"]["best_checkpoint_origin"] is None
    assert transitioned["training_state"]["completed_stage_policy_updates"] == 0
    assert transitioned["training_state"]["execution_lr_reductions"] == 0
    assert transitioned["training_state"]["execution_policy_restorations"] == 0


def test_low_only_validation_plan_covers_configured_curriculum_scenarios() -> None:
    scenarios, assignment_mode = _validation_plan("low_only", [1, 2, 3, 4], [1])

    assert assignment_mode == "capacity_aware"
    assert scenarios == [
        ("many_to_one", 1, 1),
        ("many_to_one", 2, 1),
        ("many_to_one", 3, 1),
        ("many_to_one", 4, 1),
    ]
    warmup_scenarios, warmup_assignment_mode = _validation_plan(
        "low_critic_only", [1, 2, 3, 4], [1]
    )
    assert warmup_scenarios == scenarios
    assert warmup_assignment_mode == "capacity_aware"


def test_validation_assignment_mode_override_replaces_plan_default() -> None:
    default_scenarios, default_mode = _validation_plan("high_only", [24], [4, 5, 6])
    assert default_mode == "actor"

    baseline_scenarios, baseline_mode = _validation_plan(
        "high_only", [24], [4, 5, 6], "capacity_aware"
    )
    assert baseline_scenarios == default_scenarios
    assert baseline_mode == "capacity_aware"

    low_scenarios, low_mode = _validation_plan("low_only", [24], [4, 5, 6], "actor")
    assert low_scenarios == [
        ("many_to_many", 24, 4),
        ("many_to_many", 24, 5),
        ("many_to_many", 24, 6),
    ]
    assert low_mode == "actor"
    assert low_scenarios == default_scenarios

    with pytest.raises(ValueError, match="unsupported validation assignment mode"):
        _validation_plan("high_only", [24], [4, 5, 6], "greedy")


def test_validation_only_evaluates_checkpoint_without_training_or_writing(
    tmp_path,
) -> None:
    latest = tmp_path / "latest.pt"
    setup_args = build_training_parser().parse_args(
        [
            "--device",
            "cpu",
            "--iterations",
            "1",
            "--parallel-envs",
            "1",
            "--parallel-backend",
            "thread",
            "--rollout-steps",
            "1",
            "--time-step-s",
            "0.05",
            "--bias-update-interval-s",
            "0.05",
            "--assignment-update-interval-s",
            "0.1",
            "--max-steps",
            "4",
            "--missile-boost-time-s",
            "0.1",
            "--d-model",
            "16",
            "--num-heads",
            "4",
            "--ppo-epochs",
            "1",
            "--critic-updates-per-actor",
            "1",
            "--validation-interval",
            "0",
            "--validation-seed-start",
            "4242",
            "--validation-trials-per-blue-count",
            "5",
            "--checkpoint-interval",
            "0",
            "--latest-checkpoint",
            str(latest),
            "--best-checkpoint",
            "",
            "--checkpoint",
            "",
            "--metrics-path",
            str(tmp_path / "setup_metrics.json"),
        ]
    )
    assert train(setup_args) == 0
    assert latest.exists()
    saved_validation_config = _load_torch_checkpoint(latest)["validation_config"]
    assert saved_validation_config["seed_start"] == 4242
    assert saved_validation_config["trials_per_blue_count"] == 5
    assert saved_validation_config["assignment_mode_override"] is None

    reports = {}
    for assignment_mode in ("actor", "capacity_aware"):
        metrics_path = tmp_path / f"validation_{assignment_mode}.json"
        unexpected_latest = tmp_path / f"unexpected_latest_{assignment_mode}.pt"
        unexpected_best = tmp_path / f"unexpected_best_{assignment_mode}.pt"
        validation_args = build_training_parser().parse_args(
            [
                "--device",
                "cpu",
                "--validation-only",
                "--iterations",
                "0",
                "--resume-checkpoint",
                str(latest),
                "--parallel-envs",
                "1",
                "--parallel-backend",
                "process",
                "--validation-parallel-envs",
                "2",
                "--validation-assignment-mode",
                assignment_mode,
                "--validation-seed-start",
                "7",
                "--validation-trials-per-blue-count",
                "1",
                "--latest-checkpoint",
                str(unexpected_latest),
                "--best-checkpoint",
                str(unexpected_best),
                "--metrics-path",
                str(metrics_path),
            ]
        )
        assert train(validation_args) == 0
        assert not unexpected_latest.exists()
        assert not unexpected_best.exists()
        assert not (tmp_path / "iteration_000001.pt").exists()

        report = json.loads(metrics_path.read_text(encoding="utf-8"))
        reports[assignment_mode] = report
        assert report["event"] == "validation_only"
        assert report["assignment_mode"] == assignment_mode
        assert report["checkpoint_sha256"] == hashlib.sha256(latest.read_bytes()).hexdigest()
        assert report["trial_count"] == 3.0
        assert len(report["checkpoint_selection_score"]) == 5
        assert [entry["blue_count"] for entry in report["by_scenario"]] == [4, 5, 6]
        assert {entry["red_count"] for entry in report["by_scenario"]} == {24}
        # The command line, not the resumed checkpoint, defines the validation set.
        assert report["validation_config"]["seed_start"] == 7
        assert report["validation_config"]["trials_per_blue_count"] == 1
        assert report["validation_config"]["parallel_envs"] == 2
        assert report["validation_config"]["assignment_mode_override"] == assignment_mode
        assert report["validation_config"]["assignment_deterministic"] is True
        assert report["validation_config"]["execution_deterministic"] is True
        assert report["validation_config"]["policy_seed"] is None
        assert report["validation_config"]["validation_only"] is True
        assert report["policy_mode"] == "deterministic"
        assert report["execution_policy_mode"] == "deterministic"
        _assert_json_numbers_finite(report)

    stochastic_metrics_path = tmp_path / "validation_actor_stochastic.json"
    stochastic_args = build_training_parser().parse_args(
        [
            "--device",
            "cpu",
            "--validation-only",
            "--iterations",
            "0",
            "--resume-checkpoint",
            str(latest),
            "--parallel-envs",
            "1",
            "--parallel-backend",
            "process",
            "--validation-parallel-envs",
            "2",
            "--validation-assignment-mode",
            "actor",
            "--validation-assignment-stochastic",
            "--validation-policy-seed",
            "20265001",
            "--validation-seed-start",
            "7",
            "--validation-trials-per-blue-count",
            "1",
            "--metrics-path",
            str(stochastic_metrics_path),
        ]
    )
    assert train(stochastic_args) == 0
    stochastic_report = json.loads(
        stochastic_metrics_path.read_text(encoding="utf-8")
    )
    assert stochastic_report["assignment_mode"] == "actor"
    assert stochastic_report["policy_mode"] == (
        "assignment_stochastic_execution_deterministic"
    )
    assert stochastic_report["assignment_policy_mode"] == "stochastic"
    assert stochastic_report["execution_policy_mode"] == "deterministic"
    assert stochastic_report["validation_policy_seed"] == 20265001
    assert stochastic_report["validation_config"]["assignment_deterministic"] is False
    assert stochastic_report["validation_config"]["execution_deterministic"] is True
    assert stochastic_report["validation_config"]["policy_seed"] == 20265001
    _assert_json_numbers_finite(stochastic_report)

    manifest = json.loads(
        (tmp_path / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["stop_reason"] == "validation_only"
    assert manifest["completed_iterations"] == 1
    assert manifest["validation_only"]["assignment_mode"] == "actor"
    assert manifest["validation_only"]["assignment_policy_mode"] == "stochastic"
    assert manifest["validation_only"]["execution_policy_mode"] == "deterministic"
    assert manifest["validation_only"]["validation_policy_seed"] == 20265001

    for scenario_index, scenario in enumerate(reports["capacity_aware"]["by_scenario"]):
        actor_scenario = reports["actor"]["by_scenario"][scenario_index]
        assert scenario["style"] == actor_scenario["style"]
        assert scenario["trial_count"] == actor_scenario["trial_count"]


def test_validation_only_requires_a_checkpoint_and_zero_iterations() -> None:
    with pytest.raises(SystemExit):
        train_env_module.main(["--validation-only", "--iterations", "0"])
    with pytest.raises(SystemExit):
        train_env_module.main(
            ["--validation-only", "--iterations", "1", "--resume-checkpoint", "x.pt"]
        )

    unchecked = build_training_parser().parse_args(
        ["--device", "cpu", "--validation-only", "--iterations", "0"]
    )
    with pytest.raises(ValueError, match="validation_only requires a resume checkpoint"):
        train(unchecked)


def test_stochastic_assignment_validation_requires_safe_explicit_cli() -> None:
    with pytest.raises(SystemExit):
        train_env_module.main(["--validation-assignment-stochastic"])
    with pytest.raises(SystemExit):
        train_env_module.main(
            [
                "--validation-only",
                "--iterations",
                "0",
                "--resume-checkpoint",
                "missing.pt",
                "--validation-assignment-mode",
                "actor",
                "--validation-assignment-stochastic",
            ]
        )
    with pytest.raises(SystemExit):
        train_env_module.main(
            [
                "--validation-only",
                "--iterations",
                "0",
                "--resume-checkpoint",
                "missing.pt",
                "--validation-assignment-mode",
                "capacity_aware",
                "--validation-assignment-stochastic",
                "--validation-policy-seed",
                "20265001",
            ]
        )
    with pytest.raises(SystemExit):
        train_env_module.main(
            [
                "--validation-policy-seed",
                "20265001",
            ]
        )


def test_stratified_red_count_schedule_is_reproducible_and_balanced() -> None:
    first = _stratified_red_counts([1, 2, 3, 4], 10, 20260703, 0)
    repeated = _stratified_red_counts([1, 2, 3, 4], 10, 20260703, 0)
    assert first == repeated
    assert sorted(first).count(1) == 3
    assert sorted(first).count(2) == 3
    assert sorted(first).count(3) == 2
    assert sorted(first).count(4) == 2

    totals = {count: 0 for count in (1, 2, 3, 4)}
    for update in range(4):
        for count in _stratified_red_counts(
            [1, 2, 3, 4], 10, 20260703, update
        ):
            totals[count] += 1
    assert totals == {1: 10, 2: 10, 3: 10, 4: 10}


def test_validation_scheduler_reduces_execution_lr_and_stops() -> None:
    networks = _network_bundle(SwarmModelConfig(d_model=16, num_heads=4))
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(execution_actor_learning_rate=5.0e-5),
    )
    state = {"no_improvement_validations": 0, "execution_lr_reductions": 0}

    for validation in range(1, 9):
        events, should_stop = _step_execution_validation_scheduler(
            trainer,
            state,
            improved=False,
            lr_patience=2,
            lr_factor=0.5,
            min_actor_lr=5.0e-6,
            early_stop_patience=8,
        )
        if validation == 2:
            assert events[0]["new_lr"] == pytest.approx(2.5e-5)
        if validation == 4:
            assert events[0]["new_lr"] == pytest.approx(1.25e-5)
        if validation == 6:
            assert events[0]["new_lr"] == pytest.approx(6.25e-6)
        if validation == 8:
            assert events[0]["new_lr"] == pytest.approx(5.0e-6)
        assert should_stop is (validation == 8)
    assert state == {
        "no_improvement_validations": 8,
        "execution_lr_plateau_bad_validations": 0,
        "early_stop_bad_validations": 8,
        "execution_lr_reductions": 4,
    }

    events, should_stop = _step_execution_validation_scheduler(
        trainer,
        state,
        improved=True,
        lr_patience=2,
        lr_factor=0.5,
        min_actor_lr=5.0e-6,
        early_stop_patience=8,
    )
    assert events == [] and not should_stop
    assert state["no_improvement_validations"] == 0
    assert state["execution_lr_plateau_bad_validations"] == 0
    assert state["early_stop_bad_validations"] == 0


def test_validation_scheduler_reduces_assignment_lr_and_leaves_execution_lr() -> None:
    networks = _network_bundle(SwarmModelConfig(d_model=16, num_heads=4))
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(
            assignment_actor_learning_rate=1.0e-4,
            execution_actor_learning_rate=5.0e-5,
        ),
    )
    state = {"no_improvement_validations": 0, "assignment_lr_reductions": 0}
    execution_lr = trainer.execution_actor_optimizer.param_groups[0]["lr"]

    events, should_stop = _step_assignment_validation_scheduler(
        trainer,
        state,
        improved=False,
        lr_patience=1,
        lr_factor=0.5,
        min_actor_lr=1.0e-5,
        early_stop_patience=1,
    )

    assert should_stop is True
    assert events == [
        {
            "event": "learning_rate_reduced",
            "old_lr": pytest.approx(1.0e-4),
            "new_lr": pytest.approx(5.0e-5),
        }
    ]
    assert trainer.assignment_actor_optimizer.param_groups[0]["lr"] == pytest.approx(
        5.0e-5
    )
    assert trainer.execution_actor_optimizer.param_groups[0]["lr"] == execution_lr
    assert state == {
        "no_improvement_validations": 1,
        "assignment_lr_plateau_bad_validations": 0,
        "early_stop_bad_validations": 1,
        "assignment_lr_reductions": 1,
    }


def test_execution_policy_restore_recovers_actor_and_adam_at_reduced_lr() -> None:
    networks = _network_bundle(SwarmModelConfig(d_model=16, num_heads=4))
    execution_actor = networks[1]
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(execution_actor_learning_rate=5.0e-5),
    )
    for parameter in execution_actor.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer.execution_actor_optimizer.step()
    trainer.execution_actor_optimizer.zero_grad(set_to_none=True)
    safe_actor = {
        name: tensor.detach().clone()
        for name, tensor in execution_actor.state_dict().items()
    }
    safe_optimizer = copy.deepcopy(
        trainer.execution_actor_optimizer.state_dict()
    )
    best_score = (0.8, 0.8, -0.2, -100.0, -0.01)
    checkpoint = {
        "execution_actor": safe_actor,
        "trainer": {"execution_actor_optimizer": safe_optimizer},
        "training_state": {
            "best_checkpoint_score": list(best_score),
            "best_iteration": 5,
            "best_checkpoint_origin": "warmup_pn_baseline",
        },
    }

    for parameter in execution_actor.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    trainer.execution_actor_optimizer.step()
    trainer.execution_actor_optimizer.zero_grad(set_to_none=True)
    assert any(
        not torch.equal(tensor, safe_actor[name])
        for name, tensor in execution_actor.state_dict().items()
    )

    result = _restore_execution_policy_from_checkpoint(
        checkpoint,
        execution_actor,
        trainer,
        expected_best_score=best_score,
        learning_rate=2.5e-5,
    )

    for name, tensor in execution_actor.state_dict().items():
        assert torch.equal(tensor, safe_actor[name])
    restored_optimizer = trainer.execution_actor_optimizer.state_dict()
    for parameter_id, state in safe_optimizer["state"].items():
        for name, value in state.items():
            restored = restored_optimizer["state"][parameter_id][name]
            if torch.is_tensor(value):
                assert torch.equal(restored, value)
            else:
                assert restored == value
    assert trainer.execution_actor_optimizer.param_groups[0]["lr"] == pytest.approx(
        2.5e-5
    )
    assert result == {
        "restored_best_iteration": 5,
        "restored_best_origin": "warmup_pn_baseline",
        "restored_execution_actor_optimizer": True,
    }


def test_assignment_policy_restore_recovers_actor_and_adam_at_reduced_lr() -> None:
    networks = _network_bundle(SwarmModelConfig(d_model=16, num_heads=4))
    assignment_actor = networks[0]
    trainer = MAPPOTrainer(
        *networks,
        PPOConfig(assignment_actor_learning_rate=1.0e-4),
    )
    for parameter in assignment_actor.parameters():
        parameter.grad = torch.ones_like(parameter)
    trainer.assignment_actor_optimizer.step()
    trainer.assignment_actor_optimizer.zero_grad(set_to_none=True)
    safe_actor = {
        name: tensor.detach().clone()
        for name, tensor in assignment_actor.state_dict().items()
    }
    safe_optimizer = copy.deepcopy(
        trainer.assignment_actor_optimizer.state_dict()
    )
    best_score = (0.9, 0.95, -0.1, -100.0, -0.01)
    checkpoint = {
        "assignment_actor": safe_actor,
        "trainer": {"assignment_actor_optimizer": safe_optimizer},
        "training_state": {
            "best_checkpoint_score": list(best_score),
            "best_iteration": 10,
            "best_checkpoint_origin": "policy_update",
        },
    }

    for parameter in assignment_actor.parameters():
        parameter.grad = torch.full_like(parameter, 2.0)
    trainer.assignment_actor_optimizer.step()
    trainer.assignment_actor_optimizer.zero_grad(set_to_none=True)
    assert any(
        not torch.equal(tensor, safe_actor[name])
        for name, tensor in assignment_actor.state_dict().items()
    )

    result = _restore_assignment_policy_from_checkpoint(
        checkpoint,
        assignment_actor,
        trainer,
        expected_best_score=best_score,
        learning_rate=5.0e-5,
    )

    for name, tensor in assignment_actor.state_dict().items():
        assert torch.equal(tensor, safe_actor[name])
    restored_optimizer = trainer.assignment_actor_optimizer.state_dict()
    for parameter_id, state in safe_optimizer["state"].items():
        for name, value in state.items():
            restored = restored_optimizer["state"][parameter_id][name]
            if torch.is_tensor(value):
                assert torch.equal(restored, value)
            else:
                assert restored == value
    assert trainer.assignment_actor_optimizer.param_groups[0]["lr"] == pytest.approx(
        5.0e-5
    )
    assert result == {
        "restored_best_iteration": 10,
        "restored_best_origin": "policy_update",
        "restored_assignment_actor_optimizer": True,
    }


def test_stage1_early_stop_restores_best_before_writing_latest(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    metrics = tmp_path / "metrics.json"
    manifest = tmp_path / "run_manifest.json"
    validation_results = iter(
        [
            {
                "trial_count": 2,
                "full_success_rate": 0.8,
                "average_damage_rate": 0.8,
                "ineffective_loss_rate": 0.2,
                "successful_completion_time_s": 100.0,
                "control_effort": 0.0,
                "assignment_mode": "capacity_aware",
                "by_scenario": [],
            },
            {
                "trial_count": 2,
                "full_success_rate": 0.7,
                "average_damage_rate": 0.7,
                "ineffective_loss_rate": 0.3,
                "successful_completion_time_s": 99.0,
                "control_effort": 0.01,
                "assignment_mode": "capacity_aware",
                "by_scenario": [],
            },
        ]
    )

    monkeypatch.setattr(
        train_env_module,
        "_fixed_validation_metrics",
        lambda *args, **kwargs: next(validation_results),
    )
    args = build_training_parser().parse_args(
        [
            "--device", "cpu",
            "--training-mode", "low_only",
            "--iterations", "2",
            "--low-critic-warmup-updates", "1",
            "--low-critic-warmup-critic-steps-per-update", "1",
            "--parallel-envs", "2",
            "--parallel-backend", "thread",
            "--rollout-steps", "1",
            "--red-counts", "1,2",
            "--red-count-batch-mode", "stratified",
            "--blue-counts", "1",
            "--styles", "many_to_one",
            "--time-step-s", "0.05",
            "--bias-update-interval-s", "0.05",
            "--assignment-update-interval-s", "0.1",
            "--max-steps", "4",
            "--missile-boost-time-s", "0.1",
            "--d-model", "16",
            "--num-heads", "4",
            "--ppo-epochs", "1",
            "--critic-updates-per-actor", "1",
            "--validation-interval", "1",
            "--validation-trials-per-blue-count", "1",
            "--execution-lr-plateau-patience", "0",
            "--execution-restore-best-on-early-stop",
            "--early-stop-validation-patience", "1",
            "--checkpoint-interval", "0",
            "--latest-checkpoint", str(latest),
            "--best-checkpoint", str(best),
            "--metrics-path", str(metrics),
            "--run-manifest-path", str(manifest),
        ]
    )

    assert train(args) == 0
    events = [
        json.loads(line)["event"]
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    assert events.index("early_stop_best_restored") < events.index("early_stop")

    best_checkpoint = _load_torch_checkpoint(best)
    latest_checkpoint = _load_torch_checkpoint(latest)
    for name, tensor in best_checkpoint["execution_actor"].items():
        assert torch.equal(tensor, latest_checkpoint["execution_actor"][name])
    state = latest_checkpoint["training_state"]
    assert state["execution_restore_best_on_early_stop"] is True
    assert state["execution_policy_restorations"] == 1
    assert state["execution_terminal_policy_restorations"] == 1
    assert state["best_iteration"] == 1

    metrics_data = json.loads(metrics.read_text(encoding="utf-8"))
    assert metrics_data["stop_reason"] == "early_stop_validation_patience"
    assert metrics_data["training_state"][
        "execution_terminal_policy_restorations"
    ] == 1
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["training_control"][
        "execution_restore_best_on_early_stop"
    ] is True


def test_stage2_early_stop_restores_assignment_best_and_keeps_execution_bitwise(
    tmp_path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    latest = tmp_path / "stage2_latest.pt"
    best = tmp_path / "stage2_best.pt"
    metrics = tmp_path / "stage2_metrics.json"
    manifest = tmp_path / "stage2_manifest.json"
    validation_results = iter(
        [
            {
                "trial_count": 2,
                "full_success_rate": 0.9,
                "average_damage_rate": 0.95,
                "ineffective_loss_rate": 0.1,
                "successful_completion_time_s": 100.0,
                "control_effort": 0.01,
                "assignment_mode": "actor",
                "by_scenario": [],
            },
            {
                "trial_count": 2,
                "full_success_rate": 0.8,
                "average_damage_rate": 0.9,
                "ineffective_loss_rate": 0.2,
                "successful_completion_time_s": 99.0,
                "control_effort": 0.01,
                "assignment_mode": "actor",
                "by_scenario": [],
            },
        ]
    )
    monkeypatch.setattr(
        train_env_module,
        "_fixed_validation_metrics",
        lambda *args, **kwargs: next(validation_results),
    )
    args = build_training_parser().parse_args(
        [
            "--device", "cpu",
            "--training-mode", "high_only",
            "--iterations", "2",
            "--parallel-envs", "2",
            "--parallel-backend", "thread",
            "--rollout-steps", "1",
            "--red-counts", "2",
            "--blue-counts", "1",
            "--styles", "many_to_one",
            "--time-step-s", "0.05",
            "--bias-update-interval-s", "0.05",
            "--assignment-update-interval-s", "0.1",
            "--max-steps", "4",
            "--missile-boost-time-s", "0.1",
            "--d-model", "16",
            "--num-heads", "4",
            "--ppo-epochs", "1",
            "--critic-updates-per-actor", "1",
            "--assignment-reward-learning-scale", "0.001953125",
            "--assignment-entropy-coef", "0.001",
            "--assignment-stickiness-logit-bonus", "1.0",
            "--high-potential-weight", "512",
            "--terminal-success-reward", "512",
            "--validation-interval", "1",
            "--validation-trials-per-blue-count", "1",
            "--assignment-lr-plateau-patience", "1",
            "--assignment-lr-plateau-factor", "0.5",
            "--assignment-min-actor-learning-rate", "1e-5",
            "--assignment-restore-best-on-lr-reduction",
            "--assignment-restore-best-on-early-stop",
            "--early-stop-validation-patience", "1",
            "--checkpoint-interval", "0",
            "--latest-checkpoint", str(latest),
            "--best-checkpoint", str(best),
            "--metrics-path", str(metrics),
            "--run-manifest-path", str(manifest),
        ]
    )

    assert train(args) == 0
    event_rows = [
        json.loads(line)
        for line in capsys.readouterr().out.splitlines()
        if line.startswith("{")
    ]
    events = [row["event"] for row in event_rows]
    assert events.index("learning_rate_reduced") < events.index(
        "early_stop_best_restored"
    ) < events.index("early_stop")
    reduced = next(row for row in event_rows if row["event"] == "learning_rate_reduced")
    restored = next(
        row for row in event_rows if row["event"] == "early_stop_best_restored"
    )
    assert reduced["policy"] == "assignment"
    assert reduced["assignment_policy_restored"] is True
    assert restored["policy"] == "assignment"
    assert restored["assignment_policy_restored"] is True

    best_checkpoint = _load_torch_checkpoint(best)
    latest_checkpoint = _load_torch_checkpoint(latest)
    for network_key in ("assignment_actor", "execution_actor", "execution_critic"):
        for name, tensor in best_checkpoint[network_key].items():
            assert torch.equal(tensor, latest_checkpoint[network_key][name])
    state = latest_checkpoint["training_state"]
    assert state["assignment_lr_reductions"] == 1
    assert state["assignment_policy_restorations"] == 2
    assert state["assignment_terminal_policy_restorations"] == 1
    assert state["execution_policy_restorations"] == 0
    assert state["execution_terminal_policy_restorations"] == 0
    assert state["best_iteration"] == 1
    assert latest_checkpoint["ppo_config"][
        "assignment_reward_learning_scale"
    ] == pytest.approx(1.0 / 512.0)
    assert latest_checkpoint["env_config"]["reward"][
        "terminal_success_reward"
    ] == pytest.approx(512.0)
    assert latest_checkpoint["trainer"]["assignment_actor_optimizer"][
        "param_groups"
    ][0]["lr"] == pytest.approx(5.0e-5)
    metrics_data = json.loads(metrics.read_text(encoding="utf-8"))
    assert metrics_data["stop_reason"] == "early_stop_validation_patience"
    assert all(row["execution_actor_updates"] == 0.0 for row in metrics_data["iterations"])
    assert all(row["execution_critic_updates"] == 0.0 for row in metrics_data["iterations"])
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["training_control"][
        "assignment_restore_best_on_early_stop"
    ] is True


def test_stage1_pn_comparison_builds_overall_and_scenario_metrics(tmp_path) -> None:
    configuration = {
        "seed_start": 20271000,
        "trials_per_scenario": 100,
        "blue_count": 1,
        "max_missiles_per_target": 4,
    }
    learned_scenarios = [
        {
            "red_count": red_count,
            "blue_count": 1,
            "full_success_rate": 0.95,
            "ineffective_loss_rate": 0.05,
        }
        for red_count in (1, 2, 3, 4)
    ]
    baseline_scenarios = [
        {
            "red_count": red_count,
            "blue_count": 1,
            "full_success_rate": 0.85,
            "ineffective_loss_rate": 0.15,
        }
        for red_count in (1, 2, 3, 4)
    ]
    baseline_path = tmp_path / "pn_summary.json"
    baseline_path.write_text(
        json.dumps(
            {
                "configuration": configuration,
                "overall": {
                    "full_success_rate": 0.85,
                    "ineffective_loss_rate": 0.15,
                },
                "by_scenario": baseline_scenarios,
            }
        ),
        encoding="utf-8",
    )

    comparison = _compare_to_pn_baseline(
        {
            "configuration": configuration,
            "overall": {
                "full_success_rate": 0.95,
                "ineffective_loss_rate": 0.05,
            },
            "by_scenario": learned_scenarios,
        },
        baseline_path,
    )

    assert comparison["overall"]["full_success_rate"][
        "checkpoint_minus_pn"
    ] == pytest.approx(0.10)
    assert [item["red_count"] for item in comparison["by_scenario"]] == [
        1, 2, 3, 4
    ]
    assert comparison["by_scenario"][3]["metrics"][
        "ineffective_loss_rate"
    ]["checkpoint_minus_pn"] == pytest.approx(-0.10)


def test_stage1_quality_gate_is_per_scenario_and_checkpoint_bound(tmp_path) -> None:
    checkpoint_path = tmp_path / "stage1.pt"
    checkpoint_path.write_bytes(b"checkpoint-v1")
    baseline_path = tmp_path / "pn_summary.json"
    baseline_path.write_text('{"baseline": true}\n', encoding="utf-8")

    def compared_metrics(
        learned_success: float = 0.9,
        learned_ineffective: float = 0.1,
    ) -> dict[str, object]:
        return {
            "full_success_rate": {
                "pn_zero_residual": 0.9,
                "checkpoint_residual": learned_success,
                "checkpoint_minus_pn": learned_success - 0.9,
            },
            "ineffective_loss_rate": {
                "pn_zero_residual": 0.1,
                "checkpoint_residual": learned_ineffective,
                "checkpoint_minus_pn": learned_ineffective - 0.1,
            },
        }

    summary = {
        "overall": {
            "pn_gain_all_valid": True,
            "residual_bound_all_valid": True,
            "capacity_all_valid": True,
            "finite_state_all_valid": True,
        },
        "comparison_to_pn_zero_residual": {
            "overall": compared_metrics(),
            "by_scenario": [
                {
                    "red_count": red_count,
                    "blue_count": 1,
                    "metrics": compared_metrics(),
                }
                for red_count in (1, 2, 3, 4)
            ],
        },
    }
    gate = _build_stage1_quality_gate(
        summary,
        checkpoint_path,
        baseline_path,
    )
    assert gate["passed"] is True
    assert gate["checkpoint_sha256"] == hashlib.sha256(b"checkpoint-v1").hexdigest()

    gate_path = tmp_path / "learned_summary.json"
    gate_path.write_text(
        json.dumps(
            {
                "evaluation": "stage1_low_checkpoint_residual_guidance",
                "stage1_quality_gate": gate,
            }
        ),
        encoding="utf-8",
    )
    validated = _validate_stage1_quality_gate(checkpoint_path, gate_path)
    assert validated["passed"] is True

    checkpoint_path.write_bytes(b"checkpoint-v2")
    with pytest.raises(ValueError, match="SHA256"):
        _validate_stage1_quality_gate(checkpoint_path, gate_path)

    checkpoint_path.write_bytes(b"checkpoint-v1")
    summary["comparison_to_pn_zero_residual"]["by_scenario"][2]["metrics"] = (
        compared_metrics(learned_success=0.89)
    )
    failed_gate = _build_stage1_quality_gate(
        summary,
        checkpoint_path,
        baseline_path,
    )
    assert failed_gate["passed"] is False
    assert failed_gate["by_scenario"][2]["passed"] is False


def test_stage1_fullfix_warmup_mixed_red_checkpoint_state(tmp_path) -> None:
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    metrics = tmp_path / "metrics.json"
    manifest = tmp_path / "run_manifest.json"
    args = build_training_parser().parse_args(
        [
            "--device", "cpu",
            "--training-mode", "low_only",
            "--iterations", "2",
            "--low-critic-warmup-updates", "1",
            "--low-critic-warmup-critic-steps-per-update", "3",
            "--parallel-envs", "2",
            "--parallel-backend", "thread",
            "--rollout-steps", "1",
            "--red-counts", "1,2",
            "--red-count-batch-mode", "stratified",
            "--blue-counts", "1",
            "--styles", "many_to_one",
            "--time-step-s", "0.05",
            "--bias-update-interval-s", "0.05",
            "--assignment-update-interval-s", "0.1",
            "--max-steps", "4",
            "--missile-boost-time-s", "0.1",
            "--d-model", "16",
            "--num-heads", "4",
            "--ppo-epochs", "1",
            "--critic-updates-per-actor", "1",
            "--execution-reward-learning-scale", "0.001953125",
            "--execution-value-loss", "huber",
            "--execution-advantage-normalization", "per_scenario",
            "--execution-actor-loss-weighting", "per_scenario",
            "--execution-action-distribution", "radial_tanh_disk",
            "--critic-value-head-mode", "scalar",
            "--low-time-credit-mode", "terminal_active_share",
            "--low-time-weight", "2",
            "--low-option-boundary-potential", "terminal_zero",
            "--execution-post-step-kl-rollback",
            "--execution-post-step-kl-limit", "1.0",
            "--validation-interval", "1",
            "--validation-trials-per-blue-count", "1",
            "--checkpoint-interval", "0",
            "--latest-checkpoint", str(latest),
            "--best-checkpoint", str(best),
            "--metrics-path", str(metrics),
            "--run-manifest-path", str(manifest),
        ]
    )

    assert train(args) == 0
    data = json.loads(metrics.read_text(encoding="utf-8"))
    _assert_json_numbers_finite(data)
    assert [row["training_mode"] for row in data["iterations"]] == [
        "low_critic_only",
        "low_only",
    ]
    assert data["completed_optimizer_updates"] == 2
    assert data["completed_policy_updates"] == 1
    assert data["iterations"][0]["execution_actor_steps_attempted"] == 0.0
    assert data["iterations"][0]["execution_critic_updates"] == 3.0
    assert data["iterations"][0]["best_iteration"] == 1
    assert data["iterations"][0]["best_checkpoint_origin"] == "warmup_pn_baseline"
    assert "execution_explained_variance_before_update" in data["iterations"][0]
    assert data["iterations"][1][
        "execution_scenario_r1_b1_actor_weight_share"
    ] == pytest.approx(0.5)
    assert data["iterations"][1][
        "execution_scenario_r2_b1_actor_weight_share"
    ] == pytest.approx(0.5)
    assert all(
        sorted(row["sampled_red_counts"]) == [1, 2]
        for row in data["iterations"]
    )
    checkpoint = _load_torch_checkpoint(latest)
    state = checkpoint["training_state"]
    assert state["completed_optimizer_updates"] == 2
    assert state["completed_policy_updates"] == 1
    assert state["low_critic_warmup_updates"] == 1
    assert state["low_critic_warmup_critic_steps_per_update"] == 3
    assert state["red_count_batch_mode"] == "stratified"
    assert checkpoint["ppo_config"]["execution_reward_learning_scale"] == pytest.approx(
        1.0 / 512.0
    )
    assert checkpoint["ppo_config"]["execution_advantage_normalization"] == "per_scenario"
    assert checkpoint["ppo_config"]["execution_actor_loss_weighting"] == "per_scenario"
    assert checkpoint["schema_version"] == 14
    assert checkpoint["model_config"]["execution_action_distribution"] == "radial_tanh_disk"
    assert checkpoint["model_config"]["critic_value_head_mode"] == "scalar"
    assert checkpoint["model_config"]["d_value_components"] == 1
    assert checkpoint["env_config"]["reward"]["low_time_credit_mode"] == "terminal_active_share"
    assert checkpoint["env_config"]["reward"]["low_option_boundary_potential"] == "terminal_zero"
    manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
    assert manifest_data["stop_reason"] == "max_iterations"
    assert manifest_data["completed_policy_updates"] == 1
    assert manifest_data["source_sha256"]

    blocked_stage2 = build_training_parser().parse_args(
        [
            "--device",
            "cpu",
            "--training-mode",
            "high_only",
            "--iterations",
            "0",
            "--resume-checkpoint",
            str(latest),
            "--latest-checkpoint",
            "",
            "--best-checkpoint",
            "",
            "--metrics-path",
            "",
        ]
    )
    with pytest.raises(ValueError, match="stage1-quality-gate"):
        train(blocked_stage2)

    gate_path = tmp_path / "stage1_quality_gate.json"
    gate_path.write_text(
        json.dumps(
            {
                "evaluation": "stage1_low_checkpoint_residual_guidance",
                "stage1_quality_gate": {
                    "schema_version": 1,
                    "policy_mode": "deterministic",
                    "checkpoint_sha256": hashlib.sha256(latest.read_bytes()).hexdigest(),
                    "baseline_summary_sha256": "synthetic-test-baseline",
                    "required_red_counts": [1, 2, 3, 4],
                    "evaluated_red_counts": [1, 2, 3, 4],
                    "runtime_validity": {
                        "pn_gain_all_valid": True,
                        "residual_bound_all_valid": True,
                        "capacity_all_valid": True,
                        "finite_state_all_valid": True,
                    },
                    "overall": {"passed": True},
                    "by_scenario": [
                        {"red_count": red_count, "passed": True}
                        for red_count in (1, 2, 3, 4)
                    ],
                    "passed": True,
                },
            }
        ),
        encoding="utf-8",
    )
    missing_reset_args = build_training_parser().parse_args(
        [
            "--device", "cpu",
            "--training-mode", "high_only",
            "--iterations", "0",
            "--resume-checkpoint", str(latest),
            "--stage1-quality-gate", str(gate_path),
            "--latest-checkpoint", "",
            "--best-checkpoint", "",
            "--metrics-path", "",
        ]
    )
    with pytest.raises(ValueError, match="reset-best-on-resume"):
        train(missing_reset_args)

    stage2_metrics = tmp_path / "stage2_transition_metrics.json"
    stage2_manifest = tmp_path / "stage2_transition_manifest.json"
    stage2_latest = tmp_path / "stage2" / "latest.pt"
    transitioned_args = build_training_parser().parse_args(
        [
            "--device", "cpu",
            "--training-mode", "high_only",
            "--reset-best-on-resume",
            "--iterations", "1",
            "--seed", "20260810",
            "--parallel-envs", "1",
            "--parallel-backend", "thread",
            "--red-counts", "24",
            "--blue-counts", "4,5,6",
            "--styles", "many_to_many",
            "--scenario-sampling", "random",
            "--red-count-batch-mode", "stratified",
            "--assignment-reward-learning-scale", "0.001953125",
            "--assignment-entropy-coef", "0.001",
            "--assignment-stickiness-logit-bonus", "1.0",
            "--high-potential-weight", "512",
            "--terminal-success-reward", "512",
            "--assignment-lr-plateau-patience", "2",
            "--assignment-lr-plateau-factor", "0.5",
            "--assignment-min-actor-learning-rate", "1e-5",
            "--assignment-restore-best-on-lr-reduction",
            "--assignment-restore-best-on-early-stop",
            "--early-stop-validation-patience", "4",
            "--resume-checkpoint", str(latest),
            "--stage1-quality-gate", str(gate_path),
            "--latest-checkpoint", str(stage2_latest),
            "--best-checkpoint", "",
            "--checkpoint", "",
            "--checkpoint-interval", "1",
            "--metrics-path", str(stage2_metrics),
            "--run-manifest-path", str(stage2_manifest),
            "--validation-interval", "0",
        ]
    )
    assert train(transitioned_args) == 0
    stage2_data = json.loads(stage2_metrics.read_text(encoding="utf-8"))
    assert stage2_data["model_config"]["critic_value_head_mode"] == "scalar"
    assert stage2_data["model_config"]["d_value_components"] == 1
    assert (
        stage2_data["model_config"]["assignment_critic_value_head_mode"]
        == "latent_sum"
    )
    assert stage2_data["start_iteration"] == 0
    assert stage2_data["completed_iterations"] == 1
    assert stage2_data["completed_optimizer_updates"] == 1
    assert stage2_data["completed_policy_updates"] == 1
    assert stage2_data["completed_stage_policy_updates"] == 1
    assert stage2_data["iterations"][0]["iteration"] == 1
    assert stage2_data["iterations"][0]["sampled_red_counts"] == [24]
    assert len(stage2_data["iterations"][0]["sampled_blue_counts"]) == 1
    assert stage2_data["training_state"]["best_checkpoint_score"] is None
    assert stage2_data["training_state"]["seed"] == 20260810
    expected_stage_origin = {
        "source_checkpoint": str(latest),
        "source_training_mode": "low_only",
        "source_completed_iterations": 2,
        "source_completed_optimizer_updates": 2,
        "source_completed_policy_updates": 1,
        "source_completed_stage_policy_updates": 1,
    }
    assert stage2_data["stage_origin"] == expected_stage_origin
    assert stage2_data["training_state"]["stage_origin"] == expected_stage_origin
    assert stage2_data["stage_transition"] == {
        **expected_stage_origin,
        "target_training_mode": "high_only",
        "assignment_critic_reinitialized": True,
        "assignment_critic_value_head_mode": "latent_sum",
        "checkpoint_rng_restored": False,
        "trainer_update_step_reset": True,
        "stage_counters_reset": True,
    }
    assert stage2_latest.exists()
    assert (stage2_latest.parent / "iteration_000001.pt").exists()
    stage2_checkpoint = _load_torch_checkpoint(stage2_latest)
    assert stage2_checkpoint["training_state"]["completed_iterations"] == 1
    assert stage2_checkpoint["training_state"]["stage_origin"] == expected_stage_origin
    assert stage2_checkpoint["ppo_config"][
        "assignment_reward_learning_scale"
    ] == pytest.approx(1.0 / 512.0)
    assert stage2_checkpoint["ppo_config"][
        "assignment_entropy_coef"
    ] == pytest.approx(0.001)
    assert stage2_checkpoint["model_config"][
        "assignment_stickiness_logit_bonus"
    ] == pytest.approx(1.0)
    assert stage2_checkpoint["env_config"]["reward"][
        "high_potential_weight"
    ] == pytest.approx(512.0)
    assert stage2_checkpoint["env_config"]["reward"][
        "terminal_success_reward"
    ] == pytest.approx(512.0)
    assert stage2_checkpoint["training_state"][
        "assignment_lr_plateau_patience"
    ] == 2
    assert stage2_checkpoint["training_state"][
        "assignment_restore_best_on_lr_reduction"
    ] is True
    assert stage2_checkpoint["training_state"][
        "assignment_restore_best_on_early_stop"
    ] is True

    stage2_resumed_metrics = tmp_path / "stage2_resumed_metrics.json"
    resumed_stage2_args = build_training_parser().parse_args(
        [
            "--device", "cpu",
            "--training-mode", "high_only",
            "--iterations", "0",
            "--parallel-envs", "1",
            "--parallel-backend", "thread",
            "--resume-checkpoint", str(stage2_latest),
            "--latest-checkpoint", "",
            "--best-checkpoint", "",
            "--checkpoint", "",
            "--metrics-path", str(stage2_resumed_metrics),
            "--validation-interval", "0",
        ]
    )
    assert train(resumed_stage2_args) == 0
    resumed_stage2 = json.loads(stage2_resumed_metrics.read_text(encoding="utf-8"))
    assert resumed_stage2["start_iteration"] == 1
    assert resumed_stage2["completed_iterations"] == 1
    assert resumed_stage2["completed_optimizer_updates"] == 1
    assert resumed_stage2["completed_policy_updates"] == 1
    assert resumed_stage2["completed_stage_policy_updates"] == 1
    assert resumed_stage2["stage_transition"] is None
    assert resumed_stage2["stage_origin"] == expected_stage_origin
    assert resumed_stage2["ppo_config"][
        "assignment_reward_learning_scale"
    ] == pytest.approx(1.0 / 512.0)
    assert resumed_stage2["ppo_config"][
        "assignment_entropy_coef"
    ] == pytest.approx(0.001)
    assert resumed_stage2["model_config"][
        "assignment_stickiness_logit_bonus"
    ] == pytest.approx(1.0)
    assert resumed_stage2["env_config"]["reward"][
        "high_potential_weight"
    ] == pytest.approx(512.0)
    assert resumed_stage2["env_config"]["reward"][
        "terminal_success_reward"
    ] == pytest.approx(512.0)
    assert resumed_stage2["training_state"][
        "assignment_lr_plateau_patience"
    ] == 2

    rejected_override = build_training_parser().parse_args(
        [
            "--device", "cpu",
            "--training-mode", "high_only",
            "--iterations", "0",
            "--parallel-envs", "1",
            "--parallel-backend", "thread",
            "--resume-checkpoint", str(stage2_latest),
            "--assignment-reward-learning-scale", "1.0",
            "--latest-checkpoint", "",
            "--best-checkpoint", "",
            "--checkpoint", "",
            "--metrics-path", "",
            "--validation-interval", "0",
        ]
    )
    with pytest.raises(
        ValueError,
        match="cannot override assignment_reward_learning_scale",
    ):
        train(rejected_override)


def test_stage1_dedicated_validation_pool_runs_one_exact_wave(tmp_path) -> None:
    latest = tmp_path / "latest.pt"
    best = tmp_path / "best.pt"
    metrics = tmp_path / "metrics.json"
    manifest = tmp_path / "run_manifest.json"
    args = build_training_parser().parse_args(
        [
            "--device", "cpu",
            "--training-mode", "low_only",
            "--iterations", "1",
            "--low-critic-warmup-updates", "1",
            "--low-critic-warmup-critic-steps-per-update", "1",
            "--parallel-envs", "2",
            "--parallel-backend", "process",
            "--env-worker-threads", "1",
            "--rollout-steps", "1",
            "--red-counts", "1",
            "--blue-counts", "1",
            "--styles", "many_to_one",
            "--time-step-s", "0.05",
            "--bias-update-interval-s", "0.05",
            "--assignment-update-interval-s", "0.1",
            "--max-steps", "4",
            "--missile-boost-time-s", "0.1",
            "--d-model", "16",
            "--num-heads", "4",
            "--ppo-epochs", "1",
            "--critic-updates-per-actor", "1",
            "--validation-interval", "1",
            "--validation-trials-per-blue-count", "3",
            "--validation-parallel-envs", "3",
            "--checkpoint-interval", "0",
            "--latest-checkpoint", str(latest),
            "--best-checkpoint", str(best),
            "--metrics-path", str(metrics),
            "--run-manifest-path", str(manifest),
        ]
    )

    assert train(args) == 0
    data = json.loads(metrics.read_text(encoding="utf-8"))
    validation = data["validation_config"]
    assert validation["parallel_envs"] == 3
    assert validation["waves_per_scenario"] == 1
    assert validation["actual_simulations_per_scenario"] == 3
    assert data["iterations"][0]["fixed_validation"]["trial_count"] == 3.0
    assert data["iterations"][0]["execution_critic_updates"] == 1.0
    assert latest.exists() and best.exists()
    best_state = _load_torch_checkpoint(best)["training_state"]
    assert best_state["best_iteration"] == 1
    assert best_state["best_checkpoint_origin"] == "warmup_pn_baseline"
