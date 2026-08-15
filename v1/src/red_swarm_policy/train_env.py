from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cli_utils import parse_float_pair, parse_float_range, parse_float_sequence
from .core.config import PPOConfig, SwarmModelConfig
from .env import (
    AircraftConfig,
    BlueEvasionConfig,
    BlueEvasionController,
    BlueEvasionRuleMachine,
    EnvironmentConfig,
    MissileConfig,
    RedBlueEngagementEnv,
    RewardConfig,
    SCENARIO_STYLES,
    ScenarioConfig,
    SensorConfig,
)
from .policy.actor import OverloadBiasActor, TargetAssignmentActor
from .policy.critic import OverloadBiasCritic, TargetAssignmentCritic
from .training.env_pool import ProcessEnvironmentPool
from .training.mappo import MAPPOTrainer
from .training.rollout import collect_parallel_rollout, evaluate_parallel_episodes

CHECKPOINT_SCHEMA_VERSION = 14
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = (11, 12, 13, 14)
DEFAULT_ROLLOUT_STEPS = 80
DEFAULT_SEED = 20260703
DEFAULT_STYLE = "many_to_many"
DEFAULT_RED_COUNT = 24
DEFAULT_BLUE_COUNT = 4
ASSIGNMENT_MODES = ("actor", "capacity_aware")
VALIDATION_ASSIGNMENT_MODE_CHOICES = ("auto", *ASSIGNMENT_MODES)


def _select_device(name: str) -> tuple[torch.device, str]:
    if name == "auto":
        name = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(f"CUDA device index {index} is unavailable")
        torch.cuda.set_device(index)
        gpu_name = torch.cuda.get_device_name(index)
        return torch.device(f"cuda:{index}"), f"cuda:{index} ({gpu_name})"
    return torch.device("cpu"), "cpu"


def _build_env_config(args: argparse.Namespace) -> EnvironmentConfig:
    return EnvironmentConfig(
        time_step_s=args.time_step_s,
        bias_update_interval_s=args.bias_update_interval_s,
        assignment_update_interval_s=args.assignment_update_interval_s,
        max_steps=args.max_steps,
        policy_start_mode=args.policy_start_mode,
        policy_entry_speed_tolerance_ratio=args.policy_entry_speed_tolerance_ratio,
        policy_entry_flight_path_tolerance_deg=args.policy_entry_flight_path_tolerance_deg,
        scenario=ScenarioConfig(
            red_count=args.red_count,
            blue_count=args.blue_count,
            max_missiles_per_target=args.max_missiles_per_target,
            red_launch_mach_range=parse_float_range(args.red_launch_mach_range, "red_launch_mach_range"),
            red_altitude_range_m=parse_float_range(args.red_altitude_range_m, "red_altitude_range_m"),
            blue_speed_range_mps=parse_float_range(args.blue_speed_range_mps, "blue_speed_range_mps"),
            blue_altitude_range_m=parse_float_range(args.blue_altitude_range_m, "blue_altitude_range_m"),
            speed_of_sound_mps=args.reference_speed_of_sound_mps,
            blue_cluster_center_ne_m=parse_float_pair(args.blue_cluster_center_ne_m, "blue_cluster_center_ne_m"),
            blue_cluster_radius_m=args.blue_cluster_radius_m,
            blue_heading_range_deg=parse_float_range(
                args.blue_heading_range_deg,
                "blue_heading_range_deg",
                positive=False,
            ),
            red_cluster_radius_range_m=parse_float_range(
                args.red_cluster_radius_range_m,
                "red_cluster_radius_range_m",
            ),
            red_sector_center_azimuth_deg=args.red_sector_center_azimuth_deg,
            red_sector_width_deg=args.red_sector_width_deg,
            red_heading_bias_max_deg=args.red_heading_bias_max_deg,
            position_perturb_m=args.position_perturb_m,
            velocity_perturb_mps=args.velocity_perturb_mps,
        ),
        aircraft=AircraftConfig(
            min_speed_mps=args.aircraft_min_speed_mps,
            max_speed_mps=args.aircraft_max_speed_mps,
            min_altitude_m=args.aircraft_min_altitude_m,
            max_altitude_m=args.aircraft_max_altitude_m,
        ),
        missile=MissileConfig(
            dry_mass_kg=args.missile_dry_mass_kg,
            propellant_mass_kg=args.missile_propellant_mass_kg,
            boost_duration_s=args.missile_boost_time_s,
            boost_target_mach_number=args.missile_max_mach,
            reference_speed_of_sound_mps=args.reference_speed_of_sound_mps,
            boost_climb_angle_deg=args.missile_boost_climb_angle_deg,
            boost_pitch_transition_s=args.missile_boost_pitch_transition_s,
            boost_pitch_tracking_gain=args.missile_boost_pitch_tracking_gain,
            reference_area_m2=args.missile_reference_area_m2,
            drag_coefficient=args.missile_drag_coefficient,
            drag_mach_breakpoints=parse_float_sequence(
                args.missile_drag_mach_breakpoints,
                "missile_drag_mach_breakpoints",
                minimum_length=2,
            ),
            zero_lift_drag_coefficients=parse_float_sequence(
                args.missile_zero_lift_drag_coefficients,
                "missile_zero_lift_drag_coefficients",
                minimum_length=2,
            ),
            induced_drag_factor=args.missile_induced_drag_factor,
            max_load_factor_g=args.missile_max_load_factor_g,
            max_guidance_bias_g=args.missile_max_guidance_bias_g,
            proportional_navigation_gain=args.proportional_navigation_gain,
            max_guidance_time_s=args.missile_max_guidance_time_s,
            seeker_acquisition_fov_deg=args.seeker_acquisition_fov_deg,
            seeker_tracking_fov_deg=args.seeker_tracking_fov_deg,
            fov_break_hold_s=args.fov_break_hold_s,
            post_closest_growth_m=args.post_closest_growth_m,
            post_closest_recede_speed_mps=args.post_closest_recede_speed_mps,
            lethal_radius_m=args.lethal_radius_m,
        ),
        sensor=SensorConfig(
            detection_range_m=args.detection_range_m,
            communication_delay_steps=args.communication_delay_steps,
        ),
        reward=RewardConfig(
            high_damage_weight=args.high_damage_weight,
            high_waste_weight=args.high_waste_weight,
            high_potential_weight=(
                512.0
                if args.high_potential_weight is None
                else args.high_potential_weight
            ),
            high_potential_gamma=args.high_potential_gamma,
            high_time_penalty_per_s=args.high_time_penalty_per_s,
            high_time_margin_scale_s=args.high_time_margin_scale_s,
            terminal_success_reward=(
                0.0
                if args.terminal_success_reward is None
                else args.terminal_success_reward
            ),
            terminal_failure_penalty=args.terminal_failure_penalty,
            terminal_timeout_penalty=args.terminal_timeout_penalty,
            low_damage_weight=args.low_damage_weight,
            low_potential_weight=args.low_potential_weight,
            low_potential_gamma=args.low_potential_gamma,
            low_missile_failure_penalty=args.low_missile_failure_penalty,
            low_load_penalty=args.low_load_penalty,
            low_smooth_penalty=args.low_smooth_penalty,
            low_time_credit_mode=(
                "none"
                if args.low_time_credit_mode is None
                else args.low_time_credit_mode
            ),
            low_time_weight=args.low_time_weight,
            low_option_boundary_potential=(
                "exempt"
                if args.low_option_boundary_potential is None
                else args.low_option_boundary_potential
            ),
            zem_reference_range_m=args.zem_reference_range_m,
            zem_floor_range_m=args.zem_floor_range_m,
            zem_weight=args.zem_weight,
            seeker_lock_weight=args.seeker_lock_weight,
            smooth_bias_denominator=args.smooth_bias_denominator,
            zem_time_gate_scale_s=args.zem_time_gate_scale_s,
            assignment_min_energy_fraction=args.assignment_min_energy_fraction,
            assignment_min_available_load_fraction=args.assignment_min_available_load_fraction,
            assignment_correlation_weight=args.assignment_correlation_weight,
            assignment_correlation_angle_scale_deg=args.assignment_correlation_angle_scale_deg,
            assignment_correlation_time_scale_s=args.assignment_correlation_time_scale_s,
        ),
    )


def _build_model_config(args: argparse.Namespace) -> SwarmModelConfig:
    critic_value_head_mode = (
        "latent_sum"
        if args.critic_value_head_mode is None
        else args.critic_value_head_mode
    )
    return SwarmModelConfig(
        d_model=args.d_model,
        d_bias=args.d_bias,
        d_value_components=1 if critic_value_head_mode == "scalar" else 5,
        num_heads=args.num_heads,
        max_missiles_per_target=args.max_missiles_per_target,
        execution_action_distribution=(
            "tanh_box"
            if args.execution_action_distribution is None
            else args.execution_action_distribution
        ),
        critic_value_head_mode=critic_value_head_mode,
        assignment_stickiness_logit_bonus=(
            0.0
            if args.assignment_stickiness_logit_bonus is None
            else args.assignment_stickiness_logit_bonus
        ),
    )


def _build_ppo_config(args: argparse.Namespace) -> PPOConfig:
    return PPOConfig(
        gamma_high=args.gamma_high,
        gamma_low=args.gamma_low,
        lambda_high=args.lambda_high,
        lambda_low=args.lambda_low,
        learning_rate=args.learning_rate,
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        assignment_actor_learning_rate=args.assignment_actor_learning_rate,
        execution_actor_learning_rate=args.execution_actor_learning_rate,
        assignment_critic_learning_rate=args.assignment_critic_learning_rate,
        execution_critic_learning_rate=args.execution_critic_learning_rate,
        clip_epsilon=args.clip_epsilon,
        assignment_clip_epsilon=args.assignment_clip_epsilon,
        execution_clip_epsilon=args.execution_clip_epsilon,
        epochs=args.ppo_epochs,
        critic_updates_per_actor=args.critic_updates_per_actor,
        actor_update_interval=args.actor_update_interval,
        sequence_length=args.ppo_sequence_length,
        assignment_sequence_length=args.assignment_sequence_length,
        execution_sequence_length=args.execution_sequence_length,
        assignment_target_kl=args.assignment_target_kl,
        execution_target_kl=args.execution_target_kl,
        assignment_reward_learning_scale=(
            1.0
            if args.assignment_reward_learning_scale is None
            else args.assignment_reward_learning_scale
        ),
        execution_reward_learning_scale=(
            1.0
            if args.execution_reward_learning_scale is None
            else args.execution_reward_learning_scale
        ),
        execution_value_loss=(
            "mse" if args.execution_value_loss is None else args.execution_value_loss
        ),
        execution_value_huber_delta=args.execution_value_huber_delta,
        execution_post_step_kl_rollback=args.execution_post_step_kl_rollback,
        execution_post_step_kl_limit=args.execution_post_step_kl_limit,
        effort_finetune_scale=args.effort_finetune_scale,
        execution_advantage_normalization=(
            "global"
            if args.execution_advantage_normalization is None
            else args.execution_advantage_normalization
        ),
        execution_actor_loss_weighting=(
            "active_step"
            if args.execution_actor_loss_weighting is None
            else args.execution_actor_loss_weighting
        ),
        entropy_coef=args.entropy_coef,
        assignment_entropy_coef=(
            0.001
            if args.assignment_entropy_coef is None
            else args.assignment_entropy_coef
        ),
        execution_entropy_coef=args.execution_entropy_coef,
    )


def _build_blue_evasion_config(args: argparse.Namespace) -> BlueEvasionConfig:
    return BlueEvasionConfig(
        decision_interval_s=args.blue_rule_decision_interval_s,
        detection_range_m=args.blue_rule_detection_range_m,
        critical_range_m=args.blue_rule_critical_range_m,
        lookahead_s=args.blue_rule_lookahead_s,
        effort_penalty=args.blue_rule_effort_penalty,
        switch_penalty=args.blue_rule_switch_penalty,
    )


def _build_blue_policy(
    env_config: EnvironmentConfig,
    mode: str,
    evasion_config: BlueEvasionConfig,
) -> BlueEvasionController | None:
    if mode == "straight":
        return None
    if mode != "rule":
        raise ValueError("blue_policy must be 'rule' or 'straight'")
    return BlueEvasionController(BlueEvasionRuleMachine(env_config, evasion_config))


def _numpy_rng_state() -> dict[str, Any]:
    name, keys, pos, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "name": name,
        "keys": keys.tolist(),
        "pos": int(pos),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_rng_state(state: dict[str, Any]) -> None:
    np.random.set_state(
        (
            str(state["name"]),
            np.asarray(state["keys"], dtype=np.uint32),
            int(state["pos"]),
            int(state["has_gauss"]),
            float(state["cached_gaussian"]),
        )
    )


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "torch": torch.get_rng_state(),
        "numpy": _numpy_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any], device: torch.device) -> bool:
    restored = False
    if "torch" in state:
        torch.set_rng_state(state["torch"].cpu())
        restored = True
    if "numpy" in state:
        _restore_numpy_rng_state(state["numpy"])
        restored = True
    cuda_states = state.get("torch_cuda")
    if device.type == "cuda" and torch.cuda.is_available() and cuda_states:
        index = 0 if device.index is None else device.index
        if len(cuda_states) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([rng.cpu() for rng in cuda_states])
        else:
            torch.cuda.set_rng_state(cuda_states[min(index, len(cuda_states) - 1)].cpu(), device=device)
        restored = True
    return restored


def _seed_validation_policy_rng(seed: int, device: torch.device) -> None:
    if seed < 0:
        raise ValueError("validation policy seed must be non-negative")
    torch.manual_seed(seed)
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_torch_checkpoint(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict):
        raise ValueError(f"checkpoint must be a dict: {path}")
    return checkpoint


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_stage1_quality_gate(
    checkpoint_path: Path,
    quality_gate_path: Path,
) -> dict[str, Any]:
    if not quality_gate_path.exists():
        raise FileNotFoundError(
            f"stage-one quality gate does not exist: {quality_gate_path}"
        )
    with quality_gate_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("evaluation") != "stage1_low_checkpoint_residual_guidance":
        raise ValueError("stage-one quality gate has an unexpected evaluation type")
    gate = document.get("stage1_quality_gate")
    if not isinstance(gate, dict):
        raise ValueError("stage-one quality gate document is missing stage1_quality_gate")
    if gate.get("schema_version") != 1:
        raise ValueError("unsupported stage-one quality gate schema")
    if gate.get("policy_mode") != "deterministic":
        raise ValueError("Stage 2 requires a deterministic Stage 1 quality gate")
    if gate.get("passed") is not True:
        raise ValueError("Stage 1 checkpoint did not pass its PN holdout quality gate")
    required_red_counts = [1, 2, 3, 4]
    if gate.get("required_red_counts") != required_red_counts:
        raise ValueError("Stage 1 quality gate does not require all 1/2/3/4-red scenarios")
    if gate.get("evaluated_red_counts") != required_red_counts:
        raise ValueError("Stage 1 quality gate did not evaluate all 1/2/3/4-red scenarios")
    runtime_validity = gate.get("runtime_validity")
    if not isinstance(runtime_validity, dict) or not runtime_validity or not all(
        value is True for value in runtime_validity.values()
    ):
        raise ValueError("Stage 1 quality gate runtime validity checks did not pass")
    if not isinstance(gate.get("overall"), dict) or gate["overall"].get("passed") is not True:
        raise ValueError("Stage 1 quality gate overall PN comparison did not pass")
    by_scenario = gate.get("by_scenario")
    if (
        not isinstance(by_scenario, list)
        or sorted(item.get("red_count") for item in by_scenario if isinstance(item, dict))
        != required_red_counts
        or not all(
            isinstance(item, dict) and item.get("passed") is True
            for item in by_scenario
        )
    ):
        raise ValueError("Stage 1 quality gate per-scenario PN comparisons did not pass")
    expected_sha256 = gate.get("checkpoint_sha256")
    actual_sha256 = _file_sha256(checkpoint_path)
    if expected_sha256 != actual_sha256:
        raise ValueError(
            "Stage 1 quality gate checkpoint SHA256 does not match --resume-checkpoint"
        )
    return {
        "path": str(quality_gate_path),
        "schema_version": 1,
        "passed": True,
        "policy_mode": "deterministic",
        "checkpoint_sha256": actual_sha256,
        "baseline_summary_sha256": gate.get("baseline_summary_sha256"),
        "required_red_counts": required_red_counts,
    }


def _restore_execution_policy_from_checkpoint(
    checkpoint: dict[str, Any],
    execution_actor: OverloadBiasActor,
    trainer: MAPPOTrainer,
    *,
    expected_best_score: tuple[float, ...] | None,
    learning_rate: float,
) -> dict[str, Any]:
    training_state = _training_state_from_checkpoint(checkpoint)
    saved_score = training_state.get("best_checkpoint_score")
    if expected_best_score is not None:
        if not isinstance(saved_score, list) or len(saved_score) != len(
            expected_best_score
        ):
            raise ValueError("best checkpoint score is missing or incompatible")
        if not np.allclose(
            np.asarray(saved_score, dtype=np.float64),
            np.asarray(expected_best_score, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("best checkpoint score does not match scheduler state")
    trainer_state = checkpoint.get("trainer")
    if not isinstance(trainer_state, dict) or not isinstance(
        trainer_state.get("execution_actor_optimizer"), dict
    ):
        raise ValueError("best checkpoint is missing execution actor Adam state")
    execution_actor.load_state_dict(checkpoint["execution_actor"])
    trainer.execution_actor_optimizer.load_state_dict(
        trainer_state["execution_actor_optimizer"]
    )
    for group in trainer.execution_actor_optimizer.param_groups:
        group["lr"] = float(learning_rate)
    return {
        "restored_best_iteration": training_state.get("best_iteration"),
        "restored_best_origin": training_state.get("best_checkpoint_origin"),
        "restored_execution_actor_optimizer": True,
    }


def _restore_assignment_policy_from_checkpoint(
    checkpoint: dict[str, Any],
    assignment_actor: TargetAssignmentActor,
    trainer: MAPPOTrainer,
    *,
    expected_best_score: tuple[float, ...] | None,
    learning_rate: float,
) -> dict[str, Any]:
    training_state = _training_state_from_checkpoint(checkpoint)
    saved_score = training_state.get("best_checkpoint_score")
    if expected_best_score is not None:
        if not isinstance(saved_score, list) or len(saved_score) != len(
            expected_best_score
        ):
            raise ValueError("best checkpoint score is missing or incompatible")
        if not np.allclose(
            np.asarray(saved_score, dtype=np.float64),
            np.asarray(expected_best_score, dtype=np.float64),
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError("best checkpoint score does not match scheduler state")
    trainer_state = checkpoint.get("trainer")
    if not isinstance(trainer_state, dict) or not isinstance(
        trainer_state.get("assignment_actor_optimizer"), dict
    ):
        raise ValueError("best checkpoint is missing assignment actor Adam state")
    assignment_actor.load_state_dict(checkpoint["assignment_actor"])
    trainer.assignment_actor_optimizer.load_state_dict(
        trainer_state["assignment_actor_optimizer"]
    )
    for group in trainer.assignment_actor_optimizer.param_groups:
        group["lr"] = float(learning_rate)
    return {
        "restored_best_iteration": training_state.get("best_iteration"),
        "restored_best_origin": training_state.get("best_checkpoint_origin"),
        "restored_assignment_actor_optimizer": True,
    }


def _environment_config_from_dict(data: dict[str, Any]) -> EnvironmentConfig:
    values = dict(data)
    nested = {
        "scenario": ScenarioConfig,
        "aircraft": AircraftConfig,
        "missile": MissileConfig,
        "sensor": SensorConfig,
        "reward": RewardConfig,
    }
    for key, cls in nested.items():
        if isinstance(values.get(key), dict):
            nested_values = dict(values[key])
            values[key] = cls(**nested_values)
    return EnvironmentConfig(**values)


def _configs_from_checkpoint(
    checkpoint: dict[str, Any],
    model_fallback: SwarmModelConfig,
    ppo_fallback: PPOConfig,
    env_fallback: EnvironmentConfig,
) -> tuple[SwarmModelConfig, PPOConfig, EnvironmentConfig]:
    schema_version = checkpoint.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"checkpoint schema {schema_version} is incompatible with supported schemas "
            f"{SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS}"
        )
    model_config = model_fallback
    ppo_config = ppo_fallback
    env_config = env_fallback
    if isinstance(checkpoint.get("model_config"), dict):
        model_config = SwarmModelConfig(**checkpoint["model_config"])
    if isinstance(checkpoint.get("ppo_config"), dict):
        ppo_config = PPOConfig(**checkpoint["ppo_config"])
    if isinstance(checkpoint.get("env_config"), dict):
        env_config = _environment_config_from_dict(checkpoint["env_config"])
    return model_config, ppo_config, env_config


def _training_state_from_checkpoint(checkpoint: dict[str, Any]) -> dict[str, Any]:
    state = checkpoint.get("training_state", {})
    return dict(state) if isinstance(state, dict) else {}


def _restored_best_state(
    training_state: dict[str, Any],
    *,
    reset: bool,
) -> tuple[tuple[float, ...] | None, dict[str, float] | None, str | None]:
    if reset:
        return None, None, None
    saved_score = training_state.get("best_checkpoint_score")
    score = (
        tuple(float(value) for value in saved_score)
        if isinstance(saved_score, list)
        else None
    )
    saved_metrics = training_state.get("best_checkpoint_metrics")
    metrics = (
        {
            str(key): float(value)
            for key, value in saved_metrics.items()
            if isinstance(value, (int, float, np.integer, np.floating))
            and not isinstance(value, (bool, np.bool_))
        }
        if isinstance(saved_metrics, dict)
        else None
    )
    saved_stage = training_state.get("best_checkpoint_stage")
    stage = str(saved_stage) if saved_stage is not None else None
    return score, metrics, stage


def _list_from_training_state(state: dict[str, Any], key: str) -> list[Any] | None:
    value = state.get(key)
    if isinstance(value, list) and value:
        return value
    return None


def _training_schedule(
    args: argparse.Namespace,
    env_config: EnvironmentConfig,
    resume_training_state: dict[str, Any],
) -> tuple[int, int, list[int], list[int], list[str]]:
    seed = int(resume_training_state.get("seed", args.seed)) if args.seed == DEFAULT_SEED else args.seed
    rollout_steps = args.rollout_steps
    if args.rollout_steps == DEFAULT_ROLLOUT_STEPS and "rollout_steps" in resume_training_state:
        rollout_steps = int(resume_training_state["rollout_steps"])
        rollout_step_unit = resume_training_state.get("rollout_step_unit")
        if rollout_step_unit != "assignment_decision":
            raise ValueError(f"unsupported rollout_step_unit: {rollout_step_unit}")

    red_counts = _parse_int_list(args.red_counts, env_config.scenario.red_count, "red_counts")
    saved_red_counts = _list_from_training_state(resume_training_state, "red_counts")
    if args.red_counts is None and args.red_count == DEFAULT_RED_COUNT and saved_red_counts is not None:
        red_counts = [int(item) for item in saved_red_counts]

    blue_counts = _parse_int_list(args.blue_counts, env_config.scenario.blue_count, "blue_counts")
    saved_blue_counts = _list_from_training_state(resume_training_state, "blue_counts")
    if args.blue_counts is None and args.blue_count == DEFAULT_BLUE_COUNT and saved_blue_counts is not None:
        blue_counts = [int(item) for item in saved_blue_counts]

    styles = _parse_style_list(args.styles, args.style)
    saved_styles = _list_from_training_state(resume_training_state, "styles")
    if args.styles is None and args.style == DEFAULT_STYLE and saved_styles is not None:
        styles = [str(item) for item in saved_styles]

    return seed, rollout_steps, red_counts, blue_counts, styles


def _stratified_red_counts(
    red_counts: list[int],
    batch_size: int,
    seed: int,
    optimizer_update: int,
) -> list[int]:
    if not red_counts:
        raise ValueError("red_counts must not be empty")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    unique_counts = list(dict.fromkeys(int(value) for value in red_counts))
    quotient, remainder = divmod(batch_size, len(unique_counts))
    sampled = [value for value in unique_counts for _ in range(quotient)]
    start = optimizer_update % len(unique_counts)
    sampled.extend(
        unique_counts[(start + offset) % len(unique_counts)]
        for offset in range(remainder)
    )
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, optimizer_update, 0x524544434F554E54])
    )
    rng.shuffle(sampled)
    return sampled


def _source_hashes() -> dict[str, str]:
    paths = sorted(Path("src/red_swarm_policy").rglob("*.py")) + sorted(
        Path("tests").rglob("*.py")
    )
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }


def _configuration_fingerprint(
    model_config: SwarmModelConfig,
    ppo_config: PPOConfig,
    env_config: EnvironmentConfig,
) -> str:
    payload = json.dumps(
        {
            "model_config": asdict(model_config),
            "ppo_config": asdict(ppo_config),
            "env_config": asdict(env_config),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_run_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )


def _aggregate_iteration_metrics(stats: Any, batch: Any) -> dict[str, Any]:
    grouped: dict[str, dict[str, float | int]] = {}
    if batch.scenario_red_counts is not None:
        counts = batch.scenario_red_counts.detach().cpu()
        low_active = (
            batch.episode_active_low.bool().unsqueeze(-1)
            & batch.execution_actor_inputs.agent_mask.bool()
            & batch.execution_actor_inputs.target_mask.any(dim=-1)
        )
        for red_count in sorted(set(int(value) for value in counts.tolist())):
            env_mask = batch.scenario_red_counts == red_count
            selected = batch.rewards_low[:, env_mask][low_active[:, env_mask]]
            grouped[str(red_count)] = {
                "environment_count": int(env_mask.sum().item()),
                "active_execution_agent_steps": int(selected.numel()),
                "active_low_reward_mean": (
                    float(selected.mean().detach().cpu()) if selected.numel() else 0.0
                ),
            }
    env0 = stats.per_env_final_info[0] if stats.per_env_final_info else stats.final_info
    return {
        "active_low_reward_mean": stats.active_low_reward_mean,
        "active_low_reward_nonzero_rate": stats.active_low_reward_nonzero_rate,
        "active_execution_agent_steps": stats.active_execution_agent_steps,
        "episode_high_reward_mean": stats.episode_high_reward_mean,
        "episode_low_return_mean": stats.episode_low_return_mean,
        "episode_hit_count_sum": stats.episode_hit_count_sum,
        "episode_hit_count_mean": stats.episode_hit_count_mean,
        "episode_miss_distance_mean": stats.episode_miss_distance_mean,
        "episode_miss_distance_p95": stats.episode_miss_distance_p95,
        "terminal_reason_counts": stats.terminal_reason_counts,
        "low_reward_component_sums": stats.low_reward_component_sums,
        "time_credit_unassigned_count": stats.time_credit_unassigned_count,
        "rollout_by_red_count": grouped,
        "env0_last_step_hit_count": env0.get("hit_count", 0),
        "env0_last_miss_distance_m": env0.get("miss_distance_m"),
        "env0_last_reward_components": env0.get("reward_components", {}),
    }


def _step_execution_validation_scheduler(
    trainer: MAPPOTrainer,
    state: dict[str, Any],
    *,
    improved: bool,
    lr_patience: int,
    lr_factor: float,
    min_actor_lr: float,
    early_stop_patience: int,
) -> tuple[list[dict[str, Any]], bool]:
    events: list[dict[str, Any]] = []
    if improved:
        state["no_improvement_validations"] = 0
        state["execution_lr_plateau_bad_validations"] = 0
        state["early_stop_bad_validations"] = 0
        return events, False
    early_stop_count = int(state.get("early_stop_bad_validations", 0)) + 1
    plateau_count = int(state.get("execution_lr_plateau_bad_validations", 0)) + 1
    state["no_improvement_validations"] = early_stop_count
    state["early_stop_bad_validations"] = early_stop_count
    state["execution_lr_plateau_bad_validations"] = plateau_count
    if lr_patience > 0 and plateau_count >= lr_patience:
        state["execution_lr_plateau_bad_validations"] = 0
        old_lr = float(trainer.execution_actor_optimizer.param_groups[0]["lr"])
        new_lr = max(float(min_actor_lr), old_lr * float(lr_factor))
        if new_lr < old_lr:
            for group in trainer.execution_actor_optimizer.param_groups:
                group["lr"] = new_lr
            state["execution_lr_reductions"] = int(
                state.get("execution_lr_reductions", 0)
            ) + 1
            events.append(
                {"event": "learning_rate_reduced", "old_lr": old_lr, "new_lr": new_lr}
            )
    should_stop = (
        early_stop_patience > 0 and early_stop_count >= early_stop_patience
    )
    return events, should_stop


def _step_assignment_validation_scheduler(
    trainer: MAPPOTrainer,
    state: dict[str, Any],
    *,
    improved: bool,
    lr_patience: int,
    lr_factor: float,
    min_actor_lr: float,
    early_stop_patience: int,
) -> tuple[list[dict[str, Any]], bool]:
    events: list[dict[str, Any]] = []
    if improved:
        state["no_improvement_validations"] = 0
        state["assignment_lr_plateau_bad_validations"] = 0
        state["early_stop_bad_validations"] = 0
        return events, False
    early_stop_count = int(state.get("early_stop_bad_validations", 0)) + 1
    plateau_count = int(state.get("assignment_lr_plateau_bad_validations", 0)) + 1
    state["no_improvement_validations"] = early_stop_count
    state["early_stop_bad_validations"] = early_stop_count
    state["assignment_lr_plateau_bad_validations"] = plateau_count
    if lr_patience > 0 and plateau_count >= lr_patience:
        state["assignment_lr_plateau_bad_validations"] = 0
        old_lr = float(trainer.assignment_actor_optimizer.param_groups[0]["lr"])
        new_lr = max(float(min_actor_lr), old_lr * float(lr_factor))
        if new_lr < old_lr:
            for group in trainer.assignment_actor_optimizer.param_groups:
                group["lr"] = new_lr
            state["assignment_lr_reductions"] = int(
                state.get("assignment_lr_reductions", 0)
            ) + 1
            events.append(
                {"event": "learning_rate_reduced", "old_lr": old_lr, "new_lr": new_lr}
            )
    should_stop = (
        early_stop_patience > 0 and early_stop_count >= early_stop_patience
    )
    return events, should_stop


def _iteration_training_mode(
    configured_mode: str,
    iteration: int,
    low_updates: int,
    high_updates: int,
) -> str:
    if configured_mode != "alternating":
        return configured_mode
    cycle = low_updates + high_updates
    return "low_only" if iteration % cycle < low_updates else "high_only"


def _configure_modules_for_update(
    mode: str,
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    assignment_critic: TargetAssignmentCritic,
    execution_critic: OverloadBiasCritic,
) -> dict[str, bool]:
    if mode not in {
        "joint",
        "low_only",
        "low_critic_only",
        "high_only",
        "effort_finetune",
    }:
        raise ValueError(f"unsupported update mode: {mode}")
    enabled = {
        "assignment_actor": mode in {"joint", "high_only"},
        "execution_actor": mode in {"joint", "low_only", "effort_finetune"},
        "assignment_critic": mode in {"joint", "high_only"},
        "execution_critic": mode in {"joint", "low_only", "low_critic_only"},
    }
    modules = {
        "assignment_actor": assignment_actor,
        "execution_actor": execution_actor,
        "assignment_critic": assignment_critic,
        "execution_critic": execution_critic,
    }
    for name, module in modules.items():
        trainable = enabled[name]
        module.train(trainable)
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)
    return enabled


def _save_checkpoint(
    path: Path,
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    assignment_critic: TargetAssignmentCritic,
    execution_critic: OverloadBiasCritic,
    trainer: MAPPOTrainer,
    model_config: SwarmModelConfig,
    ppo_config: PPOConfig,
    env_config: EnvironmentConfig,
    training_state: dict[str, Any],
    blue_policy: str = "rule",
    blue_evasion_config: BlueEvasionConfig = BlueEvasionConfig(),
    validation_config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "assignment_actor": assignment_actor.state_dict(),
            "execution_actor": execution_actor.state_dict(),
            "assignment_critic": assignment_critic.state_dict(),
            "execution_critic": execution_critic.state_dict(),
            "trainer": trainer.state_dict(),
            "model_config": asdict(model_config),
            "ppo_config": asdict(ppo_config),
            "env_config": asdict(env_config),
            "blue_policy": blue_policy,
            "blue_evasion_config": asdict(blue_evasion_config),
            "validation_config": {} if validation_config is None else dict(validation_config),
            "training_state": training_state,
            "rng_state": _capture_rng_state(),
        },
        path,
    )


def _restore_checkpoint(
    checkpoint: dict[str, Any],
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    assignment_critic: TargetAssignmentCritic,
    execution_critic: OverloadBiasCritic,
    trainer: MAPPOTrainer,
    device: torch.device,
    *,
    restore_assignment_critic: bool = True,
    restore_rng: bool = True,
    reset_trainer_update_step: bool = False,
) -> dict[str, Any]:
    assignment_actor.load_state_dict(checkpoint["assignment_actor"])
    execution_actor.load_state_dict(checkpoint["execution_actor"])
    if restore_assignment_critic:
        assignment_critic.load_state_dict(checkpoint["assignment_critic"])
    execution_critic.load_state_dict(checkpoint["execution_critic"])

    trainer_state = checkpoint.get("trainer")
    mode = "weights_only"
    if isinstance(trainer_state, dict):
        restored_trainer_state = dict(trainer_state)
        if not restore_assignment_critic:
            restored_trainer_state.pop("assignment_critic_optimizer", None)
        if reset_trainer_update_step:
            restored_trainer_state["update_step"] = 0
        trainer.load_state_dict(restored_trainer_state)
        mode = "full"

    training_state = _training_state_from_checkpoint(checkpoint)
    completed_iterations = int(training_state.get("completed_iterations", 0))
    restored_rng = False
    if (
        restore_rng
        and mode == "full"
        and isinstance(checkpoint.get("rng_state"), dict)
    ):
        restored_rng = _restore_rng_state(checkpoint["rng_state"], device)

    return {
        "mode": mode,
        "completed_iterations": completed_iterations,
        "restored_rng": restored_rng,
        "restored_assignment_critic": restore_assignment_critic,
        "reset_trainer_update_step": reset_trainer_update_step,
    }


def _parse_int_list(value: str | None, fallback: int, name: str) -> list[int]:
    if value is None or value.strip() == "":
        return [fallback]
    items = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not items or any(item <= 0 for item in items):
        raise ValueError(f"{name} must contain positive integers")
    return items


def _parse_style_list(value: str | None, fallback: str) -> list[str]:
    if value is None or value.strip() == "":
        return [fallback]
    styles = [part.strip() for part in value.split(",") if part.strip()]
    invalid = sorted(set(styles) - set(SCENARIO_STYLES))
    if invalid:
        raise ValueError(f"invalid styles: {invalid}")
    return styles


def _validation_plan(
    training_mode: str,
    red_counts: list[int],
    blue_counts: list[int],
    assignment_mode_override: str | None = None,
) -> tuple[list[tuple[str, int, int]], str]:
    if assignment_mode_override is not None and assignment_mode_override not in ASSIGNMENT_MODES:
        raise ValueError(
            f"unsupported validation assignment mode: {assignment_mode_override}"
        )
    if training_mode in {"low_only", "low_critic_only"}:
        scenarios = [
            (
                "many_to_one" if blue_count == 1 else "many_to_many",
                red_count,
                blue_count,
            )
            for red_count in dict.fromkeys(red_counts)
            for blue_count in dict.fromkeys(blue_counts)
        ]
        return scenarios, assignment_mode_override or "capacity_aware"
    return [
        ("many_to_many", 24, blue_count)
        for blue_count in (4, 5, 6)
    ], assignment_mode_override or "actor"


def _summarize_validation_values(
    values: list[tuple[float, float, float, float, float]],
    max_guidance_time_s: float,
) -> dict[str, float]:
    full_success = np.asarray([value[0] for value in values], dtype=np.float64)
    damage = np.asarray([value[1] for value in values], dtype=np.float64)
    ineffective = np.asarray([value[2] for value in values], dtype=np.float64)
    success_times = [value[3] for value in values if value[0] > 0.5]
    return {
        "trial_count": float(len(values)),
        "full_success_rate": float(full_success.mean()),
        "average_damage_rate": float(damage.mean()),
        "ineffective_loss_rate": float(ineffective.mean()),
        "successful_completion_time_s": (
            float(np.mean(success_times))
            if success_times
            else float(max_guidance_time_s)
        ),
        "control_effort": float(np.mean([value[4] for value in values])),
    }


def _checkpoint_selection_metrics(envs: list[RedBlueEngagementEnv]) -> dict[str, float]:
    rows: list[tuple[float, float, float, float]] = []
    for env in envs:
        if env.state is None:
            raise RuntimeError("environment state is unavailable after rollout")
        red_count = max(len(env.state.red), 1)
        blue_count = max(len(env.state.blue), 1)
        destroyed_blue = sum(not target.alive for target in env.state.blue)
        lost_red = sum(not missile.alive for missile in env.state.red)
        damage_rate = destroyed_blue / blue_count
        full_success = float(destroyed_blue == blue_count)
        ineffective_loss = max(lost_red - destroyed_blue, 0) / red_count
        success_time = env.state.time_s if full_success else env.config.missile.max_guidance_time_s
        rows.append((full_success, damage_rate, ineffective_loss, success_time))
    return {
        "full_success_rate": float(np.mean([row[0] for row in rows])),
        "average_damage_rate": float(np.mean([row[1] for row in rows])),
        "ineffective_loss_rate": float(np.mean([row[2] for row in rows])),
        "successful_completion_time_s": float(np.mean([row[3] for row in rows if row[0] > 0.5]))
        if any(row[0] > 0.5 for row in rows)
        else float(envs[0].config.missile.max_guidance_time_s),
    }


def _fixed_validation_metrics(
    env_config: EnvironmentConfig,
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    blue_policy_mode: str,
    blue_evasion_config: BlueEvasionConfig,
    *,
    training_mode: str,
    red_counts: list[int],
    blue_counts: list[int],
    seed_start: int,
    trials_per_blue_count: int,
    envs: list[RedBlueEngagementEnv] | None = None,
    env_pool: ProcessEnvironmentPool | None = None,
    assignment_mode_override: str | None = None,
    assignment_deterministic: bool = True,
    execution_deterministic: bool = True,
) -> dict[str, Any]:
    from .validate_checkpoint import _run_trial

    if trials_per_blue_count <= 0:
        raise ValueError("validation trials_per_blue_count must be positive")
    scenarios, assignment_mode = _validation_plan(
        training_mode,
        red_counts,
        blue_counts,
        assignment_mode_override,
    )
    for _, red_count, blue_count in scenarios:
        env_config.reward.validate_lexicographic_priority(red_count, blue_count)
    if env_pool is not None:
        if envs is None or len(envs) != env_pool.size:
            raise ValueError("parallel validation requires envs matching env_pool")
        values: list[tuple[float, float, float, float, float]] = []
        by_scenario: list[dict[str, Any]] = []
        trial_index = 0
        max_assignment_steps = (
            env_config.policy_horizon_steps + env_config.assignment_update_steps - 1
        ) // env_config.assignment_update_steps
        for style, red_count, blue_count in scenarios:
            scenario_values: list[tuple[float, float, float, float, float]] = []
            remaining = trials_per_blue_count
            while remaining > 0:
                active_trials = min(remaining, env_pool.size)
                seeds = [
                    seed_start + trial_index + index
                    for index in range(active_trials)
                ]
                seeds.extend([seeds[-1]] * (env_pool.size - active_trials))
                evaluated = evaluate_parallel_episodes(
                    envs,
                    env_pool,
                    assignment_actor,
                    execution_actor,
                    seeds=seeds,
                    style=style,
                    red_count=red_count,
                    blue_count=blue_count,
                    max_assignment_steps=max_assignment_steps,
                    deterministic=True,
                    assignment_mode=assignment_mode,
                    assignment_deterministic=assignment_deterministic,
                    execution_deterministic=execution_deterministic,
                )
                scenario_values.extend(
                    (
                        row.full_success,
                        row.damage_rate,
                        row.ineffective_loss_rate,
                        row.completion_time_s,
                        row.control_effort,
                    )
                    for row in evaluated[:active_trials]
                )
                trial_index += active_trials
                remaining -= active_trials
            values.extend(scenario_values)
            by_scenario.append(
                {
                    "style": style,
                    "red_count": red_count,
                    "blue_count": blue_count,
                    "lexicographic_priority_valid": True,
                    **_summarize_validation_values(
                        scenario_values,
                        env_config.missile.max_guidance_time_s,
                    ),
                }
            )
        return {
            **_summarize_validation_values(values, env_config.missile.max_guidance_time_s),
            "assignment_mode": assignment_mode,
            "by_scenario": by_scenario,
        }
    env = RedBlueEngagementEnv(
        env_config,
        blue_policy=_build_blue_policy(env_config, blue_policy_mode, blue_evasion_config),
        device="cpu",
        record_replay=False,
    )
    values: list[tuple[float, float, float, float, float]] = []
    by_scenario = []
    trial_index = 0
    for style, red_count, blue_count in scenarios:
        scenario_values = []
        for _ in range(trials_per_blue_count):
            row = _run_trial(
                env,
                assignment_actor,
                execution_actor,
                seed=seed_start + trial_index,
                style=style,
                red_count=red_count,
                blue_count=blue_count,
                max_steps=env_config.policy_horizon_steps,
                deterministic=True,
                assignment_mode=assignment_mode,
                assignment_deterministic=assignment_deterministic,
                execution_deterministic=execution_deterministic,
            )
            full_success = float(row.hit_count >= row.blue_count)
            scenario_values.append(
                (
                    full_success,
                    row.red_success_rate,
                    row.ineffective_loss_rate,
                    row.task_completion_time_s
                    if full_success > 0.5
                    else float(env_config.missile.max_guidance_time_s),
                    row.control_effort,
                )
            )
            trial_index += 1
        values.extend(scenario_values)
        by_scenario.append(
            {
                "style": style,
                "red_count": red_count,
                "blue_count": blue_count,
                "lexicographic_priority_valid": True,
                **_summarize_validation_values(
                    scenario_values,
                    env_config.missile.max_guidance_time_s,
                ),
            }
        )
    return {
        **_summarize_validation_values(values, env_config.missile.max_guidance_time_s),
        "assignment_mode": assignment_mode,
        "by_scenario": by_scenario,
    }


def _selection_score(metrics: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        metrics["full_success_rate"],
        metrics["average_damage_rate"],
        -metrics["ineffective_loss_rate"],
        -metrics["successful_completion_time_s"],
        -metrics["control_effort"],
    )


def _effort_candidate_improves(
    candidate: dict[str, float],
    baseline: dict[str, float] | None,
) -> bool:
    if baseline is None:
        return True
    preserves_higher_priorities = (
        candidate["full_success_rate"] >= baseline["full_success_rate"]
        and candidate["average_damage_rate"] >= baseline["average_damage_rate"]
        and candidate["ineffective_loss_rate"] <= baseline["ineffective_loss_rate"]
        and candidate["successful_completion_time_s"]
        <= baseline["successful_completion_time_s"] + 0.1
    )
    return (
        preserves_higher_priorities
        and candidate["control_effort"] < baseline["control_effort"]
    )


def _rollout_policy_diagnostics(batch: Any) -> dict[str, Any]:
    high_active = batch.episode_active_high.bool()
    high_agent_active = batch.assignment_actor_inputs.agent_mask.bool()
    high_mask = high_active.unsqueeze(-1) & high_agent_active
    targets = batch.assignment_actions.target
    no_target_ratio = (
        float((targets[high_mask] == 0).float().mean().cpu())
        if high_mask.any()
        else 0.0
    )
    previous_targets = batch.assignment_actor_inputs.current_assignment.argmax(dim=-1)
    target_switch_rate = (
        float((targets[high_mask] != previous_targets[high_mask]).float().mean().cpu())
        if high_mask.any()
        else 0.0
    )
    physical_target_count = batch.assignment_actor_inputs.target_mask.shape[-1] - 1
    assignment_counts = []
    for target_slot in range(1, physical_target_count + 1):
        per_transition = ((targets == target_slot) & high_mask).sum(dim=-1)
        selected = per_transition[high_active]
        assignment_counts.append(float(selected.float().mean().cpu()) if selected.numel() else 0.0)

    low_active = (
        batch.episode_active_low.bool().unsqueeze(-1)
        & batch.execution_actor_inputs.agent_mask.bool()
        & batch.execution_actor_inputs.target_mask.any(dim=-1)
    )
    action = batch.bias_matrices
    action_norm = torch.linalg.vector_norm(action, dim=-1)
    bias_g = torch.clamp(action_norm, max=1.0) * 5.0
    selected_bias = bias_g[low_active]
    bias_rms_g = (
        float(torch.sqrt(selected_bias.pow(2).mean()).cpu())
        if selected_bias.numel()
        else 0.0
    )
    bias_p95_g = (
        float(torch.quantile(selected_bias, 0.95).cpu())
        if selected_bias.numel()
        else 0.0
    )
    bias_saturation_rate = (
        float((selected_bias >= 4.95).float().mean().cpu())
        if selected_bias.numel()
        else 0.0
    )
    self_state = batch.execution_actor_inputs.self_state
    seeker_modes = self_state[..., 10:13]
    mode_names = ("locked", "lock_hold", "inertial")
    mode_fraction = {
        name: (
            float(seeker_modes[..., index][low_active].mean().cpu())
            if low_active.any()
            else 0.0
        )
        for index, name in enumerate(mode_names)
    }
    zem_m = batch.execution_actor_inputs.assigned_target[..., 0, 14] * 50000.0
    return {
        "no_target_ratio": no_target_ratio,
        "target_switch_rate": target_switch_rate,
        "mean_assignment_count_by_target": assignment_counts,
        "bias_rms_g": bias_rms_g,
        "bias_p95_g": bias_p95_g,
        "bias_saturation_rate_4_95g": bias_saturation_rate,
        "seeker_mode_fraction": mode_fraction,
        "mean_zem_m": float(zem_m[low_active].mean().cpu()) if low_active.any() else 0.0,
    }


def train(args: argparse.Namespace) -> int:
    device, device_label = _select_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    resume_checkpoint = _load_torch_checkpoint(Path(args.resume_checkpoint)) if args.resume_checkpoint else None
    stage1_quality_gate: dict[str, Any] | None = None
    if args.reset_best_on_resume and resume_checkpoint is None:
        raise ValueError("reset_best_on_resume requires a resume checkpoint")
    if args.validation_only:
        if resume_checkpoint is None:
            raise ValueError("validation_only requires a resume checkpoint")
        if args.iterations != 0:
            raise ValueError("validation_only requires iterations == 0")
    if args.validation_assignment_stochastic:
        if not args.validation_only:
            raise ValueError(
                "validation_assignment_stochastic requires validation_only"
            )
        if args.validation_assignment_mode != "actor":
            raise ValueError(
                "validation_assignment_stochastic requires "
                "validation_assignment_mode=actor"
            )
        if args.validation_policy_seed is None:
            raise ValueError(
                "validation_assignment_stochastic requires validation_policy_seed"
            )
    elif args.validation_policy_seed is not None:
        raise ValueError(
            "validation_policy_seed requires validation_assignment_stochastic"
        )
    if args.validation_policy_seed is not None and args.validation_policy_seed < 0:
        raise ValueError("validation_policy_seed must be non-negative")
    resume_training_state: dict[str, Any] = (
        _training_state_from_checkpoint(resume_checkpoint)
        if resume_checkpoint is not None
        else {}
    )
    source_training_mode = (
        str(resume_training_state.get("training_mode", "unknown"))
        if resume_checkpoint is not None
        else None
    )
    stage_transition_requested = (
        args.training_mode == "high_only" and source_training_mode == "low_only"
    )
    model_config = _build_model_config(args)
    ppo_config = _build_ppo_config(args)
    requested_assignment_scale = args.assignment_reward_learning_scale
    requested_execution_scale = args.execution_reward_learning_scale
    requested_execution_value_loss = args.execution_value_loss
    requested_action_distribution = args.execution_action_distribution
    requested_critic_value_head_mode = args.critic_value_head_mode
    requested_assignment_stickiness = args.assignment_stickiness_logit_bonus
    requested_low_time_credit_mode = args.low_time_credit_mode
    requested_low_option_boundary_potential = args.low_option_boundary_potential
    requested_execution_advantage_normalization = (
        args.execution_advantage_normalization
    )
    requested_execution_actor_loss_weighting = args.execution_actor_loss_weighting
    requested_terminal_success_reward = args.terminal_success_reward
    requested_high_potential_weight = args.high_potential_weight
    requested_assignment_entropy_coef = args.assignment_entropy_coef
    env_config = _build_env_config(args)
    blue_policy_mode = args.blue_policy
    blue_evasion_config = _build_blue_evasion_config(args)
    if env_config.policy_start_mode != "post_boost":
        raise ValueError("training requires policy_start_mode='post_boost'")
    stage_transition = False
    reinitialize_assignment_critic = False
    if resume_checkpoint is not None:
        saved_model = resume_checkpoint.get("model_config", {})
        if isinstance(saved_model, dict):
            saved_action_distribution = str(
                saved_model.get("execution_action_distribution", "tanh_box")
            )
            saved_critic_mode = str(
                saved_model.get("critic_value_head_mode", "latent_sum")
            )
            saved_assignment_stickiness = float(
                saved_model.get("assignment_stickiness_logit_bonus", 0.0)
            )
            if (
                requested_action_distribution is not None
                and requested_action_distribution != saved_action_distribution
            ):
                raise ValueError(
                    "full resume cannot override execution_action_distribution; "
                    "start a fresh run"
                )
            if (
                requested_critic_value_head_mode is not None
                and requested_critic_value_head_mode != saved_critic_mode
            ):
                raise ValueError(
                    "full resume cannot override critic_value_head_mode; start a fresh run"
                )
            if (
                requested_assignment_stickiness is not None
                and not np.isclose(
                    requested_assignment_stickiness,
                    saved_assignment_stickiness,
                )
                and not stage_transition_requested
            ):
                raise ValueError(
                    "full resume cannot override assignment_stickiness_logit_bonus; "
                    "start a fresh run or a low_only-to-high_only stage transition"
                )
        saved_env = resume_checkpoint.get("env_config", {})
        saved_reward = saved_env.get("reward", {}) if isinstance(saved_env, dict) else {}
        if isinstance(saved_reward, dict):
            saved_time_mode = str(saved_reward.get("low_time_credit_mode", "none"))
            saved_boundary_mode = str(
                saved_reward.get("low_option_boundary_potential", "exempt")
            )
            if (
                requested_low_time_credit_mode is not None
                and requested_low_time_credit_mode != saved_time_mode
            ):
                raise ValueError(
                    "full resume cannot override low_time_credit_mode; start a fresh run"
                )
            if (
                requested_low_option_boundary_potential is not None
                and requested_low_option_boundary_potential != saved_boundary_mode
            ):
                raise ValueError(
                    "full resume cannot override low_option_boundary_potential; "
                    "start a fresh run"
                )
            saved_terminal_success_reward = float(
                saved_reward.get("terminal_success_reward", 0.0)
            )
            saved_high_potential_weight = float(
                saved_reward.get("high_potential_weight", 1.0)
            )
            if (
                requested_terminal_success_reward is not None
                and not np.isclose(
                    requested_terminal_success_reward,
                    saved_terminal_success_reward,
                )
                and not stage_transition_requested
            ):
                raise ValueError(
                    "full resume cannot override terminal_success_reward; "
                    "start a fresh run or a low_only-to-high_only stage transition"
                )
            if (
                requested_high_potential_weight is not None
                and not np.isclose(
                    requested_high_potential_weight,
                    saved_high_potential_weight,
                )
                and not stage_transition_requested
            ):
                raise ValueError(
                    "full resume cannot override high_potential_weight; "
                    "start a fresh run or a low_only-to-high_only stage transition"
                )
        saved_ppo = resume_checkpoint.get("ppo_config", {})
        if isinstance(saved_ppo, dict):
            saved_assignment_scale = float(
                saved_ppo.get("assignment_reward_learning_scale", 1.0)
            )
            saved_assignment_entropy_coef = float(
                saved_ppo.get("assignment_entropy_coef", 0.01)
            )
            saved_scale = float(saved_ppo.get("execution_reward_learning_scale", 1.0))
            saved_loss = str(saved_ppo.get("execution_value_loss", "mse"))
            saved_advantage_normalization = str(
                saved_ppo.get("execution_advantage_normalization", "global")
            )
            saved_actor_loss_weighting = str(
                saved_ppo.get("execution_actor_loss_weighting", "active_step")
            )
            if (
                requested_assignment_scale is not None
                and not np.isclose(
                    requested_assignment_scale,
                    saved_assignment_scale,
                )
                and not stage_transition_requested
            ):
                raise ValueError(
                    "full resume cannot override assignment_reward_learning_scale; "
                    "start a fresh run or a low_only-to-high_only stage transition"
                )
            if (
                requested_assignment_entropy_coef is not None
                and not np.isclose(
                    requested_assignment_entropy_coef,
                    saved_assignment_entropy_coef,
                )
                and not stage_transition_requested
            ):
                raise ValueError(
                    "full resume cannot override assignment_entropy_coef; "
                    "start a fresh run or a low_only-to-high_only stage transition"
                )
            if (
                requested_execution_scale is not None
                and not np.isclose(requested_execution_scale, saved_scale)
            ):
                raise ValueError(
                    "full resume cannot override execution_reward_learning_scale; "
                    "start a fresh run for a different learning scale"
                )
            if (
                requested_execution_value_loss is not None
                and requested_execution_value_loss != saved_loss
            ):
                raise ValueError(
                    "full resume cannot override execution_value_loss; start a fresh run"
                )
            if (
                requested_execution_advantage_normalization is not None
                and requested_execution_advantage_normalization
                != saved_advantage_normalization
            ):
                raise ValueError(
                    "full resume cannot override execution_advantage_normalization; "
                    "start a fresh run"
                )
            if (
                requested_execution_actor_loss_weighting is not None
                and requested_execution_actor_loss_weighting
                != saved_actor_loss_weighting
            ):
                raise ValueError(
                    "full resume cannot override execution_actor_loss_weighting; "
                    "start a fresh run"
                )
        model_config, ppo_config, env_config = _configs_from_checkpoint(resume_checkpoint, model_config, ppo_config, env_config)
        if stage_transition_requested:
            if requested_assignment_stickiness is not None:
                model_config = replace(
                    model_config,
                    assignment_stickiness_logit_bonus=(
                        requested_assignment_stickiness
                    ),
                )
            if requested_assignment_scale is not None:
                ppo_config = replace(
                    ppo_config,
                    assignment_reward_learning_scale=requested_assignment_scale,
                )
            if requested_assignment_entropy_coef is not None:
                ppo_config = replace(
                    ppo_config,
                    assignment_entropy_coef=requested_assignment_entropy_coef,
                )
            if requested_terminal_success_reward is not None:
                env_config = replace(
                    env_config,
                    reward=replace(
                        env_config.reward,
                        terminal_success_reward=requested_terminal_success_reward,
                    ),
                )
            if requested_high_potential_weight is not None:
                env_config = replace(
                    env_config,
                    reward=replace(
                        env_config.reward,
                        high_potential_weight=requested_high_potential_weight,
                    ),
                )
        blue_policy_mode = str(resume_checkpoint.get("blue_policy", blue_policy_mode))
        saved_blue_config = resume_checkpoint.get("blue_evasion_config")
        if isinstance(saved_blue_config, dict):
            blue_evasion_config = BlueEvasionConfig(**saved_blue_config)
        requires_stage1_gate = stage_transition_requested
        stage_transition = requires_stage1_gate
        if args.stage1_quality_gate:
            stage1_quality_gate = _validate_stage1_quality_gate(
                Path(args.resume_checkpoint),
                Path(args.stage1_quality_gate),
            )
        elif requires_stage1_gate:
            raise ValueError(
                "transitioning from low_only to high_only requires "
                "--stage1-quality-gate from deterministic paired PN holdout"
            )
        if requires_stage1_gate and not args.reset_best_on_resume:
            raise ValueError(
                "transitioning from low_only to high_only requires "
                "--reset-best-on-resume so Stage 1 validation state cannot leak into Stage 2"
            )
        if requires_stage1_gate and (
            model_config.effective_assignment_critic_value_head_mode
            != "latent_sum"
        ):
            model_config = replace(
                model_config,
                assignment_critic_value_head_mode="latent_sum",
            )
            reinitialize_assignment_critic = True
    if (
        args.training_mode == "high_only"
        and model_config.effective_assignment_critic_value_head_mode != "latent_sum"
    ):
        raise ValueError(
            "high_only training requires a five-component latent-sum assignment critic"
        )
    if model_config.max_missiles_per_target != env_config.scenario.max_missiles_per_target:
        raise ValueError("model and environment max_missiles_per_target must match")

    assignment_actor = TargetAssignmentActor(model_config).to(device)
    execution_actor = OverloadBiasActor(model_config).to(device)
    assignment_critic = TargetAssignmentCritic(model_config).to(device)
    execution_critic = OverloadBiasCritic(model_config).to(device)
    trainer = MAPPOTrainer(
        assignment_actor,
        execution_actor,
        assignment_critic,
        execution_critic,
        ppo_config,
    )
    envs = [
        RedBlueEngagementEnv(
            env_config,
            blue_policy=_build_blue_policy(env_config, blue_policy_mode, blue_evasion_config),
            device="cpu",
            record_replay=False,
        )
        for _ in range(args.parallel_envs)
    ]
    start_iteration = 0
    if resume_checkpoint is not None:
        resume_info = _restore_checkpoint(
            resume_checkpoint,
            assignment_actor,
            execution_actor,
            assignment_critic,
            execution_critic,
            trainer,
            device,
            restore_assignment_critic=not reinitialize_assignment_critic,
            restore_rng=not stage_transition,
            reset_trainer_update_step=stage_transition,
        )
        source_completed_iterations = int(resume_info["completed_iterations"])
        start_iteration = 0 if stage_transition else source_completed_iterations
        print(
            json.dumps(
                {
                    "event": "resume",
                    "path": str(args.resume_checkpoint),
                    "mode": resume_info["mode"],
                    "source_completed_iterations": source_completed_iterations,
                    "start_iteration": start_iteration,
                    "completed_iterations": start_iteration,
                    "restored_rng": resume_info["restored_rng"],
                    "restored_assignment_critic": resume_info[
                        "restored_assignment_critic"
                    ],
                    "reset_trainer_update_step": resume_info[
                        "reset_trainer_update_step"
                    ],
                    "reset_best_on_resume": args.reset_best_on_resume,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    seed, rollout_steps, red_counts, blue_counts, styles = _training_schedule(args, env_config, resume_training_state)
    scenario_sampling = args.scenario_sampling
    if scenario_sampling is None:
        scenario_sampling = str(resume_training_state.get("scenario_sampling", "cyclic"))
    if scenario_sampling not in {"cyclic", "random"}:
        raise ValueError(f"unsupported scenario_sampling: {scenario_sampling}")
    red_count_batch_mode = args.red_count_batch_mode
    if red_count_batch_mode is None:
        red_count_batch_mode = str(
            resume_training_state.get("red_count_batch_mode", "homogeneous")
        )
    if red_count_batch_mode not in {"homogeneous", "stratified"}:
        raise ValueError(f"unsupported red_count_batch_mode: {red_count_batch_mode}")

    saved_stage_origin = resume_training_state.get("stage_origin")
    stage_origin = (
        dict(saved_stage_origin)
        if isinstance(saved_stage_origin, dict)
        else None
    )
    if stage_transition:
        source_optimizer_updates = int(
            resume_training_state.get(
                "completed_optimizer_updates", source_completed_iterations
            )
        )
        source_policy_updates = int(
            resume_training_state.get(
                "completed_policy_updates", source_completed_iterations
            )
        )
        stage_origin = {
            "source_checkpoint": str(args.resume_checkpoint),
            "source_training_mode": source_training_mode,
            "source_completed_iterations": source_completed_iterations,
            "source_completed_optimizer_updates": source_optimizer_updates,
            "source_completed_policy_updates": source_policy_updates,
            "source_completed_stage_policy_updates": int(
                resume_training_state.get(
                    "completed_stage_policy_updates", source_policy_updates
                )
            ),
        }
    stage_transition_metadata = (
        {
            **stage_origin,
            "target_training_mode": "high_only",
            "assignment_critic_reinitialized": reinitialize_assignment_critic,
            "assignment_critic_value_head_mode": (
                model_config.effective_assignment_critic_value_head_mode
            ),
            "checkpoint_rng_restored": False,
            "trainer_update_step_reset": True,
            "stage_counters_reset": True,
        }
        if stage_transition and stage_origin is not None
        else None
    )

    inherit_stage_control = resume_checkpoint is not None and not args.reset_best_on_resume
    stage_control_state = resume_training_state if inherit_stage_control else {}
    low_critic_warmup_updates = int(
        resume_training_state.get(
            "low_critic_warmup_updates", args.low_critic_warmup_updates
        )
        if inherit_stage_control
        else args.low_critic_warmup_updates
    )
    low_critic_warmup_critic_steps = int(
        resume_training_state.get(
            "low_critic_warmup_critic_steps_per_update",
            args.low_critic_warmup_critic_steps_per_update,
        )
        if inherit_stage_control
        else args.low_critic_warmup_critic_steps_per_update
    )
    lr_plateau_patience = int(
        resume_training_state.get(
            "execution_lr_plateau_patience", args.execution_lr_plateau_patience
        )
        if inherit_stage_control
        else args.execution_lr_plateau_patience
    )
    lr_plateau_factor = float(
        resume_training_state.get(
            "execution_lr_plateau_factor", args.execution_lr_plateau_factor
        )
        if inherit_stage_control
        else args.execution_lr_plateau_factor
    )
    assignment_lr_plateau_patience = int(
        resume_training_state.get(
            "assignment_lr_plateau_patience",
            args.assignment_lr_plateau_patience,
        )
        if inherit_stage_control
        else args.assignment_lr_plateau_patience
    )
    assignment_lr_plateau_factor = float(
        resume_training_state.get(
            "assignment_lr_plateau_factor",
            args.assignment_lr_plateau_factor,
        )
        if inherit_stage_control
        else args.assignment_lr_plateau_factor
    )
    restore_best_on_lr_reduction = bool(
        resume_training_state.get(
            "execution_restore_best_on_lr_reduction",
            args.execution_restore_best_on_lr_reduction,
        )
        if inherit_stage_control
        else args.execution_restore_best_on_lr_reduction
    )
    assignment_restore_best_on_lr_reduction = bool(
        resume_training_state.get(
            "assignment_restore_best_on_lr_reduction",
            args.assignment_restore_best_on_lr_reduction,
        )
        if inherit_stage_control
        else args.assignment_restore_best_on_lr_reduction
    )
    restore_best_on_early_stop = bool(
        resume_training_state.get(
            "execution_restore_best_on_early_stop",
            args.execution_restore_best_on_early_stop,
        )
        if inherit_stage_control
        else args.execution_restore_best_on_early_stop
    )
    assignment_restore_best_on_early_stop = bool(
        resume_training_state.get(
            "assignment_restore_best_on_early_stop",
            args.assignment_restore_best_on_early_stop,
        )
        if inherit_stage_control
        else args.assignment_restore_best_on_early_stop
    )
    early_stop_patience = int(
        resume_training_state.get(
            "early_stop_validation_patience", args.early_stop_validation_patience
        )
        if inherit_stage_control
        else args.early_stop_validation_patience
    )
    default_min_actor_lr = float(
        ppo_config.actor_learning_rate
        or ppo_config.execution_actor_learning_rate
    )
    min_actor_lr = float(
        resume_training_state.get(
            "execution_min_actor_learning_rate",
            args.execution_min_actor_learning_rate or default_min_actor_lr,
        )
        if inherit_stage_control
        else args.execution_min_actor_learning_rate or default_min_actor_lr
    )
    default_assignment_min_actor_lr = float(
        ppo_config.actor_learning_rate
        or ppo_config.assignment_actor_learning_rate
    )
    assignment_min_actor_lr = float(
        resume_training_state.get(
            "assignment_min_actor_learning_rate",
            args.assignment_min_actor_learning_rate
            or default_assignment_min_actor_lr,
        )
        if inherit_stage_control
        else args.assignment_min_actor_learning_rate
        or default_assignment_min_actor_lr
    )
    for scheduled_red_count in red_counts:
        for scheduled_blue_count in blue_counts:
            env_config.reward.validate_lexicographic_priority(
                scheduled_red_count,
                scheduled_blue_count,
            )
    best_score, best_metrics, best_checkpoint_stage = _restored_best_state(
        resume_training_state,
        reset=args.reset_best_on_resume,
    )
    completed_optimizer_updates = (
        0
        if stage_transition
        else int(
            resume_training_state.get("completed_optimizer_updates", start_iteration)
        )
    )
    completed_policy_updates = (
        0
        if stage_transition
        else int(
            resume_training_state.get("completed_policy_updates", start_iteration)
        )
    )
    completed_stage_policy_updates = (
        int(
            resume_training_state.get(
                "completed_stage_policy_updates", completed_policy_updates
            )
        )
        if inherit_stage_control
        else 0
    )
    legacy_bad_validations = int(
        resume_training_state.get("no_improvement_validations", 0)
        if inherit_stage_control
        else 0
    )
    scheduler_state: dict[str, Any] = {
        "no_improvement_validations": legacy_bad_validations,
        "execution_lr_plateau_bad_validations": int(
            stage_control_state.get(
                "execution_lr_plateau_bad_validations",
                legacy_bad_validations % lr_plateau_patience
                if lr_plateau_patience > 0
                else legacy_bad_validations,
            )
        ),
        "assignment_lr_plateau_bad_validations": int(
            stage_control_state.get(
                "assignment_lr_plateau_bad_validations",
                legacy_bad_validations % assignment_lr_plateau_patience
                if assignment_lr_plateau_patience > 0
                else legacy_bad_validations,
            )
        ),
        "early_stop_bad_validations": int(
            stage_control_state.get(
                "early_stop_bad_validations", legacy_bad_validations
            )
        ),
        "execution_lr_reductions": int(
            stage_control_state.get("execution_lr_reductions", 0)
        ),
        "assignment_lr_reductions": int(
            stage_control_state.get("assignment_lr_reductions", 0)
        ),
        "execution_policy_restorations": int(
            stage_control_state.get("execution_policy_restorations", 0)
        ),
        "execution_terminal_policy_restorations": int(
            stage_control_state.get("execution_terminal_policy_restorations", 0)
        ),
        "assignment_policy_restorations": int(
            stage_control_state.get("assignment_policy_restorations", 0)
        ),
        "assignment_terminal_policy_restorations": int(
            stage_control_state.get("assignment_terminal_policy_restorations", 0)
        ),
    }
    best_iteration = (
        resume_training_state.get("best_iteration")
        if inherit_stage_control
        else None
    )
    best_checkpoint_origin = (
        resume_training_state.get("best_checkpoint_origin")
        if inherit_stage_control
        else None
    )
    last_validation_policy_update = int(
        resume_training_state.get("last_validation_policy_update", 0)
        if inherit_stage_control
        else 0
    )
    if args.reset_best_on_resume:
        print(
            json.dumps(
                {
                    "event": "best_checkpoint_baseline_reset",
                    "training_mode": args.training_mode,
                    "weights_and_optimizers_restored": resume_checkpoint is not None,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    validation_assignment_mode_override = (
        None
        if args.validation_assignment_mode == "auto"
        else str(args.validation_assignment_mode)
    )
    validation_assignment_deterministic = not bool(
        args.validation_assignment_stochastic
    )
    validation_execution_deterministic = True
    validation_policy_seed = (
        int(args.validation_policy_seed)
        if args.validation_policy_seed is not None
        else None
    )
    validation_interval = args.validation_interval
    validation_seed_start = args.validation_seed_start
    validation_trials_per_blue_count = args.validation_trials_per_blue_count
    validation_parallel_envs = (
        args.validation_parallel_envs or args.parallel_envs
        if args.parallel_backend == "process"
        else 1
    )
    saved_validation_config = None if resume_checkpoint is None else resume_checkpoint.get("validation_config")
    if (
        isinstance(saved_validation_config, dict)
        and not args.reset_best_on_resume
        and not args.validation_only
    ):
        validation_interval = int(saved_validation_config.get("interval", validation_interval))
        validation_seed_start = int(saved_validation_config.get("seed_start", validation_seed_start))
        validation_trials_per_blue_count = int(
            saved_validation_config.get("trials_per_blue_count", validation_trials_per_blue_count)
        )
        validation_parallel_envs = int(
            saved_validation_config.get("parallel_envs", validation_parallel_envs)
        )
    low_only_validation_scenarios, low_only_assignment_mode = _validation_plan(
        "low_only",
        red_counts,
        blue_counts,
        validation_assignment_mode_override,
    )
    final_validation_scenarios, final_assignment_mode = _validation_plan(
        "full",
        red_counts,
        blue_counts,
        validation_assignment_mode_override,
    )
    validation_config = {
        "interval": validation_interval,
        "seed_start": validation_seed_start,
        "trials_per_blue_count": validation_trials_per_blue_count,
        "trials_per_scenario": validation_trials_per_blue_count,
        "parallel_envs": validation_parallel_envs,
        "assignment_mode_override": validation_assignment_mode_override,
        "assignment_deterministic": validation_assignment_deterministic,
        "execution_deterministic": validation_execution_deterministic,
        "policy_seed": validation_policy_seed,
        "validation_only": bool(args.validation_only),
        "waves_per_scenario": int(
            np.ceil(validation_trials_per_blue_count / validation_parallel_envs)
        ),
        "actual_simulations_per_scenario": int(
            np.ceil(validation_trials_per_blue_count / validation_parallel_envs)
            * validation_parallel_envs
        ),
        "low_only": {
            "assignment_mode": low_only_assignment_mode,
            "scenarios": [
                {"style": style, "red_count": red_count, "blue_count": blue_count}
                for style, red_count, blue_count in low_only_validation_scenarios
            ],
            "trial_count": len(low_only_validation_scenarios) * validation_trials_per_blue_count,
        },
        "full_engagement": {
            "assignment_mode": final_assignment_mode,
            "scenarios": [
                {"style": style, "red_count": red_count, "blue_count": blue_count}
                for style, red_count, blue_count in final_validation_scenarios
            ],
            "trial_count": len(final_validation_scenarios) * validation_trials_per_blue_count,
        },
        "max_guidance_time_s": env_config.missile.max_guidance_time_s,
    }
    latest_checkpoint = Path(args.latest_checkpoint) if args.latest_checkpoint else None
    best_checkpoint = Path(args.best_checkpoint or args.checkpoint) if (args.best_checkpoint or args.checkpoint) else None
    config_fingerprint = _configuration_fingerprint(
        model_config,
        ppo_config,
        env_config,
    )
    if args.run_manifest_path:
        run_manifest_path = Path(args.run_manifest_path)
    elif args.metrics_path:
        run_manifest_path = Path(args.metrics_path).with_name("run_manifest.json")
    else:
        run_manifest_path = Path("outputs/run_manifest.json")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint_schema_version": CHECKPOINT_SCHEMA_VERSION,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "ended_at": None,
        "stop_reason": None,
        "command_line": [sys.executable, *sys.argv],
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_device": str(device),
        "device_label": device_label,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "seed": seed,
        "validation_seed_start": validation_seed_start,
        "resume_checkpoint": args.resume_checkpoint,
        "resume_mode": None if resume_checkpoint is None else "full",
        "stage1_quality_gate": stage1_quality_gate,
        "stage_origin": stage_origin,
        "stage_transition": stage_transition_metadata,
        "model_config": asdict(model_config),
        "ppo_config": asdict(ppo_config),
        "env_config": asdict(env_config),
        "validation_config": validation_config,
        "training_control": {
            "iterations": args.iterations,
            "low_critic_warmup_updates": low_critic_warmup_updates,
            "low_critic_warmup_critic_steps_per_update": (
                low_critic_warmup_critic_steps
            ),
            "red_count_batch_mode": red_count_batch_mode,
            "execution_lr_plateau_patience": lr_plateau_patience,
            "execution_lr_plateau_factor": lr_plateau_factor,
            "execution_min_actor_learning_rate": min_actor_lr,
            "assignment_lr_plateau_patience": assignment_lr_plateau_patience,
            "assignment_lr_plateau_factor": assignment_lr_plateau_factor,
            "assignment_min_actor_learning_rate": assignment_min_actor_lr,
            "execution_restore_best_on_lr_reduction": (
                restore_best_on_lr_reduction
            ),
            "execution_restore_best_on_early_stop": restore_best_on_early_stop,
            "assignment_restore_best_on_lr_reduction": (
                assignment_restore_best_on_lr_reduction
            ),
            "assignment_restore_best_on_early_stop": (
                assignment_restore_best_on_early_stop
            ),
            "early_stop_validation_patience": early_stop_patience,
        },
        "config_fingerprint": config_fingerprint,
        "source_sha256": _source_hashes(),
    }
    _write_run_manifest(run_manifest_path, manifest)

    print(json.dumps({"event": "device", "device": device_label}, ensure_ascii=True), flush=True)
    print(
        json.dumps(
            {
                "event": "experiment_config",
                "red_counts": red_counts,
                "blue_counts": blue_counts,
                "styles": styles,
                "scenario_sampling": scenario_sampling,
                "red_count_batch_mode": red_count_batch_mode,
                "rollout_assignment_steps": rollout_steps,
                "rollout_step_unit": "assignment_decision",
                "iterations": args.iterations,
                "start_iteration": start_iteration,
                "training_mode": args.training_mode,
                "reset_best_on_resume": args.reset_best_on_resume,
                "inherited_best_checkpoint_stage": best_checkpoint_stage,
                "alternating_low_updates": args.alternating_low_updates,
                "alternating_high_updates": args.alternating_high_updates,
                "seed": seed,
                "parallel_cpu_envs": args.parallel_envs,
                "parallel_backend": args.parallel_backend,
                "env_worker_threads": args.env_worker_threads,
                "env_worker_timeout_s": args.env_worker_timeout_s,
                "assignment_sequence_length": ppo_config.high_sequence_length,
                "execution_sequence_length": ppo_config.low_sequence_length,
                "network_batch_size": args.parallel_envs,
                "blue_policy": blue_policy_mode,
                "blue_evasion_config": asdict(blue_evasion_config),
                "max_missiles_per_target": env_config.scenario.max_missiles_per_target,
                "max_guidance_time_s": env_config.missile.max_guidance_time_s,
                "validation_config": validation_config,
                "low_critic_warmup_updates": low_critic_warmup_updates,
                "low_critic_warmup_critic_steps_per_update": (
                    low_critic_warmup_critic_steps
                ),
                "validation_parallel_envs": validation_parallel_envs,
                "completed_optimizer_updates": completed_optimizer_updates,
                "completed_policy_updates": completed_policy_updates,
                "completed_stage_policy_updates": completed_stage_policy_updates,
                "execution_lr_plateau_patience": lr_plateau_patience,
                "execution_lr_plateau_factor": lr_plateau_factor,
                "execution_min_actor_learning_rate": min_actor_lr,
                "assignment_lr_plateau_patience": (
                    assignment_lr_plateau_patience
                ),
                "assignment_lr_plateau_factor": assignment_lr_plateau_factor,
                "assignment_min_actor_learning_rate": assignment_min_actor_lr,
                "execution_restore_best_on_lr_reduction": (
                    restore_best_on_lr_reduction
                ),
                "execution_restore_best_on_early_stop": (
                    restore_best_on_early_stop
                ),
                "assignment_restore_best_on_lr_reduction": (
                    assignment_restore_best_on_lr_reduction
                ),
                "assignment_restore_best_on_early_stop": (
                    assignment_restore_best_on_early_stop
                ),
                "early_stop_validation_patience": early_stop_patience,
                "stage1_quality_gate": stage1_quality_gate,
                "config_fingerprint": config_fingerprint,
                "run_manifest_path": str(run_manifest_path),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    env_pool = None
    if args.parallel_backend == "process" and args.parallel_envs > 1:
        env_pool = ProcessEnvironmentPool(
            envs,
            native_threads=args.env_worker_threads,
            timeout_s=args.env_worker_timeout_s,
        )
        print(
            json.dumps(
                {
                    "event": "environment_workers",
                    "backend": "process",
                    "count": env_pool.size,
                    "pids": [worker.pid for worker in env_pool.worker_info],
                    "torch_threads_per_worker": [
                        worker.torch_threads for worker in env_pool.worker_info
                    ],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    validation_envs = envs
    validation_env_pool = env_pool
    if (
        args.parallel_backend == "process"
        and validation_parallel_envs != args.parallel_envs
    ):
        validation_envs = [
            RedBlueEngagementEnv(
                env_config,
                blue_policy=_build_blue_policy(
                    env_config,
                    blue_policy_mode,
                    blue_evasion_config,
                ),
                device="cpu",
                record_replay=False,
            )
            for _ in range(validation_parallel_envs)
        ]
        validation_env_pool = ProcessEnvironmentPool(
            validation_envs,
            native_threads=args.env_worker_threads,
            timeout_s=args.env_worker_timeout_s,
        )
        print(
            json.dumps(
                {
                    "event": "validation_environment_workers",
                    "backend": "process",
                    "count": validation_env_pool.size,
                    "pids": [
                        worker.pid for worker in validation_env_pool.worker_info
                    ],
                    "torch_threads_per_worker": [
                        worker.torch_threads
                        for worker in validation_env_pool.worker_info
                    ],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
    if args.validation_only:
        if validation_policy_seed is not None:
            _seed_validation_policy_rng(validation_policy_seed, device)
        try:
            validation_metrics = _fixed_validation_metrics(
                env_config,
                assignment_actor,
                execution_actor,
                blue_policy_mode,
                blue_evasion_config,
                training_mode=args.training_mode,
                red_counts=red_counts,
                blue_counts=blue_counts,
                seed_start=validation_seed_start,
                trials_per_blue_count=validation_trials_per_blue_count,
                envs=validation_envs,
                env_pool=validation_env_pool,
                assignment_mode_override=validation_assignment_mode_override,
                assignment_deterministic=validation_assignment_deterministic,
                execution_deterministic=validation_execution_deterministic,
            )
        finally:
            if validation_env_pool is not None and validation_env_pool is not env_pool:
                validation_env_pool.close()
            if env_pool is not None:
                env_pool.close()
        assignment_policy_mode = (
            "capacity_aware"
            if validation_metrics["assignment_mode"] == "capacity_aware"
            else (
                "deterministic"
                if validation_assignment_deterministic
                else "stochastic"
            )
        )
        execution_policy_mode = (
            "deterministic"
            if validation_execution_deterministic
            else "stochastic"
        )
        policy_mode = (
            "deterministic"
            if assignment_policy_mode in {"capacity_aware", "deterministic"}
            and execution_policy_mode == "deterministic"
            else f"assignment_{assignment_policy_mode}_execution_{execution_policy_mode}"
        )
        validation_report = {
            "event": "validation_only",
            "schema_version": 1,
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "checkpoint": args.resume_checkpoint,
            "checkpoint_sha256": _file_sha256(Path(args.resume_checkpoint)),
            "training_mode": args.training_mode,
            "policy_mode": policy_mode,
            "assignment_policy_mode": assignment_policy_mode,
            "execution_policy_mode": execution_policy_mode,
            "validation_policy_seed": validation_policy_seed,
            "red_counts": red_counts,
            "blue_counts": blue_counts,
            "validation_config": validation_config,
            "checkpoint_selection_score": list(_selection_score(validation_metrics)),
            "config_fingerprint": config_fingerprint,
            **validation_metrics,
        }
        print(json.dumps(validation_report, ensure_ascii=True), flush=True)
        if args.metrics_path:
            metrics_path = Path(args.metrics_path)
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(
                json.dumps(validation_report, indent=2, ensure_ascii=True),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {"event": "metrics", "path": str(metrics_path)},
                    ensure_ascii=True,
                ),
                flush=True,
            )
        manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
        manifest["stop_reason"] = "validation_only"
        manifest["completed_iterations"] = start_iteration
        manifest["validation_only"] = validation_report
        _write_run_manifest(run_manifest_path, manifest)
        return 0
    metric_rows: list[dict[str, object]] = []
    stop_reason = "max_iterations"
    completed_iterations = start_iteration
    for local_iteration in range(args.iterations):
        iteration = start_iteration + local_iteration
        sampled_red_counts: list[int] | None = None
        sampled_blue_counts: list[int] | None = None
        if scenario_sampling == "random":
            scenario_rng = np.random.default_rng(
                np.random.SeedSequence([seed, iteration, 0x535441474532])
            )
            style = str(scenario_rng.choice(styles))
            sampled_blue_counts = [
                int(value)
                for value in scenario_rng.choice(
                    blue_counts,
                    size=args.parallel_envs,
                    replace=True,
                )
            ]
            blue_count = None
        else:
            blue_count = blue_counts[iteration % len(blue_counts)]
            style = styles[iteration % len(styles)]
        if red_count_batch_mode == "stratified":
            sampled_red_counts = _stratified_red_counts(
                red_counts,
                args.parallel_envs,
                seed,
                completed_optimizer_updates,
            )
            red_count = None
        elif scenario_sampling == "random":
            red_count = int(scenario_rng.choice(red_counts))
        else:
            red_count = red_counts[iteration % len(red_counts)]
        in_critic_warmup = completed_optimizer_updates < low_critic_warmup_updates
        update_mode = (
            "low_critic_only"
            if in_critic_warmup
            else _iteration_training_mode(
                args.training_mode,
                completed_stage_policy_updates,
                args.alternating_low_updates,
                args.alternating_high_updates,
            )
        )
        trainable_modules = _configure_modules_for_update(
            update_mode,
            assignment_actor,
            execution_actor,
            assignment_critic,
            execution_critic,
        )
        batch, stats = collect_parallel_rollout(
            envs,
            assignment_actor,
            execution_actor,
            assignment_critic,
            execution_critic,
            rollout_steps,
            seed=seed + iteration,
            style=style,
            red_count=red_count,
            blue_count=blue_count,
            red_counts=sampled_red_counts,
            blue_counts=sampled_blue_counts,
            deterministic=False,
            assignment_mode=(
                "capacity_aware"
                if update_mode in {"low_only", "low_critic_only"}
                else "actor"
            ),
            assignment_deterministic=update_mode == "effort_finetune",
            execution_deterministic=update_mode == "high_only",
            env_pool=env_pool,
        )
        last_metrics: dict[str, float] = trainer.update(
            batch,
            mode=update_mode,
            critic_steps_override=(
                low_critic_warmup_critic_steps
                if update_mode == "low_critic_only"
                and low_critic_warmup_critic_steps > 0
                else None
            ),
        )
        completed_optimizer_updates += 1
        policy_updated = (
            update_mode != "low_critic_only"
            and last_metrics["actor_updates"] > 0.0
        )
        if policy_updated:
            completed_policy_updates += 1
            completed_stage_policy_updates += 1
        completed_iterations = iteration + 1
        warmup_completed_now = (
            update_mode == "low_critic_only"
            and completed_optimizer_updates == low_critic_warmup_updates
        )
        rollout_selection = _checkpoint_selection_metrics(envs)
        fixed_validation = None
        selection_score = None
        validation_due = warmup_completed_now or (
            policy_updated
            and validation_interval > 0
            and completed_stage_policy_updates % validation_interval == 0
            and completed_stage_policy_updates != last_validation_policy_update
        )
        if validation_due:
            fixed_validation = _fixed_validation_metrics(
                env_config,
                assignment_actor,
                execution_actor,
                blue_policy_mode,
                blue_evasion_config,
                training_mode=update_mode,
                red_counts=red_counts,
                blue_counts=blue_counts,
                seed_start=validation_seed_start,
                trials_per_blue_count=validation_trials_per_blue_count,
                envs=validation_envs,
                env_pool=validation_env_pool,
                assignment_mode_override=validation_assignment_mode_override,
            )
            selection_score = _selection_score(fixed_validation)
            if policy_updated:
                last_validation_policy_update = completed_stage_policy_updates
        best_candidate_origin = (
            "warmup_pn_baseline" if warmup_completed_now else "policy_update"
        )
        eligible_for_best = policy_updated or warmup_completed_now
        improved = False
        if eligible_for_best and selection_score is not None and fixed_validation is not None:
            improved = (
                _effort_candidate_improves(fixed_validation, best_metrics)
                if update_mode == "effort_finetune"
                else best_score is None or selection_score > best_score
            )
        if improved:
            best_score = selection_score
            best_metrics = fixed_validation
            best_checkpoint_stage = args.training_mode
            best_iteration = completed_iterations
            best_checkpoint_origin = best_candidate_origin
        scheduler_events: list[dict[str, Any]] = []
        early_stop = False
        if fixed_validation is not None and policy_updated:
            scheduler_policy = (
                "assignment" if update_mode == "high_only" else "execution"
            )
            if scheduler_policy == "assignment":
                scheduler_events, early_stop = _step_assignment_validation_scheduler(
                    trainer,
                    scheduler_state,
                    improved=improved,
                    lr_patience=assignment_lr_plateau_patience,
                    lr_factor=assignment_lr_plateau_factor,
                    min_actor_lr=assignment_min_actor_lr,
                    early_stop_patience=early_stop_patience,
                )
            else:
                scheduler_events, early_stop = _step_execution_validation_scheduler(
                    trainer,
                    scheduler_state,
                    improved=improved,
                    lr_patience=lr_plateau_patience,
                    lr_factor=lr_plateau_factor,
                    min_actor_lr=min_actor_lr,
                    early_stop_patience=early_stop_patience,
                )
            for event in scheduler_events:
                event["policy"] = scheduler_policy
                should_restore_on_reduction = (
                    assignment_restore_best_on_lr_reduction
                    if scheduler_policy == "assignment"
                    else restore_best_on_lr_reduction
                )
                if event.get("event") == "learning_rate_reduced" and should_restore_on_reduction:
                    if best_checkpoint is None or not best_checkpoint.exists():
                        raise RuntimeError(
                            f"cannot restore {scheduler_policy} policy: "
                            "best checkpoint is unavailable"
                        )
                    if scheduler_policy == "assignment":
                        restore_info = _restore_assignment_policy_from_checkpoint(
                            _load_torch_checkpoint(best_checkpoint),
                            assignment_actor,
                            trainer,
                            expected_best_score=best_score,
                            learning_rate=float(event["new_lr"]),
                        )
                        scheduler_state["assignment_policy_restorations"] += 1
                    else:
                        restore_info = _restore_execution_policy_from_checkpoint(
                            _load_torch_checkpoint(best_checkpoint),
                            execution_actor,
                            trainer,
                            expected_best_score=best_score,
                            learning_rate=float(event["new_lr"]),
                        )
                        scheduler_state["execution_policy_restorations"] += 1
                    event.update(
                        {
                            f"{scheduler_policy}_policy_restored": True,
                            "restored_from": str(best_checkpoint),
                            **restore_info,
                        }
                    )
            for event in scheduler_events:
                print(
                    json.dumps(
                        {
                            **event,
                            "iteration": completed_iterations,
                            "completed_policy_updates": completed_policy_updates,
                            "completed_stage_policy_updates": (
                                completed_stage_policy_updates
                            ),
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
            should_restore_on_early_stop = (
                assignment_restore_best_on_early_stop
                if scheduler_policy == "assignment"
                else restore_best_on_early_stop
            )
            if early_stop and should_restore_on_early_stop:
                if best_checkpoint is None or not best_checkpoint.exists():
                    raise RuntimeError(
                        f"cannot restore {scheduler_policy} policy on early stop: "
                        "best checkpoint is unavailable"
                    )
                if scheduler_policy == "assignment":
                    current_lr = float(
                        trainer.assignment_actor_optimizer.param_groups[0]["lr"]
                    )
                    restore_info = _restore_assignment_policy_from_checkpoint(
                        _load_torch_checkpoint(best_checkpoint),
                        assignment_actor,
                        trainer,
                        expected_best_score=best_score,
                        learning_rate=current_lr,
                    )
                    scheduler_state["assignment_policy_restorations"] += 1
                    scheduler_state[
                        "assignment_terminal_policy_restorations"
                    ] += 1
                else:
                    current_lr = float(
                        trainer.execution_actor_optimizer.param_groups[0]["lr"]
                    )
                    restore_info = _restore_execution_policy_from_checkpoint(
                        _load_torch_checkpoint(best_checkpoint),
                        execution_actor,
                        trainer,
                        expected_best_score=best_score,
                        learning_rate=current_lr,
                    )
                    scheduler_state["execution_policy_restorations"] += 1
                    scheduler_state[
                        "execution_terminal_policy_restorations"
                    ] += 1
                print(
                    json.dumps(
                        {
                            "event": "early_stop_best_restored",
                            "policy": scheduler_policy,
                            f"{scheduler_policy}_policy_restored": True,
                            "restored_from": str(best_checkpoint),
                            f"retained_{scheduler_policy}_actor_learning_rate": current_lr,
                            "iteration": completed_iterations,
                            "completed_policy_updates": completed_policy_updates,
                            "completed_stage_policy_updates": (
                                completed_stage_policy_updates
                            ),
                            **restore_info,
                        },
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
        aggregate_metrics = _aggregate_iteration_metrics(stats, batch)
        policy_diagnostics = _rollout_policy_diagnostics(batch)
        row = {
            "event": "iteration",
            "iteration": iteration + 1,
            "style": style,
            "red_count": red_count,
            "blue_count": blue_count,
            "sampled_red_counts": sampled_red_counts,
            "sampled_blue_counts": sampled_blue_counts,
            "scenario_sampling": scenario_sampling,
            "red_count_batch_mode": red_count_batch_mode,
            "training_mode": update_mode,
            "trainable_modules": trainable_modules,
            "rollout_assignment_steps": stats.steps,
            "rollout_execution_steps": stats.execution_steps,
            "done": stats.done,
            "total_reward_high": stats.total_reward_high,
            "mean_reward_low": stats.mean_reward_low,
            "hit_count": aggregate_metrics["env0_last_step_hit_count"],
            "miss_distance_m": aggregate_metrics["env0_last_miss_distance_m"],
            "reward_components": aggregate_metrics["env0_last_reward_components"],
            "deprecated_metric_fields": {
                "mean_reward_low": "unmasked tensor mean; use active_low_reward_mean",
                "hit_count": "env0 last step; use episode_hit_count_sum/mean",
                "miss_distance_m": "env0 last value; use episode_miss_distance_mean/p95",
                "reward_components": "env0 last value; use per-env aggregates",
                "timescale_ratio": "optimizer ratio; use explicit ratio fields",
                "execution_advantage_std": "raw std compatibility alias",
            },
            "rollout_diagnostics": rollout_selection,
            "policy_diagnostics": policy_diagnostics,
            "fixed_validation": fixed_validation,
            "checkpoint_selection_score": None if selection_score is None else list(selection_score),
            "completed_optimizer_updates": completed_optimizer_updates,
            "completed_policy_updates": completed_policy_updates,
            "completed_stage_policy_updates": completed_stage_policy_updates,
            "policy_updated": policy_updated,
            "best_candidate_origin": best_candidate_origin,
            "best_iteration": best_iteration,
            "best_checkpoint_origin": best_checkpoint_origin,
            "no_improvement_validations": scheduler_state[
                "no_improvement_validations"
            ],
            "execution_lr_plateau_bad_validations": scheduler_state[
                "execution_lr_plateau_bad_validations"
            ],
            "assignment_lr_plateau_bad_validations": scheduler_state[
                "assignment_lr_plateau_bad_validations"
            ],
            "early_stop_bad_validations": scheduler_state[
                "early_stop_bad_validations"
            ],
            "execution_lr_reductions": scheduler_state["execution_lr_reductions"],
            "assignment_lr_reductions": scheduler_state[
                "assignment_lr_reductions"
            ],
            "execution_policy_restorations": scheduler_state[
                "execution_policy_restorations"
            ],
            "execution_terminal_policy_restorations": scheduler_state[
                "execution_terminal_policy_restorations"
            ],
            "assignment_policy_restorations": scheduler_state[
                "assignment_policy_restorations"
            ],
            "assignment_terminal_policy_restorations": scheduler_state[
                "assignment_terminal_policy_restorations"
            ],
            "assignment_actor_learning_rate": trainer.assignment_actor_optimizer.param_groups[0]["lr"],
            "execution_actor_learning_rate": trainer.execution_actor_optimizer.param_groups[0]["lr"],
            "assignment_critic_learning_rate": trainer.assignment_critic_optimizer.param_groups[0]["lr"],
            "execution_critic_learning_rate": trainer.execution_critic_optimizer.param_groups[0]["lr"],
            "config_fingerprint": config_fingerprint,
            **aggregate_metrics,
            **last_metrics,
        }
        metric_rows.append(row)
        print(json.dumps(row, ensure_ascii=True), flush=True)
        current_training_state = {
            "completed_iterations": completed_iterations,
            "completed_optimizer_updates": completed_optimizer_updates,
            "completed_policy_updates": completed_policy_updates,
            "completed_stage_policy_updates": completed_stage_policy_updates,
            "stage_origin": stage_origin,
            "seed": seed,
            "rollout_steps": rollout_steps,
            "rollout_step_unit": "assignment_decision",
            "rollout_step_duration_s": env_config.assignment_update_interval_s,
            "parallel_envs": args.parallel_envs,
            "parallel_backend": args.parallel_backend,
            "env_worker_threads": args.env_worker_threads,
            "red_counts": red_counts,
            "blue_counts": blue_counts,
            "styles": styles,
            "scenario_sampling": scenario_sampling,
            "red_count_batch_mode": red_count_batch_mode,
            "training_mode": args.training_mode,
            "alternating_low_updates": args.alternating_low_updates,
            "alternating_high_updates": args.alternating_high_updates,
            "blue_policy": blue_policy_mode,
            "blue_evasion_config": asdict(blue_evasion_config),
            "best_checkpoint_score": None if best_score is None else list(best_score),
            "best_checkpoint_metrics": best_metrics,
            "best_checkpoint_stage": best_checkpoint_stage,
            "best_iteration": best_iteration,
            "best_checkpoint_origin": best_checkpoint_origin,
            "last_validation_policy_update": last_validation_policy_update,
            "low_critic_warmup_updates": low_critic_warmup_updates,
            "low_critic_warmup_critic_steps_per_update": (
                low_critic_warmup_critic_steps
            ),
            "execution_lr_plateau_patience": lr_plateau_patience,
            "execution_lr_plateau_factor": lr_plateau_factor,
            "execution_min_actor_learning_rate": min_actor_lr,
            "assignment_lr_plateau_patience": assignment_lr_plateau_patience,
            "assignment_lr_plateau_factor": assignment_lr_plateau_factor,
            "assignment_min_actor_learning_rate": assignment_min_actor_lr,
            "execution_restore_best_on_lr_reduction": (
                restore_best_on_lr_reduction
            ),
            "execution_restore_best_on_early_stop": restore_best_on_early_stop,
            "assignment_restore_best_on_lr_reduction": (
                assignment_restore_best_on_lr_reduction
            ),
            "assignment_restore_best_on_early_stop": (
                assignment_restore_best_on_early_stop
            ),
            "early_stop_validation_patience": early_stop_patience,
            **scheduler_state,
        }
        if latest_checkpoint is not None:
            _save_checkpoint(
                latest_checkpoint,
                assignment_actor,
                execution_actor,
                assignment_critic,
                execution_critic,
                trainer,
                model_config,
                ppo_config,
                env_config,
                current_training_state,
                blue_policy_mode,
                blue_evasion_config,
                validation_config,
            )
            print(json.dumps({"event": "latest_checkpoint", "path": str(latest_checkpoint)}, ensure_ascii=True), flush=True)
        if improved and best_checkpoint is not None:
            _save_checkpoint(
                best_checkpoint,
                assignment_actor,
                execution_actor,
                assignment_critic,
                execution_critic,
                trainer,
                model_config,
                ppo_config,
                env_config,
                current_training_state,
                blue_policy_mode,
                blue_evasion_config,
                validation_config,
            )
            print(
                json.dumps(
                    {
                        "event": "best_checkpoint",
                        "path": str(best_checkpoint),
                        "score": list(best_score),
                        "origin": best_checkpoint_origin,
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
        if args.checkpoint_interval > 0 and (iteration + 1) % args.checkpoint_interval == 0:
            base = latest_checkpoint or best_checkpoint
            if base is not None:
                periodic = base.parent / f"iteration_{iteration + 1:06d}.pt"
                _save_checkpoint(
                    periodic,
                    assignment_actor,
                    execution_actor,
                    assignment_critic,
                    execution_critic,
                    trainer,
                    model_config,
                    ppo_config,
                    env_config,
                    current_training_state,
                    blue_policy_mode,
                    blue_evasion_config,
                    validation_config,
                )
                print(json.dumps({"event": "periodic_checkpoint", "path": str(periodic)}, ensure_ascii=True), flush=True)
        if early_stop:
            stop_reason = "early_stop_validation_patience"
            print(
                json.dumps(
                    {
                        "event": "early_stop",
                        "iteration": completed_iterations,
                        "completed_policy_updates": completed_policy_updates,
                        "completed_stage_policy_updates": (
                            completed_stage_policy_updates
                        ),
                        "no_improvement_validations": scheduler_state[
                            "no_improvement_validations"
                        ],
                    },
                    ensure_ascii=True,
                ),
                flush=True,
            )
            break

    if validation_env_pool is not None and validation_env_pool is not env_pool:
        validation_env_pool.close()
    if env_pool is not None:
        env_pool.close()
    training_state = {
        "completed_iterations": completed_iterations,
        "completed_optimizer_updates": completed_optimizer_updates,
        "completed_policy_updates": completed_policy_updates,
        "completed_stage_policy_updates": completed_stage_policy_updates,
        "stage_origin": stage_origin,
        "seed": seed,
        "rollout_steps": rollout_steps,
        "rollout_step_unit": "assignment_decision",
        "rollout_step_duration_s": env_config.assignment_update_interval_s,
        "red_counts": red_counts,
        "blue_counts": blue_counts,
        "styles": styles,
        "scenario_sampling": scenario_sampling,
        "red_count_batch_mode": red_count_batch_mode,
        "training_mode": args.training_mode,
        "alternating_low_updates": args.alternating_low_updates,
        "alternating_high_updates": args.alternating_high_updates,
        "parallel_envs": args.parallel_envs,
        "parallel_backend": args.parallel_backend,
        "env_worker_threads": args.env_worker_threads,
        "blue_policy": blue_policy_mode,
        "blue_evasion_config": asdict(blue_evasion_config),
        "best_checkpoint_score": None if best_score is None else list(best_score),
        "best_checkpoint_metrics": best_metrics,
        "best_checkpoint_stage": best_checkpoint_stage,
        "best_iteration": best_iteration,
        "best_checkpoint_origin": best_checkpoint_origin,
        "last_validation_policy_update": last_validation_policy_update,
        "low_critic_warmup_updates": low_critic_warmup_updates,
        "low_critic_warmup_critic_steps_per_update": (
            low_critic_warmup_critic_steps
        ),
        "execution_lr_plateau_patience": lr_plateau_patience,
        "execution_lr_plateau_factor": lr_plateau_factor,
        "execution_min_actor_learning_rate": min_actor_lr,
        "assignment_lr_plateau_patience": assignment_lr_plateau_patience,
        "assignment_lr_plateau_factor": assignment_lr_plateau_factor,
        "assignment_min_actor_learning_rate": assignment_min_actor_lr,
        "execution_restore_best_on_lr_reduction": restore_best_on_lr_reduction,
        "execution_restore_best_on_early_stop": restore_best_on_early_stop,
        "assignment_restore_best_on_lr_reduction": (
            assignment_restore_best_on_lr_reduction
        ),
        "assignment_restore_best_on_early_stop": (
            assignment_restore_best_on_early_stop
        ),
        "early_stop_validation_patience": early_stop_patience,
        "stop_reason": stop_reason,
        **scheduler_state,
    }
    if args.metrics_path:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "model_config": asdict(model_config),
                    "ppo_config": asdict(ppo_config),
                    "env_config": asdict(env_config),
                    "blue_policy": blue_policy_mode,
                    "blue_evasion_config": asdict(blue_evasion_config),
                    "validation_config": validation_config,
                    "parallel_backend": args.parallel_backend,
                    "env_worker_threads": args.env_worker_threads,
                    "scenario_sampling": scenario_sampling,
                    "red_count_batch_mode": red_count_batch_mode,
                    "start_iteration": start_iteration,
                    "completed_iterations": completed_iterations,
                    "completed_optimizer_updates": completed_optimizer_updates,
                    "completed_policy_updates": completed_policy_updates,
                    "completed_stage_policy_updates": completed_stage_policy_updates,
                    "stop_reason": stop_reason,
                    "training_state": training_state,
                    "config_fingerprint": config_fingerprint,
                    "run_manifest_path": str(run_manifest_path),
                    "resume_checkpoint": args.resume_checkpoint,
                    "stage1_quality_gate": stage1_quality_gate,
                    "stage_origin": stage_origin,
                    "stage_transition": manifest["stage_transition"],
                    "iterations": metric_rows,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )
        print(json.dumps({"event": "metrics", "path": str(metrics_path)}, ensure_ascii=True), flush=True)
    manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
    manifest["stop_reason"] = stop_reason
    manifest["completed_iterations"] = completed_iterations
    manifest["completed_optimizer_updates"] = completed_optimizer_updates
    manifest["completed_policy_updates"] = completed_policy_updates
    manifest["completed_stage_policy_updates"] = completed_stage_policy_updates
    manifest["best_iteration"] = best_iteration
    manifest["final_execution_actor_learning_rate"] = float(
        trainer.execution_actor_optimizer.param_groups[0]["lr"]
    )
    _write_run_manifest(run_manifest_path, manifest)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the red swarm policy on RedBlueEngagementEnv with MAPPO.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--training-mode",
        choices=("low_only", "high_only", "alternating", "joint", "effort_finetune"),
        default="alternating",
    )
    parser.add_argument("--alternating-low-updates", type=int, default=20)
    parser.add_argument("--alternating-high-updates", type=int, default=10)
    parser.add_argument(
        "--reset-best-on-resume",
        action="store_true",
        help="Clear inherited best score/metrics after restoring model and optimizer state.",
    )
    parser.add_argument(
        "--stage1-quality-gate",
        default=None,
        help=(
            "Stage 1 learned-checkpoint holdout summary. Required when a low_only "
            "checkpoint initializes high_only training."
        ),
    )
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument(
        "--parallel-backend",
        choices=("process", "thread"),
        default="process",
        help="CPU environment execution backend; process bypasses the CPython GIL.",
    )
    parser.add_argument("--env-worker-threads", type=int, default=1)
    parser.add_argument("--env-worker-timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=DEFAULT_ROLLOUT_STEPS,
        help="Number of target-assignment decision steps per iteration (5.0 s by default).",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--style", choices=SCENARIO_STYLES, default=DEFAULT_STYLE)
    parser.add_argument("--styles", default=None, help="Comma-separated scenario styles; overrides cycling from --style.")
    parser.add_argument("--red-count", type=int, default=DEFAULT_RED_COUNT)
    parser.add_argument("--red-counts", default=None, help="Comma-separated red missile counts sampled cyclically.")
    parser.add_argument("--blue-count", type=int, default=DEFAULT_BLUE_COUNT)
    parser.add_argument("--blue-counts", default=None, help="Comma-separated blue target counts sampled cyclically.")
    parser.add_argument(
        "--scenario-sampling",
        choices=("cyclic", "random"),
        default=None,
        help=(
            "Scenario-count schedule. random samples one configured blue count independently "
            "for every parallel environment and is reproducible from seed/iteration."
        ),
    )
    parser.add_argument(
        "--red-count-batch-mode",
        choices=("homogeneous", "stratified"),
        default=None,
        help="Use one red count per update or a reproducible within-batch stratified mix.",
    )
    parser.add_argument("--max-missiles-per-target", type=int, default=4)
    parser.add_argument("--time-step-s", type=float, default=0.005)
    parser.add_argument("--bias-update-interval-s", type=float, default=0.1)
    parser.add_argument("--assignment-update-interval-s", type=float, default=5.0)
    parser.add_argument("--max-steps", type=int, default=36000)
    parser.add_argument("--policy-start-mode", choices=("post_boost", "launch"), default="post_boost")
    parser.add_argument("--policy-entry-speed-tolerance-ratio", type=float, default=1.0e-6)
    parser.add_argument("--policy-entry-flight-path-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--red-launch-mach-range", default="0.6,0.9")
    parser.add_argument("--red-altitude-range-m", default="8000,10000")
    parser.add_argument("--blue-speed-range-mps", default="300,400")
    parser.add_argument("--blue-altitude-range-m", default="8000,12000")
    parser.add_argument("--blue-cluster-center-ne-m", default="0,0")
    parser.add_argument("--blue-cluster-radius-m", type=float, default=20000.0)
    parser.add_argument("--blue-heading-range-deg", default="-180,180")
    parser.add_argument("--red-cluster-radius-range-m", default="140000,160000")
    parser.add_argument("--red-sector-center-azimuth-deg", type=float, default=180.0)
    parser.add_argument("--red-sector-width-deg", type=float, default=60.0)
    parser.add_argument("--red-heading-bias-max-deg", type=float, default=15.0)
    parser.add_argument("--position-perturb-m", type=float, default=0.0)
    parser.add_argument("--velocity-perturb-mps", type=float, default=0.0)
    parser.add_argument("--aircraft-min-speed-mps", type=float, default=100.0)
    parser.add_argument("--aircraft-max-speed-mps", type=float, default=600.0)
    parser.add_argument("--aircraft-min-altitude-m", type=float, default=8000.0)
    parser.add_argument("--aircraft-max-altitude-m", type=float, default=12000.0)
    parser.add_argument("--missile-dry-mass-kg", type=float, default=120.0)
    parser.add_argument("--missile-propellant-mass-kg", type=float, default=45.0)
    parser.add_argument("--missile-boost-time-s", type=float, default=7.0)
    parser.add_argument("--missile-max-mach", type=float, default=6.0)
    parser.add_argument("--reference-speed-of-sound-mps", type=float, default=295.0)
    parser.add_argument("--missile-boost-climb-angle-deg", type=float, default=20.0)
    parser.add_argument("--missile-boost-pitch-transition-s", type=float, default=2.0)
    parser.add_argument("--missile-boost-pitch-tracking-gain", type=float, default=2.0)
    parser.add_argument("--missile-reference-area-m2", type=float, default=0.028)
    parser.add_argument("--missile-drag-coefficient", type=float, default=None)
    parser.add_argument(
        "--missile-drag-mach-breakpoints",
        default="0,0.8,0.95,1.05,1.2,2,3,4,5,6,8",
    )
    parser.add_argument(
        "--missile-zero-lift-drag-coefficients",
        default="0.10,0.11,0.18,0.34,0.30,0.22,0.19,0.17,0.16,0.15,0.15",
    )
    parser.add_argument("--missile-induced-drag-factor", type=float, default=0.08)
    parser.add_argument("--missile-max-load-factor-g", type=float, default=35.0)
    parser.add_argument("--missile-max-guidance-bias-g", type=float, default=5.0)
    parser.add_argument("--proportional-navigation-gain", type=float, default=3.5)
    parser.add_argument("--missile-max-guidance-time-s", type=float, default=180.0)
    parser.add_argument("--seeker-acquisition-fov-deg", type=float, default=35.0)
    parser.add_argument("--seeker-tracking-fov-deg", type=float, default=60.0)
    parser.add_argument("--fov-break-hold-s", type=float, default=0.75)
    parser.add_argument("--post-closest-growth-m", type=float, default=600.0)
    parser.add_argument("--post-closest-recede-speed-mps", type=float, default=40.0)
    parser.add_argument("--lethal-radius-m", type=float, default=5.0)
    parser.add_argument("--detection-range-m", type=float, default=200000.0)
    parser.add_argument("--communication-delay-steps", type=int, default=0)
    parser.add_argument("--blue-policy", choices=("rule", "straight"), default="rule")
    parser.add_argument("--blue-rule-decision-interval-s", type=float, default=0.1)
    parser.add_argument("--blue-rule-detection-range-m", type=float, default=60000.0)
    parser.add_argument("--blue-rule-critical-range-m", type=float, default=30000.0)
    parser.add_argument("--blue-rule-lookahead-s", type=float, default=6.0)
    parser.add_argument("--blue-rule-effort-penalty", type=float, default=0.04)
    parser.add_argument("--blue-rule-switch-penalty", type=float, default=0.02)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--d-bias", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument(
        "--execution-action-distribution",
        choices=("tanh_box", "radial_tanh_disk"),
        default=None,
    )
    parser.add_argument(
        "--critic-value-head-mode",
        choices=("latent_sum", "scalar"),
        default=None,
    )
    parser.add_argument(
        "--assignment-stickiness-logit-bonus",
        type=float,
        default=None,
        help=(
            "Add this logit bonus to each missile's still-valid current target "
            "to provide explicit assignment hysteresis."
        ),
    )
    parser.add_argument("--gamma-high", type=float, default=1.0)
    parser.add_argument("--gamma-low", type=float, default=1.0)
    parser.add_argument("--lambda-high", type=float, default=0.95)
    parser.add_argument("--lambda-low", type=float, default=0.994883803)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--actor-learning-rate", type=float, default=None)
    parser.add_argument("--critic-learning-rate", type=float, default=None)
    parser.add_argument("--assignment-actor-learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--execution-actor-learning-rate", type=float, default=5.0e-5)
    parser.add_argument("--assignment-critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--execution-critic-learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--clip-epsilon", type=float, default=None)
    parser.add_argument("--assignment-clip-epsilon", type=float, default=0.10)
    parser.add_argument("--execution-clip-epsilon", type=float, default=0.20)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--ppo-sequence-length", type=int, default=None)
    parser.add_argument("--assignment-sequence-length", type=int, default=32)
    parser.add_argument("--execution-sequence-length", type=int, default=128)
    parser.add_argument("--assignment-target-kl", type=float, default=0.01)
    parser.add_argument("--execution-target-kl", type=float, default=0.02)
    parser.add_argument(
        "--assignment-reward-learning-scale",
        type=float,
        default=None,
        help="Trainer-only scale applied to high rewards before assignment GAE.",
    )
    parser.add_argument(
        "--execution-reward-learning-scale",
        type=float,
        default=None,
        help="Trainer-only scale applied to low rewards before execution GAE.",
    )
    parser.add_argument(
        "--execution-value-loss",
        choices=("mse", "huber"),
        default=None,
    )
    parser.add_argument("--execution-value-huber-delta", type=float, default=1.0)
    parser.add_argument(
        "--execution-post-step-kl-rollback",
        action="store_true",
        help="Restore execution actor and Adam state when post-step KL exceeds the limit.",
    )
    parser.add_argument("--execution-post-step-kl-limit", type=float, default=None)
    parser.add_argument("--low-critic-warmup-updates", type=int, default=0)
    parser.add_argument(
        "--low-critic-warmup-critic-steps-per-update",
        type=int,
        default=0,
        help=(
            "Exact execution-critic optimizer steps per critic-only rollout; "
            "zero uses ppo_epochs * critic_updates_per_actor."
        ),
    )
    parser.add_argument(
        "--execution-advantage-normalization",
        choices=("global", "per_scenario"),
        default=None,
        help="Normalize execution advantages globally or within each red/blue scenario.",
    )
    parser.add_argument(
        "--execution-actor-loss-weighting",
        choices=("active_step", "per_scenario"),
        default=None,
        help="Weight execution PPO samples uniformly or give each scenario equal total weight.",
    )
    parser.add_argument("--execution-lr-plateau-patience", type=int, default=0)
    parser.add_argument("--execution-lr-plateau-factor", type=float, default=0.5)
    parser.add_argument("--assignment-lr-plateau-patience", type=int, default=0)
    parser.add_argument("--assignment-lr-plateau-factor", type=float, default=0.5)
    parser.add_argument(
        "--execution-restore-best-on-lr-reduction",
        action="store_true",
        help="Restore the validated-best execution actor and Adam state before continuing at the reduced LR.",
    )
    parser.add_argument(
        "--execution-restore-best-on-early-stop",
        action="store_true",
        help="Restore the validated-best execution actor and Adam state before writing the terminal latest checkpoint.",
    )
    parser.add_argument(
        "--execution-min-actor-learning-rate", type=float, default=None
    )
    parser.add_argument(
        "--assignment-restore-best-on-lr-reduction",
        action="store_true",
        help="Restore the validated-best assignment actor and Adam state before continuing at the reduced LR.",
    )
    parser.add_argument(
        "--assignment-restore-best-on-early-stop",
        action="store_true",
        help="Restore the validated-best assignment actor and Adam state before writing the terminal latest checkpoint.",
    )
    parser.add_argument(
        "--assignment-min-actor-learning-rate", type=float, default=None
    )
    parser.add_argument("--early-stop-validation-patience", type=int, default=0)
    parser.add_argument("--effort-finetune-scale", type=float, default=1000.0)
    parser.add_argument("--critic-updates-per-actor", type=int, default=2)
    parser.add_argument("--actor-update-interval", type=int, default=1)
    parser.add_argument("--entropy-coef", type=float, default=None)
    parser.add_argument("--assignment-entropy-coef", type=float, default=None)
    parser.add_argument("--execution-entropy-coef", type=float, default=0.001)
    parser.add_argument("--high-damage-weight", type=float, default=512.0)
    parser.add_argument("--high-waste-weight", type=float, default=64.0)
    parser.add_argument("--high-potential-weight", type=float, default=None)
    parser.add_argument("--high-potential-gamma", type=float, default=1.0)
    parser.add_argument("--high-time-penalty-per-s", type=float, default=2.0)
    parser.add_argument("--high-time-margin-scale-s", type=float, default=10.0)
    parser.add_argument("--terminal-success-reward", type=float, default=None)
    parser.add_argument("--terminal-failure-penalty", type=float, default=0.0)
    parser.add_argument("--terminal-timeout-penalty", type=float, default=0.0)
    parser.add_argument("--low-damage-weight", type=float, default=512.0)
    parser.add_argument("--low-potential-weight", type=float, default=1.0)
    parser.add_argument("--low-potential-gamma", type=float, default=1.0)
    parser.add_argument("--low-missile-failure-penalty", type=float, default=64.0)
    parser.add_argument("--low-load-penalty", type=float, default=0.0008)
    parser.add_argument("--low-smooth-penalty", type=float, default=0.0002)
    parser.add_argument(
        "--low-time-credit-mode",
        choices=("none", "terminal_active_share"),
        default=None,
    )
    parser.add_argument("--low-time-weight", type=float, default=2.0)
    parser.add_argument(
        "--low-option-boundary-potential",
        choices=("exempt", "terminal_zero"),
        default=None,
    )
    parser.add_argument("--zem-reference-range-m", type=float, default=1000.0)
    parser.add_argument("--zem-floor-range-m", type=float, default=5.0)
    parser.add_argument("--zem-weight", type=float, default=0.6)
    parser.add_argument("--seeker-lock-weight", type=float, default=0.2)
    parser.add_argument("--smooth-bias-denominator", type=float, default=8.0)
    parser.add_argument("--zem-time-gate-scale-s", type=float, default=1.0)
    parser.add_argument("--assignment-min-energy-fraction", type=float, default=0.05)
    parser.add_argument("--assignment-min-available-load-fraction", type=float, default=0.05)
    parser.add_argument("--assignment-correlation-weight", type=float, default=0.5)
    parser.add_argument("--assignment-correlation-angle-scale-deg", type=float, default=15.0)
    parser.add_argument("--assignment-correlation-time-scale-s", type=float, default=5.0)
    parser.add_argument("--checkpoint", default="outputs/env_training_checkpoint.pt")
    parser.add_argument("--best-checkpoint", default=None)
    parser.add_argument("--latest-checkpoint", default="outputs/env_training_latest.pt")
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--validation-interval", type=int, default=25)
    parser.add_argument("--validation-seed-start", type=int, default=20261000)
    parser.add_argument("--validation-trials-per-blue-count", type=int, default=16)
    parser.add_argument(
        "--validation-parallel-envs",
        type=int,
        default=None,
        help="Dedicated process-worker count for fixed validation; defaults to training workers.",
    )
    parser.add_argument(
        "--validation-assignment-mode",
        choices=VALIDATION_ASSIGNMENT_MODE_CHOICES,
        default="auto",
        help=(
            "Target assignment source for fixed validation. auto keeps the "
            "training-mode default (capacity_aware for low_only, actor otherwise); "
            "capacity_aware evaluates the heuristic assignment baseline. "
            "The override is command-line only and is never inherited on resume."
        ),
    )
    parser.add_argument(
        "--validation-assignment-stochastic",
        action="store_true",
        help=(
            "For --validation-only actor evaluation, sample high-level target "
            "assignments while keeping the low-level execution actor deterministic. "
            "Requires --validation-assignment-mode actor and "
            "--validation-policy-seed."
        ),
    )
    parser.add_argument(
        "--validation-policy-seed",
        type=int,
        default=None,
        help=(
            "Torch sampling seed applied after checkpoint RNG restoration for "
            "reproducible stochastic assignment-only validation."
        ),
    )
    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Run one fixed validation on --resume-checkpoint and exit without "
            "training. Requires --iterations 0; writes no checkpoint and lets the "
            "command line, not the checkpoint, define the validation config."
        ),
    )
    parser.add_argument("--resume-checkpoint", default=None, help="Load model, optimizer, trainer, iteration, and RNG state.")
    parser.add_argument("--metrics-path", default="outputs/env_training_metrics.json")
    parser.add_argument(
        "--run-manifest-path",
        default=None,
        help="Reproducibility manifest path; defaults beside --metrics-path.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.parallel_envs <= 0:
        parser.error("--parallel-envs must be positive")
    if args.env_worker_threads <= 0:
        parser.error("--env-worker-threads must be positive")
    if args.alternating_low_updates <= 0 or args.alternating_high_updates <= 0:
        parser.error("alternating update counts must be positive")
    if args.reset_best_on_resume and not args.resume_checkpoint:
        parser.error("--reset-best-on-resume requires --resume-checkpoint")
    if args.stage1_quality_gate and not args.resume_checkpoint:
        parser.error("--stage1-quality-gate requires --resume-checkpoint")
    if not np.isfinite(args.env_worker_timeout_s) or args.env_worker_timeout_s <= 0.0:
        parser.error("--env-worker-timeout-s must be positive and finite")
    if args.low_critic_warmup_updates < 0:
        parser.error("--low-critic-warmup-updates must be non-negative")
    if args.low_critic_warmup_critic_steps_per_update < 0:
        parser.error(
            "--low-critic-warmup-critic-steps-per-update must be non-negative"
        )
    if (
        args.low_critic_warmup_critic_steps_per_update > 0
        and args.low_critic_warmup_updates == 0
    ):
        parser.error(
            "--low-critic-warmup-critic-steps-per-update requires warm-up updates"
        )
    if args.low_critic_warmup_updates and args.training_mode != "low_only":
        parser.error("--low-critic-warmup-updates requires --training-mode low_only")
    if args.execution_lr_plateau_patience < 0:
        parser.error("--execution-lr-plateau-patience must be non-negative")
    if args.assignment_lr_plateau_patience < 0:
        parser.error("--assignment-lr-plateau-patience must be non-negative")
    if args.early_stop_validation_patience < 0:
        parser.error("--early-stop-validation-patience must be non-negative")
    if args.validation_parallel_envs is not None and args.validation_parallel_envs <= 0:
        parser.error("--validation-parallel-envs must be positive")
    if args.validation_only:
        if not args.resume_checkpoint:
            parser.error("--validation-only requires --resume-checkpoint")
        if args.iterations != 0:
            parser.error("--validation-only requires --iterations 0")
        if args.validation_trials_per_blue_count <= 0:
            parser.error(
                "--validation-only requires a positive "
                "--validation-trials-per-blue-count"
            )
    if args.validation_assignment_stochastic:
        if not args.validation_only:
            parser.error(
                "--validation-assignment-stochastic requires --validation-only"
            )
        if args.validation_assignment_mode != "actor":
            parser.error(
                "--validation-assignment-stochastic requires "
                "--validation-assignment-mode actor"
            )
        if args.validation_policy_seed is None:
            parser.error(
                "--validation-assignment-stochastic requires "
                "--validation-policy-seed"
            )
    elif args.validation_policy_seed is not None:
        parser.error(
            "--validation-policy-seed requires "
            "--validation-assignment-stochastic"
        )
    if args.validation_policy_seed is not None and args.validation_policy_seed < 0:
        parser.error("--validation-policy-seed must be non-negative")
    if not 0.0 < args.execution_lr_plateau_factor < 1.0:
        parser.error("--execution-lr-plateau-factor must be in (0, 1)")
    if not 0.0 < args.assignment_lr_plateau_factor < 1.0:
        parser.error("--assignment-lr-plateau-factor must be in (0, 1)")
    if (
        args.execution_min_actor_learning_rate is not None
        and args.execution_min_actor_learning_rate <= 0.0
    ):
        parser.error("--execution-min-actor-learning-rate must be positive")
    if (
        args.assignment_min_actor_learning_rate is not None
        and args.assignment_min_actor_learning_rate <= 0.0
    ):
        parser.error("--assignment-min-actor-learning-rate must be positive")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
