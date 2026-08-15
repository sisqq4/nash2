from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cli_utils import parse_float_pair, parse_float_range
from .core.config import PPOConfig, SwarmModelConfig
from .env import (
    BlueEvasionConfig,
    EnvironmentConfig,
    RedBlueEngagementEnv,
    SCENARIO_STYLES,
    ScenarioStyle,
    ineffective_loss_rate,
    los_kinematics,
)
from .policy.actor import OverloadBiasActor, TargetAssignmentActor
from .training.rollout import HierarchicalPolicyRuntime
from .train_env import (
    DEFAULT_SEED,
    DEFAULT_STYLE,
    _configs_from_checkpoint,
    _build_blue_policy,
    _load_torch_checkpoint,
    _parse_int_list,
    _parse_style_list,
    _select_device,
)


@dataclass(frozen=True)
class ValidationTrialMetrics:
    trial_index: int
    seed: int
    style: str
    red_count: int
    blue_count: int
    steps: int
    done: bool
    terminal_reason: str
    task_completion_time_s: float
    policy_control_time_s: float
    hit_count: int
    red_success_rate: float
    ammunition_consumed: int
    ammunition_consumption: float
    ineffective_loss_rate: float
    control_effort: float
    single_miss_distances_m: list[float]
    single_miss_distance_mean_m: float
    single_miss_distance_min_m: float
    cooperative_target_miss_distances_m: list[float]
    cooperative_miss_distance_m: float
    final_miss_distance_m: float
    total_reward_high: float
    mean_reward_low: float


def _apply_env_overrides(args: argparse.Namespace, config: EnvironmentConfig) -> EnvironmentConfig:
    scenario = config.scenario
    sensor = config.sensor
    aircraft = config.aircraft
    missile = config.missile
    reward = config.reward
    reward_updates: dict[str, float] = {}
    for field_name in (
        "high_damage_weight",
        "high_waste_weight",
        "high_potential_weight",
        "high_potential_gamma",
        "high_time_penalty_per_s",
        "high_time_margin_scale_s",
        "terminal_success_reward",
        "terminal_failure_penalty",
        "terminal_timeout_penalty",
        "low_damage_weight",
        "low_potential_weight",
        "low_potential_gamma",
        "low_missile_failure_penalty",
        "low_load_penalty",
        "low_smooth_penalty",
        "zem_reference_range_m",
        "zem_floor_range_m",
        "zem_weight",
        "seeker_lock_weight",
        "smooth_bias_denominator",
        "zem_time_gate_scale_s",
        "assignment_min_energy_fraction",
        "assignment_min_available_load_fraction",
        "assignment_correlation_weight",
        "assignment_correlation_angle_scale_deg",
        "assignment_correlation_time_scale_s",
    ):
        value = getattr(args, field_name)
        if value is not None:
            reward_updates[field_name] = value
    if reward_updates:
        reward = replace(reward, **reward_updates)

    scenario_updates: dict[str, Any] = {}
    for field_name in (
        "red_count",
        "blue_count",
        "max_missiles_per_target",
        "blue_cluster_radius_m",
        "red_sector_center_azimuth_deg",
        "red_sector_width_deg",
        "red_heading_bias_max_deg",
        "position_perturb_m",
        "velocity_perturb_mps",
    ):
        value = getattr(args, field_name)
        if value is not None:
            scenario_updates[field_name] = value
    range_fields = (
        "red_launch_mach_range",
        "red_altitude_range_m",
        "blue_speed_range_mps",
        "blue_altitude_range_m",
        "blue_heading_range_deg",
        "red_cluster_radius_range_m",
    )
    for field_name in range_fields:
        value = getattr(args, field_name)
        if value is not None:
            scenario_updates[field_name] = parse_float_range(
                value,
                field_name,
                positive=field_name != "blue_heading_range_deg",
            )
    if args.blue_cluster_center_ne_m is not None:
        scenario_updates["blue_cluster_center_ne_m"] = parse_float_pair(
            args.blue_cluster_center_ne_m,
            "blue_cluster_center_ne_m",
        )
    if scenario_updates:
        scenario = replace(scenario, **scenario_updates)

    if (
        args.detection_range_m is not None
        or args.communication_delay_steps is not None
        or args.position_noise_m is not None
        or args.velocity_noise_mps is not None
    ):
        sensor = replace(
            sensor,
            detection_range_m=args.detection_range_m if args.detection_range_m is not None else sensor.detection_range_m,
            communication_delay_steps=(
                args.communication_delay_steps
                if args.communication_delay_steps is not None
                else sensor.communication_delay_steps
            ),
            position_noise_m=args.position_noise_m if args.position_noise_m is not None else sensor.position_noise_m,
            velocity_noise_mps=args.velocity_noise_mps if args.velocity_noise_mps is not None else sensor.velocity_noise_mps,
        )

    if args.aircraft_max_load_factor_g is not None:
        aircraft = replace(aircraft, max_load_factor_g=args.aircraft_max_load_factor_g)
    if args.missile_lethal_radius_m is not None or args.missile_escape_range_m is not None:
        missile = replace(
            missile,
            lethal_radius_m=(
                args.missile_lethal_radius_m
                if args.missile_lethal_radius_m is not None
                else missile.lethal_radius_m
            ),
            escape_range_m=(
                args.missile_escape_range_m
                if args.missile_escape_range_m is not None
                else missile.escape_range_m
            ),
        )

    env_config = replace(
        config,
        time_step_s=args.time_step_s if args.time_step_s is not None else config.time_step_s,
        bias_update_interval_s=(
            args.bias_update_interval_s
            if args.bias_update_interval_s is not None
            else config.bias_update_interval_s
        ),
        assignment_update_interval_s=(
            args.assignment_update_interval_s
            if args.assignment_update_interval_s is not None
            else config.assignment_update_interval_s
        ),
        max_steps=args.max_steps if args.max_steps is not None else config.max_steps,
        policy_start_mode="post_boost",
        policy_entry_speed_tolerance_ratio=(
            args.policy_entry_speed_tolerance_ratio
            if args.policy_entry_speed_tolerance_ratio is not None
            else config.policy_entry_speed_tolerance_ratio
        ),
        scenario=scenario,
        sensor=sensor,
        aircraft=aircraft,
        missile=missile,
        reward=reward,
    )
    env_config.validate()
    return env_config


def _load_actors(
    checkpoint: dict[str, Any],
    model_config: SwarmModelConfig,
    device: torch.device,
) -> tuple[TargetAssignmentActor, OverloadBiasActor]:
    assignment_actor = TargetAssignmentActor(model_config).to(device)
    execution_actor = OverloadBiasActor(model_config).to(device)
    assignment_actor.load_state_dict(checkpoint["assignment_actor"])
    execution_actor.load_state_dict(checkpoint["execution_actor"])
    assignment_actor.eval()
    execution_actor.eval()
    return assignment_actor, execution_actor


def _update_distance_trackers(
    env: RedBlueEngagementEnv,
    red_min_distances: np.ndarray,
    blue_min_distances: np.ndarray,
) -> None:
    if env.state is None:
        return
    for red_index, red in enumerate(env.state.red):
        for blue_index, blue in enumerate(env.state.blue):
            distance = los_kinematics(red, blue).range_m
            red_min_distances[red_index] = min(red_min_distances[red_index], distance)
            blue_min_distances[blue_index] = min(blue_min_distances[blue_index], distance)
    for blue_index, blue in enumerate(env.state.blue):
        if not blue.alive:
            blue_min_distances[blue_index] = 0.0


def _finite_or_zero(values: np.ndarray) -> list[float]:
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in values]


def _terminal_reason(info: dict[str, Any], done: bool, hit_count: int) -> str:
    declared = str(info.get("termination_reason", "none"))
    if declared == "success":
        return "mission_complete"
    if declared == "timeout":
        return "timeout"
    if declared == "red_failure":
        return "red_exhausted"
    if hit_count > 0 and bool(info.get("all_blue_done", False)):
        return "mission_complete"
    if hit_count > 0:
        return "partial_hit"
    if bool(info.get("all_red_done", False)):
        return "red_exhausted"
    if bool(info.get("timeout", False)):
        return "timeout"
    if bool(info.get("all_blue_done", False)):
        return "mission_complete"
    return "ended" if done else "rollout_limit"


def _run_trial(
    env: RedBlueEngagementEnv,
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    *,
    seed: int,
    style: ScenarioStyle,
    red_count: int,
    blue_count: int,
    max_steps: int,
    deterministic: bool,
    assignment_mode: str = "actor",
    assignment_deterministic: bool | None = None,
    execution_deterministic: bool | None = None,
) -> ValidationTrialMetrics:
    if assignment_mode not in {"actor", "capacity_aware"}:
        raise ValueError("assignment_mode must be actor or capacity_aware")
    obs = env.reset(
        seed=seed,
        style=style,
        red_count=red_count,
        blue_count=blue_count,
        start_mode="post_boost",
    )
    assert env.state is not None
    runtime = HierarchicalPolicyRuntime(
        env,
        assignment_actor,
        execution_actor,
        deterministic=deterministic,
        assignment_mode=assignment_mode,
        assignment_deterministic=assignment_deterministic,
        execution_deterministic=execution_deterministic,
    )
    runtime.reset(obs)
    red_min_distances = np.full(len(env.state.red), math.inf, dtype=np.float64)
    blue_min_distances = np.full(len(env.state.blue), math.inf, dtype=np.float64)
    total_reward_high = 0.0
    low_reward_values: list[float] = []
    hit_count = 0
    last_info: dict[str, Any] = {}
    done = False

    _update_distance_trackers(env, red_min_distances, blue_min_distances)
    for _ in range(max_steps):
        policy, _ = runtime.action(obs)
        step = env.step(policy)
        runtime.observe(step)
        done = bool(step.done)
        obs = step.observation
        total_reward_high += float(step.reward_high)
        low_reward_values.extend(float(value) for value in np.asarray(step.reward_low).reshape(-1))
        last_info = dict(step.info)
        hit_count += int(step.info.get("hit_count", 0))
        _update_distance_trackers(env, red_min_distances, blue_min_distances)
        if done:
            break

    assert env.state is not None
    for red_index, red in enumerate(env.state.red):
        red_min_distances[red_index] = min(red_min_distances[red_index], float(red.min_range_m))

    single_miss = _finite_or_zero(red_min_distances)
    cooperative_target_miss = _finite_or_zero(blue_min_distances)
    alive_red = sum(1 for red in env.state.red if red.alive)
    ammunition_consumed = len(env.state.red) - alive_red
    destroyed_blue = sum(not blue.alive for blue in env.state.blue)
    ineffective_loss = ineffective_loss_rate(env.state)
    final_miss = float(last_info.get("miss_distance_m", np.min(red_min_distances)))
    if not math.isfinite(final_miss):
        final_miss = 0.0

    return ValidationTrialMetrics(
        trial_index=0,
        seed=seed,
        style=style,
        red_count=len(env.state.red),
        blue_count=len(env.state.blue),
        steps=int(env.state.step_count),
        done=done,
        terminal_reason=_terminal_reason(last_info, done, hit_count),
        task_completion_time_s=float(env.state.time_s),
        policy_control_time_s=float(env.policy_time_s),
        hit_count=hit_count,
        red_success_rate=float(np.clip(hit_count / max(len(env.state.blue), 1), 0.0, 1.0)),
        ammunition_consumed=int(ammunition_consumed),
        ammunition_consumption=float(ammunition_consumed / max(len(env.state.red), 1)),
        ineffective_loss_rate=float(ineffective_loss),
        control_effort=float(last_info.get("control_effort", env.control_effort)),
        single_miss_distances_m=single_miss,
        single_miss_distance_mean_m=float(np.mean(single_miss)) if single_miss else 0.0,
        single_miss_distance_min_m=float(np.min(single_miss)) if single_miss else 0.0,
        cooperative_target_miss_distances_m=cooperative_target_miss,
        cooperative_miss_distance_m=float(np.mean(cooperative_target_miss)) if cooperative_target_miss else 0.0,
        final_miss_distance_m=final_miss,
        total_reward_high=float(total_reward_high),
        mean_reward_low=float(np.mean(low_reward_values)) if low_reward_values else 0.0,
    )


def _summary(trials: list[ValidationTrialMetrics]) -> dict[str, float]:
    if not trials:
        return {
            "trial_count": 0.0,
            "single_miss_distance_mean_m": 0.0,
            "cooperative_miss_distance_mean_m": 0.0,
            "red_average_success_rate": 0.0,
            "red_full_success_rate": 0.0,
            "ammunition_consumed_mean": 0.0,
            "ammunition_consumption_mean": 0.0,
            "ineffective_loss_rate_mean": 0.0,
            "control_effort_mean": 0.0,
            "task_completion_time_mean_s": 0.0,
            "policy_control_time_mean_s": 0.0,
        }
    return {
        "trial_count": float(len(trials)),
        "single_miss_distance_mean_m": float(np.mean([row.single_miss_distance_mean_m for row in trials])),
        "single_miss_distance_min_m": float(np.min([row.single_miss_distance_min_m for row in trials])),
        "cooperative_miss_distance_mean_m": float(np.mean([row.cooperative_miss_distance_m for row in trials])),
        "red_average_success_rate": float(np.mean([row.red_success_rate for row in trials])),
        "red_full_success_rate": float(np.mean([row.hit_count >= row.blue_count for row in trials])),
        "ammunition_consumed_mean": float(np.mean([row.ammunition_consumed for row in trials])),
        "ammunition_consumption_mean": float(np.mean([row.ammunition_consumption for row in trials])),
        "ineffective_loss_rate_mean": float(np.mean([row.ineffective_loss_rate for row in trials])),
        "control_effort_mean": float(np.mean([row.control_effort for row in trials])),
        "task_completion_time_mean_s": float(np.mean([row.task_completion_time_s for row in trials])),
        "policy_control_time_mean_s": float(np.mean([row.policy_control_time_s for row in trials])),
        "total_reward_high_mean": float(np.mean([row.total_reward_high for row in trials])),
        "mean_reward_low_mean": float(np.mean([row.mean_reward_low for row in trials])),
    }


def _write_csv(path: Path, trials: list[ValidationTrialMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(row) for row in trials]
    if not rows:
        return
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row = dict(row)
            row["single_miss_distances_m"] = json.dumps(row["single_miss_distances_m"], ensure_ascii=True)
            row["cooperative_target_miss_distances_m"] = json.dumps(
                row["cooperative_target_miss_distances_m"],
                ensure_ascii=True,
            )
            writer.writerow(row)


def validate(args: argparse.Namespace) -> int:
    device, device_label = _select_device(args.device)
    checkpoint = _load_torch_checkpoint(Path(args.checkpoint))
    model_config, _, loaded_env_config = _configs_from_checkpoint(
        checkpoint,
        SwarmModelConfig(),
        ppo_fallback=PPOConfig(),
        env_fallback=EnvironmentConfig(),
    )
    env_config = _apply_env_overrides(args, loaded_env_config)
    if model_config.max_missiles_per_target != env_config.scenario.max_missiles_per_target:
        raise ValueError("model and environment max_missiles_per_target must match")
    assignment_actor, execution_actor = _load_actors(checkpoint, model_config, device)
    blue_policy_mode = args.blue_policy or str(checkpoint.get("blue_policy", "rule"))
    saved_blue_config = checkpoint.get("blue_evasion_config")
    blue_evasion_config = (
        BlueEvasionConfig(**saved_blue_config)
        if isinstance(saved_blue_config, dict)
        else BlueEvasionConfig()
    )
    env = RedBlueEngagementEnv(
        env_config,
        blue_policy=_build_blue_policy(env_config, blue_policy_mode, blue_evasion_config),
        device=device,
        record_replay=False,
    )

    red_fallback = args.red_count if args.red_count is not None else env_config.scenario.red_count
    blue_fallback = args.blue_count if args.blue_count is not None else env_config.scenario.blue_count
    style_fallback = args.style or DEFAULT_STYLE
    red_counts = _parse_int_list(args.red_counts, red_fallback, "red_counts")
    blue_counts = _parse_int_list(args.blue_counts, blue_fallback, "blue_counts")
    styles = _parse_style_list(args.styles, style_fallback)
    rollout_assignment_steps = args.rollout_steps
    max_steps = (
        env_config.policy_horizon_steps
        if rollout_assignment_steps is None
        else min(
            env_config.policy_horizon_steps,
            rollout_assignment_steps * env_config.assignment_update_steps,
        )
    )
    deterministic = not args.stochastic

    print(json.dumps({"event": "device", "device": device_label}, ensure_ascii=True), flush=True)
    print(
        json.dumps(
            {
                "event": "validation_config",
                "checkpoint": str(args.checkpoint),
                "trials": args.trials,
                "red_counts": red_counts,
                "blue_counts": blue_counts,
                "styles": styles,
                "rollout_assignment_steps": rollout_assignment_steps,
                "rollout_step_unit": "assignment_decision",
                "max_physics_steps": max_steps,
                "deterministic": deterministic,
                "blue_policy": blue_policy_mode,
                "blue_evasion_config": asdict(blue_evasion_config),
            },
            ensure_ascii=True,
        ),
        flush=True,
    )

    trials: list[ValidationTrialMetrics] = []
    for trial_index in range(args.trials):
        row = _run_trial(
            env,
            assignment_actor,
            execution_actor,
            seed=args.seed + trial_index,
            style=styles[trial_index % len(styles)],
            red_count=red_counts[trial_index % len(red_counts)],
            blue_count=blue_counts[trial_index % len(blue_counts)],
            max_steps=max_steps,
            deterministic=deterministic,
        )
        row = replace(row, trial_index=trial_index)
        trials.append(row)
        print(json.dumps({"event": "trial", **asdict(row)}, ensure_ascii=True), flush=True)

    summary = _summary(trials)
    result = {
        "checkpoint": str(args.checkpoint),
        "device": device_label,
        "deterministic": deterministic,
        "blue_policy": blue_policy_mode,
        "blue_evasion_config": asdict(blue_evasion_config),
        "env_config": asdict(env_config),
        "model_config": asdict(model_config),
        "schedule": {
            "seed": args.seed,
            "trials": args.trials,
            "rollout_assignment_steps": rollout_assignment_steps,
            "rollout_step_unit": "assignment_decision",
            "max_physics_steps": max_steps,
            "red_counts": red_counts,
            "blue_counts": blue_counts,
            "styles": styles,
        },
        "metric_definitions": {
            "single_miss_distances_m": "minimum miss distance reached by each red missile during one trial",
            "cooperative_miss_distance_m": "mean over blue targets of each target's closest red missile distance during one trial",
            "red_success_rate": "hit_count divided by blue_count for one trial",
            "ammunition_consumption": "dead or expended red missiles divided by red_count",
            "ineffective_loss_rate": "red missiles lost without a valid hit divided by initial red_count",
            "control_effort": "shared active-control physical-time U with 5g load and bias-smooth components",
            "task_completion_time_s": "environment time at terminal state or validation rollout limit",
            "policy_control_time_s": "task_completion_time_s minus the configured post-boost entry time",
        },
        "summary": summary,
        "trials": [asdict(row) for row in trials],
    }

    metrics_path = Path(args.metrics_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"event": "metrics", "path": str(metrics_path)}, ensure_ascii=True), flush=True)
    if args.trials_csv:
        _write_csv(Path(args.trials_csv), trials)
        print(json.dumps({"event": "trials_csv", "path": str(args.trials_csv)}, ensure_ascii=True), flush=True)
    print(json.dumps({"event": "summary", **summary}, ensure_ascii=True), flush=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate a red swarm actor checkpoint in generated environments.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--rollout-steps",
        type=int,
        default=None,
        help="Maximum 1.0 s target-assignment decision steps per trial.",
    )
    parser.add_argument("--stochastic", action="store_true", help="Sample actions instead of deterministic policy outputs.")
    parser.add_argument("--blue-policy", choices=("rule", "straight"), default=None)
    parser.add_argument("--style", choices=SCENARIO_STYLES, default=DEFAULT_STYLE)
    parser.add_argument("--styles", default=None, help="Comma-separated scenario styles sampled cyclically.")
    parser.add_argument("--red-count", type=int, default=None)
    parser.add_argument("--red-counts", default=None, help="Comma-separated red missile counts sampled cyclically.")
    parser.add_argument("--blue-count", type=int, default=None)
    parser.add_argument("--blue-counts", default=None, help="Comma-separated blue target counts sampled cyclically.")
    parser.add_argument("--max-missiles-per-target", type=int, default=None)
    parser.add_argument("--time-step-s", type=float, default=None)
    parser.add_argument("--bias-update-interval-s", type=float, default=None)
    parser.add_argument("--assignment-update-interval-s", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--policy-entry-speed-tolerance-ratio", type=float, default=None)
    parser.add_argument("--red-launch-mach-range", default=None)
    parser.add_argument("--red-altitude-range-m", default=None)
    parser.add_argument("--blue-speed-range-mps", default=None)
    parser.add_argument("--blue-altitude-range-m", default=None)
    parser.add_argument("--blue-cluster-center-ne-m", default=None)
    parser.add_argument("--blue-cluster-radius-m", type=float, default=None)
    parser.add_argument("--blue-heading-range-deg", default=None)
    parser.add_argument(
        "--red-cluster-radius-range-m",
        default=None,
        help="Comma-separated min,max red-cluster radii in meters.",
    )
    parser.add_argument("--red-sector-center-azimuth-deg", type=float, default=None)
    parser.add_argument("--red-sector-width-deg", type=float, default=None)
    parser.add_argument("--red-heading-bias-max-deg", type=float, default=None)
    parser.add_argument("--position-perturb-m", type=float, default=None)
    parser.add_argument("--velocity-perturb-mps", type=float, default=None)
    parser.add_argument("--detection-range-m", type=float, default=None)
    parser.add_argument("--communication-delay-steps", type=int, default=None)
    parser.add_argument("--position-noise-m", type=float, default=None)
    parser.add_argument("--velocity-noise-mps", type=float, default=None)
    parser.add_argument("--aircraft-max-load-factor-g", type=float, default=None)
    parser.add_argument("--missile-lethal-radius-m", type=float, default=None)
    parser.add_argument("--missile-escape-range-m", type=float, default=None)
    for option in (
        "high-damage-weight",
        "high-waste-weight",
        "high-potential-weight",
        "high-potential-gamma",
        "high-time-penalty-per-s",
        "high-time-margin-scale-s",
        "terminal-success-reward",
        "terminal-failure-penalty",
        "terminal-timeout-penalty",
        "low-damage-weight",
        "low-potential-weight",
        "low-potential-gamma",
        "low-missile-failure-penalty",
        "low-load-penalty",
        "low-smooth-penalty",
        "zem-reference-range-m",
        "zem-floor-range-m",
        "zem-weight",
        "seeker-lock-weight",
        "smooth-bias-denominator",
        "zem-time-gate-scale-s",
        "assignment-min-energy-fraction",
        "assignment-min-available-load-fraction",
        "assignment-correlation-weight",
        "assignment-correlation-angle-scale-deg",
        "assignment-correlation-time-scale-s",
    ):
        parser.add_argument(f"--{option}", type=float, default=None)
    parser.add_argument("--metrics-path", default="outputs/checkpoint_validation_metrics.json")
    parser.add_argument("--trials-csv", default="outputs/checkpoint_validation_trials.csv")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.rollout_steps is not None and args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error("--max-steps must be positive")
    return validate(args)


if __name__ == "__main__":
    raise SystemExit(main())
