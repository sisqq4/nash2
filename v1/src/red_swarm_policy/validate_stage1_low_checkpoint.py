"""Validate a stage-one low-level residual checkpoint against rule blue evasion.

The capacity-aware target allocation and the 1--4v1 fixed seed schedule are
identical to ``validate_stage1_zero_pn``.  Only the residual source changes:
the checkpoint's low-level actor supplies its deterministic two-axis bias on
each low-level decision, while proportional navigation remains at N=3.5.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .core.config import SwarmModelConfig
from .env import (
    BlueEvasionConfig,
    BlueEvasionController,
    BlueEvasionRuleMachine,
    EnvironmentConfig,
    RedBlueEngagementEnv,
    ScenarioConfig,
    ineffective_loss_rate,
    los_kinematics,
)
from .policy.actor import OverloadBiasActor, TargetAssignmentActor
from .train_env import _load_torch_checkpoint
from .training.rollout import HierarchicalPolicyRuntime
from .validate_stage1_zero_pn import DEFAULT_SEED_START, DEFAULT_TRIALS_PER_SCENARIO


BASELINE_COMPARABLE_METRICS = (
    "full_success_rate",
    "full_success_count",
    "average_damage_rate",
    "ineffective_loss_rate",
    "successful_completion_time_s",
    "all_trial_completion_time_mean_s",
    "control_effort",
    "ammunition_consumption",
    "ammunition_consumed_mean",
    "hit_count_mean",
    "final_miss_distance_mean_m",
    "red_min_miss_distance_mean_m",
    "red_min_miss_distance_min_m",
    "max_pn_load_g",
    "max_final_load_g",
)


@dataclass(frozen=True)
class TrialMetrics:
    trial_index: int
    seed: int
    style: str
    red_count: int
    blue_count: int
    initial_target_indices: list[int]
    steps: int
    task_completion_time_s: float
    terminal_reason: str
    full_success: float
    damage_rate: float
    hit_count: int
    ineffective_loss_rate: float
    ammunition_consumed: int
    ammunition_consumption: float
    control_effort: float
    final_miss_distance_m: float
    red_min_miss_distance_mean_m: float
    red_min_miss_distance_min_m: float
    max_abs_guidance_bias_g: float
    max_guidance_bias_norm_g: float
    guidance_bias_rms_g: float
    guidance_bias_saturation_rate: float
    execution_actor_call_count: int
    max_pn_load_g: float
    max_final_load_g: float
    blue_decision_count: int
    blue_mode_counts: dict[str, int]
    all_state_values_finite: bool
    pn_gain_valid: bool
    residual_bound_valid: bool
    capacity_valid: bool


@dataclass
class _Controller:
    assignment_actor: TargetAssignmentActor
    execution_actor: OverloadBiasActor
    checkpoint_schema_version: int


_WORKER_CONTROLLER: _Controller | None = None
_WORKER_ENVIRONMENTS: dict[int, RedBlueEngagementEnv] = {}


def _terminal_reason(info: dict[str, Any]) -> str:
    reason = str(info.get("termination_reason", "none"))
    if reason == "success":
        return "mission_complete"
    if reason == "red_failure":
        return "red_exhausted"
    if reason == "timeout":
        return "timeout"
    return reason


def _finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_environment(red_count: int) -> RedBlueEngagementEnv:
    config = EnvironmentConfig(
        scenario=ScenarioConfig(
            red_count=red_count,
            blue_count=1,
            max_missiles_per_target=4,
        ),
        policy_start_mode="post_boost",
    )
    blue_policy = BlueEvasionController(BlueEvasionRuleMachine(config, BlueEvasionConfig()))
    return RedBlueEngagementEnv(
        config,
        blue_policy=blue_policy,
        device="cpu",
        record_replay=False,
    )


def _load_controller(checkpoint_path: Path) -> _Controller:
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    model_data = checkpoint.get("model_config")
    if not isinstance(model_data, dict):
        raise ValueError("checkpoint model_config must be a dict")
    model_config = SwarmModelConfig(**model_data)
    if model_config.max_missiles_per_target != 4:
        raise ValueError("stage-one evaluator requires max_missiles_per_target=4")
    assignment_actor = TargetAssignmentActor(model_config).to("cpu")
    execution_actor = OverloadBiasActor(model_config).to("cpu")
    assignment_actor.load_state_dict(checkpoint["assignment_actor"])
    execution_actor.load_state_dict(checkpoint["execution_actor"])
    assignment_actor.eval()
    execution_actor.eval()
    schema_version = checkpoint.get("schema_version")
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ValueError("checkpoint schema_version must be an integer")
    return _Controller(
        assignment_actor=assignment_actor,
        execution_actor=execution_actor,
        checkpoint_schema_version=schema_version,
    )


def _worker_init(checkpoint_path: str) -> None:
    global _WORKER_CONTROLLER, _WORKER_ENVIRONMENTS
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _WORKER_CONTROLLER = _load_controller(Path(checkpoint_path))
    _WORKER_ENVIRONMENTS = {}


def _run_trial(
    env: RedBlueEngagementEnv,
    controller: _Controller,
    *,
    trial_index: int,
    seed: int,
    red_count: int,
    deterministic: bool,
) -> TrialMetrics:
    if not deterministic:
        torch.manual_seed(seed)
    obs = env.reset(
        seed=seed,
        style="many_to_one",
        red_count=red_count,
        blue_count=1,
        start_mode="post_boost",
    )
    assert env.state is not None
    state = env.state
    runtime = HierarchicalPolicyRuntime(
        env,
        controller.assignment_actor,
        controller.execution_actor,
        deterministic=deterministic,
        assignment_mode="capacity_aware",
    )
    runtime.reset(obs)

    red_min_distances = np.full(len(state.red), math.inf, dtype=np.float64)
    initial_targets: np.ndarray | None = None
    max_abs_bias = 0.0
    max_bias_norm = 0.0
    bias_sq_sum = 0.0
    bias_sample_count = 0
    bias_saturated_count = 0
    execution_actor_call_count = 0
    max_pn_load_g = 0.0
    max_final_load_g = 0.0
    all_finite = True
    blue_decision_count = 0
    blue_mode_counts: Counter[str] = Counter()
    hit_count = 0
    last_info: dict[str, Any] = {}

    while not env._episode_done:
        assert env.state is not None
        request = env.next_decision_request()
        if request.bias_due:
            execution_actor_call_count += 1
        policy, _ = runtime.action(obs)
        if initial_targets is None and runtime.assignment_output is not None:
            initial_targets = (
                runtime.assignment_output.actions.target[0].detach().cpu().numpy().astype(np.int64) - 1
            )

        blue_policy = env.decision_layer.blue_policy
        if not isinstance(blue_policy, BlueEvasionController):
            raise RuntimeError("stage-one evaluation requires BlueEvasionController")
        blue_action, blue_decision = blue_policy.action_for(env.state)
        if blue_decision is not None:
            blue_decision_count += 1
            blue_mode_counts.update(str(mode) for mode in blue_decision.modes)
        step = env.step(red_action=policy, blue_action=blue_action)
        runtime.observe(step)
        obs = step.observation
        last_info = dict(step.info)
        hit_count += int(step.info.get("hit_count", 0))

        assert env.state is not None
        state = env.state
        for red_index, red in enumerate(state.red):
            for blue in state.blue:
                red_min_distances[red_index] = min(
                    red_min_distances[red_index],
                    los_kinematics(red, blue).range_m,
                )
            red_min_distances[red_index] = min(red_min_distances[red_index], float(red.min_range_m))
            normalized_bias = np.asarray(red.guidance_bias, dtype=np.float64)
            bias_load = np.asarray(red.bias_load_body_g[1:], dtype=np.float64)
            bias_norm = float(np.linalg.norm(bias_load))
            max_abs_bias = max(max_abs_bias, float(np.max(np.abs(bias_load))))
            max_bias_norm = max(max_bias_norm, bias_norm)
            if red.alive:
                bias_sq_sum += bias_norm * bias_norm
                bias_sample_count += 1
                if bias_norm >= env.config.missile.max_guidance_bias_g - 1.0e-6:
                    bias_saturated_count += 1
            max_pn_load_g = max(max_pn_load_g, float(np.linalg.norm(red.pn_load_body_g[1:])))
            max_final_load_g = max(max_final_load_g, float(np.linalg.norm(red.final_load_body_g[1:])))
            all_finite = all_finite and bool(
                np.all(np.isfinite(red.position_m))
                and np.all(np.isfinite(red.velocity_mps))
                and np.all(np.isfinite(normalized_bias))
                and np.all(np.isfinite(bias_load))
                and np.all(np.isfinite(red.final_load_body_g))
            )
        for blue in state.blue:
            all_finite = all_finite and bool(
                np.all(np.isfinite(blue.position_m)) and np.all(np.isfinite(blue.velocity_mps))
            )
        if step.done:
            break

    assert env.state is not None
    state = env.state
    destroyed_blue = sum(not blue.alive for blue in state.blue)
    alive_red = sum(red.alive for red in state.red)
    targets = np.asarray([red.current_target_index for red in state.red], dtype=np.int64)
    capacity_counts = np.bincount(targets[targets >= 0], minlength=len(state.blue))
    capacity_valid = bool(np.all(capacity_counts <= env.config.scenario.max_missiles_per_target))
    finite_distances = [_finite_or_zero(value) for value in red_min_distances]
    final_miss_distance = _finite_or_zero(
        float(last_info.get("miss_distance_m", min(finite_distances, default=0.0)))
    )
    max_bias_g = env.config.missile.max_guidance_bias_g
    return TrialMetrics(
        trial_index=trial_index,
        seed=seed,
        style="many_to_one",
        red_count=len(state.red),
        blue_count=len(state.blue),
        initial_target_indices=[
            int(value) for value in (targets if initial_targets is None else initial_targets)
        ],
        steps=int(state.step_count),
        task_completion_time_s=float(state.time_s),
        terminal_reason=_terminal_reason(last_info),
        full_success=float(destroyed_blue == len(state.blue)),
        damage_rate=float(destroyed_blue / max(len(state.blue), 1)),
        hit_count=hit_count,
        ineffective_loss_rate=float(ineffective_loss_rate(state)),
        ammunition_consumed=len(state.red) - alive_red,
        ammunition_consumption=float((len(state.red) - alive_red) / max(len(state.red), 1)),
        control_effort=float(env.control_effort),
        final_miss_distance_m=final_miss_distance,
        red_min_miss_distance_mean_m=float(np.mean(finite_distances)),
        red_min_miss_distance_min_m=float(np.min(finite_distances)),
        max_abs_guidance_bias_g=max_abs_bias,
        max_guidance_bias_norm_g=max_bias_norm,
        guidance_bias_rms_g=(
            float(math.sqrt(bias_sq_sum / bias_sample_count)) if bias_sample_count else 0.0
        ),
        guidance_bias_saturation_rate=(
            float(bias_saturated_count / bias_sample_count) if bias_sample_count else 0.0
        ),
        execution_actor_call_count=execution_actor_call_count,
        max_pn_load_g=max_pn_load_g,
        max_final_load_g=max_final_load_g,
        blue_decision_count=blue_decision_count,
        blue_mode_counts=dict(sorted(blue_mode_counts.items())),
        all_state_values_finite=all_finite,
        pn_gain_valid=env.config.missile.proportional_navigation_gain == 3.5,
        residual_bound_valid=max_bias_norm <= max_bias_g + 1.0e-8,
        capacity_valid=capacity_valid,
    )


def _summary(rows: list[TrialMetrics]) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize no trials")
    successes = np.asarray([row.full_success for row in rows], dtype=np.float64)
    success_times = [row.task_completion_time_s for row in rows if row.full_success > 0.5]
    reasons = Counter(row.terminal_reason for row in rows)
    return {
        "trial_count": len(rows),
        "full_success_rate": float(successes.mean()),
        "full_success_count": int(successes.sum()),
        "average_damage_rate": float(np.mean([row.damage_rate for row in rows])),
        "ineffective_loss_rate": float(np.mean([row.ineffective_loss_rate for row in rows])),
        "successful_completion_time_s": float(np.mean(success_times)) if success_times else None,
        "all_trial_completion_time_mean_s": float(np.mean([row.task_completion_time_s for row in rows])),
        "control_effort": float(np.mean([row.control_effort for row in rows])),
        "ammunition_consumption": float(np.mean([row.ammunition_consumption for row in rows])),
        "ammunition_consumed_mean": float(np.mean([row.ammunition_consumed for row in rows])),
        "hit_count_mean": float(np.mean([row.hit_count for row in rows])),
        "final_miss_distance_mean_m": float(np.mean([row.final_miss_distance_m for row in rows])),
        "red_min_miss_distance_mean_m": float(
            np.mean([row.red_min_miss_distance_mean_m for row in rows])
        ),
        "red_min_miss_distance_min_m": float(
            np.min([row.red_min_miss_distance_min_m for row in rows])
        ),
        "guidance_bias_rms_g": float(np.mean([row.guidance_bias_rms_g for row in rows])),
        "guidance_bias_saturation_rate": float(
            np.mean([row.guidance_bias_saturation_rate for row in rows])
        ),
        "max_guidance_bias_norm_g": float(np.max([row.max_guidance_bias_norm_g for row in rows])),
        "max_pn_load_g": float(np.max([row.max_pn_load_g for row in rows])),
        "max_final_load_g": float(np.max([row.max_final_load_g for row in rows])),
        "execution_actor_calls_mean": float(
            np.mean([row.execution_actor_call_count for row in rows])
        ),
        "terminal_reason_counts": dict(sorted(reasons.items())),
        "pn_gain_all_valid": bool(all(row.pn_gain_valid for row in rows)),
        "residual_bound_all_valid": bool(all(row.residual_bound_valid for row in rows)),
        "capacity_all_valid": bool(all(row.capacity_valid for row in rows)),
        "finite_state_all_valid": bool(all(row.all_state_values_finite for row in rows)),
    }


def _write_csv(path: Path, rows: list[TrialMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened: list[dict[str, Any]] = []
    for row in rows:
        item = asdict(row)
        item["initial_target_indices"] = json.dumps(item["initial_target_indices"])
        item["blue_mode_counts"] = json.dumps(item["blue_mode_counts"], sort_keys=True)
        flattened.append(item)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def _metric_comparison(learned: Any, baseline: Any) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for name in BASELINE_COMPARABLE_METRICS:
        learned_value = learned.get(name)
        baseline_value = baseline.get(name)
        if learned_value is None or baseline_value is None:
            delta = None
        else:
            delta = float(learned_value) - float(baseline_value)
        comparison[name] = {
            "pn_zero_residual": baseline_value,
            "checkpoint_residual": learned_value,
            "checkpoint_minus_pn": delta,
        }
    comparison["terminal_reason_counts"] = {
        "pn_zero_residual": baseline.get("terminal_reason_counts", {}),
        "checkpoint_residual": learned.get("terminal_reason_counts", {}),
    }
    return comparison


def _compare_to_pn_baseline(
    summary: dict[str, Any],
    baseline_path: Path,
) -> dict[str, Any]:
    with baseline_path.open(encoding="utf-8") as handle:
        baseline = json.load(handle)
    expected = summary["configuration"]
    actual = baseline.get("configuration", {})
    for key in ("seed_start", "trials_per_scenario", "blue_count", "max_missiles_per_target"):
        if actual.get(key) != expected.get(key):
            raise ValueError(
                f"PN baseline {key}={actual.get(key)!r} does not match checkpoint evaluation "
                f"{expected.get(key)!r}"
            )
    baseline_scenarios = {
        int(item["red_count"]): item for item in baseline.get("by_scenario", [])
    }
    comparison_scenarios: list[dict[str, Any]] = []
    for learned in summary["by_scenario"]:
        red_count = int(learned["red_count"])
        if red_count not in baseline_scenarios:
            raise ValueError(f"PN baseline is missing the {red_count}v1 scenario")
        comparison_scenarios.append(
            {
                "red_count": red_count,
                "blue_count": int(learned["blue_count"]),
                "metrics": _metric_comparison(learned, baseline_scenarios[red_count]),
            }
        )
    return {
        "pn_baseline_summary": str(baseline_path),
        "metric_contract": {
            "full_success_rate": "trials ending with all blue targets destroyed divided by trials",
            "average_damage_rate": "destroyed blue targets divided by initial blue targets, averaged over trials",
            "ineffective_loss_rate": "red missiles lost without a valid hit divided by initial red missiles",
            "successful_completion_time_s": "environment terminal time averaged only over full-success trials",
            "control_effort": "environment shared active-control physical-time U in [0, 1]",
            "ammunition_consumption": "dead or expended red missiles divided by initial red missiles",
        },
        "overall": _metric_comparison(summary["overall"], baseline["overall"]),
        "by_scenario": comparison_scenarios,
    }


def _quality_gate_decision(metrics: dict[str, Any]) -> dict[str, Any]:
    success = metrics["full_success_rate"]
    ineffective = metrics["ineffective_loss_rate"]
    learned_success = float(success["checkpoint_residual"])
    baseline_success = float(success["pn_zero_residual"])
    learned_ineffective = float(ineffective["checkpoint_residual"])
    baseline_ineffective = float(ineffective["pn_zero_residual"])
    tolerance = 1.0e-12
    success_non_regression = learned_success + tolerance >= baseline_success
    ineffective_non_regression = (
        learned_ineffective <= baseline_ineffective + tolerance
    )
    return {
        "full_success_rate": {
            "pn_zero_residual": baseline_success,
            "checkpoint_residual": learned_success,
            "non_regression": success_non_regression,
        },
        "ineffective_loss_rate": {
            "pn_zero_residual": baseline_ineffective,
            "checkpoint_residual": learned_ineffective,
            "non_regression": ineffective_non_regression,
        },
        "passed": success_non_regression and ineffective_non_regression,
    }


def _build_stage1_quality_gate(
    summary: dict[str, Any],
    checkpoint_path: Path,
    baseline_path: Path,
) -> dict[str, Any]:
    comparison = summary["comparison_to_pn_zero_residual"]
    required_red_counts = [1, 2, 3, 4]
    evaluated_red_counts = sorted(
        int(item["red_count"]) for item in comparison["by_scenario"]
    )
    by_scenario = [
        {
            "red_count": int(item["red_count"]),
            "blue_count": int(item["blue_count"]),
            **_quality_gate_decision(item["metrics"]),
        }
        for item in comparison["by_scenario"]
    ]
    overall = _quality_gate_decision(comparison["overall"])
    validity = {
        name: bool(summary["overall"][name])
        for name in (
            "pn_gain_all_valid",
            "residual_bound_all_valid",
            "capacity_all_valid",
            "finite_state_all_valid",
        )
    }
    return {
        "schema_version": 1,
        "policy_mode": "deterministic",
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "baseline_summary_sha256": _file_sha256(baseline_path),
        "required_red_counts": required_red_counts,
        "evaluated_red_counts": evaluated_red_counts,
        "criteria": {
            "full_success_rate": "checkpoint >= paired PN, overall and every required red count",
            "ineffective_loss_rate": "checkpoint <= paired PN, overall and every required red count",
            "tolerance": 1.0e-12,
            "lower_priority_metrics_cannot_compensate": True,
        },
        "runtime_validity": validity,
        "overall": overall,
        "by_scenario": by_scenario,
        "passed": (
            evaluated_red_counts == required_red_counts
            and all(validity.values())
            and bool(overall["passed"])
            and all(bool(item["passed"]) for item in by_scenario)
        ),
    }


def _run_trial_worker(job: tuple[int, int, int, bool]) -> TrialMetrics:
    if _WORKER_CONTROLLER is None:
        raise RuntimeError("worker controller was not initialized")
    trial_index, seed, red_count, deterministic = job
    env = _WORKER_ENVIRONMENTS.setdefault(red_count, _build_environment(red_count))
    return _run_trial(
        env,
        _WORKER_CONTROLLER,
        trial_index=trial_index,
        seed=seed,
        red_count=red_count,
        deterministic=deterministic,
    )


def evaluate(
    *,
    checkpoint_path: Path,
    seed_start: int,
    trials_per_scenario: int,
    red_counts: list[int],
    workers: int = 1,
    pn_baseline_summary: Path | None = None,
    deterministic: bool = True,
) -> tuple[dict[str, Any], list[TrialMetrics]]:
    if trials_per_scenario <= 0:
        raise ValueError("trials_per_scenario must be positive")
    if not red_counts or any(count < 1 or count > 4 for count in red_counts):
        raise ValueError("red_counts must be a nonempty subset of 1,2,3,4")
    if workers <= 0:
        raise ValueError("workers must be positive")

    controller = _load_controller(checkpoint_path) if workers == 1 else None
    checkpoint_schema_version = (
        controller.checkpoint_schema_version
        if controller is not None
        else _load_torch_checkpoint(checkpoint_path).get("schema_version")
    )
    if not isinstance(checkpoint_schema_version, int) or isinstance(checkpoint_schema_version, bool):
        raise ValueError("checkpoint schema_version must be an integer")
    all_rows: list[TrialMetrics] = []
    by_scenario: list[dict[str, Any]] = []
    trial_index = 0
    for red_count in dict.fromkeys(red_counts):
        print(
            json.dumps(
                {
                    "event": "stage1_holdout_scenario_started",
                    "red_count": red_count,
                    "blue_count": 1,
                    "trials": trials_per_scenario,
                    "workers": workers,
                },
                ensure_ascii=True,
            ),
            flush=True,
        )
        jobs = [
            (
                trial_index + offset,
                seed_start + trial_index + offset,
                red_count,
                deterministic,
            )
            for offset in range(trials_per_scenario)
        ]
        if workers == 1:
            assert controller is not None
            env = _build_environment(red_count)
            rows = [
                _run_trial(
                    env,
                    controller,
                    trial_index=index,
                    seed=seed,
                    red_count=count,
                    deterministic=trial_deterministic,
                )
                for index, seed, count, trial_deterministic in jobs
            ]
        else:
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_worker_init,
                initargs=(str(checkpoint_path),),
            ) as executor:
                rows = list(executor.map(_run_trial_worker, jobs))
        all_rows.extend(rows)
        trial_index += trials_per_scenario
        scenario_summary = {
            "style": "many_to_one",
            "red_count": red_count,
            "blue_count": 1,
            **_summary(rows),
        }
        by_scenario.append(scenario_summary)
        print(
            json.dumps(
                {
                    "event": "stage1_holdout_scenario_completed",
                    "red_count": red_count,
                    "blue_count": 1,
                    "trials": trials_per_scenario,
                    "full_success_rate": scenario_summary["full_success_rate"],
                    "ineffective_loss_rate": scenario_summary["ineffective_loss_rate"],
                },
                ensure_ascii=True,
            ),
            flush=True,
        )

    result = {
            "evaluation": "stage1_low_checkpoint_residual_guidance",
            "configuration": {
                "checkpoint": str(checkpoint_path),
                "checkpoint_schema_version": checkpoint_schema_version,
                "policy_mode": (
                    "deterministic_execution_actor_with_capacity_aware_assignment"
                    if deterministic
                    else "stochastic_execution_actor_with_capacity_aware_assignment"
                ),
                "scenario_style": "many_to_one",
                "red_counts": list(dict.fromkeys(red_counts)),
                "blue_count": 1,
                "max_missiles_per_target": 4,
                "blue_policy": "BlueEvasionController(BlueEvasionRuleMachine)",
                "red_guidance": "proportional_navigation_plus_gravity_compensation_plus_checkpoint_residual",
                "proportional_navigation_gain": 3.5,
                "max_guidance_bias_g": 5.0,
                "time_step_s": 0.005,
                "bias_update_interval_s": 0.1,
                "assignment_update_interval_s": 5.0,
                "max_guidance_time_s": 180.0,
                "seed_start": seed_start,
                "trials_per_scenario": trials_per_scenario,
                "workers": workers,
                "seed_schedule": "contiguous globally across 1v1, 2v1, 3v1, 4v1",
            },
            "overall": _summary(all_rows),
            "by_scenario": by_scenario,
    }
    if pn_baseline_summary is not None:
        result["comparison_to_pn_zero_residual"] = _compare_to_pn_baseline(
            result,
            pn_baseline_summary,
        )
        if deterministic:
            result["stage1_quality_gate"] = _build_stage1_quality_gate(
                result,
                checkpoint_path,
                pn_baseline_summary,
            )
    return result, all_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a stage-one low residual checkpoint against rule blue evasion."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--trials-per-scenario", type=int, default=DEFAULT_TRIALS_PER_SCENARIO)
    parser.add_argument("--red-counts", default="1,2,3,4")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample residual actions for a diagnostic run; stochastic results cannot authorize Stage 2.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/stage1_low_checkpoint_validation")
    )
    parser.add_argument(
        "--pn-baseline-summary",
        type=Path,
        default=Path(
            "outputs/stage1_zero_pn_baseline/fixed_100_seed_20261000/stage1_zero_pn_summary.json"
        ),
        help="Matching zero-residual PN summary used for the direct metric comparison.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    red_counts = [int(value.strip()) for value in args.red_counts.split(",") if value.strip()]
    summary, rows = evaluate(
        checkpoint_path=args.checkpoint,
        seed_start=args.seed_start,
        trials_per_scenario=args.trials_per_scenario,
        red_counts=red_counts,
        workers=args.workers,
        pn_baseline_summary=args.pn_baseline_summary,
        deterministic=not args.stochastic,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "stage1_low_checkpoint_summary.json"
    trials_path = args.output_dir / "stage1_low_checkpoint_trials.csv"
    _write_csv(trials_path, rows)
    summary["artifacts"] = {
        "summary_json": str(metrics_path),
        "trials_csv": str(trials_path),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(summary, ensure_ascii=True, allow_nan=False))
    checks = summary["overall"]
    runtime_valid = all(
        (
            checks["pn_gain_all_valid"],
            checks["residual_bound_all_valid"],
            checks["capacity_all_valid"],
            checks["finite_state_all_valid"],
        )
    )
    if not runtime_valid:
        return 2
    if not args.stochastic and summary.get("stage1_quality_gate", {}).get("passed") is not True:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
