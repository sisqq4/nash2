"""Evaluate the stage-one capacity-aware, zero-residual PN baseline.

This module deliberately does not load a learned actor.  Its red command consists
only of a capacity-aware target assignment and an exact zero two-axis residual,
so the environment combines proportional navigation (N=3.5 by default) with its
normal gravity compensation.  Blue uses the same rule-evasion controller as the
stage-one curriculum.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import (
    BlueEvasionConfig,
    BlueEvasionController,
    BlueEvasionRuleMachine,
    EnvironmentConfig,
    RedAction,
    RedBlueEngagementEnv,
    ScenarioConfig,
    ineffective_loss_rate,
    los_kinematics,
)


DEFAULT_SEED_START = 20261000
DEFAULT_TRIALS_PER_SCENARIO = 100


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
    max_pn_load_g: float
    max_final_load_g: float
    blue_decision_count: int
    blue_mode_counts: dict[str, int]
    all_state_values_finite: bool
    pn_gain_valid: bool
    zero_residual_valid: bool
    capacity_valid: bool


def _capacity_aware_targets(
    env: RedBlueEngagementEnv,
    *,
    current_targets: np.ndarray | None = None,
) -> np.ndarray:
    """Faithfully use stage-one's quality-based capacity-aware allocation.

    The collector uses the initial post-boost observation to decide allocation,
    then retains it until the next 5-second high-level decision.  It does not
    require an immediate seeker observation merely to assign a target.
    """
    from .training.rollout import _capacity_aware_assignment

    assert env.state is not None
    state = env.state
    if current_targets is None:
        inputs = env.last_observation.assignment_actor_inputs
    else:
        slots = torch.as_tensor(current_targets + 1, dtype=torch.long)
        inputs = env.observation_layer.execution_inputs(state, slots)
        # `execution_inputs` has the expected target choices but not an
        # assignment actor input.  At later high-level decisions ask the full
        # observation layer for fresh assignment features.
        inputs = env.observation_layer.observe(state).assignment_actor_inputs
    slots = _capacity_aware_assignment(
        inputs,
        env.config.scenario.max_missiles_per_target,
    )[0].detach().cpu().numpy().astype(np.int64)
    return slots - 1


def _finite_or_zero(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else 0.0


def _terminal_reason(info: dict[str, Any]) -> str:
    reason = str(info.get("termination_reason", "none"))
    if reason == "success":
        return "mission_complete"
    if reason == "red_failure":
        return "red_exhausted"
    if reason == "timeout":
        return "timeout"
    return reason


def _build_environment(red_count: int) -> RedBlueEngagementEnv:
    config = EnvironmentConfig(
        scenario=ScenarioConfig(
            red_count=red_count,
            blue_count=1,
            max_missiles_per_target=4,
        ),
        policy_start_mode="post_boost",
    )
    blue_config = BlueEvasionConfig()
    blue_policy = BlueEvasionController(BlueEvasionRuleMachine(config, blue_config))
    return RedBlueEngagementEnv(
        config,
        blue_policy=blue_policy,
        device="cpu",
        record_replay=False,
    )


def run_trial(
    env: RedBlueEngagementEnv,
    *,
    trial_index: int,
    seed: int,
    red_count: int,
) -> TrialMetrics:
    obs = env.reset(
        seed=seed,
        style="many_to_one",
        red_count=red_count,
        blue_count=1,
        start_mode="post_boost",
    )
    del obs
    assert env.state is not None
    state = env.state
    config = env.config
    targets = np.full(len(state.red), -1, dtype=np.int64)
    initial_targets: np.ndarray | None = None
    action = RedAction(
        target_indices=targets,
        guidance_bias=np.zeros((len(state.red), 2), dtype=np.float64),
    )

    red_min_distances = np.full(len(state.red), math.inf, dtype=np.float64)
    max_abs_bias = 0.0
    max_pn_load_g = 0.0
    max_final_load_g = 0.0
    all_finite = True
    blue_decision_count = 0
    blue_mode_counts: Counter[str] = Counter()
    hit_count = 0
    last_info: dict[str, Any] = {}

    while not env._episode_done:
        assert env.state is not None
        state = env.state
        if env.next_decision_request().assignment_due:
            targets = _capacity_aware_targets(env, current_targets=targets)
            if initial_targets is None:
                initial_targets = targets.copy()
        # RedAction makes defensive copies, so allocation changes must be
        # explicitly transferred before physics evaluates this control frame.
        action = RedAction(
            target_indices=targets,
            guidance_bias=np.zeros((len(state.red), 2), dtype=np.float64),
        )
        blue_policy = env.decision_layer.blue_policy
        if not isinstance(blue_policy, BlueEvasionController):
            raise RuntimeError("stage-one baseline requires BlueEvasionController")
        blue_action, blue_decision = blue_policy.action_for(state)
        if blue_decision is not None:
            blue_decision_count += 1
            blue_mode_counts.update(str(mode) for mode in blue_decision.modes)
        step = env.step(red_action=action, blue_action=blue_action)
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
            max_abs_bias = max(max_abs_bias, float(np.max(np.abs(red.guidance_bias))))
            max_pn_load_g = max(max_pn_load_g, float(np.linalg.norm(red.pn_load_body_g[1:])))
            max_final_load_g = max(max_final_load_g, float(np.linalg.norm(red.final_load_body_g[1:])))
            all_finite = all_finite and bool(
                np.all(np.isfinite(red.position_m))
                and np.all(np.isfinite(red.velocity_mps))
                and np.all(np.isfinite(red.guidance_bias))
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
    capacity_counts = np.bincount(targets[targets >= 0], minlength=len(state.blue))
    capacity_valid = bool(np.all(capacity_counts <= config.scenario.max_missiles_per_target))
    zero_residual_valid = max_abs_bias == 0.0 and float(env.control_effort) == 0.0
    pn_gain_valid = config.missile.proportional_navigation_gain == 3.5
    finite_distances = [_finite_or_zero(value) for value in red_min_distances]
    final_miss_distance = _finite_or_zero(
        float(last_info.get("miss_distance_m", min(finite_distances, default=0.0)))
    )
    return TrialMetrics(
        trial_index=trial_index,
        seed=seed,
        style="many_to_one",
        red_count=len(state.red),
        blue_count=len(state.blue),
        initial_target_indices=[int(value) for value in (targets if initial_targets is None else initial_targets)],
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
        max_pn_load_g=max_pn_load_g,
        max_final_load_g=max_final_load_g,
        blue_decision_count=blue_decision_count,
        blue_mode_counts=dict(sorted(blue_mode_counts.items())),
        all_state_values_finite=all_finite,
        pn_gain_valid=pn_gain_valid,
        zero_residual_valid=zero_residual_valid,
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
        "red_min_miss_distance_mean_m": float(np.mean([row.red_min_miss_distance_mean_m for row in rows])),
        "red_min_miss_distance_min_m": float(np.min([row.red_min_miss_distance_min_m for row in rows])),
        "max_pn_load_g": float(np.max([row.max_pn_load_g for row in rows])),
        "max_final_load_g": float(np.max([row.max_final_load_g for row in rows])),
        "terminal_reason_counts": dict(sorted(reasons.items())),
        "zero_residual_all_valid": bool(all(row.zero_residual_valid for row in rows)),
        "pn_gain_all_valid": bool(all(row.pn_gain_valid for row in rows)),
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


def _run_trial_worker(job: tuple[int, int, int]) -> TrialMetrics:
    trial_index, seed, red_count = job
    return run_trial(
        _build_environment(red_count),
        trial_index=trial_index,
        seed=seed,
        red_count=red_count,
    )


def evaluate(
    *,
    seed_start: int,
    trials_per_scenario: int,
    red_counts: list[int],
    workers: int = 1,
) -> tuple[dict[str, Any], list[TrialMetrics]]:
    if trials_per_scenario <= 0:
        raise ValueError("trials_per_scenario must be positive")
    if not red_counts or any(count < 1 or count > 4 for count in red_counts):
        raise ValueError("red_counts must be a nonempty subset of 1,2,3,4")
    if workers <= 0:
        raise ValueError("workers must be positive")
    all_rows: list[TrialMetrics] = []
    by_scenario: list[dict[str, Any]] = []
    trial_index = 0
    for red_count in dict.fromkeys(red_counts):
        jobs = [
            (
                trial_index + offset,
                seed_start + trial_index + offset,
                red_count,
            )
            for offset in range(trials_per_scenario)
        ]
        if workers == 1:
            env = _build_environment(red_count)
            rows = [
                run_trial(env, trial_index=index, seed=seed, red_count=count)
                for index, seed, count in jobs
            ]
        else:
            with ProcessPoolExecutor(max_workers=workers) as executor:
                rows = list(executor.map(_run_trial_worker, jobs))
        all_rows.extend(rows)
        trial_index += trials_per_scenario
        by_scenario.append(
            {
                "style": "many_to_one",
                "red_count": red_count,
                "blue_count": 1,
                **_summary(rows),
            }
        )
    return (
        {
            "evaluation": "stage1_zero_residual_proportional_navigation_baseline",
            "configuration": {
                "scenario_style": "many_to_one",
                "red_counts": list(dict.fromkeys(red_counts)),
                "blue_count": 1,
                "max_missiles_per_target": 4,
                "blue_policy": "BlueEvasionController(BlueEvasionRuleMachine)",
                "red_guidance": "proportional_navigation_plus_gravity_compensation",
                "proportional_navigation_gain": 3.5,
                "guidance_residual_bias_g": [0.0, 0.0],
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
        },
        all_rows,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the first-stage zero-residual PN (N=3.5) baseline against rule blue evasion."
    )
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--trials-per-scenario", type=int, default=DEFAULT_TRIALS_PER_SCENARIO)
    parser.add_argument("--red-counts", default="1,2,3,4")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/stage1_zero_pn_baseline"))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    red_counts = [int(value.strip()) for value in args.red_counts.split(",") if value.strip()]
    summary, rows = evaluate(
        seed_start=args.seed_start,
        trials_per_scenario=args.trials_per_scenario,
        red_counts=red_counts,
        workers=args.workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "stage1_zero_pn_summary.json"
    trials_path = args.output_dir / "stage1_zero_pn_trials.csv"
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
    return 0 if all(
        (
            checks["zero_residual_all_valid"],
            checks["pn_gain_all_valid"],
            checks["capacity_all_valid"],
            checks["finite_state_all_valid"],
        )
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
