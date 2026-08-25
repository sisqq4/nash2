"""Evaluate Blue's existing rule machine against non-learning red PN missiles.

The baseline runs through the same :class:`BlueEscapeEnv` adapter used by Blue
Rainbow training and evaluation.  Red targets the sole Blue aircraft with a
zero residual command, while Blue actions come from the existing deterministic
rule machine.  No learned policy, checkpoint, replay buffer, or optimizer is
created.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import time

from typing import Any

import numpy as np

from .blue_rl import BlueEscapeEnv, BlueEscapeEnvConfig
from .blue_rl.config_io import configure_blue_mission_duration, load_environment_config
from .cli_utils import parse_missile_scenarios
from .env import BlueEvasionConfig, BlueEvasionRuleMachine
from .evaluate_blue_rl import _aggregate_results

DEFAULT_SEED_START = 20271000
DEFAULT_EPISODES_PER_SCENARIO = 100


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the no-learning baseline: Blue's existing rule machine "
            "against zero-residual red PN missiles in the Blue-RL test environment."
        )
    )
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--episodes-per-scenario", type=int,
                        default=DEFAULT_EPISODES_PER_SCENARIO)
    parser.add_argument("--missiles", default="1,2,3,4")
    parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--acmi-interval", type=int, default=0)

    parser.add_argument("--log-interval", type=int, default=1,
                        help="Print and archive progress every N completed episodes")
    parser.add_argument("--output", type=Path, default=Path("outputs/blue_rl/rule_baseline"))
    return parser


def _mark_as_blue_rule_baseline(summary: dict[str, object]) -> None:
    """Attach machine-readable controls that prevent baseline ambiguity."""
    summary["evaluation"] = "blue_rule_vs_red_zero_residual_pn_baseline"
    configuration = summary["configuration"]
    assert isinstance(configuration, dict)
    configuration.update({
        "baseline": True,
        "purpose": "measure_intelligent_game_strategy_effect",
        "blue_policy": "BlueEvasionRuleMachine",
        "red_policy": "fixed_target_zero_residual_proportional_navigation",
        "blue_learning_enabled": False,
        "red_learning_enabled": False,
        "blue_checkpoint": None,
        "red_checkpoint": None,
    })


def _json_safe_csv_value(value: object) -> object:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_trials(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty baseline")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: _json_safe_csv_value(value) for key, value in row.items()}
                         for row in rows)



def _emit(row: dict[str, object], jsonl_path: Path) -> None:
    encoded = json.dumps(row, ensure_ascii=True, allow_nan=False)
    print(encoded, flush=True)
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _run_episode(env: BlueEscapeEnv, rule: BlueEvasionRuleMachine, *, episode: int,
                 seed: int, missile_count: int) -> dict[str, object]:
    _, reset_info = env.reset(seed=seed, episode_index=episode, missile_count=missile_count)
    rule.reset()
    total_reward = 0.0
    decision_steps = 0
    action_counts: Counter[int] = Counter()
    blue_mode_counts: Counter[str] = Counter()
    reward_components: Counter[str] = Counter()
    last_info: dict[str, Any] = {}
    while True:
        assert env.inner.state is not None
        decision = rule.decide(env.inner.state)
        action = int(decision.action_indices[0])
        blue_mode_counts.update(decision.modes)
        _, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        decision_steps += 1
        action_counts[action] += 1
        for name, value in dict(info.get("reward_components", {})).items():
            reward_components[str(name)] += float(value)
        last_info = dict(info)
        if terminated or truncated:
            break
    return {
        "episode": episode,
        "seed": seed,
        "missile_count": missile_count,
        "reward": total_reward,
        **last_info,
        "initialization": reset_info["initialization"],
        "blue_orientation": reset_info["initialization"]["blue_orientation"],
        "simulation_time_s": float(last_info.get("time_s", 0.0)),
        "physics_steps": int(last_info.get("step_count", 0)),
        "decision_steps": decision_steps,
        "action_histogram": {str(key): value for key, value in sorted(action_counts.items())},
        "blue_mode_counts": dict(sorted(blue_mode_counts.items())),
        "reward_component_sums": dict(reward_components),
    }


def evaluate(*, seed_start: int, episodes_per_scenario: int, missile_counts: tuple[int, ...],
             decision_interval_s: float, env_config: str | None, acmi_interval: int,

             output: Path, log_interval: int = 1,
             jsonl_path: Path | None = None) -> tuple[dict[str, object], list[dict[str, object]]]:

    if episodes_per_scenario < 1:
        raise ValueError("episodes-per-scenario must be positive")
    if acmi_interval < 0:
        raise ValueError("acmi-interval must be non-negative")

    if log_interval < 1:
        raise ValueError("log-interval must be positive")
    progress_path = output / "blue_rule_baseline_progress.jsonl" if jsonl_path is None else jsonl_path
    total_episodes = len(missile_counts) * episodes_per_scenario
    started = time.monotonic()

    environment_config = configure_blue_mission_duration(load_environment_config(env_config))
    adapter_config = BlueEscapeEnvConfig(
        missile_count=missile_counts[0], max_missiles=max(missile_counts),
        decision_interval_s=decision_interval_s, record_acmi=acmi_interval > 0,
        acmi_episode_interval=acmi_interval, acmi_directory=str(output / "acmi"),
    )
    env = BlueEscapeEnv(environment_config, adapter_config)
    rule = BlueEvasionRuleMachine(
        environment_config,
        BlueEvasionConfig(decision_interval_s=decision_interval_s),
    )
    rows: list[dict[str, object]] = []
    for scenario_index, missile_count in enumerate(missile_counts):
        for offset in range(episodes_per_scenario):
            episode = scenario_index * episodes_per_scenario + offset + 1
            seed = seed_start + episode - 1
            rows.append(_run_episode(env, rule, episode=episode, seed=seed,
                                     missile_count=missile_count))

            if episode % log_interval == 0 or episode == total_episodes:
                survived = sum(bool(row["blue_survived"]) for row in rows)
                elapsed = time.monotonic() - started
                _emit({
                    "event": "baseline_progress",
                    "completed_episodes": episode,
                    "total_episodes": total_episodes,
                    "progress_fraction": episode / total_episodes,
                    "current_missile_count": missile_count,
                    "survival_rate_so_far": survived / episode,
                    "elapsed_seconds": elapsed,
                    "episodes_per_second": episode / elapsed if elapsed > 0.0 else None,
                }, progress_path)

    by_scenario = []
    for missile_count in missile_counts:
        scenario_rows = [row for row in rows if row["missile_count"] == missile_count]
        by_scenario.append({"missile_count": missile_count, **_aggregate_results(scenario_rows)})
    summary: dict[str, object] = {
        "configuration": {
            "missile_counts": list(missile_counts),
            "episodes_per_scenario": episodes_per_scenario,
            "seed_start": seed_start,
            "seed_schedule": "contiguous globally in listed scenario order",
            "decision_interval_s": decision_interval_s,
            "environment": "BlueEscapeEnv",
        },
        "statistics": _aggregate_results(rows),
        "by_scenario": by_scenario,
        "episodes": rows,
    }
    _mark_as_blue_rule_baseline(summary)
    return summary, rows


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    progress_path = args.output / "blue_rule_baseline_progress.jsonl"
    progress_path.write_text("", encoding="utf-8")
    try:
        missile_counts = parse_missile_scenarios(args.missiles)
        _emit({
            "event": "baseline_start",
            "message": "Blue rule baseline evaluation is running",
            "missile_counts": list(missile_counts),
            "episodes_per_scenario": args.episodes_per_scenario,
            "total_episodes": len(missile_counts) * args.episodes_per_scenario,
            "seed_start": args.seed_start,
            "output": str(args.output),
        }, progress_path)

        summary, rows = evaluate(
            seed_start=args.seed_start, episodes_per_scenario=args.episodes_per_scenario,
            missile_counts=missile_counts, decision_interval_s=args.decision_interval,
            env_config=args.env_config, acmi_interval=args.acmi_interval, output=args.output,

            log_interval=args.log_interval, jsonl_path=progress_path,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error

    summary_path = args.output / "blue_rule_baseline_summary.json"
    trials_path = args.output / "blue_rule_baseline_trials.csv"
    _write_trials(trials_path, rows)
    summary["artifacts"] = {"summary_json": str(summary_path), "trials_csv": str(trials_path)}
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                       allow_nan=False) + "\n", encoding="utf-8")

    _emit({"event": "baseline_complete", "completed_episodes": len(rows),
           "summary_json": str(summary_path), "trials_csv": str(trials_path)}, progress_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
