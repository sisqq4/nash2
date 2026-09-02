from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .blue_rl import (BlueEscapeEnvConfig, BlueProcessEnvironmentPool, EvaluationActionShaper,
                      EvaluationShapingConfig, RainbowDQNAgent)
from .blue_rl.config_io import configure_blue_mission_duration, load_environment_config
from .cli_utils import parse_missile_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a blue Rainbow checkpoint with batched inference")
    parser.add_argument("checkpoint"); parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--missiles", default="1", help="Comma-separated scenarios to sample, e.g. 1,2,3,4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/blue_rl/test"); parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-config", default=None); parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument("--parallel-envs", type=int, default=1); parser.add_argument("--env-worker-threads", type=int, default=1)
    parser.add_argument("--env-worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print and archive one aggregate row every N completed episodes")
    parser.add_argument("--jsonl-path", default=None,
                        help="Streaming progress path; defaults to OUTPUT/evaluation.jsonl")
    parser.add_argument("--acmi-interval", type=int, default=1,
                        help="Save one evaluation ACMI every N episodes; use 0 to disable ACMI output")
    parser.add_argument("--mechanism-threat", action="store_true",
                        help="Enable test-only short-horizon threat-relief scoring")
    parser.add_argument("--mechanism-timing", action="store_true",
                        help="Enable test-only P0/P1/P2 timing and phase scoring")
    parser.add_argument("--mechanism-direction", action="store_true",
                        help="Enable test-only LOS-normal escape-direction scoring")
    parser.add_argument("--mechanism-overload", action="store_true",
                        help="Enable test-only threat/energy-dependent overload scoring")
    parser.add_argument("--mechanism-weight", type=float, default=0.35,
                        help="Weight of each enabled normalized mechanism score")
    for name in ("threat", "timing", "direction", "overload"):
        parser.add_argument(f"--mechanism-{name}-weight", type=float, default=1.0,
                            help=f"Relative {name} score weight before phase scheduling")
    parser.add_argument("--mechanism-detail-log", action="store_true",
                        help="Store per-decision Q vectors and mechanism scores for offline analysis")
    return parser


def _emit(row: dict[str, Any], jsonl_path: Path) -> None:
    encoded = json.dumps(row, ensure_ascii=True, allow_nan=False)
    print(encoded, flush=True)
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _numeric_distribution(values: list[float], *, bins: int = 10) -> dict[str, object]:
    """Return JSON-safe descriptive statistics and an equal-width histogram."""
    data = np.asarray(values, dtype=np.float64)
    if data.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "p05": None,
                "p25": None, "median": None, "p75": None, "p95": None, "max": None,
                "histogram": {"bin_edges": [], "counts": []}}
    counts, edges = np.histogram(data, bins=min(bins, max(1, data.size)))
    quantiles = np.percentile(data, [5, 25, 50, 75, 95])
    return {
        "count": int(data.size), "mean": float(data.mean()), "std": float(data.std()),
        "min": float(data.min()), "p05": float(quantiles[0]), "p25": float(quantiles[1]),
        "median": float(quantiles[2]), "p75": float(quantiles[3]), "p95": float(quantiles[4]),
        "max": float(data.max()),
        "histogram": {"bin_edges": edges.tolist(), "counts": counts.astype(int).tolist()},
    }


def _one_meter_probability_histogram(values: list[float]) -> dict[str, object]:
    """Return occupied [m, m+1) bins; sparse output avoids enormous zero-filled JSON."""
    if not values:
        return {"bin_width_m": 1.0, "sample_count": 0, "bins": []}
    integer_bins = Counter(math.floor(max(0.0, float(value))) for value in values)
    total = len(values)
    return {
        "bin_width_m": 1.0,
        "sample_count": total,
        "bins": [
            {"lower_m": meter, "upper_m": meter + 1, "count": count,
             "probability": count / total}
            for meter, count in sorted(integer_bins.items())
        ],
    }


def _aggregate_results(rows: list[dict[str, object]]) -> dict[str, object]:
    survived = sum(bool(row["blue_survived"]) for row in rows)
    action_counts: Counter[int] = Counter()
    for row in rows:
        action_counts.update({int(action): int(count)
                              for action, count in dict(row.get("action_histogram", {})).items()})
    return {
        "episodes": len(rows), "survived": survived, "killed": len(rows) - survived,
        "survival_rate": survived / len(rows) if rows else None,
        "termination_counts": dict(sorted(Counter(str(row["termination_reason"]) for row in rows).items())),
        "red_loss_reason_counts": dict(sorted(Counter(
            str(reason) for row in rows for reason in list(row.get("red_loss_reasons", []))
        ).items())),
        "hit_count_distribution": dict(sorted(Counter(str(int(row["hit_count"])) for row in rows).items())),
        "action_distribution": {str(action): count for action, count in sorted(action_counts.items())},
        "reward": _numeric_distribution([float(row["reward"]) for row in rows]),
        "miss_distance_m": _numeric_distribution([float(row["miss_distance_m"]) for row in rows]),
        "miss_distance_probability_histogram_1m": _one_meter_probability_histogram(
            [float(row["miss_distance_m"]) for row in rows]
        ),
        "simulation_time_s": _numeric_distribution([float(row["simulation_time_s"]) for row in rows]),
        "decision_steps": _numeric_distribution([float(row["decision_steps"]) for row in rows]),
    }


def main() -> int:
    args = build_parser().parse_args()
    try: missile_scenarios = parse_missile_scenarios(args.missiles)
    except ValueError as error: raise SystemExit(str(error)) from error
    if args.episodes < 1 or args.parallel_envs < 1 or args.env_worker_threads < 1 or args.log_interval < 1: raise SystemExit("episode, worker, and log interval counts must be positive")
    if args.acmi_interval < 0 or args.env_worker_timeout_s <= 0: raise SystemExit("invalid ACMI interval or worker timeout")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else output / "evaluation.jsonl"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True); jsonl_path.write_text("", encoding="utf-8")
    environment_config = configure_blue_mission_duration(load_environment_config(args.env_config))
    shaping_config = EvaluationShapingConfig(
        threat=args.mechanism_threat, timing=args.mechanism_timing,
        direction=args.mechanism_direction, overload=args.mechanism_overload,
        weight=args.mechanism_weight, threat_weight=args.mechanism_threat_weight,
        timing_weight=args.mechanism_timing_weight,
        direction_weight=args.mechanism_direction_weight,
        overload_weight=args.mechanism_overload_weight,
    )
    agent = RainbowDQNAgent.load(args.checkpoint, args.device)
    agent.online.eval()
    for parameter in agent.online.parameters(): parameter.requires_grad_(False)
    immutable_learner_state = (agent.total_steps, agent.optimizer_updates, agent.target_updates,
                               agent.replay.size, tuple(parameter._version for parameter in agent.online.parameters()))
    slots = max(missile_scenarios)
    legacy_dim, normalized_dim = 6 + slots * 3, 6 + slots * 4
    checkpoint_schema = agent.config.observation_schema
    if checkpoint_schema == "normalized_v2" and agent.config.observation_dim == normalized_dim:
        observation_schema = "normalized_v2"
    elif checkpoint_schema == "legacy_v1" and agent.config.observation_dim == legacy_dim:
        observation_schema = "legacy_v1"
    else:
        raise ValueError(
            f"checkpoint schema/dimension ({checkpoint_schema}, {agent.config.observation_dim}) "
            f"does not match --missiles; expected legacy ({legacy_dim}) or normalized ({normalized_dim})"
        )
    config = BlueEscapeEnvConfig(missile_scenarios[0], max_missiles=slots,
                                 pad_observation_to_max_missiles=len(missile_scenarios) > 1,
                                 observation_schema=observation_schema,
                                 decision_interval_s=args.decision_interval,
                                 expose_evaluation_mechanism_state=True,
                                 acmi_episode_interval=args.acmi_interval,
                                 acmi_directory=str(output / "acmi"))
    observation_dim = normalized_dim if observation_schema == "normalized_v2" else legacy_dim
    action_dim = 29
    if agent.config.observation_dim != observation_dim or agent.config.action_dim != action_dim:
        raise ValueError(f"checkpoint dimensions ({agent.config.observation_dim}, {agent.config.action_dim}) do not match the requested scenarios ({observation_dim}, {action_dim}); check --missiles")
    pool_size = min(args.parallel_envs, args.episodes); observations = {}; episode_by_worker = {}; rewards = {}
    mechanism_states: dict[int, dict[str, object]] = {}
    shapers = {worker: EvaluationActionShaper(shaping_config) for worker in range(pool_size)}
    mechanism_interventions: dict[int, int] = {}; mechanism_diagnostics: dict[int, dict[str, object]] = {}
    mechanism_traces: dict[int, list[dict[str, object]]] = {}
    decisions: dict[int, int] = {}; action_counts: dict[int, Counter[int]] = {}; initializations = {}
    reward_component_sums: dict[int, Counter[str]] = {}; reward_diagnostic_sums: dict[int, Counter[str]] = {}
    rows: list[dict[str, object]] = []; window_rows: list[dict[str, object]] = []; next_episode = 1
    completed = 0; next_log = args.log_interval; vector_iterations = 0
    started = time.monotonic(); window_started = started
    scenario_rng = random.Random(args.seed)
    episode_scenarios = {episode: scenario_rng.choice(missile_scenarios)
                         for episode in range(1, args.episodes + 1)}
    _emit({"event": "evaluation_config", "checkpoint": args.checkpoint, "device": str(agent.device),
           "episodes": args.episodes, "red_counts": list(missile_scenarios), "blue_count": 1,
           "parallel_cpu_envs": pool_size, "env_worker_threads": args.env_worker_threads,
           "env_worker_timeout_s": args.env_worker_timeout_s, "inference_batch_size_max": pool_size,
           "seed": args.seed, "decision_interval_s": args.decision_interval,
           "acmi_interval": args.acmi_interval, "output": str(output), "evaluation_only": True,
           "evaluation_mechanisms": {"threat": shaping_config.threat, "timing": shaping_config.timing,
                                     "direction": shaping_config.direction, "overload": shaping_config.overload,
                                     "weight": shaping_config.weight},
           "checkpoint_optimizer_updates": agent.optimizer_updates,
           "checkpoint_target_updates": agent.target_updates}, jsonl_path)
    with BlueProcessEnvironmentPool(environment_config, config, pool_size,
                                    native_threads=args.env_worker_threads,
                                    timeout_s=args.env_worker_timeout_s) as pool:
        assignments = {}
        for worker in range(pool_size):
            assignments[worker] = (args.seed + next_episode, next_episode, episode_scenarios[next_episode])
            episode_by_worker[worker] = next_episode; rewards[worker] = 0.0; decisions[worker] = 0
            action_counts[worker] = Counter(); reward_component_sums[worker] = Counter()
            mechanism_interventions[worker] = 0; shapers[worker].reset()
            mechanism_traces[worker] = []
            reward_diagnostic_sums[worker] = Counter(); next_episode += 1
        for worker, (observation, reset_info) in pool.reset(assignments).items():
            observations[worker] = observation; initializations[worker] = reset_info["initialization"]
            mechanism_states[worker] = reset_info["mechanism_state"]
        while observations:
            workers = sorted(observations)
            observation_batch = np.stack([observations[w] for w in workers])
            with torch.inference_mode():
                # The hard safety mask is part of evaluation even when all
                # optional tactical shaping channels are disabled.
                q_values = agent.expected_action_values(observation_batch)
                selected = [shapers[w].select(q, mechanism_states[w])
                            for w, q in zip(workers, q_values)]
                actions = np.asarray([item[0] for item in selected], dtype=np.int64)
                for worker, (_, diagnostic) in zip(workers, selected):
                    mechanism_diagnostics[worker] = diagnostic
                    mechanism_interventions[worker] += int(bool(diagnostic["intervened"]))
                    mechanism_traces[worker].append(diagnostic)
            results = pool.step({worker: int(action) for worker, action in zip(workers, actions)})
            vector_iterations += 1
            resets = {}
            for worker, action in zip(workers, actions):
                result = results[worker]; rewards[worker] += result.reward; observations[worker] = result.observation
                mechanism_states[worker] = dict(result.info["mechanism_state"])
                decisions[worker] += 1; action_counts[worker][int(action)] += 1
                for name, value in dict(result.info.get("reward_components", {})).items():
                    reward_component_sums[worker][str(name)] += float(value)
                for name, value in dict(result.info.get("reward_diagnostics", {})).items():
                    reward_diagnostic_sums[worker][str(name)] += float(value)
                if result.terminated or result.truncated:
                    episode = episode_by_worker[worker]
                    row = {"episode": episode, "missile_count": episode_scenarios[episode],
                           "reward": rewards[worker], **result.info,
                           "initialization": initializations[worker],
                           "blue_orientation": initializations[worker]["blue_orientation"],
                           "simulation_time_s": float(result.info.get("time_s", 0.0)),
                           "physics_steps": int(result.info.get("step_count", 0)),
                           "decision_steps": decisions[worker],
                           "action_histogram": {str(key): value for key, value in sorted(action_counts[worker].items())},
                           "reward_component_sums": dict(reward_component_sums[worker]),
                           "reward_diagnostics_mean": {
                               name: value / decisions[worker]
                               for name, value in reward_diagnostic_sums[worker].items()
                           },
                           "mechanism": {"enabled": shaping_config.enabled,
                                         "interventions": mechanism_interventions[worker],
                                         "intervention_rate": mechanism_interventions[worker] / decisions[worker],
                                         "phase_switches": sum(bool(item.get("phase_changed"))
                                                               for item in mechanism_traces[worker]),
                                         "main_threat_switches": sum(
                                             left.get("main_threat_slot") != right.get("main_threat_slot")
                                             for left, right in zip(mechanism_traces[worker],
                                                                    mechanism_traces[worker][1:])),
                                         "peak_total_threat": max(
                                             (float(item.get("total_threat", 0.0))
                                              for item in mechanism_traces[worker]), default=0.0),
                                         "minimum_safe_corridor": min(
                                             (float(item.get("W_safe", 1.0))
                                              for item in mechanism_traces[worker]), default=1.0),
                                         "last": mechanism_diagnostics.get(worker, {}),
                                         **({"trace": mechanism_traces[worker]}
                                            if args.mechanism_detail_log else {})}}
                    rows.append(row); window_rows.append(row); completed += 1
                    if next_episode <= args.episodes:
                        resets[worker] = (args.seed + next_episode, next_episode, episode_scenarios[next_episode])
                        episode_by_worker[worker] = next_episode; rewards[worker] = 0.0; decisions[worker] = 0
                        action_counts[worker] = Counter(); reward_component_sums[worker] = Counter()
                        mechanism_interventions[worker] = 0; mechanism_diagnostics.pop(worker, None)
                        mechanism_traces[worker] = []
                        shapers[worker].reset()
                        reward_diagnostic_sums[worker] = Counter(); next_episode += 1
                    else:
                        observations.pop(worker); episode_by_worker.pop(worker); rewards.pop(worker)
                        decisions.pop(worker); action_counts.pop(worker); initializations.pop(worker)
                        reward_component_sums.pop(worker); reward_diagnostic_sums.pop(worker)
                        mechanism_states.pop(worker); mechanism_interventions.pop(worker)
                        mechanism_traces.pop(worker)
            if resets:
                for worker, (observation, reset_info) in pool.reset(resets).items():
                    observations[worker] = observation; initializations[worker] = reset_info["initialization"]
                    mechanism_states[worker] = reset_info["mechanism_state"]
            if completed >= next_log or (completed == args.episodes and window_rows):
                now = time.monotonic(); window_rewards = [float(row["reward"]) for row in window_rows]
                misses = [float(row["miss_distance_m"]) for row in window_rows]
                report = {
                    "event": "evaluation_progress", "completed_episodes": completed,
                    "total_episodes": args.episodes, "progress_fraction": completed / args.episodes,
                    "episode_ids": sorted(int(row["episode"]) for row in window_rows),
                    "sampled_red_counts": dict(Counter(int(row["missile_count"]) for row in window_rows)),
                    "active_parallel_envs": len(observations), "inference_batch_size": len(workers),
                    "vector_iterations": vector_iterations,
                    "reward_mean": _mean(window_rewards),
                    "reward_std": float(np.std(window_rewards)) if window_rewards else None,
                    "reward_min": min(window_rewards) if window_rewards else None,
                    "reward_max": max(window_rewards) if window_rewards else None,
                    "survival_rate": _mean([float(row["blue_survived"]) for row in window_rows]),
                    "hit_count_sum": sum(int(row["hit_count"]) for row in window_rows),
                    "miss_distance_mean_m": _mean(misses),
                    "miss_distance_p95_m": float(np.percentile(misses, 95)) if misses else None,
                    "termination_counts": dict(Counter(str(row["termination_reason"]) for row in window_rows)),
                    "rollout_diagnostics": {"wall_time_s": now - window_started,
                                            "episodes_per_hour": len(window_rows) * 3600.0 / max(now - window_started, 1e-9)},
                }
                _emit(report, jsonl_path)
                while next_log <= completed: next_log += args.log_interval
                window_rows = []; window_started = now
    rows.sort(key=lambda row: int(row["episode"]))
    by_scenario = {
        str(count): _aggregate_results(selected)
        for count in missile_scenarios
        if (selected := [row for row in rows if row["missile_count"] == count])
    }
    statistics = _aggregate_results(rows)
    orientation_names = ("toward_missile_swarm", "positive_90_deg", "negative_90_deg",
                         "away_from_missile_swarm")
    by_blue_orientation = {
        name: _aggregate_results([row for row in rows if row["blue_orientation"] == name])
        for name in orientation_names
    }
    current_learner_state = (agent.total_steps, agent.optimizer_updates, agent.target_updates,
                             agent.replay.size, tuple(parameter._version for parameter in agent.online.parameters()))
    if current_learner_state != immutable_learner_state:
        raise RuntimeError("evaluation modified learner state; training and updates are forbidden")
    elapsed = time.monotonic() - started
    summary = {"episodes": args.episodes, "missile_scenarios": list(missile_scenarios),
               "survival_rate": statistics["survival_rate"], "statistics": statistics,
               "by_blue_orientation": by_blue_orientation,
               "by_scenario": by_scenario, "parallel_envs": pool_size,
               "inference_batch_size": pool_size,
               "evaluation_only": True, "learner_state_unchanged": True,
               "evaluation_mechanisms": {"threat": shaping_config.threat, "timing": shaping_config.timing,
                                         "direction": shaping_config.direction, "overload": shaping_config.overload,
                                         "weight": shaping_config.weight},
               "training_updates_performed": False, "environment_transitions_recorded": 0,
               "optimizer_updates_during_evaluation": 0, "target_updates_during_evaluation": 0,
               "replay_transitions_added": 0,
               "checkpoint_optimizer_updates": agent.optimizer_updates,
               "checkpoint_target_updates": agent.target_updates,
               "elapsed_s": elapsed,
               "episodes_per_hour": args.episodes * 3600.0 / max(elapsed, 1e-9),
               "results": rows}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _emit({"event": "evaluation_complete", "episodes": args.episodes,
           "survival_rate": summary["survival_rate"], "by_scenario": by_scenario,
           "by_blue_orientation": by_blue_orientation,
           "statistics": statistics, "evaluation_only": True, "learner_state_unchanged": True,
           "training_updates_performed": False, "environment_transitions_recorded": 0,
           "optimizer_updates_during_evaluation": 0, "target_updates_during_evaluation": 0,
           "replay_transitions_added": 0,
           "elapsed_s": elapsed, "episodes_per_hour": summary["episodes_per_hour"],
           "evaluation_path": str(output / "evaluation.json")}, jsonl_path)
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
