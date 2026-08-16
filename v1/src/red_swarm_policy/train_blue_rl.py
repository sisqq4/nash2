from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .blue_rl import BlueEscapeEnvConfig, BlueProcessEnvironmentPool, RainbowDQNAgent, RainbowDQNConfig
from .blue_rl.config_io import configure_blue_mission_duration, load_environment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the independent v1 blue Rainbow-DQN policy")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--missiles", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output", default="outputs/blue_rl/train")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--env-config", default=None)
    parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument("--parallel-envs", type=int, default=1,
                        help="Persistent CPU environment processes sampled in parallel")
    parser.add_argument("--env-worker-threads", type=int, default=1)
    parser.add_argument("--env-worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--updates-per-transition", type=float, default=1.0,
                        help="Gradient updates scheduled per collected environment transition")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print and archive one aggregate row every N completed episodes")
    parser.add_argument("--metrics-path", default=None,
                        help="Final JSON metrics path; defaults to OUTPUT/training_metrics.json")
    parser.add_argument("--jsonl-path", default=None,
                        help="Streaming JSONL path; defaults to OUTPUT/training.jsonl")
    parser.add_argument("--acmi-interval", type=int, default=1,
                        help="Save one training ACMI every N episodes; use 0 to disable ACMI output")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    return parser


def _emit(row: dict[str, Any], rows: list[dict[str, Any]], jsonl_path: Path) -> None:
    rows.append(row)
    encoded = json.dumps(row, ensure_ascii=True, allow_nan=False)
    print(encoded, flush=True)
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _device_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda": return {}
    return {
        "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
        "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
    }


def main() -> int:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.checkpoint_interval < 1 or args.log_interval < 1:
        raise SystemExit("episodes, checkpoint interval and log interval must be positive")
    if args.acmi_interval < 0 or args.parallel_envs < 1 or args.env_worker_threads < 1:
        raise SystemExit("ACMI interval must be non-negative and worker counts positive")
    if args.env_worker_timeout_s <= 0 or args.updates_per_transition < 0 or args.batch_size < 1:
        raise SystemExit("timeout and batch size must be positive; update ratio must be non-negative")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output / "training_metrics.json"
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else output / "training.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True); jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("", encoding="utf-8")
    environment_config = configure_blue_mission_duration(load_environment_config(args.env_config))
    env_config = BlueEscapeEnvConfig(args.missiles, decision_interval_s=args.decision_interval,
                                     acmi_episode_interval=args.acmi_interval,
                                     acmi_directory=str(output / "acmi"))
    rainbow_config = RainbowDQNConfig(6 + args.missiles * 3, 29, batch_size=args.batch_size,
                                      gamma=env_config.shaping_discount, device=args.device)
    pool_size = min(args.parallel_envs, args.episodes)
    # Spawn CPU simulation workers before creating a CUDA context.  On Windows,
    # starting spawned children after CUDA initialization can indefinitely stall
    # all workers while they import PyTorch.
    with BlueProcessEnvironmentPool(environment_config, env_config, pool_size,
                                    native_threads=args.env_worker_threads,
                                    timeout_s=args.env_worker_timeout_s) as pool:
        agent = RainbowDQNAgent(rainbow_config)
        event_rows: list[dict[str, Any]] = []; summaries: list[dict[str, Any]] = []
        started = time.monotonic()
        experiment = {
            "event": "experiment_config", "device": str(agent.device), "red_count": args.missiles,
            "blue_count": 1, "scenario_sampling": "independent_seed_per_episode",
            "parallel_cpu_envs": pool_size, "parallel_backend": "process_spawn",
            "env_worker_threads": args.env_worker_threads, "env_worker_timeout_s": args.env_worker_timeout_s,
            "inference_batch_size_max": pool_size, "training_batch_size": args.batch_size,
            "training_mode": "rainbow_off_policy", "episodes": args.episodes,
            "seed": args.seed, "decision_interval_s": args.decision_interval,
            "updates_per_transition": args.updates_per_transition, "completed_environment_transitions": 0,
            "completed_optimizer_updates": 0, "completed_target_updates": 0,
            "rainbow_config": asdict(rainbow_config), "blue_environment_config": asdict(env_config),
            "environment_config": asdict(environment_config),
            "metrics_path": str(metrics_path), "jsonl_path": str(jsonl_path),
        }
        device_row: dict[str, Any] = {"event": "device", "device": str(agent.device)}
        if agent.device.type == "cuda":
            device_row.update({"cuda_device_name": torch.cuda.get_device_name(agent.device),
                               "cuda_device_count": torch.cuda.device_count()})
        _emit(device_row, event_rows, jsonl_path)
        _emit(experiment, event_rows, jsonl_path)

        observations: dict[int, np.ndarray] = {}; episode_by_worker: dict[int, int] = {}
        reward_by_worker: dict[int, float] = {}; decisions_by_worker: dict[int, int] = {}
        reward_components_by_worker: dict[int, Counter[str]] = {}
        reward_diagnostics_by_worker: dict[int, Counter[str]] = {}
        next_episode = 1; completed = 0; update_credit = 0.0
        next_checkpoint = args.checkpoint_interval; next_log = args.log_interval; vector_iterations = 0
        window_episodes: list[dict[str, Any]] = []; window_losses: list[float] = []
        window_grad_norms: list[float] = []; window_values: list[float] = []; window_actions: Counter[int] = Counter()
        window_clamp_low: list[float] = []; window_clamp_high: list[float] = []
        window_started = started; window_transition_start = 0; window_update_start = 0
        _emit({"event": "environment_workers", "backend": "process_spawn",
               "worker_count": pool_size, "workers": list(pool.worker_info)}, event_rows, jsonl_path)
        assignments = {}
        for worker in range(pool_size):
            assignments[worker] = (args.seed + next_episode, next_episode)
            episode_by_worker[worker] = next_episode; reward_by_worker[worker] = 0.0
            reward_components_by_worker[worker] = Counter()
            reward_diagnostics_by_worker[worker] = Counter()
            decisions_by_worker[worker] = 0; next_episode += 1
        for worker, (observation, _) in pool.reset(assignments).items(): observations[worker] = observation
        while observations:
            vector_iterations += 1; workers = sorted(observations)
            selected = agent.select_actions(np.stack([observations[worker] for worker in workers]))
            actions = {worker: int(action) for worker, action in zip(workers, selected)}
            window_actions.update(actions.values())
            window_values.append(agent.last_action_metrics["selected_value_mean"])
            results = pool.step(actions); resets: dict[int, tuple[int, int]] = {}
            for worker in workers:
                result = results[worker]; previous = observations[worker]
                done = result.terminated or result.truncated
                agent.observe_for_env(worker, previous, actions[worker], result.reward, result.observation, done)
                reward_by_worker[worker] += result.reward; decisions_by_worker[worker] += 1
                for name, value in dict(result.info.get("reward_components", {})).items():
                    if name != "threat_potential":
                        reward_components_by_worker[worker][str(name)] += float(value)
                for name, value in dict(result.info.get("reward_diagnostics", {})).items():
                    reward_diagnostics_by_worker[worker][str(name)] += float(value)
                observations[worker] = result.observation
                if done:
                    completed += 1
                    row = {
                        "episode": episode_by_worker[worker], "reward": reward_by_worker[worker],
                        "blue_survived": bool(result.info["blue_survived"]),
                        "termination_reason": result.info.get("termination_reason"),
                        "hit_count": int(result.info.get("hit_count", 0)),
                        "miss_distance_m": float(result.info.get("miss_distance_m", 0.0)),
                        "simulation_time_s": float(result.info.get("time_s", 0.0)),
                        "physics_steps": int(result.info.get("step_count", 0)),
                        "decision_steps": decisions_by_worker[worker],
                        "reward_components": dict(reward_components_by_worker[worker]),
                        "reward_diagnostics_mean": {
                            name: value / decisions_by_worker[worker]
                            for name, value in reward_diagnostics_by_worker[worker].items()
                        },
                        "red_loss_reasons": list(result.info.get("red_loss_reasons", [])),
                        "mean_loss": agent.last_update_metrics.get("loss"),
                        "learner_loss_at_completion": agent.last_update_metrics.get("loss"),
                        "acmi_path": result.info.get("acmi_path"),
                    }
                    summaries.append(row); window_episodes.append(row)
                    if next_episode <= args.episodes:
                        resets[worker] = (args.seed + next_episode, next_episode)
                        episode_by_worker[worker] = next_episode; reward_by_worker[worker] = 0.0
                        reward_components_by_worker[worker] = Counter()
                        reward_diagnostics_by_worker[worker] = Counter()
                        decisions_by_worker[worker] = 0; next_episode += 1
                    else:
                        observations.pop(worker); episode_by_worker.pop(worker)
                        reward_by_worker.pop(worker); decisions_by_worker.pop(worker)
                        reward_components_by_worker.pop(worker)
                        reward_diagnostics_by_worker.pop(worker)
            update_credit += len(workers) * args.updates_per_transition
            updates = math.floor(update_credit); update_credit -= updates
            for _ in range(updates):
                loss = agent.update()
                if loss is not None:
                    window_losses.append(loss)
                    window_grad_norms.append(agent.last_update_metrics["gradient_norm"])
                    window_clamp_low.append(agent.last_update_metrics["c51_clamp_low_fraction"])
                    window_clamp_high.append(agent.last_update_metrics["c51_clamp_high_fraction"])
            while completed >= next_checkpoint:
                checkpoint = output / f"blue_rainbow_ep{next_checkpoint:06d}.pt"
                agent.save(str(checkpoint))
                _emit({"event": "checkpoint", "episode_threshold": next_checkpoint,
                       "completed_episodes": completed, "path": str(checkpoint),
                       "environment_transitions": agent.total_steps,
                       "optimizer_updates": agent.optimizer_updates}, event_rows, jsonl_path)
                next_checkpoint += args.checkpoint_interval
            if completed >= next_log or (completed == args.episodes and window_episodes):
                now = time.monotonic(); rewards = [float(row["reward"]) for row in window_episodes]
                misses = [float(row["miss_distance_m"]) for row in window_episodes]
                simulation_times = [float(row["simulation_time_s"]) for row in window_episodes]
                action_total = sum(window_actions.values())
                probabilities = np.asarray(list(window_actions.values()), dtype=np.float64) / max(action_total, 1)
                report = {
                    "event": "iteration", "iteration": len([row for row in event_rows if row["event"] == "iteration"]) + 1,
                    "completed_episodes": completed,
                    "episode_ids": sorted(int(row["episode"]) for row in window_episodes),
                    "sampled_red_count": args.missiles, "sampled_blue_count": 1,
                    "active_parallel_envs": len(observations), "inference_batch_size": len(workers),
                    "vector_iterations": vector_iterations,
                    "rollout_decision_steps": agent.total_steps - window_transition_start,
                    "completed_environment_transitions": agent.total_steps,
                    "reward_mean": _mean(rewards), "reward_std": float(np.std(rewards)) if rewards else None,
                    "reward_min": min(rewards) if rewards else None, "reward_max": max(rewards) if rewards else None,
                    "survival_rate": _mean([float(row["blue_survived"]) for row in window_episodes]),
                    "hit_count_sum": sum(int(row["hit_count"]) for row in window_episodes),
                    "miss_distance_mean_m": _mean(misses),
                    "miss_distance_p95_m": float(np.percentile(misses, 95)) if misses else None,
                    "simulation_time_mean_s": _mean(simulation_times),
                    "termination_counts": dict(Counter(str(row["termination_reason"]) for row in window_episodes)),
                    "rollout_diagnostics": {
                        "wall_time_s": now - window_started,
                        "transitions_per_s": (agent.total_steps - window_transition_start) / max(now - window_started, 1e-9),
                        "episodes_per_hour": len(window_episodes) * 3600.0 / max(now - window_started, 1e-9),
                    },
                    "policy_diagnostics": {
                        "action_histogram": {str(key): value for key, value in sorted(window_actions.items())},
                        "action_entropy": float(-(probabilities * np.log(probabilities + 1e-12)).sum()),
                        "selected_value_mean": _mean(window_values),
                    },
                    "loss_mean": _mean(window_losses),
                    "loss_std": float(np.std(window_losses)) if window_losses else None,
                    "gradient_norm_mean": _mean(window_grad_norms),
                    "learning_rate": float(agent.optimizer.param_groups[0]["lr"]),
                    "replay_size": agent.replay.size, "replay_capacity": agent.config.replay_size,
                    "per_beta": agent.last_update_metrics.get("per_beta"),
                    "priority_mean": agent.last_update_metrics.get("priority_mean"),
                    "priority_max": agent.last_update_metrics.get("priority_max"),
                    "c51_clamp_low_fraction_mean": _mean(window_clamp_low),
                    "c51_clamp_high_fraction_mean": _mean(window_clamp_high),
                    "completed_optimizer_updates": agent.optimizer_updates,
                    "optimizer_updates_in_window": agent.optimizer_updates - window_update_start,
                    "completed_target_updates": agent.target_updates,
                    **_device_metrics(agent.device),
                }
                _emit(report, event_rows, jsonl_path)
                while next_log <= completed: next_log += args.log_interval
                window_episodes = []; window_losses = []; window_grad_norms = []; window_values = []
                window_clamp_low = []; window_clamp_high = []
                window_actions = Counter(); window_started = now
                window_transition_start = agent.total_steps; window_update_start = agent.optimizer_updates
            if resets:
                for worker, (observation, _) in pool.reset(resets).items(): observations[worker] = observation
        summaries.sort(key=lambda row: int(row["episode"]))
        final_checkpoint = output / "blue_rainbow.pt"; agent.save(str(final_checkpoint))
        elapsed = time.monotonic() - started
        final_summary = {
            "event": "training_complete", "episodes": args.episodes,
            "survival_rate": _mean([float(row["blue_survived"]) for row in summaries]),
            "reward_mean": _mean([float(row["reward"]) for row in summaries]),
            "elapsed_s": elapsed, "episodes_per_hour": args.episodes * 3600.0 / max(elapsed, 1e-9),
            "completed_environment_transitions": agent.total_steps,
            "completed_optimizer_updates": agent.optimizer_updates,
            "completed_target_updates": agent.target_updates, "replay_size": agent.replay.size,
            "final_checkpoint": str(final_checkpoint), **_device_metrics(agent.device),
        }
        _emit(final_summary, event_rows, jsonl_path)
        (output / "episodes.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps({"experiment_config": experiment, "iterations": [row for row in event_rows if row["event"] == "iteration"],
                                            "episodes": summaries, "final_summary": final_summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
