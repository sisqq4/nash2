from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .blue_rl import (BlueEscapeEnv, BlueEscapeEnvConfig, BlueProcessEnvironmentPool,
                      FlightEnvelopeConfig, FlightEnvelopeConstraintLayer,
                      FlightQualityTracker, RainbowDQNAgent, RainbowDQNConfig,
                      append_flight_quality_episode,
                      blue_observation_dim,
                      write_flight_quality_report)
from .blue_rl.config_io import configure_blue_mission_duration, load_environment_config
from .blue_rl.curriculum import CurriculumSchedule, balanced_score, within_forgetting_limit
from .cli_utils import parse_missile_scenarios


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the independent v1 blue Rainbow-DQN policy")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--missiles", default="1",
                        help="Comma-separated scenarios to sample, e.g. 1,2,3,4 (default: 1)")
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
    parser.add_argument("--replay-size", type=int, default=50_000,
                        help="Maximum number of transitions retained by prioritized replay")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print and archive one aggregate row every N completed episodes")
    parser.add_argument("--metrics-path", default=None,
                        help="Final JSON metrics path; defaults to OUTPUT/training_metrics.json")
    parser.add_argument("--jsonl-path", default=None,
                        help="Streaming JSONL path; defaults to OUTPUT/training.jsonl")
    parser.add_argument("--acmi-interval", type=int, default=1,
                        help="Save one training ACMI every N episodes; use 0 to disable ACMI output")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--curriculum", action="store_true",
                        help="Use the staged 1v1-to-1v4 rehearsal curriculum (normalized_v3 54-D network)")
    parser.add_argument("--curriculum-transition-episodes", type=int, default=500,
                        help="Episodes used to linearly ramp probabilities at each curriculum stage entry")
    parser.add_argument("--curriculum-eval-interval", type=int, default=500)
    parser.add_argument("--curriculum-eval-episodes", type=int, default=300,
                        help="Fixed-seed evaluation episodes per introduced scenario; 0 disables evaluation")
    parser.add_argument("--flight-quality-plot-limit", type=int, default=10)
    parser.add_argument("--baseline-survival-rate", type=float, default=None)
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


def _curriculum_evaluation(agent: RainbowDQNAgent, environment_config: Any,
                           env_config: BlueEscapeEnvConfig, scenarios: list[int],
                           episodes: int, seed: int) -> dict[str, Any]:
    """Evaluate without replay writes, optimizer steps, or NoisyNet exploration."""
    evaluation_env = BlueEscapeEnv(environment_config, replace(
        env_config, record_acmi=False, acmi_episode_interval=0,
    ))
    rates: dict[int, float] = {}; mean_times: dict[int, float] = {}
    saved_envelope = agent.config.flight_envelope_config
    envelope_config = (FlightEnvelopeConfig(**saved_envelope) if saved_envelope else
                       FlightEnvelopeConfig(action_prediction_s=env_config.decision_interval_s))
    for scenario in scenarios:
        survived: list[float] = []; times: list[float] = []
        for index in range(episodes):
            observation, reset_info = evaluation_env.reset(
                seed=seed + scenario * 100_000 + index, episode_index=index + 1,
                missile_count=scenario,
            )
            learning_active = bool(reset_info["learning_active"])
            mechanism_state = dict(reset_info["flight_quality_state"])
            constraint_layer = FlightEnvelopeConstraintLayer(envelope_config)
            done = False
            while not done:
                if learning_active:
                    q_values = agent.expected_action_values(observation, evaluation=True)[0]
                    action, diagnostic = constraint_layer.select(q_values, mechanism_state)
                    policy_action = int(diagnostic["raw_action"])
                else:
                    action = policy_action = 0
                observation, _, terminated, truncated, info = evaluation_env.step(
                    action, policy_action=policy_action
                )
                learning_active = bool(info["learning_active"])
                mechanism_state = dict(info["flight_quality_state"])
                done = terminated or truncated
            survived.append(float(info["blue_survived"])); times.append(float(info.get("time_s", 0.0)))
        rates[scenario] = float(np.mean(survived)); mean_times[scenario] = float(np.mean(times))
    return {"survival_rates": rates, "mean_survival_time_s": mean_times,
            "episodes_per_scenario": episodes, "seed_base": seed}


def main() -> int:
    args = build_parser().parse_args()
    try: missile_scenarios = parse_missile_scenarios(args.missiles)
    except ValueError as error: raise SystemExit(str(error)) from error
    if args.episodes < 1 or args.checkpoint_interval < 1 or args.log_interval < 1:
        raise SystemExit("episodes, checkpoint interval and log interval must be positive")
    if args.acmi_interval < 0 or args.parallel_envs < 1 or args.env_worker_threads < 1:
        raise SystemExit("ACMI interval must be non-negative and worker counts positive")
    if (args.env_worker_timeout_s <= 0 or args.updates_per_transition < 0
            or args.batch_size < 1 or args.replay_size < 1):
        raise SystemExit("timeout, batch size, and replay size must be positive; update ratio must be non-negative")
    curriculum = CurriculumSchedule(transition_episodes=args.curriculum_transition_episodes) if args.curriculum else None
    if curriculum is not None and args.episodes > curriculum.total_episodes:
        raise SystemExit(f"curriculum defines at most {curriculum.total_episodes} episodes")
    if args.curriculum_eval_interval < 1 or args.curriculum_eval_episodes < 0:
        raise SystemExit("curriculum evaluation interval must be positive and episodes non-negative")
    if args.flight_quality_plot_limit < 0: raise SystemExit("flight-quality plot limit must be non-negative")
    if args.baseline_survival_rate is not None and not 0.0 <= args.baseline_survival_rate <= 1.0:
        raise SystemExit("baseline survival rate must be in [0, 1]")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output / "training_metrics.json"
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else output / "training.jsonl"
    flight_quality_jsonl_path = output / "flight_quality" / "flight_quality_episodes.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True); jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path.write_text("", encoding="utf-8")
    flight_quality_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    flight_quality_jsonl_path.write_text("", encoding="utf-8")
    environment_config = configure_blue_mission_duration(load_environment_config(args.env_config))
    training_scenarios = (1, 2, 3, 4) if curriculum is not None else missile_scenarios
    env_config = BlueEscapeEnvConfig(training_scenarios[0], max_missiles=max(training_scenarios),
                                     pad_observation_to_max_missiles=curriculum is not None or len(training_scenarios) > 1,
                                     observation_schema="normalized_v3",
                                     decision_interval_s=args.decision_interval,
                                     acmi_episode_interval=args.acmi_interval,
                                     acmi_directory=str(output / "acmi"))
    envelope_config = FlightEnvelopeConfig(action_prediction_s=args.decision_interval)
    rainbow_config = RainbowDQNConfig(
        blue_observation_dim(env_config.observation_schema, max(training_scenarios)), 29,
        observation_schema=env_config.observation_schema,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        gamma=env_config.shaping_discount, device=args.device,
        flight_envelope_config=asdict(envelope_config),
    )
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
            "event": "experiment_config", "device": str(agent.device), "red_counts": list(training_scenarios),
            "blue_count": 1, "scenario_sampling": ("staged_curriculum" if curriculum is not None
                                                     else "uniform_random_per_episode"),
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
            "flight_quality_jsonl_path": str(flight_quality_jsonl_path),
            "curriculum": curriculum.describe() if curriculum is not None else None,
        }
        device_row: dict[str, Any] = {"event": "device", "device": str(agent.device)}
        if agent.device.type == "cuda":
            device_row.update({"cuda_device_name": torch.cuda.get_device_name(agent.device),
                               "cuda_device_count": torch.cuda.device_count()})
        _emit(device_row, event_rows, jsonl_path)
        _emit(experiment, event_rows, jsonl_path)

        scenario_rng = random.Random(args.seed)
        episode_scenarios = {episode: (curriculum.sample(episode, scenario_rng) if curriculum is not None
                                      else scenario_rng.choice(missile_scenarios))
                             for episode in range(1, args.episodes + 1)}
        episode_stages = {episode: curriculum.stage_at(episode)[1].name
                          for episode in range(1, args.episodes + 1)} if curriculum is not None else {}
        observations: dict[int, np.ndarray] = {}; episode_by_worker: dict[int, int] = {}
        learning_active_by_worker: dict[int, bool] = {}
        reward_by_worker: dict[int, float] = {}; decisions_by_worker: dict[int, int] = {}
        reward_components_by_worker: dict[int, Counter[str]] = {}
        reward_diagnostics_by_worker: dict[int, Counter[str]] = {}
        envelope_states: dict[int, dict[str, object]] = {}
        constraints = {worker: FlightEnvelopeConstraintLayer(envelope_config)
                       for worker in range(pool_size)}
        quality: dict[int, FlightQualityTracker] = {}; quality_episodes: list[dict[str, Any]] = []
        next_episode = 1; completed = 0; update_credit = 0.0
        next_checkpoint = args.checkpoint_interval; next_log = args.log_interval; vector_iterations = 0
        next_curriculum_eval = args.curriculum_eval_interval
        historical_best: dict[int, float] = {}; best_balanced_score = -math.inf
        best_new_rate: dict[int, float] = {}
        window_episodes: list[dict[str, Any]] = []; window_losses: list[float] = []
        window_grad_norms: list[float] = []; window_values: list[float] = []
        window_actions: Counter[int] = Counter(); window_policy_actions: Counter[int] = Counter()
        window_constraint_interventions = 0
        window_clamp_low: list[float] = []; window_clamp_high: list[float] = []
        window_started = started; window_transition_start = 0; window_update_start = 0
        _emit({"event": "environment_workers", "backend": "process_spawn",
               "worker_count": pool_size, "workers": list(pool.worker_info)}, event_rows, jsonl_path)
        assignments = {}
        for worker in range(pool_size):
            assignments[worker] = (args.seed + next_episode, next_episode, episode_scenarios[next_episode])
            episode_by_worker[worker] = next_episode; reward_by_worker[worker] = 0.0
            reward_components_by_worker[worker] = Counter()
            reward_diagnostics_by_worker[worker] = Counter()
            decisions_by_worker[worker] = 0; next_episode += 1
            quality[worker] = FlightQualityTracker()
        for worker, (observation, info) in pool.reset(assignments).items():
            observations[worker] = observation
            learning_active_by_worker[worker] = bool(info["learning_active"])
            quality[worker].add(info["flight_quality_state"])
            envelope_states[worker] = dict(info["flight_quality_state"])
            constraints[worker].reset()
        while observations:
            vector_iterations += 1; workers = sorted(observations)
            active_workers = [worker for worker in workers if learning_active_by_worker[worker]]
            actions = {worker: 0 for worker in workers}
            policy_actions = {worker: 0 for worker in workers}
            action_diagnostics: dict[int, dict[str, object]] = {}
            if active_workers:
                q_values = agent.expected_action_values(
                    np.stack([observations[worker] for worker in active_workers]), evaluation=False
                )
                selected = [constraints[worker].select(q, envelope_states[worker])
                            for worker, q in zip(active_workers, q_values)]
                for worker, q, (action, diagnostic) in zip(active_workers, q_values, selected):
                    actions[worker] = int(action)
                    policy_actions[worker] = int(diagnostic["raw_action"])
                    action_diagnostics[worker] = diagnostic
                    window_values.append(float(q[action]))
            results = pool.step(actions, policy_actions=policy_actions); resets: dict[int, tuple[int, int]] = {}
            learning_transition_count = 0
            for worker in workers:
                result = results[worker]; previous = observations[worker]
                done = result.terminated or result.truncated
                learning_transition = bool(result.info["learning_transition"])
                executed_action = int(result.info["executed_action_index"])
                if learning_transition:
                    if executed_action != actions[worker]:
                        raise RuntimeError("replay action must equal the post-constraint executed action")
                    next_mask, next_penalty, _ = constraints[worker].constraints(
                        result.info["flight_quality_state"]
                    )
                    agent.observe_for_env(
                        worker, previous, executed_action, result.reward, result.observation, done,
                        next_action_mask=next_mask, next_action_penalty=next_penalty,
                    )
                    learning_transition_count += 1
                    decisions_by_worker[worker] += 1
                    window_actions[executed_action] += 1
                    window_policy_actions[policy_actions[worker]] += 1
                    window_constraint_interventions += int(
                        bool(action_diagnostics[worker]["intervened"])
                    )
                    for name, value in dict(result.info.get("reward_components", {})).items():
                        if name != "threat_potential":
                            reward_components_by_worker[worker][str(name)] += float(value)
                    for name, value in dict(result.info.get("reward_diagnostics", {})).items():
                        reward_diagnostics_by_worker[worker][str(name)] += float(value)
                reward_by_worker[worker] += result.reward
                observations[worker] = result.observation
                learning_active_by_worker[worker] = bool(result.info["learning_active"])
                envelope_states[worker] = dict(result.info["flight_quality_state"])
                diagnostic = action_diagnostics.get(worker, {}) if learning_transition else {}
                quality[worker].add(result.info["flight_quality_state"],
                                    policy_action=policy_actions[worker] if learning_transition else None,
                                    executed_action=executed_action,
                                    safety_intervened=bool(diagnostic.get("intervened", False)),
                                    safety_reasons=list(diagnostic.get("raw_action_hard_violation_reasons", [])))
                if done:
                    completed += 1
                    row = {
                        "episode": episode_by_worker[worker], "reward": reward_by_worker[worker],
                        "missile_count": episode_scenarios[episode_by_worker[worker]],
                        "curriculum_stage": episode_stages.get(episode_by_worker[worker]),
                        "blue_survived": bool(result.info["blue_survived"]),
                        "termination_reason": result.info.get("termination_reason"),
                        "hit_count": int(result.info.get("hit_count", 0)),
                        "miss_distance_m": float(result.info.get("miss_distance_m", 0.0)),
                        "simulation_time_s": float(result.info.get("time_s", 0.0)),
                        "physics_steps": int(result.info.get("step_count", 0)),
                        "decision_steps": decisions_by_worker[worker],
                        "learning_activation_time_s": result.info.get("learning_activation_time_s"),
                        "learning_activation_range_m": result.info.get("learning_activation_range_m"),
                        "threat_observed": bool(result.info.get("learning_active", False)),
                        "reward_components": dict(reward_components_by_worker[worker]),
                        "reward_diagnostics_mean": {
                            name: value / decisions_by_worker[worker]
                            for name, value in reward_diagnostics_by_worker[worker].items()
                        },
                        "red_loss_reasons": list(result.info.get("red_loss_reasons", [])),
                        "mean_loss": (
                            agent.last_update_metrics.get("loss")
                            if decisions_by_worker[worker] > 0 else None
                        ),
                        "learner_loss_at_completion": (
                            agent.last_update_metrics.get("loss")
                            if decisions_by_worker[worker] > 0 else None
                        ),
                        "acmi_path": result.info.get("acmi_path"),
                    }
                    summaries.append(row); window_episodes.append(row)
                    quality_result = quality[worker].finish(episode=episode_by_worker[worker],
                                                            survived=bool(result.info["blue_survived"]))
                    row["flight_quality"] = {key: quality_result[key] for key in ("metrics", "verdicts", "events")}
                    quality_episodes.append(quality_result)
                    append_flight_quality_episode(quality_result, flight_quality_jsonl_path)
                    if next_episode <= args.episodes:
                        resets[worker] = (args.seed + next_episode, next_episode, episode_scenarios[next_episode])
                        episode_by_worker[worker] = next_episode; reward_by_worker[worker] = 0.0
                        reward_components_by_worker[worker] = Counter()
                        reward_diagnostics_by_worker[worker] = Counter()
                        decisions_by_worker[worker] = 0; next_episode += 1
                        quality[worker] = FlightQualityTracker()
                        constraints[worker].reset()
                    else:
                        observations.pop(worker); episode_by_worker.pop(worker)
                        learning_active_by_worker.pop(worker)
                        reward_by_worker.pop(worker); decisions_by_worker.pop(worker)
                        reward_components_by_worker.pop(worker)
                        reward_diagnostics_by_worker.pop(worker)
                        quality.pop(worker)
                        envelope_states.pop(worker)
            # Replay insertion and loss/gradient computation are both driven
            # exclusively by post-detection transitions.
            update_credit += learning_transition_count * args.updates_per_transition
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
            while (curriculum is not None and args.curriculum_eval_episodes > 0
                   and completed >= next_curriculum_eval):
                _, evaluation_stage, _ = curriculum.stage_at(min(next_curriculum_eval, args.episodes))
                introduced = [index + 1 for index, probability in enumerate(evaluation_stage.probabilities)
                              if probability > 0.0]
                evaluation = _curriculum_evaluation(
                    agent, environment_config, env_config, introduced,
                    args.curriculum_eval_episodes, args.seed + 10_000_000,
                )
                rates = {int(key): float(value) for key, value in evaluation["survival_rates"].items()}
                eligible = within_forgetting_limit(rates, historical_best)
                score = balanced_score(rates, evaluation_stage.score_weights)
                newest = max(introduced)
                if rates[newest] > best_new_rate.get(newest, -1.0):
                    best_new_rate[newest] = rates[newest]
                    agent.save(str(output / "best_new_scenario.pt"))
                if eligible and score > best_balanced_score:
                    best_balanced_score = score
                    agent.save(str(output / "best_balanced.pt"))
                for scenario, rate in rates.items():
                    historical_best[scenario] = max(historical_best.get(scenario, 0.0), rate)
                _emit({"event": "curriculum_evaluation", "completed_episodes": completed,
                       "episode_threshold": next_curriculum_eval, "stage": evaluation_stage.name,
                       **evaluation, "score": score, "score_weights": evaluation_stage.score_weights,
                       "within_five_point_forgetting_limit": eligible,
                       "historical_best_survival_rates": dict(historical_best)}, event_rows, jsonl_path)
                next_curriculum_eval += args.curriculum_eval_interval
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
                    "sampled_red_counts": dict(Counter(int(row["missile_count"]) for row in window_episodes)),
                    "sampled_blue_count": 1,
                    "active_parallel_envs": len(observations), "inference_batch_size": len(active_workers),
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
                        "policy_action_histogram": {str(key): value for key, value in sorted(window_policy_actions.items())},
                        "action_histogram": {str(key): value for key, value in sorted(window_actions.items())},
                        "action_entropy": float(-(probabilities * np.log(probabilities + 1e-12)).sum()),
                        "selected_value_mean": _mean(window_values),
                        "constraint_intervention_count": window_constraint_interventions,
                        "constraint_intervention_rate": window_constraint_interventions / max(action_total, 1),
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
                window_actions = Counter(); window_policy_actions = Counter()
                window_constraint_interventions = 0; window_started = now
                window_transition_start = agent.total_steps; window_update_start = agent.optimizer_updates
            if resets:
                for worker, (observation, info) in pool.reset(resets).items():
                    observations[worker] = observation
                    learning_active_by_worker[worker] = bool(info["learning_active"])
                    quality[worker].add(info["flight_quality_state"])
                    envelope_states[worker] = dict(info["flight_quality_state"])
                    constraints[worker].reset()
        summaries.sort(key=lambda row: int(row["episode"]))
        final_checkpoint = output / "blue_rainbow.pt"; agent.save(str(final_checkpoint))
        elapsed = time.monotonic() - started
        flight_quality_summary = write_flight_quality_report(
            sorted(quality_episodes, key=lambda item: int(item["episode"])), output / "flight_quality",
            baseline_survival_rate=args.baseline_survival_rate, plot_limit=args.flight_quality_plot_limit)
        final_summary = {
            "event": "training_complete", "episodes": args.episodes,
            "survival_rate": _mean([float(row["blue_survived"]) for row in summaries]),
            "reward_mean": _mean([float(row["reward"]) for row in summaries]),
            "elapsed_s": elapsed, "episodes_per_hour": args.episodes * 3600.0 / max(elapsed, 1e-9),
            "completed_environment_transitions": agent.total_steps,
            "completed_optimizer_updates": agent.optimizer_updates,
            "completed_target_updates": agent.target_updates, "replay_size": agent.replay.size,
            "final_checkpoint": str(final_checkpoint), **_device_metrics(agent.device),
            "flight_quality": flight_quality_summary,
        }
        final_summary["scenario_episode_counts"] = dict(Counter(
            int(row["missile_count"]) for row in summaries
        ))
        _emit(final_summary, event_rows, jsonl_path)
        (output / "episodes.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps({"experiment_config": experiment, "iterations": [row for row in event_rows if row["event"] == "iteration"],
                                            "curriculum_evaluations": [row for row in event_rows if row["event"] == "curriculum_evaluation"],
                                            "episodes": summaries, "final_summary": final_summary}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
