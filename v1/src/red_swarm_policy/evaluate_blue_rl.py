from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .blue_rl import BlueEscapeEnvConfig, BlueProcessEnvironmentPool, RainbowDQNAgent
from .blue_rl.config_io import configure_blue_mission_duration, load_environment_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a blue Rainbow checkpoint with batched inference")
    parser.add_argument("checkpoint"); parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--missiles", type=int, choices=range(1, 5), default=1); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/blue_rl/test"); parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-config", default=None); parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument("--parallel-envs", type=int, default=1); parser.add_argument("--env-worker-threads", type=int, default=1)
    parser.add_argument("--env-worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--acmi-interval", type=int, default=1,
                        help="Save one evaluation ACMI every N episodes; use 0 to disable ACMI output")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.episodes < 1 or args.parallel_envs < 1 or args.env_worker_threads < 1: raise SystemExit("episode and worker counts must be positive")
    if args.acmi_interval < 0 or args.env_worker_timeout_s <= 0: raise SystemExit("invalid ACMI interval or worker timeout")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    environment_config = configure_blue_mission_duration(load_environment_config(args.env_config))
    config = BlueEscapeEnvConfig(args.missiles, decision_interval_s=args.decision_interval,
                                 acmi_episode_interval=args.acmi_interval,
                                 acmi_directory=str(output / "acmi"))
    agent = RainbowDQNAgent.load(args.checkpoint, args.device)
    observation_dim, action_dim = 6 + args.missiles * 3, 29
    if agent.config.observation_dim != observation_dim or agent.config.action_dim != action_dim:
        raise ValueError(f"checkpoint dimensions ({agent.config.observation_dim}, {agent.config.action_dim}) do not match the requested scenario ({observation_dim}, {action_dim}); check --missiles")
    pool_size = min(args.parallel_envs, args.episodes); observations = {}; episode_by_worker = {}; rewards = {}
    rows: list[dict[str, object]] = []; next_episode = 1
    with BlueProcessEnvironmentPool(environment_config, config, pool_size,
                                    native_threads=args.env_worker_threads,
                                    timeout_s=args.env_worker_timeout_s) as pool:
        assignments = {}
        for worker in range(pool_size):
            assignments[worker] = (args.seed + next_episode, next_episode)
            episode_by_worker[worker] = next_episode; rewards[worker] = 0.0; next_episode += 1
        for worker, (observation, _) in pool.reset(assignments).items(): observations[worker] = observation
        while observations:
            workers = sorted(observations)
            actions = agent.select_actions(np.stack([observations[w] for w in workers]), evaluation=True)
            results = pool.step({worker: int(action) for worker, action in zip(workers, actions)})
            resets = {}
            for worker in workers:
                result = results[worker]; rewards[worker] += result.reward; observations[worker] = result.observation
                if result.terminated or result.truncated:
                    rows.append({"episode": episode_by_worker[worker], "reward": rewards[worker], **result.info})
                    if next_episode <= args.episodes:
                        resets[worker] = (args.seed + next_episode, next_episode)
                        episode_by_worker[worker] = next_episode; rewards[worker] = 0.0; next_episode += 1
                    else:
                        observations.pop(worker); episode_by_worker.pop(worker); rewards.pop(worker)
            if resets:
                for worker, (observation, _) in pool.reset(resets).items(): observations[worker] = observation
    rows.sort(key=lambda row: int(row["episode"]))
    summary = {"episodes": args.episodes,
               "survival_rate": sum(bool(row["blue_survived"]) for row in rows) / args.episodes,
               "parallel_envs": pool_size, "inference_batch_size": pool_size, "results": rows}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
