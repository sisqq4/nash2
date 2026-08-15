from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from .blue_rl import BlueEscapeEnv, BlueEscapeEnvConfig, RainbowDQNAgent, RainbowDQNConfig
from .blue_rl.config_io import load_environment_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the independent v1 blue Rainbow-DQN policy")
    parser.add_argument("--episodes", type=int, default=1000); parser.add_argument("--missiles", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output", default="outputs/blue_rl/train")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--env-config", default=None)
    parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument(
        "--acmi-interval",
        type=int,
        default=1,
        help="Save one training ACMI every N episodes; use 0 to disable ACMI output",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=50); args = parser.parse_args()
    if args.episodes < 1 or args.checkpoint_interval < 1: parser.error("episodes and checkpoint interval must be positive")
    if args.acmi_interval < 0: parser.error("ACMI interval must be non-negative")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    environment_config = load_environment_config(args.env_config)
    env = BlueEscapeEnv(environment_config, BlueEscapeEnvConfig(
        args.missiles,
        decision_interval_s=args.decision_interval,
        acmi_episode_interval=args.acmi_interval,
        acmi_directory=str(output / "acmi"),
    ))
    agent = RainbowDQNAgent(RainbowDQNConfig(env.observation_dim, env.action_dim, device=args.device))
    summaries = []
    for episode in range(1, args.episodes + 1):
        observation, _ = env.reset(args.seed + episode); done = False; total_reward = 0.0; losses = []
        while not done:
            action = agent.select_action(observation); next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated; agent.observe(observation, action, reward, next_observation, done)
            loss = agent.update()
            if loss is not None: losses.append(loss)
            observation, total_reward = next_observation, total_reward + reward
        summaries.append({"episode": episode, "reward": total_reward, "blue_survived": info["blue_survived"],
                          "mean_loss": sum(losses) / len(losses) if losses else None, "acmi_path": info.get("acmi_path")})
        if episode % 10 == 0: print(json.dumps(summaries[-1]))
        if episode % args.checkpoint_interval == 0:
            agent.save(str(output / f"blue_rainbow_ep{episode:06d}.pt"))
    agent.save(str(output / "blue_rainbow.pt")); (output / "episodes.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__": raise SystemExit(main())
