from __future__ import annotations

import argparse
import json
from pathlib import Path

from .blue_rl import BlueEscapeEnv, BlueEscapeEnvConfig, RainbowDQNAgent
from .blue_rl.config_io import load_environment_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a blue Rainbow checkpoint with ACMI output")
    parser.add_argument("checkpoint"); parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--missiles", type=int, choices=range(1, 5), default=1); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="outputs/blue_rl/test"); parser.add_argument("--device", default="cpu")
    parser.add_argument("--env-config", default=None); parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument("--acmi-interval", type=int, default=1,
                        help="Save one evaluation ACMI every N episodes; use 0 to disable ACMI output"); args = parser.parse_args()
    if args.episodes < 1: parser.error("episodes must be positive")
    if args.acmi_interval < 0: parser.error("ACMI interval must be non-negative")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    environment_config = load_environment_config(args.env_config)
    env = BlueEscapeEnv(environment_config, BlueEscapeEnvConfig(
        args.missiles,
        decision_interval_s=args.decision_interval,
        acmi_episode_interval=args.acmi_interval,
        acmi_directory=str(output / "acmi"),
    ))
    agent = RainbowDQNAgent.load(args.checkpoint, args.device); wins = 0; rows = []
    if agent.config.observation_dim != env.observation_dim or agent.config.action_dim != env.action_dim:
        raise ValueError(
            f"checkpoint dimensions ({agent.config.observation_dim}, {agent.config.action_dim}) do not match "
            f"the requested scenario ({env.observation_dim}, {env.action_dim}); check --missiles"
        )
    for episode in range(1, args.episodes + 1):
        observation, _ = env.reset(args.seed + episode); done = False; reward_sum = 0.0
        while not done:
            observation, reward, terminated, truncated, info = env.step(agent.select_action(observation, evaluation=True))
            reward_sum += reward; done = terminated or truncated
        wins += int(info["blue_survived"]); rows.append({"episode": episode, "reward": reward_sum, **info})
    summary = {"episodes": args.episodes, "survival_rate": wins / max(args.episodes, 1), "results": rows}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8"); print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
