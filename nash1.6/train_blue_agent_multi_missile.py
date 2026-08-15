"""Train blue agent for fixed-start 1v2 / 1v3 annulus-launched missile scenarios."""

from __future__ import annotations

import argparse
import json
import os
import time
import random
from dataclasses import asdict
from typing import Dict, Any, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import EnvConfig, TrainConfig
from env.acmi_io import write_acmi
from train_blue_agent import make_env_and_agent, save_checkpoint


def build_multi_missile_env_cfg(num_missiles: int) -> EnvConfig:
    cfg = EnvConfig()
    cfg.num_missiles = num_missiles
    cfg.reward_mode = "multi_coop"

    # Blue fixed at (0, 0, 10000 m), heading +x.
    cfg.blue_fixed_start = True
    cfg.blue_fixed_x = 0.0
    cfg.blue_fixed_y = 0.0
    cfg.blue_fixed_z = 10.0
    cfg.blue_heading_min = 0.0
    cfg.blue_heading_max = 0.0

    # Missiles sampled from annulus centered at (0,0), r in [6, 30] km, altitude in [7, 13] km.
    cfg.missile_spawn_mode = "annulus"
    cfg.missile_spawn_radius_min = 6.0
    cfg.missile_spawn_radius_max = 30.0
    cfg.missile_spawn_alt_min = 7.0
    cfg.missile_spawn_alt_max = 13.0

    # Launch-time offset follows zero-mean normal distribution.
    cfg.missile_launch_time_std = 0.5
    cfg.missile_launch_time_clip = 2.0
    return cfg


def run_train_for_scenario(num_missiles: int, episodes: int) -> None:
    env_cfg = build_multi_missile_env_cfg(num_missiles)
    train_cfg = TrainConfig(episodes=episodes, reward_mode="multi_coop")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(env_cfg.save_dir, f"{run_id}_1v{num_missiles}")
    env_cfg.save_dir = run_dir
    train_cfg.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    train_cfg.results_dir = os.path.join(run_dir, "results")
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    os.makedirs(train_cfg.results_dir, exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"env": asdict(env_cfg), "train": asdict(train_cfg)}, f, indent=2, ensure_ascii=False)

    seed = random.randint(0, 2**31 - 1)
    env, agent = make_env_and_agent(env_cfg, train_cfg, seed=seed)

    ep_rewards: List[float] = []
    success_count = 0
    per_episode_rows: List[Dict[str, Any]] = []
    success_rate_points: List[Tuple[int, float]] = []
    for ep in range(1, train_cfg.episodes + 1):
        start_time = time.time()
        obs = env.reset()
        done = False
        total_reward = 0.0
        step = 0
        episode_info: Dict[str, Any] | None = None
        while not done:
            action = agent.select_action(obs, eval_mode=False)
            nxt, rew, done, info = env.step(action)
            episode_info = info
            agent.store_transition(obs, action, rew, nxt, done)
            agent.update()
            obs = nxt
            total_reward += rew
            step += 1
        ep_rewards.append(total_reward)

        is_timeout = bool(episode_info.get("timeout", False)) if episode_info else False
        is_hit = bool(episode_info.get("hit", False)) if episode_info else False
        crashed = bool(episode_info.get("crashed", False)) if episode_info else False
        missiles_exhausted = bool(episode_info.get("missiles_exhausted", False)) if episode_info else False
        episode_success = (is_timeout or missiles_exhausted) and (not is_hit) and (not crashed)
        if episode_success:
            success_count += 1
        success_rate = success_count / ep
        per_episode_rows.append(
            {
                "episode": ep,
                "steps": step,
                "episode_reward": float(total_reward),
                "win": int(episode_success),
                "cumulative_success_rate": float(success_rate),
                "timeout": int(is_timeout),
                "hit": int(is_hit),
                "crashed": int(crashed),
                "missiles_exhausted": int(missiles_exhausted),
            }
        )

        if train_cfg.checkpoint_interval > 0 and ep % train_cfg.checkpoint_interval == 0:
            ckpt_name = f"checkpoint_ep{ep:04d}.pt"
            save_checkpoint(os.path.join(train_cfg.checkpoint_dir, ckpt_name), ep, agent, env)

        if env_cfg.log_trajectories and ep % 10 == 0:
            csv_dir = os.path.join(env_cfg.save_dir, "csv", str(ep))
            if os.path.isdir(csv_dir):
                target_name = f"session_ep{ep:04d}"
                write_acmi(
                    target_name=target_name,
                    source_dir=csv_dir,
                    time_unit=env_cfg.dt,
                    explode_time=10,
                    add_plane_explosion=not episode_success,
                )

        if ep % 10 == 0:
            success_rate_points.append((ep, success_rate))

        if ep % max(1, train_cfg.print_interval) == 0:
            win = min(train_cfg.print_interval, len(ep_rewards))
            avg = sum(ep_rewards[-win:]) / max(win, 1)
            elapsed = time.time() - start_time
            print(
                f"[1v{num_missiles}] Episode {ep}/{train_cfg.episodes}\n"
                f"平均奖励(最近{train_cfg.print_interval})={avg:.4f}\n"
                f"当前胜率={success_count}/{ep} ({success_rate * 100:.1f}%)\n"
                f"本回合步数={step}\n"
                f"本回合耗时={elapsed:.2f}s"
            )

    if success_rate_points:
        x_vals, y_vals = zip(*success_rate_points)
        plt.figure(figsize=(8, 4.5))
        plt.plot(x_vals, y_vals, marker="o")
        plt.title(f"Success Rate Curve (1v{num_missiles})")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(train_cfg.results_dir, "success_rate_curve.png"), dpi=150)
        plt.close()

    if per_episode_rows:
        pd.DataFrame(per_episode_rows).to_csv(
            os.path.join(train_cfg.results_dir, "episode_summary.csv"),
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train fixed-start multi-missile blue policy.")
    parser.add_argument("--episodes", type=int, default=1000, help="Episodes for each scenario.")
    parser.add_argument(
        "--scenario",
        choices=["1v2", "1v3", "both"],
        default="both",
        help="Which scenario(s) to train.",
    )
    args = parser.parse_args()

    scenarios = [2, 3] if args.scenario == "both" else [2 if args.scenario == "1v2" else 3]
    for n in scenarios:
        run_train_for_scenario(n, args.episodes)


if __name__ == "__main__":
    main()