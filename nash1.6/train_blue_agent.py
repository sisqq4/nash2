
"""Training script for the blue escape agent."""

from __future__ import annotations

import os
import time
import json
import gc
from collections import deque
from dataclasses import asdict
from typing import Tuple, Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from config import EnvConfig, TrainConfig
from env.escape_env import EscapeEnv
from env.acmi_io import write_acmi
from agent.dqn_agent import DQNAgent, DQNConfig



def make_env_and_agent(
    env_cfg: EnvConfig,
    train_cfg: TrainConfig,
    seed: int = 0,
) -> Tuple[EscapeEnv, DQNAgent]:
    env = EscapeEnv(env_cfg, seed=seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dqn_cfg = DQNConfig(
        obs_dim=env.observation_dim,
        action_dim=env.action_dim,
        lr=train_cfg.lr,
        gamma=train_cfg.gamma,
        batch_size=train_cfg.batch_size,
        replay_size=train_cfg.replay_size,
        start_learning=train_cfg.start_learning,
        epsilon_start=train_cfg.epsilon_start,
        epsilon_end=train_cfg.epsilon_end,
        epsilon_decay=train_cfg.epsilon_decay,
        target_update_interval=train_cfg.target_update_interval,
        device=device,
    )
    agent = DQNAgent(dqn_cfg)
    return env, agent


def save_checkpoint(
    path: str,
    episode: int,
    agent: DQNAgent,
    env: EscapeEnv,
) -> None:
    payload: Dict[str, Any] = {
        "episode": episode,
        "blue": agent.get_state(),
        "red": env.get_red_params(),
    }
    torch.save(payload, path)


def load_checkpoint(
    path: str,
    agent: DQNAgent,
    env: EscapeEnv,
    load_blue: bool = True,
    load_red: bool = True,
) -> Dict[str, Any]:
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    if load_blue:
        if "blue" not in payload:
            raise KeyError(f"Checkpoint does not contain requested blue agent state: {path}")
        agent.load_state(payload["blue"])
        if not getattr(agent, "loaded_from_checkpoint", False):
            raise RuntimeError(f"Blue agent state was not marked as loaded after reading checkpoint: {path}")
    if load_red:
        if "red" not in payload:
            raise KeyError(f"Checkpoint does not contain requested red parameters: {path}")
        env.set_red_params(payload["red"])
    return payload

def train() -> None:
    env_cfg = EnvConfig()
    train_cfg = TrainConfig()
    env_cfg.reward_mode = train_cfg.reward_mode

    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(env_cfg.save_dir, run_id)
    env_cfg.save_dir = run_dir
    train_cfg.checkpoint_dir = os.path.join(run_dir, "checkpoints")
    train_cfg.results_dir = os.path.join(run_dir, "results")

    os.makedirs(run_dir, exist_ok=True)
    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(
            {"env": asdict(env_cfg), "train": asdict(train_cfg)},
            f,
            indent=2,
            ensure_ascii=False,
        )

    if env_cfg.log_trajectories:
        os.makedirs(env_cfg.save_dir, exist_ok=True)

    env, agent = make_env_and_agent(env_cfg, train_cfg, seed=0)

    if train_cfg.load_checkpoint_path:
        load_checkpoint(
            train_cfg.load_checkpoint_path,
            agent,
            env,
            load_blue=train_cfg.load_blue,
            load_red=train_cfg.load_red,
        )

    if train_cfg.checkpoint_interval > 0:
        os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)

    episode_rewards_window = deque(maxlen=max(1, train_cfg.print_interval))
    global_step = 0
    success_count = 0
    # start_time = time.time()
    per_episode_rows: List[Dict[str, Any]] = []
    success_rate_points: List[Tuple[int, float]] = []

    os.makedirs(train_cfg.results_dir, exist_ok=True)

    for ep in range(1, train_cfg.episodes + 1):
        start_time = time.time()
        step = 0
        obs = env.reset()
        done = False
        ep_reward = 0.0
        episode_info = None

        while not done:
            action = agent.select_action(obs, eval_mode=False)
            next_obs, reward, done, info = env.step(action)
            episode_info = info

            agent.store_transition(obs, action, reward, next_obs, done)
            _ = agent.update()

            obs = next_obs
            ep_reward += reward
            global_step += 1
            step += 1

        episode_rewards_window.append(ep_reward)
        # Determine whether this episode is a successful escape (blue survives until timeout).
        episode_steps = env.step_count
        episode_success = False
        if episode_info is not None:
            is_timeout = bool(episode_info.get("timeout", False))
            is_hit = bool(episode_info.get("hit", False))
            crashed = bool(episode_info.get("crashed", False))
            missiles_exhausted = bool(episode_info.get("missiles_exhausted", False))
            if is_timeout or missiles_exhausted and (not is_hit) and (not crashed):
                success_count += 1
                episode_success = True

        cumulative_success_rate = success_count / ep if ep > 0 else 0.0
        per_episode_rows.append(
            {
                "episode": ep,
                "steps": episode_steps,
                "episode_reward": float(ep_reward),
                "win": int(episode_success),
                "cumulative_success_rate": cumulative_success_rate,
                "timeout": int(bool(episode_info.get("timeout", False))) if episode_info else 0,
                "hit": int(bool(episode_info.get("hit", False))) if episode_info else 0,
                "crashed": int(bool(episode_info.get("crashed", False))) if episode_info else 0,
                "missiles_exhausted": int(bool(episode_info.get("missiles_exhausted", False))) if episode_info else 0,
            }
        )

        window = len(episode_rewards_window)
        avg_reward = sum(episode_rewards_window) / max(window, 1)
        elapsed = time.time() - start_time
        success_rate = cumulative_success_rate
        if ep % 10 == 0:
            print(
                f"Episode 编号 - {ep}\n"
                f"平均奖励 - 最近 {train_cfg.print_interval} 个 episode 的平均奖励: {avg_reward:.3f}\n"
                f"成功率 - {success_count}/{ep} ({success_rate * 100:.1f}%)\n"
                f"本回合步数 - {step}\n"
                f"耗时 - {elapsed:.1f} 秒"
            )
            success_rate_points.append((ep, cumulative_success_rate))

        if train_cfg.checkpoint_interval > 0 and ep % train_cfg.checkpoint_interval == 0:
            ckpt_name = f"checkpoint_ep{ep:04d}.pt"
            ckpt_path = os.path.join(train_cfg.checkpoint_dir, ckpt_name)
            save_checkpoint(ckpt_path, ep, agent, env)

        if ep % 200 == 0 and per_episode_rows:
            episode_path = os.path.join(train_cfg.results_dir, "episode_summary.csv")
            pd.DataFrame(per_episode_rows).to_csv(
                episode_path,
                index=False,
                mode="a" if os.path.exists(episode_path) else "w",
                header=not os.path.exists(episode_path),
            )
            per_episode_rows.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Every 10 episodes, convert this episode to a Tacview ACMI
        if env_cfg.log_trajectories and ep % 10 == 0:
            csv_dir = os.path.join(env_cfg.save_dir, "csv", str(ep))
            if os.path.isdir(csv_dir):
                add_plane_explosion = True
                if episode_info is not None:
                    is_timeout = bool(episode_info.get("timeout", False))
                    is_hit = bool(episode_info.get("hit", False))
                    crashed = bool(episode_info.get("crashed", False))
                    missiles_exhausted = bool(episode_info.get("missiles_exhausted", False))
                    if (is_timeout or missiles_exhausted) and (not is_hit) and (not crashed):
                        add_plane_explosion = False
                target_name = f"session_ep{ep:04d}"
                write_acmi(
                    target_name=target_name,
                    source_dir=csv_dir,
                    time_unit=env_cfg.dt,
                    explode_time=10,
                    add_plane_explosion=add_plane_explosion,
                )
                print(f"[ACMI] Episode {ep}: wrote {target_name}.acmi from {csv_dir}")
            else:
                print(f"[ACMI] Episode {ep}: csv dir {csv_dir} not found, skip.")

    print("Training finished.")

    if success_rate_points:
        x_vals, y_vals = zip(*success_rate_points)
        plt.figure(figsize=(8, 4.5))
        plt.plot(x_vals, y_vals, marker="o")
        plt.title("Success Rate (Every 10 Episodes)")
        plt.xlabel("Episode")
        plt.ylabel("Success Rate")
        plt.ylim(0.0, 1.0)
        plt.grid(True, linestyle="--", alpha=0.5)
        plot_path = os.path.join(train_cfg.results_dir, "success_rate_curve.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150)
        plt.close()

    if per_episode_rows:
        df = pd.DataFrame(per_episode_rows)
        excel_path = os.path.join(train_cfg.results_dir, "episode_summary.csv")
        df.to_csv(excel_path, index=False)


if __name__ == "__main__":
    train()
