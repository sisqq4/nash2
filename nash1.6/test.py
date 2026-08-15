"""Evaluate a stored blue-agent checkpoint without updating parameters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from config import EnvConfig, TrainConfig
from env.acmi_io import write_acmi

def _is_success(info: Optional[Dict[str, Any]]) -> bool:
    if not info:
        return False
    is_hit = bool(info.get("hit", False))
    is_crashed = bool(info.get("crashed", False))
    is_timeout = bool(info.get("timeout", False))
    missiles_exhausted = bool(info.get("missiles_exhausted", False))
    return (is_timeout or missiles_exhausted) and (not is_hit) and (not is_crashed)

def _apply_config(obj: Any, cfg: Dict[str, Any]) -> None:
    for key, value in cfg.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _load_run_config(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    run_dir = Path(checkpoint_path).resolve().parent.parent
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _default_outputs_dir() -> Path:
    return Path(EnvConfig().save_dir)


def _resolve_checkpoint_path(
    checkpoint: Optional[str],
    run_id: Optional[str],
    episode: Optional[int],
    checkpoint_name: Optional[str],
) -> str:
    if checkpoint:
        checkpoint_path = Path(checkpoint)
    else:
        outputs_dir = _default_outputs_dir()
        if checkpoint_name:
            if run_id:
                checkpoint_path = outputs_dir / run_id / "checkpoints" / checkpoint_name
            else:
                checkpoint_path = outputs_dir / "checkpoints" / checkpoint_name
        else:
            if not run_id or episode is None:
                raise ValueError(
                    "Checkpoint path required: pass --checkpoint, or pass both --run-id and --episode."
                )
            checkpoint_path = outputs_dir / run_id / "checkpoints" / f"checkpoint_ep{episode:04d}.pt"

    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"Checkpoint file must be a .pt file: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)

def evaluate(
    checkpoint_path: str,
    episodes: int = 100,
    seed: int = 0,
    load_blue: bool = True,
    load_red: bool = True,
    report_interval: int = 10,
    reward_mode: Optional[str] = None,
) -> float:
    from train_blue_agent import load_checkpoint, make_env_and_agent

    env_cfg = EnvConfig()

    train_cfg = TrainConfig()
    loaded_config = _load_run_config(checkpoint_path)
    if loaded_config:
        _apply_config(env_cfg, loaded_config.get("env", {}))
        _apply_config(train_cfg, loaded_config.get("train", {}))

    if reward_mode is not None:
        env_cfg.reward_mode = reward_mode
        train_cfg.reward_mode = reward_mode

    run_dir = str(Path(checkpoint_path).resolve().parent.parent)
    env_cfg.save_dir = os.path.join(run_dir, "test")
    env_cfg.log_trajectories = True
    os.makedirs(env_cfg.save_dir, exist_ok=True)
    env_cfg.log_trajectories = False

    env, agent = make_env_and_agent(env_cfg, train_cfg, seed=seed)

    load_checkpoint(
        checkpoint_path,
        agent,
        env,
        load_blue=load_blue,
        load_red=load_red,
    )

    agent.q_net.eval()

    win_count = 0
    for ep in range(1, episodes + 1):
        obs = env.reset()
        done = False
        info = None
        step = 0

        while not done:
            action = agent.select_action(obs, eval_mode=True)
            obs, _reward, done, info = env.step(action)
            step += 1

        episode_win = _is_success(info)
        if episode_win:
            win_count += 1
        if env_cfg.log_trajectories:
            csv_dir = os.path.join(env_cfg.save_dir, "csv", str(ep))
            if os.path.isdir(csv_dir):
                add_plane_explosion = True
                if info is not None:
                    is_timeout = bool(info.get("timeout", False))
                    is_hit = bool(info.get("hit", False))
                    crashed = bool(info.get("crashed", False))
                    missiles_exhausted = bool(info.get("missiles_exhausted", False))
                    if (is_timeout or missiles_exhausted) and (not is_hit) and (not crashed):
                        add_plane_explosion = False
                target_name = f"test_ep{ep:04d}"
                write_acmi(
                    target_name=target_name,
                    source_dir=csv_dir,
                    time_unit=env_cfg.dt,
                    explode_time=10,
                    add_plane_explosion=add_plane_explosion,
                )

        if report_interval > 0 and ep % report_interval == 0:
            win_rate = win_count / ep
            print(
                f"Episode {ep:4d} | win = {int(win_count)} | "
                f"win_rate = {win_rate * 100:5.1f}% | steps = {step:6d}"
            )

    win_rate = win_count / episodes if episodes > 0 else 0.0
    print(f"Evaluation finished. Win rate: {win_rate * 100:.2f}% ({win_count}/{episodes})")
    return win_rate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained blue agent.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pt file")
    parser.add_argument("--run-id", type=str, default=None, help="Training run directory under outputs/, e.g. 20260316_175358")
    parser.add_argument("--episode", type=int, default=None, help="Checkpoint episode number, e.g. 1900 -> checkpoint_ep1900.pt")
    parser.add_argument("--checkpoint-name", type=str, default=None, help="Checkpoint file name, e.g. checkpoint_ep1900.pt")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--no-load-blue", action="store_true", help="Skip loading blue agent params")
    parser.add_argument("--load-red", action=argparse.BooleanOptionalAction, default=True, help="Load red launcher params (default: true)")
    parser.add_argument("--report-interval", type=int, default=10, help="Episodes between progress logs")
    parser.add_argument("--reward-mode", type=str, default=None,
                        help="Override reward mode (auto, short_range, mid_small_azimuth, mid_large_azimuth)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = _resolve_checkpoint_path(
        checkpoint=args.checkpoint or TrainConfig().load_checkpoint_path,
        run_id=args.run_id,
        episode=args.episode,
        checkpoint_name=args.checkpoint_name,
    )
    print(f"Using checkpoint: {checkpoint_path}")
    evaluate(
        checkpoint_path=checkpoint_path,
        episodes=args.episodes,
        seed=args.seed,
        load_blue=not args.no_load_blue,
        load_red=args.load_red,
        report_interval=args.report_interval,
        reward_mode=args.reward_mode,
    )


if __name__ == "__main__":
    main()
