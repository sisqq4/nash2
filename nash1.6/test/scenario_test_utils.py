"""Shared utilities for 1v1 evaluation scenario sweeps."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from config import EnvConfig, TrainConfig
from agent.blue_bt_agent import BlueBTAgent, MissileSnapshot, PlaneSnapshot
from env.escape_env import EscapeEnv
from env.acmi_io import write_acmi
from train_blue_agent import load_checkpoint, make_env_and_agent, save_checkpoint


def is_success(info: Optional[Dict[str, Any]]) -> bool:
    if not info:
        return False
    is_hit = bool(info.get("hit", False))
    is_crashed = bool(info.get("crashed", False))
    is_timeout = bool(info.get("timeout", False))
    missiles_exhausted = bool(info.get("missiles_exhausted", False))
    return (is_timeout or missiles_exhausted) and (not is_hit) and (not is_crashed)


def apply_config(obj: Any, cfg: Dict[str, Any]) -> None:
    for key, value in cfg.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def load_run_config(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    run_dir = Path(checkpoint_path).resolve().parent.parent
    config_path = run_dir / "config.json"
    if not config_path.is_file():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_checkpoint_path(
    checkpoint: Optional[str],
    run_id: Optional[str],
    episode: Optional[int],
    checkpoint_name: Optional[str],
) -> str:
    outputs_dir = Path(EnvConfig().save_dir)
    if checkpoint:
        checkpoint_path = Path(checkpoint)
    elif checkpoint_name:
        checkpoint_path = outputs_dir / run_id / "checkpoints" / checkpoint_name if run_id else outputs_dir / "checkpoints" / checkpoint_name
    else:
        if not run_id or episode is None:
            raise ValueError("需要 --checkpoint，或同时提供 --run-id 与 --episode。")
        checkpoint_path = outputs_dir / run_id / "checkpoints" / f"checkpoint_ep{episode:04d}.pt"

    checkpoint_path = checkpoint_path.resolve()
    if checkpoint_path.suffix != ".pt":
        raise ValueError(f"Checkpoint 必须是 .pt 文件: {checkpoint_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint 不存在: {checkpoint_path}")
    return str(checkpoint_path)


def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _build_episode_min_dist_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        min_dist_km = float(row.get("min_dist", 0.0))
        out.append(
            {
                "scenario_index": int(row["scenario_index"]),
                "scenario_name": str(row["scenario_name"]),
                "episode": int(row["episode"]),
                "global_episode": int(row["global_episode"]),
                "episode_seed": int(row["episode_seed"]),
                "min_dist_km": float(min_dist_km),
                "min_dist_m": float(min_dist_km * 1000.0),
            }
        )
    return out

def _mean_from_group(group: Dict[str, Any], field: str) -> float:
    episodes = max(int(group["episodes"]), 1)
    return float(group[f"{field}_sum"]) / episodes

def _plot_min_dist_hist_by_scenario(rows: List[Dict[str, Any]], out_dir: str, bin_size_m: float = 1.0) -> None:
    if not rows:
        return

    grouped: Dict[tuple[int, str], List[float]] = {}
    for row in rows:
        key = (int(row["scenario_index"]), str(row["scenario_name"]))
        value_m = float(row.get("min_dist", 0.0)) * 1000.0
        if not np.isfinite(value_m):
            continue
        grouped.setdefault(key, []).append(value_m)

    for (scenario_index, scenario_name), values_m in sorted(grouped.items()):
        if not values_m:
            continue
        arr = np.asarray(values_m, dtype=float)
        x_min = float(np.min(arr))
        x_max = float(np.max(arr))
        left = math.floor(x_min / bin_size_m) * bin_size_m
        right = math.ceil(x_max / bin_size_m) * bin_size_m
        if right <= left:
            right = left + bin_size_m
        bins = np.arange(left, right + bin_size_m, bin_size_m)
        if bins.size < 2:
            bins = np.array([left, left + bin_size_m], dtype=float)

        plt.figure(figsize=(9, 5))
        plt.hist(arr, bins=bins, density=True, alpha=0.8, color="#4C72B0", edgecolor="white")
        plt.xlabel("Episode minimum missile-target distance (m)")
        plt.ylabel("Probability density")
        plt.title(f"Scenario {scenario_index:02d} - {scenario_name}: min_dist distribution (bin=1m)")
        plt.grid(alpha=0.25, linestyle="--")
        plt.tight_layout()
        fig_name = f"hist_min_dist_scenario_{scenario_index:02d}_{scenario_name}.png"
        plt.savefig(os.path.join(out_dir, fig_name), dpi=180)
        plt.close()

def _collect_step_diagnostics(env: Any, step: int, time_value: float) -> Dict[str, Any]:
    idx_active, ti, tgo = env._compute_threat_scores()
    active_count = int(idx_active.size)

    primary_threat_id = -1
    primary_threat_score = 0.0
    if active_count > 0:
        k = int(np.argmax(ti))
        primary_threat_id = int(idx_active[k])
        primary_threat_score = float(ti[k])

    corridor_width = float(env.cfg.coop_corridor_ref_width)
    encirclement = 0.0
    if active_count > 0:
        rel = env.missile_pos[idx_active, :2] - env.blue_pos[None, :2]
        heading = env.blue_vel[:2]
        h_norm = float(np.linalg.norm(heading))
        if h_norm < 1e-6:
            heading = np.array([1.0, 0.0], dtype=float)
            h_norm = 1.0
        h_hat = heading / h_norm
        side = np.array([-h_hat[1], h_hat[0]], dtype=float)
        lateral = np.abs(np.dot(rel, side))
        if lateral.size > 0:
            corridor_width = float(np.min(lateral))
        encirclement = float(env._compute_collaborative_encirclement(idx_active, tgo))

    tgo_std = float(np.std(tgo)) if tgo.size > 0 else 0.0
    tgo_min = float(np.min(tgo)) if tgo.size > 0 else 0.0
    tgo_max = float(np.max(tgo)) if tgo.size > 0 else 0.0

    return {
        "step": int(step),
        "time": float(time_value),
        "active_missiles": active_count,
        "primary_threat_id": int(primary_threat_id),
        "primary_threat_score": float(primary_threat_score),
        "corridor_width": float(corridor_width),
        "tgo_std": float(tgo_std),
        "tgo_min": float(tgo_min),
        "tgo_max": float(tgo_max),
        "encirclement": float(encirclement),
    }


def _episode_multi_metrics(step_rows: List[Dict[str, Any]], initial_missiles: int) -> Dict[str, float]:
    if not step_rows:
        return {
            "threat_switch_count": 0.0,
            "threat_id_jitter_rate": 0.0,
            "corridor_width_mean": 0.0,
            "corridor_width_min": 0.0,
            "corridor_width_trend": 0.0,
            "tgo_std_mean": 0.0,
            "tgo_std_max": 0.0,
            "degrade_to_2_time": -1.0,
            "degrade_to_1_time": -1.0,
            "degrade_to_0_time": -1.0,
        }

    threat_ids = [int(r["primary_threat_id"]) for r in step_rows if int(r["primary_threat_id"]) >= 0]
    switches = 0
    for i in range(1, len(threat_ids)):
        if threat_ids[i] != threat_ids[i - 1]:
            switches += 1

    n_threat = len(threat_ids)
    jitter = switches / max(n_threat - 1, 1)

    widths = [float(r["corridor_width"]) for r in step_rows]
    tgo_std_vals = [float(r["tgo_std"]) for r in step_rows]
    times = [float(r["time"]) for r in step_rows]
    active = [int(r["active_missiles"]) for r in step_rows]

    if len(widths) > 1:
        dt = max(times[-1] - times[0], 1e-6)
        width_trend = (widths[-1] - widths[0]) / dt
    else:
        width_trend = 0.0

    def first_time_at_or_below(target: int) -> float:
        for row in step_rows:
            if int(row["active_missiles"]) <= target:
                return float(row["time"])
        return -1.0

    deg2 = first_time_at_or_below(2) if initial_missiles >= 3 else -1.0
    deg1 = first_time_at_or_below(1) if initial_missiles >= 2 else -1.0
    deg0 = first_time_at_or_below(0)

    return {
        "threat_switch_count": float(switches),
        "threat_id_jitter_rate": float(jitter),
        "corridor_width_mean": float(np.mean(widths)) if widths else 0.0,
        "corridor_width_min": float(np.min(widths)) if widths else 0.0,
        "corridor_width_trend": float(width_trend),
        "tgo_std_mean": float(np.mean(tgo_std_vals)) if tgo_std_vals else 0.0,
        "tgo_std_max": float(np.max(tgo_std_vals)) if tgo_std_vals else 0.0,
        "degrade_to_2_time": float(deg2),
        "degrade_to_1_time": float(deg1),
        "degrade_to_0_time": float(deg0),
    }

def _build_bt_inputs(env: Any, bt_team: int) -> Tuple[PlaneSnapshot, List[MissileSnapshot]]:
    roll_rad = float(env.blue_model.roll_rad or 0.0)
    plane = PlaneSnapshot(
        pos=env.blue_pos.copy(),
        vel=env.blue_vel.copy(),
        roll_rad=roll_rad,
    )
    missiles: List[MissileSnapshot] = []
    for idx in range(env.cfg.num_missiles):
        missiles.append(
            MissileSnapshot(
                pos=env.missile_pos[idx].copy(),
                team=1 - bt_team,
                is_active=bool(env.missile_alive[idx] and env.missile_launched[idx]),
            )
        )
    return plane, missiles


def _select_eval_action(
    policy_name: str,
    obs: np.ndarray,
    env: Any,
    agent: Any,
    bt_agent: Optional[BlueBTAgent],
) -> int:
    if policy_name == "dqn":
        if agent is None:
            raise ValueError("blue_eval_policy=dqn 时 DQN agent 不能为空。")
        return int(agent.select_action(obs, eval_mode=True))
    if policy_name == "bt":
        if bt_agent is None:
            raise ValueError("blue_eval_policy=bt 时 bt_agent 不能为空。")
        plane, missiles = _build_bt_inputs(env, bt_team=0)
        return int(bt_agent.get_action(plane, missiles, enemies=[]))
    raise ValueError(f"未知 blue_eval_policy: {policy_name}")

def run_scenario_sweep(
    checkpoint_path: str,
    output_root: str,
    scenarios: Iterable[Dict[str, Any]],
    episodes_per_scenario: int,
    seed: int,
    checkpoint_interval: int = 10,
    report_interval: int = 10,
    reward_mode: Optional[str] = None,
    blue_eval_policy: str = "dqn",
) -> List[Dict[str, Any]]:
    # Kept for backward compatibility.
    rows, _ = run_scenario_sweep_multi_diagnostics(
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        scenarios=scenarios,
        episodes_per_scenario=episodes_per_scenario,
        seed=seed,
        checkpoint_interval=checkpoint_interval,
        report_interval=report_interval,
        reward_mode=reward_mode,
        enable_step_diagnostics=False,
        blue_eval_policy=blue_eval_policy,
    )
    return rows


def run_scenario_sweep_multi_diagnostics(
        checkpoint_path: str,
        output_root: str,
        scenarios: Iterable[Dict[str, Any]],
        episodes_per_scenario: int,
        seed: int,
        checkpoint_interval: int = 10,
        report_interval: int = 10,
        reward_mode: Optional[str] = None,
        enable_step_diagnostics: bool = True,
        blue_eval_policy: str = "dqn",
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    env_cfg = EnvConfig()
    train_cfg = TrainConfig()

    loaded_cfg = load_run_config(checkpoint_path)
    if loaded_cfg:
        apply_config(env_cfg, loaded_cfg.get("env", {}))
        apply_config(train_cfg, loaded_cfg.get("train", {}))

    if reward_mode is not None:
        env_cfg.reward_mode = reward_mode
        train_cfg.reward_mode = reward_mode

    env_cfg.log_trajectories = True
    os.makedirs(output_root, exist_ok=True)

    with open(os.path.join(output_root, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"env": asdict(env_cfg), "train": asdict(train_cfg)}, f, ensure_ascii=False, indent=2)

    policy_name = str(blue_eval_policy).strip().lower()
    if policy_name not in {"dqn", "bt"}:
        raise ValueError(f"未知 blue_eval_policy: {policy_name}")
    bt_agent = BlueBTAgent(uid="blue_eval_bt", team=0) if policy_name == "bt" else None

    all_rows: List[Dict[str, Any]] = []
    step_rows_all: List[Dict[str, Any]] = []
    overall_episode = 0

    for scenario_idx, scenario in enumerate(scenarios, start=1):
        scenario_name = str(scenario["scenario_name"])
        scenario_dir = os.path.join(output_root, f"scenario_{scenario_idx:02d}_{scenario_name}")
        os.makedirs(scenario_dir, exist_ok=True)
        seed_rng = random.Random((seed + 1) * 1_000_003 + scenario_idx)

        sc_env_cfg = EnvConfig()
        apply_config(sc_env_cfg, asdict(env_cfg))
        env_overrides = dict(scenario.get("env_overrides", {}))
        for k, v in env_overrides.items():
            if hasattr(sc_env_cfg, k):
                setattr(sc_env_cfg, k, v)
        blue_position_keys = {
            "blue_x_min",
            "blue_x_max",
            "blue_y_min",
            "blue_y_max",
            "blue_z_min",
            "blue_z_max",
        }
        if "blue_fixed_start" not in env_overrides and any(
            k in blue_position_keys for k in env_overrides
        ):
            # Scenario sweeps express the intended blue initial position through
            # blue_*_min/max ranges. Checkpoints may have been trained with
            # blue_fixed_start=True (for example at x=0), so disable the fixed
            # checkpoint start unless the scenario explicitly asks for it.
            sc_env_cfg.blue_fixed_start = False
        if (
            "missile_spawn_mode" not in env_overrides
            and any(k.startswith("red_launch_") for k in env_overrides)
        ):
            # Scenario sweeps describe red launch geometry with red_launch_*;
            # use the launcher-backed mode so those per-scenario policies take
            # effect instead of falling back to the fixed missile_spawn_* point.
            sc_env_cfg.missile_spawn_mode = "game_theory"
        sc_env_cfg.save_dir = scenario_dir
        sc_env_cfg.log_trajectories = True

        if policy_name == "dqn":
            env, agent = make_env_and_agent(sc_env_cfg, train_cfg, seed=seed + scenario_idx)
            # NOTE:
            #   For evaluation we intentionally do NOT load red state from checkpoint.
            #   Checkpoints are saved at episode end; red nav_gains in that snapshot may be
            #   terminal values (e.g., zero after missiles expire), which would disable PN
            #   in later tests. We keep red parameters from env_cfg/config.json so red uses
            #   the same PN/drag/speed model settings as training-time configuration.
            #
            #   The DQN network input dimension depends on num_missiles, so the agent must
            #   be built after scenario overrides are applied. BT evaluation does not use
            #   the DQN weights at all, so skip blue loading to avoid irrelevant checkpoint
            #   shape mismatches when only the environment configuration is needed.
            load_checkpoint(checkpoint_path, agent, env, load_blue=True, load_red=False)
            agent.q_net.eval()
        else:
            env = EscapeEnv(sc_env_cfg, seed=seed + scenario_idx)
            agent = None

        wins = 0
        for ep in range(1, episodes_per_scenario + 1):
            overall_episode += 1
            episode_seed = int(seed_rng.randint(0, 2**32 - 1))
            env.rng = np.random.default_rng(episode_seed)
            env.launcher.rng = env.rng
            obs = env.reset()
            if bt_agent is not None:
                init_state = np.concatenate((env.blue_pos, env.blue_vel))
                bt_agent.reset(init_state)
            done = False
            info: Optional[Dict[str, Any]] = None
            ep_reward = 0.0
            step_rows_ep: List[Dict[str, Any]] = []

            while not done:
                action = _select_eval_action(
                    policy_name=policy_name,
                    obs=obs,
                    env=env,
                    agent=agent,
                    bt_agent=bt_agent,
                )
                obs, reward, done, info = env.step(action)
                ep_reward += reward

                if enable_step_diagnostics:
                    sd = _collect_step_diagnostics(env, step=env.step_count, time_value=env.time)
                    sd.update(
                        {
                            "scenario_index": scenario_idx,
                            "scenario_name": scenario_name,
                            "episode": ep,
                            "global_episode": overall_episode,
                            "blue_eval_policy": policy_name,
                            "blue_action": int(action),
                            "bt_state": "" if bt_agent is None else str(bt_agent.state),
                        }
                    )
                    step_rows_ep.append(sd)

            win = is_success(info)
            if win:
                wins += 1
            mm = _episode_multi_metrics(step_rows_ep, initial_missiles=sc_env_cfg.num_missiles)
            row = {
                "scenario_index": scenario_idx,
                "scenario_name": scenario_name,
                "episode": ep,
                "global_episode": overall_episode,
                "episode_seed": episode_seed,
                "reward": float(ep_reward),
                "win": int(win),
                "steps": int(info.get("step", 0)) if info else 0,
                "min_dist": float(info.get("min_dist", 0.0)) if info else 0.0,
                "final_dist": float(info.get("final_dist", 0.0)) if info else 0.0,
                "final_speed": float(info.get("final_speed", 0.0)) if info else 0.0,
                "avg_speed": float(info.get("avg_speed", 0.0)) if info else 0.0,
                "min_speed": float(info.get("min_speed", 0.0)) if info else 0.0,
                "avg_altitude": float(info.get("avg_altitude", 0.0)) if info else 0.0,
                "avg_roll_abs_deg": float(info.get("avg_roll_abs_deg", 0.0)) if info else 0.0,
                "avg_turn_rate_deg": float(info.get("avg_turn_rate_deg", 0.0)) if info else 0.0,
                "timeout": int(bool(info.get("timeout", False))) if info else 0,
                "hit": int(bool(info.get("hit", False))) if info else 0,
                "crashed": int(bool(info.get("crashed", False))) if info else 0,
                "missiles_exhausted": int(bool(info.get("missiles_exhausted", False))) if info else 0,
                **mm,
            }
            row.update({f"param_{k}": v for k, v in scenario.items() if k != "env_overrides"})
            all_rows.append(row)
            step_rows_all.extend(step_rows_ep)

            if ep % checkpoint_interval == 0:

                csv_dir = os.path.join(scenario_dir, "csv", str(ep))
                if os.path.isdir(csv_dir):
                    write_acmi(
                        target_name=f"test_ep{ep:04d}",
                        source_dir=csv_dir,
                        time_unit=sc_env_cfg.dt,
                        explode_time=10,
                        add_plane_explosion=not win,
                    )

            if report_interval > 0 and ep % report_interval == 0:
                print(f"[{scenario_name}] Episode {ep}/{episodes_per_scenario} | win_rate={wins/ep:.3f}")

    results_dir = os.path.join(output_root, "results")
    os.makedirs(results_dir, exist_ok=True)
    _write_csv(os.path.join(results_dir, "episode_summary.csv"), all_rows)
    _write_csv(
        os.path.join(results_dir, "episode_min_dist_all.csv"),
        _build_episode_min_dist_rows(all_rows),
    )
    _plot_min_dist_hist_by_scenario(all_rows, results_dir, bin_size_m=1.0)
    if enable_step_diagnostics:
        _write_csv(os.path.join(results_dir, "step_diagnostics.csv"), step_rows_all)

    grouped: Dict[tuple[int, str], Dict[str, Any]] = {}
    for row in all_rows:
        key = (int(row["scenario_index"]), str(row["scenario_name"]))
        if key not in grouped:
            grouped[key] = {
                "scenario_index": key[0],
                "scenario_name": key[1],
                "episodes": 0,
                "win_sum": 0,
                "reward_sum": 0.0,
                "steps_sum": 0,
                "min_dist_sum": 0.0,
                "final_dist_sum": 0.0,
                "final_speed_sum": 0.0,
                "avg_speed_sum": 0.0,
                "min_speed_sum": 0.0,
                "avg_altitude_sum": 0.0,
                "avg_roll_abs_deg_sum": 0.0,
                "avg_turn_rate_deg_sum": 0.0,
                "timeout_sum": 0,
                "hit_sum": 0,
                "crashed_sum": 0,
                "missiles_exhausted_sum": 0,
                "threat_switch_count_sum": 0.0,
                "threat_id_jitter_rate_sum": 0.0,
                "corridor_width_mean_sum": 0.0,
                "corridor_width_min_sum": 0.0,
                "corridor_width_trend_sum": 0.0,
                "tgo_std_mean_sum": 0.0,
                "tgo_std_max_sum": 0.0,
                "degrade_to_2_time_sum": 0.0,
                "degrade_to_1_time_sum": 0.0,
                "degrade_to_0_time_sum": 0.0,
            }
        g = grouped[key]
        g["episodes"] += 1
        g["win_sum"] += int(row["win"])
        g["reward_sum"] += float(row["reward"])
        g["steps_sum"] += int(row["steps"])
        g["min_dist_sum"] += float(row["min_dist"])
        g["final_dist_sum"] += float(row["final_dist"])
        g["final_speed_sum"] += float(row["final_speed"])
        g["avg_speed_sum"] += float(row["avg_speed"])
        g["min_speed_sum"] += float(row["min_speed"])
        g["avg_altitude_sum"] += float(row["avg_altitude"])
        g["avg_roll_abs_deg_sum"] += float(row["avg_roll_abs_deg"])
        g["avg_turn_rate_deg_sum"] += float(row["avg_turn_rate_deg"])
        g["timeout_sum"] += int(row["timeout"])
        g["hit_sum"] += int(row["hit"])
        g["crashed_sum"] += int(row["crashed"])
        g["missiles_exhausted_sum"] += int(row["missiles_exhausted"])
        g["threat_switch_count_sum"] += float(row["threat_switch_count"])
        g["threat_id_jitter_rate_sum"] += float(row["threat_id_jitter_rate"])
        g["corridor_width_mean_sum"] += float(row["corridor_width_mean"])
        g["corridor_width_min_sum"] += float(row["corridor_width_min"])
        g["corridor_width_trend_sum"] += float(row["corridor_width_trend"])
        g["tgo_std_mean_sum"] += float(row["tgo_std_mean"])
        g["tgo_std_max_sum"] += float(row["tgo_std_max"])
        g["degrade_to_2_time_sum"] += float(row["degrade_to_2_time"])
        g["degrade_to_1_time_sum"] += float(row["degrade_to_1_time"])
        g["degrade_to_0_time_sum"] += float(row["degrade_to_0_time"])

    result_rows: List[Dict[str, Any]] = []
    for key in sorted(grouped.keys()):
        g = grouped[key]
        episodes = max(int(g["episodes"]), 1)
        result_rows.append(
            {
                "scenario_index": g["scenario_index"],
                "scenario_name": g["scenario_name"],
                "episodes": g["episodes"],
                "win_rate": g["win_sum"] / episodes,
                "hit_rate": g["hit_sum"] / episodes,
                "crash_rate": g["crashed_sum"] / episodes,
                "timeout_rate": g["timeout_sum"] / episodes,
                "missiles_exhausted_rate": g["missiles_exhausted_sum"] / episodes,
                "avg_reward": _mean_from_group(g, "reward"),
                "avg_steps": _mean_from_group(g, "steps"),
                "avg_min_dist": _mean_from_group(g, "min_dist"),
                "avg_final_dist": _mean_from_group(g, "final_dist"),
                "avg_final_speed": _mean_from_group(g, "final_speed"),
                "avg_speed": _mean_from_group(g, "avg_speed"),
                "avg_min_speed": _mean_from_group(g, "min_speed"),
                "avg_altitude": _mean_from_group(g, "avg_altitude"),
                "avg_roll_abs_deg": _mean_from_group(g, "avg_roll_abs_deg"),
                "avg_turn_rate_deg": _mean_from_group(g, "avg_turn_rate_deg"),
                "avg_threat_switch_count": _mean_from_group(g, "threat_switch_count"),
                "avg_threat_id_jitter_rate": _mean_from_group(g, "threat_id_jitter_rate"),
                "avg_corridor_width": _mean_from_group(g, "corridor_width_mean"),
                "avg_min_corridor_width": _mean_from_group(g, "corridor_width_min"),
                "avg_corridor_width_trend": _mean_from_group(g, "corridor_width_trend"),
                "avg_tgo_std": _mean_from_group(g, "tgo_std_mean"),
                "avg_tgo_std_max": _mean_from_group(g, "tgo_std_max"),
                "avg_degrade_to_2_time": _mean_from_group(g, "degrade_to_2_time"),
                "avg_degrade_to_1_time": _mean_from_group(g, "degrade_to_1_time"),
                "avg_degrade_to_0_time": _mean_from_group(g, "degrade_to_0_time"),
            }
        )
    _write_csv(os.path.join(results_dir, "result.csv"), result_rows)

    return all_rows, step_rows_all
