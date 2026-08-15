"""1v2 协同规避测试：覆盖时间协同、空间协同、时空混合与动态扰动。"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario_test_utils import resolve_checkpoint_path, run_scenario_sweep_multi_diagnostics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1v2 协同规避测试")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="20260326_103719_1v2")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_ep1000.pt")
    parser.add_argument("--episodes-per-scenario", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", type=str, default="multi_coop")
    parser.add_argument("--blue-eval-policy", type=str, choices=["dqn", "bt"], default="dqn")
    return parser.parse_args()


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _fixed_profile(
    base: dict[str, object],
    positions: list[list[float]],
    launch_times: list[float],
    **extra: object,
) -> dict[str, object]:
    return {
        **base,
        "missile_fixed_positions": positions,
        "missile_fixed_launch_times": launch_times,
        "missile_launch_time_std": 0.0,
        "missile_launch_time_clip": 0.0,
        **extra,
    }


def _build_scenarios() -> list[dict[str, object]]:
    base = {
        "num_missiles": 2,
        "missile_update_dt": 0.01,
        "blue_x_min": 20.0,
        "blue_x_max": 20.0,
        "blue_y_min": 0.0,
        "blue_y_max": 0.0,
        "blue_z_min": 10.0,
        "blue_z_max": 10.0,
        # Force a randomized initial nose/velocity direction in the xoy plane,
        # even when the loaded training checkpoint used a fixed +x heading.
        "blue_heading_min": -180.0,
        "blue_heading_max": 180.0,
        "hit_radius": 0.005,
    }
    same_direction = [[0.0, -1.0, 10.0], [0.0, 1.0, 10.0]]
    return [
        {
            "scenario_name": "time_same_direction_large_delta_t",
            "family": "time_coord",
            "sub_type": "same_direction_large_dt",
            "env_overrides": _fixed_profile(base, same_direction, [0.0, 5.0]),
        },
        {
            "scenario_name": "time_same_direction_medium_delta_t",
            "family": "time_coord",
            "sub_type": "same_direction_mid_dt",
            "env_overrides": _fixed_profile(base, same_direction, [0.0, 3.0]),
        },
        {
            "scenario_name": "time_same_direction_near_sync",
            "family": "time_coord",
            "sub_type": "same_direction_sync",
            "env_overrides": _fixed_profile(base, same_direction, [0.0, 1.0]),
        },
        {
            "scenario_name": "space_pincer_symmetric",
            "family": "space_coord",
            "sub_type": "dual_pincer_symmetric",
            "env_overrides": _fixed_profile(base, [[5.0, 5.0, 10.0], [5.0, -5.0, 10.0]], [0.0, 0.0]),
        },
        {
            "scenario_name": "space_pincer_asymmetric",
            "family": "space_coord",
            "sub_type": "dual_pincer_asymmetric",
            "env_overrides": _fixed_profile(base, [[5.0, 5.0, 10.0], [7.0, -7.0, 10.0]], [0.0, 0.0]),
        },
        {
            "scenario_name": "space_pincer_high_low",
            "family": "space_coord",
            "sub_type": "dual_pincer_high_low",
            "env_overrides": _fixed_profile(base, [[5.0, 5.0, 12.0], [5.0, -5.0, 8.0]], [0.0, 0.0]),
        },
        {
            "scenario_name": "spatiotemporal_pincer_sync_compress",
            "family": "spatiotemporal_coord",
            "sub_type": "pincer_sync_compress",
            "env_overrides": _fixed_profile(base, [[5.0, 12.0, 10.0], [5.0, -12.0, 10.0]], [0.0, 0.0]),
        },
        {
            "scenario_name": "dynamic_disturbance_online_replan",
            "family": "dynamic_disturbance",
            "sub_type": "param_noise_delay",
            "env_overrides": _fixed_profile(
                base,
                [[2.0, -8.0, 9.8], [2.0, 8.0, 10.2]],
                [0.0, 0.7],
                missile_speed_decay_factor=0.985,
                missile_speed_decay_factor_by_missile=[0.985, 0.985],
                missile_seeker_fov_deg=45.0,
                missile_seeker_fov_deg_by_missile=[45.0, 45.0],
                missile_seeker_memory_time=1.2,
                missile_seeker_memory_time_by_missile=[1.2, 1.2],
                blue_accel=0.08,
            ),
        },
    ]

def _plot_diagnostics(step_rows: list[dict[str, float]], out_dir: Path) -> None:
    metrics = ["primary_threat_id", "corridor_width", "tgo_std", "active_missiles"]
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in step_rows:
        key = str(row["scenario_name"])
        if int(row["episode"]) == 1:
            grouped[key].append(row)

    for scenario_name, rows in grouped.items():
        rows = sorted(rows, key=lambda x: int(x["step"]))
        t = [float(r["time"]) for r in rows]
        for metric in metrics:
            y = [float(r[metric]) for r in rows]
            plt.figure(figsize=(8.5, 4.8))
            plt.plot(t, y, linewidth=1.5)
            plt.xlabel("Time (s)")
            plt.ylabel(metric)
            plt.title(f"{scenario_name}: {metric} over time")
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.tight_layout()
            plt.savefig(out_dir / f"{scenario_name}_{metric}.png", dpi=150)
            plt.close()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        episode=args.episode,
        checkpoint_name=args.checkpoint_name,
    )

    scenarios = _build_scenarios()
    output_root = Path("outputs") / f"tests_1v2_coordination_{args.blue_eval_policy}"
    all_rows, step_rows = run_scenario_sweep_multi_diagnostics(
        checkpoint_path=checkpoint_path,
        output_root=str(output_root),
        scenarios=scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        seed=args.seed,
        checkpoint_interval=10,
        report_interval=10,
        reward_mode=args.reward_mode,
        blue_eval_policy=args.blue_eval_policy,
        enable_step_diagnostics=True,
    )

    grouped: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        family = str(row["param_family"])
        sub_type = str(row["param_sub_type"])
        key = (family, sub_type)
        for k in [
            "win", "reward", "steps", "min_dist", "final_dist", "avg_speed", "hit", "timeout", "crashed", "missiles_exhausted",
            "threat_switch_count", "threat_id_jitter_rate", "corridor_width_mean", "corridor_width_min", "corridor_width_trend",
            "tgo_std_mean", "tgo_std_max", "degrade_to_1_time", "degrade_to_0_time",
        ]:
            grouped[key][k].append(float(row[k]))

    result_rows: list[dict[str, float]] = []
    for (family, sub_type), vals in sorted(grouped.items()):
        result_rows.append(
            {
                "family": family,
                "sub_type": sub_type,
                "episodes": float(len(vals["win"])),
                "win_rate": _mean(vals["win"]),
                "hit_rate": _mean(vals["hit"]),
                "crash_rate": _mean(vals["crashed"]),
                "timeout_rate": _mean(vals["timeout"]),
                "missiles_exhausted_rate": _mean(vals["missiles_exhausted"]),
                "avg_reward": _mean(vals["reward"]),
                "avg_steps": _mean(vals["steps"]),
                "avg_min_dist": _mean(vals["min_dist"]),
                "avg_final_dist": _mean(vals["final_dist"]),
                "avg_speed": _mean(vals["avg_speed"]),
                "avg_threat_switch_count": _mean(vals["threat_switch_count"]),
                "avg_threat_id_jitter_rate": _mean(vals["threat_id_jitter_rate"]),
                "avg_corridor_width": _mean(vals["corridor_width_mean"]),
                "avg_min_corridor_width": _mean(vals["corridor_width_min"]),
                "avg_corridor_width_trend": _mean(vals["corridor_width_trend"]),
                "avg_tgo_std": _mean(vals["tgo_std_mean"]),
                "avg_tgo_std_max": _mean(vals["tgo_std_max"]),
                "avg_degrade_to_1_time": _mean(vals["degrade_to_1_time"]),
                "avg_degrade_to_0_time": _mean(vals["degrade_to_0_time"]),
            }
        )

    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results_dir / "result_1v2_coordination.csv", result_rows)
    _plot_diagnostics(step_rows, results_dir)


if __name__ == "__main__":
    main()