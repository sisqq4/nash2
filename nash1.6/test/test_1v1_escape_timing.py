"""1v1 逃逸时机测试：固定不同初始距离，评估最优策略表现与逃逸率。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scenario_test_utils import resolve_checkpoint_path, run_scenario_sweep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1v1 逃逸时机测试")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="20260323_215046")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_ep1000.pt")
    parser.add_argument("--episodes-per-scenario", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--blue-eval-policy", type=str, choices=["dqn", "bt"], default="dqn")
    parser.add_argument("--min-distance-km", type=int, default=6)
    parser.add_argument("--max-distance-km", type=int, default=30)
    parser.add_argument("--distance-step-km", type=int, default=1)
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

def _plot_timing(rows: list[dict[str, float]], out_dir: Path) -> None:
    if not rows:
        return
    rows = sorted(rows, key=lambda x: x["distance_km"])
    metrics = [
        "win_rate",
        "hit_rate",
        "crash_rate",
        "timeout_rate",
        "missiles_exhausted_rate",
        "avg_reward",
        "avg_steps",
        "avg_min_dist",
        "avg_final_dist",
        "avg_final_speed",
        "avg_speed",
        "avg_min_speed",
        "avg_altitude",
        "avg_roll_abs_deg",
        "avg_turn_rate_deg",
    ]
    for metric in metrics:
        plt.figure(figsize=(8, 4.8))
        plt.plot([r["distance_km"] for r in rows], [r[metric] for r in rows], marker="o")
        plt.xlabel("Distance (km)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Distance")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_vs_distance.png", dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        episode=args.episode,
        checkpoint_name=args.checkpoint_name,
    )

    distances = list(range(args.min_distance_km, args.max_distance_km + 1, args.distance_step_km))
    scenarios = [
        {
            "scenario_name": f"distance_{distance:02d}km",
            "distance_km": float(distance),
            "env_overrides": {
                "blue_x_min": float(distance),
                "blue_x_max": float(distance),
                "blue_y_min": 0.0,
                "blue_y_max": 0.0,
                "blue_z_min": 10.0,
                "blue_z_max": 10.0,
                "red_launch_x_min": 0.0,
                "red_launch_x_max": 0.0,
                "red_launch_y_min": 0.0,
                "red_launch_y_max": 0.0,
                "red_launch_z_min": 10.0,
                "red_launch_z_max": 10.0,
                "missile_update_dt": 0.01,
                "hit_radius": 0.005,
            },
        }
        for distance in distances
    ]

    output_root = Path("outputs") / f"tests_1v1_escape_timing_{args.blue_eval_policy}"
    all_rows = run_scenario_sweep(
        checkpoint_path=checkpoint_path,
        output_root=str(output_root),
        scenarios=scenarios,
        episodes_per_scenario=args.episodes_per_scenario,
        seed=args.seed,
        checkpoint_interval=10,
        report_interval=10,
        reward_mode=args.reward_mode,
        blue_eval_policy=args.blue_eval_policy,
    )

    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        distance = int(float(row["param_distance_km"]))
        grouped[distance]["win"].append(float(row["win"]))
        grouped[distance]["reward"].append(float(row["reward"]))
        grouped[distance]["steps"].append(float(row["steps"]))
        grouped[distance]["min_dist"].append(float(row["min_dist"]))
        grouped[distance]["final_dist"].append(float(row["final_dist"]))
        grouped[distance]["final_speed"].append(float(row["final_speed"]))
        grouped[distance]["avg_speed"].append(float(row["avg_speed"]))
        grouped[distance]["min_speed"].append(float(row["min_speed"]))
        grouped[distance]["avg_altitude"].append(float(row["avg_altitude"]))
        grouped[distance]["avg_roll_abs_deg"].append(float(row["avg_roll_abs_deg"]))
        grouped[distance]["avg_turn_rate_deg"].append(float(row["avg_turn_rate_deg"]))
        grouped[distance]["hit"].append(float(row["hit"]))
        grouped[distance]["crashed"].append(float(row["crashed"]))
        grouped[distance]["timeout"].append(float(row["timeout"]))
        grouped[distance]["missiles_exhausted"].append(float(row["missiles_exhausted"]))

    result_rows: list[dict[str, float]] = []
    for distance, vals in sorted(grouped.items()):
        result_rows.append(
            {
                "distance_km": float(distance),
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
                "avg_final_speed": _mean(vals["final_speed"]),
                "avg_speed": _mean(vals["avg_speed"]),
                "avg_min_speed": _mean(vals["min_speed"]),
                "avg_altitude": _mean(vals["avg_altitude"]),
                "avg_roll_abs_deg": _mean(vals["avg_roll_abs_deg"]),
                "avg_turn_rate_deg": _mean(vals["avg_turn_rate_deg"]),
            }
        )

    results_dir = output_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results_dir / "result_timing_distance_scan.csv", result_rows)
    _plot_timing(result_rows, results_dir)


if __name__ == "__main__":
    main()
