"""1v1 逃逸方向测试：固定初始距离并扫描不同初始朝向。"""

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
    parser = argparse.ArgumentParser(description="1v1 逃逸方向测试")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--run-id", type=str, default="20260323_215046")
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="checkpoint_ep1000.pt")
    parser.add_argument("--episodes-per-scenario", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reward-mode", type=str, default=None)
    parser.add_argument("--blue-eval-policy", type=str, choices=["dqn", "bt"], default="dqn")
    parser.add_argument("--heading-jitter-deg", type=float, default=30.0, help="每个方向场景的初始航向扰动范围(±度)")
    parser.add_argument("--distance-jitter-km", type=float, default=0.0, help="初始距离扰动范围(±km)")
    parser.add_argument("--lateral-jitter-km", type=float, default=0.3, help="初始横向扰动范围(±km)")
    parser.add_argument("--min-distance-km", type=int, default=6)
    parser.add_argument("--max-distance-km", type=int, default=30)
    parser.add_argument("--distance-step-km", type=int, default=2)
    parser.add_argument("--angle-step-deg", type=int, default=90)
    return parser.parse_args()

def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def _write_result_csv(path: Path, rows: list[dict[str, float]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def _plot_direction(result_rows: list[dict[str, float]], out_dir: Path) -> None:
    if not result_rows:
        return
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
        angle_to_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
        dist_to_points: dict[int, list[tuple[float, float]]] = defaultdict(list)
        for r in result_rows:
            d = float(r["distance_km"])
            a = int(r["heading_deg"])
            v = float(r[metric])
            angle_to_points[a].append((d, v))
            dist_to_points[int(d)].append((a, v))

        plt.figure(figsize=(9, 5))
        for angle in sorted(angle_to_points):
            pts = sorted(angle_to_points[angle], key=lambda x: x[0])
            plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{angle}°")
        plt.xlabel("Distance (km)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Distance (grouped by heading)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_by_distance_per_heading.png", dpi=150)
        plt.close()

        plt.figure(figsize=(9, 5))
        for distance in sorted(dist_to_points):
            pts = sorted(dist_to_points[distance], key=lambda x: x[0])
            plt.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=f"{distance}km")
        plt.xlabel("Heading (deg)")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Heading (grouped by distance)")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.legend(ncol=3, fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"{metric}_by_heading_per_distance.png", dpi=150)
        plt.close()


def main() -> None:
    args = parse_args()
    checkpoint_path = resolve_checkpoint_path(
        checkpoint=args.checkpoint,
        run_id=args.run_id,
        episode=args.episode,
        checkpoint_name=args.checkpoint_name,
    )

    h_jitter = max(0.0, float(args.heading_jitter_deg))
    d_jitter = max(0.0, float(args.distance_jitter_km))
    y_jitter = max(0.0, float(args.lateral_jitter_km))
    distances = list(range(args.min_distance_km, args.max_distance_km + 1, args.distance_step_km))
    angles = [angle for angle in range(0, 361, args.angle_step_deg) if angle != 360]
    scenarios = [
        {
            "scenario_name": f"d{distance:02d}_h{angle:03d}",
            "distance_km": float(distance),
            "heading_deg": float(angle),
            "env_overrides": {
                "blue_x_min": float(distance) - d_jitter,
                "blue_x_max": float(distance) + d_jitter,
                "blue_y_min": -y_jitter,
                "blue_y_max": y_jitter,
                "blue_z_min": 10.0,
                "blue_z_max": 10.0,
                "blue_heading_min": float(angle) - h_jitter,
                "blue_heading_max": float(angle) + h_jitter,
                "red_launch_x_min": 0.0,
                "red_launch_x_max": 0.0,
                "red_launch_y_min": -y_jitter,
                "red_launch_y_max": y_jitter,
                "red_launch_z_min": 10.0,
                "red_launch_z_max": 10.0,
                "missile_update_dt": 0.01,
                "hit_radius": 0.005,
            },
        }
        for distance in distances
        for angle in angles
    ]

    output_root = Path("outputs") / f"tests_1v1_escape_direction_{args.blue_eval_policy}"
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

    grouped: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in all_rows:
        distance = int(float(row["param_distance_km"]))
        heading = int(float(row["param_heading_deg"]))
        key = (distance, heading)
        grouped[key]["win"].append(float(row["win"]))
        grouped[key]["reward"].append(float(row["reward"]))
        grouped[key]["steps"].append(float(row["steps"]))
        grouped[key]["min_dist"].append(float(row["min_dist"]))
        grouped[key]["final_dist"].append(float(row["final_dist"]))
        grouped[key]["final_speed"].append(float(row["final_speed"]))
        grouped[key]["avg_speed"].append(float(row["avg_speed"]))
        grouped[key]["min_speed"].append(float(row["min_speed"]))
        grouped[key]["avg_altitude"].append(float(row["avg_altitude"]))
        grouped[key]["avg_roll_abs_deg"].append(float(row["avg_roll_abs_deg"]))
        grouped[key]["avg_turn_rate_deg"].append(float(row["avg_turn_rate_deg"]))
        grouped[key]["hit"].append(float(row["hit"]))
        grouped[key]["crashed"].append(float(row["crashed"]))
        grouped[key]["timeout"].append(float(row["timeout"]))
        grouped[key]["missiles_exhausted"].append(float(row["missiles_exhausted"]))

    result_rows: list[dict[str, float]] = []
    for (distance, heading), vals in sorted(grouped.items()):
        result_rows.append(
            {
                "distance_km": float(distance),
                "heading_deg": float(heading),
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
    _write_result_csv(results_dir / "result_direction_grid.csv", result_rows)
    _plot_direction(result_rows, results_dir)


if __name__ == "__main__":
    main()
