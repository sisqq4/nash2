from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .env import (
    BlueEvasionController,
    BlueEvasionRuleMachine,
    EnvironmentConfig,
    RedAction,
    RedBlueEngagementEnv,
    ScenarioConfig,
    los_kinematics,
)


@dataclass(frozen=True)
class TrajectorySample:
    time_s: float
    step_count: int
    red_position_m: np.ndarray
    blue_position_m: np.ndarray
    red_speed_mps: float
    blue_speed_mps: float
    range_m: float
    red_alive: bool
    blue_alive: bool
    seeker_locked: bool
    guidance_mode: str
    flight_path_angle_deg: float
    pn_load_g: float
    gravity_load_g: float
    final_load_g: float
    blue_action_index: int
    blue_mode: str


def _speed(velocity_mps: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(velocity_mps, dtype=np.float64)))


def _flight_path_angle_deg(velocity_mps: np.ndarray) -> float:
    speed = max(_speed(velocity_mps), 1.0e-9)
    return math.degrees(math.asin(float(np.clip(velocity_mps[1] / speed, -1.0, 1.0))))


def _sample(
    environment: RedBlueEngagementEnv,
    *,
    blue_action_index: int,
    blue_mode: str,
) -> TrajectorySample:
    assert environment.state is not None
    state = environment.state
    red = state.red[0]
    blue = state.blue[0]
    return TrajectorySample(
        time_s=float(state.time_s),
        step_count=int(state.step_count),
        red_position_m=red.position_m.astype(np.float64).copy(),
        blue_position_m=blue.position_m.astype(np.float64).copy(),
        red_speed_mps=_speed(red.velocity_mps),
        blue_speed_mps=_speed(blue.velocity_mps),
        range_m=float(los_kinematics(red, blue).range_m),
        red_alive=bool(red.alive),
        blue_alive=bool(blue.alive),
        seeker_locked=bool(red.seeker_locked),
        guidance_mode=str(red.guidance_mode),
        flight_path_angle_deg=_flight_path_angle_deg(red.velocity_mps),
        pn_load_g=_speed(red.pn_load_body_g[1:]),
        gravity_load_g=_speed(red.gravity_load_body_g[1:]),
        final_load_g=_speed(red.final_load_body_g[1:]),
        blue_action_index=int(blue_action_index),
        blue_mode=str(blue_mode),
    )


def _terminal_reason(final_info: dict[str, Any]) -> str:
    if int(final_info.get("hit_count", 0)) > 0:
        return "valid_hit"
    loss_events = final_info.get("red_loss_events", [])
    if loss_events:
        return str(loss_events[-1].get("loss_reason", "red_loss"))
    if bool(final_info.get("timeout", False)):
        return "environment_timeout"
    return "stopped_without_terminal_event"


def run_episode(seed: int, sample_interval_steps: int) -> tuple[list[TrajectorySample], dict[str, Any]]:
    if sample_interval_steps <= 0:
        raise ValueError("sample_interval_steps must be positive")
    config = EnvironmentConfig(
        scenario=ScenarioConfig(red_count=1, blue_count=1),
        policy_start_mode="launch",
    )
    environment = RedBlueEngagementEnv(
        config,
        device="cpu",
        record_replay=False,
    )
    controller = BlueEvasionController(BlueEvasionRuleMachine(config))
    controller.reset()
    environment.reset(
        seed=seed,
        style="one_to_one",
        red_count=1,
        blue_count=1,
        start_mode="launch",
    )
    assert environment.state is not None

    initial_red_position = environment.state.red[0].position_m.astype(np.float64).copy()
    initial_blue_position = environment.state.blue[0].position_m.astype(np.float64).copy()
    initial_red_speed = _speed(environment.state.red[0].velocity_mps)
    initial_blue_speed = _speed(environment.state.blue[0].velocity_mps)
    initial_range = float(los_kinematics(environment.state.red[0], environment.state.blue[0]).range_m)

    samples: list[TrajectorySample] = []
    current_action_index = 0
    current_mode = "cruise"
    samples.append(
        _sample(
            environment,
            blue_action_index=current_action_index,
            blue_mode=current_mode,
        )
    )

    final_info: dict[str, Any] = {}
    decision_count = 0
    blue_action_counts: Counter[int] = Counter()
    blue_mode_counts: Counter[str] = Counter()
    max_pn_load_g = 0.0
    max_gravity_load_g = 0.0
    max_final_load_g = 0.0
    max_abs_bias = 0.0
    red_speed_min = math.inf
    red_speed_max = 0.0
    blue_speed_min = math.inf
    blue_speed_max = 0.0
    red_altitude_min = math.inf
    red_altitude_max = -math.inf
    blue_altitude_min = math.inf
    blue_altitude_max = -math.inf
    all_finite = True
    first_lock_time_s: float | None = None
    locked_steps = 0
    guidance_steps = 0
    boost_exit_speed_mps: float | None = None
    boost_exit_flight_path_angle_deg: float | None = None
    red_action = RedAction(
        target_indices=np.array([0], dtype=np.int64),
        guidance_bias=np.zeros((1, 2), dtype=np.float64),
    )

    while not environment._episode_done:
        assert environment.state is not None
        blue_action, decision = controller.action_for(environment.state)
        if decision is not None:
            decision_count += 1
            current_action_index = int(decision.action_indices[0])
            current_mode = str(decision.modes[0])
            blue_action_counts[current_action_index] += 1
            blue_mode_counts[current_mode] += 1

        if environment.policy_ready:
            step = environment.step(red_action=red_action, blue_action=blue_action)
        else:
            step = environment.step(blue_action=blue_action)
        final_info = dict(step.info)

        state = environment.state
        assert state is not None
        red = state.red[0]
        blue = state.blue[0]
        red_speed = _speed(red.velocity_mps)
        blue_speed = _speed(blue.velocity_mps)
        pn_load = _speed(red.pn_load_body_g[1:])
        gravity_load = _speed(red.gravity_load_body_g[1:])
        final_load = _speed(red.final_load_body_g[1:])
        max_pn_load_g = max(max_pn_load_g, pn_load)
        max_gravity_load_g = max(max_gravity_load_g, gravity_load)
        max_final_load_g = max(max_final_load_g, final_load)
        max_abs_bias = max(max_abs_bias, float(np.max(np.abs(red.guidance_bias))))
        red_speed_min = min(red_speed_min, red_speed)
        red_speed_max = max(red_speed_max, red_speed)
        blue_speed_min = min(blue_speed_min, blue_speed)
        blue_speed_max = max(blue_speed_max, blue_speed)
        red_altitude_min = min(red_altitude_min, float(red.position_m[1]))
        red_altitude_max = max(red_altitude_max, float(red.position_m[1]))
        blue_altitude_min = min(blue_altitude_min, float(blue.position_m[1]))
        blue_altitude_max = max(blue_altitude_max, float(blue.position_m[1]))
        all_finite = all_finite and bool(
            np.all(np.isfinite(red.position_m))
            and np.all(np.isfinite(red.velocity_mps))
            and np.all(np.isfinite(blue.position_m))
            and np.all(np.isfinite(blue.velocity_mps))
            and np.all(np.isfinite(red.final_load_body_g))
        )
        if state.time_s >= config.policy_entry_time_s:
            guidance_steps += 1
            if red.seeker_locked:
                locked_steps += 1
                if first_lock_time_s is None:
                    first_lock_time_s = float(state.time_s)
        if state.step_count == config.policy_entry_steps:
            boost_exit_speed_mps = red_speed
            boost_exit_flight_path_angle_deg = _flight_path_angle_deg(red.velocity_mps)

        if state.step_count % sample_interval_steps == 0 or step.done:
            samples.append(
                _sample(
                    environment,
                    blue_action_index=current_action_index,
                    blue_mode=current_mode,
                )
            )
        if step.done:
            break

    if samples[-1].step_count != environment.state.step_count:
        samples.append(
            _sample(
                environment,
                blue_action_index=current_action_index,
                blue_mode=current_mode,
            )
        )

    boost_exit_speed_error_mps = (
        None
        if boost_exit_speed_mps is None
        else abs(boost_exit_speed_mps - config.missile.max_speed_mps)
    )
    blue_speed_within_limits = (
        blue_speed_min >= config.aircraft.min_speed_mps - 1.0e-9
        and blue_speed_max <= config.aircraft.max_speed_mps + 1.0e-9
    )
    blue_altitude_within_limits = (
        blue_altitude_min >= config.aircraft.min_altitude_m - 1.0e-9
        and blue_altitude_max <= config.aircraft.max_altitude_m + 1.0e-9
    )
    load_within_limit = max_final_load_g <= config.missile.max_load_factor_g + 1.0e-9
    boost_exit_valid = (
        boost_exit_speed_error_mps is not None
        and boost_exit_speed_error_mps <= config.policy_entry_speed_tolerance_mps
    )
    boost_exit_angle_valid = (
        boost_exit_flight_path_angle_deg is not None
        and abs(boost_exit_flight_path_angle_deg - config.missile.boost_climb_angle_deg)
        <= config.policy_entry_flight_path_tolerance_deg
    )
    zero_bias_valid = max_abs_bias == 0.0
    flight_normal = bool(
        all_finite
        and blue_speed_within_limits
        and blue_altitude_within_limits
        and load_within_limit
        and boost_exit_valid
        and boost_exit_angle_valid
        and zero_bias_valid
    )
    hit = int(final_info.get("hit_count", 0)) == 1
    final_state = environment.state
    assert final_state is not None
    summary: dict[str, Any] = {
        "seed": int(seed),
        "device": "cpu",
        "scenario_style": "one_to_one",
        "red_count": 1,
        "blue_count": 1,
        "blue_policy": "BlueEvasionController(BlueEvasionRuleMachine)",
        "blue_decision_interval_s": 0.1,
        "blue_action_library_size": 29,
        "red_guidance": "proportional_navigation_plus_gravity_compensation",
        "proportional_navigation_gain": config.missile.proportional_navigation_gain,
        "guidance_bias": [0.0, 0.0],
        "time_step_s": config.time_step_s,
        "boost_duration_s": config.missile.boost_duration_s,
        "lethal_radius_m": config.missile.lethal_radius_m,
        "initial": {
            "red_position_m": initial_red_position.tolist(),
            "blue_position_m": initial_blue_position.tolist(),
            "red_speed_mps": initial_red_speed,
            "blue_speed_mps": initial_blue_speed,
            "range_m": initial_range,
        },
        "terminal": {
            "time_s": float(final_state.time_s),
            "step_count": int(final_state.step_count),
            "reason": _terminal_reason(final_info),
            "hit": hit,
            "miss_distance_m": float(final_info.get("miss_distance_m", math.nan)),
            "red_alive": bool(final_state.red[0].alive),
            "blue_alive": bool(final_state.blue[0].alive),
        },
        "guidance_metrics": {
            "first_lock_time_s": first_lock_time_s,
            "lock_fraction_after_boost": locked_steps / max(guidance_steps, 1),
            "boost_exit_speed_mps": boost_exit_speed_mps,
            "boost_exit_speed_error_mps": boost_exit_speed_error_mps,
            "boost_exit_flight_path_angle_deg": boost_exit_flight_path_angle_deg,
            "red_speed_min_mps": red_speed_min,
            "red_speed_max_mps": red_speed_max,
            "red_altitude_min_m": red_altitude_min,
            "red_altitude_max_m": red_altitude_max,
            "max_pn_load_g": max_pn_load_g,
            "max_gravity_load_g": max_gravity_load_g,
            "max_final_load_g": max_final_load_g,
            "max_abs_guidance_bias": max_abs_bias,
        },
        "blue_metrics": {
            "decision_count": decision_count,
            "action_counts": {str(key): value for key, value in sorted(blue_action_counts.items())},
            "mode_counts": dict(sorted(blue_mode_counts.items())),
            "speed_min_mps": blue_speed_min,
            "speed_max_mps": blue_speed_max,
            "altitude_min_m": blue_altitude_min,
            "altitude_max_m": blue_altitude_max,
        },
        "checks": {
            "all_state_values_finite": all_finite,
            "zero_guidance_bias": zero_bias_valid,
            "boost_exit_speed_valid": boost_exit_valid,
            "boost_exit_flight_path_angle_valid": boost_exit_angle_valid,
            "red_load_within_35g": load_within_limit,
            "blue_speed_within_100_600_mps": blue_speed_within_limits,
            "blue_altitude_within_8000_12000_m": blue_altitude_within_limits,
            "flight_normal": flight_normal,
            "intercept_feasible": hit,
        },
    }
    return samples, summary


def write_trajectory_csv(samples: list[TrajectorySample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "time_s",
                "step_count",
                "red_x_north_m",
                "red_y_up_m",
                "red_z_east_m",
                "blue_x_north_m",
                "blue_y_up_m",
                "blue_z_east_m",
                "red_speed_mps",
                "blue_speed_mps",
                "range_m",
                "red_alive",
                "blue_alive",
                "seeker_locked",
                "guidance_mode",
                "flight_path_angle_deg",
                "pn_load_g",
                "gravity_load_g",
                "final_load_g",
                "blue_action_index",
                "blue_mode",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.time_s,
                    sample.step_count,
                    *sample.red_position_m.tolist(),
                    *sample.blue_position_m.tolist(),
                    sample.red_speed_mps,
                    sample.blue_speed_mps,
                    sample.range_m,
                    int(sample.red_alive),
                    int(sample.blue_alive),
                    int(sample.seeker_locked),
                    sample.guidance_mode,
                    sample.flight_path_angle_deg,
                    sample.pn_load_g,
                    sample.gravity_load_g,
                    sample.final_load_g,
                    sample.blue_action_index,
                    sample.blue_mode,
                ]
            )


def _padded_limits(values: np.ndarray, minimum_span: float, padding_ratio: float = 0.06) -> tuple[float, float]:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    center = 0.5 * (minimum + maximum)
    span = max(maximum - minimum, minimum_span)
    half_span = 0.5 * span * (1.0 + 2.0 * padding_ratio)
    return center - half_span, center + half_span


def write_trajectory_gif(
    samples: list[TrajectorySample],
    summary: dict[str, Any],
    path: Path,
    *,
    fps: int,
    max_frames: int,
) -> int:
    if fps <= 0 or max_frames < 2:
        raise ValueError("fps must be positive and max_frames must be at least 2")
    path.parent.mkdir(parents=True, exist_ok=True)
    red_positions_km = np.stack([sample.red_position_m for sample in samples]) / 1000.0
    blue_positions_km = np.stack([sample.blue_position_m for sample in samples]) / 1000.0
    north = np.concatenate([red_positions_km[:, 0], blue_positions_km[:, 0]])
    east = np.concatenate([red_positions_km[:, 2], blue_positions_km[:, 2]])
    up = np.concatenate([red_positions_km[:, 1], blue_positions_km[:, 1]])
    north_limits = _padded_limits(north, minimum_span=20.0)
    east_limits = _padded_limits(east, minimum_span=20.0)
    up_limits = _padded_limits(up, minimum_span=4.0)

    frame_count = min(max_frames, len(samples))
    frame_indices = np.unique(np.linspace(0, len(samples) - 1, frame_count, dtype=np.int64))
    fig = plt.figure(figsize=(10.0, 7.2), dpi=90)
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.04, top=0.88)
    ax.set_xlim(*north_limits)
    ax.set_ylim(*east_limits)
    ax.set_zlim(*up_limits)
    ax.set_xlabel("x / North (km)")
    ax.set_ylabel("z / East (km)")
    ax.set_zlabel("y / Up (km)")
    ax.view_init(elev=24.0, azim=-58.0)
    horizontal_span = max(north_limits[1] - north_limits[0], east_limits[1] - east_limits[0])
    vertical_span = max(up_limits[1] - up_limits[0], 0.18 * horizontal_span)
    ax.set_box_aspect(
        (
            north_limits[1] - north_limits[0],
            east_limits[1] - east_limits[0],
            vertical_span,
        )
    )
    ax.grid(True, alpha=0.3)
    title = "Single red PN (N=3.5) + gravity compensation vs blue rule evasion"
    fig.suptitle(title, y=0.975, fontsize=12)

    red_line, = ax.plot([], [], [], color="#d62728", linewidth=2.2, label="Red missile")
    blue_line, = ax.plot([], [], [], color="#1f77b4", linewidth=2.2, label="Blue aircraft")
    red_point, = ax.plot([], [], [], marker="o", color="#d62728", markersize=7, linestyle="None")
    blue_point, = ax.plot([], [], [], marker="o", color="#1f77b4", markersize=7, linestyle="None")
    los_line, = ax.plot([], [], [], color="#666666", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.scatter(
        [red_positions_km[0, 0]],
        [red_positions_km[0, 2]],
        [red_positions_km[0, 1]],
        marker="^",
        color="#8c1d18",
        s=48,
        label="Red start",
    )
    ax.scatter(
        [blue_positions_km[0, 0]],
        [blue_positions_km[0, 2]],
        [blue_positions_km[0, 1]],
        marker="s",
        color="#174a7e",
        s=42,
        label="Blue start",
    )
    status_text = fig.text(
        0.025,
        0.895,
        "",
        ha="left",
        va="top",
        family="monospace",
        fontsize=9.2,
        color="#222222",
    )
    ax.legend(loc="upper right", fontsize=8.5)

    def update(frame_number: int) -> tuple[Any, ...]:
        sample_index = int(frame_indices[frame_number])
        sample = samples[sample_index]
        red_history = red_positions_km[: sample_index + 1]
        blue_history = blue_positions_km[: sample_index + 1]
        red_current = red_positions_km[sample_index]
        blue_current = blue_positions_km[sample_index]
        red_line.set_data(red_history[:, 0], red_history[:, 2])
        red_line.set_3d_properties(red_history[:, 1])
        blue_line.set_data(blue_history[:, 0], blue_history[:, 2])
        blue_line.set_3d_properties(blue_history[:, 1])
        red_point.set_data([red_current[0]], [red_current[2]])
        red_point.set_3d_properties([red_current[1]])
        blue_point.set_data([blue_current[0]], [blue_current[2]])
        blue_point.set_3d_properties([blue_current[1]])
        los_line.set_data([red_current[0], blue_current[0]], [red_current[2], blue_current[2]])
        los_line.set_3d_properties([red_current[1], blue_current[1]])
        phase = "BOOST" if sample.time_s < 7.0 else sample.guidance_mode.upper()
        status_text.set_text(
            f"t={sample.time_s:7.2f} s  phase={phase:7s}  range={sample.range_m / 1000.0:7.3f} km\n"
            f"red={sample.red_speed_mps:7.1f} m/s  blue={sample.blue_speed_mps:6.1f} m/s  "
            f"lock={int(sample.seeker_locked)}  PN={sample.pn_load_g:5.2f} g  "
            f"Gcomp={sample.gravity_load_g:4.2f} g\n"
            f"blue mode={sample.blue_mode:8s}  action index={sample.blue_action_index:2d}"
        )
        return red_line, blue_line, red_point, blue_point, los_line, status_text

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frame_indices),
        interval=1000.0 / fps,
        blit=False,
    )
    animation.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
    summary["gif"] = {
        "frame_count": int(len(frame_indices)),
        "fps": int(fps),
        "duration_s": len(frame_indices) / fps,
    }
    return int(len(frame_indices))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate one PN-plus-gravity-compensation missile against one rule-evasion aircraft."
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/single_pn_validation"))
    parser.add_argument("--sample-interval-s", type=float, default=0.05)
    parser.add_argument("--gif-fps", type=int, default=12)
    parser.add_argument("--gif-max-frames", type=int, default=240)
    parser.add_argument("--no-gif", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = EnvironmentConfig()
    interval_ratio = args.sample_interval_s / config.time_step_s
    sample_interval_steps = int(round(interval_ratio))
    if not math.isclose(interval_ratio, sample_interval_steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("sample interval must be an integer multiple of the 0.005 s physics step")
    samples, summary = run_episode(args.seed, sample_interval_steps)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"single_red_pn_vs_rule_blue_seed_{args.seed}"
    csv_path = args.output_dir / f"{stem}_trajectory.csv"
    summary_path = args.output_dir / f"{stem}_summary.json"
    gif_path = args.output_dir / f"{stem}.gif"
    write_trajectory_csv(samples, csv_path)
    if not args.no_gif:
        write_trajectory_gif(
            samples,
            summary,
            gif_path,
            fps=args.gif_fps,
            max_frames=args.gif_max_frames,
        )
    summary["artifacts"] = {
        "trajectory_csv": str(csv_path),
        "summary_json": str(summary_path),
        "trajectory_gif": None if args.no_gif else str(gif_path),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(summary, ensure_ascii=True, separators=(",", ":"), allow_nan=False))
    return 0 if summary["checks"]["flight_normal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
