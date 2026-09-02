"""Blue-aircraft flight-quality metrics and acceptance visualisations.

The input is deliberately a small JSON-safe decision trace so the report can be
regenerated without a simulator or checkpoint.  Coordinates are north/up/east.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G

EPS = 1.0e-9


@dataclass(frozen=True)
class FlightQualityThresholds:
    near_vertical_deg: float = 45.0
    near_vertical_min_s: float = 0.5
    spiral_fpa_deg: float = 30.0
    spiral_heading_rate_deg_s: float = 20.0
    spiral_radius_m: float = 1500.0
    spiral_min_s: float = 1.0
    reversal_window_s: float = 5.0
    reversal_angle_deg: float = 150.0
    reversal_efficiency: float = 0.30
    self_return_min_lag_s: float = 3.0
    self_return_max_lag_s: float = 10.0
    self_return_distance_m: float = 500.0
    self_return_min_path_m: float = 1500.0
    boundary_margin_m: float = 50.0
    max_boundary_time_fraction: float = 0.05
    max_safety_intervention_rate: float = 0.10
    max_survival_rate_drop: float = 0.05


def _segments(mask: np.ndarray, times: np.ndarray, minimum_s: float) -> list[tuple[int, int]]:
    """Return inclusive contiguous true runs whose sampled duration passes a threshold."""
    padded = np.r_[False, mask, False].astype(np.int8)
    changes = np.diff(padded)
    result = []
    for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1) - 1):
        if float(times[stop] - times[start]) + EPS >= minimum_s:
            result.append((int(start), int(stop)))
    return result


def _duration(mask: np.ndarray, times: np.ndarray) -> float:
    if len(times) < 2:
        return 0.0
    # A sampled state describes the interval until the next sample.  There is
    # no observable interval after the terminal sample, so never extrapolate a
    # median step beyond episode duration.
    return float(np.sum(np.diff(times) * mask[:-1]))


def _safe_percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return None if not finite.size else float(np.percentile(finite, percentile))


def _command_vector(action: int) -> np.ndarray:
    axial, normal, bank = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[action]
    return np.asarray([axial, normal * math.cos(bank), normal * math.sin(bank)])


def _opposite_action(left: object, right: object) -> bool:
    if not isinstance(left, int) or not isinstance(right, int):
        return False
    left_command, right_command = _command_vector(left), _command_vector(right)
    scales = np.linalg.norm(left_command), np.linalg.norm(right_command)
    if min(scales) < 2.0:
        return False
    cosine = float(np.dot(left_command, right_command) / max(scales[0] * scales[1], EPS))
    return cosine <= -0.90


def _event_rows(kind: str, segments: Iterable[tuple[int, int]], times: np.ndarray,
                severity: np.ndarray | None = None) -> list[dict[str, Any]]:
    rows = []
    for start, stop in segments:
        row: dict[str, Any] = {"type": kind, "start_s": float(times[start]),
                               "end_s": float(times[stop]), "start_index": start,
                               "end_index": stop}
        if severity is not None:
            row["peak"] = float(np.nanmax(severity[start:stop + 1]))
        rows.append(row)
    return rows


def diagnose_trace(trace: list[dict[str, Any]], *, survived: bool | None = None,
                   thresholds: FlightQualityThresholds = FlightQualityThresholds()) -> dict[str, Any]:
    """Compute raw metrics, hard acceptance flags, time series, and marked events."""
    if len(trace) < 2:
        raise ValueError("flight-quality trace requires at least two samples")
    times = np.asarray([sample["time_s"] for sample in trace], dtype=np.float64)
    positions = np.asarray([sample["blue_position_m"] for sample in trace], dtype=np.float64)
    velocities = np.asarray([sample["blue_velocity_mps"] for sample in trace], dtype=np.float64)
    if positions.shape != (len(trace), 3) or velocities.shape != positions.shape:
        raise ValueError("blue position and velocity samples must be three-dimensional")
    if not np.all(np.diff(times) > 0.0):
        raise ValueError("flight-quality sample times must be strictly increasing")

    speed = np.linalg.norm(velocities, axis=1)
    horizontal_speed = np.hypot(velocities[:, 0], velocities[:, 2])
    ratio = horizontal_speed / np.maximum(speed, EPS)
    fpa = np.degrees(np.arctan2(velocities[:, 1], np.maximum(horizontal_speed, EPS)))
    heading = np.unwrap(np.arctan2(velocities[:, 2], velocities[:, 0]))
    heading_rate = np.degrees(np.gradient(heading, times))
    # Heading is ill-conditioned when horizontal speed is almost zero.  Keep
    # FPA/low-horizontal-speed diagnostics active, but exclude those points
    # from turn-radius and spiral decisions.
    heading_rate_for_turn = heading_rate.copy()
    heading_rate_for_turn[horizontal_speed < 50.0] = np.nan
    radius = horizontal_speed / np.maximum(np.abs(np.radians(heading_rate_for_turn)), EPS)
    radius[np.abs(heading_rate_for_turn) < 0.5] = np.nan

    near_mask = np.abs(fpa) >= thresholds.near_vertical_deg
    near_segments = _segments(near_mask, times, thresholds.near_vertical_min_s)
    spiral_mask = ((np.abs(fpa) >= thresholds.spiral_fpa_deg)
                   & (np.abs(heading_rate_for_turn) >= thresholds.spiral_heading_rate_deg_s)
                   & (radius <= thresholds.spiral_radius_m))
    spiral_segments = _segments(spiral_mask, times, thresholds.spiral_min_s)

    reversal_windows: list[tuple[int, int]] = []
    self_return_windows: list[tuple[int, int]] = []
    reversal_angles: list[float] = []
    reversal_efficiencies: list[float] = []
    detour_ratios: list[float] = []
    for start in range(len(times) - 1):
        stop = int(np.searchsorted(times, times[start] + thresholds.reversal_window_s, side="right") - 1)
        if stop <= start or times[stop] - times[start] < 2.0:
            continue
        left, right = velocities[start], velocities[stop]
        cosine = float(np.dot(left, right) / max(np.linalg.norm(left) * np.linalg.norm(right), EPS))
        angle = math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))
        steps = np.linalg.norm(np.diff(positions[start:stop + 1], axis=0), axis=1)
        path = float(np.sum(steps)); displacement = float(np.linalg.norm(positions[stop] - positions[start]))
        efficiency = displacement / max(path, EPS)
        reversal_angles.append(angle); reversal_efficiencies.append(efficiency)
        detour_ratios.append(path / max(displacement, EPS))
        if angle >= thresholds.reversal_angle_deg and efficiency <= thresholds.reversal_efficiency:
            if not reversal_windows or start > reversal_windows[-1][1]:
                reversal_windows.append((start, stop))
            else:
                reversal_windows[-1] = (reversal_windows[-1][0], max(stop, reversal_windows[-1][1]))

    for start in range(len(times) - 1):
        first = int(np.searchsorted(times, times[start] + thresholds.self_return_min_lag_s,
                                   side="left"))
        last = int(np.searchsorted(times, times[start] + thresholds.self_return_max_lag_s,
                                  side="right"))
        for stop in range(first, min(last, len(times))):
            horizontal_displacement = float(np.linalg.norm(
                positions[stop, [0, 2]] - positions[start, [0, 2]]))
            path = float(np.sum(np.linalg.norm(
                np.diff(positions[start:stop + 1], axis=0), axis=1)))
            if (horizontal_displacement >= thresholds.self_return_distance_m
                    or path <= thresholds.self_return_min_path_m):
                continue
            if not self_return_windows or start > self_return_windows[-1][1]:
                self_return_windows.append((start, stop))
            else:
                self_return_windows[-1] = (self_return_windows[-1][0],
                                           max(stop, self_return_windows[-1][1]))
            break

    lower = float(trace[0].get("min_altitude_m", 8000.0)); upper = float(trace[0].get("max_altitude_m", 12000.0))
    boundary_mask = ((positions[:, 1] - lower <= thresholds.boundary_margin_m)
                     | (upper - positions[:, 1] <= thresholds.boundary_margin_m))
    boundary_segments = _segments(boundary_mask, times, 0.0)
    safety = np.asarray([bool(sample.get("safety_intervened", False)) for sample in trace])
    decision_mask = np.asarray([isinstance(sample.get("raw_action"), int) for sample in trace])
    physics_protection = np.asarray([bool(sample.get("physics_protection_active", False)) for sample in trace])
    executed_load = np.asarray([sample.get("executed_load_body_g", [0.0, 0.0]) for sample in trace],
                               dtype=np.float64)
    load_norm = np.linalg.norm(executed_load, axis=1)
    safety_segments = _segments(safety, times, 0.0)
    raw_actions = [sample.get("raw_action") for sample in trace]
    executed_actions = [sample.get("executed_action") for sample in trace]
    valid_actions = [(a, b) for a, b in zip(raw_actions, executed_actions)
                     if isinstance(a, int) and isinstance(b, int)]
    switches = sum(left != right for left, right in zip(executed_actions, executed_actions[1:])
                   if isinstance(left, int) and isinstance(right, int))

    events = (_event_rows("near_vertical", near_segments, times, np.abs(fpa))
              + _event_rows("spiral", spiral_segments, times, np.abs(fpa))
              + _event_rows("reversal", reversal_windows, times)
              + _event_rows("self_return", self_return_windows, times)
              + _event_rows("altitude_boundary", boundary_segments, times)
              + _event_rows("safety_intervention", safety_segments, times))
    spiral_durations = [times[stop] - times[start] for start, stop in spiral_segments]
    intervention_rate = float(np.mean(safety[decision_mask])) if np.any(decision_mask) else 0.0
    boundary_duration = _duration(boundary_mask, times)
    duration = float(times[-1] - times[0])
    boundary_fraction = boundary_duration / max(duration, EPS)
    five_second_heading_changes = []
    for start in range(len(times) - 1):
        stop = int(np.searchsorted(times, times[start] + 5.0, side="right") - 1)
        if stop > start:
            five_second_heading_changes.append(float(np.sum(
                np.abs(np.diff(np.degrees(heading[start:stop + 1]))))))
    metrics: dict[str, Any] = {
        "duration_s": duration,
        "max_flight_path_angle_deg": float(np.max(fpa)), "min_flight_path_angle_deg": float(np.min(fpa)),
        "max_abs_flight_path_angle_deg": float(np.max(np.abs(fpa))),
        **{f"time_abs_fpa_above_{limit}_deg_s": _duration(np.abs(fpa) >= limit, times)
           for limit in (30, 45, 60)},
        "near_vertical_event_count": len(near_segments),
        "min_horizontal_speed_mps": float(np.min(horizontal_speed)),
        "min_horizontal_speed_ratio": float(np.min(ratio)),
        "time_horizontal_speed_below_150_mps_s": _duration(horizontal_speed < 150.0, times),
        "time_horizontal_speed_ratio_below_0_7_s": _duration(ratio < 0.7, times),
        "low_horizontal_speed_event_count": len(_segments(horizontal_speed < 150.0, times, 0.5)),
        "max_abs_heading_rate_deg_s": float(np.max(np.abs(heading_rate))),
        "p95_abs_heading_rate_deg_s": _safe_percentile(np.abs(heading_rate), 95),
        "time_heading_rate_above_30_deg_s_s": _duration(np.abs(heading_rate) > 30.0, times),
        "continuous_turn_angle_deg": float(np.sum(np.abs(np.diff(np.degrees(heading))))),
        "full_rotation_count": float(np.sum(np.abs(np.diff(heading))) / (2.0 * math.pi)),
        "max_5s_cumulative_heading_change_deg": max(five_second_heading_changes, default=0.0),
        "min_turn_radius_m": _safe_percentile(radius, 0), "p05_turn_radius_m": _safe_percentile(radius, 5),
        "time_turn_radius_below_1000_m_s": _duration(radius < 1000.0, times),
        "tight_turn_event_count": len(_segments(radius < 1000.0, times, 1.0)),
        "spiral_event_count": len(spiral_segments), "spiral_total_duration_s": _duration(spiral_mask, times),
        "spiral_longest_duration_s": float(max(spiral_durations, default=0.0)),
        "spiral_climb_count": sum(float(np.mean(fpa[a:b + 1])) > 0 for a, b in spiral_segments),
        "spiral_dive_count": sum(float(np.mean(fpa[a:b + 1])) < 0 for a, b in spiral_segments),
        "spiral_max_abs_fpa_deg": max((float(np.max(np.abs(fpa[a:b + 1]))) for a, b in spiral_segments), default=None),
        "spiral_min_radius_m": min((float(np.nanmin(radius[a:b + 1])) for a, b in spiral_segments), default=None),
        "reversal_event_count": len(reversal_windows),
        "reversal_total_duration_s": float(sum(times[b] - times[a] for a, b in reversal_windows)),
        "minimum_displacement_efficiency": min(reversal_efficiencies, default=1.0),
        "maximum_velocity_reversal_deg": max(reversal_angles, default=0.0),
        "reversal_with_low_horizontal_speed_count": sum(
            bool(np.min(horizontal_speed[a:b + 1]) < 150.0)
            for a, b in reversal_windows),
        "self_return_event_count": len(self_return_windows),
        "minimum_return_distance_m": min((float(np.linalg.norm(
            positions[b, [0, 2]] - positions[a, [0, 2]]))
            for a, b in self_return_windows), default=None),
        "path_to_displacement_ratio_max": max(detour_ratios, default=1.0),
        "action_switch_rate_hz": switches / max(times[-1] - times[0], EPS),
        "opposite_action_switch_count": sum(
            _opposite_action(a, b) for a, b in zip(executed_actions, executed_actions[1:])),
        "safety_intervention_count": int(np.sum(safety)), "safety_intervention_rate": intervention_rate,
        "physics_protection_count": int(np.sum(physics_protection)),
        "physics_protection_rate": float(np.mean(physics_protection)),
        "time_load_above_6g_s": _duration(load_norm > 6.0, times),
        "max_executed_load_g": float(np.max(load_norm)),
        "raw_executed_action_mismatch_count": sum(a != b for a, b in valid_actions),
        "altitude_boundary_event_count": len(boundary_segments),
        "time_near_altitude_boundary_s": boundary_duration,
        "altitude_boundary_time_fraction": boundary_fraction,
        "survived": survived,
    }
    penalty = (metrics["time_abs_fpa_above_45_deg_s"] / 5.0
               + metrics["time_horizontal_speed_below_150_mps_s"] / 5.0
               + metrics["spiral_total_duration_s"] / 3.0 + len(reversal_windows)
               + len(boundary_segments) * 0.5 + intervention_rate * 5.0)
    metrics["flight_quality_score"] = float(100.0 * math.exp(-penalty))
    acceptance = {"near_vertical_pass": not near_segments, "spiral_pass": not spiral_segments,
                  "reversal_pass": not reversal_windows,
                  "altitude_boundary_pass": boundary_fraction <= thresholds.max_boundary_time_fraction,
                  "safety_intervention_pass": intervention_rate <= thresholds.max_safety_intervention_rate}
    acceptance["flight_quality_pass"] = all(acceptance.values())
    series = {"time_s": times.tolist(), "position_m": positions.tolist(), "velocity_mps": velocities.tolist(),
              "speed_mps": speed.tolist(), "horizontal_speed_mps": horizontal_speed.tolist(),
              "horizontal_speed_ratio": ratio.tolist(), "flight_path_angle_deg": fpa.tolist(),
              "heading_rate_deg_s": heading_rate.tolist(),
              "turn_radius_m": [None if not math.isfinite(value) else float(value) for value in radius],
              "safety_intervened": safety.tolist(), "raw_action": raw_actions,
              "executed_action": executed_actions,
              "executed_load_g": load_norm.tolist(),
              "physics_protection_active": physics_protection.tolist(),
              "min_altitude_m": lower, "max_altitude_m": upper,
              "altitude_recovery_margin_m": float(trace[0].get("altitude_recovery_margin_m", 0.0)),
              "red_positions_m": [sample.get("red_positions_m", []) for sample in trace]}
    return {"metrics": metrics, "acceptance": acceptance, "events": events, "series": series}


def _plot_episode(diagnostic: dict[str, Any], output: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = diagnostic["series"]; t = np.asarray(s["time_s"]); p = np.asarray(s["position_m"])
    fig = plt.figure(figsize=(16, 15)); grid = fig.add_gridspec(3, 2)
    ax3 = fig.add_subplot(grid[0, 0], projection="3d")
    ax3.plot(p[:, 0] / 1000, p[:, 2] / 1000, p[:, 1] / 1000, color="tab:blue", label="Blue")
    red = s["red_positions_m"]
    for slot in range(max((len(row) for row in red), default=0)):
        values = np.asarray([row[slot] for row in red if slot < len(row)])
        if len(values): ax3.plot(values[:, 0] / 1000, values[:, 2] / 1000, values[:, 1] / 1000, color="tab:red", alpha=.5)
    ax3.scatter(*[p[0, i] / 1000 for i in (0, 2, 1)], marker="o", color="green")
    ax3.scatter(*[p[-1, i] / 1000 for i in (0, 2, 1)], marker="x", color="black")
    ax3.set(xlabel="North x (km)", ylabel="East z (km)", zlabel="Up y (km)"); ax3.legend()
    top = fig.add_subplot(grid[0, 1]); top.plot(p[:, 0] / 1000, p[:, 2] / 1000)
    top.set(xlabel="North x (km)", ylabel="East z (km)", title="Horizontal trajectory"); top.axis("equal")
    altitude = fig.add_subplot(grid[1, 0]); altitude.plot(t, p[:, 1], label="altitude")
    margin = s["altitude_recovery_margin_m"]
    for value, color in ((s["min_altitude_m"], "red"), (s["max_altitude_m"], "red"),
                         (s["min_altitude_m"] + margin, "gold"),
                         (s["max_altitude_m"] - margin, "gold")):
        altitude.axhline(value, color=color, linestyle="--")
    altitude.set(xlabel="Time (s)", ylabel="Altitude (m)")
    angles = fig.add_subplot(grid[1, 1]); angles.plot(t, s["flight_path_angle_deg"], label="FPA (deg)"); angles.plot(t, s["heading_rate_deg_s"], label="heading rate (deg/s)")
    for value in (-45, -30, 30, 45): angles.axhline(value, color="red" if abs(value) == 45 else "gold", linestyle="--")
    angles.fill_between(t, -90, 90, where=np.abs(s["flight_path_angle_deg"]) >= 45, color="red", alpha=.12); angles.legend()
    speeds = fig.add_subplot(grid[2, 0]); speeds.plot(t, s["speed_mps"], label="total"); speeds.plot(t, s["horizontal_speed_mps"], label="horizontal")
    radius = speeds.twinx(); radius.plot(t, [np.nan if x is None else x for x in s["turn_radius_m"]], color="purple", alpha=.5, label="radius"); radius.set_ylim(0, 5000); speeds.legend()
    actions = fig.add_subplot(grid[2, 1]); actions.step(t, [np.nan if x is None else x for x in s["raw_action"]], where="post", label="raw"); actions.step(t, [np.nan if x is None else x for x in s["executed_action"]], where="post", label="executed")
    load_axis = actions.twinx(); load_axis.plot(t, s["executed_load_g"], color="tab:red", alpha=.55, label="actual load (g)"); load_axis.set_ylabel("Executed load (g)")
    for when, active in zip(t, s["safety_intervened"]):
        if active:
            for axis in (altitude, angles, speeds, actions): axis.axvline(when, color="black", alpha=.15)
    colors = {"near_vertical": "red", "spiral": "purple", "reversal": "orange"}
    for event in diagnostic["events"]:
        if event["type"] in colors:
            marked = p[event["start_index"]:event["end_index"] + 1]
            top.scatter(marked[:, 0] / 1000, marked[:, 2] / 1000,
                        s=8, color=colors[event["type"]])
            ax3.scatter(marked[:, 0] / 1000, marked[:, 2] / 1000, marked[:, 1] / 1000,
                        s=8, color=colors[event["type"]])
    actions.legend(loc="upper left"); load_axis.legend(loc="upper right"); fig.suptitle(title); fig.tight_layout(); fig.savefig(output, dpi=150); plt.close(fig)


def write_flight_quality_report(episodes: list[dict[str, Any]], output: Path, *,
                                survival_reference: float | None = None,
                                thresholds: FlightQualityThresholds = FlightQualityThresholds(),
                                plots: bool = True, max_plots: int = 20) -> dict[str, Any]:
    """Write one summary table, diagnostic JSON, event CSV, and six-panel plots."""
    output.mkdir(parents=True, exist_ok=True)
    diagnostics = []
    for episode in episodes:
        diagnostic = diagnose_trace(episode["flight_quality_trace"], survived=episode.get("blue_survived"), thresholds=thresholds)
        diagnostic["episode"] = episode.get("episode"); diagnostics.append(diagnostic)
    if plots and max_plots > 0:
        selected = sorted(diagnostics, key=lambda item: (item["acceptance"]["flight_quality_pass"],
                                                         item["metrics"]["flight_quality_score"]))[:max_plots]
        for diagnostic in selected:
            _plot_episode(diagnostic, output / f"episode_{int(diagnostic['episode']):06d}.png",
                          f"Blue flight quality — episode {diagnostic['episode']}")
    survival_values = [item["metrics"]["survived"] for item in diagnostics
                       if item["metrics"]["survived"] is not None]
    survival = float(np.mean(survival_values)) if survival_values else None
    survival_drop = None if survival_reference is None or survival is None else survival_reference - survival
    summary = {"schema_version": 1, "episodes": len(diagnostics), "survival_rate": survival,
               "survival_reference": survival_reference, "survival_rate_drop": survival_drop,
               "survival_pass": survival_drop is None or survival_drop <= thresholds.max_survival_rate_drop,
               "flight_quality_pass_rate": float(np.mean([d["acceptance"]["flight_quality_pass"] for d in diagnostics])) if diagnostics else None,
               "episodes_with_near_vertical": sum(not d["acceptance"]["near_vertical_pass"] for d in diagnostics),
               "episodes_with_spiral": sum(not d["acceptance"]["spiral_pass"] for d in diagnostics),
               "episodes_with_reversal": sum(not d["acceptance"]["reversal_pass"] for d in diagnostics),
               "episodes_touching_altitude_boundary": sum(not d["acceptance"]["altitude_boundary_pass"] for d in diagnostics),
               "episodes_with_excessive_safety_intervention": sum(not d["acceptance"]["safety_intervention_pass"] for d in diagnostics)}
    summary["acceptance_pass"] = bool(summary["survival_pass"] and summary["flight_quality_pass_rate"] == 1.0)
    fields = ["episode", "survived", "flight_quality_score", "flight_quality_pass", "near_vertical_event_count",
              "spiral_event_count", "reversal_event_count", "altitude_boundary_event_count",
              "safety_intervention_rate", "min_horizontal_speed_mps", "min_turn_radius_m"]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fields); writer.writeheader()
        for d in diagnostics:
            writer.writerow({"episode": d["episode"], **{key: d["metrics"].get(key, d["acceptance"].get(key)) for key in fields[1:]}})
    with (output / "events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, ["episode", "type", "start_s", "end_s", "start_index", "end_index", "peak"]); writer.writeheader()
        for d in diagnostics:
            for event in d["events"]: writer.writerow({"episode": d["episode"], **event})
    (output / "report.json").write_text(json.dumps({"summary": summary, "episodes": diagnostics}, indent=2), encoding="utf-8")
    return summary
