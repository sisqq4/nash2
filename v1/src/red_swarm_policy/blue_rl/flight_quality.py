"""Blue-aircraft flight-quality acceptance metrics and diagnostic reports."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G

EPS = 1.0e-9


@dataclass(frozen=True)
class FlightQualityThresholds:
    near_vertical_deg: float = 45.0
    near_vertical_min_s: float = 0.5
    low_horizontal_speed_mps: float = 150.0
    low_horizontal_ratio: float = 0.7
    heading_rate_deg_s: float = 30.0
    tight_radius_m: float = 1000.0
    spiral_fpa_deg: float = 30.0
    spiral_heading_rate_deg_s: float = 20.0
    spiral_radius_m: float = 1500.0
    spiral_min_s: float = 1.0
    reversal_window_s: float = 5.0
    reversal_angle_deg: float = 150.0
    reversal_efficiency: float = 0.30
    self_return_min_s: float = 3.0
    self_return_max_s: float = 10.0
    self_return_distance_m: float = 500.0
    self_return_path_m: float = 1500.0
    boundary_margin_m: float = 100.0
    boundary_min_s: float = 0.5


def _runs(mask: np.ndarray, time: np.ndarray, minimum_s: float) -> list[tuple[int, int]]:
    if not len(mask):
        return []
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    return [(int(a), int(b)) for a, b in zip(starts, stops)
            if float(time[b - 1] - time[a] + (np.median(np.diff(time)) if len(time) > 1 else 0.0)) >= minimum_s]


class FlightQualityTracker:
    """Collect decision-rate snapshots and derive JSON-safe acceptance metrics."""

    def __init__(self, thresholds: FlightQualityThresholds | None = None) -> None:
        self.thresholds = thresholds or FlightQualityThresholds()
        self.samples: list[dict[str, Any]] = []

    def add(self, state: dict[str, object], *, policy_action: int | None = None,
            executed_action: int | None = None, safety_intervened: bool = False,
            safety_reasons: list[str] | None = None) -> None:
        self.samples.append({"time_s": float(state["time_s"]),
                             "position_m": list(state["blue_position_m"]),
                             "velocity_mps": list(state["blue_velocity_mps"]),
                             "red_positions_m": state.get("red_positions_m", []),
                             "min_altitude_m": float(state.get("min_altitude_m", -math.inf)),
                             "max_altitude_m": float(state.get("max_altitude_m", math.inf)),
                             "policy_action": policy_action, "executed_action": executed_action,
                             "safety_intervened": bool(safety_intervened),
                             "safety_reasons": list(safety_reasons or [])})

    def finish(self, *, episode: int, survived: bool) -> dict[str, Any]:
        if len(self.samples) < 2:
            raise ValueError("flight-quality diagnostics require at least two samples")
        t = self.thresholds
        time = np.asarray([s["time_s"] for s in self.samples]); pos = np.asarray([s["position_m"] for s in self.samples])
        vel = np.asarray([s["velocity_mps"] for s in self.samples]); dt = np.diff(time, prepend=time[0])
        if len(time) > 1: dt[0] = np.median(np.diff(time))
        speed = np.linalg.norm(vel, axis=1); horizontal = np.linalg.norm(vel[:, [0, 2]], axis=1)
        ratio = horizontal / np.maximum(speed, EPS)
        fpa = np.degrees(np.arctan2(vel[:, 1], horizontal))
        heading = np.unwrap(np.arctan2(vel[:, 2], vel[:, 0])); rate = np.zeros(len(time))
        rate[1:] = np.degrees(np.diff(heading) / np.maximum(np.diff(time), EPS)); rate[0] = rate[1]
        acceleration = np.zeros_like(vel)
        acceleration[1:] = np.diff(vel, axis=0) / np.maximum(np.diff(time), EPS)[:, None]
        acceleration[0] = acceleration[1]
        actual_load = np.linalg.norm(acceleration + np.array([0.0, 9.80665, 0.0]), axis=1) / 9.80665
        radius = np.full(len(time), math.inf); valid = np.abs(np.radians(rate)) > 1e-4
        radius[valid] = horizontal[valid] / np.abs(np.radians(rate[valid]))
        near_mask = np.abs(fpa) >= t.near_vertical_deg
        low_mask = horizontal < t.low_horizontal_speed_mps
        spiral_mask = ((np.abs(fpa) >= t.spiral_fpa_deg) &
                       (np.abs(rate) >= t.spiral_heading_rate_deg_s) & (radius <= t.spiral_radius_m))
        near_runs = _runs(near_mask, time, t.near_vertical_min_s); spiral_runs = _runs(spiral_mask, time, t.spiral_min_s)
        low_runs = _runs(low_mask | (ratio < t.low_horizontal_ratio), time, .5)
        min_alt = float(self.samples[-1]["min_altitude_m"]); max_alt = float(self.samples[-1]["max_altitude_m"])
        boundary_mask = (pos[:, 1] <= min_alt + t.boundary_margin_m) | (pos[:, 1] >= max_alt - t.boundary_margin_m)
        boundary_runs = _runs(boundary_mask, time, t.boundary_min_s)
        path_steps = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        reversal_events: list[tuple[int, int]] = []; efficiencies: list[float] = []; reversal_angles: list[float] = []
        for end in range(1, len(time)):
            start = int(np.searchsorted(time, time[end] - t.reversal_window_s))
            if start >= end: continue
            cosine = float(np.dot(vel[start], vel[end]) / max(speed[start] * speed[end], EPS))
            angle = math.degrees(math.acos(float(np.clip(cosine, -1, 1))))
            path = float(path_steps[start:end].sum()); displacement = float(np.linalg.norm(pos[end] - pos[start]))
            efficiency = displacement / max(path, EPS); efficiencies.append(efficiency); reversal_angles.append(angle)
            if angle >= t.reversal_angle_deg and efficiency <= t.reversal_efficiency:
                if not reversal_events or start > reversal_events[-1][1]: reversal_events.append((start, end))
                else: reversal_events[-1] = (reversal_events[-1][0], end)
        self_return_events: list[tuple[int, int]] = []; return_distances: list[float] = []
        detour_ratios: list[float] = []
        for end in range(1, len(time)):
            first = int(np.searchsorted(time, time[end] - t.self_return_max_s))
            last = int(np.searchsorted(time, time[end] - t.self_return_min_s, side="right"))
            for start in range(first, min(last, end)):
                displacement = float(np.linalg.norm(pos[end, [0, 2]] - pos[start, [0, 2]]))
                path = float(path_steps[start:end].sum())
                detour_ratios.append(path / max(displacement, EPS))
                if displacement <= t.self_return_distance_m and path >= t.self_return_path_m:
                    return_distances.append(displacement)
                    if not self_return_events or start > self_return_events[-1][1]:
                        self_return_events.append((start, end))
                    else:
                        self_return_events[-1] = (self_return_events[-1][0], end)
                    break
        actions = [s["executed_action"] for s in self.samples if s["executed_action"] is not None]
        switches = sum(a != b for a, b in zip(actions, actions[1:])); safety = np.asarray([s["safety_intervened"] for s in self.samples])
        commands = [BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[a] for a in actions]
        maneuver_vectors = [np.array([command[0], command[1] * math.cos(command[2]) - 1.0,
                                      -command[1] * math.sin(command[2])]) for command in commands]
        opposite = sum(float(np.dot(a, b)) < 0 for a, b in zip(maneuver_vectors, maneuver_vectors[1:]))
        left_right = sum(a[2] * b[2] < 0 for a, b in zip(commands, commands[1:]))
        load_flips = sum(a[0] * b[0] < 0 or a[1] * b[1] < 0 for a, b in zip(commands, commands[1:]))
        longest_extreme_run = current_extreme_run = 0
        previous_extreme: int | None = None
        for action, command in zip(actions, commands):
            extreme = float(np.hypot(command[0], command[1])) >= 6.0
            current_extreme_run = current_extreme_run + 1 if extreme and action == previous_extreme else int(extreme)
            longest_extreme_run = max(longest_extreme_run, current_extreme_run)
            previous_extreme = action if extreme else None
        finite_radius = radius[np.isfinite(radius)]
        events = ([{"type": "near_vertical", "start_s": float(time[a]), "end_s": float(time[b - 1])} for a, b in near_runs] +
                  [{"type": "spiral", "start_s": float(time[a]), "end_s": float(time[b - 1])} for a, b in spiral_runs] +
                  [{"type": "reversal", "start_s": float(time[a]), "end_s": float(time[b])} for a, b in reversal_events] +
                  [{"type": "self_return", "start_s": float(time[a]), "end_s": float(time[b])} for a, b in self_return_events] +
                  [{"type": "altitude_boundary", "start_s": float(time[a]), "end_s": float(time[b - 1])} for a, b in boundary_runs])
        metrics = {
            "max_flight_path_angle_deg": float(fpa.max()), "min_flight_path_angle_deg": float(fpa.min()),
            "max_abs_flight_path_angle_deg": float(np.abs(fpa).max()),
            **{f"time_abs_fpa_above_{v}_deg_s": float(dt[np.abs(fpa) >= v].sum()) for v in (30, 45, 60)},
            "near_vertical_event_count": len(near_runs), "min_horizontal_speed_mps": float(horizontal.min()),
            "min_horizontal_speed_ratio": float(ratio.min()),
            "time_horizontal_speed_below_150_mps_s": float(dt[horizontal < 150].sum()),
            "time_horizontal_speed_ratio_below_0_7_s": float(dt[ratio < .7].sum()),
            "low_horizontal_speed_event_count": len(low_runs), "max_abs_heading_rate_deg_s": float(np.abs(rate).max()),
            "p95_abs_heading_rate_deg_s": float(np.percentile(np.abs(rate), 95)),
            "time_heading_rate_above_30_deg_s_s": float(dt[np.abs(rate) >= 30].sum()),
            "continuous_turn_angle_deg": float(np.abs(np.diff(np.degrees(heading))).sum()),
            "full_rotation_count": float(np.abs(np.diff(heading)).sum() / (2 * math.pi)),
            "min_turn_radius_m": float(finite_radius.min()) if len(finite_radius) else None,
            "p05_turn_radius_m": float(np.percentile(finite_radius, 5)) if len(finite_radius) else None,
            "time_turn_radius_below_1000_m_s": float(dt[radius < 1000].sum()),
            "tight_turn_event_count": len(_runs(radius < t.tight_radius_m, time, .5)),
            "max_estimated_actual_load_g": float(actual_load.max()),
            "time_estimated_actual_load_above_6_g_s": float(dt[actual_load > 6.0].sum()),
            "spiral_event_count": len(spiral_runs),
            "spiral_total_duration_s": sum(float(dt[a:b].sum()) for a, b in spiral_runs),
            "spiral_longest_duration_s": max((float(dt[a:b].sum()) for a, b in spiral_runs), default=0.0),
            "spiral_climb_count": int(sum(float(np.mean(fpa[a:b])) > 0 for a, b in spiral_runs)),
            "spiral_dive_count": int(sum(float(np.mean(fpa[a:b])) < 0 for a, b in spiral_runs)),
            "spiral_max_abs_fpa_deg": max((float(np.abs(fpa[a:b]).max()) for a, b in spiral_runs), default=0.0),
            "spiral_min_radius_m": min((float(radius[a:b].min()) for a, b in spiral_runs), default=None),
            "reversal_event_count": len(reversal_events),
            "reversal_total_duration_s": sum(float(time[b] - time[a]) for a, b in reversal_events),
            "minimum_displacement_efficiency": min(efficiencies, default=1.0),
            "maximum_velocity_reversal_deg": max(reversal_angles, default=0.0),
            "reversal_with_low_horizontal_speed_count": int(sum(
                bool(np.any(low_mask[a:b + 1])) for a, b in reversal_events)),
            "action_switch_rate_hz": switches / max(float(time[-1] - time[0]), EPS),
            "opposite_action_switch_count": opposite, "left_right_flip_count": left_right,
            "positive_negative_load_flip_count": load_flips,
            "max_same_extreme_action_duration_s": longest_extreme_run * float(np.median(np.diff(time))),
            "self_return_event_count": len(self_return_events),
            "minimum_return_distance_m": min(return_distances, default=None),
            "path_to_displacement_ratio_max": max(detour_ratios, default=1.0),
            "altitude_boundary_event_count": len(boundary_runs), "time_near_altitude_boundary_s": float(dt[boundary_mask].sum()),
            "safety_intervention_count": int(safety.sum()),
            "safety_intervention_rate": float(safety.sum() / max(len(actions), 1)),
        }
        abnormal = (metrics["time_abs_fpa_above_45_deg_s"] / 5 + metrics["spiral_total_duration_s"] / 3 +
                    metrics["reversal_event_count"] + metrics["time_near_altitude_boundary_s"] / 5 +
                    metrics["time_estimated_actual_load_above_6_g_s"] / 5 +
                    2 * metrics["safety_intervention_rate"])
        metrics["flight_quality_score"] = float(100 * math.exp(-abnormal))
        verdicts = {"near_vertical": not near_runs, "small_radius_spiral": not spiral_runs,
                    "near_stationary_reversal": not reversal_events and not self_return_events,
                    "altitude_boundary": not boundary_runs,
                    "safety_reliance": metrics["safety_intervention_rate"] < .1}
        trace = {"time_s": time.tolist(), "position_m": pos.tolist(), "velocity_mps": vel.tolist(),
                 "speed_mps": speed.tolist(), "horizontal_speed_mps": horizontal.tolist(),
                 "vertical_speed_mps": vel[:, 1].tolist(), "estimated_actual_load_g": actual_load.tolist(),
                 "flight_path_angle_deg": fpa.tolist(), "heading_rate_deg_s": rate.tolist(),
                 "turn_radius_m": [None if not math.isfinite(x) else float(x) for x in radius],
                 "policy_action": [s["policy_action"] for s in self.samples],
                 "executed_action": [s["executed_action"] for s in self.samples],
                 "commanded_load_g": [None if s["executed_action"] is None else float(np.linalg.norm(
                     BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[s["executed_action"], :2])) for s in self.samples],
                 "safety_intervened": safety.tolist(),
                 "safety_reasons": [s["safety_reasons"] for s in self.samples],
                 "min_altitude_m": min_alt, "max_altitude_m": max_alt,
                 "red_positions_m": [s["red_positions_m"] for s in self.samples]}
        return {"episode": episode, "blue_survived": survived, "metrics": metrics,
                "verdicts": verdicts, "events": sorted(events, key=lambda e: e["start_s"]), "trace": trace}


def append_flight_quality_episode(episode: dict[str, Any], path: Path) -> None:
    """Durably archive a completed episode instead of waiting for run completion."""
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(episode, ensure_ascii=True, allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def write_flight_quality_report(episodes: list[dict[str, Any]], output: Path,
                                *, baseline_survival_rate: float | None = None,
                                plot_limit: int = 10) -> dict[str, Any]:
    """Write summary table, machine-readable traces, event markers, and panels."""
    if plot_limit < 0:
        raise ValueError("plot_limit must be non-negative")
    if baseline_survival_rate is not None and not 0.0 <= baseline_survival_rate <= 1.0:
        raise ValueError("baseline_survival_rate must be in [0, 1]")
    output.mkdir(parents=True, exist_ok=True)
    survival = float(np.mean([e["blue_survived"] for e in episodes])) if episodes else None
    drop = None if baseline_survival_rate is None or survival is None else baseline_survival_rate - survival
    aggregate = {"episodes": len(episodes), "survival_rate": survival,
                 "baseline_survival_rate": baseline_survival_rate, "survival_rate_drop": drop,
                 "survival_rate_significantly_decreased": None if drop is None else drop > .05,
                 "acceptance": {name: all(bool(e["verdicts"][name]) for e in episodes)
                                for name in (episodes[0]["verdicts"] if episodes else [])}}
    payload = {"aggregate": aggregate, "episodes": episodes}
    (output / "flight_quality.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    metric_names = sorted({k for e in episodes for k in e["metrics"]})
    with (output / "flight_quality_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["episode", "blue_survived", *metric_names]); writer.writeheader()
        for e in episodes: writer.writerow({"episode": e["episode"], "blue_survived": e["blue_survived"], **e["metrics"]})
    if plot_limit > 0:
        ranked = sorted(episodes, key=lambda e: float(e["metrics"]["flight_quality_score"]))[:plot_limit]
        for episode in ranked: _plot_episode(episode, output / f"episode_{int(episode['episode']):06d}.png")
    return aggregate


def _plot_episode(episode: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    trace = episode["trace"]; time = np.asarray(trace["time_s"]); pos = np.asarray(trace["position_m"])
    fpa = np.asarray(trace["flight_path_angle_deg"]); rate = np.asarray(trace["heading_rate_deg_s"])
    speed = np.asarray(trace["speed_mps"]); horizontal = np.asarray(trace["horizontal_speed_mps"])
    radius = np.asarray([np.nan if x is None else x for x in trace["turn_radius_m"]])
    fig = plt.figure(figsize=(16, 12)); axes = [fig.add_subplot(3, 2, 1, projection="3d")]
    axes += [fig.add_subplot(3, 2, index) for index in range(2, 7)]
    axes[0].plot(pos[:, 0], pos[:, 2], pos[:, 1], color="royalblue"); axes[0].set(xlabel="North x (m)", ylabel="East z (m)", zlabel="Altitude y (m)")
    axes[1].plot(pos[:, 0], pos[:, 2]); axes[1].set(xlabel="North x (m)", ylabel="East z (m)", title="Horizontal trajectory")
    red = trace["red_positions_m"]
    red_count = max((len(frame) for frame in red), default=0)
    for slot in range(red_count):
        missile = np.asarray([frame[slot] if slot < len(frame) else [np.nan] * 3 for frame in red])
        axes[0].plot(missile[:, 0], missile[:, 2], missile[:, 1], color="firebrick", alpha=.35)
        axes[1].plot(missile[:, 0], missile[:, 2], color="firebrick", alpha=.25)
    axes[0].scatter(*pos[0, [0, 2, 1]], color="green", marker="o"); axes[0].scatter(*pos[-1, [0, 2, 1]], color="black", marker="x")
    axes[1].scatter(pos[0, 0], pos[0, 2], color="green"); axes[1].scatter(pos[-1, 0], pos[-1, 2], color="black", marker="x")
    axes[2].plot(time, pos[:, 1]); axes[2].set(ylabel="Altitude (m)")
    for altitude, style in ((trace["min_altitude_m"], "--"), (trace["max_altitude_m"], "--")):
        axes[2].axhline(altitude, color="black", ls=style, alpha=.6)
    axes[3].plot(time, fpa, label="FPA (deg)"); axes[3].plot(time, rate, label="heading rate (deg/s)", alpha=.7)
    for threshold in (-45, -30, 30, 45): axes[3].axhline(threshold, color="red" if abs(threshold) == 45 else "gold", ls="--")
    axes[3].fill_between(time, fpa, 45, where=fpa >= 45, color="red", alpha=.18)
    axes[3].fill_between(time, fpa, -45, where=fpa <= -45, color="red", alpha=.18)
    axes[3].legend()
    axes[4].plot(time, speed, label="total"); axes[4].plot(time, horizontal, label="horizontal"); axes[4].legend(); axes[4].set(ylabel="Speed (m/s)")
    axes[4].plot(time, np.abs(trace["vertical_speed_mps"]), label="|vertical|", alpha=.7); axes[4].legend()
    radius_axis = axes[4].twinx(); radius_axis.plot(time, radius, color="purple", alpha=.35); radius_axis.set(ylabel="Turn radius (m)", ylim=(0, 5000))
    axes[5].step(time, [np.nan if x is None else x for x in trace["policy_action"]], where="post", label="policy")
    axes[5].step(time, [np.nan if x is None else x for x in trace["executed_action"]], where="post", label="executed", alpha=.7)
    load_axis = axes[5].twinx(); load_axis.plot(time, [np.nan if x is None else x for x in trace["commanded_load_g"]],
                                               color="green", alpha=.45, label="commanded load")
    load_axis.plot(time, trace["estimated_actual_load_g"], color="darkgreen", ls="--", alpha=.6,
                   label="estimated actual load")
    load_axis.set_ylabel("Commanded load (g)")
    for index in np.flatnonzero(trace["safety_intervened"]): axes[5].axvline(time[index], color="black", alpha=.25)
    axes[5].legend(); axes[5].set(ylabel="Action", xlabel="Time (s)")
    colors = {"near_vertical": "red", "spiral": "purple", "reversal": "orange",
              "self_return": "darkorange", "altitude_boundary": "brown"}
    for event in episode["events"]:
        for axis in axes[2:]: axis.axvspan(event["start_s"], event["end_s"], color=colors[event["type"]], alpha=.12)
        selected = (time >= event["start_s"]) & (time <= event["end_s"])
        axes[0].plot(pos[selected, 0], pos[selected, 2], pos[selected, 1], color=colors[event["type"]], lw=3)
        axes[1].plot(pos[selected, 0], pos[selected, 2], color=colors[event["type"]], lw=3)
    fig.suptitle(f"Blue flight quality — episode {episode['episode']} — score {episode['metrics']['flight_quality_score']:.1f}")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)
