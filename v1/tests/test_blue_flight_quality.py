from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from red_swarm_policy.blue_flight_quality import diagnose_trace, write_flight_quality_report


def _trace(*, vertical: bool = False, spiral: bool = False, intervention: bool = False):
    rows = []
    for index in range(61):
        time_s = index * 0.1
        if spiral:
            angle = math.radians(index * 3.0)
            position = [1000.0 * math.sin(angle), 9000.0 + index * 20.0,
                        1000.0 * (1.0 - math.cos(angle))]
            velocity = [300.0 * math.cos(angle), 200.0, 300.0 * math.sin(angle)]
        elif vertical:
            position = [index * 10.0, 9000.0 + index * 30.0, 0.0]
            velocity = [100.0, 300.0, 0.0]
        else:
            position = [index * 30.0, 10000.0, 0.0]
            velocity = [300.0, 0.0, 0.0]
        rows.append({"time_s": time_s, "blue_position_m": position,
                     "blue_velocity_mps": velocity, "red_positions_m": [[50000.0, 9000.0, 0.0]],
                     "min_altitude_m": 8000.0, "max_altitude_m": 12000.0,
                     "raw_action": 1, "executed_action": 0 if intervention else 1,
                     "safety_intervened": intervention})
    return rows


def test_near_vertical_event_requires_duration_and_reports_horizontal_separation() -> None:
    result = diagnose_trace(_trace(vertical=True))
    assert result["metrics"]["near_vertical_event_count"] == 1
    assert result["metrics"]["time_abs_fpa_above_45_deg_s"] == pytest.approx(6.0)
    assert result["metrics"]["min_horizontal_speed_ratio"] < .4
    assert result["acceptance"]["near_vertical_pass"] is False
    assert any(event["type"] == "near_vertical" for event in result["events"])


def test_joint_spiral_detector_does_not_flag_straight_climb() -> None:
    straight = diagnose_trace(_trace(vertical=True))
    turning = diagnose_trace(_trace(spiral=True))
    assert straight["metrics"]["spiral_event_count"] == 0
    assert turning["metrics"]["spiral_event_count"] == 1
    assert turning["metrics"]["spiral_climb_count"] == 1
    assert turning["metrics"]["spiral_min_radius_m"] < 1500.0


def test_duration_never_exceeds_episode_and_brief_boundary_contact_is_not_frequent() -> None:
    trace = _trace(vertical=True)
    trace[20]["blue_position_m"][1] = 8000.0
    result = diagnose_trace(trace)
    assert result["metrics"]["time_abs_fpa_above_45_deg_s"] <= result["metrics"]["duration_s"]
    assert result["metrics"]["altitude_boundary_event_count"] == 1
    assert result["acceptance"]["altitude_boundary_pass"] is True


def test_self_return_uses_trajectory_distance_and_consolidates_sliding_windows() -> None:
    trace = []
    radius, angular_rate = 300.0, 2.0 * math.pi / 10.0
    for index in range(101):
        time_s = index * .1; angle = angular_rate * time_s
        trace.append({"time_s": time_s,
                      "blue_position_m": [radius * math.sin(angle), 10000.0,
                                          radius * (1.0 - math.cos(angle))],
                      "blue_velocity_mps": [radius * angular_rate * math.cos(angle), 0.0,
                                            radius * angular_rate * math.sin(angle)],
                      "raw_action": 0, "executed_action": 0})
    result = diagnose_trace(trace)
    assert result["metrics"]["self_return_event_count"] == 1
    assert result["metrics"]["minimum_return_distance_m"] < 500.0


def test_report_writes_summary_events_and_survival_gate(tmp_path: Path) -> None:
    episodes = [{"episode": 1, "blue_survived": True, "flight_quality_trace": _trace()},
                {"episode": 2, "blue_survived": False,
                 "flight_quality_trace": _trace(intervention=True)}]
    summary = write_flight_quality_report(episodes, tmp_path, survival_reference=.60, plots=False)
    assert summary["survival_rate"] == pytest.approx(.5)
    assert summary["survival_pass"] is False
    assert summary["episodes_with_excessive_safety_intervention"] == 1
    assert json.loads((tmp_path / "report.json").read_text())["summary"]["episodes"] == 2
    with (tmp_path / "summary.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2
