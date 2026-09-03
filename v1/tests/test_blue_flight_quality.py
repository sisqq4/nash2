from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from red_swarm_policy.blue_rl.flight_quality import FlightQualityTracker, write_flight_quality_report


def _state(time_s: float, position: list[float], velocity: list[float]) -> dict[str, object]:
    return {"time_s": time_s, "blue_position_m": position, "blue_velocity_mps": velocity,
            "red_positions_m": [[10_000.0, 9_000.0, 0.0]],
            "min_altitude_m": 8_000.0, "max_altitude_m": 12_000.0}


def test_level_straight_flight_passes_hard_acceptance() -> None:
    tracker = FlightQualityTracker()
    for index in range(21):
        tracker.add(_state(index * .1, [30.0 * index, 10_000.0, 0.0], [300.0, 0.0, 0.0]),
                    policy_action=0, executed_action=0)
    result = tracker.finish(episode=1, survived=True)
    assert all(result["verdicts"].values())
    assert result["metrics"]["flight_quality_score"] == pytest.approx(100.0)
    assert result["metrics"]["min_horizontal_speed_ratio"] == pytest.approx(1.0)


def test_sustained_vertical_tight_turn_creates_timed_events() -> None:
    tracker = FlightQualityTracker()
    radius, horizontal, vertical = 500.0, 250.0, 300.0
    omega = horizontal / radius
    for index in range(31):
        now = index * .1; angle = omega * now
        position = [radius * math.sin(angle), 10_000.0 + vertical * now,
                    radius * (1.0 - math.cos(angle))]
        velocity = [horizontal * math.cos(angle), vertical, horizontal * math.sin(angle)]
        tracker.add(_state(now, position, velocity), policy_action=7, executed_action=7,
                    safety_intervened=index in {5, 6})
    result = tracker.finish(episode=2, survived=False)
    assert result["metrics"]["near_vertical_event_count"] == 1
    assert result["metrics"]["spiral_event_count"] == 1
    assert result["metrics"]["spiral_total_duration_s"] >= 2.9
    assert result["metrics"]["safety_intervention_count"] == 2
    assert {event["type"] for event in result["events"]} >= {"near_vertical", "spiral"}


def test_report_writes_json_and_csv_and_validates_options(tmp_path: Path) -> None:
    tracker = FlightQualityTracker()
    for index in range(11):
        tracker.add(_state(index * .1, [30.0 * index, 10_000.0, 0.0], [300.0, 0.0, 0.0]),
                    policy_action=0, executed_action=0)
    episode = tracker.finish(episode=3, survived=True)
    summary = write_flight_quality_report([episode], tmp_path, baseline_survival_rate=.9, plot_limit=0)
    assert summary["survival_rate_significantly_decreased"] is False
    assert json.loads((tmp_path / "flight_quality.json").read_text())["episodes"][0]["episode"] == 3
    with (tmp_path / "flight_quality_summary.csv").open() as stream:
        assert next(csv.DictReader(stream))["flight_quality_score"] == "100.0"
    with pytest.raises(ValueError, match="plot_limit"):
        write_flight_quality_report([], tmp_path, plot_limit=-1)
    with pytest.raises(ValueError, match="baseline_survival_rate"):
        write_flight_quality_report([], tmp_path, baseline_survival_rate=1.1, plot_limit=0)
