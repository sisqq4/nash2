from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from red_swarm_policy.blue_rl.flight_quality import (
    FlightQualityTracker,
    _build_episode_figure,
    _plot_episode,
    append_flight_quality_episode,
    replot_flight_quality_episodes,
    write_flight_quality_report,
)
from red_swarm_policy.plot_localization import chinese_plot_context


def test_episode_plot_shows_large_radii_separate_units_and_all_load_legends(tmp_path: Path) -> None:
    import matplotlib
    import numpy as np

    tracker = FlightQualityTracker()
    for index in range(11):
        tracker.add(_state(index * .1, [30.0 * index, 10_000.0, 0.0], [300.0, 0.0, 0.0]),
                    policy_action=0, executed_action=0)
    episode = tracker.finish(episode=1, survived=True)
    episode["trace"]["turn_radius_m"] = [None, *[10000. * index for index in range(1, 11)]]
    episode["trace"]["estimated_actual_load_g"] = [2.] * 11
    original = json.dumps(episode, allow_nan=False)
    backend = matplotlib.get_backend()
    figure = _build_episode_figure(episode)
    try:
        axes = {axis.get_ylabel(): axis for axis in figure.axes}
        radius_axis = axes["转弯半径（米）"]
        assert radius_axis.get_ylim()[1] > 100000.
        np.testing.assert_allclose(radius_axis.lines[0].get_ydata()[1:], list(range(10000, 100001, 10000)))
        assert "航迹倾角（度）" in axes and "航向角速度（度/秒）" in axes
        assert axes["航向角速度（度/秒）"].get_ylim()[0] <= -1.
        assert axes["航向角速度（度/秒）"].get_ylim()[1] >= 1.
        assert "过载（g）" in axes and "指令过载（g）" not in axes
        assert axes["过载（g）"].get_ylim()[0] == 0.
        assert axes["过载（g）"].get_ylim()[1] > 2.
        legends = [text.get_text() for axis in figure.axes if axis.get_legend()
                   for text in axis.get_legend().get_texts()]
        assert {"指令过载", "估计实际过载", "转弯半径"} <= set(legends)
        assert figure._suptitle.get_text() == "蓝机飞行质量—第 1 回合—评分 100.0"
        with chinese_plot_context():
            figure.canvas.draw()
    finally:
        figure.clear()
    _plot_episode(episode, tmp_path / "episode.png")
    assert (tmp_path / "episode.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert json.dumps(episode, allow_nan=False) == original
    assert matplotlib.get_backend() == backend


def test_missing_turn_radius_is_marked_unavailable() -> None:
    tracker = FlightQualityTracker()
    for index in range(11):
        tracker.add(_state(index * .1, [30.0 * index, 10_000.0, 0.0], [300.0, 0.0, 0.0]))
    episode = tracker.finish(episode=1, survived=True)
    episode["trace"]["turn_radius_m"] = [None] * 11
    figure = _build_episode_figure(episode)
    try:
        assert "无可用转弯半径" in [text.get_text() for axis in figure.axes for text in axis.texts]
    finally:
        figure.clear()


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


def test_low_horizontal_speed_suppresses_false_spiral_metrics() -> None:
    tracker = FlightQualityTracker()
    horizontal, vertical = 50.0, 300.0
    for index in range(21):
        now = index * .1
        # Deliberately jump the horizontal azimuth while it is below the
        # heading-valid speed threshold.  These jumps must not become turns.
        angle = index * math.pi / 2
        velocity = [horizontal * math.cos(angle), vertical, horizontal * math.sin(angle)]
        tracker.add(_state(now, [0.0, 10_000.0 + vertical * now, 0.0], velocity),
                    policy_action=7, executed_action=7)

    result = tracker.finish(episode=9, survived=False)

    assert result["metrics"]["spiral_event_count"] == 0
    assert result["metrics"]["steep_low_horizontal_speed_event_count"] == 1
    assert result["metrics"]["max_abs_heading_rate_deg_s"] is None
    assert all(value is None for value in result["trace"]["turn_radius_m"])
    assert {event["type"] for event in result["events"]} >= {"steep_low_horizontal_speed"}


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


@pytest.mark.parametrize("actions,left_right_flips,load_flips", [
    ([0], 0, 0),
    ([0] * 10, 0, 0),
    ([5, 8, 5], 2, 0),
    ([1, 3, 1], 0, 2),
    ([11, 15, 11], 0, 2),
    ([17, 23, 26, 20], 3, 2),
])
def test_completed_episodes_are_streamed_as_jsonl(
    tmp_path: Path, actions: list[int], left_right_flips: int, load_flips: int,
) -> None:
    tracker = FlightQualityTracker()
    tracker.add(_state(0.0, [0.0, 10_000.0, 0.0], [300.0, 0.0, 0.0]))
    for index, action in enumerate(actions, start=1):
        tracker.add(_state(index * .1, [30.0 * index, 10_000.0, 0.0], [300.0, 0.0, 0.0]),
                    policy_action=action, executed_action=action)
    episode = tracker.finish(episode=7, survived=True)
    for name, expected in (("left_right_flip_count", left_right_flips),
                           ("positive_negative_load_flip_count", load_flips)):
        assert type(episode["metrics"][name]) is int
        assert episode["metrics"][name] == expected
    path = tmp_path / "flight_quality_episodes.jsonl"
    append_flight_quality_episode(episode, path)
    append_flight_quality_episode({**episode, "episode": 8}, path)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows == [episode, {**episode, "episode": 8}]


def test_offline_replot_streams_jsonl_selects_worst_and_clears_only_its_manifest(
    tmp_path: Path,
) -> None:
    source = tmp_path / "flight_quality_episodes.jsonl"
    tracker = FlightQualityTracker()
    for index in range(11):
        tracker.add(_state(index * .1, [30.0 * index, 10_000.0, 0.0],
                           [300.0, 0.0, 0.0]), policy_action=0, executed_action=0)
    template = tracker.finish(episode=1, survived=True)
    for episode_id, score in ((1, 40.0), (2, 10.0), (3, 30.0), (4, 20.0)):
        row = json.loads(json.dumps(template))
        row["episode"] = episode_id
        row["metrics"]["flight_quality_score"] = score
        append_flight_quality_episode(row, source)

    output = tmp_path / "plots"
    result = replot_flight_quality_episodes(source, output, plot_limit=2)

    assert result["records_scanned"] == 4
    assert result["selected_episodes"] == [2, 4]
    assert result["selected_scores"] == [10.0, 20.0]
    assert (output / "episode_000002.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert (output / "episode_000004.png").is_file()

    unrelated = output / "episode_999999.png"
    unrelated.write_bytes(b"unrelated")
    refreshed = replot_flight_quality_episodes(
        source, output, plot_limit=1, clear_existing=True
    )
    assert refreshed["selected_episodes"] == [2]
    assert (output / "episode_000002.png").is_file()
    assert not (output / "episode_000004.png").exists()
    assert unrelated.read_bytes() == b"unrelated"
