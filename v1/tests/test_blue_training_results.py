from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from red_swarm_policy.analyze_blue_training_results import (
    UNKNOWN_ORIENTATION, analyze_blue_training_results, main,
    record_episode_orientation, write_blue_training_results_safely,
)


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "training_metrics.json"
    episodes = [{"episode": episode, "missile_count": count, "blue_survived": escaped,
                 "reward": float(episode), "miss_distance_m": distance,
                 "flight_quality": {"metrics": {"flight_quality_score": score},
                                    "verdicts": {"near_vertical": escaped}, "events": []}}
                for episode, count, escaped, distance, score in [
                    (2, 2, True, 10000., 90.), (1, 1, False, .99, 30.), (3, 1, True, 1., 100.)]]
    source.write_text(json.dumps({"episodes": episodes, "iterations": []}, indent=3), encoding="utf-8")
    return source


def test_reset_metadata_is_keyed_by_episode_not_worker_or_completion_order() -> None:
    metadata = {}
    info = {"initialization": {"blue_orientation": "positive_90_deg", "other": [1, 2]}}
    original = copy.deepcopy(info)
    record_episode_orientation(metadata, 20, info)
    record_episode_orientation(metadata, 3, {"initialization": {"blue_orientation": "away_from_missile_swarm"}})
    record_episode_orientation(metadata, 21, {})
    assert metadata == {20: "positive_90_deg", 3: "away_from_missile_swarm"}
    assert info == original


def test_training_result_figures_use_bound_reset_metadata_and_preserve_raw_logs(tmp_path: Path) -> None:
    source = _source(tmp_path)
    original = source.read_bytes()
    orientations = {1: "toward_missile_swarm", 2: "away_from_missile_swarm", 3: "positive_90_deg"}
    # Disabling plots still saves orientation metadata for independent later plotting.
    write_blue_training_results_safely(source, episode_orientations=orientations, make_plots=False)
    assert not (tmp_path / "training_analysis").exists()
    report = analyze_blue_training_results(source, dpi=50)
    output = tmp_path / "training_analysis" / "results"
    assert report["data_stage"] == "training"
    assert report["orientation_missing_episodes"] == 0
    by_orientation = {row["group"]["blue_orientation"]: row for row in report["groups"]
                      if row["level"] == "by_blue_orientation"}
    assert by_orientation["toward_missile_swarm"]["escape_rate"] == 0.
    assert by_orientation["away_from_missile_swarm"]["miss_distance_histogram_1m"]["bins"][0]["lower_m"] == 10000
    assert by_orientation["positive_90_deg"]["flight_quality"]["metrics"]["flight_quality_score"]["mean"] == 100.
    for filename in ("escape_rate_by_missile_count_and_orientation.png", "miss_distance_distribution.png",
                     "flight_quality_score_distribution.png", "flight_quality_failure_rate_overall.png"):
        assert (output / filename).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert source.read_bytes() == original
    assert all("blue_orientation" not in row for row in json.loads(source.read_text())["episodes"])


def test_stale_orientation_metadata_is_rejected_without_touching_new_run(tmp_path: Path) -> None:
    source = _source(tmp_path)
    write_blue_training_results_safely(source, episode_orientations={1: "positive_90_deg"}, make_plots=False)
    source.write_text(source.read_text() + "\n", encoding="utf-8")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="does not match"):
        analyze_blue_training_results(source)
    assert source.read_bytes() == original
    assert not (tmp_path / "training_analysis").exists()


def test_old_training_without_orientation_is_explicitly_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from red_swarm_policy import evaluation_plots

    source = _source(tmp_path)
    captured = []
    write_plots = evaluation_plots.write_evaluation_plots

    def summarize_only(output, inputs, **kwargs):
        captured.extend(inputs[0][2])
        return write_plots(output, inputs, make_plots=False)

    monkeypatch.setattr(evaluation_plots, "write_evaluation_plots", summarize_only)
    output = tmp_path / "old_analysis"
    assert main([str(source), "--output", str(output)]) == 0
    report = json.loads((output / "plot_statistics.json").read_text())
    assert report["orientation_missing_episodes"] == 3
    assert {row["blue_orientation"] for row in captured} == {UNKNOWN_ORIENTATION}
    assert len([row for row in report["groups"] if row["level"] == "by_missile_count"]) == 2


def test_grouped_plot_failure_does_not_break_saved_training(tmp_path: Path, monkeypatch, capsys) -> None:
    from red_swarm_policy import evaluation_plots

    source = _source(tmp_path)
    original = source.read_bytes()

    def fail(*args, **kwargs):
        raise RuntimeError("plot backend failed")

    monkeypatch.setattr(evaluation_plots, "write_evaluation_plots", fail)
    assert write_blue_training_results_safely(source, episode_orientations={1: "positive_90_deg"}) is None
    assert source.read_bytes() == original
    assert (tmp_path / "training_metrics_episode_metadata.json").exists()
    assert "plot backend failed" in capsys.readouterr().err
