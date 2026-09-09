from __future__ import annotations

import builtins
import copy
import json
from pathlib import Path

import pytest

from red_swarm_policy import analyze_training as analysis


def _blue_training() -> dict[str, object]:
    """Representative existing Rainbow artifacts, including replay warm-up."""
    return {
        "iterations": [
            {"event": "iteration", "iteration": 1, "completed_episodes": 2,
             "reward_mean": -2.0, "loss_mean": None, "survival_rate": 0.5,
             "learning_rate": 0.0001, "gradient_norm_mean": None,
             "rollout_diagnostics": {"transitions_per_s": 10.0}},
            {"event": "iteration", "iteration": 2, "completed_episodes": 3,
             "reward_mean": 4.0, "loss_mean": 1.25, "survival_rate": 1.0,
             "learning_rate": 0.0001, "gradient_norm_mean": 0.4,
             "rollout_diagnostics": {"transitions_per_s": 12.0}},
        ],
        # Completion order can differ from episode order with parallel workers.
        "episodes": [
            {"episode": 3, "reward": 4.0, "blue_survived": True,
             "mean_loss": 1.25, "missile_count": 2, "miss_distance_m": 40.0,
             "simulation_time_s": 10.0, "termination_reason": "escaped"},
            {"episode": 1, "reward": -3.0, "blue_survived": False,
             "mean_loss": None, "missile_count": 2, "miss_distance_m": 2.0,
             "simulation_time_s": 3.0, "termination_reason": "hit"},
            {"episode": 2, "reward": -1.0, "blue_survived": True,
             "mean_loss": None, "missile_count": 3, "miss_distance_m": 20.0,
             "simulation_time_s": 8.0, "termination_reason": "escaped"},
        ],
        "curriculum_evaluations": [],
    }


def _assert_core_artifacts(output: Path) -> dict[str, object]:
    for filename in ("reward.png", "loss.png", "blue_escape_rate.png"):
        image = (output / filename).read_bytes()
        assert image.startswith(b"\x89PNG\r\n\x1a\n"), filename
        assert len(image) > 1000, filename
    result = json.loads((output / "analysis_summary.json").read_text(encoding="utf-8"))
    assert isinstance(result, dict)
    return result


def test_json_file_generates_plots_without_rewriting_training_artifacts(tmp_path: Path) -> None:
    source = tmp_path / "training_metrics.json"
    original = ("\n" + json.dumps(_blue_training(), ensure_ascii=False, indent=3) + "\n").encode()
    source.write_bytes(original)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"existing checkpoint sentinel")
    output = tmp_path / "analysis"

    report = analysis.analyze_training(source, output, smoothing_window=2, dpi=50)

    assert isinstance(report, dict)
    _assert_core_artifacts(output)
    assert (report["iteration_count"], report["episode_count"]) == (2, 3)
    loss = next(item for item in report["charts"]["loss"]["series"] if item["metric"] == "loss_mean")
    assert (loss["count"], loss["missing_count"]) == (1, 1)
    assert loss["mean"] == pytest.approx(1.25)
    outcome = next(item for item in report["charts"]["blue_escape_rate"]["series"]
                   if item["metric"] == "blue_survived")
    assert outcome["mean"] == pytest.approx(2 / 3)
    assert source.read_bytes() == original
    assert checkpoint.read_bytes() == b"existing checkpoint sentinel"


def test_mapping_input_is_not_mutated_by_sorting_or_derived_rates(tmp_path: Path) -> None:
    source = _blue_training()
    original = copy.deepcopy(source)

    analysis.analyze_training(source, tmp_path / "analysis", smoothing_window=2, dpi=50)

    assert source == original
    _assert_core_artifacts(tmp_path / "analysis")


def test_standalone_cli_discovers_metrics_in_training_directory(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    source = run / "training_metrics.json"
    source.write_text(json.dumps(_blue_training()), encoding="utf-8")
    original = source.read_bytes()
    output = tmp_path / "analysis"

    assert analysis.main([str(run), "--output", str(output),
                          "--smoothing-window", "2", "--dpi", "50"]) == 0

    _assert_core_artifacts(output)
    assert source.read_bytes() == original


def test_episode_array_artifact_is_supported(tmp_path: Path) -> None:
    source = tmp_path / "episodes.json"
    source.write_text(json.dumps(_blue_training()["episodes"]), encoding="utf-8")

    analysis.analyze_training(source, tmp_path / "analysis", dpi=50)

    _assert_core_artifacts(tmp_path / "analysis")


def test_missing_metrics_still_generate_required_plots_with_notice(tmp_path: Path) -> None:
    report = analysis.analyze_training(
        {"iterations": [{"iteration": 1, "loss_mean": None}], "episodes": []},
        tmp_path / "analysis", dpi=50,
    )

    _assert_core_artifacts(tmp_path / "analysis")
    assert report["charts"]["loss"]["series"] == []
    assert report["charts"]["blue_escape_rate"]["series"] == []
    assert any("loss" in note and "no finite" in note for note in report["notes"])


def test_jsonl_accepts_unfinished_tail_without_rewriting_source(tmp_path: Path) -> None:
    source = tmp_path / "training.jsonl"
    events = [{"event": "experiment_config", "episodes": 3}, *_blue_training()["iterations"]]
    original = ("\n".join(json.dumps(row) for row in events) + '\n{"event": "iteration",').encode()
    source.write_bytes(original)

    report = analysis.analyze_training(source, tmp_path / "analysis", dpi=50)

    assert isinstance(report, dict)
    assert report["iteration_count"] == 2
    assert any("incomplete final JSONL" in note for note in report["notes"])
    _assert_core_artifacts(tmp_path / "analysis")
    assert source.read_bytes() == original


def test_jsonl_rejects_corrupt_middle_record(tmp_path: Path) -> None:
    source = tmp_path / "training.jsonl"
    row = json.dumps(_blue_training()["iterations"][0])
    source.write_text(row + '\n{"broken":\n' + row + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        analysis.analyze_training(source, tmp_path / "analysis", dpi=50)


@pytest.mark.parametrize("options", [
    {"smoothing_window": 0}, {"smoothing_window": -2}, {"dpi": 0}, {"dpi": -10},
    {"smoothing_window": 1.5}, {"smoothing_window": True}, {"dpi": 40.5}, {"dpi": False},
])
def test_invalid_plot_parameters_fail_before_creating_output(tmp_path: Path, options: dict[str, object]) -> None:
    output = tmp_path / "analysis"

    with pytest.raises(ValueError):
        analysis.analyze_training(_blue_training(), output, **options)

    assert not output.exists()


def test_episode_escape_curves_keep_missing_outcomes_and_sort_parallel_completions() -> None:
    source = {"episodes": [
        {"episode": 4, "blue_survived": True},
        {"episode": 1, "blue_survived": False},
        {"episode": 3},
        {"episode": 2, "blue_survived": True},
        {"episode": 5, "blue_survived": 0.5},
    ]}
    data, _ = analysis.load_training_data(source)

    curve = analysis.build_charts(data)["blue_escape_rate"][0]

    assert curve.x == [1, 2, 3, 4, 5]
    assert curve.y == [0.0, 1.0, None, 1.0, None]
    assert curve.percentage
    assert analysis.rolling_mean(curve.y, 2) == [0.0, 0.5, None, 1.0, None]


def test_low_and_high_level_losses_are_separate_and_none_or_nonfinite_are_gaps() -> None:
    data, _ = analysis.load_training_data({"iterations": [
        {"iteration": 1, "assignment_actor_loss": None, "execution_actor_loss": 2.0,
         "assignment_critic_loss": float("nan"), "execution_critic_loss": float("inf")},
        {"iteration": 2, "assignment_actor_loss": -0.25, "execution_actor_loss": None,
         "assignment_critic_loss": 3.0, "execution_critic_loss": 4.0},
    ]})

    curves = {curve.name: curve.y for curve in analysis.build_charts(data)["loss"]}

    assert curves == {
        "assignment_actor_loss": [None, -0.25],
        "execution_actor_loss": [2.0, None],
        "assignment_critic_loss": [None, 3.0],
        "execution_critic_loss": [None, 4.0],
    }


def test_red_side_survival_uses_damage_fraction_and_labels_unfinished_rollouts() -> None:
    data, _ = analysis.load_training_data({"iterations": [
        {"iteration": 1, "fixed_validation": {"average_damage_rate": 0.6, "full_success_rate": 0.1},
         "rollout_diagnostics": {"average_damage_rate": 0.25}},
    ]})

    curves = {curve.name: curve for curve in analysis.build_charts(data)["blue_escape_rate"]}

    assert curves["fixed_validation.average_damage_rate"].y == pytest.approx([0.4])
    snapshot = curves["rollout_diagnostics.average_damage_rate"]
    assert snapshot.y == pytest.approx([0.75])
    assert "snapshot" in snapshot.label.lower()
    assert not any("full_success_rate" in key for key in curves)


def test_curriculum_evaluation_series_keep_scenarios_separate_and_missing_values() -> None:
    data, _ = analysis.load_training_data({
        "iterations": [{"iteration": 1}],
        "curriculum_evaluations": [
            {"completed_episodes": 100, "survival_rates": {"1": 0.9}},
            {"completed_episodes": 200, "survival_rates": {"1": 0.8, "2": 0.5}},
        ],
    })

    curves = {curve.name: curve for curve in analysis.build_charts(data)["validation_escape_rate"]}

    assert curves["survival_rates.1"].x == [100, 200]
    assert curves["survival_rates.1"].y == [0.9, 0.8]
    assert curves["survival_rates.2"].y == [None, 0.5]


def test_analysis_refuses_output_in_input_directory(tmp_path: Path) -> None:
    source = tmp_path / "training_metrics.json"
    source.write_text(json.dumps(_blue_training()), encoding="utf-8")
    original = source.read_bytes()

    with pytest.raises(ValueError, match="separate"):
        analysis.analyze_training(source, tmp_path)

    assert source.read_bytes() == original


def test_safe_training_hook_reports_missing_matplotlib_without_raising(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    original_import = builtins.__import__

    def import_without_matplotlib(name: str, *args: object, **kwargs: object) -> object:
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise ModuleNotFoundError("No module named 'matplotlib'")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_matplotlib)

    result = analysis.write_training_analysis_safely(_blue_training(), tmp_path / "analysis")

    assert result is None
    assert "matplotlib" in capsys.readouterr().err


def test_safe_training_hook_does_not_turn_missing_artifact_into_training_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    result = analysis.write_training_analysis_safely(tmp_path / "missing.json", tmp_path / "analysis")

    assert result is None
    assert capsys.readouterr().err
