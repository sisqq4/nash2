from __future__ import annotations

import builtins
import copy
import csv
import json
from pathlib import Path

import pytest

from red_swarm_policy import analyze_blue_evaluations as analysis
from red_swarm_policy.evaluation_plots import build_plot_statistics, one_meter_histogram, write_evaluation_plots


def _rows() -> list[dict]:
    return [
        {"episode": 1, "missile_count": 1, "blue_orientation": "toward_missile_swarm",
         "blue_survived": False, "miss_distance_m": .999,
         "flight_quality": {"metrics": {"flight_quality_score": 20., "action_switch_rate_hz": 3.},
                            "verdicts": {"near_vertical": False, "safety_reliance": True},
                            "events": [{"type": "near_vertical"}, {"type": "near_vertical"}]}},
        {"episode": 2, "missile_count": 1, "blue_orientation": "toward_missile_swarm",
         "blue_survived": True, "miss_distance_m": 1.,
         "flight_quality": {"metrics": {"flight_quality_score": 80., "action_switch_rate_hz": 1.},
                            "verdicts": {"near_vertical": True}, "events": []}},
        {"episode": 3, "missile_count": 2, "blue_orientation": "away_from_missile_swarm",
         "blue_survived": True, "miss_distance_m": 20000.25},
    ]


def _overall(report: dict, policy: str = "blue") -> dict:
    return next(row for row in report["groups"] if row["policy"] == policy and row["level"] == "overall")


def test_one_meter_bins_include_exact_boundaries_and_sparse_long_tail() -> None:
    histogram = one_meter_histogram([0., .999999, 1., 1.999999, 2., 20000.3])
    assert histogram["sample_count"] == 6
    assert [(row["lower_m"], row["upper_m"], row["count"]) for row in histogram["bins"]] == [
        (0, 1, 2), (1, 2, 2), (2, 3, 1), (20000, 20001, 1)]
    assert sum(row["probability"] for row in histogram["bins"]) == pytest.approx(1.)
    assert histogram["unlisted_bins_have_zero_count"]


@pytest.mark.parametrize("value", [-.001, float("nan"), float("inf"), True])
def test_one_meter_bins_reject_invalid_distances(value: float) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        one_meter_histogram([value])


def test_grouped_rates_and_quality_exclude_missing_observations(tmp_path: Path) -> None:
    report = build_plot_statistics([("blue", tmp_path / "evaluation.json", _rows())])
    overall = _overall(report)
    assert overall["escape_rate"] == pytest.approx(2 / 3)
    assert overall["miss_distance_histogram_1m"]["sample_count"] == 3
    assert {row["level"] for row in report["groups"]} == {
        "overall", "by_missile_count", "by_blue_orientation", "by_missile_count_and_orientation"}
    joint = [row for row in report["groups"] if row["level"] == "by_missile_count_and_orientation"]
    assert len(joint) == 2  # Unobserved cross combinations must not be zero-filled samples.
    assert {row["episodes"] for row in joint} == {1, 2}
    quality = overall["flight_quality"]
    assert quality["available_episodes"] == 2 and quality["missing_episodes"] == 1
    assert quality["metrics"]["flight_quality_score"]["mean"] == 50.
    assert quality["metrics"]["flight_quality_score"]["missing_count"] == 1
    assert quality["verdicts"]["near_vertical"]["failure_rate"] == .5
    assert quality["verdicts"]["near_vertical"]["evaluated_episodes"] == 2
    assert quality["verdicts"]["safety_reliance"]["failure_rate"] == 0.
    assert quality["verdicts"]["safety_reliance"]["evaluated_episodes"] == 1
    assert quality["events"]["near_vertical"]["event_count"] == 2
    assert quality["events"]["near_vertical"]["affected_episode_rate"] == .5


def test_multiple_policies_do_not_share_histogram_denominators(tmp_path: Path) -> None:
    report = build_plot_statistics([("one", tmp_path / "one.json", _rows()[:1]),
                                    ("many", tmp_path / "many.json", _rows())])
    assert _overall(report, "one")["miss_distance_histogram_1m"]["bins"][0]["probability"] == 1.
    assert _overall(report, "many")["miss_distance_histogram_1m"]["bins"][0]["probability"] == pytest.approx(1 / 3)


def test_figures_and_exact_tables_preserve_inputs_and_backend(tmp_path: Path) -> None:
    import matplotlib
    import numpy as np
    import random

    rows = _rows()
    original_rows = copy.deepcopy(rows)
    source = tmp_path / "evaluation.json"
    encoded = json.dumps({"results": rows}, indent=3).encode()
    source.write_bytes(encoded)
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"existing checkpoint")
    original_backend = matplotlib.get_backend()
    py_state = random.getstate()
    np_state = np.random.get_state()

    report = write_evaluation_plots(tmp_path / "analysis", [("blue", source, rows)], dpi=50)

    expected = {"escape_rate_by_missile_count.png", "escape_rate_by_blue_orientation.png",
                "escape_rate_by_missile_count_and_orientation.png", "miss_distance_distribution.png",
                "miss_distance_distribution_by_missile_count_and_orientation.png",
                "miss_distance_distribution_0_50m.png", "flight_quality_score_distribution.png",
                "flight_quality_score_by_missile_count.png", "flight_quality_score_by_blue_orientation.png",
                "flight_quality_failure_rate_overall.png", "flight_quality_event_rates.png"}
    assert expected <= set(report["figures"])
    for name in report["figures"]:
        assert (tmp_path / "analysis" / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    with (tmp_path / "analysis" / "miss_distance_histogram_1m.csv").open() as stream:
        bins = [row for row in csv.DictReader(stream) if row["level"] == "overall"]
    assert sum(int(row["count"]) for row in bins) == 3
    assert sum(float(row["probability"]) for row in bins) == pytest.approx(1.)
    assert any(row["lower_m"] == "20000" for row in bins)
    assert source.read_bytes() == encoded
    assert checkpoint.read_bytes() == b"existing checkpoint"
    assert rows == original_rows
    assert matplotlib.get_backend() == original_backend
    assert random.getstate() == py_state
    current = np.random.get_state()
    assert current[0] == np_state[0] and np.array_equal(current[1], np_state[1]) and current[2:] == np_state[2:]


def test_cli_no_plots_still_exports_new_histogram_and_quality_statistics(tmp_path: Path) -> None:
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps({"results": _rows()}), encoding="utf-8")
    output = tmp_path / "analysis"
    assert analysis.main([f"blue={source}", "--output", str(output), "--no-plots"]) == 0
    report = json.loads((output / "plot_statistics.json").read_text())
    assert _overall(report)["flight_quality"]["metrics"]["flight_quality_score"]["count"] == 2
    assert (output / "miss_distance_histogram_1m.csv").exists()
    assert not list(output.glob("*.png"))


def test_missing_optional_quality_is_not_reported_as_perfect_quality(tmp_path: Path) -> None:
    rows = [{k: v for k, v in row.items() if k != "flight_quality"} for row in _rows()]
    report = build_plot_statistics([("blue", tmp_path / "evaluation.json", rows)])
    quality = _overall(report)["flight_quality"]
    assert quality["available_episodes"] == 0 and quality["missing_episodes"] == 3
    assert quality["metrics"] == quality["verdicts"] == quality["events"] == {}


def test_automatic_wrapper_catches_missing_matplotlib_and_preserves_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "evaluation.json"
    original = json.dumps({"results": _rows()}).encode()
    source.write_bytes(original)
    real_import = builtins.__import__

    def fail_import(name: str, *args, **kwargs):
        if name.startswith("matplotlib"):
            raise ImportError("matplotlib unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_import)
    assert analysis.write_evaluation_analysis_safely(source) is None
    assert "matplotlib unavailable" in capsys.readouterr().err
    assert source.read_bytes() == original
    assert (tmp_path / "evaluation_analysis" / "plot_statistics.json").exists()


def test_original_output_directory_is_rejected_before_writing(tmp_path: Path) -> None:
    source = tmp_path / "episodes.csv"  # Protect even coincidentally named original artifacts.
    source.write_text("unchanged")
    with pytest.raises(ValueError, match="separate"):
        write_evaluation_plots(tmp_path, [("blue", source, _rows())])
    assert source.read_text() == "unchanged"


def test_baseline_auto_hook_runs_after_original_saves_and_can_be_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from red_swarm_policy import evaluate_blue_rule_baseline as baseline

    rows = _rows()
    monkeypatch.setattr(baseline, "evaluate", lambda **kwargs: ({"episodes": rows}, rows))
    calls = []

    def check_saved(source: Path, output: Path | None) -> None:
        assert source.is_file()
        assert source.with_name("blue_rule_baseline_trials.csv").is_file()
        progress = source.with_name("blue_rule_baseline_progress.jsonl").read_text()
        assert json.loads(progress.splitlines()[-1])["event"] == "baseline_complete"
        calls.append((source, output))

    monkeypatch.setattr(baseline, "write_evaluation_analysis_safely", check_saved)
    assert baseline.main(["--output", str(tmp_path / "enabled")]) == 0
    assert len(calls) == 1
    assert baseline.main(["--output", str(tmp_path / "disabled"), "--no-result-plots"]) == 0
    assert len(calls) == 1
