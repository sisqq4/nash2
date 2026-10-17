from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

import red_swarm_policy.analyze_blue_evaluation_dynamics as dynamics_module
from red_swarm_policy.analyze_blue_evaluation_dynamics import (
    EPISODE_CSV,
    LOAD_PNG,
    MANIFEST_JSON,
    PROFILE_CSV,
    SPEED_PNG,
    SUMMARY_CSV,
    SUMMARY_JSON,
    analyze_blue_evaluation_dynamics,
    main,
    safe_analyze_blue_evaluation_dynamics,
    stream_blue_evaluation_dynamics,
)


def _episode(episode: int, missile_count: int, times: list[float], speeds: list[float],
             estimated: list[float], commanded: list[float | None], *,
             survived: bool) -> dict[str, object]:
    assert len(times) == len(speeds) == len(estimated) == len(commanded)
    intervals = [0.0, *(right - left for left, right in zip(times, times[1:]))]
    count = len(times)
    return {
        "episode": episode,
        "blue_survived": survived,
        "metadata": {
            "missile_count": missile_count,
            "termination_reason": "red_failure" if survived else "success",
            "simulation_time_s": times[-1] - times[0],
        },
        "trace": {
            "time_s": times,
            "sample_interval_s": intervals,
            "position_m": [[speed * (time - times[0]), 10000.0, 0.0]
                           for time, speed in zip(times, speeds)],
            "velocity_mps": [[speed, 0.0, 0.0] for speed in speeds],
            "speed_mps": speeds,
            "horizontal_speed_mps": speeds,
            "vertical_speed_mps": [0.0] * count,
            "estimated_actual_load_g": estimated,
            "commanded_load_g": commanded,
            "policy_action": [0] * count,
            "executed_action": [0] * count,
            "safety_intervened": [False] * count,
            "safety_reasons": [[] for _ in times],
        },
    }


def _evaluation(tmp_path: Path) -> Path:
    root = tmp_path / "evaluation"
    quality = root / "flight_quality"
    quality.mkdir(parents=True)
    records = [
        _episode(1, 1, [10.0, 11.0, 12.0], [100.0, 200.0, 300.0],
                 [1.0, 4.0, 7.0], [None, 3.0, 6.0], survived=False),
        _episode(2, 2, [5.0, 6.0], [200.0, 400.0],
                 [2.0, 8.0], [2.0, 8.0], survived=True),
        _episode(3, 3, [20.0, 21.0, 22.0, 23.0], [300.0, 300.0, 500.0, 500.0],
                 [3.0, 6.0, 9.0, 12.0], [1.0, 4.0, 7.0, 10.0], survived=True),
    ]
    (quality / "flight_quality_episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    return root


def test_streaming_groups_unequal_episodes_and_does_not_pad_time_profiles(
    tmp_path: Path,
) -> None:
    root = _evaluation(tmp_path)
    rows, groups, profiles, source = stream_blue_evaluation_dynamics(root, time_bin_s=1.0)

    assert source == (root / "flight_quality" / "flight_quality_episodes.jsonl").resolve()
    assert [row["missile_count"] for row in rows] == [1, 2, 3]
    assert [group.missile_count for group in groups] == [None, 1, 2, 3]
    assert rows[0]["speed_mean_mps"] == pytest.approx(250.0)
    assert rows[0]["estimated_load_mean_g"] == pytest.approx(5.5)
    assert rows[0]["commanded_load_mean_g"] == pytest.approx(4.5)
    assert rows[0]["commanded_load_rms_g"] == pytest.approx(math.sqrt(22.5))
    assert rows[2]["commanded_load_above_3g_s"] == pytest.approx(3.0)
    assert rows[2]["commanded_load_above_6g_s"] == pytest.approx(2.0)
    assert rows[2]["commanded_load_above_9g_s"] == pytest.approx(1.0)

    overall = [row for row in profiles if row["group"] == "全部来弹数"]
    assert [row["time_bin_index"] for row in overall] == [0, 1, 2, 3]
    assert [row["speed_episode_count"] for row in overall] == [3, 3, 2, 1]
    assert overall[0]["speed_mean_mps"] == pytest.approx(200.0)
    assert overall[0]["speed_p05_mps"] == pytest.approx(110.0)
    assert overall[0]["speed_p50_mps"] == pytest.approx(200.0)
    assert overall[0]["speed_p95_mps"] == pytest.approx(290.0)
    assert overall[2]["speed_mean_mps"] == pytest.approx(400.0)
    assert overall[3]["speed_mean_mps"] == pytest.approx(500.0)
    # Episode 2 ended at relative time 1 s.  It is not carried into bins 2 or 3.
    assert overall[2]["estimated_load_episode_count"] == 2
    assert overall[3]["estimated_load_episode_count"] == 1


def test_analysis_writes_all_and_per_missile_count_statistics_and_plots(
    tmp_path: Path,
) -> None:
    root = _evaluation(tmp_path)
    output = tmp_path / "dynamics"
    manifest = analyze_blue_evaluation_dynamics(
        root, output, time_bin_s=1.0, dpi=72, make_plots=True
    )

    assert manifest["episode_count"] == 3
    assert manifest["missile_counts"] == [1, 2, 3]
    assert manifest["episodes_by_missile_count"] == {"1": 1, "2": 1, "3": 1}
    for name in (EPISODE_CSV, SUMMARY_CSV, SUMMARY_JSON, PROFILE_CSV,
                 SPEED_PNG, LOAD_PNG, MANIFEST_JSON):
        assert (output / name).is_file()
        assert (output / name).stat().st_size > 0

    document = json.loads((output / SUMMARY_JSON).read_text(encoding="utf-8"))
    overall = document["overall"]
    assert overall["episode_count"] == 3
    assert overall["survival_rate"] == pytest.approx(2 / 3)
    assert overall["speed_valid_duration_s"] == pytest.approx(6.0)
    assert overall["speed_time_weighted_mean_mps"] == pytest.approx(2200.0 / 6.0)
    assert overall["speed_time_weighted_rms_mps"] == pytest.approx(
        math.sqrt(880000.0 / 6.0)
    )
    assert overall["speed_sample_p50_mps"] == pytest.approx(300.0)
    assert overall["speed_sample_p95_mps"] == pytest.approx(500.0)
    assert overall["speed_global_peak_mps"] == pytest.approx(500.0)
    assert overall["episode_speed_peak_mean_mps"] == pytest.approx(400.0)
    assert overall["episode_speed_peak_p95_mps"] == pytest.approx(490.0)
    assert overall["estimated_load_above_3g_s"] == pytest.approx(6.0)
    assert overall["estimated_load_above_6g_s"] == pytest.approx(4.0)
    assert overall["estimated_load_above_9g_s"] == pytest.approx(1.0)
    assert overall["commanded_load_above_3g_s"] == pytest.approx(5.0)
    assert overall["commanded_load_above_6g_s"] == pytest.approx(3.0)
    assert overall["commanded_load_above_9g_s"] == pytest.approx(1.0)
    assert set(document["by_missile_count"]) == {"1", "2", "3"}
    assert document["by_missile_count"]["2"]["speed_global_peak_mps"] == 400.0

    with (output / SUMMARY_CSV).open(encoding="utf-8-sig", newline="") as stream:
        summary_rows = list(csv.DictReader(stream))
    assert [row["group"] for row in summary_rows] == [
        "全部来弹数", "1 枚来弹", "2 枚来弹", "3 枚来弹"
    ]
    with (output / PROFILE_CSV).open(encoding="utf-8-sig", newline="") as stream:
        profile_rows = list(csv.DictReader(stream))
    final_overall = next(row for row in profile_rows
                         if row["group"] == "全部来弹数" and row["time_bin_index"] == "3")
    assert final_overall["speed_episode_count"] == "1"


def test_cli_and_input_validation(tmp_path: Path) -> None:
    root = _evaluation(tmp_path)
    output = tmp_path / "cli-output"
    output.mkdir()
    for name in (SPEED_PNG, LOAD_PNG):
        (output / name).write_bytes(b"stale")
    assert main(["--evaluation-dir", str(root), "--output", str(output),
                 "--time-bin-s", "2", "--no-plots"]) == 0
    assert (output / SUMMARY_JSON).is_file()
    assert not (output / SPEED_PNG).exists()
    with pytest.raises(ValueError, match="positive finite"):
        stream_blue_evaluation_dynamics(root, time_bin_s=0.0)

    _, _, tiny_profiles, _ = stream_blue_evaluation_dynamics(root, time_bin_s=1.0e-13)
    tiny_overall = [row for row in tiny_profiles if row["group"] == "全部来弹数"]
    assert tiny_overall[0]["time_bin_index"] == 0

    broken = tmp_path / "broken"
    (broken / "flight_quality").mkdir(parents=True)
    record = _episode(1, 1, [0.0, 1.0], [100.0, 100.0],
                      [1.0, 1.0], [1.0, 1.0], survived=True)
    record.pop("metadata")
    (broken / "flight_quality" / "flight_quality_episodes.jsonl").write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="metadata"):
        stream_blue_evaluation_dynamics(broken)


def test_safe_analysis_reports_optional_postprocessing_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("模拟绘图失败")

    monkeypatch.setattr(dynamics_module, "analyze_blue_evaluation_dynamics", fail)

    assert safe_analyze_blue_evaluation_dynamics(tmp_path) is None
    assert "蓝方测试速度/过载分析失败" in capsys.readouterr().err
