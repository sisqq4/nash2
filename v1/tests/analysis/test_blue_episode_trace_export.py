from __future__ import annotations

import csv
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from red_swarm_policy.analysis.export_blue_episode_trace import (
    ACTION_MAP_CSV_NAME,
    PLOT_FILENAMES,
    SEGMENT_CSV_NAME,
    SUMMARY_CSV_NAME,
    SUMMARY_JSON_NAME,
    TRACE_CSV_NAME,
    _action_segments,
    _interval_step_data,
    _prepare_trace,
    _title_context,
    export_episode_trace,
    find_episode,
    resolve_jsonl,
)

trace_export = importlib.import_module("red_swarm_policy.analysis.export_blue_episode_trace")


def _episode(episode: int, *, missiles: int, scenario_episode: int) -> dict:
    time = [7.0, 7.1, 7.35, 7.5]
    position = [
        [100.0, 10_000.0, -50.0],
        [130.0, 10_001.0, -48.0],
        [159.0, 10_003.0, -44.0],
        [187.0, 10_004.0, -38.0],
    ]
    velocity = [
        [300.0, 0.0, 20.0],
        [298.0, 10.0, 25.0],
        [290.0, 20.0, 40.0],
        [280.0, 10.0, 60.0],
    ]
    speed = [math.sqrt(sum(component * component for component in row)) for row in velocity]
    horizontal = [math.hypot(row[0], row[2]) for row in velocity]
    red_positions = [
        [
            [-2_000.0 + 120.0 * sample + 250.0 * slot,
             10_500.0 + 20.0 * slot,
             -1_000.0 + 80.0 * sample - 150.0 * slot]
            for slot in range(missiles)
        ]
        for sample in range(len(time))
    ]
    return {
        "episode": episode,
        "blue_survived": True,
        "metrics": {"flight_quality_score": 88.5, "max_estimated_actual_load_g": 4.0},
        "verdicts": {"near_vertical": True},
        "events": [],
        "metadata": {
            "missile_count": missiles,
            "scenario_episode": scenario_episode,
            "seed": 1234 + scenario_episode,
            "termination_reason": "success",
        },
        "trace": {
            "time_s": time,
            "step_count": [1400, 1420, 1440, 1460],
            "sample_interval_s": [0.0, 0.1, 0.25, 0.15],
            "position_m": position,
            "velocity_mps": velocity,
            "speed_mps": speed,
            "horizontal_speed_mps": horizontal,
            "vertical_speed_mps": [row[1] for row in velocity],
            "estimated_actual_load_g": [1.0, 3.0, 4.0, 1.1],
            "flight_path_angle_deg": [0.0, 2.0, 4.0, 2.0],
            "heading_rate_deg_s": [0.0, 5.0, 7.0, 3.0],
            "turn_radius_m": [10_000.0, 4_000.0, 2_500.0, 5_000.0],
            "policy_action": [None, 7, 7, 0],
            "executed_action": [None, 7, 7, 0],
            "commanded_load_g": [None, 8.5, 9.0, 1.0],
            "safety_intervened": [False, True, False, False],
            "safety_reasons": [[], ["高度保护"], [], []],
            "min_altitude_m": 8_000.0,
            "max_altitude_m": 12_000.0,
            "red_positions_m": red_positions,
            "red_ids": list(range(missiles)),
            "red_position_valid": [[True] * missiles for _ in time],
            "red_state_valid": [[True] * missiles for _ in time],
        },
    }


def _write_source(path: Path) -> None:
    records = [
        _episode(1, missiles=1, scenario_episode=1),
        _episode(42, missiles=2, scenario_episode=3),
    ]
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def test_export_writes_complete_chinese_tables_mapping_summary_and_separate_plots(
    tmp_path: Path,
) -> None:
    source = tmp_path / "flight_quality_episodes.jsonl"
    _write_source(source)
    output = tmp_path / "episode"

    result = export_episode_trace(source, output, episode=42, dpi=60)

    assert result["episode"] == 42
    assert result["samples"] == 4
    assert result["action_segments"] == 3
    assert result["red_tracks"] == 2
    assert result["show_safety_interventions"] is False
    expected = {
        TRACE_CSV_NAME, SEGMENT_CSV_NAME, ACTION_MAP_CSV_NAME,
        SUMMARY_JSON_NAME, SUMMARY_CSV_NAME, *PLOT_FILENAMES,
    }
    assert expected == {path.name for path in output.iterdir()}
    for name in PLOT_FILENAMES:
        assert (output / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")

    for name in (TRACE_CSV_NAME, SEGMENT_CSV_NAME, ACTION_MAP_CSV_NAME, SUMMARY_CSV_NAME):
        assert (output / name).read_bytes().startswith(b"\xef\xbb\xbf")

    with (output / TRACE_CSV_NAME).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 4
    assert rows[0]["绝对时间_s"] == "7.0"
    assert rows[0]["相对时间_s"] == "0.0"
    assert rows[1]["策略动作序号"] == rows[1]["执行动作序号"] == "7"
    assert float(rows[1]["估算实际过载_g"]) == 3.0
    assert float(rows[1]["名义执行动作过载_g"]) == 8.5
    assert float(rows[1]["轨迹记录名义执行动作过载_g"]) == 8.5
    assert float(rows[1]["当前动作表重建名义执行动作过载_g"]) == 9.0
    assert float(rows[1]["执行动作法向过载_g"]) == 9.0
    assert rows[1]["安全约束介入"] == "是"
    assert rows[1]["安全约束原因"] == "高度保护"

    with (output / ACTION_MAP_CSV_NAME).open(encoding="utf-8-sig", newline="") as stream:
        action_rows = list(csv.DictReader(stream))
    assert len(action_rows) == 29
    action_seven = action_rows[7]
    assert action_seven["动作序号"] == "7"
    assert float(action_seven["轴向过载_g"]) == 0.0
    assert float(action_seven["法向过载_g"]) == 9.0
    assert float(action_seven["名义合成过载_g"]) == 9.0
    assert float(action_seven["滚转角_rad"]) == pytest.approx(math.acos(1 / 9))

    with (output / SEGMENT_CSV_NAME).open(encoding="utf-8-sig", newline="") as stream:
        segments = list(csv.DictReader(stream))
    assert len(segments) == 3
    assert segments[1]["策略动作序号"] == segments[1]["执行动作序号"] == "7"
    assert int(segments[1]["样本数"]) == 2
    assert int(segments[1]["安全约束介入次数"]) == 1
    assert float(segments[1]["开始相对时间_s"]) == pytest.approx(0.0)
    assert float(segments[1]["结束相对时间_s"]) == pytest.approx(0.35)
    assert float(segments[1]["持续时间_s"]) == pytest.approx(0.35)

    summary = json.loads((output / SUMMARY_JSON_NAME).read_text(encoding="utf-8"))
    assert summary["全局回合序号"] == 42
    assert summary["红弹数量"] == 2
    assert summary["场景内回合序号"] == 3
    assert summary["坐标约定"]["数组顺序"] == ["x北向", "y向上（高度）", "z东向"]
    assert "不是每个物理积分步" in summary["数据口径说明"]["估算实际过载"]
    assert "无法恢复" in summary["数据口径说明"]["名义执行动作过载"]
    validation = summary["当前动作表校验"]
    assert len(validation["当前动作表SHA256"]) == 64
    assert validation["可比较样本数"] == 3
    assert validation["不一致样本数"] == 1
    assert validation["最大绝对差_g"] == pytest.approx(0.5)


def test_stream_selector_supports_global_and_missile_scenario_identity(tmp_path: Path) -> None:
    source = tmp_path / "episodes.jsonl"
    _write_source(source)

    assert find_episode(source, episode=1)["metadata"]["missile_count"] == 1
    selected = find_episode(source, missiles=2, scenario_episode=3)
    assert selected["episode"] == 42
    with pytest.raises(LookupError, match="找不到"):
        find_episode(source, missiles=3, scenario_episode=3)
    with pytest.raises(ValueError, match="同时"):
        find_episode(source, missiles=2)


def test_evaluation_directory_resolution_and_trace_validation(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation"
    quality = evaluation / "flight_quality"
    quality.mkdir(parents=True)
    source = quality / "flight_quality_episodes.jsonl"
    _write_source(source)
    assert resolve_jsonl(evaluation_dir=evaluation) == source.resolve()
    assert resolve_jsonl(input_jsonl=source) == source.resolve()

    invalid = _episode(9, missiles=1, scenario_episode=9)
    invalid["trace"]["velocity_mps"] = invalid["trace"]["velocity_mps"][:-1]
    bad_source = tmp_path / "bad.jsonl"
    bad_source.write_text(json.dumps(invalid) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="velocity_mps"):
        export_episode_trace(bad_source, tmp_path / "bad-output", episode=9, dpi=60)


def test_interval_steps_use_preceding_nonuniform_decision_intervals() -> None:
    data = _prepare_trace(_episode(5, missiles=2, scenario_episode=4))

    edges, values = _interval_step_data(
        data, [math.nan if value is None else value for value in data["executed"]]
    )

    assert edges.tolist() == pytest.approx([0.0, 0.1, 0.35, 0.5])
    assert values.tolist() == pytest.approx([7.0, 7.0, 0.0])
    segments = _action_segments(data)
    assert segments[0]["持续时间_s"] == 0.0  # Initial state has no action interval.
    assert segments[1]["开始相对时间_s"] == pytest.approx(edges[0])
    assert segments[1]["结束相对时间_s"] == pytest.approx(edges[2])


def test_plot_contract_uses_xy_yz_zx_order_context_and_discrete_action_axis(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = _episode(42, missiles=2, scenario_episode=3)
    data = _prepare_trace(episode)
    time_calls: list[dict] = []
    projection_calls: list[dict] = []
    saved: list[dict] = []

    monkeypatch.setattr(trace_export, "_plot_time_series",
                        lambda *args, **kwargs: time_calls.append(kwargs))
    monkeypatch.setattr(trace_export, "_plot_projection",
                        lambda *args, **kwargs: projection_calls.append(kwargs))
    monkeypatch.setattr(trace_export, "_save_figure",
                        lambda *args, **kwargs: saved.append(kwargs))

    trace_export._write_plots(
        tmp_path, data, title_context=_title_context(episode), font_path=None, dpi=100,
    )

    assert len(time_calls) + len(projection_calls) + len(saved) == 9
    assert [(call["first"], call["second"]) for call in projection_calls] == [
        (0, 1), (1, 2), (2, 0),
    ]
    assert [(call["xlabel"], call["ylabel"]) for call in projection_calls] == [
        ("北向 x（米）", "高度 y（米）"),
        ("高度 y（米）", "东向 z（米）"),
        ("东向 z（米）", "北向 x（米）"),
    ]
    all_titles = [call["title"] for call in [*time_calls, *projection_calls, *saved]]
    assert all("2v1" in title and "全局回合 42" in title and "场景内回合 3" in title
               for title in all_titles)
    assert time_calls[-1]["action_axis"] is True


def test_safety_overlay_is_subtle_and_projection_marks_only_run_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    time_axis_calls: list[tuple[float, float, dict]] = []

    class TimeAxis:
        def axvspan(self, start, end, **kwargs):
            time_axis_calls.append((start, end, kwargs))

        def set_xlabel(self, _label):
            return None

        def grid(self, *_args, **_kwargs):
            return None

    time_data = {
        "relative": np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        "dt": np.asarray([0.0, 0.1, 0.1, 0.1, 0.1, 0.1]),
        "safety": np.asarray([False, True, True, False, True, False]),
    }
    trace_export._decorate_time_axis(TimeAxis(), time_data)
    assert time_axis_calls == []
    trace_export._decorate_time_axis(
        TimeAxis(), time_data, show_safety_interventions=True
    )
    np.testing.assert_allclose(
        [(start, end) for start, end, _ in time_axis_calls],
        [(0.0, 0.2), (0.3, 0.4)], rtol=0.0, atol=1.0e-12,
    )
    assert all(call[2]["alpha"] == pytest.approx(0.055) for call in time_axis_calls)
    assert 0.0 < time_axis_calls[0][2]["alpha"] < 0.12
    assert [call[2]["label"] for call in time_axis_calls] == ["安全约束介入", None]

    scatter_calls: list[tuple[np.ndarray, np.ndarray, dict]] = []

    class ProjectionAxis:
        def plot(self, *_args, **_kwargs):
            return None

        def scatter(self, x, y, **kwargs):
            scatter_calls.append((np.atleast_1d(x), np.atleast_1d(y), kwargs))

        def legend(self):
            return None

    axis = ProjectionAxis()

    def draw_without_saving(_path, **kwargs):
        kwargs["draw"](axis, object())

    monkeypatch.setattr(trace_export, "_save_figure", draw_without_saving)
    positions = np.column_stack((np.arange(6.0), np.arange(10.0, 16.0), np.arange(20.0, 26.0)))
    trace_export._plot_projection(
        tmp_path / "projection.png", {**time_data, "position": positions},
        first=0, second=1, title="投影", xlabel="x", ylabel="y",
        font_path=None, dpi=100, equal=True,
    )

    default_labels = [call[2].get("label") for call in scatter_calls]
    assert "安全介入区间起点" not in default_labels
    assert "安全介入区间终点" not in default_labels
    scatter_calls.clear()
    trace_export._plot_projection(
        tmp_path / "projection-with-safety.png", {**time_data, "position": positions},
        first=0, second=1, title="投影", xlabel="x", ylabel="y",
        font_path=None, dpi=100, equal=True, show_safety_interventions=True,
    )

    safety_starts = next(call for call in scatter_calls
                         if call[2].get("label") == "安全介入区间起点")
    safety_ends = next(call for call in scatter_calls
                       if call[2].get("label") == "安全介入区间终点")
    assert safety_starts[0].tolist() == [1.0, 4.0]
    assert safety_ends[0].tolist() == [2.0, 4.0]
    assert safety_starts[2]["marker"] == "^"
    assert safety_ends[2]["marker"] == "o"
    labels = [call[2].get("label") for call in scatter_calls]
    assert labels.count("安全介入区间起点") == 1
    assert labels.count("安全介入区间终点") == 1
    assert "安全约束介入" not in labels


def test_projection_draws_blue_and_red_tracks_and_uses_exact_terminal_point(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = _episode(11, missiles=2, scenario_episode=5)
    trace = episode["trace"]
    frozen = [9_999.0, 9_999.0, 9_999.0]
    trace["red_positions_m"][2][0] = frozen
    trace["red_positions_m"][3][0] = frozen
    trace["red_state_valid"][2][0] = False
    trace["red_state_valid"][3][0] = False
    episode["red_termination_events"] = [{
        "red_id": 0,
        "time_s": 7.2,
        "position_valid": True,
        "position_m": [-1_777.0, 10_444.0, -888.0],
    }]
    data = _prepare_trace(episode)
    assert len(data["red_tracks"]) == 2
    first_track = data["red_tracks"][0]["position"]
    assert first_track.shape == (3, 3)
    assert first_track[-1].tolist() == [-1_777.0, 10_444.0, -888.0]
    assert not np.any(np.all(first_track == np.asarray(frozen), axis=1))

    plot_calls: list[tuple[np.ndarray, np.ndarray, dict]] = []

    class Axis:
        def plot(self, x, y, **kwargs):
            plot_calls.append((np.asarray(x), np.asarray(y), kwargs))

        def scatter(self, *_args, **_kwargs):
            return None

        def legend(self):
            return None

    def draw_without_saving(_path, **kwargs):
        kwargs["draw"](Axis(), object())

    monkeypatch.setattr(trace_export, "_save_figure", draw_without_saving)
    trace_export._plot_projection(
        tmp_path / "projection.png", data, first=0, second=2,
        title="投影", xlabel="x", ylabel="z", font_path=None, dpi=100, equal=True,
    )
    assert plot_calls[0][2]["color"] == "royalblue"
    assert [call[2]["color"] for call in plot_calls[1:]] == ["firebrick", "firebrick"]
    assert plot_calls[1][0][-1] == pytest.approx(-1_777.0)
    assert plot_calls[1][1][-1] == pytest.approx(-888.0)


def test_strict_selector_safety_and_sample_interval_validation(tmp_path: Path) -> None:
    source = tmp_path / "episodes.jsonl"
    _write_source(source)
    with pytest.raises(ValueError, match="正整数"):
        find_episode(source, episode=1.0)  # type: ignore[arg-type]

    invalid_safety = _episode(8, missiles=1, scenario_episode=8)
    invalid_safety["trace"]["safety_intervened"][1] = "false"
    with pytest.raises(ValueError, match="布尔"):
        _prepare_trace(invalid_safety)

    invalid_interval = _episode(9, missiles=1, scenario_episode=9)
    invalid_interval["trace"]["sample_interval_s"][2] = 0.2
    with pytest.raises(ValueError, match="相邻 time_s"):
        _prepare_trace(invalid_interval)

    invalid_command = _episode(10, missiles=1, scenario_episode=10)
    invalid_command["trace"]["commanded_load_g"] = 9.0
    with pytest.raises(ValueError, match="commanded_load_g 必须是长度"):
        _prepare_trace(invalid_command)
