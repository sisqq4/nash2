from __future__ import annotations

import csv
from contextlib import nullcontext
import json
import math
from pathlib import Path

import pytest

from red_swarm_policy import analyze_blue_training_load as load_analysis


def _trace(episode: int, loads: list[float], *, actions: list[int | None],
           commands: list[float | None], safety: list[bool],
           sample_intervals: list[float] | None = None) -> dict:
    trace = {
        "time_s": [0.0, 1.0, 3.0],
        # Legacy training traces can omit scalar speed fields.  The analyzer
        # must recover them from [north, up, east] velocity components.
        "velocity_mps": [[3.0, 4.0, 0.0], [0.0, 0.0, 10.0], [0.0, -20.0, 0.0]],
        "estimated_actual_load_g": loads,
        "commanded_load_g": commands,
        "executed_action": actions,
        "safety_intervened": safety,
    }
    if sample_intervals is not None:
        trace["sample_interval_s"] = sample_intervals
    return {
        "episode": episode,
        "blue_survived": episode == 2,
        "trace": trace,
    }


def _run(tmp_path: Path) -> Path:
    training = tmp_path / "训练结果"
    quality = training / "flight_quality"
    quality.mkdir(parents=True)
    # Deliberately use completion order 2, 1 to verify that JSONL order is
    # preserved while metadata is joined by episode ID.
    records = [
        _trace(2, [1.0, 4.0, 10.0], actions=[None, 1, 2],
               commands=[None, 2.0, 6.0], safety=[False, True, False]),
        _trace(1, [2.0, 2.0, 2.0], actions=[None, 3, 3],
               commands=[None, 3.0, 3.0], safety=[False, False, False]),
    ]
    (quality / "flight_quality_episodes.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
    (training / "training_metrics.json").write_text(json.dumps({"episodes": [
        {"episode": 1, "missile_count": 1, "blue_survived": False,
         "termination_reason": "hit", "simulation_time_s": 3.0},
        {"episode": 2, "missile_count": 3, "blue_survived": True,
         "termination_reason": "all_missiles_lost", "simulation_time_s": 3.0},
    ]}), encoding="utf-8")
    # The second standard source is also accepted and overlays matching fields.
    (training / "episodes.json").write_text(json.dumps([
        {"episode": 1, "missile_count": 1}, {"episode": 2, "missile_count": 3},
    ]), encoding="utf-8")
    return training


def test_streaming_statistics_preserve_completion_order_and_join_metadata(tmp_path: Path) -> None:
    training = _run(tmp_path)

    rows, sources = load_analysis.stream_training_load_rows(training)

    assert [(row["completion_order"], row["episode"], row["missile_count"])
            for row in rows] == [(1, 2, 3), (2, 1, 1)]
    first = rows[0]
    # dt=[0,1,2].  The reset sample has no duration before the recorded trace.
    assert first["valid_speed_samples"] == 3
    assert first["speed_valid_duration_s"] == pytest.approx(3.0)
    assert first["speed_mean_mps"] == pytest.approx(50.0 / 3.0)
    assert first["speed_p05_mps"] == pytest.approx(5.5)
    assert first["speed_p50_mps"] == pytest.approx(10.0)
    assert first["speed_p95_mps"] == pytest.approx(19.0)
    assert first["speed_min_mps"] == pytest.approx(5.0)
    assert first["speed_max_mps"] == pytest.approx(20.0)
    assert first["speed_rms_mps"] == pytest.approx(math.sqrt(300.0))
    assert first["horizontal_speed_mean_mps"] == pytest.approx(10.0 / 3.0)
    assert first["horizontal_speed_min_mps"] == pytest.approx(0.0)
    assert first["horizontal_speed_p05_mps"] == pytest.approx(0.3)
    assert first["abs_vertical_speed_mean_mps"] == pytest.approx(40.0 / 3.0)
    assert first["abs_vertical_speed_p95_mps"] == pytest.approx(18.4)
    assert first["abs_vertical_speed_max_mps"] == pytest.approx(20.0)
    assert first["estimated_load_valid_duration_s"] == pytest.approx(3.0)
    assert first["estimated_load_mean_g"] == pytest.approx(8.0)
    assert first["estimated_load_p50_g"] == pytest.approx(4.0)
    assert first["estimated_load_p95_g"] == pytest.approx(9.4)
    assert first["estimated_load_max_g"] == pytest.approx(10.0)
    assert first["estimated_load_rms_g"] == pytest.approx(math.sqrt(72.0))
    assert first["estimated_load_above_3g_s"] == pytest.approx(3.0)
    assert first["estimated_load_above_6g_s"] == pytest.approx(2.0)
    assert first["estimated_load_above_9g_s"] == pytest.approx(2.0)
    assert first["estimated_load_above_6g_fraction"] == pytest.approx(2.0 / 3.0)
    assert first["commanded_load_valid_duration_s"] == pytest.approx(3.0)
    assert first["commanded_load_mean_g"] == pytest.approx(14.0 / 3.0)
    assert first["commanded_load_p95_g"] == pytest.approx(5.8)
    assert first["safety_intervention_count"] == 1
    assert first["safety_intervention_rate"] == pytest.approx(0.5)
    assert first["action_switch_count"] == 1
    assert first["action_switch_rate_hz"] == pytest.approx(1.0 / 3.0)
    assert len(sources["metadata_sources"]) == 2
    assert sources["missing_metadata_episode_ids"] == []


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font.*:UserWarning")
def test_analysis_writes_chinese_tables_manifest_and_eleven_separate_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = _run(tmp_path)
    output = tmp_path / "中文过载分析"
    monkeypatch.setattr(load_analysis, "chinese_plot_context",
                        lambda _font=None: nullcontext("测试字体"))

    manifest = load_analysis.analyze_blue_training_load(
        training, output, rolling_window=2, block_size=1, dpi=72,
    )

    assert manifest["episode_count"] == 2
    assert manifest["block_count"] == 2
    assert len(manifest["figures"]) == 11
    assert all((output / name).is_file() for name in manifest["figures"])
    assert len(set(manifest["figures"])) == 11
    assert {
        "speed_mean.png", "speed_min.png", "speed_max.png",
        "horizontal_speed_min.png", "abs_vertical_speed_p95.png",
    } <= set(manifest["figures"])
    with (output / "逐回合速度过载统计.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["轨迹完成序号"] == "1"
    assert rows[0]["回合编号"] == "2"
    assert rows[0]["蓝方成功规避"] == "是"
    assert float(rows[0]["总速度时间加权均值（m/s）"]) == pytest.approx(50.0 / 3.0)
    with (output / "分段速度过载统计.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        blocks = list(csv.DictReader(stream))
    assert len(blocks) == 2
    saved = json.loads(
        (output / "训练速度过载分析清单.json").read_text(encoding="utf-8")
    )
    assert saved["figures"] == manifest["figures"]
    assert saved["schema_version"] == 2
    assert saved["tables"] == ["逐回合速度过载统计.csv", "分段速度过载统计.csv"]
    assert any("velocity_mps" in note for note in saved["notes"])
    assert any("不是仿真器逐物理步直接测量" in note for note in saved["notes"])


def test_parser_and_invalid_windows() -> None:
    args = load_analysis.build_parser().parse_args([
        "--training-dir", "训练", "--output", "输出", "--rolling-window", "20",
        "--block-size", "50", "--dpi", "120", "--font", "字体.ttf",
    ])
    assert args.training_dir == Path("训练")
    assert args.output == Path("输出")
    assert args.rolling_window == 20
    assert args.block_size == 50
    with pytest.raises(ValueError, match="block_size"):
        load_analysis.summarize_blocks([], 0)


def test_public_writer_and_safe_wrappers_isolate_report_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    training = _run(tmp_path)
    rows, _sources = load_analysis.stream_training_load_rows(training)
    output = tmp_path / "direct-writer"

    def touch_plot(_rows, _key, _title, _label, path, **_kwargs) -> None:
        path.write_bytes(b"png")

    monkeypatch.setattr(load_analysis, "_render_metric_plot", touch_plot)
    manifest = load_analysis.write_blue_training_dynamics(
        rows, output, rolling_window=2, block_size=2, dpi=72,
    )
    assert manifest["episode_count"] == 2
    assert (output / "逐回合速度过载统计.csv").is_file()
    assert (output / "分段速度过载统计.csv").is_file()
    assert (output / "训练速度过载分析清单.json").is_file()

    monkeypatch.setattr(
        load_analysis, "write_blue_training_dynamics",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("绘图失败")),
    )
    assert load_analysis.safe_write_blue_training_dynamics(rows, output) is None
    assert "绘图失败" in capsys.readouterr().err

    monkeypatch.setattr(
        load_analysis, "analyze_blue_training_load",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("字体失败")),
    )
    assert load_analysis.safe_analyze_blue_training_load(training) is None
    assert "字体失败" in capsys.readouterr().err


def test_recorded_intervals_lengths_actions_and_metadata_conflicts_are_validated(
    tmp_path: Path,
) -> None:
    training = _run(tmp_path)
    quality = training / "flight_quality" / "flight_quality_episodes.jsonl"

    record = _trace(3, [1.0, 2.0, 3.0], actions=[None, 0, 29],
                    commands=[None, 1.0, 1.0], safety=[False, False, False],
                    sample_intervals=[0.0, 1.0, 2.0])
    quality.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="0–28"):
        load_analysis.stream_training_load_rows(training)

    record["trace"]["executed_action"] = [None, 0, 1]
    record["trace"]["estimated_actual_load_g"] = [1.0, 2.0]
    quality.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="estimated_actual_load_g.*长度"):
        load_analysis.stream_training_load_rows(training)

    record["trace"]["estimated_actual_load_g"] = [1.0, 2.0, 3.0]
    record["trace"]["sample_interval_s"] = [0.1, 1.0, 2.0]
    quality.write_text(json.dumps(record) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="首个值必须为 0"):
        load_analysis.stream_training_load_rows(training)

    # Restore a valid trace and make the two standard metadata sources disagree.
    record["trace"]["sample_interval_s"] = [0.0, 1.0, 2.0]
    quality.write_text(json.dumps(record) + "\n", encoding="utf-8")
    (training / "training_metrics.json").write_text(json.dumps({"episodes": [
        {"episode": 3, "missile_count": 1, "blue_survived": True},
    ]}), encoding="utf-8")
    (training / "episodes.json").write_text(json.dumps([
        {"episode": 3, "missile_count": 2, "blue_survived": True},
    ]), encoding="utf-8")
    with pytest.raises(ValueError, match="missile_count.*冲突"):
        load_analysis.stream_training_load_rows(training)


def test_recorded_speed_fields_take_precedence_and_are_strictly_validated() -> None:
    record = _trace(
        1, [1.0, 2.0, 3.0], actions=[None, 0, 1],
        commands=[None, 1.0, 2.0], safety=[False, False, False],
    )
    record["trace"].update({
        "speed_mps": [100.0, 200.0, 300.0],
        "horizontal_speed_mps": [90.0, 180.0, 270.0],
        "vertical_speed_mps": [-10.0, 20.0, -30.0],
        # Deliberately disagree with the scalar series: recorded scalars win.
        "velocity_mps": [[1.0, 1.0, 1.0]] * 3,
    })

    row = load_analysis.summarize_training_episode(record, {}, 1)

    assert row["speed_mean_mps"] == pytest.approx(800.0 / 3.0)
    assert row["speed_min_mps"] == pytest.approx(100.0)
    assert row["horizontal_speed_mean_mps"] == pytest.approx(240.0)
    assert row["horizontal_speed_p05_mps"] == pytest.approx(99.0)
    assert row["abs_vertical_speed_mean_mps"] == pytest.approx(80.0 / 3.0)
    assert row["abs_vertical_speed_p95_mps"] == pytest.approx(29.0)

    for key, invalid, message in (
        ("speed_mps", [100.0, 200.0], "speed_mps.*长度"),
        ("horizontal_speed_mps", [90.0, math.inf, 270.0],
         "horizontal_speed_mps.*有限数值"),
        ("vertical_speed_mps", [-10.0, None, -30.0],
         "vertical_speed_mps.*有限数值"),
    ):
        broken = json.loads(json.dumps(record))
        broken["trace"][key] = invalid
        with pytest.raises(ValueError, match=message):
            load_analysis.summarize_training_episode(broken, {}, 1)

    broken = json.loads(json.dumps(record))
    broken["trace"]["horizontal_speed_mps"] = [-1.0, 2.0, 3.0]
    with pytest.raises(ValueError, match="horizontal_speed_mps.*非负"):
        load_analysis.summarize_training_episode(broken, {}, 1)


def test_missing_speed_series_requires_valid_velocity_vectors() -> None:
    record = _trace(
        1, [1.0, 2.0, 3.0], actions=[None, 0, 1],
        commands=[None, 1.0, 2.0], safety=[False, False, False],
    )
    record["trace"]["velocity_mps"][1] = [1.0, 2.0]
    with pytest.raises(ValueError, match=r"velocity_mps\[1\].*长度为3"):
        load_analysis.summarize_training_episode(record, {}, 1)


def test_block_combines_time_weighted_moments_by_valid_duration() -> None:
    rows = [
        {
            "completion_order": 1, "episode": 1, "blue_survived": True,
            "valid_speed_samples": 2, "speed_valid_duration_s": 1.0,
            "speed_mean_mps": 100.0, "speed_p05_mps": 90.0,
            "speed_p50_mps": 100.0, "speed_p95_mps": 110.0,
            "speed_min_mps": 80.0, "speed_max_mps": 120.0,
            "speed_rms_mps": 100.0,
            "horizontal_speed_mean_mps": 80.0,
            "horizontal_speed_min_mps": 70.0,
            "horizontal_speed_p05_mps": 75.0,
            "abs_vertical_speed_mean_mps": 10.0,
            "abs_vertical_speed_p95_mps": 15.0,
            "abs_vertical_speed_max_mps": 20.0,
            "trace_elapsed_s": 1.0, "estimated_load_valid_duration_s": 1.0,
            "estimated_load_mean_g": 2.0, "estimated_load_rms_g": 2.0,
            "estimated_load_p50_g": 2.0, "estimated_load_p95_g": 2.0,
            "estimated_load_max_g": 2.0, "estimated_load_above_3g_s": 0.0,
            "estimated_load_above_6g_s": 0.0, "estimated_load_above_9g_s": 0.0,
            "commanded_load_valid_duration_s": 1.0, "commanded_load_mean_g": 3.0,
            "commanded_load_p95_g": 3.0, "commanded_load_max_g": 3.0,
            "executed_action_count": 1, "safety_intervention_count": 0,
            "action_switch_count": 0,
        },
        {
            "completion_order": 2, "episode": 2, "blue_survived": False,
            "valid_speed_samples": 10, "speed_valid_duration_s": 9.0,
            "speed_mean_mps": 200.0, "speed_p05_mps": 180.0,
            "speed_p50_mps": 200.0, "speed_p95_mps": 220.0,
            "speed_min_mps": 160.0, "speed_max_mps": 240.0,
            "speed_rms_mps": 200.0,
            "horizontal_speed_mean_mps": 180.0,
            "horizontal_speed_min_mps": 150.0,
            "horizontal_speed_p05_mps": 160.0,
            "abs_vertical_speed_mean_mps": 30.0,
            "abs_vertical_speed_p95_mps": 40.0,
            "abs_vertical_speed_max_mps": 50.0,
            "trace_elapsed_s": 9.0, "estimated_load_valid_duration_s": 9.0,
            "estimated_load_mean_g": 10.0, "estimated_load_rms_g": 10.0,
            "estimated_load_p50_g": 10.0, "estimated_load_p95_g": 10.0,
            "estimated_load_max_g": 10.0, "estimated_load_above_3g_s": 9.0,
            "estimated_load_above_6g_s": 9.0, "estimated_load_above_9g_s": 9.0,
            "commanded_load_valid_duration_s": 9.0, "commanded_load_mean_g": 9.0,
            "commanded_load_p95_g": 9.0, "commanded_load_max_g": 9.0,
            "executed_action_count": 9, "safety_intervention_count": 0,
            "action_switch_count": 0,
        },
    ]

    block = load_analysis.summarize_blocks(rows, 100)[0]

    assert block["valid_speed_samples"] == 12
    assert block["speed_valid_duration_s"] == pytest.approx(10.0)
    assert block["speed_mean_mps"] == pytest.approx(190.0)
    assert block["episode_speed_p05_mean_mps"] == pytest.approx(135.0)
    assert block["speed_min_mps"] == pytest.approx(80.0)
    assert block["speed_max_mps"] == pytest.approx(240.0)
    assert block["speed_rms_mps"] == pytest.approx(math.sqrt(37000.0))
    assert block["horizontal_speed_mean_mps"] == pytest.approx(170.0)
    assert block["horizontal_speed_min_mps"] == pytest.approx(70.0)
    assert block["episode_horizontal_speed_p05_mean_mps"] == pytest.approx(117.5)
    assert block["abs_vertical_speed_mean_mps"] == pytest.approx(28.0)
    assert block["episode_abs_vertical_speed_p95_mean_mps"] == pytest.approx(27.5)
    assert block["abs_vertical_speed_max_mps"] == pytest.approx(50.0)
    assert block["estimated_load_mean_g"] == pytest.approx(9.2)
    assert block["estimated_load_rms_g"] == pytest.approx(math.sqrt(90.4))
    assert block["commanded_load_mean_g"] == pytest.approx(8.4)
    assert block["estimated_load_above_6g_fraction"] == pytest.approx(0.9)


@pytest.mark.filterwarnings("ignore:Glyph .* missing from font.*:UserWarning")
def test_plot_draws_one_matching_color_rolling_curve_per_missile_group(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from matplotlib.figure import Figure

    rows = [
        {"completion_order": 1, "missile_count": 1, "estimated_load_mean_g": 1.0},
        {"completion_order": 2, "missile_count": 2, "estimated_load_mean_g": 10.0},
        {"completion_order": 3, "missile_count": 1, "estimated_load_mean_g": 3.0},
        {"completion_order": 4, "missile_count": 2, "estimated_load_mean_g": 14.0},
    ]
    captured: dict[str, object] = {}

    def capture(figure: Figure, *_args, **_kwargs) -> None:
        axis = figure.axes[0]
        captured["line_labels"] = [line.get_label() for line in axis.lines]
        captured["line_values"] = [list(line.get_ydata()) for line in axis.lines]
        captured["line_colors"] = [line.get_color() for line in axis.lines]
        captured["scatter_colors"] = [collection.get_facecolors()[0, :3]
                                      for collection in axis.collections]

    monkeypatch.setattr(load_analysis, "chinese_plot_context",
                        lambda _font=None: nullcontext("测试字体"))
    monkeypatch.setattr(Figure, "savefig", capture)
    load_analysis._render_metric_plot(
        rows, "estimated_load_mean_g", "标题", "纵轴", tmp_path / "plot.png",
        rolling_window=2, dpi=72, font_path=None,
    )

    assert captured["line_values"] == [[1.0, 2.0], [10.0, 12.0]]
    assert all("同类完成回合滚动均值" in label for label in captured["line_labels"])
    # Each group's rolling curve and raw scatter use the same base color.
    import matplotlib.colors as colors
    for line_color, scatter_color in zip(captured["line_colors"], captured["scatter_colors"]):
        assert colors.to_rgb(line_color) == pytest.approx(tuple(scatter_color))
