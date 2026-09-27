from __future__ import annotations

import json
from pathlib import Path

import pytest

from red_swarm_policy import plot_blue_results as batch


def test_batch_entry_accepts_separate_training_and_evaluation_directories(
    tmp_path: Path, monkeypatch,
) -> None:
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation（固定参数）"
    for run in (training, evaluation):
        quality = run / "flight_quality"
        quality.mkdir(parents=True)
        (quality / "flight_quality_episodes.jsonl").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "中文图表"
    calls: list[tuple[str, Path]] = []
    quality_options: list[dict] = []
    dynamics_options: list[dict] = []
    font = tmp_path / "中文字体.ttf"
    font.write_bytes(b"test double")

    def fake_training(source, destination, **kwargs):
        calls.append(("training", Path(source)))
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {"charts": {"reward": {"file": "reward.png"}}}

    def fake_grouped(source, destination, **kwargs):
        calls.append(("grouped", Path(source)))
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {"figures": ["escape_rate_overall.png"]}

    def fake_evaluation(source, destination, **kwargs):
        calls.append(("evaluation", Path(source)))
        Path(destination).mkdir(parents=True, exist_ok=True)
        (Path(destination) / "evaluation.png").write_bytes(b"png")
        return {"ok": True}

    def fake_dynamics(source, destination, **kwargs):
        calls.append(("dynamics", Path(source)))
        dynamics_options.append(kwargs)
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {
            "figures": [f"dynamics_{index:02d}.png" for index in range(1, 12)],
            "tables": ["逐回合速度过载统计.csv", "分段速度过载统计.csv"],
        }

    def fake_quality(source, destination, **kwargs):
        calls.append(("quality", Path(source)))
        quality_options.append(kwargs)
        Path(destination).mkdir(parents=True, exist_ok=True)
        name = "episode_000001.png"
        (Path(destination) / name).write_bytes(b"png")
        return {"files": [name], "records_scanned": 1}

    monkeypatch.setattr(batch, "analyze_training", fake_training)
    monkeypatch.setattr(batch, "analyze_blue_training_results", fake_grouped)
    monkeypatch.setattr(batch, "analyze_blue_training_load", fake_dynamics)
    monkeypatch.setattr(batch, "write_evaluation_analysis_safely", fake_evaluation)
    monkeypatch.setattr(batch, "replot_flight_quality_episodes", fake_quality)

    manifest = batch.generate_blue_plots(
        training, evaluation, output_root=output, font_path=font,
    )

    assert [name for name, _ in calls] == [
        "training", "grouped", "dynamics", "evaluation", "quality", "quality",
    ]
    assert calls[0][1] == training.resolve()
    assert calls[2][1] == training.resolve()
    assert calls[3][1] == evaluation.resolve()
    assert [options["font_path"] for options in quality_options] == [font, font]
    assert dynamics_options == [{"rolling_window": 20, "dpi": 160, "font_path": font}]
    assert manifest["training_dir"] == str(training.resolve())
    saved = json.loads((output / "blue_plot_batch_manifest.json").read_text(encoding="utf-8"))
    assert saved["figures"]["training_analysis"] == ["reward.png"]
    assert len(saved["figures"]["training_dynamics"]) == 11
    assert saved["tables"]["training_dynamics"] == [
        "逐回合速度过载统计.csv", "分段速度过载统计.csv",
    ]
    assert saved["outputs"]["training_dynamics"] == str(
        (output / "training_dynamics").resolve()
    )
    assert saved["figures"]["evaluation_flight_quality"] == ["episode_000001.png"]
    assert "episode_trace" not in saved
    assert "evaluation_episode" not in saved["figures"]


def test_batch_parser_exposes_input_directory_aliases() -> None:
    args = batch.build_parser().parse_args([
        "--train-dir", "训练", "--eval-dir", "测试", "--output-root", "输出",
    ])

    assert args.training_dir == Path("训练")
    assert args.evaluation_dir == Path("测试")
    assert args.output_root == Path("输出")


def _patch_standard_plotters(monkeypatch, calls: list[str]) -> None:
    def fake_training(_source, destination, **_kwargs):
        calls.append("training")
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {"charts": {"reward": {"file": "reward.png"}}}

    def fake_grouped(_source, destination, **_kwargs):
        calls.append("grouped")
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {"figures": ["escape_rate_overall.png"]}

    def fake_evaluation(_source, destination, **_kwargs):
        calls.append("evaluation")
        Path(destination).mkdir(parents=True, exist_ok=True)
        (Path(destination) / "evaluation.png").write_bytes(b"png")
        return {"ok": True}

    def fake_dynamics(_source, destination, **_kwargs):
        calls.append("dynamics")
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {
            "figures": [f"dynamics_{index:02d}.png" for index in range(1, 12)],
            "tables": ["逐回合速度过载统计.csv", "分段速度过载统计.csv"],
        }

    def fake_quality(_source, destination, **_kwargs):
        calls.append("quality")
        Path(destination).mkdir(parents=True, exist_ok=True)
        return {"files": ["episode_000001.png"], "records_scanned": 1}

    monkeypatch.setattr(batch, "analyze_training", fake_training)
    monkeypatch.setattr(batch, "analyze_blue_training_results", fake_grouped)
    monkeypatch.setattr(batch, "analyze_blue_training_load", fake_dynamics)
    monkeypatch.setattr(batch, "write_evaluation_analysis_safely", fake_evaluation)
    monkeypatch.setattr(batch, "replot_flight_quality_episodes", fake_quality)


def _episode_result(source: Path, output: Path, episode: int = 42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    return {
        "episode": episode,
        "source": str(source.resolve()),
        "output": str(output.resolve()),
        "samples": 4,
        "action_segments": 3,
        "files": ["蓝方逐时刻数据.csv", *batch.PLOT_FILENAMES],
    }


def test_full_batch_can_append_selected_episode_without_changing_default_pipeline(
    tmp_path: Path, monkeypatch,
) -> None:
    training = tmp_path / "training"
    evaluation = tmp_path / "evaluation"
    for run in (training, evaluation):
        quality = run / "flight_quality"
        quality.mkdir(parents=True)
        (quality / "flight_quality_episodes.jsonl").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    font = tmp_path / "font.ttf"
    font.write_bytes(b"font")
    calls: list[str] = []
    exported: dict[str, object] = {}
    _patch_standard_plotters(monkeypatch, calls)

    def fake_episode(source, destination, **kwargs):
        calls.append("episode")
        exported.update({"source": Path(source), "destination": Path(destination), **kwargs})
        return _episode_result(Path(source), Path(destination))

    monkeypatch.setattr(batch, "export_episode_trace", fake_episode)
    manifest = batch.generate_blue_plots(
        training, evaluation, output_root=output, episode=42, font_path=font, dpi=120,
    )

    assert calls == [
        "training", "grouped", "dynamics", "evaluation", "quality", "quality", "episode",
    ]
    assert exported["source"] == (
        evaluation / "flight_quality" / "flight_quality_episodes.jsonl"
    ).resolve()
    assert exported["destination"] == (
        output / "evaluation_episode_details" / "episode_000042"
    ).resolve()
    assert exported["episode"] == 42
    assert exported["missiles"] is None and exported["scenario_episode"] is None
    assert exported["font_path"] == font and exported["dpi"] == 120
    assert manifest["episode_trace"]["episode"] == 42
    assert manifest["figures"]["evaluation_episode"] == list(batch.PLOT_FILENAMES)
    saved = json.loads((output / "blue_plot_batch_manifest.json").read_text(encoding="utf-8"))
    assert saved["figures"]["evaluation_episode"] == list(batch.PLOT_FILENAMES)


def test_evaluation_only_mode_uses_scenario_selector_and_default_output_root(
    tmp_path: Path, monkeypatch,
) -> None:
    evaluation = tmp_path / "evaluation"
    quality = evaluation / "flight_quality"
    quality.mkdir(parents=True)
    source = quality / "flight_quality_episodes.jsonl"
    source.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "output"
    exported: dict[str, object] = {}

    def fake_episode(input_source, destination, **kwargs):
        exported.update({"source": Path(input_source), "destination": Path(destination), **kwargs})
        return _episode_result(Path(input_source), Path(destination), episode=503)

    monkeypatch.setattr(batch, "export_episode_trace", fake_episode)
    monkeypatch.setattr(
        batch, "analyze_blue_training_load",
        lambda *_args, **_kwargs: pytest.fail("仅单回合模式不得运行训练速度过载分析"),
    )
    manifest = batch.generate_evaluation_episode_plots(
        evaluation, missiles=2, scenario_episode=3, output_root=output_root, dpi=100,
    )

    expected = (
        output_root / "evaluation_episode_details" / "missiles_2_scenario_000003"
    ).resolve()
    assert exported["source"] == source.resolve()
    assert exported["destination"] == expected
    assert exported["episode"] is None
    assert exported["missiles"] == 2 and exported["scenario_episode"] == 3
    assert manifest["mode"] == "evaluation_episode_only"
    assert manifest["figures"]["evaluation_episode"] == list(batch.PLOT_FILENAMES)
    saved = json.loads((expected / "episode_plot_manifest.json").read_text(encoding="utf-8"))
    assert saved["episode_trace"]["episode"] == 503


def test_main_without_training_dispatches_only_single_episode(monkeypatch) -> None:
    received: dict[str, object] = {}

    def fake_episode_only(evaluation_dir, **kwargs):
        received.update({"evaluation_dir": evaluation_dir, **kwargs})
        return {"mode": "evaluation_episode_only", "episode_trace": {}, "figures": {}}

    monkeypatch.setattr(batch, "generate_evaluation_episode_plots", fake_episode_only)
    monkeypatch.setattr(
        batch, "generate_blue_plots",
        lambda *_args, **_kwargs: pytest.fail("不应运行完整训练/测试批处理"),
    )

    assert batch.main([
        "--evaluation-dir", "测试", "--episode", "7", "--episode-output", "单回合",
    ]) == 0
    assert received["evaluation_dir"] == Path("测试")
    assert received["episode"] == 7
    assert received["episode_output"] == Path("单回合")


@pytest.mark.parametrize("arguments, message", [
    ({"episode": 1, "missiles": 2, "scenario_episode": None}, "不能与"),
    ({"episode": None, "missiles": None, "scenario_episode": 2}, "必须同时指定"),
    ({"episode": None, "missiles": 2, "scenario_episode": None}, "只能与"),
    ({"episode": None, "missiles": None, "scenario_episode": None,
      "episode_output": "out"}, "必须与回合选择器"),
])
def test_episode_selector_combinations_are_strict(arguments, message) -> None:
    arguments.setdefault("episode_output", None)
    with pytest.raises(ValueError, match=message):
        batch._validate_episode_selector(**arguments)


def test_default_episode_directory_rules(tmp_path: Path) -> None:
    evaluation = (tmp_path / "evaluation").resolve()
    assert batch._episode_destination(
        evaluation, None, None, episode=7, missiles=None, scenario_episode=None,
    ) == evaluation / "episode_details" / "episode_000007"
    output_root = (tmp_path / "output").resolve()
    assert batch._episode_destination(
        evaluation, output_root, None, episode=None, missiles=3, scenario_episode=12,
    ) == output_root / "evaluation_episode_details" / "missiles_3_scenario_000012"


def test_cli_requires_training_directory_or_episode_selector() -> None:
    with pytest.raises(SystemExit) as error:
        batch.main(["--evaluation-dir", "测试"])
    assert error.value.code == 2
