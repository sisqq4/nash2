"""批量重绘蓝方训练、测试和飞行质量图。

本入口只读取已有结果，不启动环境、不加载检查点，也不更新模型参数。
训练目录和测试目录可以相互独立；默认把派生图写回各自原有分析目录。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

from .analyze_blue_evaluations import write_evaluation_analysis_safely
from .analyze_blue_training_load import analyze_blue_training_load
from .analyze_blue_training_results import analyze_blue_training_results
from .analyze_training import analyze_training
from ..blue_rl.flight_quality import replot_flight_quality_episodes
from .export_blue_episode_trace import PLOT_FILENAMES, export_episode_trace


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", "--train-dir", dest="training_dir", type=Path,
                        help="训练结果目录；省略时仅导出所选测试回合")
    parser.add_argument("--evaluation-dir", "--eval-dir", dest="evaluation_dir", type=Path,
                        required=True, help="测试结果目录")
    parser.add_argument(
        "--output-root", type=Path,
        help="统一输出根目录；省略时覆盖两个输入目录中的派生图，不改原始数据",
    )
    parser.add_argument("--font", type=Path, help="中文字体文件；默认自动查找系统字体")
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--smoothing-window", type=int, default=20)
    parser.add_argument("--flight-quality-plot-limit", type=int, default=10,
                        help="训练和测试各重绘评分最低的多少个回合")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--episode", type=_positive_integer, help="要详细导出的测试全局回合序号")
    selector.add_argument("--scenario-episode", type=_positive_integer,
                          help="要详细导出的场景内回合序号；必须同时指定 --missiles")
    parser.add_argument("--missiles", type=_positive_integer,
                        help="按场景内回合选择时的红弹数量")
    parser.add_argument("--episode-output", type=Path,
                        help="单回合详情的最终输出目录；省略时按选择器自动命名")
    parser.add_argument(
        "--show-safety-interventions", action="store_true",
        help="在所选单回合的时序图和轨迹图中标出安全介入；默认不标注",
    )
    return parser


def _destinations(training_dir: Path, evaluation_dir: Path,
                  output_root: Path | None) -> dict[str, Path]:
    if output_root is None:
        return {
            "training_analysis": training_dir / "training_analysis",
            "training_results": training_dir / "training_analysis" / "results",
            "training_dynamics": training_dir / "training_load_analysis",
            "evaluation_analysis": evaluation_dir / "evaluation_analysis",
            "training_flight_quality": training_dir / "flight_quality",
            "evaluation_flight_quality": evaluation_dir / "flight_quality",
            "manifest": evaluation_dir / "blue_plot_batch_manifest.json",
        }
    return {
        "training_analysis": output_root / "training_analysis",
        "training_results": output_root / "training_results",
        "training_dynamics": output_root / "training_dynamics",
        "evaluation_analysis": output_root / "evaluation_analysis",
        "training_flight_quality": output_root / "training_flight_quality",
        "evaluation_flight_quality": output_root / "evaluation_flight_quality",
        "manifest": output_root / "blue_plot_batch_manifest.json",
    }


def _flight_quality_source(run_dir: Path) -> Path:
    source = run_dir / "flight_quality" / "flight_quality_episodes.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"缺少飞行质量逐回合记录：{source}")
    return source


def _replot_flight_quality(source: Path, output: Path, plot_limit: int,
                           font_path: str | Path | None) -> dict[str, Any]:
    return replot_flight_quality_episodes(
        source, output, plot_limit=plot_limit, overwrite=True, clear_existing=True,
        font_path=font_path,
    )


def _validate_episode_selector(*, episode: int | None, missiles: int | None,
                               scenario_episode: int | None,
                               episode_output: str | Path | None = None) -> bool:
    """Validate the two supported episode identities and return whether one is selected."""
    for name, value in (("episode", episode), ("missiles", missiles),
                        ("scenario_episode", scenario_episode)):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                  or value <= 0):
            raise ValueError(f"{name} 必须是正整数")
    if episode is not None:
        if missiles is not None or scenario_episode is not None:
            raise ValueError("--episode 不能与 --missiles 或 --scenario-episode 同时使用")
        return True
    if scenario_episode is not None:
        if missiles is None:
            raise ValueError("使用 --scenario-episode 时必须同时指定 --missiles")
        return True
    if missiles is not None:
        raise ValueError("--missiles 只能与 --scenario-episode 一起使用")
    if episode_output is not None:
        raise ValueError("--episode-output 必须与回合选择器一起使用")
    return False


def _episode_destination(evaluation: Path, output_root: Path | None,
                         episode_output: str | Path | None, *,
                         episode: int | None, missiles: int | None,
                         scenario_episode: int | None) -> Path:
    if episode_output is not None:
        return Path(episode_output).resolve()
    base = (output_root / "evaluation_episode_details" if output_root is not None
            else evaluation / "episode_details")
    if episode is not None:
        name = f"episode_{episode:06d}"
    else:
        assert missiles is not None and scenario_episode is not None
        name = f"missiles_{missiles}_scenario_{scenario_episode:06d}"
    return base / name


def _export_selected_episode(evaluation: Path, *, output_root: Path | None,
                             episode_output: str | Path | None,
                             episode: int | None, missiles: int | None,
                             scenario_episode: int | None,
                             font_path: str | Path | None, dpi: int,
                             show_safety_interventions: bool) -> dict[str, Any]:
    destination = _episode_destination(
        evaluation, output_root, episode_output, episode=episode, missiles=missiles,
        scenario_episode=scenario_episode,
    )
    return export_episode_trace(
        _flight_quality_source(evaluation), destination, episode=episode, missiles=missiles,
        scenario_episode=scenario_episode, font_path=font_path, dpi=dpi,
        show_safety_interventions=show_safety_interventions,
    )


def generate_evaluation_episode_plots(
    evaluation_dir: str | Path,
    *,
    episode: int | None = None,
    missiles: int | None = None,
    scenario_episode: int | None = None,
    output_root: str | Path | None = None,
    episode_output: str | Path | None = None,
    font_path: str | Path | None = None,
    dpi: int = 160,
    show_safety_interventions: bool = False,
) -> dict[str, Any]:
    """Export only one selected evaluation episode through the batch entry point."""
    selected = _validate_episode_selector(
        episode=episode, missiles=missiles, scenario_episode=scenario_episode,
        episode_output=episode_output,
    )
    if not selected:
        raise ValueError("仅测试单回合模式必须指定 --episode 或 --missiles 与 --scenario-episode")
    evaluation = Path(evaluation_dir).resolve()
    if not evaluation.is_dir():
        raise FileNotFoundError(f"测试结果目录不存在：{evaluation}")
    root = Path(output_root).resolve() if output_root is not None else None
    episode_report = _export_selected_episode(
        evaluation, output_root=root, episode_output=episode_output, episode=episode,
        missiles=missiles, scenario_episode=scenario_episode, font_path=font_path, dpi=dpi,
        show_safety_interventions=show_safety_interventions,
    )
    manifest = {
        "mode": "evaluation_episode_only",
        "evaluation_dir": str(evaluation),
        "output_root": str(root) if root is not None else None,
        "font_path": str(Path(font_path).resolve()) if font_path is not None else None,
        "episode_trace": episode_report,
        "figures": {
            "evaluation_episode": [name for name in episode_report["files"]
                                   if name in PLOT_FILENAMES],
        },
    }
    manifest_path = Path(episode_report["output"]) / "episode_plot_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def generate_blue_plots(
    training_dir: str | Path,
    evaluation_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    font_path: str | Path | None = None,
    dpi: int = 160,
    smoothing_window: int = 20,
    flight_quality_plot_limit: int = 10,
    episode: int | None = None,
    missiles: int | None = None,
    scenario_episode: int | None = None,
    episode_output: str | Path | None = None,
    show_safety_interventions: bool = False,
) -> dict[str, Any]:
    """Generate every native Blue result figure from two existing run directories."""
    training = Path(training_dir).resolve()
    evaluation = Path(evaluation_dir).resolve()
    root = Path(output_root).resolve() if output_root is not None else None
    if not training.is_dir():
        raise FileNotFoundError(f"训练结果目录不存在：{training}")
    if not evaluation.is_dir():
        raise FileNotFoundError(f"测试结果目录不存在：{evaluation}")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi 必须是正整数")
    if (isinstance(flight_quality_plot_limit, bool)
            or not isinstance(flight_quality_plot_limit, int)
            or flight_quality_plot_limit < 0):
        raise ValueError("flight_quality_plot_limit 必须是非负整数")
    episode_selected = _validate_episode_selector(
        episode=episode, missiles=missiles, scenario_episode=scenario_episode,
        episode_output=episode_output,
    )
    if episode_selected and not 50 <= dpi <= 1200:
        raise ValueError("导出单回合详情时，dpi 必须在 50 到 1200 之间")

    outputs = _destinations(training, evaluation, root)
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)

    training_report = analyze_training(
        training, outputs["training_analysis"], smoothing_window=smoothing_window,
        dpi=dpi, font_path=font_path,
    )
    grouped_training_report = analyze_blue_training_results(
        training, outputs["training_results"], dpi=dpi, font_path=font_path,
    )
    training_dynamics_report = analyze_blue_training_load(
        training, outputs["training_dynamics"], rolling_window=smoothing_window,
        dpi=dpi, font_path=font_path,
    )
    evaluation_report = write_evaluation_analysis_safely(
        evaluation, outputs["evaluation_analysis"], font_path=font_path,
    )
    if evaluation_report is None:
        raise RuntimeError("测试结果图生成失败；详细原因见此前的标准错误输出")

    flight_reports = {
        "training": _replot_flight_quality(
            _flight_quality_source(training), outputs["training_flight_quality"],
            flight_quality_plot_limit, font_path,
        ),
        "evaluation": _replot_flight_quality(
            _flight_quality_source(evaluation), outputs["evaluation_flight_quality"],
            flight_quality_plot_limit, font_path,
        ),
    }

    episode_report = None
    if episode_selected:
        episode_report = _export_selected_episode(
            evaluation, output_root=root, episode_output=episode_output, episode=episode,
            missiles=missiles, scenario_episode=scenario_episode,
            font_path=font_path, dpi=dpi,
            show_safety_interventions=show_safety_interventions,
        )

    figures = {
        "training_analysis": [item["file"] for item in training_report["charts"].values()],
        "training_results": grouped_training_report["figures"],
        "training_dynamics": training_dynamics_report["figures"],
        "evaluation_analysis": sorted(path.name for path in outputs["evaluation_analysis"].glob("*.png")),
        "training_flight_quality": flight_reports["training"]["files"],
        "evaluation_flight_quality": flight_reports["evaluation"]["files"],
    }
    if episode_report is not None:
        figures["evaluation_episode"] = [name for name in episode_report["files"]
                                         if name in PLOT_FILENAMES]

    manifest = {
        "training_dir": str(training),
        "evaluation_dir": str(evaluation),
        "output_root": str(root) if root is not None else None,
        "font_path": str(Path(font_path).resolve()) if font_path is not None else None,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()
                    if name != "manifest"},
        "figures": figures,
        "tables": {
            "training_dynamics": training_dynamics_report["tables"],
        },
    }
    if episode_report is not None:
        manifest["episode_trace"] = episode_report
    outputs["manifest"].parent.mkdir(parents=True, exist_ok=True)
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        episode_selected = _validate_episode_selector(
            episode=args.episode, missiles=args.missiles,
            scenario_episode=args.scenario_episode, episode_output=args.episode_output,
        )
    except ValueError as error:
        parser.error(str(error))
    if args.training_dir is None and not episode_selected:
        parser.error("未指定 --training-dir 时，必须选择一个测试回合")
    try:
        if args.training_dir is None:
            manifest = generate_evaluation_episode_plots(
                args.evaluation_dir, episode=args.episode, missiles=args.missiles,
                scenario_episode=args.scenario_episode, output_root=args.output_root,
                episode_output=args.episode_output, font_path=args.font, dpi=args.dpi,
                show_safety_interventions=args.show_safety_interventions,
            )
        else:
            manifest = generate_blue_plots(
                args.training_dir,
                args.evaluation_dir,
                output_root=args.output_root,
                font_path=args.font,
                dpi=args.dpi,
                smoothing_window=args.smoothing_window,
                flight_quality_plot_limit=args.flight_quality_plot_limit,
                episode=args.episode,
                missiles=args.missiles,
                scenario_episode=args.scenario_episode,
                episode_output=args.episode_output,
                show_safety_interventions=args.show_safety_interventions,
            )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"批量绘图失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
