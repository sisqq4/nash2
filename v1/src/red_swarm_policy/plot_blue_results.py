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
from .analyze_blue_training_results import analyze_blue_training_results
from .analyze_training import analyze_training
from .blue_rl.flight_quality import replot_flight_quality_episodes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", "--train-dir", dest="training_dir", type=Path,
                        required=True, help="训练结果目录")
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
    parser.add_argument("--include-report-charts", action="store_true",
                        help="同时运行第二版报告批量图表导出（会读取完整测试轨迹）")
    parser.add_argument("--report-output", type=Path,
                        help="第二版报告图表目录；默认位于统一输出根目录或测试目录下")
    parser.add_argument("--report-dpi", type=int, default=220)
    return parser


def _destinations(training_dir: Path, evaluation_dir: Path,
                  output_root: Path | None) -> dict[str, Path]:
    if output_root is None:
        return {
            "training_analysis": training_dir / "training_analysis",
            "training_results": training_dir / "training_analysis" / "results",
            "evaluation_analysis": evaluation_dir / "evaluation_analysis",
            "training_flight_quality": training_dir / "flight_quality",
            "evaluation_flight_quality": evaluation_dir / "flight_quality",
            "report": evaluation_dir / "report_charts_zh",
            "manifest": evaluation_dir / "blue_plot_batch_manifest.json",
        }
    return {
        "training_analysis": output_root / "training_analysis",
        "training_results": output_root / "training_results",
        "evaluation_analysis": output_root / "evaluation_analysis",
        "training_flight_quality": output_root / "training_flight_quality",
        "evaluation_flight_quality": output_root / "evaluation_flight_quality",
        "report": output_root / "report_charts_zh",
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


def generate_blue_plots(
    training_dir: str | Path,
    evaluation_dir: str | Path,
    *,
    output_root: str | Path | None = None,
    font_path: str | Path | None = None,
    dpi: int = 160,
    smoothing_window: int = 20,
    flight_quality_plot_limit: int = 10,
    include_report_charts: bool = False,
    report_output: str | Path | None = None,
    report_dpi: int = 220,
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

    report_destination = Path(report_output).resolve() if report_output is not None else outputs["report"]
    if include_report_charts:
        from .export_blue_report_charts import main as export_report_charts

        arguments = [
            "--train-dir", str(training),
            "--evaluation-dir", str(evaluation),
            "--output", str(report_destination),
            "--dpi", str(report_dpi),
        ]
        if font_path is not None:
            arguments += ["--font", str(Path(font_path))]
        if export_report_charts(arguments) != 0:
            raise RuntimeError("第二版报告图表导出失败")

    manifest = {
        "training_dir": str(training),
        "evaluation_dir": str(evaluation),
        "output_root": str(root) if root is not None else None,
        "font_path": str(Path(font_path).resolve()) if font_path is not None else None,
        "outputs": {name: str(path.resolve()) for name, path in outputs.items()
                    if name != "manifest"},
        "figures": {
            "training_analysis": [item["file"] for item in training_report["charts"].values()],
            "training_results": grouped_training_report["figures"],
            "evaluation_analysis": sorted(path.name for path in outputs["evaluation_analysis"].glob("*.png")),
            "training_flight_quality": flight_reports["training"]["files"],
            "evaluation_flight_quality": flight_reports["evaluation"]["files"],
        },
        "report_charts": str(report_destination) if include_report_charts else None,
    }
    outputs["manifest"].parent.mkdir(parents=True, exist_ok=True)
    outputs["manifest"].write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = generate_blue_plots(
            args.training_dir,
            args.evaluation_dir,
            output_root=args.output_root,
            font_path=args.font,
            dpi=args.dpi,
            smoothing_window=args.smoothing_window,
            flight_quality_plot_limit=args.flight_quality_plot_limit,
            include_report_charts=args.include_report_charts,
            report_output=args.report_output,
            report_dpi=args.report_dpi,
        )
    except (OSError, ValueError, RuntimeError, ImportError) as error:
        print(f"批量绘图失败：{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
