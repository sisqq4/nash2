"""Read existing training logs and render analysis without importing the learner.

This module can also be run as a script, so offline plotting does not require
importing red_swarm_policy (whose package initializer imports PyTorch).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import fill
from typing import Any, Mapping, Sequence

try:
    from .plot_localization import chinese_plot_context
except ImportError:  # Support ``python analyze_training.py``.
    from plot_localization import chinese_plot_context


CHART_TITLES = {
    "reward": "奖励",
    "loss": "损失",
    "blue_escape_rate": "蓝方逃逸率",
    "validation_escape_rate": "固定评估逃逸率",
    "learning_rate": "学习率",
    "gradient_norm": "梯度范数",
    "optimizer_diagnostics": "优化器诊断",
    "training_progress": "训练进度",
    "episode_length": "回合长度",
}

METRIC_LABELS = {
    "actor_loss": "策略网络损失",
    "critic_loss": "价值网络损失",
    "loss": "损失",
    "learning_rate": "学习率",
    "assignment_actor_learning_rate": "分配策略网络学习率",
    "execution_actor_learning_rate": "执行策略网络学习率",
    "assignment_critic_learning_rate": "分配价值网络学习率",
    "execution_critic_learning_rate": "执行价值网络学习率",
    "gradient_norm_mean": "平均梯度范数",
    "assignment_actor_grad_norm_preclip": "分配策略网络裁剪前梯度范数",
    "execution_actor_grad_norm_preclip": "执行策略网络裁剪前梯度范数",
    "assignment_critic_grad_norm_preclip": "分配价值网络裁剪前梯度范数",
    "execution_critic_grad_norm_preclip": "执行价值网络裁剪前梯度范数",
    "policy_diagnostics.action_entropy": "策略诊断：动作熵",
    "assignment_entropy": "分配策略熵",
    "execution_entropy": "执行策略熵",
    "assignment_approx_kl": "分配策略近似 KL 散度",
    "execution_approx_kl": "执行策略近似 KL 散度",
    "assignment_clip_fraction": "分配策略裁剪比例",
    "execution_clip_fraction": "执行策略裁剪比例",
    "c51_clamp_low_fraction_mean": "C51 下界截断平均比例",
    "c51_clamp_high_fraction_mean": "C51 上界截断平均比例",
    "rollout_diagnostics.transitions_per_s": "采样诊断：每秒环境转移数",
    "rollout_diagnostics.episodes_per_hour": "采样诊断：每小时完成回合数",
    "rollout_diagnostics.wall_time_s": "采样诊断：运行耗时 (s)",
    "replay_size": "经验回放池大小",
    "completed_environment_transitions": "已完成环境转移数",
    "completed_optimizer_updates": "已完成参数更新次数",
    "cuda_memory_allocated_mb": "CUDA 已分配显存 (MB)",
}


@dataclass
class TrainingData:
    iterations: list[dict[str, Any]] = field(default_factory=list)
    episodes: list[dict[str, Any]] = field(default_factory=list)
    evaluations: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


@dataclass
class Curve:
    name: str
    label: str
    x: list[float]
    y: list[float | None]
    x_label: str
    percentage: bool = False


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _field(row: Mapping[str, Any], key: str) -> Any:
    value: Any = row
    for part in key.split("."):
        if not isinstance(value, Mapping):
            return None
        value = value.get(part)
    return value


def rolling_mean(values: Sequence[float | None], window: int) -> list[float | None]:
    """Trailing mean of finite observations; missing current points stay missing."""
    if window < 1:
        raise ValueError("smoothing_window must be positive")
    recent: deque[float | None] = deque()
    total = 0.0
    count = 0
    result: list[float | None] = []
    for value in values:
        value = _number(value)
        recent.append(value)
        if value is not None:
            total += value
            count += 1
        if len(recent) > window:
            old = recent.popleft()
            if old is not None:
                total -= old
                count -= 1
        result.append(total / count if value is not None and count else None)
    return result


def _rows(value: Any, name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise ValueError(f"{name} must be a list of objects")
    return value


def _extract(document: Any) -> TrainingData:
    structured = isinstance(document, dict) and any(
        key in document for key in ("iterations", "episodes", "curriculum_evaluations")
    ) and not isinstance(document.get("episodes"), (int, float))
    if structured:
        data = TrainingData(
            iterations=_rows(document.get("iterations"), "iterations"),
            episodes=_rows(document.get("episodes"), "episodes"),
            evaluations=_rows(document.get("curriculum_evaluations"), "curriculum_evaluations"),
        )
    else:
        rows = _rows(document if isinstance(document, list) else [document], "training records")
        data = TrainingData()
        for row in rows:
            event = row.get("event")
            if event == "iteration" or (event is None and "iteration" in row):
                data.iterations.append(row)
            elif event == "curriculum_evaluation":
                data.evaluations.append(row)
            elif event in (None, "episode") and "episode" in row:
                data.episodes.append(row)
    if not data.iterations and not data.episodes and not structured:
        raise ValueError("No training iteration or episode records found")
    # Parallel episodes finish out of order; episode IDs identify their intended order.
    if data.episodes and all(_number(row.get("episode")) is not None for row in data.episodes):
        data.episodes = sorted(data.episodes, key=lambda row: row["episode"])
    return data


def load_training_data(source: str | Path | Mapping[str, Any]) -> tuple[TrainingData, Path | None]:
    """Accept a final metrics JSON, episode list, streaming JSONL, or run directory."""
    if isinstance(source, Mapping):
        return _extract(dict(source)), None
    path = Path(source)
    if path.is_dir():
        candidates = [path / name for name in (
            "training_metrics.json", "env_training_metrics.json", "training.jsonl", "episodes.json"
        )]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
        if path.is_dir():
            raise ValueError(f"No standard training log in {path}; pass the metrics file explicitly")
    encoded = path.read_text(encoding="utf-8-sig")
    notes: list[str] = []
    if path.suffix.lower() in (".jsonl", ".ndjson", ".log"):
        records: list[dict[str, Any]] = []
        lines = encoded.splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                # Only an unfinished last write is recoverable. Never silently
                # discard corrupted complete records in the middle of a run.
                if index == len(lines) - 1 and not encoded.endswith(("\n", "\r")):
                    notes.append(f"Ignored incomplete final JSONL line {index + 1}")
                    break
                raise ValueError(f"Invalid JSONL at {path}:{index + 1}: {error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL record at {path}:{index + 1} must be an object")
            records.append(record)
        data = _extract(records)
    else:
        data = _extract(json.loads(encoded))
    data.notes.extend(notes)
    return data, path.resolve()


def _curve(rows: list[dict[str, Any]], key: str, label: str, *,
           axis: str = "iteration", percentage: bool = False,
           complement: bool = False, boolean: bool = False) -> Curve:
    x_key = "completed_episodes" if axis == "iteration" and rows and all(
        _number(row.get("completed_episodes")) is not None for row in rows
    ) else axis
    x_label = {"episode": "回合编号", "completed_episodes": "已完成回合数",
               "iteration": "训练迭代次数"}[x_key]
    xs: list[float] = []
    ys: list[float | None] = []
    for index, row in enumerate(rows, 1):
        x = _number(row.get(x_key))
        xs.append(float(index) if x is None else x)
        raw = _field(row, key)
        value = float(raw) if boolean and isinstance(raw, bool) else _number(raw)
        if boolean and value not in (None, 0.0, 1.0):
            value = None
        if percentage and value is not None and not 0 <= value <= 1:
            value = None
        if value is not None and complement:
            value = 1 - value
        ys.append(value)
    return Curve(key, label, xs, ys, x_label, percentage)


def build_charts(data: TrainingData) -> dict[str, list[Curve]]:
    """Select only logged metrics; no environment steps or policy inference."""
    charts: dict[str, list[Curve]] = {"reward": [], "loss": [], "blue_escape_rate": []}

    def add(chart: str, rows: list[dict[str, Any]], key: str, label: str, **kwargs: Any) -> None:
        curve = _curve(rows, key, label, **kwargs)
        if any(value is not None for value in curve.y):
            charts.setdefault(chart, []).append(curve)

    episodes, iterations = data.episodes, data.iterations
    add("reward", episodes, "reward", "回合奖励", axis="episode")
    for key, label in (("reward_mean", "记录窗口平均奖励"),
                       ("episode_high_reward_mean", "回合高层平均回报"),
                       ("episode_low_return_mean", "回合低层平均回报"),
                       ("active_low_reward_mean", "有效低层单步平均奖励")):
        add("reward", iterations, key, label)
    for key, label in (("loss_mean", "记录窗口平均优化器损失"),
                       ("assignment_actor_loss", "分配策略网络损失"),
                       ("execution_actor_loss", "执行策略网络损失"),
                       ("assignment_critic_loss", "分配价值网络损失"),
                       ("execution_critic_loss", "执行价值网络损失")):
        add("loss", iterations, key, label)
    if not charts["loss"]:
        for key in ("actor_loss", "critic_loss", "loss"):
            add("loss", iterations, key, METRIC_LABELS[key])
    if not charts["loss"]:
        key = "learner_loss_at_completion" if any(
            _number(row.get("learner_loss_at_completion")) is not None for row in episodes
        ) else "mean_loss"
        add("loss", episodes, key, "回合结束时的学习器损失快照", axis="episode")
        if charts["loss"]:
            data.notes.append("Episode loss is a learner snapshot, not an episode-averaged loss.")
    add("blue_escape_rate", episodes, "blue_survived", "回合逃逸结果",
        axis="episode", percentage=True, boolean=True)
    add("blue_escape_rate", iterations, "survival_rate", "训练窗口逃逸率", percentage=True)
    add("blue_escape_rate", iterations, "fixed_validation.average_damage_rate",
        "固定验证蓝方存活率（1 - 平均受损率）", percentage=True, complement=True)
    add("blue_escape_rate", iterations, "rollout_diagnostics.average_damage_rate",
        "采样阶段蓝方存活比例（过程快照，非最终逃逸结果）", percentage=True, complement=True)
    if any(curve.name.endswith("average_damage_rate") for curve in charts["blue_escape_rate"]):
        data.notes.append("Blue alive fraction = 1 - average_damage_rate, never 1 - full_success_rate. "
                          "Rollout snapshots may precede episode termination; fixed validation is shown separately.")
    scenario_keys = sorted({str(key) for row in data.evaluations
                            for key in (row.get("survival_rates") or {})})
    # JSON-loaded keys are strings; in-memory callers can also supply integer keys.
    evaluations = [{**row, "survival_rates": {str(k): v for k, v in
                    (row.get("survival_rates") or {}).items()}} for row in data.evaluations]
    for key in scenario_keys:
        add("validation_escape_rate", evaluations, f"survival_rates.{key}",
            f"固定评估逃逸率：场景 {key}", percentage=True)
    for key in ("learning_rate", "assignment_actor_learning_rate", "execution_actor_learning_rate",
                "assignment_critic_learning_rate", "execution_critic_learning_rate"):
        add("learning_rate", iterations, key, METRIC_LABELS[key])
    for key in ("gradient_norm_mean", "assignment_actor_grad_norm_preclip",
                "execution_actor_grad_norm_preclip", "assignment_critic_grad_norm_preclip",
                "execution_critic_grad_norm_preclip"):
        add("gradient_norm", iterations, key, METRIC_LABELS[key])
    for key in ("policy_diagnostics.action_entropy", "assignment_entropy", "execution_entropy",
                "assignment_approx_kl", "execution_approx_kl", "assignment_clip_fraction",
                "execution_clip_fraction", "c51_clamp_low_fraction_mean", "c51_clamp_high_fraction_mean"):
        add("optimizer_diagnostics", iterations, key, METRIC_LABELS[key])
    for key in ("rollout_diagnostics.transitions_per_s", "rollout_diagnostics.episodes_per_hour",
                "rollout_diagnostics.wall_time_s", "replay_size", "completed_environment_transitions",
                "completed_optimizer_updates", "cuda_memory_allocated_mb"):
        add("training_progress", iterations, key, METRIC_LABELS[key])
    add("episode_length", episodes, "decision_steps", "回合决策步数", axis="episode")
    add("episode_length", episodes, "simulation_time_s", "回合仿真时长 (s)", axis="episode")
    add("episode_length", iterations, "simulation_time_mean_s", "窗口平均仿真时长 (s)")
    for name, curves in charts.items():
        if not curves:
            data.notes.append(f"{name}: no finite logged values; chart contains a missing-data notice.")
    return charts


def _render_chart(path: Path, name: str, curves: list[Curve], window: int, dpi: int,
                  font_path: str | Path | None = None) -> None:
    # Explicit Agg canvas avoids switching the application's global pyplot backend
    # or displaying/blocking on GUI windows. Import only during postprocessing.
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import PercentFormatter

    with chinese_plot_context(font_path):
        columns = 2 if len(curves) > 1 else 1
        rows = max(1, math.ceil(len(curves) / columns))
        figure = Figure(figsize=(12 if columns == 2 else 9, 3.4 * rows + .5), layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(rows, columns, squeeze=False).ravel()
        figure.suptitle(CHART_TITLES.get(name, name), fontsize=15, fontweight="bold")
        try:
            if not curves:
                axes[0].text(.5, .5, "给定训练日志中没有可绘制的有限数值。\n"
                             "缺失数据不会按零值处理。",
                             ha="center", va="center", transform=axes[0].transAxes)
                axes[0].set_axis_off()
            for axis, curve in zip(axes, curves):
                values = [math.nan if value is None else value for value in curve.y]
                marker = "." if len(values) < 3 or any(value is None for value in curve.y) else None
                if curve.name == "blue_survived":
                    averaged = rolling_mean(curve.y, window)
                    axis.plot(curve.x, averaged, color="#2563eb", linewidth=1.8,
                              label=f"最近 {window} 个回合", marker=marker)
                    total = 0.0
                    count = 0
                    cumulative: list[float | None] = []
                    for value in curve.y:
                        if value is not None:
                            total += value
                            count += 1
                        cumulative.append(total / count if value is not None and count else None)
                    axis.plot(curve.x, cumulative, color="#d97706", label="累计已记录回合")
                else:
                    axis.plot(curve.x, values, color="#94a3b8", linewidth=1, label="日志记录值",
                              marker=marker)
                    if window > 1:
                        axis.plot(curve.x, rolling_mean(curve.y, window), color="#2563eb", linewidth=1.8,
                                  label=f"最近 {window} 条记录（仅有限值）",
                                  marker=marker)
                axis.set_title(fill(curve.label, width=48 if columns == 2 else 80), fontsize=10)
                axis.set_xlabel(curve.x_label)
                if curve.percentage:
                    axis.set_ylim(-.03, 1.03)
                    axis.yaxis.set_major_formatter(PercentFormatter(1.0))
                axis.grid(True, alpha=.22)
                axis.legend(fontsize=8, loc="best")
            for axis in axes[max(len(curves), 1):]:
                axis.set_axis_off()
            figure.savefig(path, dpi=dpi)
        finally:
            figure.clear()


def analyze_training(source: str | Path | Mapping[str, Any], output: str | Path | None = None,
                     *, smoothing_window: int = 20, dpi: int = 160,
                     font_path: str | Path | None = None) -> dict[str, Any]:
    """Create only derived PNG/JSON files in a dedicated analysis directory."""
    if isinstance(smoothing_window, bool) or not isinstance(smoothing_window, int) or smoothing_window < 1:
        raise ValueError("smoothing_window must be a positive integer")
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    data, source_path = load_training_data(source)
    if output is None:
        if source_path is None:
            raise ValueError("output is required for in-memory training data")
        output = source_path.parent / f"{source_path.stem}_analysis"
    output_path = Path(output).resolve()
    if source_path is not None and output_path == source_path.parent:
        raise ValueError("output must be a separate analysis directory, not the input log directory")
    charts = build_charts(data)
    output_path.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "source": str(source_path) if source_path else "in-memory training metrics",
        "output": str(output_path), "smoothing_window": smoothing_window,
        "iteration_count": len(data.iterations), "episode_count": len(data.episodes),
        "evaluation_count": len(data.evaluations), "charts": {},
        "notes": ["Smoothing uses a trailing number of records, not environment steps. "
                  "Logged window means are not reweighted by episode count. "
                  "Null/non-finite values remain gaps, including pre-update loss.", *data.notes],
    }
    for name, curves in charts.items():
        filename = f"{name}.png"
        _render_chart(output_path / filename, name, curves, smoothing_window, dpi, font_path)
        series = []
        for curve in curves:
            finite = [value for value in curve.y if value is not None]
            series.append({"metric": curve.name, "label": curve.label, "x_label": curve.x_label,
                           "count": len(finite), "missing_count": len(curve.y) - len(finite),
                           "first": finite[0], "last": finite[-1],
                           "min": min(finite), "max": max(finite), "mean": sum(finite) / len(finite)})
        report["charts"][name] = {"file": filename, "series": series}
    (output_path / "analysis_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return report


def write_training_analysis_safely(source: str | Path | Mapping[str, Any],
                                 output: str | Path | None = None, *,
                                 smoothing_window: int = 20, dpi: int = 160,
                                 font_path: str | Path | None = None) -> dict[str, Any] | None:
    """Best-effort postprocessing; failures never invalidate saved training results."""
    try:
        return analyze_training(source, output, smoothing_window=smoothing_window, dpi=dpi,
                                font_path=font_path)
    except Exception as error:
        print(f"[training analysis] Plot generation skipped: {type(error).__name__}: {error}", file=sys.stderr)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Training JSON/JSONL file or run directory")
    parser.add_argument("--output", type=Path, help="Separate directory for generated analysis")
    parser.add_argument("--smoothing-window", type=int, default=20)
    parser.add_argument("--dpi", type=int, default=160)
    parser.add_argument("--font", type=Path, help="中文字体文件；默认自动查找系统字体")
    args = parser.parse_args(argv)
    try:
        report = analyze_training(args.input, args.output, smoothing_window=args.smoothing_window,
                                  dpi=args.dpi, font_path=args.font)
    except (OSError, ValueError, ImportError) as error:
        print(f"[training analysis] {error}", file=sys.stderr)
        return 1
    print(json.dumps({"output": report["output"], "charts": list(report["charts"])}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
