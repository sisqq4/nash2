"""Stream Blue evaluation traces into speed/load monitoring artifacts.

The evaluator archives one complete episode per line in
``flight_quality/flight_quality_episodes.jsonl``.  This module deliberately
reads that JSONL instead of the much larger consolidated
``flight_quality.json`` and retains only compact episode rows, scalar samples
needed for exact percentiles, and episode-balanced time-bin values.

It is an offline reporter: no environment is created, no checkpoint is loaded,
and source evaluation artifacts are never modified.
"""
from __future__ import annotations

import argparse
from array import array
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from .analyze_blue_training_load import summarize_training_episode
    from .plot_localization import chinese_plot_context
except ImportError:  # Support ``python analyze_blue_evaluation_dynamics.py``.
    from analyze_blue_training_load import summarize_training_episode
    from plot_localization import chinese_plot_context


EPISODE_CSV = "逐回合速度过载统计.csv"
SUMMARY_CSV = "全部及分弹数速度过载汇总.csv"
SUMMARY_JSON = "全部及分弹数速度过载汇总.json"
PROFILE_CSV = "速度过载时间剖面.csv"
SPEED_PNG = "速度时间剖面.png"
LOAD_PNG = "过载时间剖面.png"
MANIFEST_JSON = "测试速度过载分析清单.json"

PROFILE_METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("speed", "总速度", "mps", "米/秒"),
    ("estimated_load", "估算实际过载", "g", "g"),
    ("commanded_load", "名义指令过载", "g", "g"),
)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    clean = np.asarray(list(values), dtype=float)
    if not clean.size:
        return None
    return float(np.percentile(clean, fraction * 100.0))


def _buffer_percentile(values: array, fraction: float) -> float | None:
    if not values:
        return None
    # np.frombuffer is a zero-copy view over the compact C-double array.
    return float(np.percentile(np.frombuffer(values, dtype=np.float64), fraction * 100.0))


def _time_weighted(values: Sequence[float | None], durations: Sequence[float], *,
                   rms: bool = False) -> float | None:
    pairs = [(value, durations[index]) for index, item in enumerate(values)
             if (value := _finite(item)) is not None and durations[index] > 0.0]
    total = math.fsum(duration for _, duration in pairs)
    if total <= 0.0:
        return None
    moment = math.fsum((value * value if rms else value) * duration
                       for value, duration in pairs) / total
    return math.sqrt(moment) if rms else moment


def _durations(trace: Mapping[str, Any]) -> list[float]:
    """Return the duration owned by each sample, matching the training reducer."""
    times = [float(value) for value in trace["time_s"]]
    if "sample_interval_s" in trace:
        return [float(value) for value in trace["sample_interval_s"]]
    return [0.0, *(right - left for left, right in zip(times, times[1:]))]


def _numeric_trace(trace: Mapping[str, Any], key: str, *, nullable: bool = False) \
        -> list[float | None]:
    raw = trace[key]
    result: list[float | None] = []
    for index, item in enumerate(raw):
        value = _finite(item)
        if value is None and not nullable:
            raise ValueError(f"trace.{key}[{index}] 必须是有限数值")
        result.append(value)
    return result


def _enrich_episode_row(row: dict[str, Any], trace: Mapping[str, Any],
                        durations: Sequence[float]) -> tuple[
                            dict[str, list[float]], dict[str, dict[int, float]]
                        ]:
    """Add symmetric actual/commanded-load metrics and make one-bin episode means."""
    speed = [float(value) for value in _numeric_trace(trace, "speed_mps")]
    estimated_nullable = _numeric_trace(trace, "estimated_actual_load_g", nullable=True)
    commanded_nullable = _numeric_trace(trace, "commanded_load_g", nullable=True)
    estimated = [float(value) for value in estimated_nullable if value is not None]
    commanded = [float(value) for value in commanded_nullable if value is not None]

    row.update({
        "estimated_load_min_g": min(estimated) if estimated else None,
        "estimated_load_p05_g": _percentile(estimated, .05),
        "commanded_load_min_g": min(commanded) if commanded else None,
        "commanded_load_p05_g": _percentile(commanded, .05),
        "commanded_load_p50_g": _percentile(commanded, .50),
        "commanded_load_rms_g": _time_weighted(commanded_nullable, durations, rms=True),
    })
    commanded_duration = float(row["commanded_load_valid_duration_s"])
    for threshold in (3, 6, 9):
        duration = math.fsum(
            durations[index] for index, value in enumerate(commanded_nullable)
            if value is not None and value > threshold
        )
        row[f"commanded_load_above_{threshold}g_s"] = duration
        row[f"commanded_load_above_{threshold}g_fraction"] = (
            duration / commanded_duration if commanded_duration > 0.0 else None
        )

    samples = {
        "speed": speed,
        "estimated_load": estimated,
        "commanded_load": commanded,
    }
    # Filled by the caller because the bin width is user configurable.
    return samples, {}


def _episode_bin_means(trace: Mapping[str, Any], time_bin_s: float) \
        -> dict[str, dict[int, float]]:
    times = [float(value) for value in trace["time_s"]]
    relative = [max(0.0, value - times[0]) for value in times]
    values = {
        "speed": _numeric_trace(trace, "speed_mps"),
        "estimated_load": _numeric_trace(trace, "estimated_actual_load_g", nullable=True),
        "commanded_load": _numeric_trace(trace, "commanded_load_g", nullable=True),
    }
    result: dict[str, dict[int, float]] = {}
    for metric, series in values.items():
        buckets: dict[int, list[float]] = defaultdict(list)
        for elapsed, value in zip(relative, series):
            if value is None:
                continue
            index = int(math.floor(elapsed / time_bin_s))
            buckets[index].append(float(value))
        result[metric] = {
            index: math.fsum(bucket) / len(bucket) for index, bucket in buckets.items()
        }
    return result


class _GroupAccumulator:
    """Compact exact accumulators for one overall or missile-count group."""

    def __init__(self, *, label: str, missile_count: int | None) -> None:
        self.label = label
        self.missile_count = missile_count
        self.rows: list[dict[str, Any]] = []
        self.samples = {name: array("d") for name, *_ in PROFILE_METRICS}
        self.profile: dict[str, dict[int, array]] = {
            name: defaultdict(lambda: array("d")) for name, *_ in PROFILE_METRICS
        }

    def add(self, row: dict[str, Any], samples: Mapping[str, Sequence[float]],
            profile: Mapping[str, Mapping[int, float]]) -> None:
        self.rows.append(row)
        for metric in self.samples:
            self.samples[metric].extend(samples[metric])
            for bin_index, value in profile[metric].items():
                self.profile[metric][bin_index].append(value)


def _combine_weighted(rows: Sequence[Mapping[str, Any]], value_key: str,
                      duration_key: str, *, rms: bool = False) -> float | None:
    pairs = []
    for row in rows:
        value = _finite(row.get(value_key))
        duration = _finite(row.get(duration_key))
        if value is not None and duration is not None and duration > 0.0:
            pairs.append((value, duration))
    total = math.fsum(duration for _, duration in pairs)
    if total <= 0.0:
        return None
    moment = math.fsum((value * value if rms else value) * duration
                       for value, duration in pairs) / total
    return math.sqrt(moment) if rms else moment


def _summary_for(group: _GroupAccumulator) -> dict[str, Any]:
    rows = group.rows
    survived = [bool(row["blue_survived"]) for row in rows
                if isinstance(row.get("blue_survived"), bool)]
    result: dict[str, Any] = {
        "group": group.label,
        "missile_count": group.missile_count,
        "episode_count": len(rows),
        "survival_rate": sum(survived) / len(survived) if survived else None,
    }
    definitions = (
        ("speed", "speed", "speed_valid_duration_s", "speed_mean_mps",
         "speed_rms_mps", "speed_max_mps", "mps"),
        ("estimated_load", "estimated_load", "estimated_load_valid_duration_s",
         "estimated_load_mean_g", "estimated_load_rms_g", "estimated_load_max_g", "g"),
        ("commanded_load", "commanded_load", "commanded_load_valid_duration_s",
         "commanded_load_mean_g", "commanded_load_rms_g", "commanded_load_max_g", "g"),
    )
    for prefix, sample_key, duration_key, mean_key, rms_key, peak_key, unit in definitions:
        duration = math.fsum(max(float(row.get(duration_key) or 0.0), 0.0) for row in rows)
        peaks = [float(value) for row in rows
                 if (value := _finite(row.get(peak_key))) is not None]
        result.update({
            f"{prefix}_sample_count": len(group.samples[sample_key]),
            f"{prefix}_valid_duration_s": duration,
            f"{prefix}_time_weighted_mean_{unit}": _combine_weighted(
                rows, mean_key, duration_key),
            f"{prefix}_time_weighted_rms_{unit}": _combine_weighted(
                rows, rms_key, duration_key, rms=True),
            f"{prefix}_sample_p05_{unit}": _buffer_percentile(
                group.samples[sample_key], .05),
            f"{prefix}_sample_p50_{unit}": _buffer_percentile(
                group.samples[sample_key], .50),
            f"{prefix}_sample_p95_{unit}": _buffer_percentile(
                group.samples[sample_key], .95),
            f"{prefix}_global_peak_{unit}": max(peaks) if peaks else None,
            f"episode_{prefix}_peak_mean_{unit}": (
                math.fsum(peaks) / len(peaks) if peaks else None),
            f"episode_{prefix}_peak_p95_{unit}": _percentile(peaks, .95),
        })
    for prefix in ("estimated_load", "commanded_load"):
        duration_key = f"{prefix}_valid_duration_s"
        valid_duration = float(result[duration_key])
        for threshold in (3, 6, 9):
            key = f"{prefix}_above_{threshold}g_s"
            duration = math.fsum(float(row.get(key) or 0.0) for row in rows)
            result[key] = duration
            result[f"{prefix}_above_{threshold}g_fraction"] = (
                duration / valid_duration if valid_duration > 0.0 else None
            )
            result[f"episode_{prefix}_above_{threshold}g_mean_s"] = (
                duration / len(rows) if rows else None
            )
    return result


def _profile_rows(groups: Sequence[_GroupAccumulator], time_bin_s: float) \
        -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for group in groups:
        bin_indices = sorted({index for metric in group.profile.values() for index in metric})
        for index in bin_indices:
            row: dict[str, Any] = {
                "group": group.label,
                "missile_count": group.missile_count,
                "time_bin_index": index,
                "time_bin_start_s": index * time_bin_s,
                "time_bin_end_s": (index + 1) * time_bin_s,
                "time_bin_center_s": (index + .5) * time_bin_s,
            }
            for metric, _, unit, _ in PROFILE_METRICS:
                values = group.profile[metric].get(index, array("d"))
                clean = list(values)
                row[f"{metric}_episode_count"] = len(clean)
                row[f"{metric}_mean_{unit}"] = (
                    math.fsum(clean) / len(clean) if clean else None)
                row[f"{metric}_p05_{unit}"] = _percentile(clean, .05)
                row[f"{metric}_p50_{unit}"] = _percentile(clean, .50)
                row[f"{metric}_p95_{unit}"] = _percentile(clean, .95)
            output.append(row)
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_profiles(profile_rows: Sequence[Mapping[str, Any]], groups: Sequence[_GroupAccumulator],
                   output: Path, *, dpi: int, font_path: str | Path | None) -> list[str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    def series(group: _GroupAccumulator, metric: str, statistic: str, unit: str) \
            -> tuple[np.ndarray, np.ndarray]:
        selected = [row for row in profile_rows if row["group"] == group.label]
        return (np.asarray([row["time_bin_center_s"] for row in selected], dtype=float),
                np.asarray([np.nan if row[f"{metric}_{statistic}_{unit}"] is None
                            else row[f"{metric}_{statistic}_{unit}"] for row in selected], dtype=float))

    count = len(groups)
    rows_count = int(math.ceil(count / 2))
    with chinese_plot_context(font_path):
        speed_figure = Figure(figsize=(13, 4.3 * rows_count), layout="constrained")
        FigureCanvasAgg(speed_figure)
        speed_axes = np.atleast_1d(speed_figure.subplots(rows_count, 2)).reshape(-1)
        for axis, group in zip(speed_axes, groups):
            x, mean = series(group, "speed", "mean", "mps")
            _, p05 = series(group, "speed", "p05", "mps")
            _, p50 = series(group, "speed", "p50", "mps")
            _, p95 = series(group, "speed", "p95", "mps")
            axis.fill_between(x, p05, p95, color="#93c5fd", alpha=.45, label="P05–P95")
            axis.plot(x, mean, color="#1d4ed8", lw=1.8, label="回合均值")
            axis.plot(x, p50, color="#1e3a8a", lw=1.2, ls="--", label="回合中位数")
            axis.set(title=group.label, xlabel="相对首样本时间（秒）", ylabel="总速度（米/秒）")
            axis.grid(True, alpha=.22); axis.legend(fontsize=8)
        for axis in speed_axes[count:]:
            axis.set_visible(False)
        speed_figure.suptitle("蓝方测试速度时间剖面（每个时间箱内回合等权）", fontsize=15)
        speed_figure.savefig(output / SPEED_PNG, dpi=dpi)
        speed_figure.clear()

        load_figure = Figure(figsize=(13, 4.3 * rows_count), layout="constrained")
        FigureCanvasAgg(load_figure)
        load_axes = np.atleast_1d(load_figure.subplots(rows_count, 2)).reshape(-1)
        colors = {"estimated_load": ("#15803d", "#86efac"),
                  "commanded_load": ("#c2410c", "#fdba74")}
        labels = {"estimated_load": "估算实际过载", "commanded_load": "名义指令过载"}
        for axis, group in zip(load_axes, groups):
            for metric in ("estimated_load", "commanded_load"):
                x, mean = series(group, metric, "mean", "g")
                _, p05 = series(group, metric, "p05", "g")
                _, p95 = series(group, metric, "p95", "g")
                line, shade = colors[metric]
                axis.fill_between(x, p05, p95, color=shade, alpha=.20)
                axis.plot(x, mean, color=line, lw=1.7, label=labels[metric])
            for threshold in (3, 6, 9):
                axis.axhline(threshold, color="#64748b", lw=.7, ls=":", alpha=.55)
            axis.set(title=group.label, xlabel="相对首样本时间（秒）", ylabel="过载（g）")
            axis.set_ylim(bottom=0.0); axis.grid(True, alpha=.22); axis.legend(fontsize=8)
        for axis in load_axes[count:]:
            axis.set_visible(False)
        load_figure.suptitle("蓝方测试过载时间剖面（实线为回合均值，淡色为P05–P95）", fontsize=15)
        load_figure.savefig(output / LOAD_PNG, dpi=dpi)
        load_figure.clear()
    return [SPEED_PNG, LOAD_PNG]


def stream_blue_evaluation_dynamics(
    evaluation_dir: str | Path,
    *,
    time_bin_s: float = 1.0,
) -> tuple[list[dict[str, Any]], list[_GroupAccumulator], list[dict[str, Any]], Path]:
    """Stream evaluation JSONL and return compact rows and accumulators."""
    if not math.isfinite(time_bin_s) or time_bin_s <= 0.0:
        raise ValueError("time_bin_s must be a positive finite number")
    root = Path(evaluation_dir).resolve()
    source = root / "flight_quality" / "flight_quality_episodes.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"未找到测试轨迹文件：{source}")

    overall = _GroupAccumulator(label="全部来弹数", missile_count=None)
    by_count: dict[int, _GroupAccumulator] = {}
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    with source.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"轨迹 JSONL 第 {line_number} 行无效：{error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"轨迹 JSONL 第 {line_number} 行不是对象")
            metadata = record.get("metadata")
            if not isinstance(metadata, dict):
                raise ValueError(f"轨迹 JSONL 第 {line_number} 行缺少 metadata 对象")
            missile_count = metadata.get("missile_count")
            if (isinstance(missile_count, bool) or not isinstance(missile_count, int)
                    or missile_count < 1):
                raise ValueError(
                    f"轨迹 JSONL 第 {line_number} 行 metadata.missile_count 必须为正整数"
                )
            episode = record.get("episode")
            if (isinstance(episode, bool) or not isinstance(episode, int) or episode < 1):
                raise ValueError(f"轨迹 JSONL 第 {line_number} 行 episode 必须为正整数")
            if episode in seen:
                raise ValueError(f"轨迹 JSONL 中存在重复回合编号 {episode}")
            seen.add(episode)

            row = summarize_training_episode(record, metadata, len(rows) + 1)
            trace = record["trace"]
            durations = _durations(trace)
            samples, _ = _enrich_episode_row(row, trace, durations)
            profile = _episode_bin_means(trace, time_bin_s)
            rows.append(row)
            group = by_count.setdefault(
                missile_count,
                _GroupAccumulator(label=f"{missile_count} 枚来弹", missile_count=missile_count),
            )
            overall.add(row, samples, profile)
            group.add(row, samples, profile)
    if not rows:
        raise ValueError(f"测试轨迹文件没有完整回合记录：{source}")
    groups = [overall, *(by_count[count] for count in sorted(by_count))]
    return rows, groups, _profile_rows(groups, time_bin_s), source


def analyze_blue_evaluation_dynamics(
    evaluation_dir: str | Path,
    output: str | Path | None = None,
    *,
    time_bin_s: float = 1.0,
    dpi: int = 180,
    font_path: str | Path | None = None,
    make_plots: bool = True,
) -> dict[str, Any]:
    """Write all-test and per-missile-count speed/load monitoring artifacts."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    root = Path(evaluation_dir).resolve()
    destination = (Path(output).resolve() if output is not None
                   else root / "evaluation_dynamics")
    rows, groups, profiles, source = stream_blue_evaluation_dynamics(
        root, time_bin_s=time_bin_s
    )
    destination.mkdir(parents=True, exist_ok=True)
    summaries = [_summary_for(group) for group in groups]

    _write_csv(destination / EPISODE_CSV, rows)
    _write_csv(destination / SUMMARY_CSV, summaries)
    _write_csv(destination / PROFILE_CSV, profiles)
    summary_document = {
        "schema_version": 1,
        "overall": summaries[0],
        "by_missile_count": {
            str(item["missile_count"]): item for item in summaries[1:]
        },
    }
    (destination / SUMMARY_JSON).write_text(
        json.dumps(summary_document, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if make_plots:
        figures = _plot_profiles(
            profiles, groups, destination, dpi=dpi, font_path=font_path
        )
    else:
        # A reused output directory must describe this run, not retain figures
        # from an earlier invocation that did request plots.
        for name in (SPEED_PNG, LOAD_PNG):
            path = destination / name
            if path.is_file():
                path.unlink()
        figures = []
    manifest = {
        "schema_version": 1,
        "evaluation_dir": str(root),
        "trace_source": str(source),
        "trace_source_size_bytes": source.stat().st_size,
        "output_dir": str(destination),
        "episode_count": len(rows),
        "missile_counts": [group.missile_count for group in groups[1:]],
        "episodes_by_missile_count": {
            str(group.missile_count): len(group.rows) for group in groups[1:]
        },
        "time_bin_s": time_bin_s,
        "tables": [EPISODE_CSV, SUMMARY_CSV, PROFILE_CSV],
        "documents": [SUMMARY_JSON],
        "figures": figures,
        "streaming": True,
        "definitions": {
            "time_weighted_statistics": (
                "均值和RMS按轨迹记录的sample_interval_s加权；旧记录缺少该字段时，"
                "使用[0, *相邻time_s差值]。"
            ),
            "sample_percentiles": "P05/P50/P95对组内全部有效采样值计算，不做时间加权。",
            "episode_peak_statistics": "先求每回合峰值，再对回合峰值等权计算均值和P95。",
            "time_profiles": (
                "以每回合首样本为0秒；先在每个时间箱内求该回合的采样均值，再在回合间"
                "等权计算mean/P05/P50/P95。回合结束后的时间箱不补值，*_episode_count给出"
                "该箱实际贡献回合数。"
            ),
            "estimated_actual_load": (
                "由FlightQualityTracker基于相邻决策级速度差分并计入重力估算，不是每个"
                "物理积分步的结构过载实测值。"
            ),
            "commanded_load": "约束后离散执行动作对应的名义合成过载。",
            "threshold_duration": "超过3g/6g/9g均使用严格大于阈值并按有效采样时长累计。",
        },
    }
    (destination / MANIFEST_JSON).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return manifest


def safe_analyze_blue_evaluation_dynamics(
    evaluation_dir: str | Path,
    output: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Run the optional post-evaluation reporter without hiding the evaluation result."""
    try:
        return analyze_blue_evaluation_dynamics(evaluation_dir, output, **kwargs)
    except Exception as error:  # Evaluation results must survive optional reporting failures.
        print(
            f"蓝方测试速度/过载分析失败：{type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="流式统计蓝方测试全部回合及各来弹数的速度和过载"
    )
    parser.add_argument("--evaluation-dir", required=True, type=Path,
                        help="测试结果目录，其中应含 flight_quality/flight_quality_episodes.jsonl")
    parser.add_argument("--output", type=Path,
                        help="输出目录；默认写入测试目录下 evaluation_dynamics")
    parser.add_argument("--time-bin-s", type=float, default=1.0,
                        help="跨回合时间剖面箱宽（秒，默认1.0）")
    parser.add_argument("--dpi", type=int, default=180, help="PNG分辨率（默认180）")
    parser.add_argument("--font", type=Path, help="可选中文字体文件；默认自动探测")
    parser.add_argument("--no-plots", action="store_true", help="只写表格和JSON，不绘图")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = analyze_blue_evaluation_dynamics(
            args.evaluation_dir,
            args.output,
            time_bin_s=args.time_bin_s,
            dpi=args.dpi,
            font_path=args.font,
            make_plots=not args.no_plots,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps({
        "output": manifest["output_dir"],
        "episodes": manifest["episode_count"],
        "missile_counts": manifest["missile_counts"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
