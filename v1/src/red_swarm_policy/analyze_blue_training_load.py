"""Stream blue training traces and report how speed and load change in training.

The trajectory JSONL can be close to one gigabyte for a 2,000 episode run, so
this module intentionally reads and reduces one completed episode at a time.
It never loads a policy, changes a checkpoint, or rewrites source results.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from .plot_localization import chinese_plot_context
except ImportError:  # Support ``python analyze_blue_training_load.py``.
    from plot_localization import chinese_plot_context


EPISODE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("completion_order", "轨迹完成序号"),
    ("episode", "回合编号"),
    ("missile_count", "来弹数量"),
    ("blue_survived", "蓝方成功规避"),
    ("termination_reason", "终止原因"),
    ("simulation_time_s", "仿真时长（s）"),
    ("trace_elapsed_s", "轨迹覆盖时长（s）"),
    ("sample_count", "轨迹采样点数"),
    ("valid_speed_samples", "有效总速度样本数"),
    ("speed_valid_duration_s", "有效总速度时长（s）"),
    ("speed_mean_mps", "总速度时间加权均值（m/s）"),
    ("speed_p05_mps", "总速度样本P05（m/s）"),
    ("speed_p50_mps", "总速度样本中位数（m/s）"),
    ("speed_p95_mps", "总速度样本P95（m/s）"),
    ("speed_min_mps", "总速度最小值（m/s）"),
    ("speed_max_mps", "总速度最大值（m/s）"),
    ("speed_rms_mps", "总速度时间加权均方根（m/s）"),
    ("horizontal_speed_mean_mps", "水平速度时间加权均值（m/s）"),
    ("horizontal_speed_min_mps", "水平速度最小值（m/s）"),
    ("horizontal_speed_p05_mps", "水平速度样本P05（m/s）"),
    ("abs_vertical_speed_mean_mps", "垂直速度绝对值时间加权均值（m/s）"),
    ("abs_vertical_speed_p95_mps", "垂直速度绝对值样本P95（m/s）"),
    ("abs_vertical_speed_max_mps", "垂直速度绝对值最大值（m/s）"),
    ("valid_estimated_load_samples", "有效估算实际过载样本数"),
    ("estimated_load_valid_duration_s", "有效估算实际过载时长（s）"),
    ("estimated_load_mean_g", "估算实际过载时间加权均值（g）"),
    ("estimated_load_p50_g", "估算实际过载样本中位数（g）"),
    ("estimated_load_p95_g", "估算实际过载样本P95（g）"),
    ("estimated_load_max_g", "估算实际过载最大值（g）"),
    ("estimated_load_rms_g", "估算实际过载时间加权均方根（g）"),
    ("estimated_load_above_3g_s", "估算实际过载超过3g时长（s）"),
    ("estimated_load_above_3g_fraction", "估算实际过载超过3g时长比例"),
    ("estimated_load_above_6g_s", "估算实际过载超过6g时长（s）"),
    ("estimated_load_above_6g_fraction", "估算实际过载超过6g时长比例"),
    ("estimated_load_above_9g_s", "估算实际过载超过9g时长（s）"),
    ("estimated_load_above_9g_fraction", "估算实际过载超过9g时长比例"),
    ("valid_commanded_load_samples", "有效名义指令过载样本数"),
    ("commanded_load_valid_duration_s", "有效名义指令过载时长（s）"),
    ("commanded_load_mean_g", "名义指令过载时间加权均值（g）"),
    ("commanded_load_p95_g", "名义指令过载样本P95（g）"),
    ("commanded_load_max_g", "名义指令过载最大值（g）"),
    ("executed_action_count", "有效执行动作数"),
    ("safety_intervention_count", "安全约束介入次数"),
    ("safety_intervention_rate", "安全约束介入比例"),
    ("action_switch_count", "动作切换次数"),
    ("action_switch_rate_hz", "动作切换频率（Hz）"),
)

BLOCK_COLUMNS: tuple[tuple[str, str], ...] = (
    ("block", "分段编号"),
    ("completion_order_start", "起始轨迹完成序号"),
    ("completion_order_end", "结束轨迹完成序号"),
    ("episode_count", "回合数"),
    ("episode_min", "最小回合编号"),
    ("episode_max", "最大回合编号"),
    ("successful_evasion_count", "成功规避回合数"),
    ("survival_rate", "成功规避率"),
    ("valid_speed_samples", "有效总速度样本总数"),
    ("speed_valid_duration_s", "有效总速度总时长（s）"),
    ("speed_mean_mps", "合并总速度时间加权均值（m/s）"),
    ("episode_speed_p05_mean_mps", "逐回合总速度样本P05的平均值（m/s）"),
    ("episode_speed_p50_mean_mps", "逐回合总速度样本中位数的平均值（m/s）"),
    ("episode_speed_p95_mean_mps", "逐回合总速度样本P95的平均值（m/s）"),
    ("speed_min_mps", "分段总速度最小值（m/s）"),
    ("speed_max_mps", "分段总速度最大值（m/s）"),
    ("speed_rms_mps", "合并总速度时间加权均方根（m/s）"),
    ("horizontal_speed_mean_mps", "合并水平速度时间加权均值（m/s）"),
    ("horizontal_speed_min_mps", "分段水平速度最小值（m/s）"),
    ("episode_horizontal_speed_p05_mean_mps", "逐回合水平速度样本P05的平均值（m/s）"),
    ("abs_vertical_speed_mean_mps", "合并垂直速度绝对值时间加权均值（m/s）"),
    ("episode_abs_vertical_speed_p95_mean_mps", "逐回合垂直速度绝对值样本P95的平均值（m/s）"),
    ("abs_vertical_speed_max_mps", "分段垂直速度绝对值最大值（m/s）"),
    ("estimated_load_mean_g", "合并估算实际过载时间加权均值（g）"),
    ("episode_p50_mean_g", "逐回合估算实际过载样本中位数的平均值（g）"),
    ("episode_p95_mean_g", "逐回合估算实际过载样本P95的平均值（g）"),
    ("estimated_load_max_g", "分段估算实际过载最大值（g）"),
    ("estimated_load_rms_g", "合并估算实际过载时间加权均方根（g）"),
    ("estimated_load_above_3g_s", "估算实际过载超过3g总时长（s）"),
    ("estimated_load_above_3g_fraction", "估算实际过载超过3g总时长比例"),
    ("estimated_load_above_6g_s", "估算实际过载超过6g总时长（s）"),
    ("estimated_load_above_6g_fraction", "估算实际过载超过6g总时长比例"),
    ("estimated_load_above_9g_s", "估算实际过载超过9g总时长（s）"),
    ("estimated_load_above_9g_fraction", "估算实际过载超过9g总时长比例"),
    ("commanded_load_mean_g", "合并名义指令过载时间加权均值（g）"),
    ("episode_commanded_p95_mean_g", "逐回合名义指令过载样本P95的平均值（g）"),
    ("commanded_load_max_g", "分段名义指令过载最大值（g）"),
    ("safety_intervention_count", "安全约束介入总次数"),
    ("safety_intervention_rate", "安全约束介入比例"),
    ("action_switch_count", "动作切换总次数"),
    ("action_switch_rate_hz", "动作切换频率（Hz）"),
)

PLOTS: tuple[tuple[str, str, str, str], ...] = (
    ("speed_mean_mps", "训练过程中蓝方总速度时间加权均值", "总速度时间加权均值（m/s）",
     "speed_mean.png"),
    ("speed_min_mps", "训练过程中蓝方最低总速度", "总速度最小值（m/s）",
     "speed_min.png"),
    ("speed_max_mps", "训练过程中蓝方最高总速度", "总速度最大值（m/s）",
     "speed_max.png"),
    ("horizontal_speed_min_mps", "训练过程中蓝方最低水平速度", "水平速度最小值（m/s）",
     "horizontal_speed_min.png"),
    ("abs_vertical_speed_p95_mps", "训练过程中蓝方垂直速度绝对值样本P95",
     "垂直速度绝对值样本P95（m/s）", "abs_vertical_speed_p95.png"),
    ("estimated_load_max_g", "训练过程中蓝方最大估算实际过载", "估算实际过载最大值（g）",
     "estimated_load_max.png"),
    ("estimated_load_p95_g", "训练过程中蓝方估算实际过载样本P95", "估算实际过载样本P95（g）",
     "estimated_load_p95.png"),
    ("estimated_load_mean_g", "训练过程中蓝方估算实际过载时间加权均值", "估算实际过载时间加权均值（g）",
     "estimated_load_mean.png"),
    ("estimated_load_above_6g_s", "训练过程中蓝方超过6g的持续时间", "超过6g时长（s）",
     "estimated_load_above_6g_duration.png"),
    ("commanded_load_max_g", "训练过程中蓝方最大名义指令过载", "名义指令过载最大值（g）",
     "commanded_load_max.png"),
    ("commanded_load_mean_g", "训练过程中蓝方名义指令过载时间加权均值", "名义指令过载时间加权均值（g）",
     "commanded_load_mean.png"),
)

_METADATA_CONFLICT_FIELDS = (
    "missile_count", "blue_survived", "simulation_time_s", "termination_reason",
)
_BLUE_ACTION_COUNT = 29


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _episode_id(value: Any, source: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{source} 中的回合编号必须为正整数")
    return value


def _percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = (len(ordered) - 1) * fraction
    left = math.floor(location)
    right = math.ceil(location)
    if left == right:
        return ordered[left]
    weight = location - left
    return ordered[left] * (1.0 - weight) + ordered[right] * weight


def _mean(values: Iterable[float | None]) -> float | None:
    finite = [value for item in values if (value := _finite(item)) is not None]
    return math.fsum(finite) / len(finite) if finite else None


def _load_episode_metadata(training_dir: Path) -> tuple[dict[int, dict[str, Any]], list[str]]:
    """Join standard summary files by episode without assuming list order."""
    indexed: dict[int, dict[str, Any]] = {}
    sources: list[str] = []
    for name in ("training_metrics.json", "episodes.json"):
        path = training_dir / name
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig") as stream:
            document = json.load(stream)
        rows = document.get("episodes") if isinstance(document, dict) else document
        if not isinstance(rows, list):
            raise ValueError(f"{path} 中没有可用的 episodes 数组")
        seen_here: set[int] = set()
        for index, item in enumerate(rows, 1):
            if not isinstance(item, dict):
                raise ValueError(f"{path} 的第 {index} 个回合记录不是对象")
            episode = _episode_id(item.get("episode"), f"{path} 的第 {index} 条记录")
            if episode in seen_here:
                raise ValueError(f"{path} 中存在重复回合编号 {episode}")
            seen_here.add(episode)
            current = indexed.setdefault(episode, {})
            for key in _METADATA_CONFLICT_FIELDS:
                previous, incoming = current.get(key), item.get(key)
                if (key in current and key in item and previous is not None
                        and incoming is not None and previous != incoming):
                    raise ValueError(
                        f"{path} 的回合 {episode} 字段 {key} 与先前元数据冲突："
                        f"{previous!r} != {incoming!r}"
                    )
            # A missing/null field in the later optional source must not erase
            # a value already supplied by the primary training summary.
            current.update({key: value for key, value in item.items()
                            if value is not None or key not in current})
        sources.append(str(path.resolve()))
    return indexed, sources


def _durations(times: Sequence[float], sample_intervals: Any = None) -> list[float]:
    if len(times) < 2:
        differences: list[float] = []
    else:
        differences = [right - left for left, right in zip(times, times[1:])]
        if any(not math.isfinite(value) or value <= 0.0 for value in differences):
            raise ValueError("轨迹采样时间必须严格递增且为有限值")
    if sample_intervals is None:
        return [0.0, *differences] if times else []
    if not isinstance(sample_intervals, list) or len(sample_intervals) != len(times):
        raise ValueError("sample_interval_s 长度必须与 time_s 相同")
    intervals = [_finite(value) for value in sample_intervals]
    if any(value is None or value < 0.0 for value in intervals):
        raise ValueError("sample_interval_s 必须只包含非负有限数值")
    clean = [float(value) for value in intervals if value is not None]
    tolerance = 1.0e-9
    if clean and abs(clean[0]) > tolerance:
        raise ValueError("sample_interval_s 的首个值必须为 0")
    for index, (recorded, derived) in enumerate(zip(clean[1:], differences), 1):
        allowed = tolerance * max(1.0, abs(recorded), abs(derived))
        if abs(recorded - derived) > allowed:
            raise ValueError(
                f"sample_interval_s[{index}] 与相邻 time_s 的差值不一致"
            )
    return clean


def _sequence(trace: Mapping[str, Any], key: str, count: int, *,
              required: bool = False) -> list[Any]:
    value = trace.get(key)
    if value is None and not required:
        return [None] * count
    if not isinstance(value, list):
        requirement = "必须是数组" if required or key in trace else "必须是数组或缺省"
        raise ValueError(f"trace.{key} {requirement}")
    if len(value) != count:
        raise ValueError(f"trace.{key} 长度 {len(value)} 与 time_s 长度 {count} 不一致")
    return list(value)


def _strict_numeric_series(trace: Mapping[str, Any], key: str, count: int, *,
                           nonnegative: bool = False) -> list[float] | None:
    """Read an optional recorded series without silently repairing bad data."""
    if key not in trace:
        return None
    raw = _sequence(trace, key, count, required=True)
    result: list[float] = []
    for index, item in enumerate(raw):
        value = _finite(item)
        if value is None:
            raise ValueError(f"trace.{key}[{index}] 必须是有限数值")
        if nonnegative and value < 0.0:
            raise ValueError(f"trace.{key}[{index}] 必须是非负有限数值")
        result.append(value)
    return result


def _velocity_series(trace: Mapping[str, Any], count: int) -> list[tuple[float, float, float]]:
    raw = _sequence(trace, "velocity_mps", count, required=True)
    result: list[tuple[float, float, float]] = []
    for index, item in enumerate(raw):
        if not isinstance(item, list) or len(item) != 3:
            raise ValueError(f"trace.velocity_mps[{index}] 必须是长度为3的有限数值数组")
        x, y, z = (_finite(value) for value in item)
        if x is None or y is None or z is None:
            raise ValueError(f"trace.velocity_mps[{index}] 必须是长度为3的有限数值数组")
        result.append((x, y, z))
    return result


def _time_weighted_mean(values: Sequence[tuple[int, float]],
                        durations: Sequence[float]) -> float | None:
    total = math.fsum(durations[index] for index, _ in values)
    if total <= 0.0:
        return None
    return math.fsum(value * durations[index] for index, value in values) / total


def _time_weighted_rms(values: Sequence[tuple[int, float]],
                       durations: Sequence[float]) -> float | None:
    total = math.fsum(durations[index] for index, _ in values)
    if total <= 0.0:
        return None
    second_moment = math.fsum(
        value * value * durations[index] for index, value in values
    ) / total
    return math.sqrt(second_moment)


def summarize_training_episode(record: Mapping[str, Any], metadata: Mapping[str, Any],
                               completion_order: int) -> dict[str, Any]:
    """Reduce one full trace to the compact row used online and offline."""
    episode = _episode_id(record.get("episode"), f"轨迹完成序号 {completion_order}")
    trace = record.get("trace")
    if not isinstance(trace, dict):
        raise ValueError(f"回合 {episode} 缺少 trace 对象")
    raw_times = trace.get("time_s")
    if not isinstance(raw_times, list) or not raw_times:
        raise ValueError(f"回合 {episode} 缺少有效 time_s")
    times = [_finite(item) for item in raw_times]
    if any(item is None for item in times):
        raise ValueError(f"回合 {episode} 的 time_s 含有非有限值")
    clean_times = [float(item) for item in times if item is not None]
    count = len(clean_times)
    dt = _durations(clean_times, trace.get("sample_interval_s"))

    speed_values = _strict_numeric_series(
        trace, "speed_mps", count, nonnegative=True
    )
    horizontal_values = _strict_numeric_series(
        trace, "horizontal_speed_mps", count, nonnegative=True
    )
    vertical_values = _strict_numeric_series(trace, "vertical_speed_mps", count)
    if speed_values is None or horizontal_values is None or vertical_values is None:
        velocities = _velocity_series(trace, count)
        if speed_values is None:
            speed_values = [math.sqrt(x * x + y * y + z * z) for x, y, z in velocities]
        if horizontal_values is None:
            horizontal_values = [math.hypot(x, z) for x, _y, z in velocities]
        if vertical_values is None:
            vertical_values = [y for _x, y, _z in velocities]
    speed = list(enumerate(speed_values))
    horizontal_speed = list(enumerate(horizontal_values))
    abs_vertical_values = [abs(value) for value in vertical_values]
    abs_vertical_speed = list(enumerate(abs_vertical_values))
    speed_duration = math.fsum(dt)

    raw_estimated = _sequence(trace, "estimated_actual_load_g", count, required=True)
    estimated = [(index, value) for index, item in enumerate(raw_estimated)
                 if (value := _finite(item)) is not None]
    estimated_values = [value for _, value in estimated]
    valid_duration = math.fsum(dt[index] for index, _ in estimated)

    raw_commanded = _sequence(trace, "commanded_load_g", count)
    commanded = [(index, value) for index, item in enumerate(raw_commanded)
                 if (value := _finite(item)) is not None]
    commanded_values = [value for _, value in commanded]
    commanded_duration = math.fsum(dt[index] for index, _ in commanded)
    raw_actions = _sequence(trace, "executed_action", count)
    for index, action in enumerate(raw_actions):
        if action is None:
            continue
        if (isinstance(action, bool) or not isinstance(action, int)
                or not 0 <= action < _BLUE_ACTION_COUNT):
            raise ValueError(
                f"回合 {episode} 的 executed_action[{index}] 必须是 0–28 的整数或 null"
            )
    action_indices = [index for index, action in enumerate(raw_actions) if action is not None]
    actions = [int(raw_actions[index]) for index in action_indices]
    raw_safety = _sequence(trace, "safety_intervened", count)
    for index, value in enumerate(raw_safety):
        if value is not None and not isinstance(value, bool):
            raise ValueError(
                f"回合 {episode} 的 safety_intervened[{index}] 必须是布尔值或 null"
            )
    safety_count = sum(bool(raw_safety[index]) for index in action_indices)
    switches = sum(left != right for left, right in zip(actions, actions[1:]))
    elapsed = clean_times[-1] - clean_times[0] if count > 1 else 0.0

    result: dict[str, Any] = {
        "completion_order": completion_order,
        "episode": episode,
        "missile_count": metadata.get("missile_count"),
        "blue_survived": metadata.get("blue_survived", record.get("blue_survived")),
        "termination_reason": metadata.get("termination_reason"),
        "simulation_time_s": metadata.get("simulation_time_s", elapsed),
        "trace_elapsed_s": elapsed,
        "sample_count": count,
        "valid_speed_samples": len(speed_values),
        "speed_valid_duration_s": speed_duration,
        "speed_mean_mps": _time_weighted_mean(speed, dt),
        "speed_p05_mps": _percentile(speed_values, 0.05),
        "speed_p50_mps": _percentile(speed_values, 0.50),
        "speed_p95_mps": _percentile(speed_values, 0.95),
        "speed_min_mps": min(speed_values),
        "speed_max_mps": max(speed_values),
        "speed_rms_mps": _time_weighted_rms(speed, dt),
        "horizontal_speed_mean_mps": _time_weighted_mean(horizontal_speed, dt),
        "horizontal_speed_min_mps": min(horizontal_values),
        "horizontal_speed_p05_mps": _percentile(horizontal_values, 0.05),
        "abs_vertical_speed_mean_mps": _time_weighted_mean(abs_vertical_speed, dt),
        "abs_vertical_speed_p95_mps": _percentile(abs_vertical_values, 0.95),
        "abs_vertical_speed_max_mps": max(abs_vertical_values),
        "valid_estimated_load_samples": len(estimated_values),
        "estimated_load_valid_duration_s": valid_duration,
        "estimated_load_mean_g": _time_weighted_mean(estimated, dt),
        "estimated_load_p50_g": _percentile(estimated_values, 0.50),
        "estimated_load_p95_g": _percentile(estimated_values, 0.95),
        "estimated_load_max_g": max(estimated_values) if estimated_values else None,
        "estimated_load_rms_g": _time_weighted_rms(estimated, dt),
        "valid_commanded_load_samples": len(commanded_values),
        "commanded_load_valid_duration_s": commanded_duration,
        "commanded_load_mean_g": _time_weighted_mean(commanded, dt),
        "commanded_load_p95_g": _percentile(commanded_values, 0.95),
        "commanded_load_max_g": max(commanded_values) if commanded_values else None,
        "executed_action_count": len(actions),
        "safety_intervention_count": safety_count,
        "safety_intervention_rate": safety_count / len(actions) if actions else None,
        "action_switch_count": switches,
        "action_switch_rate_hz": switches / elapsed if elapsed > 0.0 else None,
    }
    for threshold in (3, 6, 9):
        duration = math.fsum(dt[index] for index, value in estimated if value > threshold)
        result[f"estimated_load_above_{threshold}g_s"] = duration
        result[f"estimated_load_above_{threshold}g_fraction"] = (
            duration / valid_duration if valid_duration > 0.0 else None
        )
    return result


def stream_training_load_rows(training_dir: str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Reduce the trajectory JSONL in completion order and join episode metadata."""
    root = Path(training_dir).resolve()
    source = root / "flight_quality" / "flight_quality_episodes.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"未找到训练轨迹文件：{source}")
    metadata, metadata_sources = _load_episode_metadata(root)
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    with source.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"轨迹 JSONL 第 {line_number} 行无效：{error.msg}") from error
            if not isinstance(record, dict):
                raise ValueError(f"轨迹 JSONL 第 {line_number} 行不是对象")
            episode = _episode_id(record.get("episode"), f"轨迹 JSONL 第 {line_number} 行")
            if episode in seen:
                raise ValueError(f"轨迹 JSONL 中存在重复回合编号 {episode}")
            seen.add(episode)
            rows.append(summarize_training_episode(
                record, metadata.get(episode, {}), len(rows) + 1
            ))
    if not rows:
        raise ValueError(f"训练轨迹文件没有完整回合记录：{source}")
    return rows, {
        "trace_source": str(source.resolve()),
        "metadata_sources": metadata_sources,
        "metadata_episode_count": len(metadata),
        "missing_metadata_episode_ids": [row["episode"] for row in rows
                                         if row["episode"] not in metadata],
        "metadata_without_trace_episode_ids": sorted(set(metadata) - seen),
    }


def _weighted(rows: Sequence[Mapping[str, Any]], value_key: str, weight_key: str) -> float | None:
    pairs: list[tuple[float, float]] = []
    for row in rows:
        value, weight = _finite(row.get(value_key)), _finite(row.get(weight_key))
        if value is not None and weight is not None and weight > 0.0:
            pairs.append((value, weight))
    total = math.fsum(weight for _, weight in pairs)
    return math.fsum(value * weight for value, weight in pairs) / total if total else None


def _combined_rms(rows: Sequence[Mapping[str, Any]], value_key: str,
                  duration_key: str) -> float | None:
    """Combine already reduced RMS values using their contributing duration."""
    pairs: list[tuple[float, float]] = []
    for row in rows:
        value = _finite(row.get(value_key))
        duration = _finite(row.get(duration_key))
        if value is not None and duration is not None and duration > 0.0:
            pairs.append((value, duration))
    total = math.fsum(duration for _, duration in pairs)
    if total <= 0.0:
        return None
    second_moment = math.fsum(value * value * duration for value, duration in pairs) / total
    return math.sqrt(second_moment)


def _extreme(rows: Sequence[Mapping[str, Any]], key: str, *, maximum: bool) -> float | None:
    values = [value for row in rows if (value := _finite(row.get(key))) is not None]
    if not values:
        return None
    return max(values) if maximum else min(values)


def summarize_blocks(rows: Sequence[Mapping[str, Any]], block_size: int) -> list[dict[str, Any]]:
    if block_size < 1:
        raise ValueError("block_size must be positive")
    result: list[dict[str, Any]] = []
    for offset in range(0, len(rows), block_size):
        block = list(rows[offset:offset + block_size])
        speed_duration = math.fsum(
            max(float(row.get("speed_valid_duration_s") or 0.0), 0.0) for row in block
        )
        valid_duration = math.fsum(float(row.get("estimated_load_valid_duration_s") or 0.0)
                                   for row in block)
        action_count = sum(int(row.get("executed_action_count") or 0) for row in block)
        elapsed = math.fsum(max(float(row.get("trace_elapsed_s") or 0.0), 0.0) for row in block)
        survived = [row.get("blue_survived") for row in block
                    if isinstance(row.get("blue_survived"), bool)]
        summary: dict[str, Any] = {
            "block": len(result) + 1,
            "completion_order_start": int(block[0]["completion_order"]),
            "completion_order_end": int(block[-1]["completion_order"]),
            "episode_count": len(block),
            "episode_min": min(int(row["episode"]) for row in block),
            "episode_max": max(int(row["episode"]) for row in block),
            "successful_evasion_count": sum(survived),
            "survival_rate": sum(survived) / len(survived) if survived else None,
            "valid_speed_samples": sum(
                int(row.get("valid_speed_samples") or 0) for row in block
            ),
            "speed_valid_duration_s": speed_duration,
            "speed_mean_mps": _weighted(
                block, "speed_mean_mps", "speed_valid_duration_s"),
            "episode_speed_p05_mean_mps": _mean(
                row.get("speed_p05_mps") for row in block),
            "episode_speed_p50_mean_mps": _mean(
                row.get("speed_p50_mps") for row in block),
            "episode_speed_p95_mean_mps": _mean(
                row.get("speed_p95_mps") for row in block),
            "speed_min_mps": _extreme(block, "speed_min_mps", maximum=False),
            "speed_max_mps": _extreme(block, "speed_max_mps", maximum=True),
            "speed_rms_mps": _combined_rms(
                block, "speed_rms_mps", "speed_valid_duration_s"),
            "horizontal_speed_mean_mps": _weighted(
                block, "horizontal_speed_mean_mps", "speed_valid_duration_s"),
            "horizontal_speed_min_mps": _extreme(
                block, "horizontal_speed_min_mps", maximum=False),
            "episode_horizontal_speed_p05_mean_mps": _mean(
                row.get("horizontal_speed_p05_mps") for row in block),
            "abs_vertical_speed_mean_mps": _weighted(
                block, "abs_vertical_speed_mean_mps", "speed_valid_duration_s"),
            "episode_abs_vertical_speed_p95_mean_mps": _mean(
                row.get("abs_vertical_speed_p95_mps") for row in block),
            "abs_vertical_speed_max_mps": _extreme(
                block, "abs_vertical_speed_max_mps", maximum=True),
            "estimated_load_mean_g": _weighted(
                block, "estimated_load_mean_g", "estimated_load_valid_duration_s"),
            "episode_p50_mean_g": _mean(row.get("estimated_load_p50_g") for row in block),
            "episode_p95_mean_g": _mean(row.get("estimated_load_p95_g") for row in block),
            "estimated_load_max_g": max(
                (float(value) for row in block
                 if (value := _finite(row.get("estimated_load_max_g"))) is not None), default=None),
            "commanded_load_mean_g": _weighted(
                block, "commanded_load_mean_g", "commanded_load_valid_duration_s"),
            "episode_commanded_p95_mean_g": _mean(
                row.get("commanded_load_p95_g") for row in block),
            "commanded_load_max_g": max(
                (float(value) for row in block
                 if (value := _finite(row.get("commanded_load_max_g"))) is not None), default=None),
            "safety_intervention_count": sum(
                int(row.get("safety_intervention_count") or 0) for row in block),
            "action_switch_count": sum(int(row.get("action_switch_count") or 0) for row in block),
        }
        # RMS cannot be averaged directly. Recombine the episode second
        # moments using only each metric's contributing duration.
        summary["estimated_load_rms_g"] = _combined_rms(
            block, "estimated_load_rms_g", "estimated_load_valid_duration_s"
        )
        for threshold in (3, 6, 9):
            duration = math.fsum(float(row.get(f"estimated_load_above_{threshold}g_s") or 0.0)
                                 for row in block)
            summary[f"estimated_load_above_{threshold}g_s"] = duration
            summary[f"estimated_load_above_{threshold}g_fraction"] = (
                duration / valid_duration if valid_duration > 0.0 else None
            )
        summary["safety_intervention_rate"] = (
            summary["safety_intervention_count"] / action_count if action_count else None
        )
        summary["action_switch_rate_hz"] = (
            summary["action_switch_count"] / elapsed if elapsed > 0.0 else None
        )
        result.append(summary)
    return result


def _csv_value(key: str, value: Any) -> Any:
    if value is None:
        return ""
    if key == "blue_survived" and isinstance(value, bool):
        return "是" if value else "否"
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]],
               columns: Sequence[tuple[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=[label for _, label in columns])
        writer.writeheader()
        for row in rows:
            writer.writerow({label: _csv_value(key, row.get(key)) for key, label in columns})


def _rolling_mean(values: Sequence[float | None], window: int) -> list[float | None]:
    recent: deque[float | None] = deque()
    total = 0.0
    count = 0
    result: list[float | None] = []
    for item in values:
        value = _finite(item)
        recent.append(value)
        if value is not None:
            total += value
            count += 1
        if len(recent) > window:
            old = recent.popleft()
            if old is not None:
                total -= old
                count -= 1
        result.append(total / count if count else None)
    return result


def _render_metric_plot(rows: Sequence[Mapping[str, Any]], key: str, title: str,
                        y_label: str, path: Path, *, rolling_window: int,
                        dpi: int, font_path: str | Path | None) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    palette = ("#2563eb", "#ea580c", "#16a34a", "#7c3aed", "#64748b")
    x = [int(row["completion_order"]) for row in rows]
    values = [_finite(row.get(key)) for row in rows]
    with chinese_plot_context(font_path):
        figure = Figure(figsize=(10.5, 5.8), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        try:
            groups = sorted({row.get("missile_count") for row in rows},
                            key=lambda value: (value is None, str(value)))
            for group_index, group in enumerate(groups):
                group_rows = [row for row in rows if row.get("missile_count") == group]
                group_x = [int(row["completion_order"]) for row in group_rows]
                group_values = [_finite(row.get(key)) for row in group_rows]
                points = [(position, value) for position, value in zip(group_x, group_values)
                          if value is not None]
                if not points:
                    continue
                label = "来弹数未知" if group is None else f"{group} 枚来弹"
                color = palette[group_index % len(palette)]
                axis.scatter([item[0] for item in points], [item[1] for item in points],
                             s=10, alpha=.28, color=color, linewidths=0,
                             label=f"{label}：逐回合原始值")
                smoothed = _rolling_mean(group_values, rolling_window)
                axis.plot(group_x, [math.nan if value is None else value for value in smoothed],
                          color=color, linewidth=2.0,
                          label=f"{label}：最近 {rolling_window} 个同类完成回合滚动均值")
            if not any(value is not None for value in values):
                axis.text(.5, .5, "没有可绘制的有效数据", ha="center", va="center",
                          transform=axis.transAxes)
            axis.set_title(title, fontsize=15, fontweight="bold")
            axis.set_xlabel("轨迹完成序号（并行环境中的回合完成次序）")
            axis.set_ylabel(y_label)
            axis.set_ylim(bottom=0.0)
            axis.grid(True, alpha=.22)
            axis.legend(loc="best", fontsize=8)
            figure.savefig(path, dpi=dpi)
        finally:
            figure.clear()


def write_blue_training_dynamics(
    rows: Sequence[Mapping[str, Any]],
    output: str | Path,
    *,
    rolling_window: int = 50,
    block_size: int = 100,
    dpi: int = 180,
    font_path: str | Path | None = None,
    training_dir: str | Path | None = None,
    source_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write tables, independent trend plots, and a manifest from compact rows.

    This is the shared output layer for both the offline JSONL scanner and a
    training process that already owns the reduced per-episode rows.
    """
    if rolling_window < 1:
        raise ValueError("rolling_window must be positive")
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if dpi < 1:
        raise ValueError("dpi must be positive")
    compact_rows = [dict(row) for row in rows]
    if not compact_rows:
        raise ValueError("rows must contain at least one completed episode")
    completion_orders: list[int] = []
    episodes: set[int] = set()
    for index, row in enumerate(compact_rows, 1):
        completion_order = row.get("completion_order")
        if (isinstance(completion_order, bool)
                or not isinstance(completion_order, int) or completion_order < 1):
            raise ValueError(f"rows[{index - 1}].completion_order must be a positive integer")
        episode = _episode_id(row.get("episode"), f"rows[{index - 1}]")
        if episode in episodes:
            raise ValueError(f"rows contains duplicate episode {episode}")
        episodes.add(episode)
        completion_orders.append(completion_order)
    if len(set(completion_orders)) != len(completion_orders):
        raise ValueError("rows contains duplicate completion_order values")
    if completion_orders != sorted(completion_orders):
        raise ValueError("rows must be ordered by completion_order")

    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    blocks = summarize_blocks(compact_rows, block_size)

    episode_csv = destination / "逐回合速度过载统计.csv"
    block_csv = destination / "分段速度过载统计.csv"
    _write_csv(episode_csv, compact_rows, EPISODE_COLUMNS)
    _write_csv(block_csv, blocks, BLOCK_COLUMNS)
    figures: list[str] = []
    for key, title, y_label, filename in PLOTS:
        _render_metric_plot(compact_rows, key, title, y_label, destination / filename,
                            rolling_window=rolling_window, dpi=dpi, font_path=font_path)
        figures.append(filename)

    sources = dict(source_metadata or {})
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "training_dir": str(Path(training_dir).resolve()) if training_dir is not None else None,
        "output_dir": str(destination),
        **sources,
        "episode_count": len(compact_rows),
        "block_count": len(blocks),
        "rolling_window": rolling_window,
        "block_size": block_size,
        "tables": [episode_csv.name, block_csv.name],
        "figures": figures,
        "episode_order": (
            "传入紧凑行的 completion_order 顺序；离线模式对应 "
            "flight_quality_episodes.jsonl 中的完成顺序，回合编号另列并按编号联结训练汇总"
        ),
        "notes": [
            "若轨迹记录了 speed_mps、horizontal_speed_mps 或 vertical_speed_mps，则逐项严格校验长度与有限值；缺失的速度序列由 velocity_mps=[北向速度, 垂直速度, 东向速度] 回算。",
            "速度均值和均方根按记录时长加权；P05、P50和P95按逐采样点计算，不做时间加权。水平速度为北向与东向分量的合速度，垂直速度统计采用绝对值。",
            "估算实际过载由相邻决策采样点的速度差分加重力得到，是采样诊断值，不是仿真器逐物理步直接测量的结构过载。",
            "名义指令过载是约束后执行离散动作对应指令的合成标量，不代表物理步之间插值后的瞬时过载。",
            "过载均值和均方根按有效记录时长加权，分段合并继续按有效时长加权；首个轨迹样本的持续时间为0，不在轨迹起点之前虚构时间。",
            "P50和P95保留逐采样点分位数口径，不做时间加权；表头使用“样本中位数/样本P95”与时间加权均值区分。",
            "超过阈值的时长优先按记录的 sample_interval_s 累计；旧轨迹没有该字段时使用 [0, *相邻time_s差值]。",
            "原始训练文件只读，所有表格、图片和本清单均写入单独输出目录。",
        ],
    }
    manifest_path = destination / "训练速度过载分析清单.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def safe_write_blue_training_dynamics(
    rows: Sequence[Mapping[str, Any]],
    output: str | Path,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Run the output layer without allowing reporting errors to stop training."""
    try:
        return write_blue_training_dynamics(rows, output, **kwargs)
    except Exception as error:  # Reporting must remain isolated from training.
        print(f"蓝方训练速度过载分析写出失败：{error}", file=sys.stderr)
        return None


def analyze_blue_training_load(training_dir: str | Path, output: str | Path | None = None,
                               *, rolling_window: int = 50, block_size: int = 100,
                               dpi: int = 180,
                               font_path: str | Path | None = None) -> dict[str, Any]:
    """Scan stored trajectories and create speed/load tables and Chinese plots."""
    root = Path(training_dir).resolve()
    destination = (Path(output).resolve() if output is not None
                   else root / "training_load_analysis")
    rows, sources = stream_training_load_rows(root)
    return write_blue_training_dynamics(
        rows,
        destination,
        rolling_window=rolling_window,
        block_size=block_size,
        dpi=dpi,
        font_path=font_path,
        training_dir=root,
        source_metadata=sources,
    )


def safe_analyze_blue_training_load(
    training_dir: str | Path,
    output: str | Path | None = None,
    **kwargs: Any,
) -> dict[str, Any] | None:
    """Run the complete offline report without propagating reporting errors."""
    try:
        return analyze_blue_training_load(training_dir, output, **kwargs)
    except Exception as error:  # Training completion must survive report failures.
        print(f"蓝方训练速度过载分析失败：{error}", file=sys.stderr)
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="流式统计蓝方训练逐回合速度和过载并生成中文分图")
    parser.add_argument("--training-dir", required=True, type=Path, help="训练结果目录")
    parser.add_argument("--output", type=Path, help="输出目录；默认写入训练目录下 training_load_analysis")
    parser.add_argument("--rolling-window", type=int, default=50, help="绘图滚动均值窗口（默认 50）")
    parser.add_argument("--block-size", type=int, default=100, help="分段统计回合数（默认 100）")
    parser.add_argument("--dpi", type=int, default=180, help="PNG 分辨率（默认 180）")
    parser.add_argument("--font", type=Path, help="可选中文字体文件；未指定时自动探测")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = analyze_blue_training_load(
        args.training_dir, args.output, rolling_window=args.rolling_window,
        block_size=args.block_size, dpi=args.dpi, font_path=args.font,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
