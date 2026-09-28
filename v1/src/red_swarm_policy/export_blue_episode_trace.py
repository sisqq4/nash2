"""Export one recorded blue-aircraft evaluation episode in detail.

The exporter reads ``flight_quality_episodes.jsonl`` as a stream.  It does not
load the much larger consolidated ``flight_quality.json`` file.  Recorded
coordinates use the project convention ``[x, y, z] = [north, up, east]``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from .plot_localization import chinese_plot_context


PLOT_FILENAMES = (
    "01_速度随时间.png",
    "02_速度分量随时间.png",
    "03_过载随时间.png",
    "04_XY平面轨迹.png",
    "05_YZ平面轨迹.png",
    "06_ZX平面轨迹.png",
    "07_位置分量随时间.png",
    "08_动作序号随时间.png",
    "09_动作指令分量随时间.png",
)

TRACE_CSV_NAME = "蓝方逐时刻数据.csv"
SEGMENT_CSV_NAME = "动作区间.csv"
ACTION_MAP_CSV_NAME = "动作映射.csv"
SUMMARY_JSON_NAME = "回合摘要.json"
SUMMARY_CSV_NAME = "回合摘要.csv"
_SAFETY_SPAN_ALPHA = 0.055

_EPISODE_PREFIX = re.compile(r'^\s*\{\s*"episode"\s*:\s*(\d+)')


def _action_table_sha256() -> str:
    """Return a stable fingerprint for the action table used during export."""
    encoded = json.dumps(
        np.asarray(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G, dtype=np.float64).tolist(),
        ensure_ascii=True, allow_nan=False, separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def _positive_dpi(value: str) -> int:
    parsed = int(value)
    if not 50 <= parsed <= 1200:
        raise argparse.ArgumentTypeError("DPI 必须在 50 到 1200 之间")
    return parsed


def resolve_jsonl(*, evaluation_dir: str | Path | None = None,
                  input_jsonl: str | Path | None = None) -> Path:
    """Resolve exactly one supported trace input."""
    if (evaluation_dir is None) == (input_jsonl is None):
        raise ValueError("必须且只能指定 evaluation_dir 或 input_jsonl 之一")
    if input_jsonl is not None:
        source = Path(input_jsonl).expanduser().resolve()
    else:
        root = Path(evaluation_dir).expanduser().resolve()  # type: ignore[arg-type]
        source = root / "flight_quality" / "flight_quality_episodes.jsonl"
    if not source.is_file():
        raise FileNotFoundError(f"找不到测试回合轨迹文件：{source}")
    return source


def find_episode(source: str | Path, *, episode: int | None = None,
                 missiles: int | None = None,
                 scenario_episode: int | None = None) -> dict[str, Any]:
    """Stream *source* and return one episode selected by either identity."""
    source_path = Path(source)
    by_global_id = episode is not None
    by_scenario = missiles is not None or scenario_episode is not None
    if by_global_id == by_scenario:
        raise ValueError("请使用 episode，或同时使用 missiles 和 scenario_episode")
    if by_global_id and (isinstance(episode, bool) or not isinstance(episode, int)
                         or episode <= 0):
        raise ValueError("episode 必须是正整数")
    if by_scenario:
        if (isinstance(missiles, bool) or not isinstance(missiles, int) or missiles <= 0
                or isinstance(scenario_episode, bool)
                or not isinstance(scenario_episode, int) or scenario_episode <= 0):
            raise ValueError("missiles 和 scenario_episode 必须同时为正整数")

    with source_path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            # Episode is deliberately the first field written by the recorder.
            # Skipping JSON decoding for non-matching IDs keeps late lookups in
            # multi-gigabyte files practical.
            if by_global_id:
                prefix = _EPISODE_PREFIX.match(line)
                if prefix is not None and int(prefix.group(1)) != episode:
                    continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"JSONL 第 {line_number} 行无效：{error.msg}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(f"JSONL 第 {line_number} 行必须是对象")
            if by_global_id:
                if record.get("episode") == episode:
                    return record
            else:
                metadata = record.get("metadata")
                if (isinstance(metadata, dict)
                        and metadata.get("missile_count") == missiles
                        and metadata.get("scenario_episode") == scenario_episode):
                    return record

    selector = (f"全局回合 {episode}" if by_global_id else
                f"{missiles} 枚红弹、场景内回合 {scenario_episode}")
    raise LookupError(f"在 {source_path} 中找不到{selector}")


def _required_series(trace: dict[str, Any], name: str, length: int) -> list[Any]:
    value = trace.get(name)
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"trace.{name} 必须是长度为 {length} 的数组")
    return value


def _vector_array(trace: dict[str, Any], name: str, length: int) -> np.ndarray:
    try:
        values = np.asarray(_required_series(trace, name, length), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"trace.{name} 必须只包含数值") from error
    if values.shape != (length, 3) or not np.all(np.isfinite(values)):
        raise ValueError(f"trace.{name} 必须是 {length}x3 的有限数值数组")
    return values


def _nullable_numeric(values: Sequence[Any], *, name: str) -> np.ndarray:
    converted: list[float] = []
    for index, value in enumerate(values):
        if value is None:
            converted.append(math.nan)
            continue
        if isinstance(value, bool):
            raise ValueError(f"{name}[{index}] 不是数值")
        try:
            number = float(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name}[{index}] 不是数值") from error
        converted.append(number if math.isfinite(number) else math.nan)
    return np.asarray(converted, dtype=np.float64)


def _action_values(values: Sequence[Any], *, name: str) -> list[int | None]:
    result: list[int | None] = []
    for index, value in enumerate(values):
        if value is None:
            result.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name}[{index}] 必须是动作整数或 null")
        if not 0 <= value < len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G):
            raise ValueError(f"{name}[{index}]={value} 超出动作范围 0–28")
        result.append(value)
    return result


def _prepare_red_tracks(episode: Mapping[str, Any], trace: Mapping[str, Any],
                        time: np.ndarray) -> list[dict[str, Any]]:
    """Return active red-missile tracks, including their exact terminal points."""
    raw_positions = trace.get("red_positions_m")
    if raw_positions is None:
        return []
    count = len(time)
    if not isinstance(raw_positions, list) or len(raw_positions) != count:
        raise ValueError(f"trace.red_positions_m 必须是长度为 {count} 的数组")
    if any(not isinstance(frame, list) for frame in raw_positions):
        raise ValueError("trace.red_positions_m 的每个采样必须是红弹位置数组")

    raw_ids = trace.get("red_ids")
    if raw_ids is None:
        slot_count = max((len(frame) for frame in raw_positions), default=0)
        red_ids = list(range(slot_count))
    else:
        if (not isinstance(raw_ids, list)
                or any(isinstance(value, bool) or not isinstance(value, int)
                       for value in raw_ids)
                or len(set(raw_ids)) != len(raw_ids)):
            raise ValueError("trace.red_ids 必须是不重复的整数数组")
        red_ids = list(raw_ids)
        slot_count = len(red_ids)
    if any(len(frame) != slot_count for frame in raw_positions):
        raise ValueError("trace.red_positions_m 的红弹槽位数必须与 trace.red_ids 一致且固定")

    positions = np.full((count, slot_count, 3), np.nan, dtype=np.float64)
    finite_position = np.zeros((count, slot_count), dtype=bool)
    for sample_index, frame in enumerate(raw_positions):
        for slot, value in enumerate(frame):
            if value is None:
                continue
            try:
                point = np.asarray(value, dtype=np.float64)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"trace.red_positions_m[{sample_index}][{slot}] 必须是三维数值位置或 null"
                ) from error
            if point.shape != (3,):
                raise ValueError(
                    f"trace.red_positions_m[{sample_index}][{slot}] 必须是三维数值位置或 null"
                )
            if np.all(np.isfinite(point)):
                positions[sample_index, slot] = point
                finite_position[sample_index, slot] = True

    def validity(name: str, *, allow_null: bool = False) -> np.ndarray | None:
        raw = trace.get(name)
        if raw is None:
            return None
        if not isinstance(raw, list) or len(raw) != count:
            raise ValueError(f"trace.{name} 必须是长度为 {count} 的数组")
        result = np.zeros((count, slot_count), dtype=bool)
        for sample_index, frame in enumerate(raw):
            if not isinstance(frame, list) or len(frame) != slot_count:
                raise ValueError(f"trace.{name} 的每个采样必须包含 {slot_count} 个布尔值")
            normalized: list[bool] = []
            for value in frame:
                if isinstance(value, bool):
                    normalized.append(value)
                elif allow_null and value is None:
                    normalized.append(False)
                else:
                    raise ValueError(
                        f"trace.{name} 的每个采样必须包含 {slot_count} 个"
                        + ("布尔值或 null" if allow_null else "布尔值")
                    )
            result[sample_index] = normalized
        return result

    state_valid = validity("red_state_valid")
    position_valid = validity("red_position_valid")
    alive = validity("red_alive", allow_null=True) if state_valid is None else None
    active = finite_position.copy()
    if position_valid is not None:
        active &= position_valid
    if state_valid is not None:
        active &= state_valid
    elif alive is not None:
        active &= alive
    positions[~active] = np.nan

    raw_events = episode.get("red_termination_events", [])
    if raw_events is None:
        raw_events = []
    if not isinstance(raw_events, list) or any(not isinstance(item, dict) for item in raw_events):
        raise ValueError("red_termination_events 必须是对象数组")
    events: dict[int, dict[str, Any]] = {}
    for event in raw_events:
        red_id = event.get("red_id")
        if isinstance(red_id, bool) or not isinstance(red_id, int) or red_id not in red_ids:
            raise ValueError("red_termination_events.red_id 必须匹配 trace.red_ids")
        if red_id in events:
            raise ValueError(f"红弹 {red_id} 存在重复终止事件")
        events[red_id] = event

    result: list[dict[str, Any]] = []
    for slot, red_id in enumerate(red_ids):
        track = positions[:, slot, :]
        event = events.get(red_id)
        if event is not None:
            event_time = event.get("time_s")
            if (isinstance(event_time, bool) or not isinstance(event_time, (int, float))
                    or not math.isfinite(float(event_time))):
                raise ValueError(f"红弹 {red_id} 的终止时间必须是有限数值")
            track = track[time < float(event_time)]
            if event.get("position_valid", False):
                terminal = np.asarray(event.get("position_m"), dtype=np.float64)
                if terminal.shape != (3,) or not np.all(np.isfinite(terminal)):
                    raise ValueError(f"红弹 {red_id} 的终止位置必须是有限三维数值")
                track = np.vstack((track, terminal))
        result.append({"red_id": red_id, "position": track})
    return result


def _prepare_trace(episode: dict[str, Any]) -> dict[str, Any]:
    trace = episode.get("trace")
    if not isinstance(trace, dict):
        raise ValueError("回合缺少 trace 对象")
    raw_time = trace.get("time_s")
    if not isinstance(raw_time, list) or not raw_time:
        raise ValueError("trace.time_s 必须是非空数组")
    time = _nullable_numeric(raw_time, name="trace.time_s")
    if not np.all(np.isfinite(time)) or np.any(np.diff(time) <= 0):
        raise ValueError("trace.time_s 必须是严格递增的有限数值")
    count = len(time)
    position = _vector_array(trace, "position_m", count)
    velocity = _vector_array(trace, "velocity_mps", count)
    red_tracks = _prepare_red_tracks(episode, trace, time)
    relative = time - time[0]

    sample_interval = trace.get("sample_interval_s")
    if sample_interval is None:
        dt = np.diff(time, prepend=time[0])
    else:
        dt = _nullable_numeric(
            _required_series(trace, "sample_interval_s", count),
            name="trace.sample_interval_s",
        )
        if not np.all(np.isfinite(dt)) or np.any(dt < 0):
            raise ValueError("trace.sample_interval_s 必须是非负有限数值")
        differences = np.diff(time)
        if (not math.isclose(float(dt[0]), 0.0, rel_tol=0.0, abs_tol=1.0e-9)
                or not np.allclose(dt[1:], differences, rtol=1.0e-7, atol=1.0e-9)):
            raise ValueError("trace.sample_interval_s 必须与相邻 time_s 的差一致，首项必须为 0")

    step_count = trace.get("step_count", [None] * count)
    if not isinstance(step_count, list) or len(step_count) != count:
        raise ValueError(f"trace.step_count 必须是长度为 {count} 的数组")

    speed = _nullable_numeric(trace.get("speed_mps", np.linalg.norm(velocity, axis=1)),
                              name="trace.speed_mps")
    horizontal = _nullable_numeric(
        trace.get("horizontal_speed_mps", np.linalg.norm(velocity[:, [0, 2]], axis=1)),
        name="trace.horizontal_speed_mps",
    )
    vertical = _nullable_numeric(trace.get("vertical_speed_mps", velocity[:, 1]),
                                 name="trace.vertical_speed_mps")
    fpa = _nullable_numeric(
        trace.get("flight_path_angle_deg", np.degrees(np.arctan2(vertical, horizontal))),
        name="trace.flight_path_angle_deg",
    )
    heading_rate = _nullable_numeric(
        trace.get("heading_rate_deg_s", [None] * count), name="trace.heading_rate_deg_s"
    )
    turn_radius = _nullable_numeric(
        trace.get("turn_radius_m", [None] * count), name="trace.turn_radius_m"
    )
    estimated_load = _nullable_numeric(
        _required_series(trace, "estimated_actual_load_g", count),
        name="trace.estimated_actual_load_g",
    )
    policy = _action_values(_required_series(trace, "policy_action", count),
                            name="trace.policy_action")
    executed = _action_values(_required_series(trace, "executed_action", count),
                              name="trace.executed_action")

    safety_raw = trace.get("safety_intervened", [False] * count)
    if not isinstance(safety_raw, list) or len(safety_raw) != count:
        raise ValueError(f"trace.safety_intervened 必须是长度为 {count} 的数组")
    if any(not isinstance(value, bool) for value in safety_raw):
        raise ValueError("trace.safety_intervened 必须只包含布尔值")
    safety = np.asarray(safety_raw, dtype=bool)
    reasons = trace.get("safety_reasons", [[] for _ in range(count)])
    if not isinstance(reasons, list) or len(reasons) != count:
        raise ValueError(f"trace.safety_reasons 必须是长度为 {count} 的数组")
    normalized_reasons: list[list[str]] = []
    for value in reasons:
        if value is None:
            normalized_reasons.append([])
        elif isinstance(value, list):
            normalized_reasons.append([str(item) for item in value])
        else:
            normalized_reasons.append([str(value)])

    commands = np.full((count, 3), np.nan, dtype=np.float64)
    reconstructed_nominal_load = np.full(count, np.nan, dtype=np.float64)
    for index, action in enumerate(executed):
        if action is None:
            continue
        command = BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[action]
        commands[index] = command
        reconstructed_nominal_load[index] = float(np.linalg.norm(command[:2]))

    recorded_raw = trace.get("commanded_load_g", [None] * count)
    if not isinstance(recorded_raw, list) or len(recorded_raw) != count:
        raise ValueError(f"trace.commanded_load_g 必须是长度为 {count} 的数组")
    recorded_nominal_load = _nullable_numeric(
        recorded_raw, name="trace.commanded_load_g"
    )
    # The recorded value is the historical source of truth.  Older traces may
    # omit it, in which case the display series falls back to the current table
    # while both source columns remain available in the CSV.
    nominal_load = np.where(
        np.isfinite(recorded_nominal_load), recorded_nominal_load,
        reconstructed_nominal_load,
    )

    for name, values in (("speed_mps", speed), ("horizontal_speed_mps", horizontal),
                         ("vertical_speed_mps", vertical), ("flight_path_angle_deg", fpa),
                         ("heading_rate_deg_s", heading_rate), ("turn_radius_m", turn_radius),
                         ("estimated_actual_load_g", estimated_load)):
        if len(values) != count:
            raise ValueError(f"trace.{name} 必须是长度为 {count} 的数组")

    return {
        "time": time, "relative": relative, "dt": dt, "step_count": step_count,
        "position": position, "velocity": velocity, "speed": speed,
        "horizontal_speed": horizontal, "vertical_speed": vertical, "fpa": fpa,
        "heading_rate": heading_rate, "turn_radius": turn_radius,
        "estimated_load": estimated_load, "policy": policy, "executed": executed,
        "safety": safety, "safety_reasons": normalized_reasons,
        "commands": commands, "nominal_load": nominal_load,
        "recorded_nominal_load": recorded_nominal_load,
        "reconstructed_nominal_load": reconstructed_nominal_load,
        "red_tracks": red_tracks,
    }


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "是" if value else "否"
    return value


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _trace_rows(data: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for index in range(len(data["time"])):
        position = data["position"][index]
        velocity = data["velocity"][index]
        command = data["commands"][index]
        yield {
            "绝对时间_s": data["time"][index], "相对时间_s": data["relative"][index],
            "物理步序号": data["step_count"][index], "采样间隔_s": data["dt"][index],
            "北向位置x_m": position[0], "高度y_m": position[1], "东向位置z_m": position[2],
            "北向速度vx_m_s": velocity[0], "垂直速度vy_m_s": velocity[1],
            "东向速度vz_m_s": velocity[2], "总速度_m_s": data["speed"][index],
            "水平速度_m_s": data["horizontal_speed"][index],
            "垂直速度_m_s": data["vertical_speed"][index],
            "航迹倾角_deg": data["fpa"][index],
            "航向角速率_deg_s": data["heading_rate"][index],
            "转弯半径_m": data["turn_radius"][index],
            "估算实际过载_g": data["estimated_load"][index],
            "名义执行动作过载_g": data["nominal_load"][index],
            "轨迹记录名义执行动作过载_g": data["recorded_nominal_load"][index],
            "当前动作表重建名义执行动作过载_g": data["reconstructed_nominal_load"][index],
            "策略动作序号": data["policy"][index], "执行动作序号": data["executed"][index],
            "执行动作轴向过载_g": command[0], "执行动作法向过载_g": command[1],
            "执行动作滚转角_deg": math.degrees(command[2]) if math.isfinite(command[2]) else math.nan,
            "安全约束介入": bool(data["safety"][index]),
            "安全约束原因": "; ".join(data["safety_reasons"][index]),
        }


TRACE_FIELDS = (
    "绝对时间_s", "相对时间_s", "物理步序号", "采样间隔_s",
    "北向位置x_m", "高度y_m", "东向位置z_m",
    "北向速度vx_m_s", "垂直速度vy_m_s", "东向速度vz_m_s",
    "总速度_m_s", "水平速度_m_s", "垂直速度_m_s",
    "航迹倾角_deg", "航向角速率_deg_s", "转弯半径_m",
    "估算实际过载_g", "名义执行动作过载_g",
    "轨迹记录名义执行动作过载_g", "当前动作表重建名义执行动作过载_g",
    "策略动作序号", "执行动作序号", "执行动作轴向过载_g",
    "执行动作法向过载_g", "执行动作滚转角_deg",
    "安全约束介入", "安全约束原因",
)


def _action_segments(data: dict[str, Any]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    count = len(data["time"])
    if count == 0:
        return segments
    start = 0
    keys = list(zip(data["policy"], data["executed"]))
    for stop in range(1, count + 1):
        if stop < count and keys[stop] == keys[start]:
            continue
        indices = range(start, stop)
        executed = data["executed"][start]
        command = (BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[executed]
                   if executed is not None else np.full(3, np.nan))
        interval_start = float(data["relative"][start] - data["dt"][start])
        interval_end = float(data["relative"][stop - 1])
        reasons = sorted({reason for index in indices
                          for reason in data["safety_reasons"][index]})
        segments.append({
            "区间序号": len(segments) + 1,
            "开始相对时间_s": max(0.0, interval_start),
            "结束相对时间_s": interval_end,
            "持续时间_s": max(0.0, interval_end - max(0.0, interval_start)),
            "样本数": stop - start,
            "策略动作序号": data["policy"][start], "执行动作序号": executed,
            "轴向过载_g": command[0], "法向过载_g": command[1],
            "滚转角_deg": math.degrees(command[2]) if math.isfinite(command[2]) else math.nan,
            "名义合成过载_g": (float(np.linalg.norm(command[:2]))
                              if np.all(np.isfinite(command[:2])) else math.nan),
            "安全约束介入次数": int(sum(bool(data["safety"][index]) for index in indices)),
            "安全约束原因": "; ".join(reasons),
        })
        start = stop
    return segments


SEGMENT_FIELDS = (
    "区间序号", "开始相对时间_s", "结束相对时间_s", "持续时间_s", "样本数",
    "策略动作序号", "执行动作序号", "轴向过载_g", "法向过载_g", "滚转角_deg",
    "名义合成过载_g", "安全约束介入次数", "安全约束原因",
)


def _action_map_rows() -> Iterable[dict[str, Any]]:
    for index, command in enumerate(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G):
        yield {
            "动作序号": index, "轴向过载_g": command[0], "法向过载_g": command[1],
            "滚转角_rad": command[2], "滚转角_deg": math.degrees(command[2]),
            "名义合成过载_g": float(np.linalg.norm(command[:2])),
        }


ACTION_MAP_FIELDS = (
    "动作序号", "轴向过载_g", "法向过载_g", "滚转角_rad", "滚转角_deg", "名义合成过载_g",
)


def _safety_index_runs(data: dict[str, Any]) -> list[tuple[int, int]]:
    """Return inclusive sample-index bounds for consecutive interventions."""
    active = np.asarray(data["safety"], dtype=bool)
    indices = np.flatnonzero(active)
    if indices.size == 0:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1) + 1
    groups = np.split(indices, splits)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _safety_spans(data: dict[str, Any]) -> list[tuple[float, float]]:
    spans: list[tuple[float, float]] = []
    for first, last in _safety_index_runs(data):
        start = max(0.0, float(data["relative"][first] - data["dt"][first]))
        end = float(data["relative"][last])
        spans.append((start, end))
    return spans


def _interval_step_data(data: dict[str, Any], values: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return interval values and edges for end-of-decision samples.

    Sample ``i`` describes the command applied over
    ``[relative[i] - dt[i], relative[i]]``.  The initial state normally has a
    zero interval and is omitted, preventing a one-decision shift to the right.
    """
    converted = np.asarray(values, dtype=np.float64)
    if converted.shape != data["relative"].shape:
        raise ValueError("阶梯图数值长度必须与时间轴一致")
    positive = np.flatnonzero(np.asarray(data["dt"]) > 1.0e-12)
    if positive.size == 0:
        return np.asarray([0.0, 0.0]), np.asarray([math.nan])
    first, last = int(positive[0]), int(positive[-1])
    if not np.array_equal(positive, np.arange(first, last + 1)):
        raise ValueError("正采样区间必须连续")
    edges = np.concatenate((
        [max(0.0, float(data["relative"][first] - data["dt"][first]))],
        np.asarray(data["relative"][first:last + 1], dtype=np.float64),
    ))
    return edges, converted[first:last + 1]


def _decorate_time_axis(axis: Any, data: dict[str, Any], *,
                        show_safety_interventions: bool = False) -> None:
    if show_safety_interventions:
        for index, (start, end) in enumerate(_safety_spans(data)):
            axis.axvspan(start, end, color="crimson", alpha=_SAFETY_SPAN_ALPHA,
                        label="安全约束介入" if index == 0 else None)
    axis.set_xlabel("相对首样本时间（秒）")
    axis.grid(True, alpha=.25)


def _save_figure(path: Path, *, title: str, xlabel: str, ylabel: str,
                 font_path: str | Path | None, dpi: int,
                 draw: Any, equal: bool = False) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.text import Text

    with chinese_plot_context(font_path) as font_name:
        figure = Figure(figsize=(10, 6), layout="constrained")
        FigureCanvasAgg(figure)
        axis = figure.add_subplot(1, 1, 1)
        draw(axis, figure)
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=.25)
        if equal:
            axis.set_aspect("equal", adjustable="datalim")
        for text in figure.findobj(match=Text):
            text.set_fontfamily(font_name)
        figure.savefig(path, dpi=dpi)
        figure.clear()


def _plot_time_series(path: Path, data: dict[str, Any], *, title: str, ylabel: str,
                      lines: Sequence[tuple[Sequence[float], str, str]],
                      font_path: str | Path | None, dpi: int, step: bool = False,
                      action_axis: bool = False,
                      show_safety_interventions: bool = False) -> None:
    def draw(axis: Any, _figure: Any) -> None:
        for values, label, color in lines:
            if step:
                edges, interval_values = _interval_step_data(data, values)
                axis.stairs(interval_values, edges, baseline=None, label=label, color=color)
            else:
                axis.plot(data["relative"], values, label=label, color=color)
        _decorate_time_axis(
            axis, data, show_safety_interventions=show_safety_interventions
        )
        if action_axis:
            from matplotlib.ticker import MaxNLocator

            axis.set_ylim(-0.5, len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G) - 0.5)
            axis.yaxis.set_major_locator(MaxNLocator(integer=True, steps=[1, 2, 5, 10]))
        axis.legend()

    _save_figure(path, title=title, xlabel="相对首样本时间（秒）", ylabel=ylabel,
                 font_path=font_path, dpi=dpi, draw=draw)


def _plot_projection(path: Path, data: dict[str, Any], *, first: int, second: int,
                     title: str, xlabel: str, ylabel: str,
                     font_path: str | Path | None, dpi: int, equal: bool,
                     show_safety_interventions: bool = False) -> None:
    position = data["position"]

    def draw(axis: Any, _figure: Any) -> None:
        axis.plot(position[:, first], position[:, second], color="royalblue", label="蓝方轨迹")
        for index, red_track in enumerate(data.get("red_tracks", [])):
            red_position = np.asarray(red_track["position"], dtype=np.float64)
            axis.plot(
                red_position[:, first], red_position[:, second], color="firebrick", alpha=.6,
                label="红方导弹轨迹" if index == 0 else None,
            )
        if show_safety_interventions:
            safety_runs = _safety_index_runs(data)
            if safety_runs:
                starts = np.asarray([start for start, _ in safety_runs], dtype=np.int64)
                ends = np.asarray([end for _, end in safety_runs], dtype=np.int64)
                axis.scatter(position[starts, first], position[starts, second], s=34,
                             color="crimson", marker="^", label="安全介入区间起点", zorder=3)
                axis.scatter(position[ends, first], position[ends, second], s=24,
                             facecolors="white", edgecolors="crimson", linewidths=1.2,
                             marker="o", label="安全介入区间终点", zorder=3)
        axis.scatter(position[0, first], position[0, second], color="green", s=55,
                     label="起点", zorder=4)
        axis.scatter(position[-1, first], position[-1, second], color="black", marker="x",
                     s=65, label="终点", zorder=4)
        axis.legend()

    _save_figure(path, title=title, xlabel=xlabel, ylabel=ylabel,
                 font_path=font_path, dpi=dpi, draw=draw, equal=equal)


def _write_plots(output: Path, data: dict[str, Any], *, title_context: str,
                 font_path: str | Path | None, dpi: int,
                 show_safety_interventions: bool = False) -> None:
    colors = ("royalblue", "darkorange", "seagreen")

    def titled(value: str) -> str:
        return f"{value}{title_context}"
    _plot_time_series(
        output / PLOT_FILENAMES[0], data, title=titled("蓝方速度随时间变化"), ylabel="速度（米/秒）",
        lines=((data["speed"], "总速度", colors[0]),
               (data["horizontal_speed"], "水平速度", colors[1]),
               (np.abs(data["vertical_speed"]), "垂直速度绝对值", colors[2])),
        font_path=font_path, dpi=dpi,
        show_safety_interventions=show_safety_interventions,
    )
    _plot_time_series(
        output / PLOT_FILENAMES[1], data, title=titled("蓝方速度分量随时间变化"), ylabel="速度分量（米/秒）",
        lines=((data["velocity"][:, 0], "北向速度 vx", colors[0]),
               (data["velocity"][:, 1], "垂直速度 vy", colors[1]),
               (data["velocity"][:, 2], "东向速度 vz", colors[2])),
        font_path=font_path, dpi=dpi,
        show_safety_interventions=show_safety_interventions,
    )
    _plot_time_series(
        output / PLOT_FILENAMES[2], data, title=titled("蓝方过载随时间变化"), ylabel="过载（g）",
        lines=((data["estimated_load"], "估算实际过载", colors[0]),
               (data["nominal_load"], "名义执行动作过载", colors[1])),
        font_path=font_path, dpi=dpi,
        show_safety_interventions=show_safety_interventions,
    )
    _plot_projection(output / PLOT_FILENAMES[3], data, first=0, second=1,
                     title=titled("蓝方与红弹 X-Y 平面轨迹"), xlabel="北向 x（米）", ylabel="高度 y（米）",
                     font_path=font_path, dpi=dpi, equal=True,
                     show_safety_interventions=show_safety_interventions)
    _plot_projection(output / PLOT_FILENAMES[4], data, first=1, second=2,
                     title=titled("蓝方与红弹 Y-Z 平面轨迹"), xlabel="高度 y（米）", ylabel="东向 z（米）",
                     font_path=font_path, dpi=dpi, equal=True,
                     show_safety_interventions=show_safety_interventions)
    _plot_projection(output / PLOT_FILENAMES[5], data, first=2, second=0,
                     title=titled("蓝方与红弹 Z-X 水平面轨迹"), xlabel="东向 z（米）", ylabel="北向 x（米）",
                     font_path=font_path, dpi=dpi, equal=True,
                     show_safety_interventions=show_safety_interventions)
    _plot_time_series(
        output / PLOT_FILENAMES[6], data, title=titled("蓝方位置分量随时间变化"), ylabel="位置（米）",
        lines=((data["position"][:, 0], "北向位置 x", colors[0]),
               (data["position"][:, 1], "高度 y", colors[1]),
               (data["position"][:, 2], "东向位置 z", colors[2])),
        font_path=font_path, dpi=dpi,
        show_safety_interventions=show_safety_interventions,
    )
    policy = [math.nan if value is None else value for value in data["policy"]]
    executed = [math.nan if value is None else value for value in data["executed"]]
    _plot_time_series(
        output / PLOT_FILENAMES[7], data, title=titled("蓝方动作序号随时间变化"), ylabel="动作序号",
        lines=((policy, "策略动作", colors[0]), (executed, "执行动作", colors[1])),
        font_path=font_path, dpi=dpi, step=True, action_axis=True,
        show_safety_interventions=show_safety_interventions,
    )

    def draw_commands(axis: Any, figure: Any) -> None:
        edges, axial = _interval_step_data(data, data["commands"][:, 0])
        _, normal = _interval_step_data(data, data["commands"][:, 1])
        _, bank = _interval_step_data(data, np.degrees(data["commands"][:, 2]))
        axis.stairs(axial, edges, baseline=None, label="轴向过载指令", color=colors[0])
        axis.stairs(normal, edges, baseline=None, label="法向过载指令", color=colors[1])
        bank_axis = axis.twinx()
        bank_axis.stairs(bank, edges, baseline=None, label="滚转角指令", color=colors[2])
        bank_axis.set_ylabel("滚转角（度）")
        _decorate_time_axis(
            axis, data, show_safety_interventions=show_safety_interventions
        )
        handles, labels = axis.get_legend_handles_labels()
        extra_handles, extra_labels = bank_axis.get_legend_handles_labels()
        axis.legend(handles + extra_handles, labels + extra_labels)

    _save_figure(output / PLOT_FILENAMES[8], title=titled("蓝方动作指令分量随时间变化"),
                 xlabel="相对首样本时间（秒）", ylabel="过载指令（g）",
                 font_path=font_path, dpi=dpi, draw=draw_commands)


def _flatten(prefix: str, value: Any) -> Iterable[tuple[str, Any]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from _flatten(child_prefix, child)
    elif isinstance(value, list):
        yield prefix, json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    else:
        yield prefix, value


def _title_context(episode: Mapping[str, Any]) -> str:
    metadata = episode.get("metadata") if isinstance(episode.get("metadata"), dict) else {}
    parts: list[str] = []
    missiles = metadata.get("missile_count")
    if isinstance(missiles, int) and not isinstance(missiles, bool) and missiles > 0:
        parts.append(f"{missiles}v1")
    episode_id = episode.get("episode")
    if isinstance(episode_id, int) and not isinstance(episode_id, bool) and episode_id > 0:
        parts.append(f"全局回合 {episode_id}")
    scenario_episode = metadata.get("scenario_episode")
    if (isinstance(scenario_episode, int) and not isinstance(scenario_episode, bool)
            and scenario_episode > 0):
        parts.append(f"场景内回合 {scenario_episode}")
    return f"（{'，'.join(parts)}）" if parts else ""


def _action_table_validation(data: Mapping[str, Any]) -> dict[str, Any]:
    recorded = np.asarray(data["recorded_nominal_load"], dtype=np.float64)
    rebuilt = np.asarray(data["reconstructed_nominal_load"], dtype=np.float64)
    comparable = np.isfinite(recorded) & np.isfinite(rebuilt)
    differences = np.abs(recorded[comparable] - rebuilt[comparable])
    mismatch = differences > 1.0e-9
    return {
        "当前动作表SHA256": _action_table_sha256(),
        "指纹算法": "SHA-256(JSON(float64动作表))",
        "轨迹记录标量样本数": int(np.isfinite(recorded).sum()),
        "当前动作表可重建样本数": int(np.isfinite(rebuilt).sum()),
        "可比较样本数": int(comparable.sum()),
        "不一致样本数": int(mismatch.sum()),
        "最大绝对差_g": float(differences.max()) if differences.size else None,
        "一致性容差_g": 1.0e-9,
    }


def export_episode_trace(source: str | Path, output: str | Path, *,
                         episode: int | None = None,
                         missiles: int | None = None,
                         scenario_episode: int | None = None,
                         font_path: str | Path | None = None,
                         dpi: int = 180,
                         show_safety_interventions: bool = False) -> dict[str, Any]:
    """Export CSV, JSON, and nine standalone Chinese plots for one episode."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or not 50 <= dpi <= 1200:
        raise ValueError("dpi 必须是 50 到 1200 之间的整数")
    if not isinstance(show_safety_interventions, bool):
        raise TypeError("show_safety_interventions 必须是布尔值")
    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    selected = find_episode(source_path, episode=episode, missiles=missiles,
                            scenario_episode=scenario_episode)
    data = _prepare_trace(selected)
    episode_id = selected.get("episode")
    if isinstance(episode_id, bool) or not isinstance(episode_id, int) or episode_id <= 0:
        raise ValueError("所选记录的 episode 必须是正整数")
    destination = Path(output).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    metadata = selected.get("metadata") if isinstance(selected.get("metadata"), dict) else {}

    _write_csv(destination / TRACE_CSV_NAME, TRACE_FIELDS, _trace_rows(data))
    segments = _action_segments(data)
    _write_csv(destination / SEGMENT_CSV_NAME, SEGMENT_FIELDS, segments)
    _write_csv(destination / ACTION_MAP_CSV_NAME, ACTION_MAP_FIELDS, _action_map_rows())
    _write_plots(destination, data, title_context=_title_context(selected),
                 font_path=font_path, dpi=dpi,
                 show_safety_interventions=show_safety_interventions)

    summary: dict[str, Any] = {
        "全局回合序号": episode_id,
        "红弹数量": metadata.get("missile_count"),
        "场景内回合序号": metadata.get("scenario_episode"),
        "蓝方存活": selected.get("blue_survived"),
        "样本数": len(data["time"]),
        "绝对起始时间_s": float(data["time"][0]),
        "绝对结束时间_s": float(data["time"][-1]),
        "记录时长_s": float(data["relative"][-1]),
        "红弹轨迹数": len(data["red_tracks"]),
        "图中标注安全介入": show_safety_interventions,
        "坐标约定": {"坐标系": "惯性坐标系", "数组顺序": ["x北向", "y向上（高度）", "z东向"]},
        "数据口径说明": {
            "估算实际过载": (
                "estimated_actual_load_g 由相邻决策级速度样本差分并计入重力得到；"
                "它不是每个物理积分步的传感器实测过载。"
            ),
            "名义执行动作过载": (
                "优先采用轨迹中记录的 commanded_load_g；另用导出时当前代码的离散动作表重建标量和"
                "动作分量并单独列出。环境在一个决策区间内对动作指令进行插值，旧记录无法恢复各物理步"
                "的完整插值指令；若动作表指纹不同，应以轨迹记录标量为历史来源。"
            ),
            "采样": "首样本及每个完成的环境决策步；相对时间以所选轨迹首样本为零点。",
        },
        "来源JSONL": str(source_path),
        "当前动作表校验": _action_table_validation(data),
        "元数据": metadata,
        "飞行质量指标": selected.get("metrics", {}),
        "飞行质量判定": selected.get("verdicts", {}),
        "事件": selected.get("events", []),
        "输出文件": [TRACE_CSV_NAME, SEGMENT_CSV_NAME, ACTION_MAP_CSV_NAME,
                     SUMMARY_JSON_NAME, SUMMARY_CSV_NAME, *PLOT_FILENAMES],
    }
    (destination / SUMMARY_JSON_NAME).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    _write_csv(destination / SUMMARY_CSV_NAME, ("项目", "值"),
               ({"项目": key, "值": value} for key, value in _flatten("", summary)))
    return {"episode": episode_id, "source": str(source_path), "output": str(destination),
            "samples": len(data["time"]), "action_segments": len(segments),
            "red_tracks": len(data["red_tracks"]),
            "show_safety_interventions": show_safety_interventions,
            "files": summary["输出文件"]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从测试结果中导出某一回合蓝机的完整时序表和独立中文图片。"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--evaluation-dir", type=Path,
                        help="测试输出目录，其中应含 flight_quality/flight_quality_episodes.jsonl")
    source.add_argument("--input-jsonl", type=Path, help="直接指定 flight_quality_episodes.jsonl")
    selector = parser.add_mutually_exclusive_group(required=True)
    selector.add_argument("--episode", type=_positive_integer, help="全局回合序号")
    selector.add_argument("--scenario-episode", type=_positive_integer,
                          help="某一红弹数量场景内的回合序号；必须同时指定 --missiles")
    parser.add_argument("--missiles", type=_positive_integer, help="红弹数量")
    parser.add_argument("--output", type=Path, help="输出目录；默认写到测试目录的 episode_details 下")
    parser.add_argument("--font", type=Path, help="可选中文字体文件")
    parser.add_argument("--dpi", type=_positive_dpi, default=180, help="PNG 分辨率，默认 180")
    parser.add_argument(
        "--show-safety-interventions", action="store_true",
        help="在时序图和轨迹图中显式标出安全介入；默认不标注",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.scenario_episode is not None and args.missiles is None:
        parser.error("使用 --scenario-episode 时必须同时指定 --missiles")
    if args.episode is not None and args.missiles is not None:
        parser.error("--missiles 只能与 --scenario-episode 一起使用")
    source = resolve_jsonl(evaluation_dir=args.evaluation_dir, input_jsonl=args.input_jsonl)
    if args.output is not None:
        output = args.output
    else:
        root = args.evaluation_dir if args.evaluation_dir is not None else source.parent
        selector = (f"episode_{args.episode:06d}" if args.episode is not None else
                    f"missiles_{args.missiles}_scenario_{args.scenario_episode:06d}")
        output = Path(root) / "episode_details" / selector
    result = export_episode_trace(
        source, output, episode=args.episode, missiles=args.missiles,
        scenario_episode=args.scenario_episode, font_path=args.font, dpi=args.dpi,
        show_safety_interventions=args.show_safety_interventions,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
