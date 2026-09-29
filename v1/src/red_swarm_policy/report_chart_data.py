"""Read existing Blue report inputs; never load a policy or run an environment.

``read_run(root)`` returns the original training, evaluation and evaluation
metadata dictionaries, with ``source`` paths added. Training initial headings
are joined from the SHA-256-bound episode sidecar when available.

``load_trace_summary(root, evaluation)`` streams one evaluation JSONL record at
a time. Its output has ``episodes`` (one row per matched episode), ``peak``
(the largest recorded difference-derived load), ``peak_window`` (every recorded
point within one second of that peak), ``global``, ``missing`` and ``warnings``.
Episode IDs, not JSONL order, join the trace with ``evaluation['results']``.
Conflicting run IDs, duplicate episode IDs and contradictory episode metadata
raise ValueError instead of combining different experiments. Missing optional
values remain None; missing traces are explicitly listed.

The recorded ``estimated_actual_load_g`` uses velocity differences plus gravity;
it is a sampled diagnostic, not an independently measured physical load.
``commanded_load_g`` is the nominal resultant of the executed action, not the
interpolated command at every physics step. Boundary contact is at the recorded
hard altitude bounds (tolerance 1e-6 m), distinct from a near-boundary verdict.
No training trajectory file or expanded ``flight_quality.json`` is read.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterator


def _read_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8-sig") as stream:
        result = json.load(stream)
    if not isinstance(result, dict):
        raise ValueError(f"JSON 顶层应为对象：{path}")
    return result


def _records(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8-sig") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL 记录损坏：{path}:{number}：{error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL 记录应为对象：{path}:{number}")
            yield number, row


def _episode_id(row: dict[str, Any], source: Any) -> int:
    value = row.get("episode")
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"回合编号应为正整数：{source}：{value!r}")
    return value


def _index(rows: Any, source: Any) -> dict[int, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError(f"逐回合记录应为数组：{source}")
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"逐回合记录应为对象：{source}")
        episode = _episode_id(row, source)
        if episode in result:
            raise ValueError(f"回合编号重复：{source}：{episode}")
        result[episode] = row
    return result


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _extreme(values: list[Any], maximum: bool = False) -> float | None:
    valid = [number for item in values if (number := _finite(item)) is not None]
    return (max(valid) if maximum else min(valid)) if valid else None


def read_run(root: str | Path, *, training_dir: str | Path | None = None,
             evaluation_dir: str | Path | None = None
             ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Read one training directory and one evaluation directory without changing them.

    Return the three original JSON objects (training_metrics.json,
    evaluation.json and evaluation_metadata.json). Each has an absolute
    ``source`` string. Training keeps ``experiment_config`` and all other
    original keys; ``blue_orientation`` may be added to individual episodes
    only from a verified metadata sidecar. An unavailable optional metadata
    file yields a dictionary with ``source`` and ``warnings``.

    ``training_dir`` and ``evaluation_dir`` allow a newly generated evaluation
    to be combined with the training artifacts that produced its checkpoint.
    When omitted, the historical ``root/train`` and ``root/evaluation`` layout
    is used.
    """
    root = Path(root).resolve()
    training_root = (Path(training_dir).resolve() if training_dir is not None
                     else root / "train")
    evaluation_root = (Path(evaluation_dir).resolve() if evaluation_dir is not None
                       else root / "evaluation")
    training_path = training_root / "training_metrics.json"
    if training_path.is_file():
        training = _read_object(training_path)
    else:
        training_path = training_root / "training.jsonl"
        training = {"iterations": [], "episodes": [], "curriculum_evaluations": []}
        for _, record in _records(training_path):
            event = record.get("event")
            if event == "experiment_config":
                training["experiment_config"] = record
            elif event == "iteration":
                training["iterations"].append(record)
            elif event == "episode":
                training["episodes"].append(record)
            elif event == "curriculum_evaluation":
                training["curriculum_evaluations"].append(record)
        episode_path = training_root / "episodes.json"
        if not training["episodes"] and episode_path.is_file():
            with episode_path.open(encoding="utf-8-sig") as stream:
                episodes = json.load(stream)
            training["episodes"] = episodes if isinstance(episodes, list) else episodes.get("episodes", [])
        training["notes"] = ["最终训练汇总不存在，已读取流式训练日志；未补造缺失的逐回合值。"]
    training["source"] = str(training_path)
    episodes = _index(training.get("episodes", []), training_path)
    training["episodes"] = [episodes[key] for key in sorted(episodes)]

    sidecar_path = training_path.with_name(training_path.stem + "_episode_metadata.json")
    if sidecar_path.is_file():
        sidecar = _read_object(sidecar_path)
        digest = hashlib.sha256(training_path.read_bytes()).hexdigest()
        if sidecar.get("metrics_sha256") != digest:
            raise ValueError(f"训练朝向元数据与训练汇总的 SHA-256 不一致：{sidecar_path}")
        for episode, entry in _index(sidecar.get("episodes"), sidecar_path).items():
            if episode not in episodes:
                raise ValueError(f"训练朝向元数据包含未知回合：{episode}")
            orientation = entry.get("blue_orientation")
            if not isinstance(orientation, str) or not orientation.strip():
                raise ValueError(f"训练朝向元数据缺少有效朝向：{episode}")
            existing = episodes[episode].get("blue_orientation")
            if existing is not None and existing != orientation:
                raise ValueError(f"训练朝向元数据与逐回合朝向冲突：{episode}")
            episodes[episode]["blue_orientation"] = orientation
        training["episode_metadata_source"] = str(sidecar_path)

    evaluation_path = evaluation_root / "evaluation.json"
    evaluation = _read_object(evaluation_path)
    evaluation["source"] = str(evaluation_path)
    _index(evaluation.get("results"), evaluation_path)
    metadata_path = evaluation_root / "flight_quality" / "evaluation_metadata.json"
    metadata = _read_object(metadata_path) if metadata_path.is_file() else {
        "warnings": ["评估元数据文件缺失，无法核对环境配置和记录运行编号。"]}
    metadata["source"] = str(metadata_path)
    if evaluation.get("run_id") and metadata.get("run_id") and evaluation["run_id"] != metadata["run_id"]:
        raise ValueError("评估汇总与评估元数据的运行编号不一致。")
    return training, evaluation, metadata


def _sample(trace: dict[str, Any], field: str, index: int) -> Any:
    values = trace.get(field)
    return values[index] if isinstance(values, list) and 0 <= index < len(values) else None


def _component(value: Any, index: int) -> float | None:
    return _finite(value[index]) if isinstance(value, list) and len(value) > index else None


def _contact(extreme: float | None, bound: float | None, missing: int, *, upper: bool) -> bool | None:
    if extreme is None or bound is None:
        return None
    touched = extreme >= bound - 1e-6 if upper else extreme <= bound + 1e-6
    return True if touched else (None if missing else False)


def _point(trace: dict[str, Any], episode: int, index: int) -> dict[str, Any]:
    position = _sample(trace, "position_m", index)
    velocity = _sample(trace, "velocity_mps", index)
    time = _finite(_sample(trace, "time_s", index))
    previous_time = _finite(_sample(trace, "time_s", index - 1))
    interval = time - previous_time if time is not None and previous_time is not None else None
    vertical = _finite(_sample(trace, "vertical_speed_mps", index))
    return {"episode": episode, "sample_index": index, "time_s": time,
            "altitude_m": _component(position, 1),
            "vertical_speed_mps": vertical if vertical is not None else _component(velocity, 1),
            "commanded_load_g": _finite(_sample(trace, "commanded_load_g", index)),
            "estimated_actual_load_g": _finite(_sample(trace, "estimated_actual_load_g", index)),
            "speed_mps": _finite(_sample(trace, "speed_mps", index)),
            "position_m": position, "velocity_mps": velocity,
            "step_count": _sample(trace, "step_count", index), "interval_s": interval,
            "sample_interval_s": _finite(_sample(trace, "sample_interval_s", index)),
            "policy_action": _sample(trace, "policy_action", index),
            "executed_action": _sample(trace, "executed_action", index),
            "safety_intervened": _sample(trace, "safety_intervened", index)}


def load_trace_summary(root: str | Path, evaluation: dict[str, Any], *,
                       evaluation_dir: str | Path | None = None) -> dict[str, Any]:
    """Stream evaluated trajectories and return exact extrema and peak context.

    ``episodes`` rows contain: episode, samples, min/max_altitude_m,
    min/max_speed_mps, min/max_commanded_load_g, min/max_estimated_actual_load_g,
    above_20g_samples, lower/upper_boundary_contact, start_s, end_s,
    missing_fields and run_id. Unknown contact is None, never False.
    ``peak`` includes previous/current altitude and vertical speed and actual
    elapsed interval. ``peak_window`` preserves every recorded point within
    ±1 s inclusive, without smoothing, resampling or a full-episode case plot.
    """
    root = Path(root).resolve()
    evaluation_root = (Path(evaluation_dir).resolve() if evaluation_dir is not None
                       else root / "evaluation")
    source = evaluation_root / "flight_quality" / "flight_quality_episodes.jsonl"
    metadata_path = source.with_name("evaluation_metadata.json")
    manifest = _read_object(metadata_path) if metadata_path.is_file() else {}
    expected = _index(evaluation.get("results"), evaluation.get("source", "evaluation.results"))
    run_id = evaluation.get("run_id") or manifest.get("run_id")
    if evaluation.get("run_id") and manifest.get("run_id") and evaluation["run_id"] != manifest["run_id"]:
        raise ValueError("评估汇总与轨迹元数据运行编号不一致。")
    result: dict[str, Any] = {
        "source": str(source), "episodes": [], "peak": {}, "peak_window": [], "global": {},
        "missing": {"trace_file": not source.is_file(), "episodes": [], "run_id_episodes": [],
                    "unknown_episode_ids": [], "trace_episodes": [], "field_episode_counts": {}},
        "warnings": [], "metadata": {"run_id": run_id, "boundary_tolerance_m": 1e-6,
            "peak_window_half_width_s": 1.0, "above_load_threshold_g": 20.0,
            "commanded_load_definition": "名义执行动作合成指令，并非逐物理步实际插值指令",
            "estimated_load_definition": "记录的速度差分加重力合成过载估计，非实测物理过载"}}
    seen: set[int] = set()
    best_load = -math.inf
    required = ("position_m", "speed_mps", "commanded_load_g", "estimated_actual_load_g")
    if source.is_file():
        for line_number, record in _records(source):
            episode = _episode_id(record, f"{source}:{line_number}")
            if episode in seen:
                raise ValueError(f"轨迹回合编号重复：{episode}")
            seen.add(episode)
            record_metadata = record.get("metadata") or {}
            record_run_id = record_metadata.get("run_id")
            if record_run_id and run_id and record_run_id != run_id:
                raise ValueError(f"轨迹回合 {episode} 的运行编号与评估汇总不一致。")
            if not record_run_id:
                result["missing"]["run_id_episodes"].append(episode)
            elif run_id is None:
                run_id = record_run_id
                result["metadata"]["run_id"] = run_id
            if episode not in expected:
                result["missing"]["unknown_episode_ids"].append(episode)
                continue
            row = expected[episode]
            for field in ("seed", "missile_count", "termination_reason"):
                if row.get(field) is not None and record_metadata.get(field) is not None and row[field] != record_metadata[field]:
                    raise ValueError(f"回合 {episode} 的轨迹与评估字段 {field} 不一致。")
            if "blue_survived" in record and "blue_survived" in row and record["blue_survived"] != row["blue_survived"]:
                raise ValueError(f"回合 {episode} 的轨迹与评估逃逸结果不一致。")
            trace = record.get("trace")
            if not isinstance(trace, dict) or not isinstance(trace.get("time_s"), list) or not trace["time_s"]:
                result["missing"]["trace_episodes"].append(episode)
                continue
            times = [_finite(value) for value in trace["time_s"]]
            if any(value is None for value in times) or any(right <= left for left, right in zip(times, times[1:])):
                raise ValueError(f"回合 {episode} 的采样时间无效或未严格递增。")
            count = len(times)
            missing_fields = {}
            for field in required:
                values = trace.get(field)
                if not isinstance(values, list):
                    missing_fields[field] = count
                elif field == "position_m":
                    missing_fields[field] = sum(_component(_sample(trace, field, i), 1) is None for i in range(count))
                else:
                    missing_fields[field] = sum(_finite(_sample(trace, field, i)) is None for i in range(count))
                if missing_fields[field]:
                    counts = result["missing"]["field_episode_counts"]
                    counts[field] = counts.get(field, 0) + 1
                if isinstance(values, list) and len(values) != count:
                    result["warnings"].append(f"回合 {episode} 的 {field} 长度为 {len(values)}，时间轴长度为 {count}；缺失样本保持空值。")
            altitudes = [_component(_sample(trace, "position_m", i), 1) for i in range(count)]
            speeds = [_sample(trace, "speed_mps", i) for i in range(count)]
            commands = [_sample(trace, "commanded_load_g", i) for i in range(count)]
            loads = [_sample(trace, "estimated_actual_load_g", i) for i in range(count)]
            low, high = _finite(trace.get("min_altitude_m")), _finite(trace.get("max_altitude_m"))
            min_altitude, max_altitude = _extreme(altitudes), _extreme(altitudes, True)
            summary = {"episode": episode, "samples": count,
                "min_altitude_m": min_altitude, "max_altitude_m": max_altitude,
                "min_speed_mps": _extreme(speeds), "max_speed_mps": _extreme(speeds, True),
                "min_commanded_load_g": _extreme(commands), "max_commanded_load_g": _extreme(commands, True),
                "min_estimated_actual_load_g": _extreme(loads), "max_estimated_actual_load_g": _extreme(loads, True),
                "above_20g_samples": sum(value > 20 for item in loads if (value := _finite(item)) is not None),
                "valid_estimated_load_samples": sum(_finite(item) is not None for item in loads),
                "lower_boundary_contact": _contact(min_altitude, low, missing_fields["position_m"], upper=False),
                "upper_boundary_contact": _contact(max_altitude, high, missing_fields["position_m"], upper=True),
                "lower_boundary_m": low, "upper_boundary_m": high,
                "start_s": times[0], "end_s": times[-1], "run_id": record_run_id,
                "missing_fields": missing_fields}
            result["episodes"].append(summary)
            candidates = [(value, index) for index, item in enumerate(loads) if (value := _finite(item)) is not None]
            if candidates:
                peak_load, peak_index = max(candidates, key=lambda item: item[0])
                if peak_load > best_load:
                    best_load = peak_load
                    peak = _point(trace, episode, peak_index)
                    previous = _point(trace, episode, peak_index - 1) if peak_index else {}
                    peak["previous_altitude_m"] = previous.get("altitude_m")
                    peak["previous_vertical_speed_mps"] = previous.get("vertical_speed_mps")
                    peak["run_id"] = record_run_id
                    result["peak"] = peak
                    result["peak_window"] = [_point(trace, episode, i) for i, time in enumerate(times)
                                             if abs(time - times[peak_index]) <= 1.0 + 1e-9]
    summaries = result["episodes"]
    summaries.sort(key=lambda row: row["episode"])
    matched_ids = {row["episode"] for row in summaries}
    result["missing"]["episodes"] = sorted(set(expected) - matched_ids)
    result["global"] = {"expected_episodes": len(expected), "matched_episodes": len(summaries),
        "total_samples": sum(row["samples"] for row in summaries),
        "above_20g_samples": sum(row["above_20g_samples"] for row in summaries),
        "above_20g_episodes": sum(row["above_20g_samples"] > 0 for row in summaries),
        "valid_estimated_load_samples": sum(row["valid_estimated_load_samples"] for row in summaries),
        "lower_boundary_contact_episodes": sum(row["lower_boundary_contact"] is True for row in summaries),
        "upper_boundary_contact_episodes": sum(row["upper_boundary_contact"] is True for row in summaries),
        "lower_boundary_observed_episodes": sum(row["lower_boundary_contact"] is not None for row in summaries),
        "upper_boundary_observed_episodes": sum(row["upper_boundary_contact"] is not None for row in summaries)}
    for field in ("altitude_m", "speed_mps", "commanded_load_g", "estimated_actual_load_g"):
        result["global"]["min_" + field] = _extreme([row["min_" + field] for row in summaries])
        result["global"]["max_" + field] = _extreme([row["max_" + field] for row in summaries], True)
    for key, description in (("episodes", "缺少可用轨迹的评估回合"),
                             ("run_id_episodes", "缺少运行编号的轨迹回合"),
                             ("unknown_episode_ids", "不属于本次评估而被排除的轨迹回合")):
        if result["missing"][key]:
            result["warnings"].append(f"{description}：{len(result['missing'][key])} 个。")
    if not source.is_file():
        result["warnings"].append("评估逐回合轨迹文件不存在；数值动力学图不能由汇总数据补造。")
    return result
