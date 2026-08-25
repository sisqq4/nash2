"""Offline extraction and comparison of Blue-policy evaluation results.

This module deliberately consumes existing evaluation artifacts; it never creates an
environment or loads a checkpoint.  Thus reports can be regenerated without changing
training or test behaviour.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable, Sequence

DEFAULT_DIMENSIONS = ("missile_count", "blue_orientation")
REQUIRED_FIELDS = ("blue_survived", "miss_distance_m", "missile_count", "blue_orientation")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract, compare, and plot Blue evaluation artifacts")
    parser.add_argument("inputs", nargs="+", metavar="LABEL=PATH",
                        help="Named evaluation.json, baseline summary, or directory containing one")
    parser.add_argument("--output", type=Path, default=Path("outputs/blue_rl/analysis"))
    parser.add_argument("--dimensions", default=",".join(DEFAULT_DIMENSIONS),
                        help="Comma-separated row fields (nested fields use dots); use 'none' for overall only")
    parser.add_argument("--baseline", default=None,
                        help="Label used for delta columns (default: first input)")
    parser.add_argument("--miss-distance-bins", type=int, default=20)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-invalid-inputs", action="store_true",
                        help="Skip unreadable/incompatible inputs and record them in analysis.json")
    return parser


def _artifact_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = (path / "evaluation.json", path / "blue_rule_baseline_summary.json",
                  path / "stage1_zero_pn_summary.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    found = sorted(path.glob("*summary.json")) if path.is_dir() else []
    if len(found) == 1:
        return found[0]
    raise ValueError(f"cannot identify one evaluation artifact under {path}")


def _rows(document: object, source: Path) -> list[dict[str, Any]]:
    if isinstance(document, list):
        rows = document
    elif isinstance(document, dict):
        rows = next((document[key] for key in ("results", "episodes", "trials")
                     if isinstance(document.get(key), list)), None)
    else:
        rows = None
    if rows is None:
        raise ValueError(f"{source} has no per-episode results/episodes/trials array")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{source} contains a non-object episode row")
    return rows


def load_input(spec: str) -> tuple[str, Path, list[dict[str, Any]]]:
    if "=" not in spec:
        raise ValueError(f"input must be LABEL=PATH: {spec!r}")
    label, raw_path = spec.split("=", 1)
    if not label.strip() or not raw_path:
        raise ValueError(f"input must have a non-empty label and path: {spec!r}")
    path = _artifact_path(Path(raw_path))
    document = json.loads(path.read_text(encoding="utf-8"))
    return label.strip(), path, _rows(document, path)


def _nested(row: dict[str, Any], field: str) -> object:
    value: object = row
    for part in field.split("."):
        if not isinstance(value, dict) or part not in value:
            return "<missing>"
        value = value[part]
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _finite(rows: Iterable[dict[str, Any]], field: str) -> list[float]:
    values = []
    for row in rows:
        value = _nested(row, field)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            values.append(float(value))
    return values


def _switch_value(row: dict[str, Any]) -> object:
    """Accept both the current evaluator layout and a future top-level field."""
    mechanism = row.get("mechanism")
    # The current evaluator writes zero with an empty trace when all evaluation
    # mechanisms are disabled.  That zero means "not observed", not "no switch".
    if isinstance(mechanism, dict) and mechanism.get("enabled") is False \
            and "main_threat_switches" not in row:
        return "<missing>"
    nested = _nested(row, "mechanism.main_threat_switches")
    return row.get("main_threat_switches", nested)


def inspect_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Report missing/invalid fields instead of silently turning them into results."""
    missing = {field: sum(_nested(row, field) == "<missing>" for row in rows)
               for field in REQUIRED_FIELDS}
    invalid_survival = sum("blue_survived" in row and not isinstance(row["blue_survived"], bool)
                           for row in rows)
    invalid_miss = sum("miss_distance_m" in row and
                       (isinstance(row["miss_distance_m"], bool) or
                        not isinstance(row["miss_distance_m"], (int, float)) or
                        not math.isfinite(row["miss_distance_m"])) for row in rows)
    switch_available = sum(isinstance(_switch_value(row), (int, float)) and
                           not isinstance(_switch_value(row), bool) for row in rows)
    warnings = []
    if switch_available != len(rows):
        warnings.append(
            "main-threat switches are unavailable for some episodes; this metric can only be "
            "compared when the evaluator recorded mechanism.main_threat_switches"
        )
    return {"missing_required_fields": missing, "invalid_blue_survived": invalid_survival,
            "invalid_miss_distance_m": invalid_miss,
            "main_threat_switches_available": switch_available,
            "main_threat_switches_missing": len(rows) - switch_available,
            "warnings": warnings}


def validate_rows(rows: list[dict[str, Any]], source: Path) -> None:
    if not rows:
        raise ValueError(f"{source} contains no episode rows")
    quality = inspect_rows(rows)
    missing = quality["missing_required_fields"]
    assert isinstance(missing, dict)
    bad = [name for name, count in missing.items() if count]
    if bad or quality["invalid_blue_survived"] or quality["invalid_miss_distance_m"]:
        raise ValueError(
            f"{source} is not a compatible Blue evaluation artifact: missing={bad}, "
            f"invalid_blue_survived={quality['invalid_blue_survived']}, "
            f"invalid_miss_distance_m={quality['invalid_miss_distance_m']}"
        )


def _describe(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "p25": None,
                "median": None, "p75": None, "max": None}
    ordered = sorted(float(value) for value in values)

    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent
        lower = math.floor(position); upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"count": len(ordered), "mean": statistics.fmean(ordered),
            "std": statistics.pstdev(ordered), "min": ordered[0],
            "p25": percentile(.25), "median": percentile(.5),
            "p75": percentile(.75), "max": ordered[-1]}


def _wilson(successes: int, total: int) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - radius, center + radius


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    survived = sum(row["blue_survived"] is True for row in rows)
    ci_low, ci_high = _wilson(survived, len(rows))
    mechanism_switches = [_switch_value(row) for row in rows]
    switches = [float(value) for value in mechanism_switches
                if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {
        "episodes": len(rows), "survived": survived, "killed": len(rows) - survived,
        "escape_rate": survived / len(rows) if rows else None,
        "escape_rate_ci95_low": ci_low, "escape_rate_ci95_high": ci_high,
        "miss_distance_m": _describe(_finite(rows, "miss_distance_m")),
        "main_threat_switches": _describe(switches),
        "reward": _describe(_finite(rows, "reward")),
        "simulation_time_s": _describe(_finite(rows, "simulation_time_s")),
        "decision_steps": _describe(_finite(rows, "decision_steps")),
        "hit_count": _describe(_finite(rows, "hit_count")),
        "termination_counts": dict(sorted(Counter(str(row.get("termination_reason", "<missing>"))
                                                   for row in rows).items())),
        "red_loss_reason_counts": dict(sorted(Counter(str(reason) for row in rows
                                                       for reason in row.get("red_loss_reasons", [])).items())),
    }


def extract(inputs: Sequence[tuple[str, Path, list[dict[str, Any]]]], dimensions: Sequence[str],
            baseline: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report: dict[str, Any] = {"schema_version": 1, "baseline": baseline,
                              "dimensions": list(dimensions), "inputs": {}, "groups": []}
    flat: list[dict[str, Any]] = []
    group_summaries: dict[tuple[str, tuple[object, ...]], dict[str, Any]] = {}
    for label, path, rows in inputs:
        report["inputs"][label] = {"path": str(path), "episodes": len(rows),
                                   "data_quality": inspect_rows(rows)}
        grouped: dict[tuple[object, ...], list[dict[str, Any]]] = defaultdict(list)
        grouped[tuple()].extend(rows)
        if dimensions:
            for row in rows:
                grouped[tuple(_nested(row, dimension) for dimension in dimensions)].append(row)
        for key, selected in grouped.items():
            level = "overall" if not key else "stratified"
            summary = summarize(selected)
            record = {"policy": label, "level": level,
                      "group": dict(zip(dimensions, key)), "metrics": summary}
            report["groups"].append(record)
            group_summaries[(label, key)] = summary
    if baseline not in report["inputs"]:
        raise ValueError(f"unknown baseline label {baseline!r}")
    for record in report["groups"]:
        key = tuple() if record["level"] == "overall" else tuple(
            record["group"].get(dimension) for dimension in dimensions)
        base = group_summaries.get((baseline, key))
        metrics = record["metrics"]
        delta = None if base is None or metrics["escape_rate"] is None or base["escape_rate"] is None \
            else metrics["escape_rate"] - base["escape_rate"]
        miss_delta = None
        if base is not None and metrics["miss_distance_m"]["mean"] is not None \
                and base["miss_distance_m"]["mean"] is not None:
            miss_delta = metrics["miss_distance_m"]["mean"] - base["miss_distance_m"]["mean"]
        record["comparison_to_baseline"] = {"escape_rate_delta": delta,
                                             "miss_distance_mean_delta_m": miss_delta}
        row = {"policy": record["policy"], "level": record["level"], **record["group"],
               "episodes": metrics["episodes"], "escape_rate": metrics["escape_rate"],
               "escape_rate_ci95_low": metrics["escape_rate_ci95_low"],
               "escape_rate_ci95_high": metrics["escape_rate_ci95_high"],
               "escape_rate_delta": delta}
        for metric in ("miss_distance_m", "main_threat_switches", "reward", "simulation_time_s",
                       "decision_steps", "hit_count"):
            for statistic, value in metrics[metric].items():
                row[f"{metric}_{statistic}"] = value
        row["miss_distance_mean_delta_m"] = miss_delta
        flat.append(row)
    return report, flat


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _episode_rows(inputs: Sequence[tuple[str, Path, list[dict[str, Any]]]],
                  dimensions: Sequence[str]) -> list[dict[str, Any]]:
    """Create a compact, non-lossy-enough episode table for downstream analysis."""
    result = []
    for label, source, rows in inputs:
        for row in rows:
            mechanism = row.get("mechanism") if isinstance(row.get("mechanism"), dict) else {}
            result.append({
                "policy": label, "source": str(source), "episode": row.get("episode"),
                **{field: _nested(row, field) for field in dimensions},
                "missile_count": row["missile_count"],
                "blue_orientation": row["blue_orientation"],
                "blue_survived": row["blue_survived"],
                "miss_distance_m": row["miss_distance_m"],
                "main_threat_switches": _switch_value(row)
                if isinstance(_switch_value(row), (int, float)) else None,
                "reward": row.get("reward"), "hit_count": row.get("hit_count"),
                "termination_reason": row.get("termination_reason"),
                "simulation_time_s": row.get("simulation_time_s"),
                "decision_steps": row.get("decision_steps"),
                "mechanisms_enabled": mechanism.get("enabled"),
                "mechanism_intervention_rate": mechanism.get("intervention_rate"),
                "red_loss_reasons": json.dumps(row.get("red_loss_reasons", []), ensure_ascii=False),
            })
    return result


def _plots(output: Path, inputs: Sequence[tuple[str, Path, list[dict[str, Any]]]],
           bins: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [item[0] for item in inputs]
    rows_by_label = {label: rows for label, _, rows in inputs}
    fig, axis = plt.subplots(figsize=(9, 5))
    for label in labels:
        values = _finite(rows_by_label[label], "miss_distance_m")
        if values:
            axis.hist(values, bins=bins, density=True, histtype="step", linewidth=2, label=label)
    axis.set(xlabel="Miss distance (m)", ylabel="Probability density", title="Miss-distance distribution")
    axis.legend(); axis.grid(alpha=.25); fig.tight_layout()
    fig.savefig(output / "miss_distance_distribution.png", dpi=160); plt.close(fig)

    missile_counts = sorted({int(row["missile_count"]) for _, _, rows in inputs for row in rows
                             if isinstance(row.get("missile_count"), int)})
    if missile_counts:
        fig, axis = plt.subplots(figsize=(9, 5)); width = .8 / len(labels)
        x = list(range(len(missile_counts)))
        for index, label in enumerate(labels):
            rates = [summarize([row for row in rows_by_label[label]
                                if row.get("missile_count") == count])["escape_rate"] or 0.0
                     for count in missile_counts]
            positions = [value + (index - (len(labels) - 1) / 2) * width for value in x]
            axis.bar(positions, rates, width, label=label)
        axis.set_xticks(x, [str(value) for value in missile_counts]); axis.set_ylim(0, 1)
        axis.set(xlabel="Red missile count", ylabel="Escape rate", title="Escape rate by missile count")
        axis.legend(); axis.grid(axis="y", alpha=.25); fig.tight_layout()
        fig.savefig(output / "escape_rate_by_missile_count.png", dpi=160); plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.miss_distance_bins < 1:
        raise SystemExit("--miss-distance-bins must be positive")
    skipped_inputs: list[dict[str, str]] = []
    try:
        inputs = []
        for spec in args.inputs:
            try:
                item = load_input(spec)
                validate_rows(item[2], item[1])
                inputs.append(item)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                if not args.skip_invalid_inputs:
                    raise
                skipped_inputs.append({"input": spec, "error": str(error)})
        if not inputs:
            raise ValueError("no valid evaluation inputs remain")
        labels = [item[0] for item in inputs]
        if len(labels) != len(set(labels)):
            raise ValueError("input labels must be unique")
        dimensions = () if args.dimensions.strip().lower() == "none" else tuple(
            field.strip() for field in args.dimensions.split(",") if field.strip())
        baseline = args.baseline or labels[0]
        report, flat = extract(inputs, dimensions, baseline)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    args.output.mkdir(parents=True, exist_ok=True)
    report["skipped_inputs"] = skipped_inputs
    (args.output / "analysis.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(args.output / "metrics.csv", flat)
    _write_csv(args.output / "episodes.csv", _episode_rows(inputs, dimensions))
    if not args.no_plots:
        _plots(args.output, inputs, args.miss_distance_bins)
    print(json.dumps({"output": str(args.output), "inputs": labels, "groups": len(flat)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
