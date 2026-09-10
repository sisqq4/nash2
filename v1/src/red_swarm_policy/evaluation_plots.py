"""Derived evaluation figures; reads episode metrics without running a policy."""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
from textwrap import fill
from typing import Any, Callable, Sequence

from .analyze_blue_evaluations import _describe, _finite, _wilson, validate_rows


EvaluationInput = tuple[str, Path, list[dict[str, Any]]]
ORIENTATIONS = ("toward_missile_swarm", "positive_90_deg", "negative_90_deg", "away_from_missile_swarm")
ORIENTATION_LABELS = {
    "toward_missile_swarm": "Toward", "positive_90_deg": "+90 deg",
    "negative_90_deg": "-90 deg", "away_from_missile_swarm": "Away",
}
LEVELS = {"overall": (), "by_missile_count": ("missile_count",),
          "by_blue_orientation": ("blue_orientation",),
          "by_missile_count_and_orientation": ("missile_count", "blue_orientation")}
QUALITY_PLOTS = {
    "flight_quality_score": "Flight quality score (0-100)",
    "max_abs_flight_path_angle_deg": "Maximum absolute flight-path angle (deg)",
    "min_horizontal_speed_mps": "Minimum horizontal speed (m/s)",
    "min_horizontal_speed_ratio": "Minimum horizontal / total speed ratio",
    "max_estimated_actual_load_g": "Maximum estimated actual load (g)",
    "action_switch_rate_hz": "Action switch rate (Hz)",
    "safety_intervention_rate": "Safety intervention fraction",
    "time_near_altitude_boundary_s": "Time near altitude boundary (s)",
    "spiral_total_duration_s": "Spiral duration (s)",
    "steep_low_horizontal_speed_total_duration_s": "Steep, low-horizontal-speed duration (s)",
}


def one_meter_histogram(values: Sequence[float]) -> dict[str, Any]:
    """Exact left-closed/right-open meter bins, stored sparsely including all tails."""
    if any(isinstance(value, bool) or not isinstance(value, (int, float))
           or not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Miss distances must be finite, non-negative numbers")
    counts = Counter(math.floor(value) for value in values)
    total = len(values)
    return {"bin_width_m": 1, "sample_count": total,
            "interval": "[lower_m, upper_m)", "unlisted_bins_have_zero_count": True,
            "bins": [{"lower_m": lower, "upper_m": lower + 1, "count": count,
                      "probability": count / total} for lower, count in sorted(counts.items())]}


def _quality(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("flight_quality")
    return value if isinstance(value, dict) else {}


def _quality_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    qualities = [_quality(row) for row in rows]
    metric_names = sorted({name for quality in qualities
                           for name in (quality.get("metrics") or {})})
    metrics = {}
    for name in metric_names:
        values = _finite(rows, f"flight_quality.metrics.{name}")
        metrics[name] = {**_describe(values), "missing_count": len(rows) - len(values)}
    verdict_names = sorted({name for quality in qualities
                            for name in (quality.get("verdicts") or {})})
    verdicts = {}
    for name in verdict_names:
        observed = [(quality.get("verdicts") or {}).get(name) for quality in qualities]
        valid = [value for value in observed if isinstance(value, bool)]
        failed = valid.count(False)
        ci_low, ci_high = _wilson(failed, len(valid))
        verdicts[name] = {"evaluated_episodes": len(valid), "missing_episodes": len(rows) - len(valid),
                          "failed_episodes": failed,
                          "failure_rate": failed / len(valid) if valid else None,
                          "ci95_low": ci_low, "ci95_high": ci_high}
    event_rows = [quality["events"] for quality in qualities if isinstance(quality.get("events"), list)]
    event_names = sorted({str(event["type"]) for events in event_rows
                          for event in events if isinstance(event, dict) and "type" in event})
    events = {}
    for name in event_names:
        counts = [sum(isinstance(event, dict) and event.get("type") == name for event in row)
                  for row in event_rows]
        affected = sum(count > 0 for count in counts)
        events[name] = {"event_count": sum(counts), "affected_episodes": affected,
                        "evaluated_episodes": len(event_rows),
                        "missing_episodes": len(rows) - len(event_rows),
                        "affected_episode_rate": affected / len(event_rows) if event_rows else None}
    return {"available_episodes": sum(bool(quality) for quality in qualities),
            "missing_episodes": sum(not quality for quality in qualities),
            "metrics": metrics, "verdicts": verdicts, "events": events,
            "event_records_available_episodes": len(event_rows)}


def build_plot_statistics(inputs: Sequence[EvaluationInput]) -> dict[str, Any]:
    if not inputs or len({label for label, _, _ in inputs}) != len(inputs):
        raise ValueError("Evaluation inputs must be non-empty and have unique labels")
    groups = []
    for label, source, rows in inputs:
        validate_rows(rows, source)
        for level, dimensions in LEVELS.items():
            selected: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
            for row in rows:
                selected[tuple(row[name] for name in dimensions)].append(row)
            for key, members in sorted(selected.items(), key=lambda item: str(item[0])):
                successes = sum(row["blue_survived"] is True for row in members)
                ci_low, ci_high = _wilson(successes, len(members))
                groups.append({"policy": label, "level": level, "group": dict(zip(dimensions, key)),
                               "episodes": len(members), "survived": successes,
                               "escape_rate": successes / len(members),
                               "escape_rate_ci95_low": ci_low, "escape_rate_ci95_high": ci_high,
                               "miss_distance_m": _describe(_finite(members, "miss_distance_m")),
                               "miss_distance_histogram_1m": one_meter_histogram(
                                   [row["miss_distance_m"] for row in members]),
                               "flight_quality": _quality_statistics(members)})
    return {"schema_version": 1, "inputs": {label: str(path) for label, path, _ in inputs},
            "groups": groups, "notes": [
                "Escape rate counts blue_survived=True episodes; bars include 95% Wilson intervals and sample counts.",
                "Miss distance uses each episode's existing miss_distance_m, one observation per episode (not per missile).",
                "All distances, including hit episodes and long tails, are included in [m,m+1) bins; unlisted bins are zero.",
                "Histogram probability divides by all episodes in that policy/group; zoom views do not renormalize.",
                "Missing policy/scenario combinations are not counted as failures.",
                "Flight-quality values are read as recorded; no new thresholds or trajectory reconstruction are applied.",
                "Verdict True means pass; failure rates exclude missing verdicts and report their denominators.",
            ]}


def _group_key(group: dict[str, Any]) -> tuple[Any, Any]:
    return group.get("missile_count"), group.get("blue_orientation")


def _sort_key(group: dict[str, Any]) -> tuple[Any, ...]:
    orientation = group.get("blue_orientation")
    return (group.get("missile_count", 0),
            ORIENTATIONS.index(orientation) if orientation in ORIENTATIONS else len(ORIENTATIONS),
            str(orientation))


def _group_label(group: dict[str, Any]) -> str:
    parts = []
    if "missile_count" in group:
        parts.append(f"{group['missile_count']} missiles")
    if "blue_orientation" in group:
        value = group["blue_orientation"]
        parts.append(ORIENTATION_LABELS.get(value, str(value)))
    return " / ".join(parts) or "Overall"


def _group_specs(report: dict[str, Any], level: str) -> list[dict[str, Any]]:
    unique = {_group_key(row["group"]): row["group"] for row in report["groups"] if row["level"] == level}
    return sorted(unique.values(), key=_sort_key)


def _select(report: dict[str, Any], level: str, group: dict[str, Any]) -> list[dict[str, Any]]:
    return [row for row in report["groups"] if row["level"] == level and row["group"] == group]


def _missing(axis: Any, message: str = "No recorded flight-quality data") -> None:
    axis.text(.5, .5, message, ha="center", va="center", transform=axis.transAxes, color="#64748b")
    axis.set_axis_off()


def _panels(output: Path, name: str, title: str, items: list[Any],
            draw: Callable[[Any, Any], None], dpi: int) -> list[str]:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    files = []
    for start in range(0, max(1, len(items)), 6):
        page = items[start:start + 6] or [None]
        columns = 2 if len(page) > 1 else 1
        figure = Figure(figsize=(13 if columns == 2 else 9, 4.2 * math.ceil(len(page) / columns) + .6),
                        layout="constrained")
        FigureCanvasAgg(figure)
        axes = figure.subplots(math.ceil(len(page) / columns), columns, squeeze=False).ravel()
        figure.suptitle(title, fontsize=15, fontweight="bold")
        try:
            for axis, item in zip(axes, page):
                draw(axis, item)
                axis.grid(axis="y", alpha=.2)
            for axis in axes[len(page):]:
                axis.set_axis_off()
            filename = f"{name}{f'_{start // 6 + 1:02d}' if start else ''}.png"
            figure.savefig(output / filename, dpi=dpi)
            files.append(filename)
        finally:
            figure.clear()
    return files


def _rate_bars(axis: Any, entries: list[tuple[str, float | None, int, float | None, float | None]],
               title: str, ylabel: str) -> None:
    from matplotlib.ticker import PercentFormatter

    if not entries or all(rate is None for _, rate, _, _, _ in entries):
        _missing(axis)
        axis.set_title(fill(title, 60), fontsize=11)
        return
    for i, (_, rate, n, low, high) in enumerate(entries):
        if rate is None:
            axis.text(i, .04, "N/A", ha="center", fontsize=8)
            continue
        axis.bar(i, rate, color="#3b82f6", width=.65)
        if low is not None and high is not None:
            axis.errorbar(i, rate, yerr=[[max(0., rate - low)], [max(0., high - rate)]],
                          fmt="none", color="#334155", capsize=3)
        axis.text(i, min(1.11, (high if high is not None else rate) + .035),
                  f"{rate:.1%}\nn={n}", ha="center", va="bottom", fontsize=8)
    axis.set_xticks(range(len(entries)), [fill(label, 18) for label, *_ in entries],
                    rotation=20 if len(entries) > 4 else 0)
    axis.set_ylim(0, 1.28)
    axis.set_yticks([0, .25, .5, .75, 1])
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set(title=fill(title, 60), ylabel=ylabel)


def _escape_charts(output: Path, report: dict[str, Any], dpi: int) -> list[str]:
    files = []
    labels = list(report["inputs"])
    for level in ("overall", "by_missile_count", "by_blue_orientation"):
        def draw(axis: Any, label: str) -> None:
            entries = []
            for group in _group_specs(report, level):
                row = next((row for row in _select(report, level, group) if row["policy"] == label), None)
                entries.append((_group_label(group), row["escape_rate"] if row else None,
                                row["episodes"] if row else 0, row["escape_rate_ci95_low"] if row else None,
                                row["escape_rate_ci95_high"] if row else None))
            _rate_bars(axis, entries, label, "Blue escape rate")
        filename = "escape_rate_overall" if level == "overall" else f"escape_rate_{level}"
        files += _panels(output, filename, "Blue escape rate — 95% Wilson intervals", labels, draw, dpi)

    def heatmap(axis: Any, label: str) -> None:
        import numpy as np
        rows = [row for row in report["groups"]
                if row["level"] == "by_missile_count_and_orientation" and row["policy"] == label]
        counts = sorted({row["group"]["missile_count"] for row in rows})
        orientations = sorted({row["group"]["blue_orientation"] for row in rows},
                              key=lambda value: _sort_key({"blue_orientation": value}))
        matrix = np.full((len(counts), len(orientations)), np.nan)
        lookup = {_group_key(row["group"]): row for row in rows}
        for y, count in enumerate(counts):
            for x, orientation in enumerate(orientations):
                row = lookup.get((count, orientation))
                if row:
                    matrix[y, x] = row["escape_rate"]
                text = f"{row['escape_rate']:.1%}\nn={row['episodes']}" if row else "N/A"
                axis.text(x, y, text, ha="center", va="center", fontsize=9)
        axis.imshow(matrix, vmin=0, vmax=1, cmap="YlGnBu", alpha=.65, aspect="auto")
        axis.set_xticks(range(len(orientations)), [ORIENTATION_LABELS.get(v, v) for v in orientations])
        axis.set_yticks(range(len(counts)), [str(v) for v in counts])
        axis.set(title=fill(label, 60), xlabel="Initial blue orientation", ylabel="Missile count")
        axis.grid(False)
    files += _panels(output, "escape_rate_by_missile_count_and_orientation",
                     "Blue escape rate — missile count × orientation", labels, heatmap, dpi)
    return files


def _histogram_charts(output: Path, report: dict[str, Any], dpi: int) -> list[str]:
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    files = []
    for level in LEVELS:
        groups = _group_specs(report, level)
        def draw(axis: Any, group: dict[str, Any], zoom: bool = False) -> None:
            for row in _select(report, level, group):
                histogram = row["miss_distance_histogram_1m"]
                bins = [entry for entry in histogram["bins"] if not zoom or entry["lower_m"] < 50]
                xs = [entry["lower_m"] for entry in bins]
                ys = [entry["probability"] for entry in bins]
                tail = sum(entry["count"] for entry in histogram["bins"] if entry["lower_m"] >= 50)
                label = f"{row['policy']} (n={histogram['sample_count']}"
                label += f", >=50 m: {tail})" if zoom else ")"
                bars = axis.bar(xs, ys, width=1., align="edge", alpha=.35, label=fill(label, 45))
                if bars:
                    color = bars[0].get_facecolor()
                    # Centers make occupied 1 m bins visible even for kilometer-scale tails.
                    axis.plot([x + .5 for x in xs], ys, linestyle="none", marker=".",
                              markersize=3, color=color[:3])
            axis.set(title=fill(_group_label(group), 60), xlabel="Miss distance (m); bins [m, m+1)",
                     ylabel="Probability per 1 m bin")
            axis.set_xlim(left=0, right=50 if zoom else None)
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.yaxis.set_major_formatter(PercentFormatter(1))
            axis.legend(fontsize=8)
        name = "miss_distance_distribution" if level == "overall" else f"miss_distance_distribution_{level}"
        files += _panels(output, name, "Miss distance — exact 1 m bins, all episodes", groups, draw, dpi)
        if any(row["miss_distance_m"]["max"] >= 50 for row in report["groups"] if row["level"] == level):
            files += _panels(output, name + "_0_50m", "Miss distance — 0–50 m detail (same denominator)",
                             groups, lambda axis, group: draw(axis, group, True), dpi)
    return files


def _quality_charts(output: Path, inputs: Sequence[EvaluationInput],
                    report: dict[str, Any], dpi: int) -> list[str]:
    files = []
    lookup = {label: rows for label, _, rows in inputs}
    metric_names = [name for name in QUALITY_PLOTS if any(
        _finite(rows, f"flight_quality.metrics.{name}") for rows in lookup.values())]

    def distributions(axis: Any, metric: str | None) -> None:
        available = [(label, _finite(rows, f"flight_quality.metrics.{metric}"))
                     for label, rows in lookup.items()]
        available = [(label, values) for label, values in available if values]
        if not available:
            _missing(axis)
            return
        for i, (label, values) in enumerate(available, 1):
            axis.boxplot([values], positions=[i], widths=.5, manage_ticks=False, showmeans=True)
        axis.set_xticks(range(1, len(available) + 1),
                        [fill(f"{label}\nn={len(values)}", 22) for label, values in available])
        axis.set_title(fill(QUALITY_PLOTS.get(metric, metric), 55), fontsize=11)
    files += _panels(output, "flight_quality_metrics", "Flight quality — recorded metric distributions",
                     metric_names, distributions, dpi)

    def score_histogram(axis: Any, _: Any) -> None:
        available = False
        for label, rows in lookup.items():
            values = _finite(rows, "flight_quality.metrics.flight_quality_score")
            if values:
                available = True
                axis.hist(values, bins=list(range(0, 101, 5)),
                          weights=[1 / len(values)] * len(values), alpha=.4,
                          label=fill(f"{label} (n={len(values)})", 40))
        if not available:
            _missing(axis)
            return
        from matplotlib.ticker import PercentFormatter
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set(xlabel="Flight quality score", ylabel="Probability per 5-point bin", xlim=(0, 100))
        axis.legend(fontsize=8)
    files += _panels(output, "flight_quality_score_distribution", "Flight quality score distribution",
                     [None], score_histogram, dpi)

    if not any(_quality(row) for rows in lookup.values() for row in rows):
        files += _panels(output, "flight_quality_failure_rate_overall", "Flight quality — acceptance failures",
                         [None], lambda axis, _: _missing(axis), dpi)
        return files

    for level in ("by_missile_count", "by_blue_orientation", "by_missile_count_and_orientation"):
        def score_boxes(axis: Any, group: dict[str, Any]) -> None:
            available = []
            for label, rows in lookup.items():
                selected = [row for row in rows if all(row.get(k) == v for k, v in group.items())]
                values = _finite(selected, "flight_quality.metrics.flight_quality_score")
                if values:
                    available.append((label, values))
            if not available:
                _missing(axis)
            else:
                axis.boxplot([values for _, values in available], showmeans=True)
                axis.set_xticks(range(1, len(available) + 1),
                                [fill(f"{label}\nn={len(values)}", 22) for label, values in available])
                axis.set_ylabel("Flight quality score")
                axis.set_ylim(-2, 105)
            axis.set_title(fill(_group_label(group), 60), fontsize=11)
        files += _panels(output, f"flight_quality_score_{level}", "Flight quality by scenario",
                         _group_specs(report, level), score_boxes, dpi)

    for level in LEVELS:
        def verdict_bars(axis: Any, item: tuple[dict[str, Any], str]) -> None:
            group, label = item
            entries = []
            for row in _select(report, level, group):
                if row["policy"] != label:
                    continue
                for name, verdict in row["flight_quality"]["verdicts"].items():
                    entries.append((name.replace('_', ' '), verdict["failure_rate"],
                                    verdict["evaluated_episodes"], verdict["ci95_low"], verdict["ci95_high"]))
            _rate_bars(axis, entries, f"{label}: {_group_label(group)}", "Failed checks / observed episodes")
        files += _panels(output, f"flight_quality_failure_rate_{level}",
                         "Flight quality — recorded acceptance failures",
                         [(group, label) for group in _group_specs(report, level) for label in lookup], verdict_bars, dpi)

    def event_bars(axis: Any, label: str) -> None:
        quality = next(row["flight_quality"] for row in report["groups"]
                       if row["level"] == "overall" and row["policy"] == label)
        entries = [(name.replace("_", " "), value["affected_episode_rate"], value["evaluated_episodes"], None, None)
                   for name, value in quality["events"].items()]
        if not entries and quality["event_records_available_episodes"]:
            _missing(axis, f"No events in {quality['event_records_available_episodes']} recorded episodes")
            axis.set_title(fill(label, 60))
        else:
            _rate_bars(axis, entries, label, "Episodes with event / observed episodes")
    files += _panels(output, "flight_quality_event_rates", "Flight quality — episodes with recorded events",
                     list(lookup), event_bars, dpi)
    return files


def write_evaluation_plots(output: Path, inputs: Sequence[EvaluationInput], *, dpi: int = 160,
                           make_plots: bool = True) -> dict[str, Any]:
    """Write derived figures and exact counts; preserve all original evaluation artifacts."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    output = Path(output).resolve()
    if any(output == source.resolve().parent for _, source, _ in inputs):
        raise ValueError("Use a separate analysis output directory, not the source artifact directory")
    report = build_plot_statistics(inputs)
    output.mkdir(parents=True, exist_ok=True)
    report["figures"] = []
    (output / "plot_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "miss_distance_histogram_1m.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["policy", "level", "missile_count", "blue_orientation",
                                                    "sample_count", "lower_m", "upper_m", "count", "probability"])
        writer.writeheader()
        for row in report["groups"]:
            histogram = row["miss_distance_histogram_1m"]
            for entry in histogram["bins"]:
                writer.writerow({"policy": row["policy"], "level": row["level"], **row["group"],
                                 "sample_count": histogram["sample_count"], **entry})
    if make_plots:
        files = _escape_charts(output, report, dpi)
        files += _histogram_charts(output, report, dpi)
        files += _quality_charts(output, inputs, report, dpi)
        report["figures"] = files
        (output / "plot_statistics.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report
