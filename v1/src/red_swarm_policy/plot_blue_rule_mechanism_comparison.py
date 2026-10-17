"""Plot an existing Blue rule-mechanism/Rainbow comparison without rerunning it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .blue_rl.flight_quality import replot_flight_quality_episodes
from .plot_localization import chinese_plot_context


VARIANT_LABELS = {
    "rule_raw": "原始规则机",
    "rule_threat": "威胁规则",
    "rule_timing": "时机规则",
    "rule_direction": "方向规则",
    "rule_overload": "过载规则",
    "rainbow_full": "完整 Rainbow RL",
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_single_episode(path: Path) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"episode record must be an object: {path}:{line_number}")
            records.append(value)
    if len(records) != 1:
        raise ValueError(f"expected exactly one episode in {path}, found {len(records)}")
    return records[0]


def _red_track(episode: dict[str, Any], slot: int) -> np.ndarray:
    trace = episode["trace"]
    time = np.asarray(trace["time_s"], dtype=float)
    frames = trace["red_positions_m"]
    points = np.asarray(
        [frame[slot] if slot < len(frame) else [np.nan, np.nan, np.nan] for frame in frames],
        dtype=float,
    )
    valid = trace.get("red_state_valid")
    if valid is not None:
        slot_valid = np.asarray(
            [bool(frame[slot]) if slot < len(frame) else False for frame in valid], dtype=bool
        )
        points[~slot_valid] = np.nan
    red_ids = trace.get("red_ids", [])
    red_id = red_ids[slot] if slot < len(red_ids) else slot
    terminal = next(
        (event for event in episode.get("red_termination_events", [])
         if event.get("red_id") == red_id),
        None,
    )
    if terminal is not None:
        points = points[time < float(terminal["time_s"])]
        if terminal.get("position_valid", False):
            points = np.vstack([points, np.asarray(terminal["position_m"], dtype=float)])
    return points


def _render_horizontal_overview(
    episodes: dict[str, dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    comparison: dict[str, Any],
    path: Path,
    *,
    font_path: str | Path | None,
    dpi: int,
) -> None:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    variants = list(comparison["variant_order"])
    with chinese_plot_context(font_path) as font_name:
        figure = Figure(figsize=(16, 10), layout="constrained")
        FigureCanvasAgg(figure)
        axes = [figure.add_subplot(2, 3, index + 1) for index in range(len(variants))]
        for axis, variant in zip(axes, variants):
            episode = episodes[variant]
            summary = summaries[variant]
            trace = episode["trace"]
            blue = np.asarray(trace["position_m"], dtype=float)
            axis.plot(blue[:, 0], blue[:, 2], color="royalblue", linewidth=2.0)
            red_count = max((len(frame) for frame in trace["red_positions_m"]), default=0)
            for slot in range(red_count):
                red = _red_track(episode, slot)
                axis.plot(red[:, 0], red[:, 2], color="firebrick", alpha=0.28, linewidth=1.5)
            axis.scatter(blue[0, 0], blue[0, 2], color="green", marker="o", s=35, zorder=4)
            axis.scatter(blue[-1, 0], blue[-1, 2], color="black", marker="x", s=45, zorder=4)
            survived = bool(summary["blue_survived"])
            outcome = "逃脱" if survived else "被命中"
            miss_distance = float(summary["miss_distance_m"])
            axis.set_title(
                f"{VARIANT_LABELS.get(variant, variant)}｜{outcome}｜最近弹距 {miss_distance:.2f} m"
            )
            axis.set_xlabel("北向 x（米）")
            axis.set_ylabel("东向 z（米）")
            axis.grid(alpha=0.2)
            axis.set_aspect("equal", adjustable="box")
            if survived:
                for spine in axis.spines.values():
                    spine.set_color("forestgreen")
                    spine.set_linewidth(2.5)
        axes[0].legend(
            handles=[
                Line2D([0], [0], color="royalblue", linewidth=2, label="蓝机"),
                Line2D([0], [0], color="firebrick", alpha=0.4, label="红方导弹"),
                Line2D([0], [0], marker="o", color="green", linestyle="none", label="起点"),
                Line2D([0], [0], marker="x", color="black", linestyle="none", label="终点"),
            ],
            fontsize=8,
        )
        all_x = np.concatenate([
            np.asarray(line.get_xdata(), dtype=float) for axis in axes for line in axis.lines
        ])
        all_z = np.concatenate([
            np.asarray(line.get_ydata(), dtype=float) for axis in axes for line in axis.lines
        ])
        finite = np.isfinite(all_x) & np.isfinite(all_z)
        x_min, x_max = float(all_x[finite].min()), float(all_x[finite].max())
        z_min, z_max = float(all_z[finite].min()), float(all_z[finite].max())
        x_pad = max(1.0, (x_max - x_min) * 0.03)
        z_pad = max(1.0, (z_max - z_min) * 0.03)
        for axis in axes:
            axis.set_xlim(x_min - x_pad, x_max + x_pad)
            axis.set_ylim(z_min - z_pad, z_max + z_pad)
        figure.suptitle(
            "同一初始场景下的蓝方策略水平轨迹比较\n"
            f"seed={comparison['seed']}，红弹数={comparison['missile_count']}，"
            f"初始状态指纹={comparison['initialization_fingerprint'][:12]}…"
        )
        for text in figure.findobj(match=Text):
            text.set_fontfamily(font_name)
        figure.savefig(path, dpi=dpi)
        figure.clear()


def generate_comparison_plots(
    comparison_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    font_path: str | Path | None = None,
    dpi: int = 160,
) -> dict[str, Any]:
    """Render six native quality panels and one same-scale trajectory overview."""
    if isinstance(dpi, bool) or not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    source = Path(comparison_dir).resolve()
    comparison = _read_json(source / "comparison.json")
    if not bool(comparison.get("identical_initialization_verified")):
        raise ValueError("comparison does not verify identical initialization")
    variants = comparison.get("variant_order")
    if not isinstance(variants, list) or not variants:
        raise ValueError("comparison.json has no variant_order")
    summaries = {
        item["variant"]: item for item in comparison.get("variants", [])
        if isinstance(item, dict) and isinstance(item.get("variant"), str)
    }
    missing = [variant for variant in variants if variant not in summaries]
    if missing:
        raise ValueError(f"comparison summaries are missing variants: {missing}")
    fingerprints = {summaries[variant].get("initialization_fingerprint") for variant in variants}
    if fingerprints != {comparison.get("initialization_fingerprint")}:
        raise ValueError("variant initialization fingerprints do not match comparison.json")

    output = Path(output_dir).resolve() if output_dir is not None else source / "plots"
    output.mkdir(parents=True, exist_ok=True)
    episodes: dict[str, dict[str, Any]] = {}
    individual: dict[str, list[str]] = {}
    for variant in variants:
        episode_source = source / variant / "flight_quality_episodes.jsonl"
        episodes[variant] = _read_single_episode(episode_source)
        result = replot_flight_quality_episodes(
            episode_source,
            output / variant,
            plot_limit=1,
            overwrite=True,
            clear_existing=True,
            font_path=font_path,
        )
        individual[variant] = [str((output / variant / name).resolve()) for name in result["files"]]

    overview = output / "strategy_horizontal_trajectory_overview.png"
    _render_horizontal_overview(
        episodes, summaries, comparison, overview, font_path=font_path, dpi=dpi
    )
    manifest = {
        "source_comparison": str((source / "comparison.json").resolve()),
        "output": str(output.resolve()),
        "generated_from_existing_results": True,
        "simulation_rerun": False,
        "identical_initialization_verified": True,
        "initialization_fingerprint": comparison["initialization_fingerprint"],
        "variant_order": variants,
        "individual_flight_quality_plots": individual,
        "horizontal_trajectory_overview": str(overview.resolve()),
    }
    temporary = output / "plot_manifest.json.tmp"
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "plot_manifest.json")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comparison_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--dpi", type=int, default=160)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = generate_comparison_plots(
        args.comparison_dir, output_dir=args.output, font_path=args.font, dpi=args.dpi
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
