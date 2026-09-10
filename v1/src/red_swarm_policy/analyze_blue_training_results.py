"""Plot grouped Blue training outcomes without modifying the original logs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

from .analyze_training import load_training_data


UNKNOWN_ORIENTATION = "<not recorded>"


def record_episode_orientation(orientations: dict[int, str], episode: int, reset_info: Mapping[str, Any]) -> None:
    """Copy only the orientation already returned by reset; never infer it from flight."""
    initialization = reset_info.get("initialization")
    value = initialization.get("blue_orientation") if isinstance(initialization, Mapping) else None
    if isinstance(value, str) and value.strip():
        orientations[episode] = value


def _metadata_path(source: Path) -> Path:
    return source.with_name(source.stem + "_episode_metadata.json")


def _load_orientations(source: Path, metadata_path: Path) -> dict[int, str]:
    document = json.loads(metadata_path.read_text(encoding="utf-8"))
    if document.get("metrics_sha256") != hashlib.sha256(source.read_bytes()).hexdigest():
        raise ValueError("Episode metadata does not match this training log; refusing stale orientation labels")
    result = {}
    for row in document["episodes"]:
        episode, orientation = row.get("episode"), row.get("blue_orientation")
        if isinstance(episode, bool) or not isinstance(episode, int) or episode < 1:
            raise ValueError("Episode metadata contains an invalid episode ID")
        if episode in result or not isinstance(orientation, str) or not orientation.strip():
            raise ValueError("Episode metadata contains duplicate IDs or invalid orientations")
        result[episode] = orientation
    return result


def analyze_blue_training_results(source: str | Path, output: str | Path | None = None, *,
                                  episode_metadata_path: str | Path | None = None,
                                  dpi: int = 160) -> dict[str, Any]:
    """Reuse the result figures with actual training outcomes and optional reset metadata."""
    from .evaluation_plots import write_evaluation_plots

    data, path = load_training_data(source)
    assert path is not None
    if not data.episodes:
        raise ValueError("Grouped training plots require episode records; use training_metrics.json or episodes.json")
    metadata_path = Path(episode_metadata_path) if episode_metadata_path is not None else _metadata_path(path)
    orientations = _load_orientations(path, metadata_path) if metadata_path.is_file() else {}
    if episode_metadata_path is not None and not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    rows = []
    missing = 0
    for row in data.episodes:
        initialization = row.get("initialization") or {}
        candidates = [row.get("blue_orientation"),
                      initialization.get("blue_orientation") if isinstance(initialization, Mapping) else None,
                      orientations.get(row.get("episode"))]
        orientation = next((value for value in candidates if isinstance(value, str) and value.strip()), None)
        missing += orientation is None
        rows.append({**row, "blue_orientation": orientation or UNKNOWN_ORIENTATION})
    destination = Path(output) if output is not None else path.parent / "training_analysis" / "results"
    report = write_evaluation_plots(destination, [("blue_training", path, rows)], dpi=dpi)
    report["data_stage"] = "training"
    report["orientation_missing_episodes"] = missing
    report["episode_metadata_path"] = str(metadata_path.resolve()) if metadata_path.is_file() else None
    report["notes"].append("Training outcomes include exploration and changing policies; they are not fixed-policy test results.")
    if missing:
        report["notes"].append(f"{missing} episodes have no recorded initial orientation; their orientation group is '{UNKNOWN_ORIENTATION}'.")
    (destination / "plot_statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    return report


def write_blue_training_results_safely(source: str | Path, output: str | Path | None = None, *,
                                       episode_orientations: Mapping[int, str] | None = None,
                                       make_plots: bool = True) -> dict[str, Any] | None:
    """Archive additional reset metadata after original saves; isolate plotting errors."""
    try:
        path = Path(source)
        if episode_orientations is not None:
            # Bind metadata to the exact saved metrics, so an old sidecar from a
            # previous run can never silently relabel a new run's episodes.
            metadata = {"schema_version": 1, "metrics_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "episodes": [{"episode": episode, "blue_orientation": value}
                                     for episode, value in sorted(episode_orientations.items())]}
            _metadata_path(path).write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        if not make_plots:
            return None
        return analyze_blue_training_results(path, output)
    except Exception as error:
        print(f"[blue training results] Analysis skipped: {type(error).__name__}: {error}", file=sys.stderr)
        return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Training metrics JSON, episodes JSON, or run directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--episode-metadata", type=Path, help="Optional saved initial-orientation metadata")
    parser.add_argument("--dpi", type=int, default=160)
    args = parser.parse_args(argv)
    try:
        report = analyze_blue_training_results(args.input, args.output,
                                               episode_metadata_path=args.episode_metadata, dpi=args.dpi)
    except (OSError, ValueError, ImportError) as error:
        print(f"[blue training results] {error}", file=sys.stderr)
        return 1
    print(json.dumps({"figures": report["figures"], "orientation_missing_episodes": report["orientation_missing_episodes"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
