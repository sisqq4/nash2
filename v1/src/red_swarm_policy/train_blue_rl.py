from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
import uuid
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .analysis.analyze_blue_training_load import safe_analyze_blue_training_load
from .analysis.analyze_training import write_training_analysis_safely
from .analysis.analyze_blue_training_results import record_episode_orientation, write_blue_training_results_safely
from .blue_rl import (BlueEscapeEnv, BlueEscapeEnvConfig, BlueProcessEnvironmentPool,
                      FlightEnvelopeConfig, FlightEnvelopeConstraintLayer,
                      FlightQualityTracker, MechanismRewardConfig,
                      RainbowDQNAgent, RainbowDQNConfig,
                      append_flight_quality_episode,
                      blue_observation_dim,
                      parse_mechanism_rewards,
                      write_flight_quality_report)
from .blue_rl.config_io import configure_blue_mission_duration, load_environment_config
from .blue_rl.curriculum import CurriculumSchedule, balanced_score, within_forgetting_limit
from .cli_utils import parse_missile_scenarios
from .env import GUIDANCE_CONTRACT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the independent v1 blue Rainbow-DQN policy")
    parser.add_argument(
        "--episodes", type=int, default=1000,
        help="Total episode target, including episodes restored by --resume-from",
    )
    parser.add_argument("--missiles", default="1",
                        help="Comma-separated scenarios to sample, e.g. 1,2,3,4 (default: 1)")
    parser.add_argument("--seed", type=int, default=0); parser.add_argument("--output", default="outputs/blue_rl/train")
    parser.add_argument("--device", default="cpu"); parser.add_argument("--env-config", default=None)
    parser.add_argument("--red-sector-width-deg", type=float, default=None,
                        help="Red spawn sector width in degrees (0, 360]; default: scenario configuration (60)")
    parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument(
        "--reward-mechanisms", type=parse_mechanism_rewards, default="all",
        metavar="SELECTION",
        help=("Training reward mechanisms: all, none, or a comma-separated subset of "
              "threat,timing,direction,overload (default: all)"),
    )
    parser.add_argument("--parallel-envs", type=int, default=1,
                        help="Persistent CPU environment processes sampled in parallel")
    parser.add_argument("--env-worker-threads", type=int, default=1)
    parser.add_argument("--env-worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--updates-per-transition", type=float, default=1.0,
                        help="Gradient updates scheduled per collected environment transition")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=50_000,
                        help="Maximum number of transitions retained by prioritized replay")
    parser.add_argument("--log-interval", type=int, default=10,
                        help="Print and archive one aggregate row every N completed episodes")
    parser.add_argument("--metrics-path", default=None,
                        help="Final JSON metrics path; defaults to OUTPUT/training_metrics.json")
    parser.add_argument("--jsonl-path", default=None,
                        help="Streaming JSONL path; defaults to OUTPUT/training.jsonl")
    parser.add_argument("--no-training-plots", action="store_true",
                        help="Skip the analysis plots normally generated after training is saved")
    parser.add_argument("--plot-smoothing-window", type=int, default=20,
                        help="Positive trailing smoothing window for training analysis (default: 20)")
    parser.add_argument("--training-plots-dir", default=None,
                        help="Analysis output directory; defaults to OUTPUT/training_analysis")
    parser.add_argument("--acmi-interval", type=int, default=1,
                        help="Save one training ACMI every N episodes; use 0 to disable ACMI output")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument(
        "--checkpoint-eval-episodes", type=int, default=100,
        help=("Fixed holdout episodes evaluated at every checkpoint boundary without "
              "learner updates or per-episode artifacts (default: 100)"),
    )
    parser.add_argument(
        "--resume-from", default=None, metavar="PATH",
        help=("Resume a full blue-training snapshot (.pt), resume manifest (.json), "
              "or the original training output directory"),
    )
    parser.add_argument(
        "--resume-save-interval", type=int, default=100,
        help=("Drain active environments and atomically save a full resumable state "
              "after every N completed episodes; the safe boundary may be a few "
              "episodes later when environments run in parallel (default: 100)"),
    )
    parser.add_argument("--curriculum", action="store_true",
                        help="Use the staged 1v1-to-1v4 rehearsal curriculum (normalized_v4 network)")
    parser.add_argument("--curriculum-transition-episodes", type=int, default=500,
                        help="Episodes used to linearly ramp probabilities at each curriculum stage entry")
    parser.add_argument("--curriculum-eval-interval", type=int, default=500)
    parser.add_argument("--curriculum-eval-episodes", type=int, default=300,
                        help="Fixed-seed evaluation episodes per introduced scenario; 0 disables evaluation")
    parser.add_argument("--flight-quality-plot-limit", type=int, default=10)
    parser.add_argument("--baseline-survival-rate", type=float, default=None)
    return parser


def _training_plot_directories(
    output: str | Path,
    training_plots_dir: str | Path | None,
) -> tuple[Path, Path]:
    """Resolve generic and speed/load post-training plot destinations."""
    training_output = Path(output)
    if training_plots_dir is None:
        return training_output / "training_analysis", training_output / "training_load_analysis"
    analysis_output = Path(training_plots_dir)
    return analysis_output, analysis_output / "speed_load"


def _write_post_training_plots(
    *,
    training_dir: str | Path,
    metrics_path: str | Path,
    training_plots_dir: str | Path | None,
    smoothing_window: int,
) -> tuple[Path, Path]:
    """Run isolated reporting only after durable training artifacts exist."""
    analysis_output, speed_load_output = _training_plot_directories(
        training_dir, training_plots_dir,
    )
    write_training_analysis_safely(
        metrics_path, analysis_output, smoothing_window=smoothing_window,
    )
    safe_analyze_blue_training_load(
        training_dir,
        speed_load_output,
        rolling_window=smoothing_window,
        block_size=100,
        dpi=180,
    )
    return analysis_output, speed_load_output


def _emit(row: dict[str, Any], rows: list[dict[str, Any]], jsonl_path: Path) -> None:
    rows.append(row)
    encoded = json.dumps(row, ensure_ascii=True, allow_nan=False)
    print(encoded, flush=True)
    with jsonl_path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _device_metrics(device: torch.device) -> dict[str, float]:
    if device.type != "cuda": return {}
    return {
        "cuda_memory_allocated_mb": torch.cuda.memory_allocated(device) / 2**20,
        "cuda_memory_reserved_mb": torch.cuda.memory_reserved(device) / 2**20,
        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 2**20,
    }


BLUE_RESUME_SCHEMA_VERSION = 1
BLUE_RESUME_KIND = "blue_rainbow_full_training_state"


def _capture_training_rng_state(device: torch.device | None = None) -> dict[str, object]:
    state: dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
        if device is not None and device.type == "cuda":
            state["torch_cuda_active"] = torch.cuda.get_rng_state(device)
            state["torch_cuda_active_index"] = (
                torch.cuda.current_device() if device.index is None else int(device.index)
            )
    return state


def _restore_training_rng_state(state: dict[str, object], device: torch.device) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    np.random.set_state(state["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(torch.as_tensor(state["torch"], dtype=torch.uint8).cpu())
    cuda_states = state.get("torch_cuda")
    if device.type != "cuda" or not torch.cuda.is_available() or not cuda_states:
        return
    saved = [torch.as_tensor(value, dtype=torch.uint8).cpu() for value in cuda_states]
    if len(saved) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(saved)
    active_state = state.get("torch_cuda_active")
    if active_state is not None:
        torch.cuda.set_rng_state(
            torch.as_tensor(active_state, dtype=torch.uint8).cpu(), device=device
        )
    elif len(saved) != torch.cuda.device_count():
        saved_index = int(state.get("torch_cuda_active_index", 0))
        torch.cuda.set_rng_state(saved[min(saved_index, len(saved) - 1)], device=device)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _resume_entry_path(manifest_path: Path, entry: dict[str, object]) -> Path:
    value = Path(str(entry["path"]))
    candidate = (value if value.is_absolute() else manifest_path.parent / value).resolve()
    resume_directory = manifest_path.parent.resolve()
    try:
        candidate.relative_to(resume_directory)
    except ValueError as error:
        raise ValueError(f"resume entry escapes its directory: {value}") from error
    if (candidate.parent != resume_directory or not candidate.name.startswith("state_")
            or candidate.suffix.lower() != ".pt"):
        raise ValueError(f"invalid resume generation path: {value}")
    return candidate


def _validate_resume_file(path: Path, entry: dict[str, object] | None = None) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"resume snapshot does not exist: {path}")
    if entry is not None:
        expected_size = int(entry.get("size_bytes", -1))
        if expected_size >= 0 and path.stat().st_size != expected_size:
            raise ValueError(f"resume snapshot size mismatch: {path}")
        expected_hash = entry.get("sha256")
        if expected_hash is not None and _sha256(path) != str(expected_hash):
            raise ValueError(f"resume snapshot SHA256 mismatch: {path}")


def _load_resume_snapshot(source: str | Path) -> tuple[dict[str, Any], Path]:
    requested = Path(source)
    if requested.is_dir():
        direct_manifest = requested / "latest.json"
        nested_manifest = requested / "resume" / "latest.json"
        requested = direct_manifest if direct_manifest.is_file() else nested_manifest
    if requested.suffix.lower() != ".json":
        _validate_resume_file(requested)
        payload = torch.load(requested, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict):
            raise ValueError("resume snapshot root must be a dictionary")
        return payload, requested

    manifest = json.loads(requested.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("resume manifest root must be a dictionary")
    errors: list[str] = []
    for label in ("latest", "previous"):
        entry = manifest.get(label)
        if not isinstance(entry, dict):
            continue
        try:
            candidate = _resume_entry_path(requested, entry)
            _validate_resume_file(candidate, entry)
            payload = torch.load(candidate, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise ValueError("snapshot root must be a dictionary")
            return payload, candidate
        except Exception as error:
            errors.append(f"{label}: {error}")
    raise RuntimeError(
        f"no valid resume generation in {requested}: " + "; ".join(errors)
    )


def _save_resume_generation(
    directory: Path,
    payload: dict[str, Any],
    *,
    snapshot_id: str,
    completed_episodes: int,
    reason: str,
    preferred_previous_path: Path | None = None,
) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "latest.json"
    final_path = directory / f"state_{snapshot_id}.pt"
    temporary = directory / f".{final_path.name}.tmp"
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, final_path)
    entry: dict[str, object] = {
        "path": final_path.name,
        "snapshot_id": snapshot_id,
        "completed_episodes": int(completed_episodes),
        "reason": reason,
        "size_bytes": final_path.stat().st_size,
        "sha256": _sha256(final_path),
    }
    previous_manifest: dict[str, object] = {}
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                previous_manifest = loaded
        except (OSError, ValueError):
            previous_manifest = {}
    old_latest = previous_manifest.get("latest")
    old_previous = previous_manifest.get("previous")

    previous_entry: dict[str, object] | None = None
    seen_paths = {final_path.resolve()}
    if preferred_previous_path is not None:
        raw_candidates: list[dict[str, object]] = []
        for manifest_entry in (old_latest, old_previous):
            if not isinstance(manifest_entry, dict):
                continue
            try:
                if (_resume_entry_path(manifest_path, manifest_entry)
                        == preferred_previous_path.resolve()):
                    raw_candidates.append(manifest_entry)
                    break
            except (KeyError, TypeError, ValueError):
                continue
        if not raw_candidates:
            try:
                preferred = preferred_previous_path.resolve()
                preferred.relative_to(directory.resolve())
                _validate_resume_file(preferred)
                raw_candidates.append({
                    "path": preferred.name,
                    "snapshot_id": preferred.stem.removeprefix("state_"),
                    "completed_episodes": -1,
                    "reason": "previous",
                    "size_bytes": preferred.stat().st_size,
                    "sha256": _sha256(preferred),
                })
            except (OSError, ValueError):
                pass
    else:
        raw_candidates = [
            entry for entry in (old_latest, old_previous) if isinstance(entry, dict)
        ]
    for candidate_entry in raw_candidates:
        try:
            candidate_path = _resume_entry_path(manifest_path, candidate_entry)
            _validate_resume_file(candidate_path, candidate_entry)
        except (OSError, KeyError, TypeError, ValueError):
            continue
        if candidate_path.resolve() in seen_paths:
            continue
        previous_entry = dict(candidate_entry)
        seen_paths.add(candidate_path.resolve())
        break

    manifest: dict[str, object] = {
        "schema_version": BLUE_RESUME_SCHEMA_VERSION,
        "kind": BLUE_RESUME_KIND,
        "latest": entry,
        "previous": previous_entry,
    }
    _atomic_write_json(manifest_path, manifest)

    keep = set(seen_paths)
    for obsolete_entry in (old_latest, old_previous):
        if not isinstance(obsolete_entry, dict):
            continue
        try:
            obsolete = _resume_entry_path(manifest_path, obsolete_entry)
        except (KeyError, TypeError, ValueError):
            continue
        if obsolete.resolve() not in keep:
            obsolete.unlink(missing_ok=True)
    return final_path, manifest_path


def _truncate_committed_log(path: Path, committed_bytes: int) -> None:
    if committed_bytes < 0:
        raise ValueError("committed log byte count must be non-negative")
    if not path.exists():
        if committed_bytes == 0:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"")
            return
        raise FileNotFoundError(f"resume log is missing: {path}")
    if path.stat().st_size < committed_bytes:
        raise ValueError(f"resume log is shorter than its committed offset: {path}")
    with path.open("r+b") as stream:
        stream.truncate(committed_bytes)


def _resume_contract(
    *,
    args: argparse.Namespace,
    training_scenarios: tuple[int, ...],
    curriculum: CurriculumSchedule | None,
    environment_config: Any,
    env_config: BlueEscapeEnvConfig,
    rainbow_config: RainbowDQNConfig,
) -> dict[str, Any]:
    """Describe every setting that changes the training trajectory."""
    rainbow = asdict(rainbow_config)
    rainbow.pop("device", None)
    blue_environment = asdict(env_config)
    acmi_directory = blue_environment.get("acmi_directory")
    if acmi_directory:
        blue_environment["acmi_directory"] = str(Path(str(acmi_directory)).resolve())
    return {
        "seed": int(args.seed),
        "missile_scenarios": list(training_scenarios),
        "updates_per_transition": float(args.updates_per_transition),
        # Draining at each full-state boundary changes parallel batch
        # composition, so this cadence is part of the learning trajectory.
        "resume_save_interval": int(args.resume_save_interval),
        "checkpoint_interval": int(args.checkpoint_interval),
        "checkpoint_eval_episodes": int(args.checkpoint_eval_episodes),
        "curriculum_eval_interval": int(args.curriculum_eval_interval),
        "curriculum_eval_episodes": int(args.curriculum_eval_episodes),
        "curriculum": curriculum.describe() if curriculum is not None else None,
        "environment_config": asdict(environment_config),
        "blue_environment_config": blue_environment,
        "rainbow_config": rainbow,
    }


def _resume_int(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"resume {name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"resume {name} must be at least {minimum}")
    return result


def _validate_resume_artifacts(
    output: Path,
    retained_checkpoints: list[dict[str, Any]],
    artifacts: object,
) -> None:
    if not isinstance(artifacts, dict):
        raise ValueError("resume snapshot is missing artifacts")
    root = output.resolve()
    allowed_special = {"best_new_scenario.pt", "best_balanced.pt"}
    artifact_keys: set[str] = set()
    for relative_name, contents in artifacts.items():
        if not isinstance(relative_name, str) or not isinstance(contents, bytes):
            raise ValueError("resume artifacts must map relative paths to bytes")
        relative = Path(relative_name)
        destination = (root / relative).resolve()
        try:
            normalized = destination.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"resume artifact escapes the output directory: {relative_name}") from error
        is_checkpoint = (
            len(relative.parts) == 2
            and relative.parts[0] == "best_checkpoints"
            and relative.name.startswith("blue_rainbow_ep")
            and relative.name.removeprefix("blue_rainbow_ep").removesuffix(".pt").isdigit()
            and relative.suffix.lower() == ".pt"
        )
        if not is_checkpoint and normalized not in allowed_special:
            raise ValueError(f"unsupported resume artifact path: {relative_name}")
        if normalized in artifact_keys:
            raise ValueError(f"duplicate resume artifact path: {relative_name}")
        artifact_keys.add(normalized)

    retained_keys: set[str] = set()
    for index, candidate in enumerate(retained_checkpoints):
        required_candidate = {
            "completed_episodes", "episode_thresholds", "final_state",
            "checkpoint_path", "validation_episodes", "survived_count",
            "survival_rate", "successful_evasion_count", "timeout_victory_count",
            "failure_count", "termination_counts", "scenario_counts",
            "scenario_survival_rates", "environment_transitions",
            "optimizer_updates", "target_updates",
        }
        if not isinstance(candidate, dict) or not required_candidate.issubset(candidate):
            raise ValueError(f"retained checkpoint {index} is malformed")
        candidate_completed = _resume_int(
            candidate["completed_episodes"], f"retained checkpoint {index} episode"
        )
        validation_episodes = _resume_int(
            candidate["validation_episodes"],
            f"retained checkpoint {index} validation_episodes",
            minimum=1,
        )
        survived = _resume_int(
            candidate["survived_count"], f"retained checkpoint {index} survived_count"
        )
        failed = _resume_int(
            candidate["failure_count"], f"retained checkpoint {index} failure_count"
        )
        if survived + failed != validation_episodes:
            raise ValueError(f"retained checkpoint {index} validation counts are inconsistent")
        for name in (
            "successful_evasion_count", "timeout_victory_count",
            "environment_transitions", "optimizer_updates", "target_updates",
        ):
            _resume_int(candidate[name], f"retained checkpoint {index} {name}")
        if not isinstance(candidate["final_state"], bool):
            raise TypeError(f"retained checkpoint {index} final_state must be boolean")
        if not isinstance(candidate["episode_thresholds"], list):
            raise TypeError(f"retained checkpoint {index} thresholds must be a list")
        for threshold in candidate["episode_thresholds"]:
            _resume_int(threshold, f"retained checkpoint {index} threshold", minimum=1)
        for name in ("termination_counts", "scenario_counts", "scenario_survival_rates"):
            if not isinstance(candidate[name], dict):
                raise TypeError(f"retained checkpoint {index} {name} must be a dictionary")
        survival_rate = float(candidate["survival_rate"])
        if not math.isfinite(survival_rate) or not 0.0 <= survival_rate <= 1.0:
            raise ValueError(f"retained checkpoint {index} survival_rate is invalid")
        checkpoint_path = Path(str(candidate["checkpoint_path"])).resolve()
        try:
            relative = checkpoint_path.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"retained checkpoint {index} is outside the output directory"
            ) from error
        key = relative.as_posix()
        valid_name = (
            len(relative.parts) == 2
            and relative.parts[0] == "best_checkpoints"
            and relative.name.startswith("blue_rainbow_ep")
            and relative.name.removeprefix("blue_rainbow_ep").removesuffix(".pt").isdigit()
            and relative.suffix.lower() == ".pt"
        )
        if not valid_name:
            raise ValueError(f"retained checkpoint {index} has an invalid path")
        expected_name = f"blue_rainbow_ep{candidate_completed:06d}.pt"
        if relative.name != expected_name:
            raise ValueError(f"retained checkpoint {index} path does not match its episode")
        retained_keys.add(key)
    if len(retained_keys) != len(retained_checkpoints):
        raise ValueError("retained checkpoint paths must be unique")
    checkpoint_artifact_keys = {
        key for key in artifact_keys if key.startswith("best_checkpoints/")
    }
    if retained_keys != checkpoint_artifact_keys:
        raise ValueError("retained checkpoints and embedded checkpoint artifacts differ")


def _validate_resume_payload(
    payload: dict[str, Any],
    *,
    expected_contract: dict[str, Any],
    output: Path,
    target_episodes: int,
) -> dict[str, Any]:
    required_root = {
        "schema_version", "kind", "snapshot_id", "reason", "output_directory",
        "critical_config", "runtime_config", "agent_state", "rng_state",
        "training_state", "artifacts",
    }
    missing_root = required_root - set(payload)
    extra_root = set(payload) - required_root
    if missing_root or extra_root:
        raise ValueError(
            f"invalid resume snapshot keys: missing={sorted(missing_root)}, "
            f"extra={sorted(extra_root)}"
        )
    if payload.get("schema_version") != BLUE_RESUME_SCHEMA_VERSION:
        raise ValueError(
            "unsupported blue resume schema version: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("kind") != BLUE_RESUME_KIND:
        raise ValueError(
            "--resume-from requires a full blue-training snapshot; "
            "a lightweight policy checkpoint cannot continue replay/optimizer state"
        )
    if not isinstance(payload.get("snapshot_id"), str) or not payload["snapshot_id"]:
        raise ValueError("resume snapshot_id must be a non-empty string")
    if payload.get("reason") not in {"periodic", "final"}:
        raise ValueError(f"unsupported resume snapshot reason: {payload.get('reason')!r}")
    saved_output = Path(str(payload.get("output_directory", ""))).resolve()
    if saved_output != output.resolve():
        raise ValueError(
            f"resume output directory mismatch: saved={saved_output}, current={output.resolve()}"
        )
    saved_contract = payload.get("critical_config")
    if not isinstance(saved_contract, dict):
        raise ValueError("resume snapshot is missing critical_config")
    if saved_contract != expected_contract:
        differing = sorted(
            key for key in set(saved_contract) | set(expected_contract)
            if saved_contract.get(key) != expected_contract.get(key)
        )
        raise ValueError(
            "resume training configuration mismatch for: " + ", ".join(differing)
        )
    runtime_config = payload.get("runtime_config")
    if not isinstance(runtime_config, dict) or set(runtime_config) != {
        "device", "parallel_envs", "env_worker_threads",
    }:
        raise ValueError("resume runtime_config is malformed")
    if not isinstance(runtime_config["device"], str):
        raise TypeError("resume runtime device must be a string")
    _resume_int(runtime_config["parallel_envs"], "runtime parallel_envs", minimum=1)
    _resume_int(
        runtime_config["env_worker_threads"], "runtime env_worker_threads", minimum=1
    )
    training_state = payload.get("training_state")
    if not isinstance(training_state, dict):
        raise ValueError("resume snapshot is missing training_state")
    required_training = {
        "completed_episodes", "next_episode", "next_checkpoint", "next_log",
        "next_curriculum_eval", "next_resume_save", "resume_generation",
        "requested_resume_thresholds", "actual_completed_episodes",
        "vector_iterations", "update_credit", "elapsed_s", "window_elapsed_s",
        "window_transition_start", "window_update_start", "window_episodes",
        "window_losses", "window_grad_norms", "window_values", "window_actions",
        "window_policy_actions", "window_constraint_interventions",
        "window_clamp_low", "window_clamp_high", "historical_best",
        "best_balanced_score", "best_new_rate", "summaries", "quality_episodes",
        "episode_orientations", "event_rows", "checkpoint_evaluations",
        "retained_checkpoints", "current_state_already_validated", "log_paths",
        "log_offsets",
    }
    missing_training = required_training - set(training_state)
    extra_training = set(training_state) - required_training
    if missing_training or extra_training:
        raise ValueError(
            f"invalid resume training_state keys: missing={sorted(missing_training)}, "
            f"extra={sorted(extra_training)}"
        )
    completed = _resume_int(training_state["completed_episodes"], "completed_episodes")
    if target_episodes < completed:
        raise ValueError(
            f"--episodes is the total target and must be at least {completed} when resuming"
        )
    if not isinstance(payload.get("agent_state"), dict):
        raise ValueError("resume snapshot is missing agent_state")
    if not isinstance(payload.get("rng_state"), dict):
        raise ValueError("resume snapshot is missing rng_state")
    rng_state = payload["rng_state"]
    if not {"python", "numpy", "torch"}.issubset(rng_state):
        raise ValueError("resume RNG state is incomplete")
    allowed_rng_keys = {
        "python", "numpy", "torch", "torch_cuda", "torch_cuda_active",
        "torch_cuda_active_index",
    }
    if set(rng_state) - allowed_rng_keys:
        raise ValueError("resume RNG state has unsupported keys")
    if not isinstance(rng_state["python"], tuple) or not isinstance(rng_state["numpy"], tuple):
        raise ValueError("resume Python/NumPy RNG states are malformed")
    if (not isinstance(rng_state["torch"], torch.Tensor)
            or rng_state["torch"].dtype != torch.uint8
            or rng_state["torch"].ndim != 1):
        raise ValueError("resume Torch RNG state is malformed")
    try:
        random.Random().setstate(rng_state["python"])
        np.random.RandomState().set_state(rng_state["numpy"])
        torch.Generator(device="cpu").set_state(rng_state["torch"].cpu())
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("resume CPU RNG state cannot be restored") from error
    cuda_states = rng_state.get("torch_cuda")
    if cuda_states is not None and (
        not isinstance(cuda_states, list)
        or any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.uint8
            or value.ndim != 1
            for value in cuda_states
        )
    ):
        raise ValueError("resume CUDA RNG states are malformed")
    active_cuda_state = rng_state.get("torch_cuda_active")
    active_cuda_index = rng_state.get("torch_cuda_active_index")
    if (active_cuda_state is None) != (active_cuda_index is None):
        raise ValueError("resume active CUDA RNG state/index must appear together")
    if active_cuda_state is not None:
        if not cuda_states:
            raise ValueError("resume active CUDA RNG state requires CUDA RNG states")
        if (not isinstance(active_cuda_state, torch.Tensor)
                or active_cuda_state.dtype != torch.uint8
                or active_cuda_state.ndim != 1):
            raise ValueError("resume active CUDA RNG state is malformed")
        _resume_int(active_cuda_index, "active CUDA device index")

    next_episode = _resume_int(training_state["next_episode"], "next_episode", minimum=1)
    if next_episode != completed + 1:
        raise ValueError("resume next_episode must equal completed_episodes + 1")
    for name in (
        "next_checkpoint", "next_log", "next_curriculum_eval", "next_resume_save",
        "resume_generation", "vector_iterations", "window_transition_start",
        "window_update_start", "window_constraint_interventions",
    ):
        _resume_int(training_state[name], name, minimum=0)
    if _resume_int(training_state["resume_generation"], "resume_generation") < 1:
        raise ValueError("resume_generation must be positive")
    if _resume_int(training_state["next_checkpoint"], "next_checkpoint", minimum=1) <= completed:
        raise ValueError("resume next_checkpoint must be after completed_episodes")
    if _resume_int(training_state["next_log"], "next_log", minimum=1) <= completed:
        raise ValueError("resume next_log must be after completed_episodes")
    if _resume_int(training_state["next_resume_save"], "next_resume_save", minimum=1) <= completed:
        raise ValueError("resume next_resume_save must be after completed_episodes")
    if _resume_int(
        training_state["actual_completed_episodes"], "actual_completed_episodes"
    ) != completed:
        raise ValueError("resume actual_completed_episodes is inconsistent")
    update_credit = float(training_state["update_credit"])
    if not math.isfinite(update_credit) or not 0.0 <= update_credit < 1.0:
        raise ValueError("resume update_credit must be finite and in [0, 1)")
    for name in ("elapsed_s", "window_elapsed_s"):
        value = float(training_state[name])
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"resume {name} must be finite and non-negative")
    thresholds = training_state["requested_resume_thresholds"]
    if not isinstance(thresholds, list):
        raise TypeError("resume requested_resume_thresholds must be a list")
    parsed_thresholds = [
        _resume_int(value, "requested_resume_threshold", minimum=1) for value in thresholds
    ]
    if parsed_thresholds != sorted(set(parsed_thresholds)) or any(
        value > completed for value in parsed_thresholds
    ):
        raise ValueError("resume requested thresholds must be unique, sorted, and completed")
    list_fields = (
        "window_episodes", "window_losses", "window_grad_norms", "window_values",
        "window_clamp_low", "window_clamp_high", "summaries", "quality_episodes",
        "event_rows", "checkpoint_evaluations", "retained_checkpoints",
    )
    for name in list_fields:
        if not isinstance(training_state[name], list):
            raise TypeError(f"resume {name} must be a list")
    dict_fields = (
        "window_actions", "window_policy_actions", "historical_best", "best_new_rate",
        "episode_orientations", "log_paths", "log_offsets",
    )
    for name in dict_fields:
        if not isinstance(training_state[name], dict):
            raise TypeError(f"resume {name} must be a dictionary")
    if not isinstance(training_state["current_state_already_validated"], bool):
        raise TypeError("resume current_state_already_validated must be boolean")
    summary_ids = []
    for row in training_state["summaries"]:
        if not isinstance(row, dict) or "episode" not in row:
            raise ValueError("resume summaries contain a malformed row")
        summary_ids.append(_resume_int(row["episode"], "summary episode", minimum=1))
    if sorted(summary_ids) != list(range(1, completed + 1)):
        raise ValueError("resume summaries must contain every completed episode exactly once")
    quality_ids = []
    for row in training_state["quality_episodes"]:
        if not isinstance(row, dict) or "episode" not in row:
            raise ValueError("resume quality_episodes contain a malformed row")
        quality_ids.append(_resume_int(row["episode"], "quality episode", minimum=1))
    if sorted(quality_ids) != list(range(1, completed + 1)):
        raise ValueError("resume flight-quality rows must match completed episodes")
    window_ids = []
    for row in training_state["window_episodes"]:
        if not isinstance(row, dict) or "episode" not in row:
            raise ValueError("resume window_episodes contain a malformed row")
        window_ids.append(_resume_int(row["episode"], "window episode", minimum=1))
    if len(window_ids) != len(set(window_ids)) or not set(window_ids).issubset(summary_ids):
        raise ValueError("resume window episodes must be unique completed episodes")
    log_offsets = training_state["log_offsets"]
    if set(log_offsets) != {"training_jsonl_bytes", "flight_quality_jsonl_bytes"}:
        raise ValueError("resume log_offsets has invalid keys")
    for name, value in log_offsets.items():
        _resume_int(value, name)
    retained = training_state["retained_checkpoints"]
    _validate_resume_artifacts(output, retained, payload["artifacts"])
    return training_state


def _collect_resume_artifacts(output: Path, paths: list[Path]) -> dict[str, bytes]:
    root = output.resolve()
    artifacts: dict[str, bytes] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"resume artifact is missing: {path}")
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ValueError(f"resume artifact is outside the output directory: {path}") from error
        artifacts[relative.as_posix()] = path.read_bytes()
    return artifacts


def _restore_resume_artifacts(output: Path, artifacts: object) -> None:
    if not isinstance(artifacts, dict):
        raise ValueError("resume artifacts must be a dictionary")
    root = output.resolve()
    validated: list[tuple[Path, bytes]] = []
    for relative_name, contents in artifacts.items():
        if not isinstance(relative_name, str) or not isinstance(contents, bytes):
            raise ValueError("resume artifact entries must map relative paths to bytes")
        destination = (root / Path(relative_name)).resolve()
        try:
            relative = destination.relative_to(root)
        except ValueError as error:
            raise ValueError(f"resume artifact escapes the output directory: {relative_name}") from error
        is_checkpoint = (
            len(relative.parts) == 2
            and relative.parts[0] == "best_checkpoints"
            and relative.name.startswith("blue_rainbow_ep")
            and relative.name.removeprefix("blue_rainbow_ep").removesuffix(".pt").isdigit()
            and relative.suffix.lower() == ".pt"
        )
        if not is_checkpoint and relative.as_posix() not in {
            "best_new_scenario.pt", "best_balanced.pt",
        }:
            raise ValueError(f"unsupported resume artifact path: {relative_name}")
        validated.append((destination, contents))

    # These files are derived from the saved leaderboard/curriculum state.  A
    # rollback removes files created after the chosen safe generation.
    best_directory = output / "best_checkpoints"
    for path in best_directory.glob("blue_rainbow_ep*.pt"):
        path.unlink()
    for name in ("best_new_scenario.pt", "best_balanced.pt"):
        (output / name).unlink(missing_ok=True)
    for destination, contents in validated:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".resume.tmp")
        with temporary.open("wb") as stream:
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)


def _build_checkpoint_validation_plan(
    *, seed: int, episodes: int, missile_scenarios: tuple[int, ...]
) -> list[dict[str, int]]:
    """Build one balanced, fixed holdout plan shared by every checkpoint."""
    if episodes < 1:
        raise ValueError("checkpoint validation episodes must be positive")
    if not missile_scenarios:
        raise ValueError("checkpoint validation requires at least one missile scenario")
    counts = [int(missile_scenarios[index % len(missile_scenarios)])
              for index in range(episodes)]
    random.Random(seed).shuffle(counts)
    return [
        {"episode": index + 1, "seed": seed + index + 1, "missile_count": count}
        for index, count in enumerate(counts)
    ]


def _checkpoint_validation_score(result: dict[str, Any]) -> tuple[int, int, int]:
    """Rank survival first, then decisive evasions, then fewer timeout victories."""
    return (
        int(result["survived_count"]),
        int(result["successful_evasion_count"]),
        -int(result["timeout_victory_count"]),
    )


def _select_retained_checkpoints(
    candidates: list[dict[str, Any]], limit: int = 3,
) -> list[dict[str, Any]]:
    """Return the best ``limit`` candidates plus exact ties at the cutoff."""
    if limit < 1:
        raise ValueError("checkpoint retention limit must be positive")
    ordered = sorted(candidates, key=_checkpoint_validation_score, reverse=True)
    if len(ordered) <= limit:
        return ordered
    cutoff = _checkpoint_validation_score(ordered[limit - 1])
    return [item for item in ordered if _checkpoint_validation_score(item) >= cutoff]


def _checkpoint_validation_learner_state(agent: RainbowDQNAgent) -> tuple[object, ...]:
    return (
        agent.total_steps,
        agent.optimizer_updates,
        agent.target_updates,
        agent.replay.size,
        tuple(parameter._version for parameter in agent.online.parameters()),
        tuple(parameter._version for parameter in agent.target.parameters()),
    )


def _evaluate_checkpoint_without_updates(
    agent: RainbowDQNAgent,
    pool: BlueProcessEnvironmentPool,
    plan: list[dict[str, int]],
    envelope_config: FlightEnvelopeConfig,
) -> dict[str, Any]:
    """Run a compact deterministic holdout and return aggregates only."""
    if not plan:
        raise ValueError("checkpoint validation plan must not be empty")
    learner_state = _checkpoint_validation_learner_state(agent)
    online_training = bool(agent.online.training)
    started = time.monotonic()
    observations: dict[int, np.ndarray] = {}
    learning_active: dict[int, bool] = {}
    envelope_states: dict[int, dict[str, object]] = {}
    trial_by_worker: dict[int, int] = {}
    constraints = {
        worker: FlightEnvelopeConstraintLayer(envelope_config)
        for worker in range(min(pool.size, len(plan)))
    }
    survived_count = successful_evasion_count = timeout_victory_count = 0
    termination_counts: Counter[str] = Counter()
    scenario_counts: Counter[int] = Counter()
    scenario_survived: Counter[int] = Counter()
    next_trial = 0
    try:
        assignments: dict[int, tuple[int, int, int]] = {}
        for worker in constraints:
            trial = plan[next_trial]
            assignments[worker] = (
                int(trial["seed"]), int(trial["episode"]), int(trial["missile_count"])
            )
            trial_by_worker[worker] = next_trial
            constraints[worker].reset()
            next_trial += 1
        for worker, (observation, info) in pool.reset(assignments).items():
            observations[worker] = observation
            learning_active[worker] = bool(info["learning_active"])
            envelope_states[worker] = dict(info["flight_quality_state"])

        while observations:
            workers = sorted(observations)
            active_workers = [worker for worker in workers if learning_active[worker]]
            actions = {worker: 0 for worker in workers}
            policy_actions = {worker: 0 for worker in workers}
            diagnostics: dict[int, dict[str, object]] = {}
            if active_workers:
                q_values = agent.expected_action_values(
                    np.stack([observations[worker] for worker in active_workers]),
                    evaluation=True,
                )
                for worker, values in zip(active_workers, q_values):
                    action, diagnostic = constraints[worker].select(
                        values, envelope_states[worker]
                    )
                    actions[worker] = int(action)
                    policy_actions[worker] = int(diagnostic["raw_action"])
                    diagnostics[worker] = diagnostic
            action_masks = {
                worker: np.asarray(
                    diagnostics[worker]["effective_action_mask"], dtype=bool
                ) if worker in diagnostics else np.ones(29, dtype=bool)
                for worker in workers
            }
            fallback_flags = {
                worker: bool(diagnostics.get(worker, {}).get("fallback_required", False))
                for worker in workers
            }
            results = pool.step(
                actions,
                policy_actions=policy_actions,
                action_masks=action_masks,
                fallback_required=fallback_flags,
            )
            resets: dict[int, tuple[int, int, int]] = {}
            for worker in workers:
                result = results[worker]
                observations[worker] = result.observation
                learning_active[worker] = bool(result.info["learning_active"])
                envelope_states[worker] = dict(result.info["flight_quality_state"])
                if not (result.terminated or result.truncated):
                    continue
                trial = plan[trial_by_worker[worker]]
                missile_count = int(trial["missile_count"])
                survived = bool(result.info["blue_survived"])
                reason = str(result.info.get("termination_reason", "none"))
                scenario_counts[missile_count] += 1
                termination_counts[reason] += 1
                if survived:
                    survived_count += 1
                    scenario_survived[missile_count] += 1
                    successful_evasion_count += int(reason == "red_failure")
                    timeout_victory_count += int(reason == "timeout")
                if next_trial < len(plan):
                    following = plan[next_trial]
                    resets[worker] = (
                        int(following["seed"]),
                        int(following["episode"]),
                        int(following["missile_count"]),
                    )
                    trial_by_worker[worker] = next_trial
                    constraints[worker].reset()
                    next_trial += 1
                else:
                    observations.pop(worker)
                    learning_active.pop(worker)
                    envelope_states.pop(worker)
                    trial_by_worker.pop(worker)
            if resets:
                for worker, (observation, info) in pool.reset(resets).items():
                    observations[worker] = observation
                    learning_active[worker] = bool(info["learning_active"])
                    envelope_states[worker] = dict(info["flight_quality_state"])
                    constraints[worker].reset()
    finally:
        agent.online.train(online_training)

    if _checkpoint_validation_learner_state(agent) != learner_state:
        raise RuntimeError("checkpoint validation modified learner state")
    total = len(plan)
    return {
        "validation_episodes": total,
        "survived_count": survived_count,
        "survival_rate": survived_count / total,
        "successful_evasion_count": successful_evasion_count,
        "timeout_victory_count": timeout_victory_count,
        "failure_count": total - survived_count,
        "termination_counts": dict(sorted(termination_counts.items())),
        "scenario_counts": {str(key): value for key, value in sorted(scenario_counts.items())},
        "scenario_survival_rates": {
            str(key): scenario_survived[key] / value
            for key, value in sorted(scenario_counts.items())
        },
        "elapsed_s": time.monotonic() - started,
    }


def _curriculum_evaluation(agent: RainbowDQNAgent, environment_config: Any,
                           env_config: BlueEscapeEnvConfig, scenarios: list[int],
                           episodes: int, seed: int) -> dict[str, Any]:
    """Evaluate without replay writes, optimizer steps, or NoisyNet exploration."""
    evaluation_env = BlueEscapeEnv(environment_config, replace(
        env_config, record_acmi=False, acmi_episode_interval=0,
    ))
    rates: dict[int, float] = {}; mean_times: dict[int, float] = {}
    saved_envelope = agent.config.flight_envelope_config
    envelope_config = (FlightEnvelopeConfig(**saved_envelope) if saved_envelope else
                       FlightEnvelopeConfig(action_prediction_s=env_config.decision_interval_s))
    for scenario in scenarios:
        survived: list[float] = []; times: list[float] = []
        for index in range(episodes):
            observation, reset_info = evaluation_env.reset(
                seed=seed + scenario * 100_000 + index, episode_index=index + 1,
                missile_count=scenario,
            )
            learning_active = bool(reset_info["learning_active"])
            mechanism_state = dict(reset_info["flight_quality_state"])
            constraint_layer = FlightEnvelopeConstraintLayer(envelope_config)
            done = False
            while not done:
                if learning_active:
                    q_values = agent.expected_action_values(observation, evaluation=True)[0]
                    action, diagnostic = constraint_layer.select(q_values, mechanism_state)
                    policy_action = int(diagnostic["raw_action"])
                else:
                    action = policy_action = 0
                observation, _, terminated, truncated, info = evaluation_env.step(
                    action, policy_action=policy_action,
                    action_mask=np.asarray(diagnostic["effective_action_mask"], dtype=bool)
                    if learning_active else None,
                    fallback_required=bool(diagnostic.get("fallback_required", False))
                    if learning_active else False,
                )
                learning_active = bool(info["learning_active"])
                mechanism_state = dict(info["flight_quality_state"])
                done = terminated or truncated
            survived.append(float(info["blue_survived"])); times.append(float(info.get("time_s", 0.0)))
        rates[scenario] = float(np.mean(survived)); mean_times[scenario] = float(np.mean(times))
    return {"survival_rates": rates, "mean_survival_time_s": mean_times,
            "episodes_per_scenario": episodes, "seed_base": seed}


def main() -> int:
    args = build_parser().parse_args()
    if args.plot_smoothing_window < 1:
        raise SystemExit("--plot-smoothing-window must be positive")
    try: missile_scenarios = parse_missile_scenarios(args.missiles)
    except ValueError as error: raise SystemExit(str(error)) from error
    if (args.episodes < 1 or args.checkpoint_interval < 1 or args.log_interval < 1
            or args.checkpoint_eval_episodes < 1 or args.resume_save_interval < 1):
        raise SystemExit(
            "episodes, checkpoint interval, log interval, and checkpoint evaluation episodes "
            "and resume save interval must be positive"
        )
    if args.acmi_interval < 0 or args.parallel_envs < 1 or args.env_worker_threads < 1:
        raise SystemExit("ACMI interval must be non-negative and worker counts positive")
    if (args.env_worker_timeout_s <= 0 or args.updates_per_transition < 0
            or args.batch_size < 1 or args.replay_size < 1):
        raise SystemExit("timeout, batch size, and replay size must be positive; update ratio must be non-negative")
    curriculum = CurriculumSchedule(transition_episodes=args.curriculum_transition_episodes) if args.curriculum else None
    if curriculum is not None and args.episodes > curriculum.total_episodes:
        raise SystemExit(f"curriculum defines at most {curriculum.total_episodes} episodes")
    if args.curriculum_eval_interval < 1 or args.curriculum_eval_episodes < 0:
        raise SystemExit("curriculum evaluation interval must be positive and episodes non-negative")
    if args.flight_quality_plot_limit < 0: raise SystemExit("flight-quality plot limit must be non-negative")
    if args.baseline_survival_rate is not None and not 0.0 <= args.baseline_survival_rate <= 1.0:
        raise SystemExit("baseline survival rate must be in [0, 1]")
    if args.red_sector_width_deg is not None and not 0.0 < args.red_sector_width_deg <= 360.0:
        raise SystemExit("--red-sector-width-deg must be in (0, 360]")
    random.seed(args.seed); np.random.seed(args.seed)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    metrics_path = Path(args.metrics_path) if args.metrics_path else output / "training_metrics.json"
    jsonl_path = Path(args.jsonl_path) if args.jsonl_path else output / "training.jsonl"
    flight_quality_jsonl_path = output / "flight_quality" / "flight_quality_episodes.jsonl"
    metrics_path.parent.mkdir(parents=True, exist_ok=True); jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    flight_quality_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    environment_config = configure_blue_mission_duration(load_environment_config(args.env_config))
    if args.red_sector_width_deg is not None:
        environment_config = replace(environment_config, scenario=replace(
            environment_config.scenario, red_sector_width_deg=args.red_sector_width_deg,
        ))
        environment_config.validate()
    training_scenarios = (1, 2, 3, 4) if curriculum is not None else missile_scenarios
    mechanism_reward_config = MechanismRewardConfig().with_mechanisms(args.reward_mechanisms)
    env_config = BlueEscapeEnvConfig(
        training_scenarios[0], max_missiles=max(training_scenarios),
        pad_observation_to_max_missiles=curriculum is not None or len(training_scenarios) > 1,
        observation_schema="normalized_v4",
        decision_interval_s=args.decision_interval,
        acmi_episode_interval=args.acmi_interval,
        acmi_directory=str(output / "acmi"),
        mechanism_reward=mechanism_reward_config,
    )
    checkpoint_eval_env_config = replace(
        env_config,
        record_acmi=False,
        acmi_episode_interval=0,
        record_red_telemetry=False,
    )
    checkpoint_eval_seed = args.seed + 20_000_000
    checkpoint_validation_plan = _build_checkpoint_validation_plan(
        seed=checkpoint_eval_seed,
        episodes=args.checkpoint_eval_episodes,
        missile_scenarios=tuple(training_scenarios),
    )
    envelope_config = FlightEnvelopeConfig(action_prediction_s=args.decision_interval)
    rainbow_config = RainbowDQNConfig(
        blue_observation_dim(env_config.observation_schema, max(training_scenarios)), 29,
        observation_schema=env_config.observation_schema,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        gamma=env_config.shaping_discount, device=args.device,
        atoms=61, value_min=-14.0, value_max=12.0,
        flight_envelope_config=asdict(envelope_config),
        mechanism_reward_config=asdict(env_config.mechanism_reward),
    )
    critical_config = _resume_contract(
        args=args,
        training_scenarios=tuple(training_scenarios),
        curriculum=curriculum,
        environment_config=environment_config,
        env_config=env_config,
        rainbow_config=rainbow_config,
    )
    resume_payload: dict[str, Any] | None = None
    resume_path: Path | None = None
    resume_snapshot_id: str | None = None
    resume_rng_state: dict[str, object] | None = None
    resume_runtime_config: dict[str, Any] | None = None
    restored_training_state: dict[str, Any] | None = None
    resume_log_offsets: dict[str, Any] | None = None
    if args.resume_from is not None:
        try:
            resume_payload, resume_path = _load_resume_snapshot(args.resume_from)
            restored_training_state = _validate_resume_payload(
                resume_payload,
                expected_contract=critical_config,
                output=output,
                target_episodes=args.episodes,
            )
            resume_snapshot_id = str(resume_payload["snapshot_id"])
            resume_runtime_config = dict(resume_payload["runtime_config"])
            saved_log_paths = restored_training_state.get("log_paths")
            expected_log_paths = {
                "training_jsonl": str(jsonl_path.resolve()),
                "flight_quality_jsonl": str(flight_quality_jsonl_path.resolve()),
            }
            if saved_log_paths != expected_log_paths:
                raise ValueError(
                    "resume log path mismatch; use the original output/jsonl paths"
                )
            resume_log_offsets = restored_training_state.get("log_offsets")
            if not isinstance(resume_log_offsets, dict):
                raise ValueError("resume snapshot is missing committed log offsets")
            int(resume_log_offsets["training_jsonl_bytes"])
            int(resume_log_offsets["flight_quality_jsonl_bytes"])
        except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
            raise SystemExit(f"cannot resume blue training: {error}") from error
    else:
        jsonl_path.write_text("", encoding="utf-8")
        flight_quality_jsonl_path.write_text("", encoding="utf-8")
        resume_directory = output / "resume"
        if resume_directory.is_dir():
            for stale_snapshot in resume_directory.glob("state_*.pt"):
                stale_snapshot.unlink()
            for stale_temporary in resume_directory.glob(".state_*.pt.tmp"):
                stale_temporary.unlink()
            (resume_directory / "latest.json").unlink(missing_ok=True)
            (resume_directory / "latest.json.tmp").unlink(missing_ok=True)

    restored_completed = (
        int(restored_training_state["completed_episodes"])
        if restored_training_state is not None else 0
    )
    remaining_episodes = max(args.episodes - restored_completed, 1)
    pool_size = min(args.parallel_envs, remaining_episodes)
    # Spawn CPU simulation workers before creating a CUDA context.  On Windows,
    # starting spawned children after CUDA initialization can indefinitely stall
    # all workers while they import PyTorch.
    checkpoint_eval_pool_size = min(args.parallel_envs, args.checkpoint_eval_episodes)
    with BlueProcessEnvironmentPool(
        environment_config, env_config, pool_size,
        native_threads=args.env_worker_threads,
        timeout_s=args.env_worker_timeout_s,
    ) as pool, BlueProcessEnvironmentPool(
        environment_config, checkpoint_eval_env_config, checkpoint_eval_pool_size,
        native_threads=args.env_worker_threads,
        timeout_s=args.env_worker_timeout_s,
    ) as checkpoint_eval_pool:
        # Keep CUDA initialization in the parent until every Windows worker has
        # spawned.  A resumed RNG state is restored after constructing/loading
        # the agent because both operations consume Torch random numbers.
        torch.manual_seed(args.seed)
        if torch.device(args.device).type == "cuda":
            torch.cuda.manual_seed_all(args.seed)
        agent = RainbowDQNAgent(rainbow_config)
        if resume_payload is not None:
            try:
                resume_agent_state = resume_payload.pop("agent_state")
                agent.load_training_state_dict(resume_agent_state)
                del resume_agent_state
                assert resume_log_offsets is not None
                _truncate_committed_log(
                    jsonl_path, int(resume_log_offsets["training_jsonl_bytes"])
                )
                _truncate_committed_log(
                    flight_quality_jsonl_path,
                    int(resume_log_offsets["flight_quality_jsonl_bytes"]),
                )
                resume_artifacts = resume_payload.pop("artifacts")
                _restore_resume_artifacts(output, resume_artifacts)
                del resume_artifacts
                resume_rng_state = resume_payload.pop("rng_state")
                for derived_output in {
                    metrics_path,
                    output / "training_metrics.json",
                    output / "episodes.json",
                    output / "blue_rainbow.pt",
                }:
                    derived_output.unlink(missing_ok=True)
            except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
                raise SystemExit(f"cannot restore blue training state: {error}") from error

        state = restored_training_state or {}
        event_rows: list[dict[str, Any]] = list(state.get("event_rows", []))
        summaries: list[dict[str, Any]] = list(state.get("summaries", []))
        checkpoint_evaluations: list[dict[str, Any]] = list(
            state.get("checkpoint_evaluations", [])
        )
        retained_checkpoints: list[dict[str, Any]] = list(
            state.get("retained_checkpoints", [])
        )
        last_checkpoint_validation_state: tuple[object, ...] | None = None
        best_checkpoint_directory = output / "best_checkpoints"
        best_checkpoint_directory.mkdir(parents=True, exist_ok=True)
        leaderboard_path = best_checkpoint_directory / "leaderboard.json"
        if restored_training_state is None:
            for stale_checkpoint in best_checkpoint_directory.glob("blue_rainbow_ep*.pt"):
                stale_checkpoint.unlink()
            leaderboard_path.unlink(missing_ok=True)
        started = time.monotonic()
        elapsed_before_resume = float(state.get("elapsed_s", 0.0))
        experiment = {
            "event": "experiment_config", "device": str(agent.device), "red_counts": list(training_scenarios),
            "blue_count": 1, "scenario_sampling": ("staged_curriculum" if curriculum is not None
                                                     else "uniform_random_per_episode"),
            "parallel_cpu_envs": pool_size, "parallel_backend": "process_spawn",
            "env_worker_threads": args.env_worker_threads, "env_worker_timeout_s": args.env_worker_timeout_s,
            "inference_batch_size_max": pool_size, "training_batch_size": args.batch_size,
            "training_mode": "rainbow_off_policy", "episodes": args.episodes,
            "guidance_contract": GUIDANCE_CONTRACT,
            "active_reward_mechanisms": list(mechanism_reward_config.active_mechanisms()),
            "seed": args.seed, "decision_interval_s": args.decision_interval,
            "updates_per_transition": args.updates_per_transition, "completed_environment_transitions": 0,
            "completed_optimizer_updates": 0, "completed_target_updates": 0,
            "rainbow_config": asdict(rainbow_config), "blue_environment_config": asdict(env_config),
            "environment_config": asdict(environment_config),
            "checkpoint_validation": {
                "checkpoint_interval": args.checkpoint_interval,
                "episodes": args.checkpoint_eval_episodes,
                "seed_base": checkpoint_eval_seed,
                "fixed_plan": True,
                "scenario_sampling": "balanced_over_training_scenarios",
                "scenario_counts": dict(Counter(
                    int(trial["missile_count"]) for trial in checkpoint_validation_plan
                )),
                "parallel_cpu_envs": checkpoint_eval_pool_size,
                "per_episode_artifacts": False,
                "ranking": [
                    "survival_rate_desc",
                    "successful_evasion_count_desc",
                    "timeout_victory_count_asc",
                ],
                "retention_limit": 3,
                "retain_exact_cutoff_ties": True,
            },
            "metrics_path": str(metrics_path), "jsonl_path": str(jsonl_path),
            "flight_quality_jsonl_path": str(flight_quality_jsonl_path),
            "curriculum": curriculum.describe() if curriculum is not None else None,
        }
        scenario_rng = random.Random(args.seed)
        episode_scenarios = {episode: (curriculum.sample(episode, scenario_rng) if curriculum is not None
                                      else scenario_rng.choice(missile_scenarios))
                             for episode in range(1, args.episodes + 1)}
        episode_stages = {episode: curriculum.stage_at(episode)[1].name
                          for episode in range(1, args.episodes + 1)} if curriculum is not None else {}
        observations: dict[int, np.ndarray] = {}; episode_by_worker: dict[int, int] = {}
        episode_orientations: dict[int, str] = dict(state.get("episode_orientations", {}))
        learning_active_by_worker: dict[int, bool] = {}
        reward_by_worker: dict[int, float] = {}; decisions_by_worker: dict[int, int] = {}
        reward_components_by_worker: dict[int, Counter[str]] = {}
        reward_diagnostics_by_worker: dict[int, Counter[str]] = {}
        envelope_states: dict[int, dict[str, object]] = {}
        constraints = {worker: FlightEnvelopeConstraintLayer(envelope_config)
                       for worker in range(pool_size)}
        quality: dict[int, FlightQualityTracker] = {}
        quality_episodes: list[dict[str, Any]] = list(state.get("quality_episodes", []))
        next_episode = int(state.get("next_episode", 1))
        completed = int(state.get("completed_episodes", 0))
        update_credit = float(state.get("update_credit", 0.0))
        next_checkpoint = int(state.get("next_checkpoint", args.checkpoint_interval))
        next_log = int(state.get("next_log", args.log_interval))
        vector_iterations = int(state.get("vector_iterations", 0))
        next_curriculum_eval = int(
            state.get("next_curriculum_eval", args.curriculum_eval_interval)
        )
        next_resume_save = int(state.get("next_resume_save", args.resume_save_interval))
        while next_resume_save <= completed:
            next_resume_save += args.resume_save_interval
        resume_generation = int(state.get("resume_generation", 0))
        last_resume_snapshot_path = resume_path
        historical_best: dict[int, float] = dict(state.get("historical_best", {}))
        best_balanced_score = float(state.get("best_balanced_score", -math.inf))
        best_new_rate: dict[int, float] = dict(state.get("best_new_rate", {}))
        window_episodes: list[dict[str, Any]] = list(state.get("window_episodes", []))
        window_losses: list[float] = list(state.get("window_losses", []))
        window_grad_norms: list[float] = list(state.get("window_grad_norms", []))
        window_values: list[float] = list(state.get("window_values", []))
        window_actions: Counter[int] = Counter(state.get("window_actions", {}))
        window_policy_actions: Counter[int] = Counter(state.get("window_policy_actions", {}))
        window_constraint_interventions = int(state.get("window_constraint_interventions", 0))
        window_clamp_low: list[float] = list(state.get("window_clamp_low", []))
        window_clamp_high: list[float] = list(state.get("window_clamp_high", []))
        window_elapsed_before_resume = float(state.get("window_elapsed_s", 0.0))
        window_started = started
        window_transition_start = int(state.get("window_transition_start", agent.total_steps))
        window_update_start = int(state.get("window_update_start", agent.optimizer_updates))
        draining_for_resume = False
        drain_requested_thresholds: list[int] = []

        def leaderboard_entry(candidate: dict[str, Any]) -> dict[str, Any]:
            return {
                "completed_episodes": int(candidate["completed_episodes"]),
                "episode_thresholds": list(candidate["episode_thresholds"]),
                "final_state": bool(candidate["final_state"]),
                "checkpoint_path": str(candidate["checkpoint_path"]),
                "validation_episodes": int(candidate["validation_episodes"]),
                "survived_count": int(candidate["survived_count"]),
                "survival_rate": float(candidate["survival_rate"]),
                "successful_evasion_count": int(candidate["successful_evasion_count"]),
                "timeout_victory_count": int(candidate["timeout_victory_count"]),
                "failure_count": int(candidate["failure_count"]),
                "termination_counts": dict(candidate["termination_counts"]),
                "scenario_counts": dict(candidate["scenario_counts"]),
                "scenario_survival_rates": dict(candidate["scenario_survival_rates"]),
                "environment_transitions": int(candidate["environment_transitions"]),
                "optimizer_updates": int(candidate["optimizer_updates"]),
                "target_updates": int(candidate["target_updates"]),
                "ranking_score": list(_checkpoint_validation_score(candidate)),
            }

        def write_checkpoint_leaderboard() -> None:
            ordered_entries = [leaderboard_entry(item) for item in retained_checkpoints]
            best_score = (_checkpoint_validation_score(retained_checkpoints[0])
                          if retained_checkpoints else None)
            best_entries = [
                entry for item, entry in zip(retained_checkpoints, ordered_entries)
                if best_score is not None and _checkpoint_validation_score(item) == best_score
            ]
            _atomic_write_json(leaderboard_path, {
                "ranking": [
                    "survived_count_desc",
                    "successful_evasion_count_desc",
                    "timeout_victory_count_asc",
                ],
                "retention_limit": 3,
                "retain_exact_cutoff_ties": True,
                "evaluation_count": len(checkpoint_evaluations),
                "best_completed_episodes": [
                    int(entry["completed_episodes"]) for entry in best_entries
                ],
                "best_checkpoints": best_entries,
                "retained_checkpoints": ordered_entries,
            })

        def validate_checkpoint(
            episode_thresholds: list[int], *, final_state: bool = False,
        ) -> None:
            nonlocal retained_checkpoints, last_checkpoint_validation_state
            _emit({
                "event": "checkpoint_validation_start",
                "episode_thresholds": list(episode_thresholds),
                "completed_episodes": completed,
                "final_state": final_state,
                "validation_episodes": args.checkpoint_eval_episodes,
                "environment_transitions": agent.total_steps,
                "optimizer_updates": agent.optimizer_updates,
            }, event_rows, jsonl_path)
            validation = _evaluate_checkpoint_without_updates(
                agent, checkpoint_eval_pool, checkpoint_validation_plan, envelope_config,
            )
            checkpoint_path = (
                best_checkpoint_directory / f"blue_rainbow_ep{completed:06d}.pt"
            ).resolve()
            candidate: dict[str, Any] = {
                "completed_episodes": completed,
                "episode_thresholds": list(episode_thresholds),
                "final_state": final_state,
                "checkpoint_path": str(checkpoint_path),
                "environment_transitions": agent.total_steps,
                "optimizer_updates": agent.optimizer_updates,
                "target_updates": agent.target_updates,
                **validation,
            }
            selected = _select_retained_checkpoints(
                [*retained_checkpoints, candidate], limit=3,
            )
            candidate_retained = any(item is candidate for item in selected)
            removed = [
                item for item in retained_checkpoints
                if not any(item is kept for kept in selected)
            ]
            if candidate_retained:
                agent.save(str(checkpoint_path))
            for removed_candidate in removed:
                Path(str(removed_candidate["checkpoint_path"])).unlink(missing_ok=True)
            retained_checkpoints = selected
            last_checkpoint_validation_state = _checkpoint_validation_learner_state(agent)

            evaluation_record = {
                "event": "checkpoint_validation",
                **candidate,
                "candidate_retained_after_evaluation": candidate_retained,
                "checkpoint_path": str(checkpoint_path) if candidate_retained else None,
                "removed_checkpoint_paths": [
                    str(item["checkpoint_path"]) for item in removed
                ],
                "ranking_score": list(_checkpoint_validation_score(candidate)),
                "retained_checkpoints": [
                    leaderboard_entry(item) for item in retained_checkpoints
                ],
            }
            checkpoint_evaluations.append(evaluation_record)
            write_checkpoint_leaderboard()
            _emit(evaluation_record, event_rows, jsonl_path)

        def retire_worker(worker: int) -> None:
            observations.pop(worker, None)
            episode_by_worker.pop(worker, None)
            learning_active_by_worker.pop(worker, None)
            reward_by_worker.pop(worker, None)
            decisions_by_worker.pop(worker, None)
            reward_components_by_worker.pop(worker, None)
            reward_diagnostics_by_worker.pop(worker, None)
            quality.pop(worker, None)
            envelope_states.pop(worker, None)

        def prepare_assignments(worker_ids: list[int]) -> dict[int, tuple[int, int, int]]:
            nonlocal next_episode
            assignments: dict[int, tuple[int, int, int]] = {}
            for worker in worker_ids:
                if next_episode > args.episodes:
                    break
                assignments[worker] = (
                    args.seed + next_episode,
                    next_episode,
                    episode_scenarios[next_episode],
                )
                episode_by_worker[worker] = next_episode
                reward_by_worker[worker] = 0.0
                reward_components_by_worker[worker] = Counter()
                reward_diagnostics_by_worker[worker] = Counter()
                decisions_by_worker[worker] = 0
                quality[worker] = FlightQualityTracker()
                constraints[worker].reset()
                next_episode += 1
            return assignments

        def reset_assignments(assignments: dict[int, tuple[int, int, int]]) -> None:
            if not assignments:
                return
            for worker, (observation, info) in pool.reset(assignments).items():
                record_episode_orientation(
                    episode_orientations, episode_by_worker[worker], info
                )
                observations[worker] = observation
                learning_active_by_worker[worker] = bool(info["learning_active"])
                quality[worker].add(info["flight_quality_state"])
                envelope_states[worker] = dict(info["flight_quality_state"])
                constraints[worker].reset()

        def save_resume_snapshot(
            *, reason: str, requested_thresholds: list[int], actual_completed: int,
        ) -> tuple[Path, Path]:
            nonlocal resume_generation, last_resume_snapshot_path
            if observations:
                raise RuntimeError("full resume state can only be saved after draining all workers")
            if next_episode != completed + 1:
                raise RuntimeError(
                    "safe resume boundary requires next_episode == completed_episodes + 1"
                )
            agent.assert_resume_safe()

            artifact_paths = [
                Path(str(candidate["checkpoint_path"]))
                for candidate in retained_checkpoints
            ]
            artifact_paths.extend(
                path for path in (
                    output / "best_new_scenario.pt",
                    output / "best_balanced.pt",
                ) if path.is_file()
            )
            artifacts = _collect_resume_artifacts(output, artifact_paths)
            now = time.monotonic()
            resume_generation += 1
            snapshot_id = (
                f"g{resume_generation:06d}_ep{completed:06d}_"
                f"t{agent.total_steps:012d}_{uuid.uuid4().hex[:12]}"
            )
            training_state: dict[str, Any] = {
                "completed_episodes": completed,
                "next_episode": next_episode,
                "next_checkpoint": next_checkpoint,
                "next_log": next_log,
                "next_curriculum_eval": next_curriculum_eval,
                "next_resume_save": next_resume_save,
                "resume_generation": resume_generation,
                "requested_resume_thresholds": list(requested_thresholds),
                "actual_completed_episodes": int(actual_completed),
                "vector_iterations": vector_iterations,
                "update_credit": update_credit,
                "elapsed_s": elapsed_before_resume + now - started,
                "window_elapsed_s": window_elapsed_before_resume + now - window_started,
                "window_transition_start": window_transition_start,
                "window_update_start": window_update_start,
                "window_episodes": window_episodes,
                "window_losses": window_losses,
                "window_grad_norms": window_grad_norms,
                "window_values": window_values,
                "window_actions": dict(window_actions),
                "window_policy_actions": dict(window_policy_actions),
                "window_constraint_interventions": window_constraint_interventions,
                "window_clamp_low": window_clamp_low,
                "window_clamp_high": window_clamp_high,
                "historical_best": historical_best,
                "best_balanced_score": best_balanced_score,
                "best_new_rate": best_new_rate,
                "summaries": summaries,
                "quality_episodes": quality_episodes,
                "episode_orientations": episode_orientations,
                "event_rows": event_rows,
                "checkpoint_evaluations": checkpoint_evaluations,
                "retained_checkpoints": retained_checkpoints,
                "current_state_already_validated": (
                    last_checkpoint_validation_state
                    == _checkpoint_validation_learner_state(agent)
                ),
                "log_paths": {
                    "training_jsonl": str(jsonl_path.resolve()),
                    "flight_quality_jsonl": str(flight_quality_jsonl_path.resolve()),
                },
                "log_offsets": {
                    "training_jsonl_bytes": jsonl_path.stat().st_size,
                    "flight_quality_jsonl_bytes": flight_quality_jsonl_path.stat().st_size,
                },
            }
            payload: dict[str, Any] = {
                "schema_version": BLUE_RESUME_SCHEMA_VERSION,
                "kind": BLUE_RESUME_KIND,
                "snapshot_id": snapshot_id,
                "reason": reason,
                "output_directory": str(output.resolve()),
                "critical_config": critical_config,
                "runtime_config": {
                    "device": str(agent.device),
                    "parallel_envs": int(args.parallel_envs),
                    "env_worker_threads": int(args.env_worker_threads),
                },
                "agent_state": agent.training_state_dict(),
                "rng_state": _capture_training_rng_state(agent.device),
                "training_state": training_state,
                "artifacts": artifacts,
            }
            result = _save_resume_generation(
                output / "resume",
                payload,
                snapshot_id=snapshot_id,
                completed_episodes=completed,
                reason=reason,
                preferred_previous_path=last_resume_snapshot_path,
            )
            last_resume_snapshot_path = result[0]
            return result

        if restored_training_state is not None:
            if bool(restored_training_state.get("current_state_already_validated", False)):
                last_checkpoint_validation_state = _checkpoint_validation_learner_state(agent)
            write_checkpoint_leaderboard()
            assert (resume_path is not None and resume_rng_state is not None
                    and resume_runtime_config is not None)
            _restore_training_rng_state(resume_rng_state, agent.device)
            resume_rng_state = None
            if resume_payload is not None:
                resume_payload.clear()
                resume_payload = None
            _emit({
                "event": "training_resumed",
                "snapshot_path": str(resume_path),
                "snapshot_id": resume_snapshot_id,
                "restored_completed_episodes": completed,
                "target_episodes": args.episodes,
                "device": str(agent.device),
                "parallel_cpu_envs": pool_size,
                "trajectory_equivalent_runtime": (
                    str(resume_runtime_config["device"]) == str(agent.device)
                    and int(resume_runtime_config["parallel_envs"]) == args.parallel_envs
                    and int(resume_runtime_config["env_worker_threads"])
                    == args.env_worker_threads
                ),
                "saved_runtime_config": resume_runtime_config,
            }, event_rows, jsonl_path)
        else:
            device_row: dict[str, Any] = {"event": "device", "device": str(agent.device)}
            if agent.device.type == "cuda":
                device_row.update({
                    "cuda_device_name": torch.cuda.get_device_name(agent.device),
                    "cuda_device_count": torch.cuda.device_count(),
                })
            _emit(device_row, event_rows, jsonl_path)
            _emit(experiment, event_rows, jsonl_path)

        _emit({"event": "environment_workers", "backend": "process_spawn",
               "worker_count": pool_size, "workers": list(pool.worker_info)}, event_rows, jsonl_path)
        if next_episode != completed + 1:
            raise RuntimeError("resume state has inconsistent next_episode/completed_episodes")
        initial_workers = list(range(min(pool_size, args.episodes - completed)))
        reset_assignments(prepare_assignments(initial_workers))
        while observations:
            vector_iterations += 1; workers = sorted(observations)
            active_workers = [worker for worker in workers if learning_active_by_worker[worker]]
            actions = {worker: 0 for worker in workers}
            policy_actions = {worker: 0 for worker in workers}
            action_diagnostics: dict[int, dict[str, object]] = {}
            if active_workers:
                q_values = agent.expected_action_values(
                    np.stack([observations[worker] for worker in active_workers]), evaluation=False
                )
                selected = [constraints[worker].select(q, envelope_states[worker])
                            for worker, q in zip(active_workers, q_values)]
                for worker, q, (action, diagnostic) in zip(active_workers, q_values, selected):
                    actions[worker] = int(action)
                    policy_actions[worker] = int(diagnostic["raw_action"])
                    action_diagnostics[worker] = diagnostic
                    window_values.append(float(q[action]))
            action_masks = {
                worker: np.asarray(action_diagnostics[worker]["effective_action_mask"], dtype=bool)
                if worker in action_diagnostics else np.ones(29, dtype=bool)
                for worker in workers
            }
            fallback_flags = {
                worker: bool(action_diagnostics.get(worker, {}).get("fallback_required", False))
                for worker in workers
            }
            results = pool.step(
                actions, policy_actions=policy_actions, action_masks=action_masks,
                fallback_required=fallback_flags,
            )
            resets: dict[int, tuple[int, int, int]] = {}
            finished_workers: list[int] = []
            learning_transition_count = 0
            for worker in workers:
                result = results[worker]; previous = observations[worker]
                done = result.terminated or result.truncated
                learning_transition = bool(result.info["learning_transition"])
                executed_action = int(result.info["executed_action_index"])
                if learning_transition:
                    if executed_action != actions[worker]:
                        raise RuntimeError("replay action must equal the post-constraint executed action")
                    next_mask, next_penalty, _ = constraints[worker].constraints(
                        result.info["flight_quality_state"]
                    )
                    agent.observe_for_env(
                        worker, previous, executed_action, result.reward, result.observation, done,
                        next_action_mask=next_mask, next_action_penalty=next_penalty,
                    )
                    learning_transition_count += 1
                    decisions_by_worker[worker] += 1
                    window_actions[executed_action] += 1
                    window_policy_actions[policy_actions[worker]] += 1
                    window_constraint_interventions += int(
                        bool(action_diagnostics[worker]["intervened"])
                    )
                    for name, value in dict(result.info.get("reward_components", {})).items():
                        if name != "threat_potential":
                            reward_components_by_worker[worker][str(name)] += float(value)
                    for name, value in dict(result.info.get("reward_diagnostics", {})).items():
                        reward_diagnostics_by_worker[worker][str(name)] += float(value)
                reward_by_worker[worker] += result.reward
                observations[worker] = result.observation
                learning_active_by_worker[worker] = bool(result.info["learning_active"])
                envelope_states[worker] = dict(result.info["flight_quality_state"])
                diagnostic = action_diagnostics.get(worker, {}) if learning_transition else {}
                quality[worker].add(result.info["flight_quality_state"],
                                    policy_action=policy_actions[worker] if learning_transition else None,
                                    executed_action=executed_action,
                                    safety_intervened=bool(diagnostic.get("intervened", False)),
                                    safety_reasons=list(diagnostic.get("raw_action_hard_violation_reasons", [])))
                if done:
                    completed += 1
                    row = {
                        "episode": episode_by_worker[worker], "reward": reward_by_worker[worker],
                        "missile_count": episode_scenarios[episode_by_worker[worker]],
                        "curriculum_stage": episode_stages.get(episode_by_worker[worker]),
                        "blue_survived": bool(result.info["blue_survived"]),
                        "termination_reason": result.info.get("termination_reason"),
                        "hit_count": int(result.info.get("hit_count", 0)),
                        "miss_distance_m": float(result.info.get("miss_distance_m", 0.0)),
                        "simulation_time_s": float(result.info.get("time_s", 0.0)),
                        "physics_steps": int(result.info.get("step_count", 0)),
                        "decision_steps": decisions_by_worker[worker],
                        "learning_activation_time_s": result.info.get("learning_activation_time_s"),
                        "learning_activation_range_m": result.info.get("learning_activation_range_m"),
                        "threat_observed": bool(result.info.get("learning_active", False)),
                        "reward_components": dict(reward_components_by_worker[worker]),
                        "reward_diagnostics_mean": {
                            name: value / decisions_by_worker[worker]
                            for name, value in reward_diagnostics_by_worker[worker].items()
                        },
                        "red_loss_reasons": list(result.info.get("red_loss_reasons", [])),
                        "mean_loss": (
                            agent.last_update_metrics.get("loss")
                            if decisions_by_worker[worker] > 0 else None
                        ),
                        "learner_loss_at_completion": (
                            agent.last_update_metrics.get("loss")
                            if decisions_by_worker[worker] > 0 else None
                        ),
                        "acmi_path": result.info.get("acmi_path"),
                    }
                    summaries.append(row); window_episodes.append(row)
                    quality_result = quality[worker].finish(episode=episode_by_worker[worker],
                                                            survived=bool(result.info["blue_survived"]))
                    row["flight_quality"] = {key: quality_result[key] for key in ("metrics", "verdicts", "events")}
                    quality_episodes.append(quality_result)
                    append_flight_quality_episode(quality_result, flight_quality_jsonl_path)
                    finished_workers.append(worker)

            if (not draining_for_resume and completed >= next_resume_save
                    and completed < args.episodes and next_episode <= args.episodes):
                draining_for_resume = True
                drain_requested_thresholds = []
                threshold = next_resume_save
                while threshold <= completed:
                    drain_requested_thresholds.append(threshold)
                    threshold += args.resume_save_interval
            for worker in finished_workers:
                if draining_for_resume or next_episode > args.episodes:
                    retire_worker(worker)
                else:
                    resets.update(prepare_assignments([worker]))
            # Replay insertion and loss/gradient computation are both driven
            # exclusively by post-detection transitions.
            update_credit += learning_transition_count * args.updates_per_transition
            updates = math.floor(update_credit); update_credit -= updates
            for _ in range(updates):
                loss = agent.update()
                if loss is not None:
                    window_losses.append(loss)
                    window_grad_norms.append(agent.last_update_metrics["gradient_norm"])
                    window_clamp_low.append(agent.last_update_metrics["c51_clamp_low_fraction"])
                    window_clamp_high.append(agent.last_update_metrics["c51_clamp_high_fraction"])
            crossed_checkpoint_thresholds: list[int] = []
            while completed >= next_checkpoint:
                crossed_checkpoint_thresholds.append(next_checkpoint)
                next_checkpoint += args.checkpoint_interval
            if crossed_checkpoint_thresholds:
                validate_checkpoint(crossed_checkpoint_thresholds)
            while (curriculum is not None and args.curriculum_eval_episodes > 0
                   and completed >= next_curriculum_eval):
                _, evaluation_stage, _ = curriculum.stage_at(min(next_curriculum_eval, args.episodes))
                introduced = [index + 1 for index, probability in enumerate(evaluation_stage.probabilities)
                              if probability > 0.0]
                evaluation = _curriculum_evaluation(
                    agent, environment_config, env_config, introduced,
                    args.curriculum_eval_episodes, args.seed + 10_000_000,
                )
                rates = {int(key): float(value) for key, value in evaluation["survival_rates"].items()}
                eligible = within_forgetting_limit(rates, historical_best)
                score = balanced_score(rates, evaluation_stage.score_weights)
                newest = max(introduced)
                if rates[newest] > best_new_rate.get(newest, -1.0):
                    best_new_rate[newest] = rates[newest]
                    agent.save(str(output / "best_new_scenario.pt"))
                if eligible and score > best_balanced_score:
                    best_balanced_score = score
                    agent.save(str(output / "best_balanced.pt"))
                for scenario, rate in rates.items():
                    historical_best[scenario] = max(historical_best.get(scenario, 0.0), rate)
                _emit({"event": "curriculum_evaluation", "completed_episodes": completed,
                       "episode_threshold": next_curriculum_eval, "stage": evaluation_stage.name,
                       **evaluation, "score": score, "score_weights": evaluation_stage.score_weights,
                       "within_five_point_forgetting_limit": eligible,
                       "historical_best_survival_rates": dict(historical_best)}, event_rows, jsonl_path)
                next_curriculum_eval += args.curriculum_eval_interval
            if completed >= next_log or (completed == args.episodes and window_episodes):
                now = time.monotonic(); rewards = [float(row["reward"]) for row in window_episodes]
                misses = [float(row["miss_distance_m"]) for row in window_episodes]
                simulation_times = [float(row["simulation_time_s"]) for row in window_episodes]
                action_total = sum(window_actions.values())
                probabilities = np.asarray(list(window_actions.values()), dtype=np.float64) / max(action_total, 1)
                report = {
                    "event": "iteration", "iteration": len([row for row in event_rows if row["event"] == "iteration"]) + 1,
                    "completed_episodes": completed,
                    "episode_ids": sorted(int(row["episode"]) for row in window_episodes),
                    "sampled_red_counts": dict(Counter(int(row["missile_count"]) for row in window_episodes)),
                    "sampled_blue_count": 1,
                    "active_parallel_envs": len(observations), "inference_batch_size": len(active_workers),
                    "vector_iterations": vector_iterations,
                    "rollout_decision_steps": agent.total_steps - window_transition_start,
                    "completed_environment_transitions": agent.total_steps,
                    "reward_mean": _mean(rewards), "reward_std": float(np.std(rewards)) if rewards else None,
                    "reward_min": min(rewards) if rewards else None, "reward_max": max(rewards) if rewards else None,
                    "survival_rate": _mean([float(row["blue_survived"]) for row in window_episodes]),
                    "hit_count_sum": sum(int(row["hit_count"]) for row in window_episodes),
                    "miss_distance_mean_m": _mean(misses),
                    "miss_distance_p95_m": float(np.percentile(misses, 95)) if misses else None,
                    "simulation_time_mean_s": _mean(simulation_times),
                    "termination_counts": dict(Counter(str(row["termination_reason"]) for row in window_episodes)),
                    "rollout_diagnostics": {
                        "wall_time_s": window_elapsed_before_resume + now - window_started,
                        "transitions_per_s": (agent.total_steps - window_transition_start) / max(window_elapsed_before_resume + now - window_started, 1e-9),
                        "episodes_per_hour": len(window_episodes) * 3600.0 / max(window_elapsed_before_resume + now - window_started, 1e-9),
                    },
                    "policy_diagnostics": {
                        "policy_action_histogram": {str(key): value for key, value in sorted(window_policy_actions.items())},
                        "action_histogram": {str(key): value for key, value in sorted(window_actions.items())},
                        "action_entropy": float(-(probabilities * np.log(probabilities + 1e-12)).sum()),
                        "selected_value_mean": _mean(window_values),
                        "constraint_intervention_count": window_constraint_interventions,
                        "constraint_intervention_rate": window_constraint_interventions / max(action_total, 1),
                    },
                    "loss_mean": _mean(window_losses),
                    "loss_std": float(np.std(window_losses)) if window_losses else None,
                    "gradient_norm_mean": _mean(window_grad_norms),
                    "learning_rate": float(agent.optimizer.param_groups[0]["lr"]),
                    "replay_size": agent.replay.size, "replay_capacity": agent.config.replay_size,
                    "per_beta": agent.last_update_metrics.get("per_beta"),
                    "priority_mean": agent.last_update_metrics.get("priority_mean"),
                    "priority_max": agent.last_update_metrics.get("priority_max"),
                    "c51_clamp_low_fraction_mean": _mean(window_clamp_low),
                    "c51_clamp_high_fraction_mean": _mean(window_clamp_high),
                    "completed_optimizer_updates": agent.optimizer_updates,
                    "optimizer_updates_in_window": agent.optimizer_updates - window_update_start,
                    "completed_target_updates": agent.target_updates,
                    **_device_metrics(agent.device),
                }
                _emit(report, event_rows, jsonl_path)
                while next_log <= completed: next_log += args.log_interval
                window_episodes = []; window_losses = []; window_grad_norms = []; window_values = []
                window_clamp_low = []; window_clamp_high = []
                window_actions = Counter(); window_policy_actions = Counter()
                window_constraint_interventions = 0; window_started = now
                window_elapsed_before_resume = 0.0
                window_transition_start = agent.total_steps; window_update_start = agent.optimizer_updates
            if resets:
                reset_assignments(resets)
            if draining_for_resume and not observations:
                crossed_resume_thresholds: list[int] = []
                while next_resume_save <= completed:
                    crossed_resume_thresholds.append(next_resume_save)
                    next_resume_save += args.resume_save_interval
                if not crossed_resume_thresholds:
                    crossed_resume_thresholds = list(drain_requested_thresholds)
                snapshot_path, manifest_path = save_resume_snapshot(
                    reason="periodic",
                    requested_thresholds=crossed_resume_thresholds,
                    actual_completed=completed,
                )
                _emit({
                    "event": "resume_snapshot",
                    "reason": "periodic",
                    "requested_episode_thresholds": crossed_resume_thresholds,
                    "completed_episodes": completed,
                    "snapshot_path": str(snapshot_path),
                    "manifest_path": str(manifest_path),
                }, event_rows, jsonl_path)
                draining_for_resume = False
                drain_requested_thresholds = []
                remaining = args.episodes - completed
                if remaining > 0:
                    reset_assignments(
                        prepare_assignments(list(range(min(pool_size, remaining))))
                    )
        summaries.sort(key=lambda row: int(row["episode"]))
        current_validation_state = _checkpoint_validation_learner_state(agent)
        if current_validation_state != last_checkpoint_validation_state:
            validate_checkpoint([], final_state=True)
        final_resume_thresholds: list[int] = []
        while next_resume_save <= completed:
            final_resume_thresholds.append(next_resume_save)
            next_resume_save += args.resume_save_interval
        final_resume_path, resume_manifest_path = save_resume_snapshot(
            reason="final",
            requested_thresholds=final_resume_thresholds,
            actual_completed=completed,
        )
        _emit({
            "event": "resume_snapshot",
            "reason": "final",
            "requested_episode_thresholds": final_resume_thresholds,
            "completed_episodes": completed,
            "snapshot_path": str(final_resume_path),
            "manifest_path": str(resume_manifest_path),
        }, event_rows, jsonl_path)
        final_checkpoint = output / "blue_rainbow.pt"; agent.save(str(final_checkpoint))
        elapsed = elapsed_before_resume + time.monotonic() - started
        flight_quality_summary = write_flight_quality_report(
            sorted(quality_episodes, key=lambda item: int(item["episode"])), output / "flight_quality",
            baseline_survival_rate=args.baseline_survival_rate, plot_limit=args.flight_quality_plot_limit)
        best_score = (_checkpoint_validation_score(retained_checkpoints[0])
                      if retained_checkpoints else None)
        best_checkpoint_entries = [
            leaderboard_entry(candidate) for candidate in retained_checkpoints
            if best_score is not None and _checkpoint_validation_score(candidate) == best_score
        ]
        final_summary = {
            "event": "training_complete", "episodes": args.episodes,
            "guidance_contract": GUIDANCE_CONTRACT,
            "survival_rate": _mean([float(row["blue_survived"]) for row in summaries]),
            "reward_mean": _mean([float(row["reward"]) for row in summaries]),
            "elapsed_s": elapsed, "episodes_per_hour": args.episodes * 3600.0 / max(elapsed, 1e-9),
            "completed_environment_transitions": agent.total_steps,
            "completed_optimizer_updates": agent.optimizer_updates,
            "completed_target_updates": agent.target_updates, "replay_size": agent.replay.size,
            "final_checkpoint": str(final_checkpoint),
            "resume_snapshot": str(final_resume_path),
            "resume_manifest": str(resume_manifest_path),
            **_device_metrics(agent.device),
            "flight_quality": flight_quality_summary,
            "checkpoint_validation": {
                "evaluation_count": len(checkpoint_evaluations),
                "episodes_per_evaluation": args.checkpoint_eval_episodes,
                "best_completed_episode": (
                    int(best_checkpoint_entries[0]["completed_episodes"])
                    if len(best_checkpoint_entries) == 1 else None
                ),
                "best_completed_episodes": [
                    int(entry["completed_episodes"]) for entry in best_checkpoint_entries
                ],
                "best_episode_thresholds": [
                    list(entry["episode_thresholds"]) for entry in best_checkpoint_entries
                ],
                "best_checkpoint_paths": [
                    str(entry["checkpoint_path"]) for entry in best_checkpoint_entries
                ],
                "best_checkpoints": best_checkpoint_entries,
                "retained_checkpoints": [
                    leaderboard_entry(candidate) for candidate in retained_checkpoints
                ],
                "leaderboard_path": str(leaderboard_path),
            },
        }
        final_summary["scenario_episode_counts"] = dict(Counter(
            int(row["missile_count"]) for row in summaries
        ))
        _emit(final_summary, event_rows, jsonl_path)
        (output / "episodes.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        metrics_path.write_text(json.dumps({"experiment_config": experiment, "iterations": [row for row in event_rows if row["event"] == "iteration"],
                                            "checkpoint_evaluations": checkpoint_evaluations,
                                            "curriculum_evaluations": [row for row in event_rows if row["event"] == "curriculum_evaluation"],
                                            "episodes": summaries, "final_summary": final_summary}, indent=2), encoding="utf-8")
    training_analysis_dir, _ = _training_plot_directories(output, args.training_plots_dir)
    if not args.no_training_plots:
        _write_post_training_plots(
            training_dir=output,
            metrics_path=metrics_path,
            training_plots_dir=args.training_plots_dir,
            smoothing_window=args.plot_smoothing_window,
        )
    write_blue_training_results_safely(
        metrics_path,
        training_analysis_dir / "results",
        episode_orientations=episode_orientations,
        make_plots=not args.no_training_plots,
    )
    return 0


if __name__ == "__main__": raise SystemExit(main())
