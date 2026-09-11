"""Passive, JSON-safe state recording and provenance for evaluation artifacts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def finite_vector(value: Any) -> tuple[list[float | None], bool]:
    """Keep missing/non-finite components distinct from real zero coordinates."""
    if value is None or len(value) != 3:
        return [None, None, None], False
    vector = [float(component) if component is not None and math.isfinite(float(component))
              else None for component in value]
    return vector, all(component is not None for component in vector)


class RedTelemetryTracker:
    """Keep episode-local slots fixed and share the blue trace's sample times."""

    def __init__(self, ids: list[int]) -> None:
        self.ids = list(ids)
        if len(set(self.ids)) != len(self.ids):
            raise ValueError("red telemetry IDs must be unique")
        self.samples: list[dict[str, Any]] = []
        self.termination_events: list[dict[str, Any]] = []
        self._terminated_ids: set[int] = set()

    def add(self, state: dict[str, Any]) -> dict[str, Any]:
        if list(state["red_ids"]) != self.ids:
            raise ValueError("red telemetry slots must remain fixed throughout an episode")
        time_s = float(state["time_s"])
        if not math.isfinite(time_s) or (self.samples and time_s <= self.samples[-1]["time_s"]):
            raise ValueError("red telemetry sample times must be finite and strictly increasing")
        sample: dict[str, Any] = {
            "time_s": time_s,
            "step_count": int(state["step_count"]),
            "sample_interval_s": time_s - self.samples[-1]["time_s"] if self.samples else 0.0,
            **{key: [] for key in ("red_positions_m", "red_velocities_mps", "red_alive",
                                   "red_loss_reasons", "red_position_valid", "red_velocity_valid",
                                   "red_state_valid")},
        }
        for slot in range(len(self.ids)):
            def item(key: str) -> Any:
                values = state.get(key, [])
                return values[slot] if slot < len(values) else None

            position, position_valid = finite_vector(item("red_positions_m"))
            velocity, velocity_valid = finite_vector(item("red_velocities_mps"))
            alive = item("red_alive")
            alive = None if alive is None else bool(alive)
            sample["red_positions_m"].append(position)
            sample["red_velocities_mps"].append(velocity)
            sample["red_alive"].append(alive)
            sample["red_loss_reasons"].append(item("red_loss_reasons") or None)
            sample["red_position_valid"].append(position_valid)
            sample["red_velocity_valid"].append(velocity_valid)
            sample["red_state_valid"].append(alive is True and position_valid and velocity_valid)
        for raw_event in state.get("red_termination_events", []):
            event = copy.deepcopy(raw_event)
            red_id = int(event["red_id"])
            if red_id not in self.ids:
                raise ValueError("red termination event refers to an unknown ID")
            if red_id in self._terminated_ids:
                continue
            for key in ("position_m", "velocity_mps", "blue_position_m", "blue_velocity_mps"):
                event[key], event[key.removesuffix("_mps").removesuffix("_m") + "_valid"] = finite_vector(event.get(key))
            self.termination_events.append(event)
            self._terminated_ids.add(red_id)
        self.samples.append(sample)
        return sample

    def trace(self) -> dict[str, Any]:
        return {"red_ids": self.ids.copy(),
                **{key: [sample[key] for sample in self.samples]
                   for key in self.samples[0] if key != "time_s"}}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_evaluation_metadata(output: Path, *, checkpoint: Path,
                              environment_config: dict[str, Any],
                              adapter_config: dict[str, Any],
                              evaluation_options: dict[str, Any]) -> dict[str, Any]:
    """Write configs once; episodes reference this manifest by run ID and filename."""
    source_root = Path(__file__).resolve().parents[1]
    source_files = {path.relative_to(source_root).as_posix(): _sha256_file(path)
                    for path in sorted(source_root.rglob("*.py"))}
    source_digest = hashlib.sha256(json.dumps(source_files, sort_keys=True).encode("utf-8")).hexdigest()
    metadata = {
        "schema_version": 2,
        "run_id": uuid.uuid4().hex,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": _sha256_file(checkpoint)},
        "code": {"source_sha256": source_digest, "files_sha256": source_files},
        "environment_config": environment_config,
        "adapter_config": adapter_config,
        "evaluation_options": evaluation_options,
        "coordinates": {"frame": "inertial", "axes": ["north", "up", "east"],
                        "position_unit": "m", "velocity_unit": "m/s", "time_unit": "s"},
        "sampling": {"trace": "reset and each completed environment decision step",
                     "first_sample_interval_s": 0.0,
                     "termination_events": "physics-step detection, with final red and blue states",
                     "id_scope": "episode", "id_base": 0,
                     "red_state_valid": "alive and finite position and velocity",
                     "inactive_slots": "retained; frozen values are not active-state samples"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    return metadata
