"""Stable Python entry point used by the CPython C API bridge.

The exported C ABI deliberately knows nothing about NumPy arrays or Python
objects.  This module owns the frozen Rainbow policy and all stateful
``BluePolicySession`` instances, translating primitive bridge arguments into
the existing deployment runtime's dictionaries.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any


OBSERVATION_DIM = 104
ACTION_COUNT = 29
_GUIDANCE_MODES = ("boost", "locked", "lock_hold", "inertial")
_RUNTIME_MODES = {"inactive": 0, "cruise": 1, "rainbow": 2}

_policy: Any | None = None
_session_type: Any | None = None
_json_api: Any | None = None
_sessions: dict[int, Any] = {}
_next_session = 1
_bundle_dir: Path | None = None
_native_dll_directory_handles: list[Any] = []
_native_dll_search_paths: list[str] = []


def _configure_native_dll_search() -> None:
    """Make native dependencies discoverable in embedded Conda on Windows."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidate = Path(sys.prefix) / "Library" / "bin"
    if candidate.is_dir() and not _native_dll_directory_handles:
        # Keep the handle alive while NumPy/MKL extension modules are loaded.
        _native_dll_directory_handles.append(os.add_dll_directory(str(candidate)))
        _native_dll_search_paths.append(str(candidate.resolve()))


def initialize(
    bundle_dir: str, interpreter_ownership: str = "borrowed"
) -> dict[str, object]:
    """Load and validate the model once for the lifetime of the bridge."""
    global _policy, _session_type, _json_api, _bundle_dir

    if interpreter_ownership not in {"owned", "borrowed"}:
        raise ValueError("interpreter_ownership must be owned or borrowed")

    root = Path(bundle_dir).expanduser().resolve(strict=True)
    runtime_dir = root / "blue_policy_runtime"
    model_dir = root / "model"
    if not runtime_dir.is_dir():
        raise FileNotFoundError(f"missing runtime package: {runtime_dir}")
    if not model_dir.is_dir():
        raise FileNotFoundError(f"missing model directory: {model_dir}")

    if _policy is not None:
        if root != _bundle_dir:
            raise RuntimeError(
                f"policy is already initialized from {_bundle_dir}; "
                f"cannot switch to {root}"
            )
        return _policy_info()

    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    _configure_native_dll_search()
    from blue_policy_json import BluePolicyJsonApi
    from blue_policy_runtime import BluePolicySession, FrozenRainbowNumpy

    manifest_path = model_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing model manifest: {manifest_path}")

    # The manifest names the inference artifact, so replacing the checkpoint
    # does not require a C++ rebuild.
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_name = manifest.get("inference_model_file")
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("model manifest is missing inference_model_file")
    policy = FrozenRainbowNumpy(model_dir / model_name, manifest_path)
    if policy.observation_dim != OBSERVATION_DIM:
        raise ValueError(
            f"model observation dimension is {policy.observation_dim}, "
            f"expected {OBSERVATION_DIM}"
        )
    if policy.action_dim != ACTION_COUNT:
        raise ValueError(
            f"model action count is {policy.action_dim}, expected {ACTION_COUNT}"
        )

    _policy = policy
    _session_type = BluePolicySession
    _bundle_dir = root
    _json_api = BluePolicyJsonApi(
        policy,
        BluePolicySession,
        root,
        interpreter_ownership=interpreter_ownership,
        native_dll_search_dirs=_native_dll_search_paths,
    )
    return _policy_info()


def _policy_info() -> dict[str, object]:
    policy = _require_policy()
    manifest = policy.manifest
    return {
        "model_id": str(manifest.get("model_id", "")),
        "observation_dim": policy.observation_dim,
        "action_dim": policy.action_dim,
        "bundle_dir": str(_bundle_dir),
    }


def _require_policy() -> Any:
    if _policy is None:
        raise RuntimeError("blue policy is not initialized")
    return _policy


def predict(observation: list[float] | tuple[float, ...]) -> int:
    """Run direct C51 inference for one normalized-v4 observation."""
    policy = _require_policy()
    if len(observation) != OBSERVATION_DIM:
        raise ValueError(
            f"observation length is {len(observation)}, expected {OBSERVATION_DIM}"
        )
    q_values = policy.expected_action_values(observation)[0]
    return int(q_values.argmax())


def dispatch_json(request_json: str) -> str:
    """Dispatch one strict JSON v1 request and return one JSON response."""
    if _json_api is None:
        raise RuntimeError("blue policy JSON API is not initialized")
    return str(_json_api.dispatch_json(request_json))


def create_session(input_mode: int, episode_id: int) -> int:
    """Create an independent stateful adapter/controller session."""
    global _next_session
    policy = _require_policy()
    if input_mode not in (0, 1):
        raise ValueError("input_mode must be 0 (exact) or 1 (kinematics_only)")
    mode_name = "exact" if input_mode == 0 else "kinematics_only"
    session = _session_type(policy, input_mode=mode_name)
    session.reset(episode_id=int(episode_id))
    handle = _next_session
    _next_session += 1
    _sessions[handle] = session
    return handle


def reset_session(handle: int, episode_id: int) -> None:
    _get_session(handle).reset(episode_id=int(episode_id))


def destroy_session(handle: int) -> None:
    try:
        del _sessions[int(handle)]
    except KeyError as error:
        raise KeyError(f"unknown blue policy session {handle}") from error


def _get_session(handle: int) -> Any:
    try:
        return _sessions[int(handle)]
    except KeyError as error:
        raise KeyError(f"unknown blue policy session {handle}") from error


def step(
    handle: int,
    step_count: int,
    time_s: float,
    blue_position_m: tuple[float, float, float],
    blue_velocity_mps: tuple[float, float, float],
    blue_alive: bool,
    actual_load_command_body_g: tuple[float, float, float],
    tracks: list[
        tuple[
            int,
            tuple[float, float, float],
            tuple[float, float, float],
            bool,
            float,
            int,
        ]
    ],
) -> tuple[object, ...]:
    """Run one 0.005 s frame and return a fixed-layout primitive tuple."""
    session = _get_session(handle)
    converted_tracks: list[dict[str, object]] = []
    for index, track in enumerate(tracks):
        if len(track) != 6:
            raise ValueError(f"track {index} must contain six fields")
        track_id, position, velocity, alive, energy, guidance_mode = track
        item: dict[str, object] = {
            "track_id": int(track_id),
            "position_m": position,
            "velocity_mps": velocity,
            "alive": bool(alive),
        }
        if session.input_mode == "exact":
            if not 0 <= int(guidance_mode) < len(_GUIDANCE_MODES):
                raise ValueError(
                    f"track {index} guidance mode {guidance_mode} is invalid"
                )
            item["energy"] = float(energy)
            item["guidance_mode"] = _GUIDANCE_MODES[int(guidance_mode)]
        converted_tracks.append(item)

    result = session.step(
        {
            "step_count": int(step_count),
            "time_s": float(time_s),
            "blue": {
                "position_m": blue_position_m,
                "velocity_mps": blue_velocity_mps,
                "alive": bool(blue_alive),
            },
            "actual_load_command_body_g": actual_load_command_body_g,
            "tracks": converted_tracks,
        }
    )
    visible_ids = tuple(int(value) for value in result["visible_track_ids"])
    if len(visible_ids) > 3:
        raise RuntimeError("runtime returned more than three visible track IDs")
    mode = str(result["mode"])
    if mode not in _RUNTIME_MODES:
        raise RuntimeError(f"runtime returned unsupported mode {mode!r}")

    return (
        _RUNTIME_MODES[mode],
        bool(result["active"]),
        bool(result["decision_updated"]),
        int(result["raw_action_index"]),
        int(result["action_index"]),
        tuple(float(value) for value in result["target_command_body_g"]),
        tuple(float(value) for value in result["applied_command_body_g"]),
        float(result["valid_until_s"]),
        bool(result["action_intervened"]),
        bool(result["safety_intervened"]),
        bool(result["fallback_required"]),
        int(result["safe_action_count"]),
        int(result["decision_count"]),
        visible_ids,
    )


def shutdown() -> None:
    """Release policy/session objects without touching CPython itself."""
    global _policy, _session_type, _json_api, _bundle_dir, _next_session
    if _json_api is not None:
        _json_api.close()
    _sessions.clear()
    _policy = None
    _session_type = None
    _json_api = None
    _bundle_dir = None
    _next_session = 1
    while _native_dll_directory_handles:
        _native_dll_directory_handles.pop().close()
    _native_dll_search_paths.clear()
