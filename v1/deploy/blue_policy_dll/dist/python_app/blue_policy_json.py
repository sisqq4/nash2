"""Strict JSON v1 adapter for the blue-policy DLL API v2.

The public protocol uses WGS-84 geodetic positions and north/up/east velocity
components.  Each aircraft owns an independent stateful ``BluePolicySession``;
the immutable Rainbow weights are shared between sessions.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import sys
import threading
from typing import Any


PROTOCOL_VERSION = 1
DLL_API_VERSION = 2
MAX_REQUEST_BYTES = 1_048_576
MAX_DEPTH = 32
MAX_TRACKS = 256
MAX_SESSIONS = 1024
MAX_INT64 = 2**63 - 1
MIN_INT64 = -(2**63)

_SESSION_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
_GUIDANCE_MODES = {"boost", "locked", "lock_hold", "inertial"}
_WGS84_A_M = 6_378_137.0
_WGS84_E2 = 6.69437999014e-3
_G0 = 9.80665
_TIME_TOLERANCE_S = 1.0e-9


class ProtocolError(Exception):
    """Stable business error converted to an ``ok=false`` response."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | None = None,
        details: dict[str, object] | None = None,
        retryable: bool = False,
        state_changed: bool | None = False,
        reset_required: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.path = path
        self.details = {} if details is None else details
        self.retryable = retryable
        self.state_changed = state_changed
        self.reset_required = reset_required


@dataclass(frozen=True)
class _Origin:
    latitude_deg: float
    longitude_deg: float
    ecef_m: tuple[float, float, float]
    north: tuple[float, float, float]
    up: tuple[float, float, float]
    east: tuple[float, float, float]


@dataclass
class _SessionState:
    runtime: Any
    aircraft_id: int
    aircraft_type: str
    input_mode: str
    episode_id: int
    lock: Any = field(default_factory=threading.RLock)
    origin: _Origin | None = None
    last_step_count: int | None = None
    last_scenario_time_s: float | None = None
    tainted: bool = False


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _add_scaled(
    first: tuple[float, float, float],
    first_scale: float,
    second: tuple[float, float, float],
    second_scale: float,
    third: tuple[float, float, float],
    third_scale: float,
) -> tuple[float, float, float]:
    return tuple(
        first[index] * first_scale
        + second[index] * second_scale
        + third[index] * third_scale
        for index in range(3)
    )  # type: ignore[return-value]


def _basis(
    latitude_deg: float, longitude_deg: float
) -> tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)
    sin_lon = math.sin(longitude)
    cos_lon = math.cos(longitude)
    north = (-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat)
    up = (cos_lat * cos_lon, cos_lat * sin_lon, sin_lat)
    east = (-sin_lon, cos_lon, 0.0)
    return north, up, east


def _ecef(
    latitude_deg: float, longitude_deg: float, altitude_m: float
) -> tuple[float, float, float]:
    latitude = math.radians(latitude_deg)
    longitude = math.radians(longitude_deg)
    sin_lat = math.sin(latitude)
    cos_lat = math.cos(latitude)
    radius = _WGS84_A_M / math.sqrt(1.0 - _WGS84_E2 * sin_lat * sin_lat)
    return (
        (radius + altitude_m) * cos_lat * math.cos(longitude),
        (radius + altitude_m) * cos_lat * math.sin(longitude),
        (radius * (1.0 - _WGS84_E2) + altitude_m) * sin_lat,
    )


def _make_origin(latitude_deg: float, longitude_deg: float) -> _Origin:
    north, up, east = _basis(latitude_deg, longitude_deg)
    return _Origin(
        latitude_deg=latitude_deg,
        longitude_deg=longitude_deg,
        ecef_m=_ecef(latitude_deg, longitude_deg, 0.0),
        north=north,
        up=up,
        east=east,
    )


def _position_in_model_frame(
    geodetic: dict[str, float], origin: _Origin
) -> tuple[float, float, float]:
    value = _ecef(
        geodetic["latitude_deg"],
        geodetic["longitude_deg"],
        geodetic["altitude_m"],
    )
    delta = tuple(value[index] - origin.ecef_m[index] for index in range(3))
    return (
        _dot(delta, origin.north),
        geodetic["altitude_m"],
        _dot(delta, origin.east),
    )


def _velocity_in_model_frame(
    velocity_nue_mps: tuple[float, float, float],
    geodetic: dict[str, float],
    origin: _Origin,
) -> tuple[float, float, float]:
    entity_north, entity_up, entity_east = _basis(
        geodetic["latitude_deg"], geodetic["longitude_deg"]
    )
    ecef_velocity = _add_scaled(
        entity_north,
        velocity_nue_mps[0],
        entity_up,
        velocity_nue_mps[1],
        entity_east,
        velocity_nue_mps[2],
    )
    return (
        _dot(ecef_velocity, origin.north),
        _dot(ecef_velocity, origin.up),
        _dot(ecef_velocity, origin.east),
    )


def _velocity_from_model_frame(
    velocity_model_mps: tuple[float, float, float],
    geodetic: dict[str, float],
    origin: _Origin,
) -> tuple[float, float, float]:
    ecef_velocity = _add_scaled(
        origin.north,
        velocity_model_mps[0],
        origin.up,
        velocity_model_mps[1],
        origin.east,
        velocity_model_mps[2],
    )
    entity_north, entity_up, entity_east = _basis(
        geodetic["latitude_deg"], geodetic["longitude_deg"]
    )
    return (
        _dot(ecef_velocity, entity_north),
        _dot(ecef_velocity, entity_up),
        _dot(ecef_velocity, entity_east),
    )


def _norm(vector: tuple[float, float, float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _unit(
    vector: tuple[float, float, float],
    fallback: tuple[float, float, float],
) -> tuple[float, float, float]:
    length = _norm(vector)
    if length <= 1.0e-12:
        return fallback
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _next_velocity(
    position_m: tuple[float, float, float],
    velocity_mps: tuple[float, float, float],
    applied_command_body_g: tuple[float, float, float],
    config: Any,
) -> tuple[float, float, float]:
    """Propagate one physical frame using the training aircraft dynamics."""
    forward = _unit(velocity_mps, (1.0, 0.0, 0.0))
    inertial_up = (0.0, 1.0, 0.0)
    projection = _dot(inertial_up, forward)
    local_up = tuple(
        inertial_up[index] - projection * forward[index] for index in range(3)
    )
    if _norm(local_up) <= 1.0e-12:
        inertial_east = (0.0, 0.0, 1.0)
        east_projection = _dot(inertial_east, forward)
        local_up = tuple(
            inertial_east[index] - east_projection * forward[index]
            for index in range(3)
        )
    local_up = _unit(local_up, (0.0, 1.0, 0.0))
    local_right = _unit(_cross(forward, local_up), (0.0, 0.0, 1.0))

    axial_load_g, normal_load_g, bank_rad = applied_command_body_g
    normal_direction = tuple(
        math.cos(bank_rad) * local_up[index]
        - math.sin(bank_rad) * local_right[index]
        for index in range(3)
    )
    acceleration = tuple(
        axial_load_g * _G0 * forward[index]
        + normal_load_g * _G0 * normal_direction[index]
        + (0.0, -_G0, 0.0)[index]
        for index in range(3)
    )
    dt = float(config.time_step_s)
    next_position = tuple(
        position_m[index]
        + velocity_mps[index] * dt
        + 0.5 * acceleration[index] * dt * dt
        for index in range(3)
    )
    raw_velocity = tuple(
        velocity_mps[index] + acceleration[index] * dt for index in range(3)
    )
    raw_speed = _norm(raw_velocity)
    speed = min(
        float(config.aircraft_max_speed_mps),
        max(float(config.aircraft_min_speed_mps), raw_speed),
    )
    next_velocity = tuple(
        value * speed
        for value in _unit(raw_velocity, forward)
    )
    if next_position[1] < float(config.aircraft_min_altitude_m):
        next_velocity = (
            next_velocity[0], max(0.0, next_velocity[1]), next_velocity[2]
        )
    elif next_position[1] > float(config.aircraft_max_altitude_m):
        next_velocity = (
            next_velocity[0], min(0.0, next_velocity[1]), next_velocity[2]
        )
    if not all(math.isfinite(value) for value in next_velocity):
        raise RuntimeError("aircraft propagation produced non-finite velocity")
    return next_velocity


def _reject_constant(value: str) -> None:
    raise ProtocolError(
        "BPJ_INVALID_JSON",
        f"non-standard JSON number {value!r} is not allowed",
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ProtocolError(
                "BPJ_INVALID_JSON",
                f"duplicate JSON key {key!r}",
                details={"key": key},
            )
        result[key] = value
    return result


def _json_depth(value: object) -> int:
    if isinstance(value, dict):
        return 1 + max((_json_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_json_depth(item) for item in value), default=0)
    return 0


def _bundle_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        [*root.rglob("*.py"), root / "model" / "manifest.json"],
        key=lambda value: value.relative_to(root).as_posix(),
    )
    for path in files:
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative)
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def _protocol_error(
    code: str, message: str, path: str, **details: object
) -> ProtocolError:
    return ProtocolError(code, message, path=path, details=dict(details))


def _object(value: object, path: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise _protocol_error("BPJ_TYPE_MISMATCH", "must be an object", path)
    return value


def _strict_keys(
    value: dict[str, object],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    path: str,
) -> None:
    missing = sorted(required - value.keys())
    if missing:
        name = missing[0]
        raise _protocol_error(
            "BPJ_MISSING_FIELD",
            f"missing required field {name!r}",
            f"{path}.{name}",
        )
    unknown = sorted(value.keys() - required - optional)
    if unknown:
        name = unknown[0]
        raise _protocol_error(
            "BPJ_UNKNOWN_FIELD",
            f"unknown field {name!r}",
            f"{path}.{name}",
        )


def _integer(
    value: object,
    path: str,
    *,
    minimum: int = MIN_INT64,
    maximum: int = MAX_INT64,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _protocol_error("BPJ_TYPE_MISMATCH", "must be an integer", path)
    if value < minimum or value > maximum:
        raise _protocol_error(
            "BPJ_OUT_OF_RANGE",
            f"must be in [{minimum}, {maximum}]",
            path,
            minimum=minimum,
            maximum=maximum,
            actual=value,
        )
    return value


def _number(
    value: object,
    path: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _protocol_error("BPJ_TYPE_MISMATCH", "must be a number", path)
    result = float(value)
    if not math.isfinite(result):
        raise _protocol_error("BPJ_OUT_OF_RANGE", "must be finite", path)
    if minimum is not None and result < minimum:
        raise _protocol_error(
            "BPJ_OUT_OF_RANGE", f"must be >= {minimum}", path, minimum=minimum
        )
    if maximum is not None and result > maximum:
        raise _protocol_error(
            "BPJ_OUT_OF_RANGE", f"must be <= {maximum}", path, maximum=maximum
        )
    return result


def _string(
    value: object,
    path: str,
    *,
    minimum_length: int = 1,
    maximum_length: int = 128,
) -> str:
    if not isinstance(value, str):
        raise _protocol_error("BPJ_TYPE_MISMATCH", "must be a string", path)
    if not minimum_length <= len(value) <= maximum_length:
        raise _protocol_error(
            "BPJ_OUT_OF_RANGE",
            f"length must be in [{minimum_length}, {maximum_length}]",
            path,
        )
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise _protocol_error("BPJ_TYPE_MISMATCH", "must be boolean", path)
    return value


def _vector3(value: object, path: str) -> tuple[float, float, float]:
    if not isinstance(value, list) or len(value) != 3:
        raise _protocol_error(
            "BPJ_TYPE_MISMATCH", "must be a three-element array", path
        )
    return (
        _number(value[0], f"{path}[0]"),
        _number(value[1], f"{path}[1]"),
        _number(value[2], f"{path}[2]"),
    )


def _geodetic(value: object, path: str) -> dict[str, float]:
    result = _object(value, path)
    _strict_keys(
        result,
        required={"longitude_deg", "latitude_deg", "altitude_m"},
        path=path,
    )
    return {
        "longitude_deg": _number(
            result["longitude_deg"], f"{path}.longitude_deg", minimum=-180.0, maximum=180.0
        ),
        "latitude_deg": _number(
            result["latitude_deg"], f"{path}.latitude_deg", minimum=-90.0, maximum=90.0
        ),
        "altitude_m": _number(
            result["altitude_m"], f"{path}.altitude_m", minimum=-1_000.0, maximum=100_000.0
        ),
    }


def _session_id(value: object, path: str = "$.session_id") -> str:
    result = _string(value, path, maximum_length=128)
    if _SESSION_PATTERN.fullmatch(result) is None:
        raise _protocol_error(
            "BPJ_OUT_OF_RANGE",
            "must match [A-Za-z0-9_.:-]+",
            path,
        )
    return result


def _request_id(value: object) -> int | str:
    if isinstance(value, bool):
        raise _protocol_error(
            "BPJ_TYPE_MISMATCH", "must be a non-negative int64 or string", "$.request_id"
        )
    if isinstance(value, int):
        return _integer(value, "$.request_id", minimum=0)
    return _string(value, "$.request_id", maximum_length=64)


class BluePolicyJsonApi:
    """Thread-safe registry and strict public protocol implementation."""

    def __init__(
        self,
        policy: Any,
        session_type: Any,
        bundle_dir: Path,
        *,
        interpreter_ownership: str,
        native_dll_search_dirs: list[str],
    ) -> None:
        self.policy = policy
        self.session_type = session_type
        self.bundle_dir = bundle_dir
        self.interpreter_ownership = interpreter_ownership
        self.native_dll_search_dirs = list(native_dll_search_dirs)
        self.sessions: dict[str, _SessionState] = {}
        self.aircraft_sessions: dict[int, str] = {}
        self.registry_lock = threading.RLock()
        for name in getattr(policy, "_ARRAY_NAMES", ()):
            array = getattr(policy, name, None)
            if hasattr(array, "setflags"):
                array.setflags(write=False)

    def close(self) -> None:
        with self.registry_lock:
            self.sessions.clear()
            self.aircraft_sessions.clear()

    def dispatch_json(self, request_text: str) -> str:
        request_id: int | str | None = None
        try:
            if not isinstance(request_text, str):
                raise ProtocolError(
                    "BPJ_INVALID_UTF8", "request must be decoded UTF-8 text"
                )
            if request_text.startswith("\ufeff") or "\x00" in request_text:
                raise ProtocolError(
                    "BPJ_INVALID_UTF8", "UTF-8 BOM and embedded NUL are not allowed"
                )
            if len(request_text.encode("utf-8")) > MAX_REQUEST_BYTES:
                raise ProtocolError(
                    "BPJ_OUT_OF_RANGE",
                    f"request exceeds {MAX_REQUEST_BYTES} bytes",
                    path="$",
                )
            try:
                request = json.loads(
                    request_text,
                    parse_constant=_reject_constant,
                    object_pairs_hook=_reject_duplicate_pairs,
                )
            except ProtocolError:
                raise
            except json.JSONDecodeError as error:
                raise ProtocolError(
                    "BPJ_INVALID_JSON",
                    f"invalid JSON at line {error.lineno}, column {error.colno}",
                    details={"line": error.lineno, "column": error.colno},
                ) from error
            if _json_depth(request) > MAX_DEPTH:
                raise ProtocolError(
                    "BPJ_OUT_OF_RANGE",
                    f"JSON nesting depth exceeds {MAX_DEPTH}",
                    path="$",
                )
            if isinstance(request, dict) and "request_id" in request:
                try:
                    request_id = _request_id(request["request_id"])
                except ProtocolError:
                    request_id = None
            result = self._dispatch(_object(request, "$"))
            response: dict[str, object] = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": True,
                "result": result,
            }
        except ProtocolError as error:
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "path": error.path,
                    "retryable": error.retryable,
                    "state_changed": error.state_changed,
                    "reset_required": error.reset_required,
                    "details": error.details,
                },
            }
        except Exception as error:  # Defensive ABI boundary.
            response = {
                "protocol_version": PROTOCOL_VERSION,
                "request_id": request_id,
                "ok": False,
                "error": {
                    "code": "BPJ_INTERNAL_ERROR",
                    "message": str(error) or type(error).__name__,
                    "path": None,
                    "retryable": False,
                    "state_changed": None,
                    "reset_required": True,
                    "details": {},
                },
            }
        return json.dumps(
            response,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )

    def _dispatch(self, request: dict[str, object]) -> dict[str, object]:
        common = {"protocol_version", "request_id", "op"}
        missing = common - request.keys()
        if missing:
            name = sorted(missing)[0]
            raise _protocol_error(
                "BPJ_MISSING_FIELD", f"missing required field {name!r}", f"$.{name}"
            )
        version = _integer(request["protocol_version"], "$.protocol_version")
        if version != PROTOCOL_VERSION:
            raise _protocol_error(
                "BPJ_UNSUPPORTED_PROTOCOL",
                f"unsupported protocol_version {version}",
                "$.protocol_version",
                supported=PROTOCOL_VERSION,
            )
        _request_id(request["request_id"])
        operation = _string(request["op"], "$.op", maximum_length=32)
        if operation == "info":
            _strict_keys(request, required=common, path="$")
            return self._info()
        if operation == "reset":
            return self._reset(request)
        if operation == "step":
            return self._step(request)
        if operation == "close_session":
            return self._close_session(request)
        raise _protocol_error(
            "BPJ_UNSUPPORTED_OPERATION",
            f"unsupported operation {operation!r}",
            "$.op",
        )

    def _info(self) -> dict[str, object]:
        manifest = self.policy.manifest
        contract = manifest["model_contract"]
        runtime = manifest["runtime_contract"]
        return {
            "operation": "info",
            "dll_api_version": DLL_API_VERSION,
            "json_protocol_version": PROTOCOL_VERSION,
            "capabilities": [
                "json_v1",
                "geodetic_input_v1",
                "next_velocity_output_v1",
                "multi_aircraft_concurrent_safe_v1",
            ],
            "model_id": str(manifest["model_id"]),
            "model_sha256": str(manifest["inference_model_sha256"]),
            "python_bundle_sha256": _bundle_sha256(self.bundle_dir),
            "python_version": platform.python_version(),
            "numpy_version": str(getattr(sys.modules.get("numpy"), "__version__", "")),
            "architecture": f"{sys.platform}-{platform.machine().lower()}",
            "python_module_dir": str(Path(__file__).resolve().parent),
            "sys_prefix": sys.prefix,
            "interpreter_ownership": self.interpreter_ownership,
            "native_dll_search_dirs": self.native_dll_search_dirs,
            "observation_schema": str(contract["observation_schema"]),
            "observation_dim": int(contract["observation_dim"]),
            "action_count": int(contract["action_dim"]),
            "atom_count": int(contract["atoms"]),
            "threat_slot_count": int(contract["missile_slots"]),
            "physics_step_s": float(runtime["time_step_s"]),
            "decision_interval_s": float(runtime["decision_interval_s"]),
            "detection_range_m": float(runtime["detection_range_m"]),
            "detection_boundary": "strictly_less_than",
            "coordinates": {
                "public_position": "WGS-84 longitude_deg,latitude_deg,altitude_m",
                "public_velocity_order": "north,up,east",
                "internal_axes": "x=north,y=up,z=east",
                "position_unit": "m",
                "velocity_unit": "m/s",
                "time_unit": "s",
            },
            "supported_input_modes": ["exact", "kinematics_only"],
            "aircraft_type_handling": "metadata_only_shared_model_v1",
            "velocity_output_semantics": "next_physics_frame_velocity",
            "limits": {
                "request_size_bytes": MAX_REQUEST_BYTES,
                "max_depth": MAX_DEPTH,
                "max_tracks": MAX_TRACKS,
                "max_sessions": MAX_SESSIONS,
            },
            "concurrency_mode": "serialized_gil_v1",
        }

    def _reset(self, request: dict[str, object]) -> dict[str, object]:
        required = {
            "protocol_version",
            "request_id",
            "op",
            "session_id",
            "aircraft_id",
            "aircraft_type",
            "episode_id",
            "input_mode",
        }
        _strict_keys(request, required=required, path="$")
        session_id = _session_id(request["session_id"])
        aircraft_id = _integer(request["aircraft_id"], "$.aircraft_id", minimum=0)
        aircraft_type = _string(
            request["aircraft_type"], "$.aircraft_type", maximum_length=64
        )
        episode_id = _integer(request["episode_id"], "$.episode_id", minimum=0)
        input_mode = _string(request["input_mode"], "$.input_mode", maximum_length=32)
        if input_mode not in {"exact", "kinematics_only"}:
            raise _protocol_error(
                "BPJ_OUT_OF_RANGE",
                "input_mode must be exact or kinematics_only",
                "$.input_mode",
            )

        runtime = self.session_type(self.policy, input_mode=input_mode)
        runtime.reset(episode_id=episode_id)
        replacement = _SessionState(
            runtime=runtime,
            aircraft_id=aircraft_id,
            aircraft_type=aircraft_type,
            input_mode=input_mode,
            episode_id=episode_id,
        )
        with self.registry_lock:
            owner = self.aircraft_sessions.get(aircraft_id)
            if owner is not None and owner != session_id:
                raise ProtocolError(
                    "BPJ_AIRCRAFT_IN_USE",
                    f"aircraft_id {aircraft_id} is already bound to session {owner!r}",
                    path="$.aircraft_id",
                )
            if session_id not in self.sessions and len(self.sessions) >= MAX_SESSIONS:
                raise ProtocolError(
                    "BPJ_OUT_OF_RANGE",
                    f"session limit {MAX_SESSIONS} reached",
                    path="$.session_id",
                )
            previous = self.sessions.get(session_id)
            if previous is not None:
                self.aircraft_sessions.pop(previous.aircraft_id, None)
            self.sessions[session_id] = replacement
            self.aircraft_sessions[aircraft_id] = session_id
        return {
            "operation": "reset",
            "session_id": session_id,
            "aircraft_id": aircraft_id,
            "aircraft_type": aircraft_type,
            "episode_id": episode_id,
            "input_mode": input_mode,
            "reset": True,
        }

    def _session(self, session_id: str) -> _SessionState:
        with self.registry_lock:
            state = self.sessions.get(session_id)
        if state is None:
            raise ProtocolError(
                "BPJ_SESSION_NOT_FOUND",
                f"unknown session_id {session_id!r}; call reset first",
                path="$.session_id",
            )
        return state

    def _validate_ownship(
        self, value: object, state: _SessionState
    ) -> tuple[
        dict[str, float],
        tuple[float, float, float],
        bool,
        tuple[float, float, float] | None,
    ]:
        path = "$.ownship"
        ownship = _object(value, path)
        _strict_keys(
            ownship,
            required={"geodetic_position", "velocity_north_up_east_mps", "alive"},
            optional={"actual_load_command_body_g"},
            path=path,
        )
        geodetic = _geodetic(ownship["geodetic_position"], f"{path}.geodetic_position")
        minimum_altitude = float(state.runtime.config.aircraft_min_altitude_m)
        maximum_altitude = float(state.runtime.config.aircraft_max_altitude_m)
        if not minimum_altitude <= geodetic["altitude_m"] <= maximum_altitude:
            raise _protocol_error(
                "BPJ_OUT_OF_RANGE",
                f"ownship altitude must be in [{minimum_altitude}, {maximum_altitude}] m",
                f"{path}.geodetic_position.altitude_m",
            )
        velocity = _vector3(
            ownship["velocity_north_up_east_mps"],
            f"{path}.velocity_north_up_east_mps",
        )
        speed = _norm(velocity)
        minimum_speed = float(state.runtime.config.aircraft_min_speed_mps)
        maximum_speed = float(state.runtime.config.aircraft_max_speed_mps)
        if not minimum_speed <= speed <= maximum_speed:
            raise _protocol_error(
                "BPJ_OUT_OF_RANGE",
                f"ownship speed must be in [{minimum_speed}, {maximum_speed}] m/s",
                f"{path}.velocity_north_up_east_mps",
                actual=speed,
            )
        alive = _boolean(ownship["alive"], f"{path}.alive")
        actual = (
            _vector3(
                ownship["actual_load_command_body_g"],
                f"{path}.actual_load_command_body_g",
            )
            if "actual_load_command_body_g" in ownship
            else None
        )
        if actual is not None:
            max_load = float(state.runtime.config.aircraft_max_load_factor_g)
            if abs(actual[0]) > max_load or abs(actual[1]) > max_load:
                raise _protocol_error(
                    "BPJ_OUT_OF_RANGE",
                    f"axial and normal load feedback must be in [-{max_load}, {max_load}] g",
                    f"{path}.actual_load_command_body_g",
                    minimum=-max_load,
                    maximum=max_load,
                )
            if abs(actual[2]) > math.pi:
                raise _protocol_error(
                    "BPJ_OUT_OF_RANGE",
                    "bank feedback must be in [-pi, pi] radians",
                    f"{path}.actual_load_command_body_g[2]",
                    minimum=-math.pi,
                    maximum=math.pi,
                )
        return geodetic, velocity, alive, actual

    def _validate_missiles(
        self, value: object, input_mode: str
    ) -> list[dict[str, object]]:
        if not isinstance(value, list):
            raise _protocol_error(
                "BPJ_TYPE_MISMATCH", "must be an array", "$.incoming_missiles"
            )
        if len(value) > MAX_TRACKS:
            raise _protocol_error(
                "BPJ_OUT_OF_RANGE",
                f"must contain at most {MAX_TRACKS} missiles",
                "$.incoming_missiles",
            )
        result: list[dict[str, object]] = []
        seen: set[int] = set()
        for index, raw in enumerate(value):
            path = f"$.incoming_missiles[{index}]"
            missile = _object(raw, path)
            _strict_keys(
                missile,
                required={
                    "missile_id",
                    "geodetic_position",
                    "velocity_north_up_east_mps",
                    "alive",
                },
                optional={"missile_type", "energy", "guidance_mode"},
                path=path,
            )
            missile_id = _integer(missile["missile_id"], f"{path}.missile_id")
            if missile_id in seen:
                raise ProtocolError(
                    "BPJ_DUPLICATE_TRACK_ID",
                    f"duplicate missile_id {missile_id}",
                    path=f"{path}.missile_id",
                )
            seen.add(missile_id)
            item: dict[str, object] = {
                "missile_id": missile_id,
                "geodetic_position": _geodetic(
                    missile["geodetic_position"], f"{path}.geodetic_position"
                ),
                "velocity_north_up_east_mps": _vector3(
                    missile["velocity_north_up_east_mps"],
                    f"{path}.velocity_north_up_east_mps",
                ),
                "alive": _boolean(missile["alive"], f"{path}.alive"),
            }
            if "missile_type" in missile:
                item["missile_type"] = _string(
                    missile["missile_type"], f"{path}.missile_type", maximum_length=64
                )
            if input_mode == "exact" and (
                "energy" not in missile or "guidance_mode" not in missile
            ):
                missing = "energy" if "energy" not in missile else "guidance_mode"
                raise _protocol_error(
                    "BPJ_MISSING_FIELD",
                    f"exact mode requires {missing}",
                    f"{path}.{missing}",
                )
            if "energy" in missile:
                item["energy"] = _number(
                    missile["energy"], f"{path}.energy", minimum=0.0, maximum=1.0
                )
            if "guidance_mode" in missile:
                guidance = _string(
                    missile["guidance_mode"], f"{path}.guidance_mode", maximum_length=32
                )
                if guidance not in _GUIDANCE_MODES:
                    raise _protocol_error(
                        "BPJ_OUT_OF_RANGE",
                        "unsupported guidance_mode",
                        f"{path}.guidance_mode",
                    )
                item["guidance_mode"] = guidance
            result.append(item)
        return result

    def _step(self, request: dict[str, object]) -> dict[str, object]:
        required = {
            "protocol_version",
            "request_id",
            "op",
            "session_id",
            "aircraft_id",
            "aircraft_type",
            "step_count",
            "scenario_time_s",
            "ownship",
            "incoming_missiles",
        }
        _strict_keys(
            request,
            required=required,
            optional={"include_diagnostics"},
            path="$",
        )
        session_id = _session_id(request["session_id"])
        aircraft_id = _integer(request["aircraft_id"], "$.aircraft_id", minimum=0)
        aircraft_type = _string(
            request["aircraft_type"], "$.aircraft_type", maximum_length=64
        )
        step_count = _integer(request["step_count"], "$.step_count", minimum=0)
        scenario_time_s = _number(
            request["scenario_time_s"], "$.scenario_time_s", minimum=0.0
        )
        include_diagnostics = (
            _boolean(request["include_diagnostics"], "$.include_diagnostics")
            if "include_diagnostics" in request
            else False
        )
        state = self._session(session_id)
        with state.lock:
            if state.aircraft_id != aircraft_id or state.aircraft_type != aircraft_type:
                raise ProtocolError(
                    "BPJ_AIRCRAFT_MISMATCH",
                    "aircraft identity does not match the session binding",
                    path="$.aircraft_id",
                    details={
                        "expected_aircraft_id": state.aircraft_id,
                        "expected_aircraft_type": state.aircraft_type,
                    },
                )
            if state.tainted:
                raise ProtocolError(
                    "BPJ_SESSION_TAINTED",
                    "session must be reset after an unrecoverable internal error",
                    path="$.session_id",
                    reset_required=True,
                )
            if state.last_step_count is not None:
                expected_step = state.last_step_count + 1
                expected_time = state.last_scenario_time_s + float(
                    state.runtime.config.time_step_s
                )
                if step_count != expected_step or not math.isclose(
                    scenario_time_s,
                    expected_time,
                    rel_tol=0.0,
                    abs_tol=_TIME_TOLERANCE_S,
                ):
                    raise ProtocolError(
                        "BPJ_SEQUENCE_ERROR",
                        "step_count and scenario_time_s must advance by one physics frame",
                        path="$.step_count",
                        details={
                            "expected_step_count": expected_step,
                            "expected_scenario_time_s": expected_time,
                        },
                    )

            own_geodetic, own_velocity_nue, own_alive, actual_load = (
                self._validate_ownship(request["ownship"], state)
            )
            missiles = self._validate_missiles(
                request["incoming_missiles"], state.input_mode
            )
            candidate_origin = state.origin or _make_origin(
                own_geodetic["latitude_deg"], own_geodetic["longitude_deg"]
            )
            own_position = _position_in_model_frame(own_geodetic, candidate_origin)
            own_velocity = _velocity_in_model_frame(
                own_velocity_nue, own_geodetic, candidate_origin
            )
            tracks: list[dict[str, object]] = []
            for missile in missiles:
                geodetic = missile["geodetic_position"]
                assert isinstance(geodetic, dict)
                velocity_nue = missile["velocity_north_up_east_mps"]
                assert isinstance(velocity_nue, tuple)
                track: dict[str, object] = {
                    "track_id": missile["missile_id"],
                    "position_m": _position_in_model_frame(geodetic, candidate_origin),
                    "velocity_mps": _velocity_in_model_frame(
                        velocity_nue, geodetic, candidate_origin
                    ),
                    "alive": missile["alive"],
                }
                if state.input_mode == "exact":
                    track["energy"] = missile["energy"]
                    track["guidance_mode"] = missile["guidance_mode"]
                tracks.append(track)

            frame: dict[str, object] = {
                "step_count": step_count,
                "time_s": scenario_time_s,
                "blue": {
                    "position_m": own_position,
                    "velocity_mps": own_velocity,
                    "alive": own_alive,
                },
                "tracks": tracks,
            }
            if actual_load is not None:
                frame["actual_load_command_body_g"] = actual_load

            try:
                snapshot = copy.deepcopy(
                    state.runtime, memo={id(self.policy): self.policy}
                )
            except Exception as error:
                state.tainted = True
                raise ProtocolError(
                    "BPJ_SESSION_TAINTED",
                    f"could not snapshot session state: {error}",
                    path="$.session_id",
                    state_changed=None,
                    reset_required=True,
                ) from error

            try:
                internal = state.runtime.step(frame, include_diagnostics=False)
                applied = tuple(
                    float(value) for value in internal["applied_command_body_g"]
                )
                next_model_velocity = _next_velocity(
                    own_position,
                    own_velocity,
                    applied,
                    state.runtime.config,
                )
                next_public_velocity = _velocity_from_model_frame(
                    next_model_velocity, own_geodetic, candidate_origin
                )
            except Exception as error:
                state.runtime = snapshot
                raise ProtocolError(
                    "BPJ_INTERNAL_ERROR",
                    f"blue policy step failed: {error}",
                    path=None,
                    state_changed=False,
                    reset_required=False,
                ) from error

            state.origin = candidate_origin
            state.last_step_count = step_count
            state.last_scenario_time_s = scenario_time_s
            result: dict[str, object] = {
                "operation": "step",
                "session_id": session_id,
                "aircraft_id": aircraft_id,
                "step_count": step_count,
                "scenario_time_s": scenario_time_s,
                "velocity_effective_time_s": float(internal["valid_until_s"]),
                "next_velocity_north_up_east_mps": list(next_public_velocity),
                "active": bool(internal["active"]),
                "fallback_required": bool(internal["fallback_required"]),
            }
            if include_diagnostics:
                result["diagnostics"] = {
                    "mode": str(internal["mode"]),
                    "decision_updated": bool(internal["decision_updated"]),
                    "visible_missile_ids": list(internal["visible_track_ids"]),
                    "decision_count": int(internal["decision_count"]),
                    "action_intervened": bool(internal["action_intervened"]),
                    "safety_intervened": bool(internal["safety_intervened"]),
                    "applied_load_command_body_g": list(applied),
                }
            return result

    def _close_session(self, request: dict[str, object]) -> dict[str, object]:
        required = {
            "protocol_version",
            "request_id",
            "op",
            "session_id",
            "aircraft_id",
        }
        _strict_keys(request, required=required, path="$")
        session_id = _session_id(request["session_id"])
        aircraft_id = _integer(request["aircraft_id"], "$.aircraft_id", minimum=0)
        with self.registry_lock:
            state = self.sessions.get(session_id)
            if state is None:
                return {
                    "operation": "close_session",
                    "session_id": session_id,
                    "aircraft_id": aircraft_id,
                    "closed": False,
                }
            if state.aircraft_id != aircraft_id:
                raise ProtocolError(
                    "BPJ_AIRCRAFT_MISMATCH",
                    "aircraft_id does not match the session binding",
                    path="$.aircraft_id",
                    details={"expected_aircraft_id": state.aircraft_id},
                )
            self.sessions.pop(session_id)
            self.aircraft_sessions.pop(aircraft_id, None)
        return {
            "operation": "close_session",
            "session_id": session_id,
            "aircraft_id": aircraft_id,
            "closed": True,
        }
