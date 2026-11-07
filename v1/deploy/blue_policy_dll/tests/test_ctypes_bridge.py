from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path

import numpy as np
import pytest


class InitConfig(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("python_home_utf8", ctypes.c_char_p),
        ("python_module_dir_utf8", ctypes.c_char_p),
        ("bundle_dir_utf8", ctypes.c_char_p),
    ]


class JsonResult(ctypes.Structure):
    _fields_ = [
        ("struct_size", ctypes.c_uint32),
        ("data_utf8", ctypes.c_void_p),
        ("size_bytes", ctypes.c_size_t),
    ]


def _load_bridge() -> tuple[ctypes.CDLL, bytes]:
    raw_path = os.environ.get("BLUE_POLICY_DLL")
    if not raw_path:
        pytest.skip("set BLUE_POLICY_DLL to run the borrowed-interpreter test")
    dll_path = Path(raw_path).resolve(strict=True)
    app_dir = dll_path.parent / "python_app"
    bridge = ctypes.CDLL(str(dll_path))
    bridge.blue_policy_api_version.restype = ctypes.c_uint32
    bridge.blue_policy_initialize.argtypes = [ctypes.POINTER(InitConfig)]
    bridge.blue_policy_initialize.restype = ctypes.c_int32
    bridge.blue_policy_predict.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_int32),
    ]
    bridge.blue_policy_predict.restype = ctypes.c_int32
    bridge.blue_policy_call_json.argtypes = [
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(JsonResult),
    ]
    bridge.blue_policy_call_json.restype = ctypes.c_int32
    bridge.blue_policy_release_json.argtypes = [ctypes.POINTER(JsonResult)]
    bridge.blue_policy_release_json.restype = None
    bridge.blue_policy_last_error.restype = ctypes.c_char_p
    bridge.blue_policy_shutdown.restype = ctypes.c_int32
    return bridge, os.fsencode(app_dir)


def _initialize(bridge: ctypes.CDLL, app_dir: bytes) -> None:
    config = InitConfig(
        struct_size=ctypes.sizeof(InitConfig),
        python_home_utf8=None,
        python_module_dir_utf8=app_dir,
        bundle_dir_utf8=app_dir,
    )
    status = bridge.blue_policy_initialize(ctypes.byref(config))
    assert status == 0, bridge.blue_policy_last_error().decode("utf-8")


def _call_json(
    bridge: ctypes.CDLL, request: dict[str, object]
) -> dict[str, object]:
    payload = json.dumps(
        request,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    ).encode("utf-8")
    result = JsonResult(struct_size=ctypes.sizeof(JsonResult))
    status = bridge.blue_policy_call_json(
        payload,
        len(payload),
        ctypes.byref(result),
    )
    native_error = bridge.blue_policy_last_error()
    assert status == 0, (
        native_error.decode("utf-8") if native_error else "unknown native error"
    )
    assert result.data_utf8
    assert result.size_bytes > 0
    raw = ctypes.string_at(result.data_utf8, result.size_bytes)
    assert ctypes.string_at(result.data_utf8 + result.size_bytes, 1) == b"\0"
    response = json.loads(raw.decode("utf-8"))

    bridge.blue_policy_release_json(ctypes.byref(result))
    assert result.struct_size == ctypes.sizeof(JsonResult)
    assert not result.data_utf8
    assert result.size_bytes == 0

    # Releasing an already-empty result is explicitly part of the C ABI.
    bridge.blue_policy_release_json(ctypes.byref(result))
    assert result.struct_size == ctypes.sizeof(JsonResult)
    assert not result.data_utf8
    assert result.size_bytes == 0
    return response


def _assert_success(response: dict[str, object], request_id: object) -> dict[str, object]:
    assert response["protocol_version"] == 1
    assert response["request_id"] == request_id
    assert response["ok"] is True
    result = response["result"]
    assert isinstance(result, dict)
    return result


def test_bridge_borrows_existing_interpreter_and_can_be_reopened() -> None:
    bridge, app_dir = _load_bridge()
    assert bridge.blue_policy_api_version() == 2
    _initialize(bridge, app_dir)

    # Keep the API v1 POD inference entry point working for existing hosts.
    observation = (ctypes.c_float * 104)()
    action = ctypes.c_int32(-1)
    assert bridge.blue_policy_predict(observation, 104, ctypes.byref(action)) == 0
    assert 0 <= action.value < 29

    info = _assert_success(
        _call_json(
            bridge,
            {
                "protocol_version": 1,
                "request_id": "info-1",
                "op": "info",
            },
        ),
        "info-1",
    )
    assert info["operation"] == "info"
    assert info["dll_api_version"] == 2
    assert "multi_aircraft_concurrent_safe_v1" in info["capabilities"]
    assert info["velocity_output_semantics"] == "next_physics_frame_velocity"

    reset = _assert_success(
        _call_json(
            bridge,
            {
                "protocol_version": 1,
                "request_id": 2,
                "op": "reset",
                "session_id": "blue-101",
                "aircraft_id": 101,
                "aircraft_type": "fighter",
                "episode_id": 7,
                "input_mode": "kinematics_only",
            },
        ),
        2,
    )
    assert reset == {
        "operation": "reset",
        "session_id": "blue-101",
        "aircraft_id": 101,
        "aircraft_type": "fighter",
        "episode_id": 7,
        "input_mode": "kinematics_only",
        "reset": True,
    }

    step_response = _call_json(
        bridge,
        {
            "protocol_version": 1,
            "request_id": "step-0",
            "op": "step",
            "session_id": "blue-101",
            "aircraft_id": 101,
            "aircraft_type": "fighter",
            "step_count": 0,
            "scenario_time_s": 0.0,
            "ownship": {
                "geodetic_position": {
                    "longitude_deg": 120.0,
                    "latitude_deg": 30.0,
                    "altitude_m": 10_000.0,
                },
                "velocity_north_up_east_mps": [300.0, 0.0, 0.0],
                "alive": True,
                "actual_load_command_body_g": [0.0, 1.0, 0.0],
            },
            "incoming_missiles": [
                {
                    "missile_id": 9001,
                    "missile_type": "test-missile",
                    "geodetic_position": {
                        "longitude_deg": 120.0,
                        "latitude_deg": 29.55,
                        "altitude_m": 10_000.0,
                    },
                    "velocity_north_up_east_mps": [1_000.0, 0.0, 0.0],
                    "alive": True,
                }
            ],
            "include_diagnostics": True,
        },
    )
    step = _assert_success(step_response, "step-0")
    assert step["operation"] == "step"
    assert step["session_id"] == "blue-101"
    assert step["aircraft_id"] == 101
    assert step["step_count"] == 0
    assert step["scenario_time_s"] == 0.0
    next_velocity = step["next_velocity_north_up_east_mps"]
    assert isinstance(next_velocity, list)
    assert len(next_velocity) == 3
    assert np.isfinite(np.asarray(next_velocity, dtype=np.float64)).all()
    assert "action_index" not in step
    assert "raw_action_index" not in step
    assert "action_index" not in step["diagnostics"]
    assert "raw_action_index" not in step["diagnostics"]
    assert "action_index" not in json.dumps(step_response)

    closed = _assert_success(
        _call_json(
            bridge,
            {
                "protocol_version": 1,
                "request_id": 4,
                "op": "close_session",
                "session_id": "blue-101",
                "aircraft_id": 101,
            },
        ),
        4,
    )
    assert closed == {
        "operation": "close_session",
        "session_id": "blue-101",
        "aircraft_id": 101,
        "closed": True,
    }
    assert bridge.blue_policy_shutdown() == 0

    # The host interpreter remains alive and the logical bridge can reopen.
    np.testing.assert_array_equal(np.arange(3), [0, 1, 2])
    different = InitConfig(
        struct_size=ctypes.sizeof(InitConfig),
        python_home_utf8=None,
        python_module_dir_utf8=app_dir + b"-different",
        bundle_dir_utf8=app_dir,
    )
    assert bridge.blue_policy_initialize(ctypes.byref(different)) == 7
    _initialize(bridge, app_dir)
    reopened = _assert_success(
        _call_json(
            bridge,
            {
                "protocol_version": 1,
                "request_id": "reopened",
                "op": "info",
            },
        ),
        "reopened",
    )
    assert reopened["dll_api_version"] == 2
    assert bridge.blue_policy_shutdown() == 0
