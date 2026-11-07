from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import sys

import jsonschema
import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY_DIR = ROOT / "python_app"
BUNDLE_DIR = ROOT.parent / "blue_policy_exe"
SCHEMA_DIR = ROOT.parents[1] / "docs" / "schemas"
DESIGN_DOCUMENT = ROOT.parents[1] / "docs" / "BLUE_POLICY_DLL_DESIGN.md"
DELIVERY_README = ROOT / "README.md"
sys.path.insert(0, str(ENTRY_DIR))

import blue_policy_entry as entry  # noqa: E402
from blue_policy_json import (  # noqa: E402
    _make_origin,
    _position_in_model_frame,
    _velocity_from_model_frame,
    _velocity_in_model_frame,
)


@pytest.fixture(scope="module", autouse=True)
def initialized_policy() -> None:
    entry.initialize(str(BUNDLE_DIR), "borrowed")
    yield
    entry.shutdown()


def call(request: dict[str, object]) -> dict[str, object]:
    return json.loads(entry.dispatch_json(json.dumps(request)))


def reset_request(index: int, *, input_mode: str = "kinematics_only") -> dict[str, object]:
    return {
        "protocol_version": 1,
        "request_id": f"reset-{index}",
        "op": "reset",
        "session_id": f"blue-{index}",
        "aircraft_id": index,
        "aircraft_type": "fighter-test",
        "episode_id": 1,
        "input_mode": input_mode,
    }


def step_request(
    index: int,
    *,
    step_count: int = 0,
    time_s: float = 0.0,
    missiles: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "protocol_version": 1,
        "request_id": f"step-{index}-{step_count}",
        "op": "step",
        "session_id": f"blue-{index}",
        "aircraft_id": index,
        "aircraft_type": "fighter-test",
        "step_count": step_count,
        "scenario_time_s": time_s,
        "ownship": {
            "geodetic_position": {
                "longitude_deg": 108.0 + index * 1.0e-4,
                "latitude_deg": 34.0,
                "altitude_m": 10_000.0,
            },
            "velocity_north_up_east_mps": [300.0, 0.0, 0.0],
            "alive": True,
        },
        "incoming_missiles": [] if missiles is None else missiles,
    }


def test_info_declares_api_v2_and_concurrency() -> None:
    response = call({"protocol_version": 1, "request_id": 1, "op": "info"})
    assert response["ok"] is True
    result = response["result"]
    assert result["dll_api_version"] == 2
    assert result["concurrency_mode"] == "serialized_gil_v1"
    assert "geodetic_input_v1" in result["capabilities"]
    assert "next_velocity_output_v1" in result["capabilities"]


def test_public_step_returns_aircraft_velocity_not_action_index() -> None:
    assert call(reset_request(101))["ok"] is True
    response = call(step_request(101))
    assert response["ok"] is True
    result = response["result"]
    assert result["aircraft_id"] == 101
    assert result["velocity_effective_time_s"] == pytest.approx(0.005)
    assert result["next_velocity_north_up_east_mps"] == pytest.approx(
        [300.0, 0.0, 0.0], abs=1.0e-9
    )
    assert "action_index" not in result
    assert "raw_action_index" not in result
    assert "target_command_body_g" not in result


def test_wgs84_local_frame_and_velocity_round_trip() -> None:
    origin = _make_origin(34.0, 108.0)
    north = _position_in_model_frame(
        {"latitude_deg": 35.0, "longitude_deg": 108.0, "altitude_m": 10_000.0},
        origin,
    )
    assert 110_000.0 < north[0] < 112_000.0
    assert abs(north[2]) < 1_000.0
    geodetic = {"latitude_deg": 34.2, "longitude_deg": 108.3, "altitude_m": 9_500.0}
    public_velocity = (900.0, 10.0, -40.0)
    model_velocity = _velocity_in_model_frame(public_velocity, geodetic, origin)
    restored = _velocity_from_model_frame(model_velocity, geodetic, origin)
    assert restored == pytest.approx(public_velocity, abs=1.0e-9)


def test_invalid_frame_does_not_consume_sequence() -> None:
    assert call(reset_request(102))["ok"] is True
    missile = {
        "missile_id": 9001,
        "geodetic_position": {
            "longitude_deg": 108.0,
            "latitude_deg": 34.4,
            "altitude_m": 9_500.0,
        },
        "velocity_north_up_east_mps": [-1_000.0, 0.0, 0.0],
        "alive": True,
    }
    invalid = step_request(102, missiles=[missile, dict(missile)])
    failed = call(invalid)
    assert failed["ok"] is False
    assert failed["error"]["code"] == "BPJ_DUPLICATE_TRACK_ID"
    assert failed["error"]["state_changed"] is False
    assert call(step_request(102))["ok"] is True


def test_out_of_envelope_load_feedback_is_rejected_atomically() -> None:
    assert call(reset_request(104))["ok"] is True
    invalid = step_request(104)
    invalid["ownship"]["actual_load_command_body_g"] = [11.0, 1.0, 0.0]
    failed = call(invalid)
    assert failed["ok"] is False
    assert failed["error"]["code"] == "BPJ_OUT_OF_RANGE"
    assert failed["error"]["state_changed"] is False
    assert call(step_request(104))["ok"] is True


def test_exact_mode_requires_energy_and_guidance() -> None:
    assert call(reset_request(103, input_mode="exact"))["ok"] is True
    missile = {
        "missile_id": 9002,
        "geodetic_position": {
            "longitude_deg": 108.0,
            "latitude_deg": 34.4,
            "altitude_m": 9_500.0,
        },
        "velocity_north_up_east_mps": [-1_000.0, 0.0, 0.0],
        "alive": True,
    }
    response = call(step_request(103, missiles=[missile]))
    assert response["ok"] is False
    assert response["error"]["code"] == "BPJ_MISSING_FIELD"


def test_duplicate_json_key_is_rejected() -> None:
    raw = '{"protocol_version":1,"request_id":1,"op":"info","op":"step"}'
    response = json.loads(entry.dispatch_json(raw))
    assert response["ok"] is False
    assert response["error"]["code"] == "BPJ_INVALID_JSON"


def test_multiple_aircraft_are_state_isolated_and_concurrent_safe() -> None:
    aircraft = list(range(200, 208))
    for index in aircraft:
        assert call(reset_request(index))["ok"] is True
    with ThreadPoolExecutor(max_workers=len(aircraft)) as executor:
        responses = list(executor.map(lambda index: call(step_request(index)), aircraft))
    assert all(response["ok"] is True for response in responses)
    assert {response["result"]["aircraft_id"] for response in responses} == set(aircraft)
    for index in aircraft:
        assert call(
            {
                "protocol_version": 1,
                "request_id": f"close-{index}",
                "op": "close_session",
                "session_id": f"blue-{index}",
                "aircraft_id": index,
            }
        )["ok"] is True


def test_public_requests_and_responses_match_schemas() -> None:
    request_schema = json.loads(
        (SCHEMA_DIR / "blue_policy_request_v1.schema.json").read_text(encoding="utf-8")
    )
    response_schema = json.loads(
        (SCHEMA_DIR / "blue_policy_response_v1.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.Draft202012Validator.check_schema(request_schema)
    jsonschema.Draft202012Validator.check_schema(response_schema)
    request_validator = jsonschema.Draft202012Validator(request_schema)
    response_validator = jsonschema.Draft202012Validator(response_schema)

    reset = reset_request(500)
    request_validator.validate(reset)
    reset_response = call(reset)
    response_validator.validate(reset_response)

    step = step_request(500)
    request_validator.validate(step)
    step_response = call(step)
    response_validator.validate(step_response)
    assert "action_index" not in step_response["result"]

    info = {"protocol_version": 1, "request_id": "schema-info", "op": "info"}
    request_validator.validate(info)
    response_validator.validate(call(info))

    close = {
        "protocol_version": 1,
        "request_id": "schema-close",
        "op": "close_session",
        "session_id": "blue-500",
        "aircraft_id": 500,
    }
    request_validator.validate(close)
    response_validator.validate(call(close))


def test_documented_json_examples_match_the_public_schemas() -> None:
    request_schema = json.loads(
        (SCHEMA_DIR / "blue_policy_request_v1.schema.json").read_text(encoding="utf-8")
    )
    response_schema = json.loads(
        (SCHEMA_DIR / "blue_policy_response_v1.schema.json").read_text(encoding="utf-8")
    )
    request_validator = jsonschema.Draft202012Validator(request_schema)
    response_validator = jsonschema.Draft202012Validator(response_schema)

    example_count = 0
    for document in (DESIGN_DOCUMENT, DELIVERY_README):
        text = document.read_text(encoding="utf-8")
        blocks = re.findall(r"```json\s*\n(.*?)\n```", text, flags=re.DOTALL)
        assert blocks, f"no JSON examples found in {document}"
        for index, block in enumerate(blocks):
            value = json.loads(block)
            validator = response_validator if "ok" in value else request_validator
            errors = sorted(validator.iter_errors(value), key=lambda error: list(error.path))
            assert not errors, (
                f"invalid JSON example {index + 1} in {document}: "
                + "; ".join(error.message for error in errors)
            )
            example_count += 1
    assert example_count >= 10
