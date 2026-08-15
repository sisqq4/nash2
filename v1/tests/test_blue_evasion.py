from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from red_swarm_policy import (
    BlueEvasionConfig,
    BlueEvasionController,
    BlueEvasionRuleMachine,
    EngagementState,
    EnvironmentConfig,
    RedBlueEngagementEnv,
    ScenarioConfig,
    SensorConfig,
    ThreeDoFState,
)
from red_swarm_policy.env import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G
from red_swarm_policy.run_blue_evasion import build_parser, run_blue_evasion_episode


def _state(
    position_m: tuple[float, float, float],
    velocity_mps: tuple[float, float, float],
    mass_kg: float,
) -> ThreeDoFState:
    return ThreeDoFState(
        position_m=np.asarray(position_m, dtype=np.float64),
        velocity_mps=np.asarray(velocity_mps, dtype=np.float64),
        mass_kg=mass_kg,
    )


def _rule_config() -> EnvironmentConfig:
    return EnvironmentConfig(
        time_step_s=0.005,
        bias_update_interval_s=0.1,
        assignment_update_interval_s=1.0,
        scenario=ScenarioConfig(red_count=1, blue_count=1),
        sensor=SensorConfig(detection_range_m=60000.0),
    )


def _engagement(config: EnvironmentConfig, red_range_m: float) -> EngagementState:
    blue = _state((0.0, 10000.0, 0.0), (350.0, 0.0, 0.0), 1.0)
    red = _state((-red_range_m, 10000.0, 0.0), (1500.0, 0.0, 0.0), config.missile.dry_mass_kg)
    red.current_target_index = 0
    red.seeker_locked = True
    red.age_s = config.missile.boost_duration_s
    return EngagementState(red=[red], blue=[blue])


def test_blue_detection_gate_is_exactly_60_km() -> None:
    config = _rule_config()
    rule = BlueEvasionRuleMachine(config)

    outside = rule.decide(_engagement(config, 60000.001))
    rule.reset()
    inside = rule.decide(_engagement(config, 59999.999))

    assert BlueEvasionConfig().detection_range_m == 60000.0
    assert outside.action_indices.tolist() == [0]
    assert outside.modes == ("cruise",)
    assert outside.output_record()["primary_threat_ranges_m"] == [None]
    json.dumps(outside.output_record(), allow_nan=False)
    assert inside.primary_threat_indices.tolist() == [0]
    assert inside.modes == ("critical",)


def test_close_locked_threat_selects_a_non_cruise_library_action() -> None:
    config = _rule_config()
    decision = BlueEvasionRuleMachine(config).decide(_engagement(config, 20000.0))

    assert 0 <= int(decision.action_indices[0]) <= 28
    assert int(decision.action_indices[0]) != 0
    assert decision.primary_threat_indices.tolist() == [0]
    assert decision.primary_closing_speeds_mps[0] == pytest.approx(1150.0)
    assert decision.output_record()["blue_action_library_entries"] == [int(decision.action_indices[0]) + 1]


def test_controller_updates_once_per_20_physics_steps() -> None:
    config = _rule_config()
    state = _engagement(config, 60000.001)
    controller = BlueEvasionController(BlueEvasionRuleMachine(config))

    first_action, first_decision = controller.action_for(state)
    assert first_decision is not None
    for step_count in range(1, 20):
        state.step_count = step_count
        held_action, updated = controller.action_for(state)
        assert updated is None
        np.testing.assert_array_equal(held_action["action_indices"], first_action["action_indices"])
    state.step_count = 20
    _, second_decision = controller.action_for(state)

    assert controller.decision_steps == 20
    assert second_decision is not None
    assert second_decision.step_count == 20


def test_selected_index_is_applied_through_the_environment_blue_action_interface() -> None:
    config = EnvironmentConfig(
        max_steps=1421,
        scenario=ScenarioConfig(red_count=1, blue_count=1),
        sensor=SensorConfig(detection_range_m=60000.0),
    )
    environment = RedBlueEngagementEnv(config, device="cpu", record_replay=True)
    environment.reset(seed=20260716, start_mode="post_boost")
    assert environment.state is not None
    state = environment.state
    state.blue[0].position_m = np.array([0.0, 10000.0, 0.0])
    state.blue[0].velocity_mps = np.array([350.0, 0.0, 0.0])
    state.red[0].position_m = np.array([-20000.0, 10000.0, 0.0])
    state.red[0].velocity_mps = np.array([1500.0, 0.0, 0.0])
    state.red[0].current_target_index = 0
    state.red[0].seeker_locked = True

    action, decision = BlueEvasionController(BlueEvasionRuleMachine(config)).action_for(state)
    assert decision is not None
    selected_index = int(decision.action_indices[0])
    environment.step(blue_action=action)

    assert selected_index != 0
    np.testing.assert_allclose(
        environment.replay_layer[-1].action.blue.load_command_body_g[0],
        BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[selected_index],
        rtol=0.0,
        atol=0.0,
    )


def test_environment_runner_outputs_indices_at_0p1_s_and_steps_at_0p005_s() -> None:
    network_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config = EnvironmentConfig(
        time_step_s=0.005,
        bias_update_interval_s=0.1,
        assignment_update_interval_s=1.0,
        max_steps=21,
        policy_start_mode="launch",
        scenario=ScenarioConfig(red_count=2, blue_count=1),
        sensor=SensorConfig(detection_range_m=60000.0),
    )
    environment = RedBlueEngagementEnv(config, device=network_device, record_replay=True)
    controller = BlueEvasionController(BlueEvasionRuleMachine(config))
    records: list[dict[str, object]] = []

    summary = run_blue_evasion_episode(
        environment,
        controller,
        seed=20260716,
        duration_s=0.105,
        start_mode="launch",
        emit=records.append,
    )

    assert summary.physics_steps == 21
    assert summary.decisions == 2
    assert summary.final_time_s == pytest.approx(0.105)
    assert [record["step_count"] for record in records] == [0, 20]
    assert [record["time_s"] for record in records] == pytest.approx([0.0, 0.1])
    assert all(0 <= index <= 28 for record in records for index in record["blue_action_api_indices"])
    assert len(environment.replay_layer) == 0
    assert environment.last_observation is not None
    assert environment.last_observation.assignment_actor_inputs.self_state.device == network_device


def test_blue_evasion_cli_defaults_match_required_timing_and_detection() -> None:
    args = build_parser().parse_args([])

    assert args.time_step_s == 0.005
    assert args.decision_interval_s == 0.1
    assert args.detection_range_m == 60000.0
    assert args.device == "cuda:0"

    with pytest.raises(ValueError, match="integer multiple"):
        BlueEvasionRuleMachine(
            _rule_config(),
            BlueEvasionConfig(decision_interval_s=0.103),
        )
