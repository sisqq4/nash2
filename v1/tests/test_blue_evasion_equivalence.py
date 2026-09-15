from __future__ import annotations

import numpy as np
import pytest

from red_swarm_policy import (
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
from red_swarm_policy.env.blue_evasion_kernel import propagate_aircraft_candidates


def _vehicle(
    position_m: np.ndarray | tuple[float, float, float],
    velocity_mps: np.ndarray | tuple[float, float, float],
    mass_kg: float = 1.0,
) -> ThreeDoFState:
    return ThreeDoFState(
        position_m=np.asarray(position_m, dtype=np.float64),
        velocity_mps=np.asarray(velocity_mps, dtype=np.float64),
        mass_kg=mass_kg,
    )


def _config(red_count: int = 12, blue_count: int = 4) -> EnvironmentConfig:
    return EnvironmentConfig(
        scenario=ScenarioConfig(red_count=red_count, blue_count=blue_count),
        sensor=SensorConfig(detection_range_m=60000.0),
    )


@pytest.mark.parametrize(
    ("position_m", "velocity_mps"),
    [
        ((0.0, 10000.0, 0.0), (350.0, 0.0, 0.0)),
        ((0.0, 8000.01, 0.0), (100.01, -30.0, 0.0)),
        ((0.0, 11999.99, 0.0), (599.99, 30.0, 0.0)),
        ((0.0, 10000.0, 0.0), (1.0e-12, 0.0, 0.0)),
        ((0.0, 10000.0, 0.0), (0.0, 350.0, 0.0)),
        ((0.0, 10000.0, 0.0), (0.0, -350.0, 1.0e-12)),
    ],
)
def test_vectorized_candidate_propagation_matches_reference(
    position_m: tuple[float, float, float],
    velocity_mps: tuple[float, float, float],
) -> None:
    config = _config(red_count=1, blue_count=1)
    rule = BlueEvasionRuleMachine(config, execution_backend="reference")
    blue = _vehicle(position_m, velocity_mps)
    aircraft = config.aircraft

    positions, velocities = propagate_aircraft_candidates(
        blue.position_m,
        blue.velocity_mps,
        axial_acceleration_mps2=rule._axial_acceleration_mps2,
        normal_acceleration_mps2=rule._normal_acceleration_mps2,
        cos_bank=rule._cos_bank,
        sin_bank=rule._sin_bank,
        time_step_s=config.time_step_s,
        decision_steps=rule.decision_steps,
        min_speed_mps=aircraft.min_speed_mps,
        max_speed_mps=aircraft.max_speed_mps,
        min_altitude_m=aircraft.min_altitude_m,
        max_altitude_m=aircraft.max_altitude_m,
    )

    for action_index, command in enumerate(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G):
        expected = rule._propagate_blue(blue, command)
        np.testing.assert_allclose(
            positions[action_index],
            expected.position_m,
            rtol=1.0e-13,
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            velocities[action_index],
            expected.velocity_mps,
            rtol=1.0e-13,
            atol=1.0e-11,
        )


def _random_threat_state(
    rng: np.random.Generator,
    config: EnvironmentConfig,
) -> EngagementState:
    blue: list[ThreeDoFState] = []
    for _ in range(config.scenario.blue_count):
        direction = rng.normal(size=3)
        direction[1] *= 0.15
        direction /= np.linalg.norm(direction)
        blue.append(
            _vehicle(
                (
                    rng.uniform(-5000.0, 5000.0),
                    rng.uniform(8100.0, 11900.0),
                    rng.uniform(-5000.0, 5000.0),
                ),
                direction * rng.uniform(150.0, 550.0),
            )
        )

    red: list[ThreeDoFState] = []
    for red_index in range(config.scenario.red_count):
        target_index = red_index % config.scenario.blue_count
        target = blue[target_index]
        line = rng.normal(size=3)
        line /= np.linalg.norm(line)
        missile = _vehicle(
            target.position_m + line * rng.uniform(5000.0, 59000.0),
            target.velocity_mps - line * rng.uniform(700.0, 1800.0),
            config.missile.dry_mass_kg,
        )
        missile.current_target_index = target_index
        missile.seeker_locked = red_index % 3 == 0
        missile.age_s = config.missile.boost_duration_s
        red.append(missile)
    return EngagementState(red=red, blue=blue)


def test_vectorized_scores_and_decisions_match_reference_on_random_threats() -> None:
    config = _config()
    rng = np.random.default_rng(20260904)
    for _ in range(25):
        state = _random_threat_state(rng, config)
        reference = BlueEvasionRuleMachine(config, execution_backend="reference")
        vectorized = BlueEvasionRuleMachine(
            config,
            execution_backend="vectorized_guarded",
        )
        previous = rng.integers(
            0,
            len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G),
            size=config.scenario.blue_count,
            dtype=np.int64,
        )
        reference._previous_indices = previous.copy()
        vectorized._previous_indices = previous.copy()

        for blue_index in range(config.scenario.blue_count):
            threats = reference._assess_threats(state, blue_index)
            assert threats
            expected_scores = reference._score_all_actions_reference(
                state,
                blue_index,
                threats,
                int(previous[blue_index]),
            )
            actual_scores = vectorized._score_all_actions_vectorized(
                state,
                blue_index,
                threats,
                int(previous[blue_index]),
            )
            np.testing.assert_allclose(
                actual_scores,
                expected_scores,
                rtol=1.0e-12,
                atol=1.0e-12,
            )

        expected = reference.decide(state)
        actual = vectorized.decide(state)
        np.testing.assert_array_equal(actual.action_indices, expected.action_indices)
        assert actual.modes == expected.modes
        np.testing.assert_array_equal(
            actual.primary_threat_indices,
            expected.primary_threat_indices,
        )
        np.testing.assert_allclose(
            actual.selected_scores,
            expected.selected_scores,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def _symmetric_tie_state(config: EnvironmentConfig) -> EngagementState:
    blue = _vehicle((0.0, 10000.0, 0.0), (350.0, 0.0, 0.0))
    red = _vehicle(
        (-20000.0, 10000.0, 0.0),
        (1500.0, 0.0, 0.0),
        config.missile.dry_mass_kg,
    )
    red.current_target_index = 0
    red.seeker_locked = True
    red.age_s = config.missile.boost_duration_s
    return EngagementState(red=[red], blue=[blue])


def test_guarded_backend_preserves_reference_tie_break() -> None:
    config = _config(red_count=1, blue_count=1)
    state = _symmetric_tie_state(config)
    reference = BlueEvasionRuleMachine(config, execution_backend="reference").decide(state)
    vectorized_rule = BlueEvasionRuleMachine(
        config,
        execution_backend="vectorized_guarded",
    )
    vectorized = vectorized_rule.decide(state)

    assert reference.action_indices.tolist() == [7]
    np.testing.assert_array_equal(vectorized.action_indices, reference.action_indices)
    assert vectorized_rule.runtime_statistics()["near_tie_fallback_count"] == 1


def test_shadow_applies_reference_action_and_reports_no_mismatch() -> None:
    config = _config()
    state = _random_threat_state(np.random.default_rng(17), config)
    expected = BlueEvasionRuleMachine(config, execution_backend="reference").decide(state)
    shadow_rule = BlueEvasionRuleMachine(config, execution_backend="shadow")
    actual = shadow_rule.decide(state)
    statistics = shadow_rule.runtime_statistics()

    np.testing.assert_array_equal(actual.action_indices, expected.action_indices)
    np.testing.assert_allclose(actual.selected_scores, expected.selected_scores, rtol=0.0, atol=0.0)
    assert statistics["execution_backend"] == "shadow"
    assert statistics["shadow_action_mismatch_count"] == 0
    assert statistics["max_score_abs_error"] < 1.0e-12


def test_reference_and_vectorized_backends_keep_environment_in_lockstep() -> None:
    config = EnvironmentConfig(
        max_steps=1461,
        policy_start_mode="post_boost",
        scenario=ScenarioConfig(red_count=4, blue_count=2),
        sensor=SensorConfig(detection_range_m=60000.0),
    )
    reference_env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    vectorized_env = RedBlueEngagementEnv(config, device="cpu", record_replay=False)
    reference_env.reset(seed=20260904, start_mode="post_boost")
    vectorized_env.reset(seed=20260904, start_mode="post_boost")
    reference_rule = BlueEvasionRuleMachine(config, execution_backend="reference")
    vectorized_rule = BlueEvasionRuleMachine(
        config,
        execution_backend="vectorized_guarded",
    )
    reference_controller = BlueEvasionController(reference_rule)
    vectorized_controller = BlueEvasionController(vectorized_rule)
    for environment in (reference_env, vectorized_env):
        assert environment.state is not None
        for red_index, missile in enumerate(environment.state.red):
            target_index = red_index % len(environment.state.blue)
            target = environment.state.blue[target_index]
            line = np.array(
                [1.0, 0.03 * (red_index - 1.5), 0.1 * (-1.0) ** red_index],
                dtype=np.float64,
            )
            line /= np.linalg.norm(line)
            missile.position_m = target.position_m + line * (18000.0 + 500.0 * red_index)
            missile.velocity_mps = target.velocity_mps - line * (1300.0 + 20.0 * red_index)
            missile.current_target_index = target_index
            missile.seeker_locked = True
            missile.age_s = config.missile.boost_duration_s

    for _ in range(60):
        assert reference_env.state is not None
        assert vectorized_env.state is not None
        reference_action, reference_decision = reference_controller.action_for(
            reference_env.state
        )
        vectorized_action, vectorized_decision = vectorized_controller.action_for(
            vectorized_env.state
        )
        assert (reference_decision is None) == (vectorized_decision is None)
        np.testing.assert_array_equal(
            vectorized_action["action_indices"],
            reference_action["action_indices"],
        )
        reference_step = reference_env.step(blue_action=reference_action)
        vectorized_step = vectorized_env.step(blue_action=vectorized_action)
        assert vectorized_step.done == reference_step.done
        for actual, expected in zip(
            vectorized_env.state.blue + vectorized_env.state.red,
            reference_env.state.blue + reference_env.state.red,
        ):
            np.testing.assert_allclose(actual.position_m, expected.position_m, rtol=0.0, atol=0.0)
            np.testing.assert_allclose(actual.velocity_mps, expected.velocity_mps, rtol=0.0, atol=0.0)
            assert actual.alive == expected.alive


def test_invalid_execution_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="execution_backend"):
        BlueEvasionRuleMachine(_config(), execution_backend="fast")  # type: ignore[arg-type]

