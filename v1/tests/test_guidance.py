from __future__ import annotations

import math

import numpy as np
import pytest

from red_swarm_policy.env.guidance import GUIDANCE_CONTRACT, ProportionalNavigationGuidance
from red_swarm_policy.env.math_utils import G0
from red_swarm_policy.env.physics import ThreeDoFPhysicsLayer
from red_swarm_policy.env.types import (
    EnvironmentConfig,
    MissileConfig,
    ScenarioConfig,
    ThreeDoFState,
    los_kinematics,
)


def _state(
    position_m: tuple[float, float, float] | np.ndarray,
    velocity_mps: tuple[float, float, float] | np.ndarray,
    *,
    mass_kg: float = 1.0,
    alive: bool = True,
) -> ThreeDoFState:
    return ThreeDoFState(
        position_m=np.asarray(position_m, dtype=np.float64),
        velocity_mps=np.asarray(velocity_mps, dtype=np.float64),
        mass_kg=mass_kg,
        alive=alive,
    )


def _pure_pn_oracle(
    gain: float,
    missile_velocity_mps: np.ndarray,
    relative_position_m: np.ndarray,
    relative_velocity_mps: np.ndarray,
) -> np.ndarray:
    speed = float(np.linalg.norm(missile_velocity_mps))
    velocity_unit = missile_velocity_mps / speed
    los_angular_velocity = np.cross(
        relative_position_m,
        relative_velocity_mps,
    ) / float(np.dot(relative_position_m, relative_position_m))
    return gain * speed * np.cross(los_angular_velocity, velocity_unit)


def test_pure_pn_matches_independent_three_dimensional_oracle() -> None:
    config = MissileConfig()
    missile = _state((300.0, 9100.0, -500.0), (930.0, 120.0, -75.0))
    target = _state((7800.0, 11200.0, -1800.0), (-140.0, 260.0, 90.0))

    actual = ProportionalNavigationGuidance(config).command(missile, target)
    expected = _pure_pn_oracle(
        config.proportional_navigation_gain,
        missile.velocity_mps,
        target.position_m - missile.position_m,
        target.velocity_mps - missile.velocity_mps,
    )

    np.testing.assert_allclose(actual, expected, rtol=1.0e-14, atol=1.0e-12)
    assert float(np.dot(actual, missile.velocity_mps)) == pytest.approx(0.0, abs=1.0e-10)
    assert ProportionalNavigationGuidance.contract == GUIDANCE_CONTRACT
    assert GUIDANCE_CONTRACT == "pure_proportional_navigation_vm_v1"


@pytest.mark.parametrize("look_angle_deg", (35.0, 60.0))
def test_pure_pn_has_no_projected_tpn_cosine_attenuation(
    look_angle_deg: float,
) -> None:
    config = EnvironmentConfig(scenario=ScenarioConfig(red_count=1, blue_count=1))
    angle = math.radians(look_angle_deg)
    missile = _state(
        (0.0, 9000.0, 0.0),
        (1000.0, 0.0, 0.0),
        mass_kg=config.missile.dry_mass_kg,
    )
    missile.age_s = config.missile.boost_duration_s
    missile.current_target_index = 0
    missile.seeker_locked = True
    missile.guidance_mode = "locked"
    missile.target_estimate_valid = True
    missile.target_estimate_target_index = 0
    target = _state(
        (10000.0 * math.cos(angle), 9000.0, 10000.0 * math.sin(angle)),
        (300.0, 0.0, 0.0),
    )
    kinematics = los_kinematics(missile, target)
    expected_acceleration = (
        config.missile.proportional_navigation_gain
        * np.linalg.norm(missile.velocity_mps)
        * np.linalg.norm(np.cross(kinematics.los_unit, kinematics.los_rate_radps))
    )

    stepped = ThreeDoFPhysicsLayer(config)._step_missile(
        missile,
        target,
        np.zeros(2, dtype=np.float64),
        target_index=0,
    )

    assert stepped.guidance_mode == "locked"
    assert stepped.pn_load_body_g[0] == 0.0
    assert stepped.pn_load_body_g[1] == pytest.approx(0.0, abs=1.0e-12)
    assert stepped.pn_load_body_g[2] == pytest.approx(expected_acceleration / G0)


def test_pure_pn_continues_guiding_while_range_is_opening() -> None:
    config = MissileConfig()
    missile = _state((0.0, 0.0, 0.0), (100.0, 0.0, 0.0))
    target = _state((1000.0, 0.0, 100.0), (200.0, 0.0, 200.0))
    kinematics = los_kinematics(missile, target)

    assert kinematics.closing_speed_mps < 0.0
    command = ProportionalNavigationGuidance(config).command(missile, target)
    assert np.linalg.norm(command) > 0.0
    np.testing.assert_allclose(
        command,
        _pure_pn_oracle(
            config.proportional_navigation_gain,
            missile.velocity_mps,
            kinematics.relative_position_m,
            kinematics.relative_velocity_mps,
        ),
    )


def test_optimized_physics_helpers_treat_input_states_as_read_only() -> None:
    config = EnvironmentConfig(scenario=ScenarioConfig(red_count=1, blue_count=1))
    physics = ThreeDoFPhysicsLayer(config)
    missile = _state(
        (0.0, 9000.0, 0.0),
        (1000.0, 0.0, 0.0),
        mass_kg=config.missile.dry_mass_kg,
    )
    missile.age_s = config.missile.boost_duration_s
    missile.current_target_index = 0
    missile.seeker_locked = True
    missile.guidance_mode = "locked"
    target = _state((20_000.0, 10_000.0, 500.0), (300.0, 0.0, 0.0))
    missile_before = missile.copy()
    target_before = target.copy()

    stepped_missile = physics._step_missile(
        missile, target, np.zeros(2, dtype=np.float64), target_index=0
    )
    stepped_aircraft = physics._step_aircraft(
        target, np.array([0.0, 1.0, 0.0], dtype=np.float64)
    )

    np.testing.assert_array_equal(missile.position_m, missile_before.position_m)
    np.testing.assert_array_equal(missile.velocity_mps, missile_before.velocity_mps)
    assert missile.age_s == missile_before.age_s
    assert missile.guidance_mode == missile_before.guidance_mode
    np.testing.assert_array_equal(target.position_m, target_before.position_m)
    np.testing.assert_array_equal(target.velocity_mps, target_before.velocity_mps)
    assert stepped_missile is not missile
    assert stepped_aircraft is not target


def test_pure_pn_is_rotation_covariant() -> None:
    config = MissileConfig()
    guidance = ProportionalNavigationGuidance(config)
    missile = _state((100.0, -40.0, 25.0), (700.0, 130.0, -80.0))
    target = _state((5200.0, 1800.0, -900.0), (-50.0, 310.0, 140.0))
    angle = math.radians(67.0)
    rotation = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0.0],
            [math.sin(angle), math.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    original = guidance.command(missile, target)
    rotated = guidance.command(
        _state(rotation @ missile.position_m, rotation @ missile.velocity_mps),
        _state(rotation @ target.position_m, rotation @ target.velocity_mps),
    )

    np.testing.assert_allclose(rotated, rotation @ original, rtol=1.0e-14, atol=1.0e-11)


def test_pure_pn_zero_and_inactive_boundaries() -> None:
    config = MissileConfig()
    guidance = ProportionalNavigationGuidance(config)
    missile = _state((0.0, 0.0, 0.0), (500.0, 0.0, 0.0))
    constant_bearing = _state((10000.0, 0.0, 0.0), (200.0, 0.0, 0.0))

    np.testing.assert_array_equal(guidance.command(missile, constant_bearing), np.zeros(3))
    np.testing.assert_array_equal(guidance.command(missile, None), np.zeros(3))
    np.testing.assert_array_equal(
        guidance.command(missile, _state((1.0, 0.0, 0.0), (0.0, 0.0, 0.0), alive=False)),
        np.zeros(3),
    )
    np.testing.assert_array_equal(
        guidance.command(
            _state((0.0, 0.0, 0.0), (500.0, 0.0, 0.0), alive=False),
            constant_bearing,
        ),
        np.zeros(3),
    )
    np.testing.assert_array_equal(
        guidance.command_from_kinematics(
            missile_velocity_mps=np.zeros(3),
            los_unit=np.array([1.0, 0.0, 0.0]),
            los_rate_radps=np.array([0.0, 0.1, 0.0]),
        ),
        np.zeros(3),
    )


@pytest.mark.parametrize(
    ("name", "value", "message"),
    (
        ("missile_velocity_mps", np.zeros(2), "shape"),
        ("los_unit", np.array([1.0, np.nan, 0.0]), "finite"),
        ("los_rate_radps", np.array([0.0, np.inf, 0.0]), "finite"),
    ),
)
def test_pure_pn_rejects_invalid_kinematics(
    name: str,
    value: np.ndarray,
    message: str,
) -> None:
    arguments = {
        "missile_velocity_mps": np.array([500.0, 0.0, 0.0]),
        "los_unit": np.array([1.0, 0.0, 0.0]),
        "los_rate_radps": np.array([0.0, 0.1, 0.0]),
    }
    arguments[name] = value
    with pytest.raises(ValueError, match=message):
        ProportionalNavigationGuidance(MissileConfig()).command_from_kinematics(
            **arguments
        )
