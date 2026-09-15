from __future__ import annotations

import numpy as np

from .math_utils import EPS, G0, UP_AXIS


def _dot3_batch(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return row-wise 3-vector dot products with an explicit operation order."""

    return (
        left[:, 0] * right[:, 0]
        + left[:, 1] * right[:, 1]
        + left[:, 2] * right[:, 2]
    )


def _norm3_batch(values: np.ndarray) -> np.ndarray:
    return np.sqrt(_dot3_batch(values, values))


def _unit3_batch(values: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    lengths = _norm3_batch(values)
    result = np.zeros_like(values, dtype=np.float64)
    valid = lengths > EPS
    result[valid] = values[valid] / lengths[valid, None]
    if fallback is not None and np.any(~valid):
        fallback_values = np.asarray(fallback, dtype=np.float64)
        if fallback_values.shape == (3,):
            result[~valid] = fallback_values
        else:
            result[~valid] = fallback_values[~valid]
    return result


def velocity_local_frame_batch(
    velocity_mps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized equivalent of ``velocity_local_frame`` for shape ``[A, 3]``."""

    velocity = np.asarray(velocity_mps, dtype=np.float64)
    if velocity.ndim != 2 or velocity.shape[1] != 3:
        raise ValueError(f"velocity_mps shape {velocity.shape} must be (A, 3)")

    forward = _unit3_batch(
        velocity,
        np.array([1.0, 0.0, 0.0], dtype=np.float64),
    )
    inertial_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up_projection = forward[:, UP_AXIS]
    local_up_raw = inertial_up[None, :] - up_projection[:, None] * forward
    local_up = _unit3_batch(local_up_raw)

    vertical = _norm3_batch(local_up) <= EPS
    if np.any(vertical):
        east = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        vertical_forward = forward[vertical]
        east_projection = vertical_forward[:, 2]
        fallback_raw = east[None, :] - east_projection[:, None] * vertical_forward
        local_up[vertical] = _unit3_batch(
            fallback_raw,
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
        )

    local_right = _unit3_batch(
        np.cross(forward, local_up),
        np.array([0.0, 0.0, 1.0], dtype=np.float64),
    )
    return forward, local_up, local_right


def propagate_aircraft_candidates(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    axial_acceleration_mps2: np.ndarray,
    normal_acceleration_mps2: np.ndarray,
    cos_bank: np.ndarray,
    sin_bank: np.ndarray,
    time_step_s: float,
    decision_steps: int,
    min_speed_mps: float,
    max_speed_mps: float,
    min_altitude_m: float,
    max_altitude_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate all action candidates while preserving the sequential steps."""

    axial = np.asarray(axial_acceleration_mps2, dtype=np.float64)
    normal = np.asarray(normal_acceleration_mps2, dtype=np.float64)
    bank_cos = np.asarray(cos_bank, dtype=np.float64)
    bank_sin = np.asarray(sin_bank, dtype=np.float64)
    action_count = axial.shape[0]
    expected = (action_count,)
    if normal.shape != expected or bank_cos.shape != expected or bank_sin.shape != expected:
        raise ValueError("candidate action constants must have the same one-dimensional shape")
    if decision_steps <= 0:
        raise ValueError("decision_steps must be positive")

    initial_position = np.asarray(position_m, dtype=np.float64)
    initial_velocity = np.asarray(velocity_mps, dtype=np.float64)
    if initial_position.shape != (3,) or initial_velocity.shape != (3,):
        raise ValueError("position_m and velocity_mps must have shape (3,)")
    position = np.repeat(initial_position[None, :], action_count, axis=0)
    velocity = np.repeat(initial_velocity[None, :], action_count, axis=0)
    half_dt_squared = 0.5 * time_step_s * time_step_s

    for _ in range(decision_steps):
        forward, local_up, local_right = velocity_local_frame_batch(velocity)
        normal_direction = bank_cos[:, None] * local_up - bank_sin[:, None] * local_right
        acceleration = axial[:, None] * forward
        acceleration += normal[:, None] * normal_direction
        acceleration[:, UP_AXIS] += -G0

        position = position + velocity * time_step_s + acceleration * half_dt_squared
        raw_velocity = velocity + acceleration * time_step_s
        raw_speed = _norm3_batch(raw_velocity)
        speed = np.clip(raw_speed, min_speed_mps, max_speed_mps)
        velocity = _unit3_batch(raw_velocity, forward) * speed[:, None]

        below = position[:, UP_AXIS] < min_altitude_m
        if np.any(below):
            position[below, UP_AXIS] = min_altitude_m
            velocity[below, UP_AXIS] = np.maximum(0.0, velocity[below, UP_AXIS])
        above = position[:, UP_AXIS] > max_altitude_m
        if np.any(above):
            position[above, UP_AXIS] = max_altitude_m
            velocity[above, UP_AXIS] = np.minimum(0.0, velocity[above, UP_AXIS])

    return position, velocity


def boundary_penalty_batch(
    position_m: np.ndarray,
    velocity_mps: np.ndarray,
    *,
    min_altitude_m: float,
    max_altitude_m: float,
    altitude_margin_m: float,
    altitude_prediction_s: float,
    min_speed_mps: float,
    max_speed_mps: float,
    speed_margin_mps: float,
) -> np.ndarray:
    altitude_forecast = position_m[:, UP_AXIS] + velocity_mps[:, UP_AXIS] * altitude_prediction_s
    lower_guard = min_altitude_m + altitude_margin_m
    upper_guard = max_altitude_m - altitude_margin_m
    altitude_scale = max(altitude_margin_m, 1.0)
    altitude_penalty = (
        np.maximum(0.0, lower_guard - altitude_forecast)
        + np.maximum(0.0, altitude_forecast - upper_guard)
    ) / altitude_scale

    speed = _norm3_batch(velocity_mps)
    lower_speed = min_speed_mps + speed_margin_mps
    upper_speed = max_speed_mps - speed_margin_mps
    speed_scale = max(speed_margin_mps, 1.0)
    speed_penalty = (
        np.maximum(0.0, lower_speed - speed)
        + np.maximum(0.0, speed - upper_speed)
    ) / speed_scale
    return 4.0 * altitude_penalty + 2.0 * speed_penalty


def score_aircraft_candidates(
    *,
    original_velocity_mps: np.ndarray,
    predicted_position_m: np.ndarray,
    predicted_velocity_mps: np.ndarray,
    red_position_m: np.ndarray,
    red_velocity_mps: np.ndarray,
    threat_severity: np.ndarray,
    action_effort: np.ndarray,
    previous_index: int,
    decision_interval_s: float,
    lookahead_s: float,
    critical_range_m: float,
    detection_range_m: float,
    max_load_factor_g: float,
    effort_penalty: float,
    switch_penalty: float,
    min_altitude_m: float,
    max_altitude_m: float,
    altitude_margin_m: float,
    altitude_prediction_s: float,
    min_speed_mps: float,
    max_speed_mps: float,
    speed_margin_mps: float,
) -> np.ndarray:
    """Score all candidates, retaining the reference threat accumulation order."""

    predicted_position = np.asarray(predicted_position_m, dtype=np.float64)
    predicted_velocity = np.asarray(predicted_velocity_mps, dtype=np.float64)
    red_position = np.asarray(red_position_m, dtype=np.float64)
    red_velocity = np.asarray(red_velocity_mps, dtype=np.float64)
    severity = np.asarray(threat_severity, dtype=np.float64)
    effort = np.asarray(action_effort, dtype=np.float64)
    action_count = predicted_position.shape[0]
    if predicted_position.shape != (action_count, 3):
        raise ValueError("predicted_position_m must have shape (A, 3)")
    if predicted_velocity.shape != (action_count, 3) or effort.shape != (action_count,):
        raise ValueError("predicted velocity and action effort shapes do not match candidates")
    threat_count = severity.shape[0]
    if red_position.shape != (threat_count, 3) or red_velocity.shape != (threat_count, 3):
        raise ValueError("red state arrays must have shape (T, 3)")

    original_velocity = np.asarray(original_velocity_mps, dtype=np.float64)
    acceleration = (predicted_velocity - original_velocity[None, :]) / decision_interval_s
    max_acceleration = max_load_factor_g * G0
    acceleration_scale = max(max_acceleration, EPS)
    threat_weight = sum(float(value) for value in severity)
    evasion_utility = np.zeros(action_count, dtype=np.float64)

    for threat_index in range(threat_count):
        predicted_red_position = red_position[threat_index] + red_velocity[threat_index] * decision_interval_s
        relative_position = predicted_red_position[None, :] - predicted_position
        distance = np.maximum(_norm3_batch(relative_position), EPS)
        los_unit = relative_position / distance[:, None]
        relative_velocity = red_velocity[threat_index][None, :] - predicted_velocity
        relative_speed_squared = _dot3_batch(relative_velocity, relative_velocity)
        time_to_closest = np.zeros(action_count, dtype=np.float64)
        moving = relative_speed_squared > EPS
        if np.any(moving):
            numerator = -_dot3_batch(relative_position[moving], relative_velocity[moving])
            time_to_closest[moving] = np.clip(
                numerator / relative_speed_squared[moving], 0.0, lookahead_s,
            )
        closest_vector = relative_position + relative_velocity * time_to_closest[:, None]
        predicted_miss = _norm3_batch(closest_vector)
        end_range = _norm3_batch(relative_position + relative_velocity * lookahead_s)
        relative_los_speed = _dot3_batch(relative_velocity, los_unit)
        transverse_velocity = relative_velocity - relative_los_speed[:, None] * los_unit
        acceleration_los = _dot3_batch(acceleration, los_unit)
        transverse_acceleration = acceleration - acceleration_los[:, None] * los_unit
        lateral_score = _norm3_batch(transverse_acceleration) / acceleration_scale
        away_score = _dot3_batch(acceleration, -los_unit) / acceleration_scale
        miss_score = np.clip(predicted_miss / critical_range_m, 0.0, 2.0)
        range_score = np.clip(end_range / detection_range_m, 0.0, 2.0)
        transverse_score = np.clip(_norm3_batch(transverse_velocity) / 600.0, 0.0, 2.0)
        local_utility = (
            0.35 * miss_score
            + 0.15 * range_score
            + 0.40 * lateral_score
            + 0.20 * away_score
            + 0.10 * transverse_score
        )
        evasion_utility += float(severity[threat_index]) * local_utility

    evasion_utility /= max(threat_weight, EPS)
    urgency = float(np.clip(max(float(value) for value in severity), 0.0, 1.0))
    vertical_acceleration = np.abs(acceleration[:, UP_AXIS]) / acceleration_scale
    boundary_penalty = boundary_penalty_batch(
        predicted_position,
        predicted_velocity,
        min_altitude_m=min_altitude_m,
        max_altitude_m=max_altitude_m,
        altitude_margin_m=altitude_margin_m,
        altitude_prediction_s=altitude_prediction_s,
        min_speed_mps=min_speed_mps,
        max_speed_mps=max_speed_mps,
        speed_margin_mps=speed_margin_mps,
    )
    change_penalty = np.full(action_count, switch_penalty, dtype=np.float64)
    if 0 <= previous_index < action_count:
        change_penalty[previous_index] = 0.0
    return (
        urgency * evasion_utility
        - effort_penalty * effort
        - 0.20 * vertical_acceleration
        - boundary_penalty
        - change_penalty
    )
