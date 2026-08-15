
"""Basic kinematics for blue aircraft and red missiles."""

from typing import Tuple
import numpy as np


def update_blue_state(
    pos: np.ndarray,
    vel: np.ndarray,
    action: np.ndarray,
    dt: float,
    accel_mag: float,
    v_max: float,
    v_min: float,
    max_sustained_pitch: float | None = None,
    roll_state: float | None = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Update blue aircraft state with a roll/pitch/yaw-based maneuver model.

    Actions:
        action: [nx, nz, roll, pitch] from the blue action library.
        - nx: tangential load factor (forward acceleration, in g)
        - nz: normal load factor (lift, in g)
        - roll: target bank angle (rad)
        - pitch: 0 = pull back to level flight, -1 = keep current pitch
    """
    pos = pos.astype(float)
    vel = vel.astype(float)

    action = np.asarray(action, dtype=float).reshape(-1)
    if action.shape[0] != 4:
        raise ValueError("blue action must be a 4D vector [nx, nz, roll, pitch].")

    nx, nz, target_roll, pitch_cmd = action

    speed = np.linalg.norm(vel)
    if speed < 1e-6:
        yaw = 0.0
        pitch = 0.0
    else:
        yaw = float(np.arctan2(vel[1], vel[0]))
        pitch = float(np.arcsin(np.clip(vel[2] / speed, -1.0, 1.0)))

    current_roll = float(roll_state) if roll_state is not None else float(target_roll)
    roll_rate = np.deg2rad(180.0)
    roll_diff = target_roll - current_roll
    roll_diff = np.arctan2(np.sin(roll_diff), np.cos(roll_diff))
    roll_step = np.clip(roll_diff, -roll_rate * dt, roll_rate * dt)
    current_roll += roll_step

    k_induced = 0.04
    drag_penalty = k_induced * (nz ** 2 - 1.0) if nz > 1.0 else 0.0
    dv = accel_mag * (nx - drag_penalty - np.sin(pitch))
    speed = speed + dv * dt
    speed = float(np.clip(speed, v_min, v_max))

    v_safe = max(speed, 1e-6)
    dpitch = (accel_mag / v_safe) * (nz * np.cos(current_roll) - np.cos(pitch))

    # Pitch command: when pitch_cmd == 0, bias acceleration to level the aircraft.
    if pitch_cmd >= 0.0:
        dpitch = dpitch - 0.5 * pitch

    pitch_limit = 1.48
    if max_sustained_pitch is not None:
        pitch_limit = min(pitch_limit, float(max_sustained_pitch))
    new_pitch = pitch + dpitch * dt
    if abs(new_pitch) < pitch_limit:
        pitch = new_pitch

    denom = max(np.cos(pitch), 1e-3)
    dyaw = (accel_mag * nz * np.sin(current_roll)) / (v_safe * denom)
    yaw = yaw + dyaw * dt
    yaw = float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    vel = np.array(
        [
            speed * np.cos(pitch) * np.cos(yaw),
            speed * np.cos(pitch) * np.sin(yaw),
            speed * np.sin(pitch),
        ]
    )
    pos = pos + vel * dt
    return pos, vel, current_roll


def update_missiles_pn(
    missile_pos: np.ndarray,
    missile_vel: np.ndarray,
    blue_pos: np.ndarray,
    blue_vel: np.ndarray,
    missile_speed: float | np.ndarray,
    dt: float,
    nav_gain,
    max_overload_g: float | np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Proportional-navigation-like update for a batch of missiles.

    Args:
        missile_pos: (M, 3)
        missile_vel: (M, 3)
        blue_pos: (3,)
        blue_vel: (3,)
        missile_speed: scalar speed (kept constant) [km/s]
        dt: [s]
        nav_gain: scalar or shape (M,) PN navigation constant N (dimensionless)
        max_overload_g: optional max lateral load factor [g]
    """
    g0_km_s2 = 9.80665 / 1000.0
    M = missile_pos.shape[0]
    new_pos = missile_pos.astype(float).copy()
    new_vel = missile_vel.astype(float).copy()
    blue_pos = blue_pos.astype(float).reshape(1, 3)

    nav_array = np.asarray(nav_gain, dtype=float)
    if nav_array.shape == ():
        nav_array = np.full(M, float(nav_array))
    else:
        assert nav_array.shape[0] == M, "nav_gain array must have shape (M,)"

    speed_array = np.asarray(missile_speed, dtype=float)
    if speed_array.shape == ():
        speed_array = np.full(M, float(speed_array))
    else:
        assert speed_array.shape[0] == M, "missile_speed array must have shape (M,)"

    max_g_array = None
    if max_overload_g is not None:
        max_g_array = np.asarray(max_overload_g, dtype=float)
        if max_g_array.shape == ():
            max_g_array = np.full(M, float(max_g_array))
        else:
            assert max_g_array.shape[0] == M, "max_overload_g array must have shape (M,)"

    for i in range(M):
        p = new_pos[i]
        v = new_vel[i]
        N_gain = float(nav_array[i])
        target_speed = float(speed_array[i])
        max_g = None if max_g_array is None else float(max_g_array[i])

        speed = np.linalg.norm(v)
        if speed < 1e-6:
            # If speed is zero, keep it zero (e.g. not yet launched)
            new_pos[i] = p
            new_vel[i] = v
            continue

        u = v / speed

        r = blue_pos[0] - p
        r_norm = np.linalg.norm(r)
        if r_norm < 1e-6:
            new_pos[i] = p
            new_vel[i] = v
            continue

            # Relative kinematics for PN guidance.
        rel_vel = blue_vel - v
        los = r / r_norm
        closing_speed = -float(np.dot(rel_vel, los))
        los_omega = np.cross(r, rel_vel) / max(r_norm ** 2, 1e-9)

        if closing_speed > 0.0 and N_gain != 0.0:
            # Standard 3D PN command: a_n = N * Vc * (omega_LOS x u_m)
            a_cmd = N_gain * closing_speed * np.cross(los_omega, u)
            u_dot = a_cmd / max(speed, 1e-6)
            delta_u = u_dot * dt
            if max_g is not None and max_g > 0.0 and speed > 1e-6:
                omega_max = (max_g * g0_km_s2) / speed
                max_delta = omega_max * dt
                delta_norm = np.linalg.norm(delta_u)
                if delta_norm > max_delta > 0.0:
                    delta_u = delta_u * (max_delta / delta_norm)
            u_new = u + delta_u
            u_norm = np.linalg.norm(u_new)
            if u_norm > 1e-6:
                u = u_new / u_norm

        v = u * target_speed
        p = p + v * dt

        new_pos[i] = p
        new_vel[i] = v

    return new_pos, new_vel
