from __future__ import annotations

import math

import numpy as np

from .guidance import ProportionalNavigationGuidance
from .math_utils import EPS, G0, UP_AXIS, air_density, clip_norm, norm, speed_of_sound, unit, velocity_local_frame
from .seeker import seeker_boresight_angle_deg, seeker_fov_limit_deg
from .types import EngagementState, EnvironmentConfig, JointAction, ThreeDoFState, los_kinematics


class ThreeDoFPhysicsLayer:
    def __init__(self, config: EnvironmentConfig) -> None:
        config.validate()
        self.config = config
        self.guidance = ProportionalNavigationGuidance(config.missile)

    def step(self, state: EngagementState, action: JointAction) -> EngagementState:
        target_indices = np.asarray(action.red.target_indices, dtype=np.int64)
        guidance_bias = np.asarray(action.red.guidance_bias, dtype=np.float64)
        if target_indices.shape != (len(state.red),):
            raise ValueError(f"target_indices shape {target_indices.shape} must be {(len(state.red),)}")
        if guidance_bias.shape != (len(state.red), 2):
            raise ValueError(f"guidance_bias shape {guidance_bias.shape} must be {(len(state.red), 2)}")
        if not np.all(np.isfinite(guidance_bias)):
            raise ValueError("guidance_bias must contain only finite values")
        if np.any((guidance_bias < -1.0) | (guidance_bias > 1.0)):
            raise ValueError("guidance_bias values must be in [-1, 1]")
        next_state = state.copy()
        old_blue = [vehicle.copy() for vehicle in state.blue]
        old_red = [vehicle.copy() for vehicle in state.red]
        for index, blue in enumerate(old_blue):
            if blue.alive:
                next_state.blue[index] = self._step_aircraft(blue, action.blue.load_command_body_g[index])
        for index, missile in enumerate(old_red):
            if not missile.alive:
                next_state.red[index].guidance_bias = np.zeros(2, dtype=np.float64)
                next_state.red[index].pn_load_body_g = np.zeros(3, dtype=np.float64)
                next_state.red[index].bias_load_body_g = np.zeros(3, dtype=np.float64)
                next_state.red[index].gravity_load_body_g = np.zeros(3, dtype=np.float64)
                next_state.red[index].final_load_body_g = np.zeros(3, dtype=np.float64)
                continue
            target_index = int(target_indices[index])
            target = old_blue[target_index] if 0 <= target_index < len(old_blue) else None
            next_state.red[index] = self._step_missile(
                missile,
                target,
                guidance_bias[index],
                target_index=target_index,
            )
        next_state.time_s = round(state.time_s + self.config.time_step_s, 12)
        next_state.step_count = state.step_count + 1
        self._update_ranges(next_state, target_indices)
        return next_state

    def _step_aircraft(self, state: ThreeDoFState, command: np.ndarray) -> ThreeDoFState:
        axial_load_g, normal_load_g, bank_rad = np.asarray(command, dtype=np.float64)
        frame = velocity_local_frame(state.velocity_mps)
        normal_direction = math.cos(bank_rad) * frame[:, 1] - math.sin(bank_rad) * frame[:, 2]
        acceleration = axial_load_g * G0 * frame[:, 0]
        acceleration += normal_load_g * G0 * normal_direction
        acceleration += np.array([0.0, -G0, 0.0], dtype=np.float64)
        next_state = self._integrate(state, acceleration, mass_flow_rate_kg_s=0.0)
        next_state.bank_angle_rad = float(bank_rad)
        speed = float(np.clip(norm(next_state.velocity_mps), self.config.aircraft.min_speed_mps, self.config.aircraft.max_speed_mps))
        next_state.velocity_mps = unit(next_state.velocity_mps, frame[:, 0]) * speed
        if next_state.position_m[UP_AXIS] < self.config.aircraft.min_altitude_m:
            next_state.position_m[UP_AXIS] = self.config.aircraft.min_altitude_m
            next_state.velocity_mps[UP_AXIS] = max(0.0, next_state.velocity_mps[UP_AXIS])
        elif next_state.position_m[UP_AXIS] > self.config.aircraft.max_altitude_m:
            next_state.position_m[UP_AXIS] = self.config.aircraft.max_altitude_m
            next_state.velocity_mps[UP_AXIS] = min(0.0, next_state.velocity_mps[UP_AXIS])
        next_state.energy = float(np.clip(speed / self.config.aircraft.max_speed_mps, 0.0, 1.0))
        return next_state

    def _step_missile(
        self,
        state: ThreeDoFState,
        target: ThreeDoFState | None,
        normalized_bias: np.ndarray,
        *,
        target_index: int | None = None,
    ) -> ThreeDoFState:
        guidance_bias = np.asarray(normalized_bias, dtype=np.float64)
        if guidance_bias.shape != (2,):
            raise ValueError(f"normalized_bias shape {guidance_bias.shape} must be (2,)")
        if not np.all(np.isfinite(guidance_bias)):
            raise ValueError("normalized_bias must contain only finite values")
        if np.any((guidance_bias < -1.0) | (guidance_bias > 1.0)):
            raise ValueError("normalized_bias values must be in [-1, 1]")
        if target_index is not None and target_index != state.current_target_index:
            self._reset_target_tracking(state)
        frame = velocity_local_frame(state.velocity_mps)
        boost_duration_s = self.config.missile.boost_duration_s
        boosting = state.age_s < boost_duration_s
        target_valid = target is not None and target.alive
        pn_load_body_g = np.zeros(3, dtype=np.float64)
        bias_load_body_g = np.zeros(3, dtype=np.float64)
        gravity_load_body_g = np.zeros(3, dtype=np.float64)
        final_load_body_g = np.zeros(3, dtype=np.float64)
        if boosting:
            state.guidance_mode = "boost"
            gravity_load_body_g = self._gravity_compensation_load_body_g(frame)
            final_load_body_g = self._boost_climb_load_body_g(
                state,
                frame,
                gravity_load_body_g,
            )
            locked = False
        else:
            guidance_target, locked = self._update_guidance_target(state, target, target_index)
            gravity_load_body_g = self._gravity_compensation_load_body_g(frame)
            if guidance_target is not None:
                pn_acceleration = self.guidance.command(state, guidance_target)
                pn_acceleration -= np.dot(pn_acceleration, frame[:, 0]) * frame[:, 0]
                pn_load_body_g[1:] = (frame[:, 1:].T @ pn_acceleration) / G0
                bias_norm = norm(guidance_bias)
                bias_direction = guidance_bias / max(1.0, bias_norm)
                bias_load_body_g[1:] = (
                    bias_direction * self.config.missile.max_guidance_bias_g
                )
            final_load_body_g[1:] = clip_norm(
                pn_load_body_g[1:] + bias_load_body_g[1:] + gravity_load_body_g[1:],
                self.config.missile.max_load_factor_g,
            )
        drag = self._missile_drag_acceleration(state, final_load_body_g)
        gravity = np.array([0.0, -G0, 0.0], dtype=np.float64)
        acceleration = gravity - drag
        acceleration += frame[:, 1:] @ (final_load_body_g[1:] * G0)
        if boosting:
            remaining = max(self.config.missile.boost_duration_s - state.age_s, self.config.time_step_s)
            required_speed_accel = max(0.0, (self.config.missile.max_speed_mps - norm(state.velocity_mps)) / remaining)
            axial_gravity = float(np.dot(gravity, frame[:, 0]))
            thrust_acceleration = max(0.0, required_speed_accel + norm(drag) - axial_gravity)
            acceleration += frame[:, 0] * thrust_acceleration
        mass_flow = self.config.missile.mass_flow_rate_kg_s if boosting else 0.0
        next_state = self._integrate(state, acceleration, mass_flow)
        next_state.seeker_locked = locked
        next_state.fov_out_time_s = state.fov_out_time_s
        next_state.guidance_bias = guidance_bias.copy()
        next_state.pn_load_body_g = pn_load_body_g
        next_state.bias_load_body_g = bias_load_body_g
        next_state.gravity_load_body_g = gravity_load_body_g
        next_state.final_load_body_g = final_load_body_g
        speed = norm(next_state.velocity_mps)
        if boosting and next_state.age_s >= boost_duration_s:
            next_state.age_s = boost_duration_s
            next_state.velocity_mps = unit(next_state.velocity_mps, frame[:, 0]) * self.config.missile.max_speed_mps
            next_state.fuel_mass_kg = 0.0
            next_state.mass_kg = self.config.missile.dry_mass_kg
            speed = self.config.missile.max_speed_mps
        next_state.energy = float(np.clip(speed / self.config.missile.max_speed_mps, 0.0, 1.0))
        if next_state.position_m[UP_AXIS] <= 0.0 or next_state.age_s >= self.config.missile.max_guidance_time_s:
            next_state.alive = False
        if next_state.age_s >= self.config.missile.boost_duration_s and speed < self.config.missile.min_speed_mps:
            next_state.alive = False
        return next_state

    @staticmethod
    def _reset_target_tracking(state: ThreeDoFState) -> None:
        state.seeker_locked = False
        state.fov_out_time_s = 0.0
        state.guidance_mode = "inertial"
        state.target_estimate_valid = False
        state.target_estimate_target_index = -1
        state.target_estimate_position_m = np.zeros(3, dtype=np.float64)
        state.target_estimate_velocity_mps = np.zeros(3, dtype=np.float64)
        state.target_estimate_age_s = 0.0

    def _update_guidance_target(
        self,
        state: ThreeDoFState,
        target: ThreeDoFState | None,
        target_index: int | None,
    ) -> tuple[ThreeDoFState | None, bool]:
        if target is None or not target.alive:
            self._reset_target_tracking(state)
            return None, False
        effective_target_index = state.current_target_index if target_index is None else target_index
        angle_deg = seeker_boresight_angle_deg(state, target)
        limit = seeker_fov_limit_deg(state, effective_target_index, self.config.missile)
        if angle_deg <= limit:
            state.fov_out_time_s = 0.0
            state.seeker_locked = True
            state.guidance_mode = "locked"
            state.target_estimate_valid = True
            state.target_estimate_target_index = int(effective_target_index)
            state.target_estimate_position_m = target.position_m.astype(np.float64).copy()
            state.target_estimate_velocity_mps = target.velocity_mps.astype(np.float64).copy()
            state.target_estimate_age_s = 0.0
            return target, True

        if not state.target_estimate_valid:
            state.target_estimate_valid = True
            state.target_estimate_target_index = int(effective_target_index)
            state.target_estimate_position_m = target.position_m.astype(np.float64).copy()
            state.target_estimate_velocity_mps = target.velocity_mps.astype(np.float64).copy()
            state.target_estimate_age_s = 0.0
        else:
            state.target_estimate_position_m = (
                state.target_estimate_position_m
                + state.target_estimate_velocity_mps * self.config.time_step_s
            )
            state.target_estimate_age_s += self.config.time_step_s

        was_held = state.seeker_locked or state.guidance_mode == "lock_hold"
        if was_held:
            state.fov_out_time_s += self.config.time_step_s
        held_lock = was_held and state.fov_out_time_s < self.config.missile.fov_break_hold_s
        state.seeker_locked = held_lock
        state.guidance_mode = "lock_hold" if held_lock else "inertial"
        estimate = target.copy()
        estimate.position_m = state.target_estimate_position_m.copy()
        estimate.velocity_mps = state.target_estimate_velocity_mps.copy()
        return estimate, held_lock

    def _gravity_compensation_load_body_g(self, frame: np.ndarray) -> np.ndarray:
        gravity = np.array([0.0, -G0, 0.0], dtype=np.float64)
        load = np.zeros(3, dtype=np.float64)
        load[1:] = -(frame[:, 1:].T @ gravity) / G0
        return load

    def _boost_climb_load_body_g(
        self,
        state: ThreeDoFState,
        frame: np.ndarray,
        gravity_load_body_g: np.ndarray,
    ) -> np.ndarray:
        speed = max(norm(state.velocity_mps), EPS)
        flight_path_angle = math.asin(float(np.clip(state.velocity_mps[UP_AXIS] / speed, -1.0, 1.0)))
        if not math.isfinite(state.boost_initial_flight_path_angle_rad):
            state.boost_initial_flight_path_angle_rad = flight_path_angle
        transition_s = min(
            self.config.missile.boost_pitch_transition_s,
            self.config.missile.boost_duration_s,
        )
        phase = float(np.clip(state.age_s / transition_s, 0.0, 1.0))
        smooth_phase = phase * phase * (3.0 - 2.0 * phase)
        target_angle = math.radians(self.config.missile.boost_climb_angle_deg)
        angle_delta = target_angle - state.boost_initial_flight_path_angle_rad
        reference_angle = state.boost_initial_flight_path_angle_rad + angle_delta * smooth_phase
        reference_rate = 0.0
        if phase < 1.0:
            reference_rate = angle_delta * 6.0 * phase * (1.0 - phase) / transition_s
        tracking_rate = reference_rate + self.config.missile.boost_pitch_tracking_gain * (
            reference_angle - flight_path_angle
        )
        maneuver_load_body_g = np.zeros(3, dtype=np.float64)
        maneuver_load_body_g[1] = speed * tracking_rate / G0
        return clip_norm(
            gravity_load_body_g + maneuver_load_body_g,
            self.config.missile.max_load_factor_g,
        )

    def _integrate(self, state: ThreeDoFState, acceleration: np.ndarray, mass_flow_rate_kg_s: float) -> ThreeDoFState:
        dt = self.config.time_step_s
        next_state = state.copy()
        next_state.position_m = state.position_m + state.velocity_mps * dt + 0.5 * acceleration * dt * dt
        next_state.velocity_mps = state.velocity_mps + acceleration * dt
        fuel_used = min(state.fuel_mass_kg, max(0.0, mass_flow_rate_kg_s) * dt)
        next_state.fuel_mass_kg = state.fuel_mass_kg - fuel_used
        next_state.mass_kg = state.mass_kg - fuel_used
        next_state.age_s = round(state.age_s + dt, 12)
        return next_state

    def _missile_drag_acceleration(self, state: ThreeDoFState, final_load_body_g: np.ndarray) -> np.ndarray:
        load = np.asarray(final_load_body_g, dtype=np.float64)
        if load.shape != (3,):
            raise ValueError(f"final_load_body_g shape {load.shape} must be (3,)")
        speed = norm(state.velocity_mps)
        if speed <= EPS:
            return np.zeros(3, dtype=np.float64)
        altitude_m = float(state.position_m[UP_AXIS])
        density = air_density(altitude_m)
        dynamic_pressure_pa = 0.5 * density * speed * speed
        reference_area_m2 = self.config.missile.reference_area_m2
        mach = speed / max(speed_of_sound(altitude_m), EPS)
        coefficient = self._zero_lift_drag_coefficient(mach)
        if dynamic_pressure_pa > EPS:
            lift_coefficient = (
                state.mass_kg * G0 * norm(load[1:])
                / (dynamic_pressure_pa * reference_area_m2)
            )
            coefficient += self.config.missile.induced_drag_factor * lift_coefficient * lift_coefficient
        drag_force = dynamic_pressure_pa * coefficient * reference_area_m2
        return unit(state.velocity_mps) * (drag_force / max(state.mass_kg, EPS))

    def _zero_lift_drag_coefficient(self, mach: float) -> float:
        fixed_coefficient = self.config.missile.drag_coefficient
        if fixed_coefficient is not None:
            return float(fixed_coefficient)
        return float(
            np.interp(
                mach,
                self.config.missile.drag_mach_breakpoints,
                self.config.missile.zero_lift_drag_coefficients,
            )
        )

    @staticmethod
    def _update_ranges(state: EngagementState, target_indices: np.ndarray) -> None:
        targets = np.asarray(target_indices, dtype=np.int64)
        if targets.shape != (len(state.red),):
            raise ValueError(f"target_indices shape {targets.shape} must be {(len(state.red),)}")
        for missile_index, missile in enumerate(state.red):
            target_index = int(targets[missile_index])
            if missile.min_range_target_index != target_index:
                missile.min_range_m = math.inf
                missile.min_range_target_index = target_index
            if missile.alive and 0 <= target_index < len(state.blue) and state.blue[target_index].alive:
                missile.min_range_m = min(missile.min_range_m, los_kinematics(missile, state.blue[target_index]).range_m)
