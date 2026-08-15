from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .math_utils import norm, unit
from .seeker import seeker_target_visible
from .types import EngagementState, EnvironmentConfig, JointAction, ThreeDoFState, los_kinematics


def mission_completion(state: EngagementState, initial_blue_count: int | None = None) -> float:
    """Return the destroyed-blue fraction D(s)."""
    denominator = max(initial_blue_count or len(state.blue), 1)
    return sum(not target.alive for target in state.blue) / denominator


def ineffective_loss_rate(state: EngagementState, initial_red_count: int | None = None) -> float:
    """Return the monotonic fraction of red missiles lost without a valid hit."""
    denominator = max(initial_red_count or len(state.red), 1)
    invalid_losses = sum(
        not missile.alive and missile.loss_reason != "valid_hit"
        for missile in state.red
    )
    return invalid_losses / denominator


def normalized_terminal_time(
    config: EnvironmentConfig,
    state: EngagementState,
) -> float:
    """Return task time T with the same 0.1 s success resolution as mission J."""
    all_blue_done = not any(target.alive for target in state.blue)
    if not all_blue_done:
        return 1.0
    return float(
        math.ceil(state.time_s / 0.1 - 1.0e-12)
        * 0.1
        / config.missile.max_guidance_time_s
    )


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + math.exp(-float(np.clip(value, -60.0, 60.0)))))


def assignment_pair_quality(
    config: EnvironmentConfig,
    missile: ThreeDoFState,
    target: ThreeDoFState,
    *,
    target_index: int | None = None,
) -> float:
    """Return a gated heuristic reachability score in [0, 1], not a hit probability."""
    if not missile.alive or not target.alive:
        return 0.0
    relative_position = target.position_m - missile.position_m
    distance = norm(relative_position)
    los_unit = unit(relative_position, np.array([1.0, 0.0, 0.0]))
    kinematics = los_kinematics(missile, target)
    closing_speed = max(kinematics.closing_speed_mps, 0.0)
    if closing_speed <= 0.0:
        return 0.0
    effective_target_index = (
        missile.current_target_index if target_index is None else int(target_index)
    )
    if not seeker_target_visible(
        missile,
        target,
        effective_target_index,
        config.missile,
        detection_range_m=config.sensor.detection_range_m,
    ):
        return 0.0
    relative_velocity_sq = float(np.dot(kinematics.relative_velocity_mps, kinematics.relative_velocity_mps))
    closest_approach_time_s = -float(np.dot(relative_position, kinematics.relative_velocity_mps)) / (
        relative_velocity_sq + 1.0e-9
    )
    if closest_approach_time_s <= 0.0:
        return 0.0
    estimated_time_s = max(distance / max(closing_speed, 1.0), closest_approach_time_s)
    remaining_s = max(config.missile.max_guidance_time_s - missile.age_s, 0.0)
    time_margin_s = remaining_s - estimated_time_s
    if time_margin_s <= 0.0:
        return 0.0
    time_quality = time_margin_s / (
        time_margin_s + config.reward.high_time_margin_scale_s
    )
    velocity_unit = unit(missile.velocity_mps, np.array([1.0, 0.0, 0.0]))
    angle_rad = math.acos(float(np.clip(np.dot(velocity_unit, los_unit), -1.0, 1.0)))
    acquisition_fov_rad = math.radians(config.missile.seeker_acquisition_fov_deg)
    angle_quality = math.exp(-((angle_rad / acquisition_fov_rad) ** 2))
    used_load = norm(
        np.asarray(missile.pn_load_body_g, dtype=np.float64)[1:]
        + np.asarray(missile.gravity_load_body_g, dtype=np.float64)[1:]
        + np.asarray(missile.bias_load_body_g, dtype=np.float64)[1:]
    )
    available_load = float(
        np.clip(1.0 - used_load / config.missile.max_load_factor_g, 0.0, 1.0)
    )
    energy_quality = float(
        np.clip(
            (missile.energy - config.reward.assignment_min_energy_fraction)
            / max(1.0 - config.reward.assignment_min_energy_fraction, 1.0e-9),
            0.0,
            1.0,
        )
    )
    load_quality = float(
        np.clip(
            (available_load - config.reward.assignment_min_available_load_fraction)
            / max(1.0 - config.reward.assignment_min_available_load_fraction, 1.0e-9),
            0.0,
            1.0,
        )
    )
    return float(
        np.clip(
            time_quality * angle_quality * energy_quality * load_quality,
            0.0,
            1.0,
        )
    )


def assignment_feasibility_potential(
    config: EnvironmentConfig,
    state: EngagementState,
    *,
    initial_blue_count: int | None = None,
    terminal: bool = False,
) -> float:
    """Return state potential Phi_H using the assignment already stored in state."""
    if terminal:
        return 0.0
    targets = np.asarray([red.current_target_index for red in state.red], dtype=np.int64)
    denominator = max(initial_blue_count or len(state.blue), 1)
    destroyed = sum(not target.alive for target in state.blue)
    coverage = 0.0
    for blue_index, blue in enumerate(state.blue):
        if not blue.alive:
            continue
        assigned: list[tuple[int, float]] = []
        for red_index, red in enumerate(state.red):
            if red.alive and int(targets[red_index]) == blue_index:
                assigned.append(
                    (
                        red_index,
                        assignment_pair_quality(
                            config,
                            red,
                            blue,
                            target_index=blue_index,
                        ),
                    )
                )
        residual_noncoverage = 1.0
        for red_index, quality in assigned:
            max_correlation = 0.0
            for peer_index, _ in assigned:
                if peer_index == red_index:
                    continue
                first = state.red[red_index]
                second = state.red[peer_index]
                first_los = los_kinematics(first, blue)
                second_los = los_kinematics(second, blue)
                angle_rad = math.acos(
                    float(
                        np.clip(
                            np.dot(first_los.los_unit, second_los.los_unit),
                            -1.0,
                            1.0,
                        )
                    )
                )
                first_tgo = first_los.range_m / max(first_los.closing_speed_mps, 1.0)
                second_tgo = second_los.range_m / max(second_los.closing_speed_mps, 1.0)
                angular_similarity = math.exp(
                    -(
                        angle_rad
                        / math.radians(config.reward.assignment_correlation_angle_scale_deg)
                    ) ** 2
                )
                time_similarity = math.exp(
                    -(
                        abs(first_tgo - second_tgo)
                        / config.reward.assignment_correlation_time_scale_s
                    ) ** 2
                )
                max_correlation = max(max_correlation, angular_similarity * time_similarity)
            effective_quality = quality * (
                1.0 - config.reward.assignment_correlation_weight * max_correlation
            )
            residual_noncoverage *= 1.0 - float(np.clip(effective_quality, 0.0, 1.0))
        coverage += 1.0 - residual_noncoverage
    return float(np.clip((destroyed + coverage) / denominator, 0.0, 1.0))


def _track_confidence(
    config: EnvironmentConfig,
    missile: ThreeDoFState,
    target_index: int | None = None,
) -> float:
    estimate_target_index = (
        missile.target_estimate_target_index
        if missile.target_estimate_target_index >= 0
        else missile.current_target_index
    )
    if target_index is not None and estimate_target_index != target_index:
        return 0.0
    if missile.guidance_mode == "locked":
        return 1.0
    if missile.guidance_mode == "lock_hold" and missile.target_estimate_valid:
        horizon = max(config.missile.fov_break_hold_s, config.time_step_s)
        return float(np.clip(1.0 - missile.target_estimate_age_s / horizon, 0.0, 1.0))
    if missile.target_estimate_valid:
        horizon = max(config.missile.fov_break_hold_s, config.time_step_s)
        return float(0.25 * np.clip(1.0 - missile.target_estimate_age_s / horizon, 0.0, 1.0))
    return 0.0


def low_intercept_potential(
    config: EnvironmentConfig,
    missile: ThreeDoFState,
    target: ThreeDoFState,
    *,
    target_index: int | None = None,
) -> tuple[float, float, float, float]:
    """Return (Phi_L, ZEM score, track confidence, time-to-go)."""
    relative_position = target.position_m - missile.position_m
    relative_velocity = target.velocity_mps - missile.velocity_mps
    tau = -float(np.dot(relative_position, relative_velocity)) / (
        float(np.dot(relative_velocity, relative_velocity)) + 1.0e-9
    )
    remaining_s = max(config.missile.max_guidance_time_s - missile.age_s, 0.0)
    clamped_tau = float(np.clip(tau, 0.0, remaining_s))
    zem_distance = norm(relative_position + relative_velocity * clamped_tau)
    zem_gate = _sigmoid(tau / config.reward.zem_time_gate_scale_s) * _sigmoid(
        (remaining_s - tau) / config.reward.zem_time_gate_scale_s
    )
    zem_score = (
        math.exp(
            -max(zem_distance, config.reward.zem_floor_range_m)
            / config.reward.zem_reference_range_m
        )
        * zem_gate
    )
    closing_score = float(
        np.clip(los_kinematics(missile, target).closing_speed_mps / 1500.0, 0.0, 1.0)
    )
    confidence = _track_confidence(config, missile, target_index)
    closing_weight = max(
        1.0 - config.reward.zem_weight - config.reward.seeker_lock_weight,
        0.0,
    )
    potential = (
        config.reward.zem_weight * zem_score
        + closing_weight * closing_score
        + config.reward.seeker_lock_weight * confidence
    )
    return float(np.clip(potential, 0.0, 1.0)), float(zem_score), confidence, float(tau)


@dataclass
class _HighWindow:
    start_step: int
    start_completion: float
    start_waste: float
    start_potential: float


@dataclass
class _LowWindow:
    start_step: int
    active: np.ndarray
    target_indices: np.ndarray
    start_potential: np.ndarray
    start_load_integral: np.ndarray
    start_smooth_integral: np.ndarray
    hit_red_indices: set[int]
    invalid_red_indices: set[int]


@dataclass
class _ControlEffortAccumulator:
    """Physical-time, active-control-only U accumulator shared by every runner."""

    config: EnvironmentConfig
    initial_red_count: int = 1
    policy_horizon_s: float = 1.0
    load_integral: np.ndarray | None = None
    smooth_integral: np.ndarray | None = None
    smooth_rate: np.ndarray | None = None
    smooth_remaining_s: np.ndarray | None = None

    def reset(self, red_count: int) -> None:
        self.initial_red_count = max(int(red_count), 1)
        self.policy_horizon_s = max(self.config.policy_horizon_s, self.config.time_step_s)
        self.load_integral = np.zeros(self.initial_red_count, dtype=np.float64)
        self.smooth_integral = np.zeros(self.initial_red_count, dtype=np.float64)
        self.smooth_rate = np.zeros(self.initial_red_count, dtype=np.float64)
        self.smooth_remaining_s = np.zeros(self.initial_red_count, dtype=np.float64)

    @property
    def penalty_weight(self) -> float:
        reward = self.config.reward
        return reward.low_load_penalty + reward.low_smooth_penalty

    @property
    def load_mix(self) -> float:
        weight = self.penalty_weight
        return 0.8 if weight <= 1.0e-12 else self.config.reward.low_load_penalty / weight

    @property
    def smooth_mix(self) -> float:
        return 1.0 - self.load_mix

    def _require_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if (
            self.load_integral is None
            or self.smooth_integral is None
            or self.smooth_rate is None
            or self.smooth_remaining_s is None
        ):
            raise RuntimeError("control effort accumulator was not reset")
        return (
            self.load_integral,
            self.smooth_integral,
            self.smooth_rate,
            self.smooth_remaining_s,
        )

    @staticmethod
    def _normalized_bias(bias: np.ndarray) -> np.ndarray:
        values = np.asarray(bias, dtype=np.float64)
        norms = np.linalg.norm(values, axis=-1, keepdims=True)
        return values / np.maximum(norms, 1.0)

    def accumulate(
        self,
        previous: EngagementState,
        current: EngagementState,
        action: JointAction,
        previous_action: JointAction | None,
        *,
        bias_updated: bool,
    ) -> None:
        load_integral, smooth_integral, smooth_rate, smooth_remaining_s = self._require_arrays()
        if len(current.red) != self.initial_red_count:
            raise ValueError("control effort red count changed within an episode")
        targets = np.asarray(action.red.target_indices, dtype=np.int64)
        active = np.zeros(self.initial_red_count, dtype=bool)
        for red_index, missile in enumerate(previous.red):
            target_index = int(targets[red_index])
            active[red_index] = (
                missile.alive
                and 0 <= target_index < len(previous.blue)
                and previous.blue[target_index].alive
            )
        current_bias = self._normalized_bias(action.red.guidance_bias)
        previous_bias = (
            np.zeros_like(current_bias)
            if previous_action is None
            else self._normalized_bias(previous_action.red.guidance_bias)
        )
        if bias_updated:
            bias_delta = current_bias - previous_bias
            changed = active & (np.sum(bias_delta * bias_delta, axis=-1) > 1.0e-15)
            if changed.any():
                smooth_rate[changed] = np.clip(
                    np.sum(bias_delta[changed] * bias_delta[changed], axis=-1)
                    / self.config.reward.smooth_bias_denominator,
                    0.0,
                    1.0,
                )
                smooth_remaining_s[changed] = self.config.bias_update_interval_s

        actual_load = np.stack(
            [missile.bias_load_body_g[1:] for missile in current.red],
            axis=0,
        )
        load_ratio_square = np.clip(
            np.linalg.norm(actual_load, axis=-1)
            / max(self.config.missile.max_guidance_bias_g, 1.0e-9),
            0.0,
            1.0,
        ) ** 2
        active_smooth_rate = np.where(smooth_remaining_s > 0.0, smooth_rate, 0.0)
        dt = self.config.time_step_s
        load_integral += active * load_ratio_square * dt
        smooth_integral += active * active_smooth_rate * dt
        smooth_remaining_s[:] = np.maximum(smooth_remaining_s - dt, 0.0)
        smooth_rate[smooth_remaining_s <= 0.0] = 0.0

    def effort_integral_by_red(self) -> np.ndarray:
        load_integral, smooth_integral, _, _ = self._require_arrays()
        return self.load_mix * load_integral + self.smooth_mix * smooth_integral

    def effort_increment(self, start_integral: np.ndarray) -> np.ndarray:
        return np.maximum(self.effort_integral_by_red() - start_integral, 0.0)

    def normalized_effort_increment(self, start_integral: np.ndarray) -> np.ndarray:
        return self.effort_increment(start_integral) / (
            self.initial_red_count * self.policy_horizon_s
        )

    @property
    def control_effort(self) -> float:
        normalized = self.effort_integral_by_red().sum() / (
            self.initial_red_count * self.policy_horizon_s
        )
        return float(np.clip(normalized, 0.0, 1.0))


class HierarchicalRewardLayer:
    """Stateful high/low reward windows after policy entry."""

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config
        self.initial_red_count = config.scenario.red_count
        self.initial_blue_count = config.scenario.blue_count
        self._high: _HighWindow | None = None
        self._low: _LowWindow | None = None
        self._control = _ControlEffortAccumulator(config)
        self._control.reset(self.initial_red_count)

    @property
    def max_low_windows(self) -> int:
        return max(
            int(math.ceil(self.config.policy_horizon_s / self.config.bias_update_interval_s)),
            1,
        )

    def reset(self, state: EngagementState) -> None:
        self.initial_red_count = len(state.red)
        self.initial_blue_count = len(state.blue)
        self._high = None
        self._low = None
        self._control.reset(self.initial_red_count)

    @property
    def control_effort(self) -> float:
        return self._control.control_effort

    def high_potential(self, state: EngagementState, terminal: bool = False) -> float:
        return assignment_feasibility_potential(
            self.config,
            state,
            initial_blue_count=self.initial_blue_count,
            terminal=terminal,
        )

    def evaluate_transition(
        self,
        previous: EngagementState,
        current: EngagementState,
        action: JointAction,
        previous_action: JointAction | None,
        event_info: dict[str, Any],
        *,
        done: bool,
        bias_updated: bool,
    ) -> tuple[float, np.ndarray, dict[str, Any]]:
        targets = np.asarray(action.red.target_indices, dtype=np.int64)
        if self._high is None:
            self._begin_high(previous)
        if self._low is None:
            self._begin_low(previous, targets)
        assert self._low is not None
        self._control.accumulate(
            previous,
            current,
            action,
            previous_action,
            bias_updated=bias_updated,
        )
        self._low.hit_red_indices.update(int(index) for index in event_info.get("hit_red_indices", []))
        for event in event_info.get("red_loss_events", []):
            if event.get("loss_reason") != "valid_hit":
                self._low.invalid_red_indices.add(int(event["red_index"]))

        event_boundary = bool(
            event_info.get("hit_red_indices")
            or event_info.get("red_loss_events")
        )
        high_boundary = (
            current.step_count % self.config.assignment_update_steps == 0
            or event_boundary
        )
        low_boundary = (
            current.step_count % self.config.bias_update_steps == 0
            or event_boundary
        )
        reward_high = 0.0
        reward_low = np.zeros(len(current.red), dtype=np.float32)
        high_components = self._empty_high_components()
        low_components = self._empty_low_components(len(current.red))
        if low_boundary or done:
            reward_low, low_components = self._finish_low(
                current,
                event_info,
                terminated=done,
            )
            self._low = None
        if high_boundary or done:
            reward_high, high_components = self._finish_high(current, event_info, done)
            self._high = None
        return reward_high, reward_low, {
            "reward_components": high_components,
            "reward_low_components": low_components,
            "high_reward_settled": bool(high_boundary or done),
            "low_reward_settled": bool(low_boundary or done),
            "assignment_event_boundary": event_boundary,
            "control_effort": self.control_effort,
        }

    def _begin_high(self, state: EngagementState) -> None:
        self._high = _HighWindow(
            start_step=state.step_count,
            start_completion=mission_completion(state, self.initial_blue_count),
            start_waste=ineffective_loss_rate(state, self.initial_red_count),
            start_potential=self.high_potential(state),
        )

    def _finish_high(
        self,
        state: EngagementState,
        event_info: dict[str, Any],
        done: bool,
    ) -> tuple[float, dict[str, Any]]:
        assert self._high is not None
        reward = self.config.reward
        completion_delta = mission_completion(state, self.initial_blue_count) - self._high.start_completion
        waste_delta = ineffective_loss_rate(state, self.initial_red_count) - self._high.start_waste
        next_potential = self.high_potential(state, terminal=done)
        elapsed_s = (state.step_count - self._high.start_step) * self.config.time_step_s
        discount = reward.high_potential_gamma**elapsed_s
        potential_delta = discount * next_potential - self._high.start_potential
        terminal_reason = str(event_info.get("termination_reason", "none"))
        if not done:
            terminal_reason = "none"
        normalized_time = normalized_terminal_time(self.config, state) if done else 1.0
        control_effort = self.control_effort
        terminal_outcome_adjustment = 0.0
        if done and terminal_reason == "success":
            terminal_outcome_adjustment = reward.terminal_success_reward
        elif done and terminal_reason == "red_failure":
            terminal_outcome_adjustment = -reward.terminal_failure_penalty
        elif done and terminal_reason == "timeout":
            terminal_outcome_adjustment = -reward.terminal_timeout_penalty
        terminal_reward = (
            -reward.high_time_penalty_per_s * normalized_time
            - self._control.penalty_weight * control_effort
            + terminal_outcome_adjustment
            if done
            else 0.0
        )
        components: dict[str, Any] = {
            "completion_delta": float(completion_delta),
            "waste_delta": float(waste_delta),
            "potential_current": float(self._high.start_potential),
            "potential_next": float(next_potential),
            "potential_discount": float(discount),
            "potential_delta": float(potential_delta),
            "elapsed_s": float(elapsed_s),
            "damage_reward": float(reward.high_damage_weight * completion_delta),
            "waste_penalty": float(-reward.high_waste_weight * waste_delta),
            "potential_reward": float(reward.high_potential_weight * potential_delta),
            "time_penalty": float(-reward.high_time_penalty_per_s * normalized_time if done else 0.0),
            "control_penalty": float(
                -self._control.penalty_weight * control_effort
                if done else 0.0
            ),
            "control_effort": float(control_effort),
            "terminal_outcome_adjustment": float(terminal_outcome_adjustment),
            "terminal_reward": float(terminal_reward),
            "terminal_reason": terminal_reason,
        }
        total = (
            components["damage_reward"]
            + components["waste_penalty"]
            + components["potential_reward"]
            + terminal_reward
        )
        return float(total), components

    def _begin_low(
        self,
        state: EngagementState,
        targets: np.ndarray,
    ) -> None:
        n_red = len(state.red)
        active = np.zeros(n_red, dtype=bool)
        potentials = np.zeros(n_red, dtype=np.float64)
        for red_index, missile in enumerate(state.red):
            target_index = int(targets[red_index])
            valid = missile.alive and 0 <= target_index < len(state.blue) and state.blue[target_index].alive
            active[red_index] = valid
            if valid:
                potentials[red_index] = low_intercept_potential(
                    self.config,
                    missile,
                    state.blue[target_index],
                    target_index=target_index,
                )[0]
        self._low = _LowWindow(
            start_step=state.step_count,
            active=active,
            target_indices=targets.copy(),
            start_potential=potentials,
            start_load_integral=self._control.load_integral.copy(),
            start_smooth_integral=self._control.smooth_integral.copy(),
            hit_red_indices=set(),
            invalid_red_indices=set(),
        )

    def _finish_low(
        self,
        state: EngagementState,
        event_info: dict[str, Any],
        *,
        terminated: bool,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        assert self._low is not None
        reward = self.config.reward
        n_red = len(state.red)
        next_potential = np.zeros(n_red, dtype=np.float64)
        zem = np.zeros(n_red, dtype=np.float64)
        confidence = np.zeros(n_red, dtype=np.float64)
        potential_delta = np.zeros(n_red, dtype=np.float64)
        hit_event = np.zeros(n_red, dtype=np.float64)
        invalid_event = np.zeros(n_red, dtype=np.float64)
        time_credit = np.zeros(n_red, dtype=np.float64)
        option_boundary_reason = ["none"] * n_red
        elapsed_s = (state.step_count - self._low.start_step) * self.config.time_step_s
        discount = reward.low_potential_gamma ** (
            elapsed_s / self.config.bias_update_interval_s
        )
        for red_index in range(n_red):
            if not self._low.active[red_index]:
                continue
            target_index = int(self._low.target_indices[red_index])
            target_was_destroyed = (
                0 <= target_index < len(state.blue)
                and not state.blue[target_index].alive
            )
            target_changed = state.red[red_index].current_target_index != target_index
            pair_continues = (
                not terminated
                and state.red[red_index].alive
                and 0 <= target_index < len(state.blue)
                and state.blue[target_index].alive
            )
            if pair_continues:
                next_potential[red_index], zem[red_index], confidence[red_index], _ = low_intercept_potential(
                    self.config,
                    state.red[red_index],
                    state.blue[target_index],
                    target_index=target_index,
                )
            if target_changed:
                option_boundary_reason[red_index] = "target_changed"
            elif target_was_destroyed:
                option_boundary_reason[red_index] = "target_destroyed"
            if (
                option_boundary_reason[red_index] == "none"
                or reward.low_option_boundary_potential == "terminal_zero"
            ):
                potential_delta[red_index] = (
                    discount * next_potential[red_index]
                    - self._low.start_potential[red_index]
                )
            hit_event[red_index] = float(red_index in self._low.hit_red_indices)
            invalid_event[red_index] = float(red_index in self._low.invalid_red_indices)

        load_cost = np.maximum(
            self._control.load_integral - self._low.start_load_integral,
            0.0,
        )
        smooth_cost = np.maximum(
            self._control.smooth_integral - self._low.start_smooth_integral,
            0.0,
        )
        start_effort = (
            self._control.load_mix * self._low.start_load_integral
            + self._control.smooth_mix * self._low.start_smooth_integral
        )
        control_effort_increment = self._control.normalized_effort_increment(start_effort)
        control_penalty = self._control.penalty_weight * control_effort_increment
        normalized_time = normalized_terminal_time(self.config, state) if terminated else 0.0
        active_count = int(self._low.active.sum())
        if (
            terminated
            and reward.low_time_credit_mode == "terminal_active_share"
            and active_count > 0
        ):
            time_credit[self._low.active] = (
                -reward.low_time_weight * normalized_time / active_count
            )
        values = (
            (reward.low_damage_weight / self.initial_blue_count) * hit_event
            - (reward.low_missile_failure_penalty / self.initial_red_count) * invalid_event
            + reward.low_potential_weight * potential_delta
            - control_penalty
            + time_credit
        )
        values = np.where(self._low.active, values, 0.0).astype(np.float32)
        components: dict[str, Any] = {
            "active_mask": self._low.active.tolist(),
            "potential_current": self._low.start_potential.tolist(),
            "potential_next": next_potential.tolist(),
            "zem": zem.tolist(),
            "track_confidence": confidence.tolist(),
            "potential_delta": potential_delta.tolist(),
            "potential_discount": float(discount),
            "elapsed_s": float(elapsed_s),
            "hit_event": hit_event.tolist(),
            "miss_event": invalid_event.tolist(),
            "load_cost": load_cost.tolist(),
            "smooth_cost": smooth_cost.tolist(),
            "control_effort_increment": control_effort_increment.tolist(),
            "normalized_terminal_time": float(normalized_time),
            "time_credit": time_credit.tolist(),
            "time_credit_total": float(time_credit.sum()),
            "time_credit_unassigned": bool(
                terminated
                and reward.low_time_credit_mode == "terminal_active_share"
                and active_count == 0
            ),
            "option_boundary_reason": option_boundary_reason,
            "reward_low": values.tolist(),
        }
        return values, components

    @staticmethod
    def _empty_high_components() -> dict[str, Any]:
        return {
            "completion_delta": 0.0,
            "waste_delta": 0.0,
            "potential_current": 0.0,
            "potential_next": 0.0,
            "potential_discount": 1.0,
            "potential_delta": 0.0,
            "elapsed_s": 0.0,
            "damage_reward": 0.0,
            "waste_penalty": 0.0,
            "potential_reward": 0.0,
            "time_penalty": 0.0,
            "control_penalty": 0.0,
            "control_effort": 0.0,
            "terminal_outcome_adjustment": 0.0,
            "terminal_reward": 0.0,
            "terminal_reason": "none",
        }

    @staticmethod
    def _empty_low_components(n_red: int) -> dict[str, Any]:
        zeros = [0.0] * n_red
        return {
            "active_mask": [False] * n_red,
            "potential_current": zeros.copy(),
            "potential_next": zeros.copy(),
            "zem": zeros.copy(),
            "track_confidence": zeros.copy(),
            "potential_delta": zeros.copy(),
            "potential_discount": 1.0,
            "elapsed_s": 0.0,
            "hit_event": zeros.copy(),
            "miss_event": zeros.copy(),
            "load_cost": zeros.copy(),
            "smooth_cost": zeros.copy(),
            "control_effort_increment": zeros.copy(),
            "normalized_terminal_time": 0.0,
            "time_credit": zeros.copy(),
            "time_credit_total": 0.0,
            "time_credit_unassigned": False,
            "option_boundary_reason": ["none"] * n_red,
            "reward_low": zeros.copy(),
        }
