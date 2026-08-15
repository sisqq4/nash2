from __future__ import annotations

import math
from collections import deque
from typing import Deque

import numpy as np
import torch
from torch import Tensor

from ..policy.actor import AssignmentActorInputs, OverloadBiasActorInputs
from ..policy.critic import AssignmentCriticInputs, OverloadBiasCriticInputs
from .math_utils import norm, speed_of_sound, unit, velocity_local_frame
from .reward import assignment_pair_quality, ineffective_loss_rate, mission_completion
from .seeker import seeker_target_visible
from .types import EngagementState, EnvironmentConfig, EnvironmentObservation, ThreeDoFState, los_kinematics


POSITION_SCALE = np.array([200000.0, 20000.0, 200000.0], dtype=np.float64)
RELATIVE_POSITION_SCALE = np.array([200000.0, 20000.0, 200000.0], dtype=np.float64)
VELOCITY_SCALE_MPS = 2000.0
LOS_RATE_SCALE_RADPS = 0.1
ZEM_SCALE_M = 50000.0
FEATURE_CLIP = 5.0


class ObservationLayer:
    def __init__(self, config: EnvironmentConfig, device: torch.device | str | None = None) -> None:
        self.config = config
        self.device = device
        delay = config.sensor.communication_delay_steps
        self._history: Deque[EngagementState] | None = deque(maxlen=delay + 1) if delay > 0 else None
        self._rng = np.random.default_rng()
        self._last_delayed_state: EngagementState | None = None
        self._last_history_step_count: int | None = None

    def reset(self, seed: int | None = None) -> None:
        if self._history is not None:
            self._history.clear()
        self._rng = np.random.default_rng(seed)
        self._last_delayed_state = None
        self._last_history_step_count = None

    def advance(self, state: EngagementState) -> None:
        if self._history is None or self._last_history_step_count == state.step_count:
            return
        self._history.append(state.copy())
        self._last_history_step_count = state.step_count

    def observe(
        self,
        state: EngagementState,
        previous: EnvironmentObservation | None = None,
    ) -> EnvironmentObservation:
        delayed = self._delayed_state(state)
        self._last_delayed_state = delayed.copy()
        n_red = len(state.red)
        n_blue = len(state.blue)
        target_slots = n_blue + 1
        red_scale = float(max(n_red, 1))
        blue_scale = float(max(n_blue, 1))
        assignment_matrix = self._assignment_matrix(state)
        assignment_counts = assignment_matrix.sum(axis=0)
        alive_red = sum(red.alive for red in state.red)
        alive_blue = sum(blue.alive for blue in state.blue)
        unassigned = max(0.0, float(alive_red) - float(assignment_matrix.sum()))

        self_state = np.zeros((1, n_red, 13), dtype=np.float32)
        friend_entities = np.zeros((1, n_red, n_red, 11), dtype=np.float32)
        friend_mask = np.zeros((1, n_red, n_red), dtype=bool)
        target_entities = np.zeros((1, n_red, target_slots, 8), dtype=np.float32)
        pair_state = np.zeros((1, n_red, target_slots, 11), dtype=np.float32)
        current_assignment = np.zeros((1, n_red, target_slots), dtype=np.float32)
        target_entity_mask = np.zeros((1, n_red, target_slots), dtype=bool)
        target_mask = np.zeros((1, n_red, target_slots), dtype=bool)
        target_entity_mask[:, :, 0] = True
        target_mask[:, :, 0] = True
        environment_context = np.zeros((1, n_red, 5), dtype=np.float32)
        target_assignment_counts = np.zeros((1, n_red, target_slots), dtype=np.float32)
        agent_mask = np.zeros((1, n_red), dtype=bool)

        total_capacity = max(
            self.config.scenario.max_missiles_per_target * max(alive_blue, 1),
            1,
        )
        used_capacity = float(assignment_counts.sum())
        for red_index, red in enumerate(state.red):
            agent_mask[0, red_index] = red.alive
            self_state[0, red_index] = self._red_self_features(red)
            current_slot = red.current_target_index + 1 if 0 <= red.current_target_index < n_blue else 0
            current_assignment[0, red_index, current_slot] = 1.0
            for friend_index, friend in enumerate(delayed.red):
                if red_index == friend_index:
                    continue
                friend_mask[0, red_index, friend_index] = True
                friend_entities[0, red_index, friend_index] = self._friend_features(red, friend)

            for blue_index, blue in enumerate(state.blue):
                if not red.alive or red.age_s < self.config.policy_entry_time_s:
                    continue
                observed_blue, confidence = self._assignment_observed_target(
                    red,
                    blue,
                    blue_index,
                )
                if observed_blue is None:
                    continue
                slot = blue_index + 1
                target_entities[0, red_index, slot] = self._target_features(
                    red,
                    observed_blue,
                    confidence,
                )
                pair_state[0, red_index, slot] = self._pair_features(
                    red,
                    observed_blue,
                    target_index=blue_index,
                    assignment_count=float(assignment_counts[blue_index]),
                    current_target=red.current_target_index == blue_index,
                )
                target_entity_mask[0, red_index, slot] = True
                target_mask[0, red_index, slot] = blue.alive
                target_assignment_counts[0, red_index, slot] = (
                    assignment_counts[blue_index] / self.config.scenario.max_missiles_per_target
                )

            environment_context[0, red_index] = self._clip(
                np.array(
                    [
                        self._remaining_global_fraction(state),
                        alive_red / red_scale,
                        alive_blue / blue_scale,
                        unassigned / red_scale,
                        max(total_capacity - used_capacity, 0.0) / total_capacity,
                    ],
                    dtype=np.float64,
                )
            )

        assignment_actor_inputs = AssignmentActorInputs(
            self_state=self._tensor(self_state),
            friend_entities=self._tensor(friend_entities),
            friend_mask=self._tensor(friend_mask, dtype=torch.bool),
            target_entities=self._tensor(target_entities),
            pair_state=self._tensor(pair_state),
            current_assignment=self._tensor(current_assignment),
            target_mask=self._tensor(target_mask, dtype=torch.bool),
            environment_context=self._tensor(environment_context),
            target_assignment_counts=self._tensor(target_assignment_counts),
            target_entity_mask=self._tensor(target_entity_mask, dtype=torch.bool),
            agent_mask=self._tensor(agent_mask, dtype=torch.bool),
        )
        current_target_slots = np.array(
            [red.current_target_index + 1 if 0 <= red.current_target_index < n_blue else 0 for red in state.red],
            dtype=np.int64,
        )
        assignment_critic_inputs, execution_critic_inputs = self._critic_inputs(
            state,
            assignment_counts,
            unassigned,
        )
        execution_actor_inputs = (
            self.execution_inputs(state, current_target_slots)
            if previous is None
            else previous.execution_actor_inputs
        )
        return EnvironmentObservation(
            assignment_actor_inputs=assignment_actor_inputs,
            execution_actor_inputs=execution_actor_inputs,
            assignment_critic_inputs=assignment_critic_inputs,
            execution_critic_inputs=execution_critic_inputs,
            assignment_matrix=assignment_matrix,
        )

    def observe_execution(
        self,
        state: EngagementState,
        previous: EnvironmentObservation,
    ) -> EnvironmentObservation:
        delayed = self._delayed_state(state)
        self._last_delayed_state = delayed.copy()
        current_target_slots = np.array(
            [
                red.current_target_index + 1
                if 0 <= red.current_target_index < len(state.blue)
                else 0
                for red in state.red
            ],
            dtype=np.int64,
        )
        return EnvironmentObservation(
            assignment_actor_inputs=previous.assignment_actor_inputs,
            execution_actor_inputs=self.execution_inputs(state, current_target_slots),
            assignment_critic_inputs=previous.assignment_critic_inputs,
            execution_critic_inputs=self._execution_critic_input(state),
            assignment_matrix=self._assignment_matrix(state),
        )

    def execution_inputs(
        self,
        state: EngagementState,
        target_slots: Tensor | np.ndarray,
        hidden: Tensor | None = None,
    ) -> OverloadBiasActorInputs:
        n_red = len(state.red)
        n_blue = len(state.blue)
        slots = self._target_slot_vector(target_slots, n_red, n_blue)
        delayed = self._last_delayed_state if self._last_delayed_state is not None else state
        assignment_counts = np.bincount(
            np.clip(slots - 1, -1, n_blue - 1)[slots > 0],
            minlength=n_blue,
        ) if n_blue > 0 else np.zeros(0, dtype=np.int64)

        self_state = np.zeros((1, n_red, 20), dtype=np.float32)
        friends = np.zeros((1, n_red, n_red, 14), dtype=np.float32)
        friend_mask = np.zeros((1, n_red, n_red), dtype=bool)
        assigned_target = np.zeros((1, n_red, 1, 17), dtype=np.float32)
        target_mask = np.zeros((1, n_red, 1), dtype=bool)
        context = np.zeros((1, n_red, 4), dtype=np.float32)
        agent_mask = np.array([[red.alive for red in state.red]], dtype=bool)

        for red_index, red in enumerate(state.red):
            target_slot = int(slots[red_index])
            target_changed = target_slot != (
                red.current_target_index + 1 if 0 <= red.current_target_index < n_blue else 0
            )
            self_state[0, red_index] = self._execution_self_features(red, target_changed)
            target_index = target_slot - 1
            target = state.blue[target_index] if 0 <= target_index < n_blue else None
            for friend_index, delayed_friend in enumerate(delayed.red):
                if red_index == friend_index:
                    continue
                same_live_target = (
                    red.alive
                    and state.red[friend_index].alive
                    and target_slot > 0
                    and int(slots[friend_index]) == target_slot
                )
                if not same_live_target:
                    continue
                friend_mask[0, red_index, friend_index] = True
                friends[0, red_index, friend_index] = self._execution_friend_features(
                    red,
                    delayed_friend,
                    target,
                )

            confidence = 0.0
            observed_target: ThreeDoFState | None = None
            if target is not None and red.alive and target.alive:
                if self._detected(red, target, target_index):
                    observed_target = self._noisy_state(target)
                    confidence = 1.0
                elif red.target_estimate_valid and (
                    (
                        red.target_estimate_target_index
                        if red.target_estimate_target_index >= 0
                        else red.current_target_index
                    )
                    == target_index
                ):
                    observed_target = target.copy()
                    observed_target.position_m = red.target_estimate_position_m.copy()
                    observed_target.velocity_mps = red.target_estimate_velocity_mps.copy()
                    confidence = self._track_confidence(red, target_index)
                target_mask[0, red_index, 0] = True
                assigned_target[0, red_index, 0] = self._execution_target_features(
                    red,
                    observed_target,
                    target_alive=True,
                    confidence=confidence,
                )

            same_target_count = int(np.count_nonzero(slots == target_slot)) - 1 if target_slot > 0 else 0
            assignment_age = float(state.parameters.get("assignment_age_s", 0.0))
            context[0, red_index] = self._clip(
                np.array(
                    [
                        self._remaining_global_fraction(state),
                        max(same_target_count, 0) / self.config.scenario.max_missiles_per_target,
                        (
                            assignment_counts[target_index]
                            / self.config.scenario.max_missiles_per_target
                            if 0 <= target_index < n_blue
                            else 0.0
                        ),
                        assignment_age / max(self.config.assignment_update_interval_s, 1.0e-9),
                    ],
                    dtype=np.float64,
                )
            )

        hidden_tensor = None if hidden is None else hidden.to(self.device)
        return OverloadBiasActorInputs(
            self_state=self._tensor(self_state),
            same_target_friends=self._tensor(friends),
            friend_mask=self._tensor(friend_mask, dtype=torch.bool),
            assigned_target=self._tensor(assigned_target),
            target_mask=self._tensor(target_mask, dtype=torch.bool),
            environment_context=self._tensor(context),
            agent_mask=self._tensor(agent_mask, dtype=torch.bool),
            hidden=hidden_tensor,
        )

    @staticmethod
    def _target_slot_vector(target_slots: Tensor | np.ndarray, n_red: int, n_blue: int) -> np.ndarray:
        values = target_slots.detach().cpu().numpy() if isinstance(target_slots, Tensor) else np.asarray(target_slots)
        values = np.asarray(values).reshape(-1)
        if values.shape != (n_red,):
            raise ValueError(f"target_slots must contain exactly {n_red} entries")
        if not np.issubdtype(values.dtype, np.integer):
            if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
                raise ValueError("target_slots must contain finite integers")
        slots = values.astype(np.int64)
        if ((slots < 0) | (slots > n_blue)).any():
            raise ValueError(f"target_slots values must be in [0, {n_blue}]")
        return slots

    def _delayed_state(self, state: EngagementState) -> EngagementState:
        if self._history is None:
            return state
        self.advance(state)
        if len(self._history) <= self.config.sensor.communication_delay_steps:
            return state
        return self._history[0]

    def _tensor(self, value: np.ndarray, dtype: torch.dtype = torch.float32) -> Tensor:
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    @staticmethod
    def _clip(value: np.ndarray) -> np.ndarray:
        return np.clip(value, -FEATURE_CLIP, FEATURE_CLIP).astype(np.float32)

    def _detected(self, red: ThreeDoFState, blue: ThreeDoFState, blue_index: int) -> bool:
        return seeker_target_visible(
            red,
            blue,
            blue_index,
            self.config.missile,
            detection_range_m=self.config.sensor.detection_range_m,
        )

    def _assignment_observed_target(
        self,
        red: ThreeDoFState,
        blue: ThreeDoFState,
        blue_index: int,
    ) -> tuple[ThreeDoFState | None, float]:
        if not blue.alive:
            return None, 0.0
        if self._detected(red, blue, blue_index):
            return self._noisy_state(blue), 1.0
        estimate_index = (
            red.target_estimate_target_index
            if red.target_estimate_target_index >= 0
            else red.current_target_index
        )
        if (
            red.current_target_index != blue_index
            or not red.target_estimate_valid
            or estimate_index != blue_index
        ):
            return None, 0.0
        predicted = blue.copy()
        predicted.position_m = red.target_estimate_position_m.copy()
        predicted.velocity_mps = red.target_estimate_velocity_mps.copy()
        return predicted, self._track_confidence(red, blue_index)

    def _noisy_state(self, state: ThreeDoFState) -> ThreeDoFState:
        noisy = state.copy()
        if self.config.sensor.position_noise_m > 0.0:
            noisy.position_m += self._rng.normal(0.0, self.config.sensor.position_noise_m, 3)
        if self.config.sensor.velocity_noise_mps > 0.0:
            noisy.velocity_mps += self._rng.normal(0.0, self.config.sensor.velocity_noise_mps, 3)
        return noisy

    def _mach(self, entity: ThreeDoFState) -> float:
        return norm(entity.velocity_mps) / max(speed_of_sound(float(entity.position_m[1])), 1.0)

    def _remaining_guidance(self, missile: ThreeDoFState) -> float:
        if not missile.alive:
            return 0.0
        return float(
            np.clip(
                max(self.config.missile.max_guidance_time_s - missile.age_s, 0.0)
                / max(self.config.remaining_guidance_horizon_s, self.config.time_step_s),
                0.0,
                1.0,
            )
        )

    def _available_load(self, missile: ThreeDoFState) -> float:
        if not missile.alive:
            return 0.0
        used = norm(
            np.asarray(missile.pn_load_body_g, dtype=np.float64)[1:]
            + np.asarray(missile.gravity_load_body_g, dtype=np.float64)[1:]
            + np.asarray(missile.bias_load_body_g, dtype=np.float64)[1:]
        )
        return float(np.clip(1.0 - used / self.config.missile.max_load_factor_g, 0.0, 1.0))

    def _track_confidence(
        self,
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
        horizon = max(self.config.missile.fov_break_hold_s, self.config.time_step_s)
        decay = float(np.clip(1.0 - missile.target_estimate_age_s / horizon, 0.0, 1.0))
        if missile.guidance_mode == "lock_hold" and missile.target_estimate_valid:
            return decay
        if missile.target_estimate_valid:
            return 0.25 * decay
        return 0.0

    def _red_self_features(self, red: ThreeDoFState) -> np.ndarray:
        return self._clip(
            np.concatenate(
                [
                    red.position_m / POSITION_SCALE,
                    red.velocity_mps / VELOCITY_SCALE_MPS,
                    np.array(
                        [
                            self._mach(red),
                            self._available_load(red),
                            float(red.alive),
                            self._remaining_guidance(red),
                            self._track_confidence(red),
                            red.target_estimate_age_s / max(self.config.missile.fov_break_hold_s, 1.0e-9),
                            norm(red.bias_load_body_g[1:]) / max(self.config.missile.max_guidance_bias_g, 1.0e-9),
                        ]
                    ),
                ]
            )
        )

    def _friend_features(self, observer: ThreeDoFState, friend: ThreeDoFState) -> np.ndarray:
        frame = velocity_local_frame(observer.velocity_mps)
        relative_position = frame.T @ (friend.position_m - observer.position_m)
        relative_velocity = frame.T @ (friend.velocity_mps - observer.velocity_mps)
        return self._clip(
            np.concatenate(
                [
                    relative_position / RELATIVE_POSITION_SCALE,
                    relative_velocity / VELOCITY_SCALE_MPS,
                    np.array(
                        [
                            self._mach(friend),
                            self._available_load(friend),
                            float(friend.alive),
                            self._remaining_guidance(friend),
                            self._track_confidence(friend),
                        ]
                    ),
                ]
            )
        )

    def _target_features(
        self,
        observer: ThreeDoFState,
        blue: ThreeDoFState,
        confidence: float = 1.0,
    ) -> np.ndarray:
        frame = velocity_local_frame(observer.velocity_mps)
        relative_position = frame.T @ (blue.position_m - observer.position_m)
        relative_velocity = frame.T @ (blue.velocity_mps - observer.velocity_mps)
        return self._clip(
            np.concatenate(
                [
                    relative_position / RELATIVE_POSITION_SCALE,
                    relative_velocity / VELOCITY_SCALE_MPS,
                    np.array([float(blue.alive), confidence]),
                ]
            )
        )

    def _pair_features(
        self,
        missile: ThreeDoFState,
        target: ThreeDoFState,
        *,
        target_index: int,
        assignment_count: float,
        current_target: bool,
    ) -> np.ndarray:
        kinematics = los_kinematics(missile, target)
        frame = velocity_local_frame(missile.velocity_mps)
        local_los_rate = frame.T @ kinematics.los_rate_radps
        velocity_unit = unit(missile.velocity_mps, np.array([1.0, 0.0, 0.0]))
        off_boresight = math.acos(float(np.clip(np.dot(velocity_unit, kinematics.los_unit), -1.0, 1.0)))
        tgo = (
            kinematics.range_m / max(kinematics.closing_speed_mps, 1.0)
            if kinematics.closing_speed_mps > 0.0
            else self.config.missile.max_guidance_time_s * FEATURE_CLIP
        )
        rel_velocity_sq = float(np.dot(kinematics.relative_velocity_mps, kinematics.relative_velocity_mps))
        tau = -float(np.dot(kinematics.relative_position_m, kinematics.relative_velocity_mps)) / (
            rel_velocity_sq + 1.0e-9
        )
        zem = (
            norm(kinematics.relative_position_m + kinematics.relative_velocity_mps * tau)
            if tau > 0.0
            else kinematics.range_m
        )
        capacity = self.config.scenario.max_missiles_per_target
        return self._clip(
            np.array(
                [
                    kinematics.range_m / 200000.0,
                    kinematics.closing_speed_mps / VELOCITY_SCALE_MPS,
                    tgo / self.config.missile.max_guidance_time_s,
                    off_boresight / math.pi,
                    local_los_rate[1] / LOS_RATE_SCALE_RADPS,
                    local_los_rate[2] / LOS_RATE_SCALE_RADPS,
                    zem / ZEM_SCALE_M,
                    assignment_pair_quality(
                        self.config,
                        missile,
                        target,
                        target_index=target_index,
                    ),
                    float(current_target),
                    assignment_count / capacity,
                    max(capacity - assignment_count, 0.0) / capacity,
                ],
                dtype=np.float64,
            )
        )

    def _execution_self_features(self, red: ThreeDoFState, target_changed: bool) -> np.ndarray:
        modes = np.array(
            [
                float(red.guidance_mode == "locked"),
                float(red.guidance_mode == "lock_hold"),
                float(red.guidance_mode == "inertial"),
            ],
            dtype=np.float64,
        )
        total_lateral = (
            np.asarray(red.pn_load_body_g, dtype=np.float64)[1:]
            + np.asarray(red.gravity_load_body_g, dtype=np.float64)[1:]
            + np.asarray(red.bias_load_body_g, dtype=np.float64)[1:]
        )
        saturation_margin = np.clip(
            1.0 - norm(total_lateral) / self.config.missile.max_load_factor_g,
            0.0,
            1.0,
        )
        return self._clip(
            np.concatenate(
                [
                    red.position_m / POSITION_SCALE,
                    red.velocity_mps / VELOCITY_SCALE_MPS,
                    np.array(
                        [
                            self._mach(red),
                            self._available_load(red),
                            self._remaining_guidance(red),
                            float(red.alive),
                        ]
                    ),
                    modes,
                    np.array(
                        [red.target_estimate_age_s / max(self.config.missile.fov_break_hold_s, 1.0e-9)]
                    ),
                    np.asarray(red.pn_load_body_g, dtype=np.float64)[1:]
                    / self.config.missile.max_load_factor_g,
                    np.asarray(red.guidance_bias, dtype=np.float64),
                    np.array([saturation_margin, float(target_changed)]),
                ]
            )
        )

    def _execution_friend_features(
        self,
        observer: ThreeDoFState,
        friend: ThreeDoFState,
        target: ThreeDoFState | None,
    ) -> np.ndarray:
        frame = velocity_local_frame(observer.velocity_mps)
        relative_position = frame.T @ (friend.position_m - observer.position_m)
        relative_velocity = frame.T @ (friend.velocity_mps - observer.velocity_mps)
        tgo = 0.0
        zem = 0.0
        if target is not None:
            kin = los_kinematics(friend, target)
            tgo = kin.range_m / max(kin.closing_speed_mps, 1.0) / self.config.missile.max_guidance_time_s
            rv2 = float(np.dot(kin.relative_velocity_mps, kin.relative_velocity_mps))
            tau = -float(np.dot(kin.relative_position_m, kin.relative_velocity_mps)) / (rv2 + 1.0e-9)
            zem = (
                norm(kin.relative_position_m + kin.relative_velocity_mps * tau) / ZEM_SCALE_M
                if tau > 0.0
                else kin.range_m / ZEM_SCALE_M
            )
        return self._clip(
            np.concatenate(
                [
                    relative_position / RELATIVE_POSITION_SCALE,
                    relative_velocity / VELOCITY_SCALE_MPS,
                    np.array(
                        [
                            self._mach(friend),
                            self._remaining_guidance(friend),
                            self._available_load(friend),
                            float(friend.alive),
                            tgo,
                            zem,
                        ]
                    ),
                    np.asarray(friend.guidance_bias, dtype=np.float64),
                ]
            )
        )

    def _execution_target_features(
        self,
        observer: ThreeDoFState,
        target: ThreeDoFState | None,
        *,
        target_alive: bool,
        confidence: float,
    ) -> np.ndarray:
        if target is None:
            values = np.zeros(17, dtype=np.float64)
            values[-1] = float(target_alive)
            return self._clip(values)
        kin = los_kinematics(observer, target)
        frame = velocity_local_frame(observer.velocity_mps)
        relative_position = frame.T @ kin.relative_position_m
        relative_velocity = frame.T @ kin.relative_velocity_mps
        local_los = frame.T @ kin.los_unit
        local_los_rate = frame.T @ kin.los_rate_radps
        tgo = kin.range_m / max(kin.closing_speed_mps, 1.0)
        rv2 = float(np.dot(kin.relative_velocity_mps, kin.relative_velocity_mps))
        tau = -float(np.dot(kin.relative_position_m, kin.relative_velocity_mps)) / (rv2 + 1.0e-9)
        zem = (
            norm(kin.relative_position_m + kin.relative_velocity_mps * tau)
            if tau > 0.0
            else kin.range_m
        )
        return self._clip(
            np.concatenate(
                [
                    relative_position / RELATIVE_POSITION_SCALE,
                    relative_velocity / VELOCITY_SCALE_MPS,
                    local_los,
                    local_los_rate[1:] / LOS_RATE_SCALE_RADPS,
                    np.array(
                        [
                            kin.range_m / 200000.0,
                            kin.closing_speed_mps / VELOCITY_SCALE_MPS,
                            tgo / self.config.missile.max_guidance_time_s,
                            zem / ZEM_SCALE_M,
                            confidence,
                            float(target_alive),
                        ]
                    ),
                ]
            )
        )

    def _critic_inputs(
        self,
        state: EngagementState,
        assignment_counts: np.ndarray,
        unassigned: float,
    ) -> tuple[AssignmentCriticInputs, OverloadBiasCriticInputs]:
        n_red = len(state.red)
        n_blue = len(state.blue)
        global_red = np.zeros((1, n_red, 15), dtype=np.float32)
        red_mask = np.ones((1, n_red), dtype=bool)
        global_blue = np.zeros((1, n_blue, 8), dtype=np.float32)
        blue_mask = np.ones((1, n_blue), dtype=bool)
        applied_bias = np.zeros((1, n_red, 2), dtype=np.float32)
        pair_state = np.zeros((1, n_red, n_blue, 11), dtype=np.float32)
        current_assignment = self._assignment_matrix(state).reshape(1, n_red, n_blue)
        for red_index, red in enumerate(state.red):
            global_red[0, red_index] = self._global_red_features(red)
            if red.alive:
                applied_bias[0, red_index] = np.asarray(red.guidance_bias, dtype=np.float32)
            for blue_index, blue in enumerate(state.blue):
                pair_state[0, red_index, blue_index] = self._pair_features(
                    red,
                    blue,
                    target_index=blue_index,
                    assignment_count=float(assignment_counts[blue_index]),
                    current_target=red.current_target_index == blue_index,
                )
        for blue_index, blue in enumerate(state.blue):
            global_blue[0, blue_index] = self._global_blue_features(blue)
        red_scale = float(max(n_red, 1))
        blue_scale = float(max(n_blue, 1))
        context = self._clip(
            np.array(
                [
                    self._remaining_global_fraction(state),
                    sum(red.alive for red in state.red) / red_scale,
                    sum(blue.alive for blue in state.blue) / blue_scale,
                    mission_completion(state, n_blue),
                    ineffective_loss_rate(state, n_red),
                    unassigned / red_scale,
                    math.log1p(n_red) / math.log1p(24.0),
                    math.log1p(n_blue) / math.log1p(6.0),
                ],
                dtype=np.float64,
            )
        ).reshape(1, 8)
        shared = {
            "global_red": self._tensor(global_red),
            "red_mask": self._tensor(red_mask, dtype=torch.bool),
            "global_blue": self._tensor(global_blue),
            "blue_mask": self._tensor(blue_mask, dtype=torch.bool),
            "global_context": self._tensor(context),
            "pair_state": self._tensor(pair_state),
            "current_assignment": self._tensor(current_assignment),
        }
        return (
            AssignmentCriticInputs(
                **shared,
                target_assignment_counts=self._tensor(
                    (assignment_counts / self.config.scenario.max_missiles_per_target).reshape(1, n_blue)
                ),
            ),
            OverloadBiasCriticInputs(
                **shared,
                applied_bias=self._tensor(applied_bias),
            ),
        )

    def _execution_critic_input(self, state: EngagementState) -> OverloadBiasCriticInputs:
        matrix = self._assignment_matrix(state)
        counts = matrix.sum(axis=0)
        alive_red = sum(red.alive for red in state.red)
        unassigned = max(0.0, float(alive_red) - float(matrix.sum()))
        return self._critic_inputs(state, counts, unassigned)[1]

    def _global_red_features(self, red: ThreeDoFState) -> np.ndarray:
        return self._clip(
            np.concatenate(
                [
                    red.position_m / POSITION_SCALE,
                    red.velocity_mps / VELOCITY_SCALE_MPS,
                    np.array(
                        [
                            self._mach(red),
                            self._available_load(red),
                            float(red.alive),
                            self._remaining_guidance(red),
                            self._track_confidence(red),
                            red.target_estimate_age_s / max(self.config.missile.fov_break_hold_s, 1.0e-9),
                        ]
                    ),
                    np.asarray(red.guidance_bias, dtype=np.float64),
                    np.array(
                        [norm(red.final_load_body_g[1:]) / self.config.missile.max_load_factor_g]
                    ),
                ]
            )
        )

    def _global_blue_features(self, blue: ThreeDoFState) -> np.ndarray:
        return self._clip(
            np.concatenate(
                [
                    blue.position_m / POSITION_SCALE,
                    blue.velocity_mps / VELOCITY_SCALE_MPS,
                    np.array([float(blue.alive), float(np.clip(blue.energy, 0.0, 1.0))]),
                ]
            )
        )

    def _remaining_global_fraction(self, state: EngagementState) -> float:
        return float(
            np.clip(
                (self.config.max_steps - state.step_count)
                / max(self.config.policy_horizon_steps, 1),
                0.0,
                1.0,
            )
        )

    @staticmethod
    def _assignment_matrix(state: EngagementState) -> np.ndarray:
        matrix = np.zeros((len(state.red), len(state.blue)), dtype=np.float32)
        for red_index, red in enumerate(state.red):
            target_index = int(red.current_target_index)
            if red.alive and 0 <= target_index < len(state.blue) and state.blue[target_index].alive:
                matrix[red_index, target_index] = 1.0
        return matrix
