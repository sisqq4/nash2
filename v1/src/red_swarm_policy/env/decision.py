from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np

from ..policy.actor import PolicyOutput
from .actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G, blue_aircraft_load_commands_body_g
from .types import (
    BlueAction,
    EngagementState,
    EnvironmentConfig,
    EnvironmentObservation,
    JointAction,
    RedAction,
    los_kinematics,
)


def _coerce_three_column_matrix(value: Any, row_count: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.size == 0:
        return np.zeros((row_count, 3), dtype=np.float64)
    matrix = array.reshape(-1, array.shape[-1] if array.ndim > 1 else 1)
    if matrix.shape[0] < row_count:
        matrix = np.pad(matrix, ((0, row_count - matrix.shape[0]), (0, 0)))
    matrix = matrix[:row_count]
    if matrix.shape[1] < 3:
        matrix = np.pad(matrix, ((0, 0), (0, 3 - matrix.shape[1])))
    return matrix[:, :3].copy()


class IntelligentDecisionLayer:
    def __init__(
        self,
        config: EnvironmentConfig,
        red_policy: Callable[[EnvironmentObservation, EngagementState], Any] | None = None,
        blue_policy: Callable[[EngagementState], Any] | None = None,
        self_correction_model: Callable[[JointAction, EngagementState, EnvironmentObservation], JointAction] | None = None,
    ) -> None:
        self.config = config
        self.red_policy = red_policy
        self.blue_policy = blue_policy
        self.self_correction_model = self_correction_model

    def select_actions(self, state: EngagementState, observation: EnvironmentObservation, red_action: Any = None, blue_action: Any = None) -> JointAction:
        red_raw = self.red_policy(observation, state) if red_action is None and self.red_policy is not None else red_action
        blue_raw = self.blue_policy(state) if blue_action is None and self.blue_policy is not None else blue_action
        joint = JointAction(red=self._coerce_red_action(red_raw, state), blue=self._coerce_blue_action(blue_raw, state))
        if self.self_correction_model is None:
            return joint
        corrected = self.self_correction_model(joint, state, observation)
        if not isinstance(corrected, JointAction):
            raise TypeError("self_correction_model must return JointAction")
        return JointAction(
            red=self._validate_red_action(corrected.red, state),
            blue=self._coerce_blue_action(corrected.blue, state),
        )

    def select_blue_action(self, state: EngagementState, blue_action: Any = None) -> BlueAction:
        """Public blue-policy interface used during red policy warmup."""
        raw = self.blue_policy(state) if blue_action is None and self.blue_policy is not None else blue_action
        return self._coerce_blue_action(raw, state)

    def zero_red_action(self, state: EngagementState) -> RedAction:
        """Return the deterministic no-target/zero-bias warmup command."""
        return RedAction(
            target_indices=np.full(len(state.red), -1, dtype=np.int64),
            guidance_bias=np.zeros((len(state.red), 2), dtype=np.float64),
        )

    def _coerce_red_action(self, raw: Any, state: EngagementState) -> RedAction:
        n_red = len(state.red)
        if raw is None:
            return self._default_red_action(state)
        if isinstance(raw, JointAction):
            return self._validate_red_action(raw.red, state)
        if isinstance(raw, RedAction):
            return self._validate_red_action(raw, state)
        if isinstance(raw, PolicyOutput):
            target_slots = raw.assignment.actions.target.detach().cpu().numpy()
            guidance_bias = raw.execution.bias_matrix.detach().cpu().numpy()
            if target_slots.shape == (1, n_red):
                target_slots = target_slots[0]
            elif target_slots.shape != (n_red,):
                raise ValueError(
                    f"assignment target shape {target_slots.shape} must be {(1, n_red)} or {(n_red,)}"
                )
            if guidance_bias.shape == (1, n_red, 2):
                guidance_bias = guidance_bias[0]
            elif guidance_bias.shape != (n_red, 2):
                raise ValueError(
                    f"execution bias_matrix shape {guidance_bias.shape} must be {(1, n_red, 2)} or {(n_red, 2)}"
                )
            if raw.execution.action_distribution == "radial_tanh_disk":
                action_norm = np.linalg.norm(guidance_bias, axis=-1)
                if np.any(action_norm >= 1.0 + 1.0e-6):
                    raise ValueError(
                        "radial_tanh_disk policy produced an action outside the unit disk"
                    )
            return self._validate_red_action(
                RedAction(
                    target_indices=target_slots - 1,
                    guidance_bias=guidance_bias,
                ),
                state,
            )
        if isinstance(raw, dict):
            return self._validate_red_action(
                RedAction(
                    target_indices=np.asarray(raw.get("target_indices", np.full(n_red, -1))),
                    guidance_bias=np.asarray(raw.get("guidance_bias", np.zeros((n_red, 2)))),
                ),
                state,
            )
        raise TypeError("unsupported red action type")

    def _coerce_blue_action(self, raw: Any, state: EngagementState) -> BlueAction:
        n_blue = len(state.blue)
        if raw is None:
            return self._default_blue_action(state)
        if isinstance(raw, JointAction):
            raw = raw.blue
        if isinstance(raw, (int, np.integer)):
            return BlueAction(np.repeat(blue_aircraft_load_commands_body_g(np.array([raw])), n_blue, axis=0))
        if isinstance(raw, dict) and "action_indices" in raw:
            indices = np.asarray(raw["action_indices"], dtype=np.int64).reshape(-1)
            if indices.size == 1:
                indices = np.repeat(indices, n_blue)
            if indices.size < n_blue:
                indices = np.pad(indices, (0, n_blue - indices.size), constant_values=0)
            return BlueAction(blue_aircraft_load_commands_body_g(indices[:n_blue]))
        if not isinstance(raw, (BlueAction, dict)):
            raw_array = np.asarray(raw)
            if raw_array.ndim <= 1 and raw_array.size in (1, n_blue) and np.issubdtype(raw_array.dtype, np.integer):
                indices = raw_array.astype(np.int64).reshape(-1)
                if indices.size == 1:
                    indices = np.repeat(indices, n_blue)
                return BlueAction(blue_aircraft_load_commands_body_g(indices))
        load_command_raw = raw.load_command_body_g if isinstance(raw, BlueAction) else raw.get("load_command_body_g", np.zeros((n_blue, 3))) if isinstance(raw, dict) else raw
        commands = _coerce_three_column_matrix(load_command_raw, n_blue)
        commands[:, 0] = np.clip(commands[:, 0], -self.config.aircraft.max_load_factor_g, self.config.aircraft.max_load_factor_g)
        commands[:, 1] = np.clip(commands[:, 1], -self.config.aircraft.max_load_factor_g, self.config.aircraft.max_load_factor_g)
        commands[:, 2] = np.clip(commands[:, 2], -math.pi, math.pi)
        return BlueAction(commands)

    def _validate_red_action(self, action: RedAction, state: EngagementState) -> RedAction:
        n_red = len(state.red)
        n_blue = len(state.blue)
        targets = np.asarray(action.target_indices, dtype=np.int64)
        guidance_bias = np.asarray(action.guidance_bias, dtype=np.float64)
        if targets.shape != (n_red,):
            raise ValueError(f"target_indices shape {targets.shape} must be {(n_red,)}")
        if guidance_bias.shape != (n_red, 2):
            raise ValueError(f"guidance_bias shape {guidance_bias.shape} must be {(n_red, 2)}")
        if not np.all(np.isfinite(guidance_bias)):
            raise ValueError("guidance_bias must contain only finite values")
        if np.any((guidance_bias < -1.0) | (guidance_bias > 1.0)):
            raise ValueError("guidance_bias values must be in [-1, 1]")
        targets = np.where((targets >= 0) & (targets < n_blue), targets, -1)
        alive_mask = np.asarray([red.alive for red in state.red], dtype=bool)
        alive_targets = targets[alive_mask]
        counts = np.bincount(alive_targets[alive_targets >= 0], minlength=n_blue)
        capacity = self.config.scenario.max_missiles_per_target
        if np.any(counts > capacity):
            raise ValueError(f"each target may be assigned at most {capacity} alive red missiles")
        return RedAction(
            target_indices=targets,
            guidance_bias=guidance_bias,
        )

    def _default_red_action(self, state: EngagementState) -> RedAction:
        targets = np.full(len(state.red), -1, dtype=np.int64)
        counts = np.zeros(len(state.blue), dtype=np.int64)
        capacity = self.config.scenario.max_missiles_per_target
        for i, red in enumerate(state.red):
            ranges = [
                (j, los_kinematics(red, blue).range_m)
                for j, blue in enumerate(state.blue)
                if red.alive and blue.alive and counts[j] < capacity
            ]
            if ranges:
                targets[i] = min(ranges, key=lambda x: x[1])[0]
                counts[targets[i]] += 1
        return RedAction(
            target_indices=targets,
            guidance_bias=np.zeros((len(state.red), 2), dtype=np.float64),
        )

    def _default_blue_action(self, state: EngagementState) -> BlueAction:
        return BlueAction(np.repeat(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[[0]], len(state.blue), axis=0))
