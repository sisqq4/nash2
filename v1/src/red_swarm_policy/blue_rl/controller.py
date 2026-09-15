from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from ..env.actions import BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G as ACTIONS
from ..env.types import EngagementState, EnvironmentConfig
from .environment import BlueEscapeEnvConfig, blue_action_context, blue_observation_dim
from .flight_envelope import FlightEnvelopeConfig, FlightEnvelopeConstraintLayer
from .mechanism_reward import (
    BlueMechanismStateEstimator,
    MechanismRewardConfig,
    encode_normalized_v4,
)
from .policy import DiscreteBluePolicy


@dataclass
class BlueRLDecision:
    step_count: int
    time_s: float
    action_indices: np.ndarray
    modes: tuple[str, ...]
    diagnostics: dict[int, dict[str, object]]

    def output_record(self) -> dict[str, object]:
        return {
            "event": "blue_rl_decision",
            "step_count": self.step_count,
            "time_s": self.time_s,
            "action_indices": self.action_indices.tolist(),
            "modes": list(self.modes),
            "diagnostics": self.diagnostics,
        }


@dataclass
class _AircraftMemory:
    constraint: FlightEnvelopeConstraintLayer
    estimator: BlueMechanismStateEstimator
    active: bool = False
    last_step: int | None = None
    action: int = 0
    start_command: np.ndarray = field(default_factory=lambda: ACTIONS[0].copy())
    slots: tuple[int, ...] = ()
    mechanism: dict[str, object] = field(default_factory=dict)


class BlueRLController:
    """Deploy a v1 blue policy with v7's per-aircraft runtime contract.

    Each live blue aircraft keeps independent threat-estimator and constraint
    state. Policy observations for all aircraft due at the same physics frame
    are inferred as one batch when the policy exposes ``expected_action_values``.
    The returned continuous commands interpolate over the same decision period
    used by :class:`BlueEscapeEnv`, avoiding a train/deployment actuator mismatch.
    """

    def __init__(
        self,
        policy: DiscreteBluePolicy,
        environment: EnvironmentConfig,
        config: BlueEscapeEnvConfig = BlueEscapeEnvConfig(),
    ) -> None:
        config.validate(environment)
        self.policy = policy
        self.environment = environment
        self.config = config

        ratio = config.decision_interval_s / environment.time_step_s
        if (
            not math.isfinite(ratio)
            or ratio < 1.0
            or not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1.0e-9)
        ):
            raise ValueError(
                "blue decision interval must be a positive multiple of the physics step"
            )
        self.decision_steps = int(round(ratio))

        policy_config = getattr(policy, "config", None)
        self.schema = str(
            getattr(policy_config, "observation_schema", config.observation_schema)
        )
        observation_dim = getattr(policy_config, "observation_dim", None)
        if observation_dim is None:
            self.missile_slots = (
                config.max_missiles
                if config.pad_observation_to_max_missiles
                else config.missile_count
            )
            self.observation_dim = blue_observation_dim(
                self.schema, self.missile_slots
            )
        else:
            self.observation_dim = int(observation_dim)
            slots = [
                count
                for count in range(1, 5)
                if blue_observation_dim(self.schema, count) == self.observation_dim
            ]
            if len(slots) != 1:
                raise ValueError(
                    "checkpoint is incompatible with the v1 1-4 slot observation contract"
                )
            self.missile_slots = slots[0]
        action_dim = getattr(policy_config, "action_dim", len(ACTIONS))
        if int(action_dim) != len(ACTIONS):
            raise ValueError("checkpoint is incompatible with the v1 29 action contract")

        saved_envelope = getattr(policy_config, "flight_envelope_config", None)
        self.envelope_config = (
            FlightEnvelopeConfig(**saved_envelope)
            if saved_envelope
            else FlightEnvelopeConfig(
                action_prediction_s=config.decision_interval_s
            )
        )
        if not math.isclose(
            self.envelope_config.action_prediction_s,
            config.decision_interval_s,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "controller decision interval must match the checkpoint flight envelope"
            )
        saved_mechanism = getattr(policy_config, "mechanism_reward_config", None)
        self.mechanism_config = (
            MechanismRewardConfig(**saved_mechanism)
            if saved_mechanism
            else config.mechanism_reward
        )
        self.reset()

    def reset(self) -> None:
        self._memories: dict[int, _AircraftMemory] = {}
        self._last_seen_step: int | None = None
        self.last_decision: BlueRLDecision | None = None
        self.last_action_diagnostic: dict[str, object] = {}
        self._decision_count = 0
        self._aircraft_decisions = 0

    def _memory(self, index: int) -> _AircraftMemory:
        if index not in self._memories:
            self._memories[index] = _AircraftMemory(
                FlightEnvelopeConstraintLayer(self.envelope_config),
                BlueMechanismStateEstimator(self.mechanism_config),
            )
        return self._memories[index]

    @property
    def constraint_layer(self) -> FlightEnvelopeConstraintLayer:
        """Compatibility alias for the first blue aircraft's constraint state."""
        return self._memory(0).constraint

    @property
    def mechanism_estimator(self) -> BlueMechanismStateEstimator:
        """Compatibility alias for the first blue aircraft's estimator state."""
        return self._memory(0).estimator

    def _command(
        self, memory: _AircraftMemory, step: int, *, next_frame: bool
    ) -> np.ndarray:
        if memory.last_step is None:
            return ACTIONS[0].copy()
        fraction = np.clip(
            (step - memory.last_step + int(next_frame)) / self.decision_steps,
            0.0,
            1.0,
        )
        return FlightEnvelopeConstraintLayer._interpolated_commands(
            memory.start_command,
            ACTIONS[memory.action],
            float(fraction),
        )

    def _visible(self, state: EngagementState, blue_index: int) -> tuple[int, ...]:
        blue = state.blue[blue_index]
        distances = [
            (float(np.linalg.norm(red.position_m - blue.position_m)), index)
            for index, red in enumerate(state.red)
            if red.alive
        ]
        nearest = sorted(
            (distance, index)
            for distance, index in distances
            if distance < self.config.threat_detection_range_m
        )
        # Select by distance, then restore scenario order as in the v1 trainer.
        return tuple(sorted(index for _, index in nearest[: self.missile_slots]))

    def _snapshot(
        self,
        state: EngagementState,
        index: int,
        slots: tuple[int, ...],
        command: np.ndarray,
    ) -> dict[str, object]:
        memory = self._memory(index)
        blue = state.blue[index]
        red = [state.red[red_index] for red_index in slots]
        aircraft = self.environment.aircraft
        return {
            "blue_position_m": blue.position_m.tolist(),
            "blue_velocity_mps": blue.velocity_mps.tolist(),
            "time_s": float(state.time_s),
            "red_positions_m": np.asarray(
                [missile.position_m for missile in red], dtype=np.float64
            ).reshape(-1, 3),
            "red_velocities_mps": np.asarray(
                [missile.velocity_mps for missile in red], dtype=np.float64
            ).reshape(-1, 3),
            "red_alive": [True] * len(red),
            "red_energy": [float(missile.energy) for missile in red],
            "red_guidance_modes": [missile.guidance_mode for missile in red],
            "min_altitude_m": aircraft.min_altitude_m,
            "max_altitude_m": aircraft.max_altitude_m,
            "min_speed_mps": aircraft.min_speed_mps,
            "max_speed_mps": aircraft.max_speed_mps,
            "max_load_factor_g": aircraft.max_load_factor_g,
            "previous_executed_action_index": memory.action,
            "actual_load_command_body_g": command.tolist(),
            "mechanism_emergency_gate": float(
                memory.mechanism.get("emergency_gate", 0.0)
            ),
        }

    def _encode(
        self, snapshot: dict[str, object], memory: _AircraftMemory
    ) -> np.ndarray:
        position = np.asarray(snapshot["blue_position_m"], dtype=np.float64)
        velocity = np.asarray(snapshot["blue_velocity_mps"], dtype=np.float64)
        if self.schema == "legacy_v1":
            values = [*(position / 1000.0), *(velocity / 1000.0)]
        else:
            values = [0.0, position[1] / 20000.0, 0.0, *(velocity / 2000.0)]
        if self.schema == "normalized_v4":
            values.extend(
                encode_normalized_v4(
                    snapshot, memory.mechanism, self.missile_slots
                )
            )
        else:
            red_positions = snapshot["red_positions_m"]
            for slot in range(self.missile_slots):
                present = slot < len(red_positions)
                relative = (
                    np.asarray(red_positions[slot], dtype=np.float64) - position
                    if present
                    else np.zeros(3, dtype=np.float64)
                )
                values.extend(
                    relative
                    / (1000.0 if self.schema == "legacy_v1" else 200000.0)
                )
                if self.schema != "legacy_v1":
                    values.append(float(present))
        if self.schema in ("normalized_v3", "normalized_v4"):
            values.extend(
                blue_action_context(
                    memory.action, snapshot["actual_load_command_body_g"]
                )
            )
        observation = np.asarray(values, dtype=np.float32)
        if observation.shape != (self.observation_dim,):
            raise RuntimeError(
                f"encoded blue observation has shape {observation.shape}, "
                f"expected ({self.observation_dim},)"
            )
        return observation

    @staticmethod
    def _remap_slots(memory: _AircraftMemory, slots: tuple[int, ...]) -> None:
        # Preserve filter history without allowing a slot swap to change identity.
        for name in ("primary_slot", "primary_candidate"):
            old_local = getattr(memory.estimator, name)
            entity = (
                memory.slots[old_local]
                if old_local is not None and old_local < len(memory.slots)
                else None
            )
            new_local = slots.index(entity) if entity in slots else None
            setattr(memory.estimator, name, new_local)
            if name == "primary_candidate" and new_local is None:
                memory.estimator.primary_candidate_count = 0
        memory.slots = slots

    def _policy_values(self, observations: np.ndarray) -> np.ndarray:
        action_values = getattr(self.policy, "expected_action_values", None)
        if callable(action_values):
            values = np.asarray(
                action_values(observations, evaluation=True), dtype=np.float64
            )
            if values.shape != (len(observations), len(ACTIONS)):
                raise ValueError(
                    "expected_action_values must return [batch, 29] values"
                )
            return values
        values = np.zeros((len(observations), len(ACTIONS)), dtype=np.float64)
        for row, observation in zip(values, observations):
            action = int(self.policy.select_action(observation, evaluation=True))
            if not 0 <= action < len(ACTIONS):
                raise ValueError("blue policy returned an action outside [0, 29)")
            row[action] = 1.0
        return values

    def action_for(
        self, state: EngagementState
    ) -> tuple[dict[str, np.ndarray], BlueRLDecision | None]:
        step = int(state.step_count)
        if self._last_seen_step is not None and step < self._last_seen_step:
            self.reset()
        self._last_seen_step = step

        pending: list[
            tuple[
                int,
                _AircraftMemory,
                dict[str, object],
                np.ndarray,
                np.ndarray,
            ]
        ] = []
        for index, blue in enumerate(state.blue):
            if not blue.alive:
                self._memories.pop(index, None)
                continue
            memory = self._memory(index)
            if (
                memory.last_step is not None
                and step - memory.last_step < self.decision_steps
            ):
                continue
            slots = self._visible(state, index)
            memory.active = memory.active or bool(slots)
            if not memory.active:
                continue
            self._remap_slots(memory, slots)
            command = self._command(memory, step, next_frame=False)
            snapshot = self._snapshot(state, index, slots, command)
            if self.schema == "normalized_v4":
                mask, _, _ = memory.constraint.constraints(snapshot)
                memory.mechanism = memory.estimator.observe(snapshot, mask)
                snapshot["mechanism_emergency_gate"] = float(
                    memory.mechanism["emergency_gate"]
                )
            pending.append(
                (index, memory, snapshot, command, self._encode(snapshot, memory))
            )

        diagnostics: dict[int, dict[str, object]] = {}
        if pending:
            observations = np.stack([row[4] for row in pending])
            q_values = self._policy_values(observations)
            for (index, memory, snapshot, command, _), values in zip(
                pending, q_values
            ):
                memory.action, diagnostic = memory.constraint.select(values, snapshot)
                memory.last_step = step
                memory.start_command = command.copy()
                diagnostics[index] = {
                    **diagnostic,
                    "red_indices": list(memory.slots),
                }
            self.last_action_diagnostic = dict(diagnostics.get(0, {}))
            self._decision_count += 1
            self._aircraft_decisions += len(pending)

        commands = np.repeat(ACTIONS[[0]], len(state.blue), axis=0)
        indices = np.zeros(len(state.blue), dtype=np.int64)
        modes: list[str] = []
        for index, blue in enumerate(state.blue):
            memory = self._memories.get(index)
            if memory is not None:
                commands[index] = self._command(memory, step, next_frame=True)
                indices[index] = memory.action
            modes.append(
                "inactive"
                if not blue.alive
                else "rainbow"
                if memory is not None and memory.active
                else "cruise"
            )

        decision = None
        if pending:
            decision = BlueRLDecision(
                step,
                float(state.time_s),
                indices.copy(),
                tuple(modes),
                diagnostics,
            )
            self.last_decision = decision
        # Continuous commands are authoritative; indices describe the target action.
        return {"load_command_body_g": commands}, decision

    def __call__(self, state: EngagementState) -> dict[str, np.ndarray]:
        return self.action_for(state)[0]

    def encode(
        self,
        state: EngagementState,
        *,
        blue_index: int = 0,
        snapshot: dict[str, object] | None = None,
        mechanism: dict[str, object] | None = None,
    ) -> np.ndarray:
        """Encode one aircraft for compatibility with earlier v1 callers."""
        if not 0 <= blue_index < len(state.blue):
            raise ValueError("blue_index is out of range")
        memory = self._memory(blue_index)
        slots = self._visible(state, blue_index)
        if slots != memory.slots:
            self._remap_slots(memory, slots)
        command = self._command(memory, int(state.step_count), next_frame=False)
        physical = (
            self._snapshot(state, blue_index, slots, command)
            if snapshot is None
            else snapshot
        )
        if mechanism is not None:
            memory.mechanism = mechanism
        elif self.schema == "normalized_v4" and not memory.mechanism:
            mask, _, _ = memory.constraint.constraints(physical)
            memory.mechanism = memory.estimator.observe(physical, mask)
        return self._encode(physical, memory)

    def runtime_statistics(self) -> dict[str, object]:
        policy_config = getattr(self.policy, "config", None)
        return {
            "policy": "rainbow",
            "device": str(getattr(policy_config, "device", "unknown")),
            "observation_schema": self.schema,
            "observation_dim": self.observation_dim,
            "missile_slots": self.missile_slots,
            "decision_count": self._decision_count,
            "aircraft_decisions": self._aircraft_decisions,
        }
