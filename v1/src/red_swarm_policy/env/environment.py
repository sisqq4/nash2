from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import torch

from .adjudication import AdjudicationLayer
from .decision import IntelligentDecisionLayer
from .math_utils import norm
from .observation import ObservationLayer
from .physics import ThreeDoFPhysicsLayer
from .replay import ReplayBuffer
from .reward import HierarchicalRewardLayer, assignment_feasibility_potential, low_intercept_potential
from .scenario import ScenarioGenerator
from .types import (
    BlueAction,
    ControlDecisionRequest,
    EngagementState,
    EnvironmentConfig,
    EnvironmentObservation,
    EnvironmentStep,
    JointAction,
    PolicyStartMode,
    RedAction,
    ReplayTransition,
    ScenarioStyle,
)


class RedBlueEngagementEnv:
    def __init__(
        self,
        config: EnvironmentConfig = EnvironmentConfig(),
        red_policy: Callable[[EnvironmentObservation, EngagementState], Any] | None = None,
        blue_policy: Callable[[EngagementState], Any] | None = None,
        self_correction_model: Callable[[JointAction, EngagementState, EnvironmentObservation], JointAction] | None = None,
        device: torch.device | str | None = None,
        record_replay: bool = True,
    ) -> None:
        config.validate()
        self.config = config
        self.scenario_layer = ScenarioGenerator(config.scenario, config.missile, config.aircraft)
        self.physics_layer = ThreeDoFPhysicsLayer(config)
        self.observation_layer = ObservationLayer(config, device=device)
        self.decision_layer = IntelligentDecisionLayer(config, red_policy, blue_policy, self_correction_model)
        self.adjudication_layer = AdjudicationLayer(config)
        self.reward_layer = HierarchicalRewardLayer(config)
        self.replay_layer = ReplayBuffer()
        self.record_replay = bool(record_replay)
        self.state: EngagementState | None = None
        self.last_observation: EnvironmentObservation | None = None
        self.previous_action: JointAction | None = None
        self.start_mode: PolicyStartMode = config.policy_start_mode
        self._policy_entry_validated = False
        self._episode_done = False
        self._seed: int | None = None
        self._network_call_counts: dict[str, int] = {}
        self._first_network_call: dict[str, dict[str, float | int]] = {}
        self._assignment_redecision_due = True

    @property
    def policy_ready(self) -> bool:
        return self._policy_entry_validated and self.state is not None

    @property
    def policy_time_s(self) -> float:
        if self.state is None:
            return 0.0
        return max(0.0, self.state.time_s - self.config.policy_entry_time_s)

    @property
    def policy_remaining_fraction(self) -> float:
        if self.state is None:
            return 1.0
        return float(
            np.clip(
                (self.config.max_steps - self.state.step_count)
                / max(self.config.policy_horizon_steps, 1),
                0.0,
                1.0,
            )
        )

    def policy_status(self) -> dict[str, Any]:
        speeds = [] if self.state is None else [norm(red.velocity_mps) for red in self.state.red]
        flight_path_angles_deg = [] if self.state is None else [
            self._flight_path_angle_deg(red.velocity_mps) for red in self.state.red
        ]
        return {
            "ready": self.policy_ready,
            "start_mode": self.start_mode,
            "network_entry_time_s": self.config.policy_entry_time_s,
            "network_entry_step": self.config.policy_entry_steps,
            "policy_time_s": self.policy_time_s,
            "policy_remaining_fraction": self.policy_remaining_fraction,
            "entry_speed_tolerance_mps": self.config.policy_entry_speed_tolerance_mps,
            "entry_flight_path_tolerance_deg": self.config.policy_entry_flight_path_tolerance_deg,
            "red_speed_mps": [float(speed) for speed in speeds],
            "red_flight_path_angle_deg": flight_path_angles_deg,
            "guidance_modes": [] if self.state is None else [red.guidance_mode for red in self.state.red],
            "control_effort": self.control_effort,
            "network_call_counts": dict(self._network_call_counts),
            "first_network_call": dict(self._first_network_call),
        }

    @property
    def control_effort(self) -> float:
        return self.reward_layer.control_effort

    def next_decision_request(self) -> ControlDecisionRequest:
        """Return the exact schedule that ``step`` will use for its next action."""
        if self.state is None or not self.policy_ready:
            raise RuntimeError("policy decision request is unavailable before policy entry")
        initial = self.previous_action is None
        event_due = self._assignment_redecision_due
        periodic = self.state.step_count % self.config.assignment_update_steps == 0
        assignment_due = initial or event_due or periodic
        bias_due = (
            initial
            or assignment_due
            or self.state.step_count % self.config.bias_update_steps == 0
        )
        reason = (
            "initial"
            if initial
            else "event"
            if event_due
            else "periodic"
            if periodic
            else "held"
        )
        return ControlDecisionRequest(
            assignment_due=assignment_due,
            bias_due=bias_due,
            reason=reason,
        )

    def record_network_call(self, network_name: str) -> None:
        """Record an external Actor/Critic forward and reject pre-entry calls."""
        if not self.policy_ready or self.state is None:
            raise RuntimeError(f"network '{network_name}' called before post-boost policy entry")
        self._network_call_counts[network_name] = self._network_call_counts.get(network_name, 0) + 1
        self._first_network_call.setdefault(
            network_name,
            {"time_s": float(self.state.time_s), "step_count": int(self.state.step_count)},
        )

    def assignment_potential(self) -> float:
        if self.state is None:
            raise RuntimeError("environment state is unavailable")
        return assignment_feasibility_potential(
            self.config,
            self.state,
            initial_blue_count=self.reward_layer.initial_blue_count,
        )

    def execution_potential(self, red_index: int, blue_index: int) -> tuple[float, float, float, float]:
        if self.state is None:
            raise RuntimeError("environment state is unavailable")
        return low_intercept_potential(
            self.config,
            self.state.red[red_index],
            self.state.blue[blue_index],
            target_index=blue_index,
        )

    def reset(
        self,
        seed: int | None = None,
        style: ScenarioStyle | None = None,
        red_count: int | None = None,
        blue_count: int | None = None,
        *,
        start_mode: PolicyStartMode | None = None,
    ) -> EnvironmentObservation:
        mode = self.config.policy_start_mode if start_mode is None else start_mode
        if mode not in ("post_boost", "launch"):
            raise ValueError("start_mode must be 'post_boost' or 'launch'")
        self.replay_layer.clear()
        self.observation_layer.reset(seed)
        blue_policy = self.decision_layer.blue_policy
        reset_blue_policy = getattr(blue_policy, "reset", None)
        if callable(reset_blue_policy):
            reset_blue_policy()
        self.previous_action = None
        self._seed = seed
        self.start_mode = mode
        self._policy_entry_validated = False
        self._episode_done = False
        self._network_call_counts.clear()
        self._first_network_call.clear()
        self._assignment_redecision_due = True
        requested_red_count = self.config.scenario.red_count if red_count is None else int(red_count)
        requested_blue_count = self.config.scenario.blue_count if blue_count is None else int(blue_count)
        self.config.reward.validate_lexicographic_priority(
            requested_red_count,
            requested_blue_count,
        )
        self.state = self.scenario_layer.generate(
            seed=seed,
            style=style,
            red_count=red_count,
            blue_count=blue_count,
        )
        if mode == "post_boost":
            self._run_boost_warmup()
            self._validate_policy_entry()
            self.reward_layer.reset(self.state)
            self._policy_entry_validated = True
            self.observation_layer.reset(seed)
        self.last_observation = self.observation_layer.observe(self.state)
        return self.last_observation

    def step(self, red_action: Any = None, blue_action: Any = None) -> EnvironmentStep:
        if self.state is None or self.last_observation is None:
            self.reset()
        assert self.state is not None
        assert self.last_observation is not None
        if self._episode_done:
            raise RuntimeError("episode is done; call reset before stepping again")
        if not self.policy_ready:
            return self._step_boost_frame(blue_action)

        previous = self.state.copy()
        observation = self.last_observation
        action = self.decision_layer.select_actions(
            self.state,
            observation,
            red_action=red_action,
            blue_action=blue_action,
        )
        decision_request = self.next_decision_request()
        assignment_updated = decision_request.assignment_due
        bias_updated = decision_request.bias_due
        if self.previous_action is not None:
            self._hold_previous_red_action(
                action,
                self.previous_action,
                hold_assignment=not assignment_updated,
                hold_bias=not bias_updated,
            )
        if assignment_updated:
            self._assignment_redecision_due = False
        next_state = self.physics_layer.step(self.state, action)
        adjudication = self.adjudication_layer.evaluate(previous, next_state, action)
        self._write_current_targets(next_state, action)
        reward_high, reward_low, reward_info = self.reward_layer.evaluate_transition(
            previous,
            next_state,
            action,
            self.previous_action,
            adjudication.info,
            done=adjudication.done,
            bias_updated=bias_updated,
        )
        adjudication.reward_high = reward_high
        adjudication.reward_low = reward_low
        adjudication.info.update(reward_info)
        self._assignment_redecision_due = bool(
            reward_info["high_reward_settled"] and not adjudication.done
        )
        previous_assignment_age = float(previous.parameters.get("assignment_age_s", 0.0))
        next_state.parameters["assignment_age_s"] = (
            self.config.time_step_s
            if assignment_updated
            else previous_assignment_age + self.config.time_step_s
        )
        adjudication.info.update(
            {
                "policy_updated": bias_updated,
                "assignment_updated": assignment_updated,
                "assignment_redecision_required": self._assignment_redecision_due,
                "decision_reason": decision_request.reason,
                "bias_updated": bias_updated,
                "boost_warmup": False,
                "policy_time_s": max(0.0, next_state.time_s - self.config.policy_entry_time_s),
                "policy_remaining_fraction": float(
                    np.clip(
                        (self.config.max_steps - next_state.step_count)
                        / max(self.config.policy_horizon_steps, 1),
                        0.0,
                        1.0,
                    )
                ),
                "guidance_bias_matrix": action.red.guidance_bias.tolist(),
                "pn_load_body_g": np.stack([missile.pn_load_body_g for missile in next_state.red]).tolist(),
                "bias_load_body_g": np.stack([missile.bias_load_body_g for missile in next_state.red]).tolist(),
                "gravity_load_body_g": np.stack([missile.gravity_load_body_g for missile in next_state.red]).tolist(),
                "final_load_body_g": np.stack([missile.final_load_body_g for missile in next_state.red]).tolist(),
                "guidance_modes": [missile.guidance_mode for missile in next_state.red],
                "target_estimate_age_s": [float(missile.target_estimate_age_s) for missile in next_state.red],
                "red_alive": [bool(missile.alive) for missile in next_state.red],
                "blue_alive": [bool(target.alive) for target in next_state.blue],
                "current_target_indices": [int(missile.current_target_index) for missile in next_state.red],
            }
        )
        self.state = next_state
        self._episode_done = bool(adjudication.done)
        self.previous_action = self._copy_action(action)
        if reward_info["high_reward_settled"] or adjudication.done:
            next_observation = self.observation_layer.observe(
                next_state,
                previous=self.last_observation,
            )
            self.last_observation = next_observation
        elif next_state.step_count % self.config.bias_update_steps == 0:
            next_observation = self.observation_layer.observe_execution(
                next_state,
                self.last_observation,
            )
            self.last_observation = next_observation
        else:
            self.observation_layer.advance(next_state)
            next_observation = self.last_observation
        assignment_matrix = self._assignment_matrix(next_state)
        adjudication.info["target_assignment_matrix"] = assignment_matrix.astype(int).tolist()
        if self.record_replay:
            self.replay_layer.append(
                ReplayTransition(
                    state=previous,
                    action=action,
                    reward_high=adjudication.reward_high,
                    reward_low=adjudication.reward_low.copy(),
                    next_state=next_state.copy(),
                    observation=observation,
                    next_observation=next_observation,
                    info=dict(adjudication.info),
                )
            )
        return EnvironmentStep(
            observation=next_observation,
            reward_high=adjudication.reward_high,
            reward_low=adjudication.reward_low,
            done=adjudication.done,
            info=adjudication.info,
            assignment_matrix=assignment_matrix,
            terminated=adjudication.terminated,
            truncated=adjudication.truncated,
            termination_reason=adjudication.termination_reason,
        )

    def _run_boost_warmup(self) -> None:
        for _ in range(self.config.policy_entry_steps):
            self._advance_boost_without_observation()

    def _advance_boost_without_observation(self, blue_action: Any = None) -> None:
        assert self.state is not None
        red = self.decision_layer.zero_red_action(self.state)
        blue = self.decision_layer.select_blue_action(self.state, blue_action)
        action = JointAction(red=red, blue=blue)
        next_state = self.physics_layer.step(self.state, action)
        self._write_current_targets(next_state, action)
        self._assert_warmup_survival(next_state)
        self.state = next_state

    def _step_boost_frame(self, blue_action: Any = None) -> EnvironmentStep:
        assert self.state is not None
        assert self.last_observation is not None
        if self.state.step_count >= self.config.policy_entry_steps:
            self._validate_policy_entry()
            self.reward_layer.reset(self.state)
            self._policy_entry_validated = True
            self.previous_action = None
            self.last_observation = self.observation_layer.observe(self.state)
            raise RuntimeError("policy entry reached; call step again with a post-boost action")
        self._advance_boost_without_observation(blue_action)
        assert self.state is not None
        entry_now = self.state.step_count == self.config.policy_entry_steps
        if entry_now:
            self._validate_policy_entry()
            self.reward_layer.reset(self.state)
            self._policy_entry_validated = True
            self.previous_action = None
            self.observation_layer.reset(self._seed)
            self.last_observation = self.observation_layer.observe(self.state)
        info = {
            "hit_count": 0,
            "hit_pairs": [],
            "red_loss_events": [],
            "all_blue_done": False,
            "all_red_done": False,
            "timeout": False,
            "time_s": self.state.time_s,
            "step_count": self.state.step_count,
            "boost_warmup": True,
            "network_entry_reached": entry_now,
            "policy_updated": False,
            "assignment_updated": False,
            "assignment_redecision_required": False,
            "bias_updated": False,
            "high_reward_settled": False,
            "low_reward_settled": False,
            "policy_time_s": self.policy_time_s,
            "policy_remaining_fraction": self.policy_remaining_fraction,
            "guidance_bias_matrix": np.zeros((len(self.state.red), 2)).tolist(),
            "pn_load_body_g": np.zeros((len(self.state.red), 3)).tolist(),
            "bias_load_body_g": np.zeros((len(self.state.red), 3)).tolist(),
            "gravity_load_body_g": np.stack(
                [missile.gravity_load_body_g for missile in self.state.red]
            ).tolist(),
            "final_load_body_g": np.stack(
                [missile.final_load_body_g for missile in self.state.red]
            ).tolist(),
            "guidance_modes": [missile.guidance_mode for missile in self.state.red],
            "target_estimate_age_s": [float(missile.target_estimate_age_s) for missile in self.state.red],
        }
        assignment_matrix = self._assignment_matrix(self.state)
        return EnvironmentStep(
            observation=self.last_observation,
            reward_high=0.0,
            reward_low=np.zeros(len(self.state.red), dtype=np.float32),
            done=False,
            info=info,
            assignment_matrix=assignment_matrix,
        )

    def _validate_policy_entry(self) -> None:
        assert self.state is not None
        errors: list[str] = []
        tolerance = self.config.policy_entry_speed_tolerance_mps
        angle_tolerance_deg = self.config.policy_entry_flight_path_tolerance_deg
        validate_flight_path = (
            self.config.missile.boost_duration_s
            >= self.config.missile.boost_pitch_transition_s
        )
        for red_index, missile in enumerate(self.state.red):
            speed_error = abs(norm(missile.velocity_mps) - self.config.missile.max_speed_mps)
            angle_error_deg = abs(
                self._flight_path_angle_deg(missile.velocity_mps)
                - self.config.missile.boost_climb_angle_deg
            )
            if not missile.alive:
                errors.append(f"red[{red_index}] is not alive")
            if missile.age_s < self.config.policy_entry_time_s:
                errors.append(f"red[{red_index}] age_s={missile.age_s}")
            if speed_error > tolerance:
                errors.append(f"red[{red_index}] speed_error_mps={speed_error}")
            if validate_flight_path and angle_error_deg > angle_tolerance_deg:
                errors.append(f"red[{red_index}] flight_path_angle_error_deg={angle_error_deg}")
            if not math_isclose(missile.fuel_mass_kg, 0.0):
                errors.append(f"red[{red_index}] fuel_mass_kg={missile.fuel_mass_kg}")
            if not math_isclose(missile.mass_kg, self.config.missile.dry_mass_kg):
                errors.append(f"red[{red_index}] mass_kg={missile.mass_kg}")
        if self.state.step_count != self.config.policy_entry_steps:
            errors.append(f"step_count={self.state.step_count}")
        if not math_isclose(self.state.time_s, self.config.policy_entry_time_s):
            errors.append(f"time_s={self.state.time_s}")
        if errors:
            raise RuntimeError("invalid post-boost policy entry: " + "; ".join(errors))

    @staticmethod
    def _assert_warmup_survival(state: EngagementState) -> None:
        if any(not missile.alive for missile in state.red) or any(not target.alive for target in state.blue):
            raise RuntimeError(
                "scenario configuration error: red or blue entity was destroyed during boost warmup"
            )

    @staticmethod
    def _write_current_targets(state: EngagementState, action: JointAction) -> None:
        targets = np.asarray(action.red.target_indices, dtype=np.int64)
        if targets.shape != (len(state.red),):
            raise ValueError(f"target_indices shape {targets.shape} must be {(len(state.red),)}")
        for index, missile in enumerate(state.red):
            target_index = int(targets[index])
            missile.current_target_index = target_index if 0 <= target_index < len(state.blue) else -1

    @staticmethod
    def _assignment_matrix(state: EngagementState) -> np.ndarray:
        matrix = np.zeros((len(state.red), len(state.blue)), dtype=np.float32)
        for red_index, red in enumerate(state.red):
            target_index = int(red.current_target_index)
            if red.alive and 0 <= target_index < len(state.blue) and state.blue[target_index].alive:
                matrix[red_index, target_index] = 1.0
        return matrix

    @staticmethod
    def _hold_previous_red_action(
        action: JointAction,
        previous_action: JointAction,
        *,
        hold_assignment: bool,
        hold_bias: bool,
    ) -> None:
        if hold_assignment:
            action.red.target_indices = previous_action.red.target_indices.copy()
        if hold_bias:
            action.red.guidance_bias = previous_action.red.guidance_bias.copy()

    @staticmethod
    def _copy_action(action: JointAction) -> JointAction:
        return JointAction(
            red=RedAction(
                target_indices=action.red.target_indices.copy(),
                guidance_bias=action.red.guidance_bias.copy(),
            ),
            blue=BlueAction(load_command_body_g=action.blue.load_command_body_g.copy()),
        )

    @staticmethod
    def _flight_path_angle_deg(velocity_mps: np.ndarray) -> float:
        speed = max(norm(velocity_mps), 1.0e-9)
        return math.degrees(math.asin(float(np.clip(velocity_mps[1] / speed, -1.0, 1.0))))


def math_isclose(left: float, right: float) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=1.0e-9))
