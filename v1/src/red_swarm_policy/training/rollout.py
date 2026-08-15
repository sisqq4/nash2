from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from ..env import EnvironmentObservation, RedBlueEngagementEnv, ScenarioStyle, ineffective_loss_rate
from ..policy.actor import (
    AssignmentActions,
    AssignmentActorInputs,
    AssignmentPolicyOutput,
    OverloadBiasActor,
    OverloadBiasActorInputs,
    PolicyOutput,
    TargetAssignmentActor,
)
from ..policy.critic import (
    AssignmentCriticInputs,
    OverloadBiasCritic,
    OverloadBiasCriticInputs,
    TargetAssignmentCritic,
)
from .env_pool import ProcessEnvironmentPool, ThreadEnvironmentPool
from .mappo import MAPPOBatch


@dataclass(frozen=True)
class RolloutStats:
    steps: int
    execution_steps: int
    done: bool
    total_reward_high: float
    mean_reward_low: float
    final_info: dict[str, Any]
    per_env_final_info: tuple[dict[str, Any], ...] = ()
    active_low_reward_mean: float = 0.0
    active_low_reward_nonzero_rate: float = 0.0
    active_execution_agent_steps: int = 0
    episode_high_reward_mean: float = 0.0
    episode_low_return_mean: float = 0.0
    episode_hit_count_sum: int = 0
    episode_hit_count_mean: float = 0.0
    episode_miss_distance_mean: float | None = None
    episode_miss_distance_p95: float | None = None
    terminal_reason_counts: dict[str, int] = field(default_factory=dict)
    low_reward_component_sums: dict[str, float] = field(default_factory=dict)
    time_credit_unassigned_count: int = 0


@dataclass(frozen=True)
class ParallelEpisodeMetrics:
    full_success: float
    damage_rate: float
    ineffective_loss_rate: float
    completion_time_s: float
    control_effort: float


def _finite_miss_distances(final_infos: Sequence[dict[str, Any]]) -> list[float]:
    values: list[float] = []
    for info in final_infos:
        raw = info.get("miss_distance_m")
        candidates = raw if isinstance(raw, (list, tuple, np.ndarray)) else [raw]
        for candidate in candidates:
            if isinstance(candidate, (int, float, np.integer, np.floating)):
                value = float(candidate)
                if np.isfinite(value):
                    values.append(value)
    return values


def _build_rollout_stats(
    batch: MAPPOBatch,
    final_infos: Sequence[dict[str, Any]],
    *,
    done: bool,
    episode_hit_counts: Sequence[int] | None = None,
) -> RolloutStats:
    low_active = (
        batch.episode_active_low.bool().unsqueeze(-1)
        & batch.execution_actor_inputs.agent_mask.bool()
        & batch.execution_actor_inputs.target_mask.any(dim=-1)
    )
    active_rewards = batch.rewards_low[low_active]
    active_count = int(active_rewards.numel())
    red_counts = batch.scenario_red_counts
    if red_counts is None:
        red_counts = batch.execution_critic_inputs.red_mask[0].sum(dim=-1).long()
    per_env_low_return = batch.rewards_low.sum(dim=(0, 2)) / red_counts.clamp_min(1)
    hits = (
        [int(value) for value in episode_hit_counts]
        if episode_hit_counts is not None
        else [int(info.get("hit_count", 0)) for info in final_infos]
    )
    enriched_infos = []
    terminal_reason_counts: dict[str, int] = {}
    low_reward_component_sums: dict[str, float] = {}
    time_credit_unassigned_count = 0
    for info, hit_count in zip(final_infos, hits):
        enriched = dict(info)
        enriched["episode_hit_count"] = hit_count
        enriched_infos.append(enriched)
        reason = info.get("termination_reason")
        if reason is None:
            components = info.get("reward_components", {})
            if isinstance(components, dict):
                reason = components.get("terminal_reason")
        if reason:
            key = str(reason)
            terminal_reason_counts[key] = terminal_reason_counts.get(key, 0) + 1
        component_sums = info.get("episode_low_reward_component_sums", {})
        if isinstance(component_sums, dict):
            for name, raw in component_sums.items():
                if isinstance(raw, (int, float, np.integer, np.floating)):
                    value = float(raw)
                    if np.isfinite(value):
                        low_reward_component_sums[str(name)] = (
                            low_reward_component_sums.get(str(name), 0.0) + value
                        )
        time_credit_unassigned_count += int(
            info.get("episode_time_credit_unassigned_count", 0)
        )
    misses = _finite_miss_distances(final_infos)
    aggregate_info = dict(enriched_infos[0]) if enriched_infos else {}
    aggregate_info["parallel_env_count"] = len(enriched_infos)
    aggregate_info["per_env"] = enriched_infos
    total_high = float(batch.rewards_high.sum().detach().cpu())
    return RolloutStats(
        steps=batch.rewards_high.shape[0],
        execution_steps=batch.rewards_low.shape[0],
        done=done,
        total_reward_high=total_high,
        mean_reward_low=float(batch.rewards_low.mean().detach().cpu()),
        final_info=aggregate_info,
        per_env_final_info=tuple(enriched_infos),
        active_low_reward_mean=(
            float(active_rewards.mean().detach().cpu()) if active_count else 0.0
        ),
        active_low_reward_nonzero_rate=(
            float((active_rewards != 0).float().mean().detach().cpu())
            if active_count
            else 0.0
        ),
        active_execution_agent_steps=active_count,
        episode_high_reward_mean=total_high / max(len(enriched_infos), 1),
        episode_low_return_mean=float(per_env_low_return.mean().detach().cpu()),
        episode_hit_count_sum=sum(hits),
        episode_hit_count_mean=sum(hits) / max(len(hits), 1),
        episode_miss_distance_mean=float(np.mean(misses)) if misses else None,
        episode_miss_distance_p95=float(np.quantile(misses, 0.95)) if misses else None,
        terminal_reason_counts=terminal_reason_counts,
        low_reward_component_sums=low_reward_component_sums,
        time_credit_unassigned_count=time_credit_unassigned_count,
    )


def _validate_batch_scenario_padding(batch: MAPPOBatch) -> None:
    if batch.scenario_red_counts is None or batch.scenario_blue_counts is None:
        raise ValueError("MAPPOBatch requires scenario red/blue count metadata")
    batch_size = batch.rewards_high.shape[1]
    red_slots = batch.rewards_low.shape[2]
    if tuple(batch.scenario_red_counts.shape) != (batch_size,):
        raise ValueError("scenario_red_counts must have shape [B]")
    if tuple(batch.scenario_blue_counts.shape) != (batch_size,):
        raise ValueError("scenario_blue_counts must have shape [B]")
    if ((batch.scenario_red_counts <= 0) | (batch.scenario_red_counts > red_slots)).any():
        raise ValueError("scenario_red_counts are inconsistent with the padded red axis")
    red_index = torch.arange(red_slots, device=batch.rewards_low.device)
    padded = red_index.unsqueeze(0) >= batch.scenario_red_counts.unsqueeze(1)
    if not padded.any():
        return
    low_padded = padded.unsqueeze(0).expand(batch.rewards_low.shape[0], -1, -1)
    high_padded = padded.unsqueeze(0).expand(
        batch.assignment_actions.target.shape[0], -1, -1
    )
    critic_padded = padded.unsqueeze(0).expand(
        batch.execution_critic_inputs.red_mask.shape[0], -1, -1
    )
    checks = (
        not batch.execution_actor_inputs.agent_mask[low_padded].any(),
        not batch.assignment_actor_inputs.agent_mask[high_padded].any(),
        not batch.execution_critic_inputs.red_mask[critic_padded].any(),
        torch.count_nonzero(batch.bias_matrices[low_padded]) == 0,
        torch.count_nonzero(batch.old_execution_log_prob[low_padded]) == 0,
        torch.count_nonzero(batch.rewards_low[low_padded]) == 0,
        bool((batch.dones_low[low_padded] == 1).all()),
        torch.count_nonzero(batch.assignment_actions.target[high_padded]) == 0,
    )
    if not all(bool(check) for check in checks):
        raise RuntimeError("padded red rows violated MAPPO batch invariants")


def _stack_optional(values: list[Tensor | None]) -> Tensor | None:
    present = [value is not None for value in values]
    if not any(present):
        return None
    if not all(present):
        raise ValueError("optional rollout tensor must be present at every time step or none")
    return torch.stack([value.detach() for value in values if value is not None], dim=0)


def _stack_assignment_inputs(values: list[AssignmentActorInputs]) -> AssignmentActorInputs:
    return AssignmentActorInputs(
        self_state=torch.stack([x.self_state.detach() for x in values], dim=0),
        friend_entities=torch.stack([x.friend_entities.detach() for x in values], dim=0),
        friend_mask=torch.stack([x.friend_mask.detach() for x in values], dim=0),
        target_entities=torch.stack([x.target_entities.detach() for x in values], dim=0),
        pair_state=torch.stack([x.pair_state.detach() for x in values], dim=0),
        current_assignment=torch.stack([x.current_assignment.detach() for x in values], dim=0),
        target_mask=torch.stack([x.target_mask.detach() for x in values], dim=0),
        environment_context=torch.stack([x.environment_context.detach() for x in values], dim=0),
        target_assignment_counts=torch.stack(
            [x.target_assignment_counts.detach() for x in values],
            dim=0,
        ),
        target_entity_mask=torch.stack([x.target_entity_mask.detach() for x in values], dim=0),
        agent_mask=torch.stack([x.agent_mask.detach() for x in values], dim=0),
        hidden=_stack_optional([x.hidden for x in values]),
    )


def _stack_execution_inputs(values: list[OverloadBiasActorInputs]) -> OverloadBiasActorInputs:
    return OverloadBiasActorInputs(
        self_state=torch.stack([x.self_state.detach() for x in values], dim=0),
        same_target_friends=torch.stack([x.same_target_friends.detach() for x in values], dim=0),
        friend_mask=torch.stack([x.friend_mask.detach() for x in values], dim=0),
        assigned_target=torch.stack([x.assigned_target.detach() for x in values], dim=0),
        target_mask=torch.stack([x.target_mask.detach() for x in values], dim=0),
        environment_context=torch.stack([x.environment_context.detach() for x in values], dim=0),
        agent_mask=torch.stack([x.agent_mask.detach() for x in values], dim=0),
        hidden=_stack_optional([x.hidden for x in values]),
    )


def _stack_assignment_actions(values: list[AssignmentActions]) -> AssignmentActions:
    return AssignmentActions(
        target=torch.stack([x.target.detach() for x in values], dim=0),
        order=_stack_optional([x.order for x in values]),
    )


def _stack_assignment_critic_inputs(values: list[AssignmentCriticInputs]) -> AssignmentCriticInputs:
    return AssignmentCriticInputs(
        global_red=torch.stack([x.global_red.detach() for x in values], dim=0),
        red_mask=torch.stack([x.red_mask.detach() for x in values], dim=0),
        global_blue=torch.stack([x.global_blue.detach() for x in values], dim=0),
        blue_mask=torch.stack([x.blue_mask.detach() for x in values], dim=0),
        global_context=torch.stack([x.global_context.detach() for x in values], dim=0),
        target_assignment_counts=torch.stack(
            [x.target_assignment_counts.detach() for x in values],
            dim=0,
        ),
        pair_state=torch.stack([x.pair_state.detach() for x in values], dim=0),
        current_assignment=torch.stack([x.current_assignment.detach() for x in values], dim=0),
    )


def _stack_execution_critic_inputs(values: list[OverloadBiasCriticInputs]) -> OverloadBiasCriticInputs:
    return OverloadBiasCriticInputs(
        global_red=torch.stack([x.global_red.detach() for x in values], dim=0),
        red_mask=torch.stack([x.red_mask.detach() for x in values], dim=0),
        global_blue=torch.stack([x.global_blue.detach() for x in values], dim=0),
        blue_mask=torch.stack([x.blue_mask.detach() for x in values], dim=0),
        applied_bias=torch.stack([x.applied_bias.detach() for x in values], dim=0),
        global_context=torch.stack([x.global_context.detach() for x in values], dim=0),
        pair_state=torch.stack([x.pair_state.detach() for x in values], dim=0),
        current_assignment=torch.stack([x.current_assignment.detach() for x in values], dim=0),
        hidden=_stack_optional([x.hidden for x in values]),
    )


def _pad_axis(value: Tensor, axis: int, size: int, fill: float | bool | int = 0) -> Tensor:
    axis = axis if axis >= 0 else value.dim() + axis
    if axis < 0 or axis >= value.dim():
        raise ValueError("padding axis is out of range")
    if value.shape[axis] > size:
        raise ValueError("padding target is smaller than tensor width")
    if value.shape[axis] == size:
        return value
    shape = list(value.shape)
    shape[axis] = size
    result = value.new_full(shape, fill)
    slices = [slice(None)] * value.dim()
    slices[axis] = slice(0, value.shape[axis])
    result[tuple(slices)] = value
    return result


def _pad_axes(
    value: Tensor,
    sizes: dict[int, int],
    fill: float | bool | int = 0,
) -> Tensor:
    for axis, size in sizes.items():
        value = _pad_axis(value, axis, size, fill)
    return value


def _concat_assignment_inputs(values: list[AssignmentActorInputs]) -> AssignmentActorInputs:
    red_slots = max(value.self_state.shape[1] for value in values)
    target_slots = max(value.target_entities.shape[2] for value in values)

    return AssignmentActorInputs(
        self_state=torch.cat([_pad_axis(x.self_state, 1, red_slots) for x in values]),
        friend_entities=torch.cat(
            [_pad_axes(x.friend_entities, {1: red_slots, 2: red_slots}) for x in values]
        ),
        friend_mask=torch.cat(
            [_pad_axes(x.friend_mask, {1: red_slots, 2: red_slots}, False) for x in values]
        ),
        target_entities=torch.cat(
            [_pad_axes(x.target_entities, {1: red_slots, 2: target_slots}) for x in values]
        ),
        pair_state=torch.cat(
            [_pad_axes(x.pair_state, {1: red_slots, 2: target_slots}) for x in values]
        ),
        current_assignment=torch.cat(
            [_pad_axes(x.current_assignment, {1: red_slots, 2: target_slots}) for x in values]
        ),
        target_mask=torch.cat(
            [_pad_axes(x.target_mask, {1: red_slots, 2: target_slots}, False) for x in values]
        ),
        environment_context=torch.cat(
            [_pad_axis(x.environment_context, 1, red_slots) for x in values]
        ),
        target_assignment_counts=torch.cat(
            [_pad_axes(x.target_assignment_counts, {1: red_slots, 2: target_slots}) for x in values]
        ),
        target_entity_mask=torch.cat(
            [_pad_axes(x.target_entity_mask, {1: red_slots, 2: target_slots}, False) for x in values]
        ),
        agent_mask=torch.cat(
            [_pad_axis(x.agent_mask, 1, red_slots, False) for x in values]
        ),
        hidden=None,
    )


def _concat_execution_inputs(values: list[OverloadBiasActorInputs]) -> OverloadBiasActorInputs:
    red_slots = max(value.self_state.shape[1] for value in values)
    return OverloadBiasActorInputs(
        self_state=torch.cat([_pad_axis(x.self_state, 1, red_slots) for x in values]),
        same_target_friends=torch.cat(
            [_pad_axes(x.same_target_friends, {1: red_slots, 2: red_slots}) for x in values]
        ),
        friend_mask=torch.cat(
            [_pad_axes(x.friend_mask, {1: red_slots, 2: red_slots}, False) for x in values]
        ),
        assigned_target=torch.cat(
            [_pad_axis(x.assigned_target, 1, red_slots) for x in values]
        ),
        target_mask=torch.cat(
            [_pad_axis(x.target_mask, 1, red_slots, False) for x in values]
        ),
        environment_context=torch.cat(
            [_pad_axis(x.environment_context, 1, red_slots) for x in values]
        ),
        agent_mask=torch.cat(
            [_pad_axis(x.agent_mask, 1, red_slots, False) for x in values]
        ),
        hidden=None,
    )


def _concat_assignment_critic_inputs(values: list[AssignmentCriticInputs]) -> AssignmentCriticInputs:
    red_slots = max(value.global_red.shape[1] for value in values)
    blue_slots = max(value.global_blue.shape[1] for value in values)

    return AssignmentCriticInputs(
        global_red=torch.cat([_pad_axis(x.global_red, 1, red_slots) for x in values]),
        red_mask=torch.cat([_pad_axis(x.red_mask, 1, red_slots, False) for x in values]),
        global_blue=torch.cat([_pad_axis(x.global_blue, 1, blue_slots) for x in values]),
        blue_mask=torch.cat([_pad_axis(x.blue_mask, 1, blue_slots, False) for x in values]),
        global_context=torch.cat([x.global_context for x in values], dim=0),
        target_assignment_counts=torch.cat(
            [_pad_axis(x.target_assignment_counts, 1, blue_slots) for x in values]
        ),
        pair_state=torch.cat(
            [_pad_axes(x.pair_state, {1: red_slots, 2: blue_slots}) for x in values]
        ),
        current_assignment=torch.cat(
            [_pad_axes(x.current_assignment, {1: red_slots, 2: blue_slots}) for x in values]
        ),
    )


def _concat_execution_critic_inputs(values: list[OverloadBiasCriticInputs]) -> OverloadBiasCriticInputs:
    red_slots = max(value.global_red.shape[1] for value in values)
    blue_slots = max(value.global_blue.shape[1] for value in values)

    return OverloadBiasCriticInputs(
        global_red=torch.cat([_pad_axis(x.global_red, 1, red_slots) for x in values]),
        red_mask=torch.cat([_pad_axis(x.red_mask, 1, red_slots, False) for x in values]),
        global_blue=torch.cat([_pad_axis(x.global_blue, 1, blue_slots) for x in values]),
        blue_mask=torch.cat([_pad_axis(x.blue_mask, 1, blue_slots, False) for x in values]),
        applied_bias=torch.cat([_pad_axis(x.applied_bias, 1, red_slots) for x in values]),
        global_context=torch.cat([x.global_context for x in values], dim=0),
        pair_state=torch.cat(
            [_pad_axes(x.pair_state, {1: red_slots, 2: blue_slots}) for x in values]
        ),
        current_assignment=torch.cat(
            [_pad_axes(x.current_assignment, {1: red_slots, 2: blue_slots}) for x in values]
        ),
        hidden=None,
    )


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def _cpu_rollout_storage(value: Any) -> Any:
    """Detach a rollout snapshot and release accelerator storage promptly."""

    if isinstance(value, Tensor):
        return value.detach().to("cpu")
    return value.detached().to("cpu")


def _capacity_aware_assignment(
    inputs: AssignmentActorInputs,
    max_missiles_per_target: int,
) -> Tensor:
    batch, agent_count, target_count = inputs.target_mask.shape
    selected = torch.zeros(
        batch,
        agent_count,
        dtype=torch.long,
        device=inputs.target_mask.device,
    )
    for batch_index in range(batch):
        counts = torch.zeros(
            target_count - 1,
            dtype=torch.long,
            device=inputs.target_mask.device,
        )
        for agent_index in range(agent_count):
            if not inputs.agent_mask[batch_index, agent_index]:
                continue
            valid = inputs.target_mask[batch_index, agent_index, 1:].bool()
            valid &= counts < max_missiles_per_target
            if not valid.any():
                continue
            quality = inputs.pair_state[batch_index, agent_index, 1:, 7]
            coverage_bonus = (counts == 0).to(dtype=quality.dtype)
            load_penalty = counts.to(dtype=quality.dtype) / max_missiles_per_target
            score = quality + coverage_bonus - 0.25 * load_penalty
            score = score.masked_fill(~valid, torch.finfo(score.dtype).min)
            target_index = int(score.argmax().item())
            selected[batch_index, agent_index] = target_index + 1
            counts[target_index] += 1
    return selected


def _capacity_aware_assignment_output(
    inputs: AssignmentActorInputs,
    max_missiles_per_target: int,
    d_model: int,
) -> AssignmentPolicyOutput:
    target = _capacity_aware_assignment(inputs, max_missiles_per_target)
    probabilities = torch.nn.functional.one_hot(
        target,
        num_classes=inputs.target_mask.shape[-1],
    ).to(dtype=inputs.self_state.dtype)
    hidden = inputs.hidden
    if hidden is None:
        hidden = torch.zeros(
            (*inputs.self_state.shape[:2], d_model),
            dtype=inputs.self_state.dtype,
            device=inputs.self_state.device,
        )
    conditional = torch.zeros_like(target, dtype=inputs.self_state.dtype)
    order = torch.arange(
        target.shape[1],
        dtype=torch.long,
        device=target.device,
    ).expand(target.shape[0], -1)
    return AssignmentPolicyOutput(
        actions=AssignmentActions(target=target, order=order),
        next_hidden=hidden,
        log_prob=conditional,
        entropy=conditional,
        joint_log_prob=conditional.sum(dim=-1),
        joint_entropy=conditional.sum(dim=-1),
        target_probabilities=probabilities,
        assignment_matrix=probabilities[..., 1:],
    )


class HierarchicalPolicyRuntime:
    """Single-environment controller driven only by ``next_decision_request``.

    Checkpoint validation and scenario visualization use this runtime directly.
    Batched training uses the same environment request semantics while retaining
    batched network execution in its collector.
    """

    def __init__(
        self,
        env: RedBlueEngagementEnv,
        assignment_actor: TargetAssignmentActor,
        execution_actor: OverloadBiasActor,
        *,
        deterministic: bool = True,
        assignment_mode: str = "actor",
        assignment_deterministic: bool | None = None,
        execution_deterministic: bool | None = None,
    ) -> None:
        if assignment_mode not in {"actor", "capacity_aware"}:
            raise ValueError("assignment_mode must be actor or capacity_aware")
        device = _module_device(assignment_actor)
        if _module_device(execution_actor) != device:
            raise ValueError("assignment and execution actors must use the same device")
        self.env = env
        self.assignment_actor = assignment_actor
        self.execution_actor = execution_actor
        self.device = device
        self.deterministic = deterministic
        self.assignment_deterministic = (
            deterministic
            if assignment_deterministic is None
            else assignment_deterministic
        )
        self.execution_deterministic = (
            deterministic
            if execution_deterministic is None
            else execution_deterministic
        )
        self.assignment_mode = assignment_mode
        self.assignment_hidden: Tensor | None = None
        self.execution_hidden: Tensor | None = None
        self.assignment_output: AssignmentPolicyOutput | None = None
        self.policy: PolicyOutput | None = None
        self._last_assignment_due = False

    def reset(self, observation: EnvironmentObservation) -> None:
        inputs = observation.assignment_actor_inputs.to(self.device)
        batch_size, agent_count = inputs.self_state.shape[:2]
        self.assignment_hidden = torch.zeros(
            batch_size,
            agent_count,
            self.assignment_actor.config.d_model,
            dtype=inputs.self_state.dtype,
            device=self.device,
        )
        self.execution_hidden = torch.zeros_like(self.assignment_hidden)
        self.assignment_output = None
        self.policy = None
        self._last_assignment_due = False

    def action(self, observation: EnvironmentObservation) -> tuple[PolicyOutput, Tensor]:
        if self.assignment_hidden is None or self.execution_hidden is None:
            raise RuntimeError("controller runtime must be reset before use")
        if self.env.state is None:
            raise RuntimeError("environment state is unavailable")
        request = self.env.next_decision_request()
        target_changed = torch.zeros(
            self.execution_hidden.shape[:2],
            dtype=torch.bool,
            device=self.device,
        )
        if request.assignment_due:
            current_assignment = replace(
                observation.assignment_actor_inputs.to(self.device),
                hidden=self.assignment_hidden,
            )
            with torch.no_grad():
                if self.assignment_mode == "capacity_aware":
                    assignment_output = _capacity_aware_assignment_output(
                        current_assignment,
                        self.env.config.scenario.max_missiles_per_target,
                        self.assignment_actor.config.d_model,
                    )
                else:
                    self.env.record_network_call("assignment_actor")
                    assignment_output = self.assignment_actor(
                        current_assignment,
                        deterministic=self.assignment_deterministic,
                    )
            previous_slots = torch.as_tensor(
                [
                    red.current_target_index + 1
                    if 0 <= red.current_target_index < len(self.env.state.blue)
                    else 0
                    for red in self.env.state.red
                ],
                dtype=torch.long,
                device=self.device,
            ).reshape_as(assignment_output.actions.target)
            target_changed = assignment_output.actions.target != previous_slots
            self.execution_hidden = self.execution_hidden * (~target_changed).unsqueeze(-1)
            self.assignment_hidden = assignment_output.next_hidden.detach()
            self.assignment_output = assignment_output

        if self.assignment_output is None:
            raise RuntimeError("controller has no assignment output")
        if request.bias_due:
            with torch.no_grad():
                current_execution = self.env.observation_layer.execution_inputs(
                    self.env.state,
                    self.assignment_output.actions.target,
                    hidden=self.execution_hidden,
                ).to(self.device)
                self.env.record_network_call("execution_actor")
                execution_output = self.execution_actor(
                    current_execution,
                    deterministic=self.execution_deterministic,
                )
            self.execution_hidden = execution_output.next_hidden.detach()
            self.policy = PolicyOutput(
                assignment=self.assignment_output,
                execution=execution_output,
            )
        if self.policy is None:
            raise RuntimeError("controller has no execution output")
        self._last_assignment_due = request.assignment_due
        return self.policy, target_changed

    def observe(self, step) -> None:
        if bool(step.info.get("assignment_updated", False)) != self._last_assignment_due:
            raise RuntimeError("controller decision schedule disagrees with environment")


def collect_rollout(
    env: RedBlueEngagementEnv,
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    assignment_critic: TargetAssignmentCritic,
    execution_critic: OverloadBiasCritic,
    steps: int,
    *,
    seed: int | None = None,
    style: ScenarioStyle | None = None,
    red_count: int | None = None,
    blue_count: int | None = None,
    deterministic: bool = False,
    assignment_mode: str = "actor",
    assignment_deterministic: bool | None = None,
    execution_deterministic: bool | None = None,
) -> tuple[MAPPOBatch, RolloutStats]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    device = _module_device(assignment_actor)
    modules = (execution_actor, assignment_critic, execution_critic)
    if any(_module_device(module) != device for module in modules):
        raise ValueError("all actors and critics must be on the same device")
    if env.config.policy_start_mode != "post_boost":
        raise ValueError("collect_rollout requires policy_start_mode='post_boost'")
    if assignment_mode not in {"actor", "capacity_aware"}:
        raise ValueError("assignment_mode must be actor or capacity_aware")
    assignment_deterministic = (
        deterministic if assignment_deterministic is None else assignment_deterministic
    )
    execution_deterministic = (
        deterministic if execution_deterministic is None else execution_deterministic
    )

    observation = env.reset(
        seed=seed,
        style=style,
        red_count=red_count,
        blue_count=blue_count,
        start_mode="post_boost",
    )
    assignment_template = observation.assignment_actor_inputs.to(device)
    batch_size, agent_count = assignment_template.self_state.shape[:2]
    d_model = assignment_actor.config.d_model
    assignment_hidden = torch.zeros(
        batch_size,
        agent_count,
        d_model,
        dtype=assignment_template.self_state.dtype,
        device=device,
    )
    execution_hidden = torch.zeros_like(assignment_hidden)
    critic_hidden = torch.zeros(
        batch_size,
        agent_count,
        d_model,
        dtype=assignment_template.self_state.dtype,
        device=device,
    )

    assignment_inputs: list[AssignmentActorInputs] = []
    execution_inputs: list[OverloadBiasActorInputs] = []
    assignment_actions: list[AssignmentActions] = []
    bias_matrices: list[Tensor] = []
    old_assignment_log_probs: list[Tensor] = []
    old_execution_log_probs: list[Tensor] = []
    rewards_high: list[Tensor] = []
    durations_high_s: list[Tensor] = []
    rewards_low: list[Tensor] = []
    durations_low_s: list[Tensor] = []
    dones_high: list[Tensor] = []
    dones_low: list[Tensor] = []
    episode_active_high: list[Tensor] = []
    episode_active_low: list[Tensor] = []
    assignment_critic_inputs: list[AssignmentCriticInputs] = []
    execution_critic_inputs: list[OverloadBiasCriticInputs] = []
    final_info: dict[str, Any] = {}

    for _ in range(steps):
        if env.state is None:
            raise RuntimeError("environment state is unavailable during rollout")
        if not env.next_decision_request().assignment_due:
            raise RuntimeError("rollout high transition is not due according to the environment")
        current_assignment = replace(
            observation.assignment_actor_inputs.to(device),
            hidden=assignment_hidden,
        )
        current_assignment_critic = observation.assignment_critic_inputs.to(device)
        episode_active_high.append(torch.ones(batch_size, dtype=torch.bool))
        with torch.no_grad():
            if assignment_mode == "capacity_aware":
                assignment_output = _capacity_aware_assignment_output(
                    current_assignment,
                    env.config.scenario.max_missiles_per_target,
                    assignment_actor.config.d_model,
                )
            else:
                env.record_network_call("assignment_actor")
                assignment_output = assignment_actor(
                    current_assignment,
                    deterministic=assignment_deterministic,
                )
        previous_target_slots = np.asarray(
            [
                red.current_target_index + 1
                if 0 <= red.current_target_index < len(env.state.blue)
                else 0
                for red in env.state.red
            ],
            dtype=np.int64,
        )
        selected_target_slots = (
            assignment_output.actions.target[0].detach().cpu().numpy().astype(np.int64)
        )
        target_changed = selected_target_slots != previous_target_slots
        target_changed_tensor = torch.as_tensor(
            target_changed,
            dtype=torch.bool,
            device=device,
        ).reshape(batch_size, agent_count)
        keep_target_hidden = (~target_changed_tensor).unsqueeze(-1)
        execution_hidden = execution_hidden * keep_target_hidden
        critic_hidden = critic_hidden * keep_target_hidden
        if dones_low:
            dones_low[-1] = torch.maximum(
                dones_low[-1],
                target_changed_tensor.to(
                    device="cpu",
                    dtype=dones_low[-1].dtype,
                ),
            )

        accumulated_reward_high = 0.0
        environment_step = None
        for low_index in range(env.config.bias_updates_per_assignment):
            if env.state is None:
                raise RuntimeError("environment state is unavailable during execution rollout")
            if not env.next_decision_request().bias_due:
                raise RuntimeError("rollout low transition is not due according to the environment")
            current_execution_critic = replace(
                observation.execution_critic_inputs.to(device),
                hidden=critic_hidden,
            )
            episode_active_low.append(torch.ones(batch_size, dtype=torch.bool))
            with torch.no_grad():
                current_execution = (
                    env.observation_layer.execution_inputs(
                        env.state,
                        assignment_output.actions.target,
                        hidden=execution_hidden,
                    ).to(device)
                    if low_index == 0
                    else replace(
                        observation.execution_actor_inputs.to(device),
                        hidden=execution_hidden,
                    )
                )
                env.record_network_call("execution_actor")
                execution_output = execution_actor(
                    current_execution,
                    deterministic=execution_deterministic,
                )
                env.record_network_call("execution_critic")
                critic_output = execution_critic(current_execution_critic)
            policy = PolicyOutput(assignment=assignment_output, execution=execution_output)
            accumulated_reward_low = np.zeros(agent_count, dtype=np.float64)
            low_frames_executed = 0
            for _ in range(env.config.bias_update_steps):
                environment_step = env.step(policy)
                low_frames_executed += 1
                accumulated_reward_high += float(environment_step.reward_high)
                accumulated_reward_low += np.asarray(
                    environment_step.reward_low,
                    dtype=np.float64,
                )
                if (
                    environment_step.done
                    or environment_step.info.get("high_reward_settled", False)
                ):
                    break
            if environment_step is None:
                raise RuntimeError("rollout did not execute an environment step")

            execution_inputs.append(_cpu_rollout_storage(current_execution))
            bias_matrices.append(_cpu_rollout_storage(execution_output.bias_matrix))
            old_execution_log_probs.append(
                _cpu_rollout_storage(execution_output.log_prob)
            )
            rewards_low.append(
                torch.as_tensor(accumulated_reward_low, dtype=torch.float32)
            )
            durations_low_s.append(
                torch.as_tensor(
                    low_frames_executed * env.config.time_step_s,
                    dtype=torch.float32,
                )
            )
            red_alive = np.asarray(environment_step.info["red_alive"], dtype=bool)
            blue_alive = np.asarray(environment_step.info["blue_alive"], dtype=bool)
            selected_targets = assignment_output.actions.target[0].detach().cpu().numpy() - 1
            pair_alive = red_alive & (selected_targets >= 0)
            valid_target = selected_targets >= 0
            pair_alive[valid_target] &= blue_alive[selected_targets[valid_target]]
            dones_low.append(
                torch.as_tensor(
                    bool(environment_step.done) | ~pair_alive,
                    dtype=torch.float32,
                ).reshape(batch_size, agent_count)
            )
            execution_critic_inputs.append(
                _cpu_rollout_storage(current_execution_critic)
            )
            execution_hidden = execution_output.next_hidden.detach()
            critic_hidden = critic_output.next_hidden.detach()
            observation = environment_step.observation
            final_info = dict(environment_step.info)
            if (
                environment_step.done
                or environment_step.info.get("high_reward_settled", False)
            ):
                break
        if environment_step is None:
            raise RuntimeError("rollout did not execute an environment step")

        assignment_inputs.append(_cpu_rollout_storage(current_assignment))
        assignment_actions.append(_cpu_rollout_storage(assignment_output.actions))
        old_assignment_log_probs.append(
            _cpu_rollout_storage(assignment_output.joint_log_prob)
        )
        rewards_high.append(
            torch.as_tensor(accumulated_reward_high, dtype=torch.float32)
        )
        durations_high_s.append(
            torch.as_tensor(
                environment_step.info["reward_components"]["elapsed_s"],
                dtype=torch.float32,
            )
        )
        dones_high.append(
            torch.as_tensor(float(environment_step.done), dtype=torch.float32)
        )
        assignment_critic_inputs.append(
            _cpu_rollout_storage(current_assignment_critic)
        )

        assignment_hidden = assignment_output.next_hidden.detach()
        if environment_step.done:
            break

    assignment_critic_inputs.append(
        _cpu_rollout_storage(observation.assignment_critic_inputs)
    )
    execution_critic_inputs.append(
        _cpu_rollout_storage(
            replace(
                observation.execution_critic_inputs.to(device),
                hidden=critic_hidden,
            )
        )
    )
    reward_high_tensor = torch.stack(rewards_high, dim=0).reshape(-1, batch_size)
    reward_low_tensor = torch.stack(rewards_low, dim=0).reshape(-1, batch_size, agent_count)
    done_high_tensor = torch.stack(dones_high, dim=0).reshape(-1, batch_size)
    done_low_tensor = torch.stack(dones_low, dim=0).reshape(-1, batch_size, agent_count)
    batch = MAPPOBatch(
        assignment_actor_inputs=_stack_assignment_inputs(assignment_inputs),
        execution_actor_inputs=_stack_execution_inputs(execution_inputs),
        assignment_actions=_stack_assignment_actions(assignment_actions),
        bias_matrices=torch.stack(bias_matrices, dim=0),
        old_assignment_log_prob=torch.stack(old_assignment_log_probs, dim=0),
        old_execution_log_prob=torch.stack(old_execution_log_probs, dim=0),
        rewards_high=reward_high_tensor,
        durations_high_s=torch.stack(durations_high_s, dim=0).reshape(-1, batch_size),
        high_reference_interval_s=env.config.assignment_update_interval_s,
        rewards_low=reward_low_tensor,
        durations_low_s=torch.stack(durations_low_s, dim=0).reshape(-1, batch_size),
        low_reference_interval_s=env.config.bias_update_interval_s,
        dones_high=done_high_tensor,
        dones_low=done_low_tensor,
        episode_active_high=torch.stack(episode_active_high, dim=0),
        episode_active_low=torch.stack(episode_active_low, dim=0),
        assignment_critic_inputs=_stack_assignment_critic_inputs(assignment_critic_inputs),
        execution_critic_inputs=_stack_execution_critic_inputs(execution_critic_inputs),
        scenario_red_counts=torch.as_tensor(
            [agent_count], dtype=torch.long
        ),
        scenario_blue_counts=torch.as_tensor(
            [len(env.state.blue) if env.state is not None else 0],
            dtype=torch.long,
        ),
    )
    if device.type == "cuda":
        batch = batch.pin_memory()
    _validate_batch_scenario_padding(batch)
    stats = _build_rollout_stats(
        batch,
        [final_info],
        done=bool(done_high_tensor[-1, 0].item()),
    )
    return batch, stats


def collect_parallel_rollout(
    envs: list[RedBlueEngagementEnv],
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    assignment_critic: TargetAssignmentCritic,
    execution_critic: OverloadBiasCritic,
    steps: int,
    *,
    seed: int | None = None,
    style: ScenarioStyle | None = None,
    red_count: int | None = None,
    blue_count: int | None = None,
    red_counts: Sequence[int | None] | None = None,
    blue_counts: Sequence[int | None] | None = None,
    deterministic: bool = False,
    assignment_mode: str = "actor",
    assignment_deterministic: bool | None = None,
    execution_deterministic: bool | None = None,
    env_pool: ProcessEnvironmentPool | None = None,
) -> tuple[MAPPOBatch, RolloutStats]:
    """Collect one synchronized rollout from N CPU environments with batched networks."""
    if not envs:
        raise ValueError("envs must contain at least one environment")
    if red_counts is not None and len(red_counts) != len(envs):
        raise ValueError("red_counts must match environment count")
    if blue_counts is not None and len(blue_counts) != len(envs):
        raise ValueError("blue_counts must match environment count")
    if len(envs) == 1:
        single_red_count = red_counts[0] if red_counts is not None else red_count
        single_blue_count = (
            blue_counts[0]
            if blue_counts is not None
            else blue_count
        )
        return collect_rollout(
            envs[0],
            assignment_actor,
            execution_actor,
            assignment_critic,
            execution_critic,
            steps,
            seed=seed,
            style=style,
            red_count=single_red_count,
            blue_count=single_blue_count,
            deterministic=deterministic,
            assignment_mode=assignment_mode,
            assignment_deterministic=assignment_deterministic,
            execution_deterministic=execution_deterministic,
        )
    if steps <= 0:
        raise ValueError("steps must be positive")
    reference = envs[0].config
    if any(env.config != reference for env in envs):
        raise ValueError("parallel environments must use identical configurations")
    if reference.policy_start_mode != "post_boost":
        raise ValueError("collect_parallel_rollout requires post_boost environments")
    device = _module_device(assignment_actor)
    if any(
        _module_device(module) != device
        for module in (execution_actor, assignment_critic, execution_critic)
    ):
        raise ValueError("all actors and critics must be on the same device")
    if env_pool is not None and env_pool.size != len(envs):
        raise ValueError("environment pool size must match envs")
    owns_pool = env_pool is None
    runner = ThreadEnvironmentPool(envs) if env_pool is None else env_pool
    try:
        return _collect_parallel_with_pool(
            envs,
            runner,
            assignment_actor,
            execution_actor,
            assignment_critic,
            execution_critic,
            steps,
            seed=seed,
            style=style,
            red_count=red_count,
            blue_count=blue_count,
            red_counts=red_counts,
            blue_counts=blue_counts,
            deterministic=deterministic,
            assignment_mode=assignment_mode,
            assignment_deterministic=assignment_deterministic,
            execution_deterministic=execution_deterministic,
            device=device,
        )
    finally:
        if owns_pool:
            runner.close()


def _collect_parallel_with_pool(
    envs: list[RedBlueEngagementEnv],
    env_pool: ThreadEnvironmentPool | ProcessEnvironmentPool,
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    assignment_critic: TargetAssignmentCritic,
    execution_critic: OverloadBiasCritic,
    steps: int,
    *,
    seed: int | None,
    style: ScenarioStyle | None,
    red_count: int | None,
    blue_count: int | None,
    red_counts: Sequence[int | None] | None,
    blue_counts: Sequence[int | None] | None,
    deterministic: bool,
    assignment_mode: str,
    assignment_deterministic: bool | None,
    execution_deterministic: bool | None,
    device: torch.device,
) -> tuple[MAPPOBatch, RolloutStats]:
    if assignment_mode not in {"actor", "capacity_aware"}:
        raise ValueError("assignment_mode must be actor or capacity_aware")
    assignment_deterministic = (
        deterministic if assignment_deterministic is None else assignment_deterministic
    )
    execution_deterministic = (
        deterministic if execution_deterministic is None else execution_deterministic
    )
    reference = envs[0].config
    observations = env_pool.reset(
        seed=seed,
        style=style,
        red_count=red_count,
        blue_count=blue_count,
        red_counts=red_counts,
        blue_counts=blue_counts,
    )
    template = _concat_assignment_inputs(
        [observation.assignment_actor_inputs for observation in observations]
    ).to(device)
    batch_size, agent_count = template.self_state.shape[:2]
    if batch_size != len(envs):
        raise RuntimeError("each environment must contribute exactly one batch row")
    d_model = assignment_actor.config.d_model
    assignment_hidden = torch.zeros(
        batch_size,
        agent_count,
        d_model,
        dtype=template.self_state.dtype,
        device=device,
    )
    execution_hidden = torch.zeros_like(assignment_hidden)
    critic_hidden = torch.zeros(
        batch_size,
        agent_count,
        d_model,
        dtype=template.self_state.dtype,
        device=device,
    )
    terminated = np.zeros(batch_size, dtype=bool)
    final_infos: list[dict[str, Any]] = [{} for _ in envs]
    scenario_red_counts = np.asarray(
        [observation.assignment_actor_inputs.self_state.shape[1] for observation in observations],
        dtype=np.int64,
    )
    scenario_blue_counts = np.asarray(
        [observation.assignment_critic_inputs.global_blue.shape[1] for observation in observations],
        dtype=np.int64,
    )
    episode_hit_counts = np.zeros(batch_size, dtype=np.int64)
    episode_low_component_sums: list[dict[str, float]] = [
        {} for _ in range(batch_size)
    ]
    episode_time_credit_unassigned = np.zeros(batch_size, dtype=np.int64)

    assignment_inputs: list[AssignmentActorInputs] = []
    execution_inputs: list[OverloadBiasActorInputs] = []
    assignment_actions: list[AssignmentActions] = []
    bias_matrices: list[Tensor] = []
    old_assignment_log_probs: list[Tensor] = []
    old_execution_log_probs: list[Tensor] = []
    rewards_high: list[Tensor] = []
    durations_high_s: list[Tensor] = []
    rewards_low: list[Tensor] = []
    durations_low_s: list[Tensor] = []
    dones_high: list[Tensor] = []
    dones_low: list[Tensor] = []
    episode_active_high: list[Tensor] = []
    episode_active_low: list[Tensor] = []
    assignment_critic_inputs: list[AssignmentCriticInputs] = []
    execution_critic_inputs: list[OverloadBiasCriticInputs] = []

    for _ in range(steps):
        high_active = ~terminated.copy()
        current_assignment = replace(
            _concat_assignment_inputs(
                [observation.assignment_actor_inputs for observation in observations]
            ).to(device),
            hidden=assignment_hidden,
        )
        current_assignment_critic = _concat_assignment_critic_inputs(
            [observation.assignment_critic_inputs for observation in observations]
        ).to(device)
        with torch.no_grad():
            if assignment_mode == "capacity_aware":
                assignment_output = _capacity_aware_assignment_output(
                    current_assignment,
                    reference.scenario.max_missiles_per_target,
                    assignment_actor.config.d_model,
                )
            else:
                assignment_output = assignment_actor(
                    current_assignment,
                    deterministic=assignment_deterministic,
                )
        assignment_agent_mask = current_assignment.agent_mask.bool()
        assignment_output = replace(
            assignment_output,
            actions=AssignmentActions(
                target=torch.where(
                    assignment_agent_mask,
                    assignment_output.actions.target,
                    torch.zeros_like(assignment_output.actions.target),
                ),
                order=assignment_output.actions.order,
            ),
            next_hidden=assignment_output.next_hidden
            * assignment_agent_mask.unsqueeze(-1),
        )
        target_slots = (
            assignment_output.actions.target.detach().cpu().numpy().astype(np.int64)
        )
        previous_target_slots = (
            current_assignment.current_assignment.argmax(dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.int64)
        )
        target_changed = (target_slots != previous_target_slots) & high_active[:, None]
        target_changed_tensor = torch.as_tensor(
            target_changed,
            dtype=torch.bool,
            device=device,
        )
        keep_target_hidden = (~target_changed_tensor).unsqueeze(-1)
        execution_hidden = execution_hidden * keep_target_hidden
        critic_hidden = critic_hidden * keep_target_hidden
        if dones_low:
            dones_low[-1] = torch.maximum(
                dones_low[-1],
                target_changed_tensor.to(
                    device="cpu",
                    dtype=dones_low[-1].dtype,
                ),
            )
        target_indices = target_slots - 1
        accumulated_reward_high = np.zeros(batch_size, dtype=np.float64)
        segment_duration_s = np.zeros(batch_size, dtype=np.float64)
        segment_complete = terminated.copy()
        last_done = terminated.copy()

        for low_index in range(reference.bias_updates_per_assignment):
            low_active = ~segment_complete.copy()
            per_env_execution = (
                env_pool.execution_inputs(
                    target_slots,
                    high_active
                    if assignment_mode == "actor"
                    else np.zeros_like(high_active),
                )
                if low_index == 0
                else [observation.execution_actor_inputs for observation in observations]
            )
            current_execution = replace(
                _concat_execution_inputs(per_env_execution).to(device),
                hidden=execution_hidden,
            )
            current_execution_critic = replace(
                _concat_execution_critic_inputs(
                    [observation.execution_critic_inputs for observation in observations]
                ).to(device),
                hidden=critic_hidden,
            )
            with torch.no_grad():
                execution_output = execution_actor(
                    current_execution,
                    deterministic=execution_deterministic,
                )
                critic_output = execution_critic(current_execution_critic)
            execution_agent_mask = current_execution.agent_mask.bool()
            execution_output = replace(
                execution_output,
                bias_matrix=execution_output.bias_matrix
                * execution_agent_mask.unsqueeze(-1),
                log_prob=execution_output.log_prob
                * execution_agent_mask.to(dtype=execution_output.log_prob.dtype),
                entropy=execution_output.entropy
                * execution_agent_mask.to(dtype=execution_output.entropy.dtype),
                next_hidden=execution_output.next_hidden
                * execution_agent_mask.unsqueeze(-1),
            )
            accumulated_reward_low = np.zeros((batch_size, agent_count), dtype=np.float64)
            low_duration_s = np.zeros(batch_size, dtype=np.float64)
            active_indices = [
                index for index in range(batch_size) if not segment_complete[index]
            ]
            guidance_bias = execution_output.bias_matrix.detach().cpu().numpy()
            results = env_pool.advance(
                active_indices,
                target_indices,
                guidance_bias,
                reference.bias_update_steps,
                collect_metrics=True,
            )
            for env_index, result in results.items():
                observations[env_index] = result.observation
                accumulated_reward_high[env_index] += result.reward_high
                n_red = int(scenario_red_counts[env_index])
                accumulated_reward_low[env_index, :n_red] += result.reward_low
                episode_hit_counts[env_index] += int(result.hit_count)
                for name, value in result.low_reward_component_sums.items():
                    episode_low_component_sums[env_index][name] = (
                        episode_low_component_sums[env_index].get(name, 0.0)
                        + float(value)
                    )
                episode_time_credit_unassigned[env_index] += int(
                    result.time_credit_unassigned_count
                )
                low_duration_s[env_index] = result.frames_executed * reference.time_step_s
                final_infos[env_index] = result.info
                terminated[env_index] = terminated[env_index] or result.done
                if result.done or result.info.get("high_reward_settled", False):
                    segment_complete[env_index] = True
                    reward_components = result.info.get("reward_components", {})
                    segment_duration_s[env_index] = float(
                        reward_components.get(
                            "elapsed_s",
                            result.frames_executed * reference.time_step_s,
                        )
                    )

            execution_inputs.append(_cpu_rollout_storage(current_execution))
            bias_matrices.append(_cpu_rollout_storage(execution_output.bias_matrix))
            old_execution_log_probs.append(
                _cpu_rollout_storage(execution_output.log_prob)
            )
            rewards_low.append(
                torch.as_tensor(accumulated_reward_low, dtype=torch.float32)
            )
            durations_low_s.append(
                torch.as_tensor(low_duration_s, dtype=torch.float32)
            )
            last_done = terminated.copy()
            agent_done = np.ones((batch_size, agent_count), dtype=bool)
            for env_index, result in results.items():
                red_alive = np.asarray(result.info["red_alive"], dtype=bool)
                blue_alive = np.asarray(result.info["blue_alive"], dtype=bool)
                n_red = int(scenario_red_counts[env_index])
                selected = target_indices[env_index, :n_red]
                pair_alive = red_alive & (selected >= 0)
                valid_target = selected >= 0
                pair_alive[valid_target] &= blue_alive[selected[valid_target]]
                agent_done[env_index, :n_red] = bool(result.done) | ~pair_alive
            dones_low.append(torch.as_tensor(agent_done, dtype=torch.float32))
            execution_critic_inputs.append(
                _cpu_rollout_storage(current_execution_critic)
            )
            active_after = torch.as_tensor(~terminated, dtype=torch.bool, device=device)
            execution_hidden = execution_output.next_hidden.detach() * active_after[:, None, None]
            critic_hidden = critic_output.next_hidden.detach() * active_after[:, None, None]
            episode_active_low.append(torch.as_tensor(low_active, dtype=torch.bool))
            if segment_complete.all():
                break

        assignment_inputs.append(_cpu_rollout_storage(current_assignment))
        assignment_actions.append(_cpu_rollout_storage(assignment_output.actions))
        old_assignment_log_probs.append(
            _cpu_rollout_storage(assignment_output.joint_log_prob)
        )
        rewards_high.append(
            torch.as_tensor(accumulated_reward_high, dtype=torch.float32)
        )
        durations_high_s.append(
            torch.as_tensor(segment_duration_s, dtype=torch.float32)
        )
        dones_high.append(torch.as_tensor(last_done, dtype=torch.float32))
        assignment_critic_inputs.append(
            _cpu_rollout_storage(current_assignment_critic)
        )
        active_after = torch.as_tensor(~terminated, dtype=torch.bool, device=device)
        assignment_hidden = assignment_output.next_hidden.detach() * active_after[:, None, None]
        episode_active_high.append(torch.as_tensor(high_active, dtype=torch.bool))
        if terminated.all():
            break

    assignment_critic_inputs.append(
        _cpu_rollout_storage(
            _concat_assignment_critic_inputs(
                [observation.assignment_critic_inputs for observation in observations]
            )
        )
    )
    execution_critic_inputs.append(
        _cpu_rollout_storage(
            replace(
                _concat_execution_critic_inputs(
                    [observation.execution_critic_inputs for observation in observations]
                ).to(device),
                hidden=critic_hidden,
            )
        )
    )
    env_pool.sync_states(envs)

    for env_index, info in enumerate(final_infos):
        info["episode_low_reward_component_sums"] = dict(
            episode_low_component_sums[env_index]
        )
        info["episode_time_credit_unassigned_count"] = int(
            episode_time_credit_unassigned[env_index]
        )

    reward_high_tensor = torch.stack(rewards_high, dim=0)
    reward_low_tensor = torch.stack(rewards_low, dim=0)
    done_high_tensor = torch.stack(dones_high, dim=0)
    done_low_tensor = torch.stack(dones_low, dim=0)
    batch = MAPPOBatch(
        assignment_actor_inputs=_stack_assignment_inputs(assignment_inputs),
        execution_actor_inputs=_stack_execution_inputs(execution_inputs),
        assignment_actions=_stack_assignment_actions(assignment_actions),
        bias_matrices=torch.stack(bias_matrices, dim=0),
        old_assignment_log_prob=torch.stack(old_assignment_log_probs, dim=0),
        old_execution_log_prob=torch.stack(old_execution_log_probs, dim=0),
        rewards_high=reward_high_tensor,
        durations_high_s=torch.stack(durations_high_s, dim=0),
        high_reference_interval_s=reference.assignment_update_interval_s,
        rewards_low=reward_low_tensor,
        durations_low_s=torch.stack(durations_low_s, dim=0),
        low_reference_interval_s=reference.bias_update_interval_s,
        dones_high=done_high_tensor,
        dones_low=done_low_tensor,
        episode_active_high=torch.stack(episode_active_high, dim=0),
        episode_active_low=torch.stack(episode_active_low, dim=0),
        assignment_critic_inputs=_stack_assignment_critic_inputs(assignment_critic_inputs),
        execution_critic_inputs=_stack_execution_critic_inputs(execution_critic_inputs),
        scenario_red_counts=torch.as_tensor(
            scenario_red_counts, dtype=torch.long
        ),
        scenario_blue_counts=torch.as_tensor(
            scenario_blue_counts, dtype=torch.long
        ),
    )
    if device.type == "cuda":
        batch = batch.pin_memory()
    _validate_batch_scenario_padding(batch)
    stats = _build_rollout_stats(
        batch,
        final_infos,
        done=bool(terminated.all()),
        episode_hit_counts=episode_hit_counts,
    )
    return batch, stats


def evaluate_parallel_episodes(
    envs: list[RedBlueEngagementEnv],
    env_pool: ProcessEnvironmentPool,
    assignment_actor: TargetAssignmentActor,
    execution_actor: OverloadBiasActor,
    *,
    seeds: list[int],
    style: ScenarioStyle,
    red_count: int,
    blue_count: int,
    max_assignment_steps: int,
    deterministic: bool = True,
    assignment_mode: str = "actor",
    assignment_deterministic: bool | None = None,
    execution_deterministic: bool | None = None,
) -> list[ParallelEpisodeMetrics]:
    """Evaluate one fixed-seed episode per worker without retaining training tensors."""
    if len(envs) != env_pool.size or len(seeds) != env_pool.size:
        raise ValueError("envs, seeds, and process pool must have identical sizes")
    if max_assignment_steps <= 0:
        raise ValueError("max_assignment_steps must be positive")
    if assignment_mode not in {"actor", "capacity_aware"}:
        raise ValueError("assignment_mode must be actor or capacity_aware")
    assignment_deterministic = (
        deterministic
        if assignment_deterministic is None
        else assignment_deterministic
    )
    execution_deterministic = (
        deterministic
        if execution_deterministic is None
        else execution_deterministic
    )
    reference = envs[0].config
    device = _module_device(assignment_actor)
    if _module_device(execution_actor) != device:
        raise ValueError("assignment and execution actors must use the same device")
    observations = env_pool.reset(
        seed=None,
        style=style,
        red_count=red_count,
        blue_count=blue_count,
        seeds=seeds,
    )
    template = _concat_assignment_inputs(
        [observation.assignment_actor_inputs for observation in observations]
    ).to(device)
    batch_size, agent_count = template.self_state.shape[:2]
    d_model = assignment_actor.config.d_model
    assignment_hidden = torch.zeros(
        batch_size,
        agent_count,
        d_model,
        dtype=template.self_state.dtype,
        device=device,
    )
    execution_hidden = torch.zeros_like(assignment_hidden)
    terminated = np.zeros(batch_size, dtype=bool)
    physics_frames = np.zeros(batch_size, dtype=np.int64)
    max_physics_steps = max_assignment_steps * reference.assignment_update_steps
    final_infos: list[dict[str, Any]] = [{} for _ in envs]

    while True:
        high_active = (~terminated) & (physics_frames < max_physics_steps)
        if not high_active.any():
            break
        current_assignment = replace(
            _concat_assignment_inputs(
                [observation.assignment_actor_inputs for observation in observations]
            ).to(device),
            hidden=assignment_hidden,
        )
        with torch.no_grad():
            if assignment_mode == "capacity_aware":
                assignment_output = _capacity_aware_assignment_output(
                    current_assignment,
                    reference.scenario.max_missiles_per_target,
                    assignment_actor.config.d_model,
                )
            else:
                assignment_output = assignment_actor(
                    current_assignment,
                    deterministic=assignment_deterministic,
                )
        target_slots = assignment_output.actions.target.detach().cpu().numpy().astype(np.int64)
        previous_target_slots = np.stack(
            [
                observation.assignment_actor_inputs.current_assignment[0]
                .argmax(dim=-1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.int64)
                for observation in observations
            ],
            axis=0,
        )
        target_changed = (target_slots != previous_target_slots) & high_active[:, None]
        target_changed_tensor = torch.as_tensor(target_changed, dtype=torch.bool, device=device)
        execution_hidden = execution_hidden * (~target_changed_tensor).unsqueeze(-1)
        high_active_tensor = torch.as_tensor(high_active, dtype=torch.bool, device=device)
        assignment_hidden = torch.where(
            high_active_tensor[:, None, None],
            assignment_output.next_hidden.detach(),
            assignment_hidden,
        )
        target_indices = target_slots - 1
        segment_complete = ~high_active.copy()
        first_low = True

        while not segment_complete.all():
            low_active = ~segment_complete.copy()
            per_env_execution = env_pool.execution_inputs(
                target_slots,
                high_active if first_low else np.zeros_like(high_active),
            )
            first_low = False
            current_execution = replace(
                _concat_execution_inputs(per_env_execution).to(device),
                hidden=execution_hidden,
            )
            with torch.no_grad():
                execution_output = execution_actor(
                    current_execution,
                    deterministic=execution_deterministic,
                )
            active_indices = [index for index in range(batch_size) if low_active[index]]
            results = env_pool.advance(
                active_indices,
                target_indices,
                execution_output.bias_matrix.detach().cpu().numpy(),
                reference.bias_update_steps,
                record_critic=False,
                collect_metrics=False,
            )
            for env_index, result in results.items():
                observations[env_index] = result.observation
                physics_frames[env_index] += result.frames_executed
                final_infos[env_index] = dict(result.info)
                terminated[env_index] = terminated[env_index] or result.done
                if (
                    result.done
                    or result.info.get("high_reward_settled", False)
                    or physics_frames[env_index] >= max_physics_steps
                ):
                    segment_complete[env_index] = True
            low_active_tensor = torch.as_tensor(low_active, dtype=torch.bool, device=device)
            execution_hidden = torch.where(
                low_active_tensor[:, None, None],
                execution_output.next_hidden.detach(),
                execution_hidden,
            )
            terminated_tensor = torch.as_tensor(terminated, dtype=torch.bool, device=device)
            execution_hidden = execution_hidden * (~terminated_tensor)[:, None, None]

    env_pool.sync_states(envs)
    rows: list[ParallelEpisodeMetrics] = []
    for index, env in enumerate(envs):
        if env.state is None:
            raise RuntimeError("environment state is unavailable after evaluation")
        blue_total = max(len(env.state.blue), 1)
        destroyed_blue = sum(not target.alive for target in env.state.blue)
        full_success = float(destroyed_blue == blue_total)
        rows.append(
            ParallelEpisodeMetrics(
                full_success=full_success,
                damage_rate=float(destroyed_blue / blue_total),
                ineffective_loss_rate=float(ineffective_loss_rate(env.state)),
                completion_time_s=(
                    float(env.state.time_s)
                    if full_success > 0.5
                    else float(reference.missile.max_guidance_time_s)
                ),
                control_effort=float(final_infos[index].get("control_effort", 0.0)),
            )
        )
    return rows
