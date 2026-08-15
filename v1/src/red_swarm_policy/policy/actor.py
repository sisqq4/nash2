from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn
from torch.distributions import Normal

from ..core.config import SwarmModelConfig
from ..core.masks import ensure_at_least_one_valid, masked_argmax, masked_categorical
from ..core.networks import (
    AttentionPool,
    MaskedCrossAttentionBlock,
    MaskedSelfAttentionBlock,
    build_mlp,
)

EPS = 1.0e-6


@dataclass
class AssignmentActorInputs:
    self_state: Tensor
    friend_entities: Tensor
    friend_mask: Tensor
    target_entities: Tensor
    pair_state: Tensor
    current_assignment: Tensor
    target_mask: Tensor
    environment_context: Tensor
    target_assignment_counts: Tensor
    target_entity_mask: Tensor
    agent_mask: Tensor
    hidden: Optional[Tensor] = None

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> "AssignmentActorInputs":
        def optional(value: Optional[Tensor]) -> Optional[Tensor]:
            return None if value is None else transform(value)

        return AssignmentActorInputs(
            self_state=transform(self.self_state),
            friend_entities=transform(self.friend_entities),
            friend_mask=transform(self.friend_mask),
            target_entities=transform(self.target_entities),
            pair_state=transform(self.pair_state),
            current_assignment=transform(self.current_assignment),
            target_mask=transform(self.target_mask),
            environment_context=transform(self.environment_context),
            target_assignment_counts=transform(self.target_assignment_counts),
            target_entity_mask=transform(self.target_entity_mask),
            agent_mask=transform(self.agent_mask),
            hidden=optional(self.hidden),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "AssignmentActorInputs":
        return self._map_tensors(
            lambda value: value.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "AssignmentActorInputs":
        return self._map_tensors(lambda value: value.pin_memory())

    def detached(self) -> "AssignmentActorInputs":
        return self._map_tensors(lambda value: value.detach())


@dataclass
class OverloadBiasActorInputs:
    self_state: Tensor
    same_target_friends: Tensor
    friend_mask: Tensor
    assigned_target: Tensor
    target_mask: Tensor
    environment_context: Tensor
    agent_mask: Tensor
    hidden: Optional[Tensor] = None

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> "OverloadBiasActorInputs":
        return OverloadBiasActorInputs(
            self_state=transform(self.self_state),
            same_target_friends=transform(self.same_target_friends),
            friend_mask=transform(self.friend_mask),
            assigned_target=transform(self.assigned_target),
            target_mask=transform(self.target_mask),
            environment_context=transform(self.environment_context),
            agent_mask=transform(self.agent_mask),
            hidden=None if self.hidden is None else transform(self.hidden),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "OverloadBiasActorInputs":
        return self._map_tensors(
            lambda value: value.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "OverloadBiasActorInputs":
        return self._map_tensors(lambda value: value.pin_memory())

    def detached(self) -> "OverloadBiasActorInputs":
        return self._map_tensors(lambda value: value.detach())


@dataclass
class AssignmentActions:
    target: Tensor
    order: Tensor | None = None

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> "AssignmentActions":
        return AssignmentActions(
            target=transform(self.target),
            order=None if self.order is None else transform(self.order),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "AssignmentActions":
        return self._map_tensors(
            lambda value: value.to(device, non_blocking=non_blocking)
        )

    def detached(self) -> "AssignmentActions":
        return self._map_tensors(lambda value: value.detach())

    def pin_memory(self) -> "AssignmentActions":
        return self._map_tensors(lambda value: value.pin_memory())


@dataclass
class AssignmentPolicyOutput:
    actions: AssignmentActions
    next_hidden: Tensor
    log_prob: Tensor
    entropy: Tensor
    joint_log_prob: Tensor
    joint_entropy: Tensor
    target_probabilities: Tensor
    assignment_matrix: Tensor


@dataclass
class OverloadBiasOutput:
    bias_matrix: Tensor
    next_hidden: Tensor
    log_prob: Tensor
    entropy: Tensor
    action_distribution: str = "tanh_box"


@dataclass
class PolicyOutput:
    assignment: AssignmentPolicyOutput
    execution: OverloadBiasOutput


@dataclass
class _AssignmentEvaluation:
    log_prob: Tensor
    entropy: Tensor
    joint_log_prob: Tensor
    joint_entropy: Tensor
    next_hidden: Tensor


@dataclass
class _EncodedAssignmentObservation:
    next_hidden: Tensor
    target_tokens: Tensor
    target_mask: Tensor
    current_assignment: Tensor


@dataclass
class _EncodedExecutionObservation:
    next_hidden: Tensor
    active: Tensor


@dataclass
class _OverloadBiasEvaluation:
    log_prob: Tensor
    entropy: Tensor
    next_hidden: Tensor


def _atanh(x: Tensor) -> Tensor:
    x = x.clamp(min=-1.0 + EPS, max=1.0 - EPS)
    return 0.5 * (torch.log1p(x) - torch.log1p(-x))


def _tanh_normal_sample(
    mu: Tensor,
    log_std: Tensor,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    dist = Normal(mu, torch.exp(log_std))
    pre_tanh = mu if deterministic else dist.rsample()
    action = torch.tanh(pre_tanh)
    log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + EPS)
    return action, log_prob.sum(dim=-1), dist.entropy().sum(dim=-1)


def _tanh_normal_log_prob(action: Tensor, mu: Tensor, log_std: Tensor) -> Tensor:
    action = action.clamp(min=-1.0 + EPS, max=1.0 - EPS)
    pre_tanh = _atanh(action)
    dist = Normal(mu, torch.exp(log_std))
    log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + EPS)
    return log_prob.sum(dim=-1)


def _radial_tanh_log_abs_det_jacobian(radius: Tensor) -> Tensor:
    radius = radius.clamp_min(0.0)
    small = radius < 1.0e-4
    log_sech_squared = 2.0 * (
        math.log(2.0) - radius - torch.nn.functional.softplus(-2.0 * radius)
    )
    safe_radius = radius.clamp_min(torch.finfo(radius.dtype).tiny)
    log_radial = torch.log(torch.tanh(radius).clamp_min(torch.finfo(radius.dtype).tiny))
    log_radial = log_radial - torch.log(safe_radius)
    exact = log_sech_squared + log_radial
    series = -(4.0 / 3.0) * radius.pow(2)
    return torch.where(small, series, exact)


def _radial_tanh_forward(pre_tanh: Tensor) -> tuple[Tensor, Tensor]:
    radius = torch.linalg.vector_norm(pre_tanh, dim=-1)
    safe_radius = radius.clamp_min(EPS)
    exact_scale = torch.tanh(radius) / safe_radius
    series_scale = 1.0 - radius.pow(2) / 3.0
    scale = torch.where(radius < 1.0e-4, series_scale, exact_scale)
    action = pre_tanh * scale.unsqueeze(-1)
    return action, _radial_tanh_log_abs_det_jacobian(radius)


def _radial_tanh_inverse(action: Tensor) -> tuple[Tensor, Tensor]:
    action_radius = torch.linalg.vector_norm(action, dim=-1)
    if bool((action_radius >= 1.0).any()):
        raise ValueError("radial_tanh_disk action norm must be smaller than one")
    radius = _atanh(action_radius)
    safe_action_radius = action_radius.clamp_min(EPS)
    exact_scale = radius / safe_action_radius
    series_scale = 1.0 + action_radius.pow(2) / 3.0
    scale = torch.where(action_radius < 1.0e-4, series_scale, exact_scale)
    pre_tanh = action * scale.unsqueeze(-1)
    return pre_tanh, _radial_tanh_log_abs_det_jacobian(radius)


def _radial_tanh_sample(
    mu: Tensor,
    log_std: Tensor,
    deterministic: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    distribution = Normal(mu, torch.exp(log_std))
    pre_tanh = mu if deterministic else distribution.rsample()
    action, log_abs_det = _radial_tanh_forward(pre_tanh)
    log_prob = distribution.log_prob(pre_tanh).sum(dim=-1) - log_abs_det
    entropy = distribution.entropy().sum(dim=-1)
    return action, log_prob, entropy


def _radial_tanh_log_prob(action: Tensor, mu: Tensor, log_std: Tensor) -> Tensor:
    pre_tanh, log_abs_det = _radial_tanh_inverse(action)
    distribution = Normal(mu, torch.exp(log_std))
    return distribution.log_prob(pre_tanh).sum(dim=-1) - log_abs_det


class TargetAssignmentActor(nn.Module):
    def __init__(self, config: SwarmModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.self_encoder = build_mlp(config.d_self, d, d, depth=2)
        self.friend_encoder = build_mlp(config.d_friend, d, d, depth=2)
        self.target_encoder = build_mlp(config.d_target, d, d, depth=2)
        self.relation_encoder = build_mlp(config.d_pair + 1, d, d, depth=2)
        self.assignment_encoder = build_mlp(1, d, d, depth=2)
        self.partial_assignment_encoder = build_mlp(1, d, d, depth=2)
        self.context_encoder = build_mlp(config.d_actor_context, d, d, depth=2)
        self.missile_attn = MaskedSelfAttentionBlock(d, config.num_heads)
        self.target_attn = MaskedSelfAttentionBlock(d, config.num_heads)
        self.missile_pool = AttentionPool(d)
        self.assignment_pool = AttentionPool(d)
        self.cross_attn = nn.MultiheadAttention(d, config.num_heads, batch_first=True)
        self.base_proj = build_mlp(5 * d, d, d, depth=3)
        self.gru = nn.GRUCell(d, d)
        self.target_head = build_mlp(3 * d, d, 1, depth=2)

    def _validate_inputs(self, inputs: AssignmentActorInputs) -> tuple[int, int, int, int]:
        if inputs.self_state.dim() != 3 or inputs.self_state.shape[-1] != self.config.d_self:
            raise ValueError(f"self_state must have shape [B, R, {self.config.d_self}]")
        batch, agents = inputs.self_state.shape[:2]
        if inputs.friend_entities.dim() != 4 or inputs.friend_entities.shape[:2] != (batch, agents):
            raise ValueError("friend_entities must have shape [B, R, F, d_friend]")
        if inputs.friend_entities.shape[-1] != self.config.d_friend:
            raise ValueError(f"friend_entities feature width must be {self.config.d_friend}")
        friend_count = inputs.friend_entities.shape[-2]
        if tuple(inputs.friend_mask.shape) != (batch, agents, friend_count):
            raise ValueError("friend_mask must match friend_entities entity dimensions")
        if inputs.target_entities.dim() != 4 or inputs.target_entities.shape[:2] != (batch, agents):
            raise ValueError("target_entities must have shape [B, R, U, d_target]")
        if inputs.target_entities.shape[-1] != self.config.d_target:
            raise ValueError(f"target_entities feature width must be {self.config.d_target}")
        target_count = inputs.target_entities.shape[-2]
        if target_count <= self.config.no_target_index:
            raise ValueError("target_entities must contain the reserved no-target slot")
        expected_target_shape = (batch, agents, target_count)
        if tuple(inputs.pair_state.shape) != (*expected_target_shape, self.config.d_pair):
            raise ValueError(
                f"pair_state must have shape [B, R, U, {self.config.d_pair}]"
            )
        if tuple(inputs.current_assignment.shape) != expected_target_shape:
            raise ValueError("current_assignment must have shape [B, R, U]")
        for name, value in (
            ("target_mask", inputs.target_mask),
            ("target_entity_mask", inputs.target_entity_mask),
            ("target_assignment_counts", inputs.target_assignment_counts),
        ):
            if tuple(value.shape) != expected_target_shape:
                raise ValueError(f"{name} shape {tuple(value.shape)} does not match {expected_target_shape}")
        if tuple(inputs.environment_context.shape) != (batch, agents, self.config.d_actor_context):
            raise ValueError(
                "environment_context must have shape "
                f"[B, R, {self.config.d_actor_context}]"
            )
        if tuple(inputs.agent_mask.shape) != (batch, agents):
            raise ValueError("agent_mask must have shape [B, R]")
        return batch, agents, friend_count, target_count

    def _safe_targets(self, inputs: AssignmentActorInputs) -> tuple[Tensor, Tensor]:
        safe_mask, all_invalid = ensure_at_least_one_valid(
            inputs.target_mask,
            default_index=self.config.no_target_index,
        )
        targets = inputs.target_entities.clone()
        if all_invalid.any():
            flat_targets = targets.reshape(-1, targets.shape[-2], targets.shape[-1])
            flat_invalid = all_invalid.reshape(-1)
            flat_targets[flat_invalid, self.config.no_target_index] = 0.0
        return targets, safe_mask

    def _encode(self, inputs: AssignmentActorInputs) -> _EncodedAssignmentObservation:
        batch, agents, friend_count, target_count = self._validate_inputs(inputs)
        d = self.config.d_model
        device = inputs.self_state.device
        dtype = inputs.self_state.dtype
        agent_mask = inputs.agent_mask.to(device=device).bool()

        self_token = self.self_encoder(inputs.self_state)
        friend_tokens = self.friend_encoder(
            inputs.friend_entities.reshape(batch * agents, friend_count, self.config.d_friend)
        )
        friend_mask = inputs.friend_mask.reshape(batch * agents, friend_count).to(device=device).bool()
        own_tokens = self_token.reshape(batch * agents, 1, d)
        own_mask = torch.ones(batch * agents, 1, dtype=torch.bool, device=device)
        missile_mask = torch.cat([own_mask, friend_mask], dim=-1)
        missile_tokens = self.missile_attn(
            torch.cat([own_tokens, friend_tokens], dim=-2),
            missile_mask,
        )
        current_missile = missile_tokens[:, 0].reshape(batch, agents, d)
        missile_context = self.missile_pool(missile_tokens, missile_mask).reshape(batch, agents, d)

        target_entities, target_mask = self._safe_targets(inputs)
        target_tokens = self.target_encoder(
            target_entities.reshape(batch * agents, target_count, self.config.d_target)
        )
        relation_features = torch.cat(
            [
                inputs.pair_state.to(device=device, dtype=dtype),
                inputs.current_assignment.to(device=device, dtype=dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        target_tokens = target_tokens + self.relation_encoder(
            relation_features.reshape(batch * agents, target_count, self.config.d_pair + 1)
        )
        target_entity_mask = inputs.target_entity_mask.to(device=device).bool() | target_mask
        target_entity_mask, _ = ensure_at_least_one_valid(
            target_entity_mask,
            default_index=self.config.no_target_index,
        )
        flat_target_entity_mask = target_entity_mask.reshape(batch * agents, target_count)
        target_tokens = self.target_attn(target_tokens, flat_target_entity_mask)
        target_tokens = target_tokens.reshape(batch, agents, target_count, d)

        assignment_counts = inputs.target_assignment_counts.to(device=device, dtype=dtype)
        assignment_tokens = self.assignment_encoder(assignment_counts.unsqueeze(-1))
        assignment_context = self.assignment_pool(
            assignment_tokens.reshape(batch * agents, target_count, d),
            flat_target_entity_mask,
        ).reshape(batch, agents, d)
        context_token = self.context_encoder(inputs.environment_context.to(device=device, dtype=dtype))

        query = current_missile.reshape(batch * agents, 1, d)
        keys = target_tokens.reshape(batch * agents, target_count, d)
        cross_context, _ = self.cross_attn(
            query,
            keys,
            keys,
            key_padding_mask=~flat_target_entity_mask,
            need_weights=False,
        )
        cross_context = cross_context.reshape(batch, agents, d)
        fused = self.base_proj(
            torch.cat(
                [current_missile, missile_context, cross_context, context_token, assignment_context],
                dim=-1,
            )
        )
        hidden = torch.zeros_like(fused) if inputs.hidden is None else inputs.hidden
        if hidden.shape != fused.shape:
            raise ValueError(f"hidden shape {tuple(hidden.shape)} does not match {tuple(fused.shape)}")
        hidden = hidden.to(device=device, dtype=dtype)
        next_hidden = self.gru(
            fused.reshape(batch * agents, d),
            hidden.reshape(batch * agents, d),
        ).reshape(batch, agents, d)
        next_hidden = next_hidden.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        return _EncodedAssignmentObservation(
            next_hidden=next_hidden,
            target_tokens=target_tokens,
            target_mask=target_mask,
            current_assignment=inputs.current_assignment.to(
                device=device,
                dtype=dtype,
            ),
        )

    def _capacity_constrained_actions(
        self,
        encoded: _EncodedAssignmentObservation,
        agent_mask: Tensor,
        *,
        deterministic: bool,
        target_actions: Tensor | None = None,
        order: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch, agents, target_count = encoded.target_mask.shape
        device = encoded.next_hidden.device
        dtype = encoded.next_hidden.dtype
        active_agents = agent_mask.to(device=device).bool()
        if order is None:
            if deterministic:
                order = torch.arange(agents, device=device).expand(batch, agents)
            else:
                order = torch.stack([torch.randperm(agents, device=device) for _ in range(batch)])
        else:
            order = order.to(device=device, dtype=torch.long)
        if tuple(order.shape) != (batch, agents):
            raise ValueError(f"assignment order shape {tuple(order.shape)} must be {(batch, agents)}")
        expected = torch.arange(agents, device=device).expand(batch, agents)
        if not torch.equal(torch.sort(order, dim=-1).values, expected):
            raise ValueError("assignment order must be a permutation of agent indices")

        provided_targets = None
        if target_actions is not None:
            provided_targets = target_actions.to(device=device, dtype=torch.long)
            if tuple(provided_targets.shape) != (batch, agents):
                raise ValueError(f"target actions shape {tuple(provided_targets.shape)} must be {(batch, agents)}")

        targets = torch.zeros(batch, agents, dtype=torch.long, device=device)
        log_prob = torch.zeros(batch, agents, dtype=dtype, device=device)
        entropy = torch.zeros_like(log_prob)
        probabilities = torch.zeros(
            batch,
            agents,
            target_count,
            dtype=dtype,
            device=device,
        )
        counts = torch.zeros(batch, max(target_count - 1, 0), dtype=torch.long, device=device)
        batch_indices = torch.arange(batch, device=device)
        capacity = self.config.max_missiles_per_target
        sticky_physical_targets = (
            encoded.current_assignment
            * encoded.target_mask.to(dtype=dtype)
        )
        sticky_physical_targets[..., self.config.no_target_index] = 0.0

        for position in range(agents):
            agent_indices = order[:, position]
            agent_base = encoded.next_hidden[batch_indices, agent_indices]
            agent_targets = encoded.target_tokens[batch_indices, agent_indices]
            padded_counts = torch.zeros(
                batch,
                target_count,
                dtype=dtype,
                device=device,
            )
            if target_count > 1:
                padded_counts[:, 1:] = counts.to(dtype=dtype) / float(capacity)
            partial_tokens = self.partial_assignment_encoder(padded_counts.unsqueeze(-1))
            row_pair = torch.cat(
                [
                    agent_base.unsqueeze(1).expand(-1, target_count, -1),
                    agent_targets,
                    partial_tokens,
                ],
                dim=-1,
            )
            row_logits = self.target_head(row_pair).squeeze(-1)
            if self.config.assignment_stickiness_logit_bonus > 0.0:
                row_logits = row_logits + (
                    self.config.assignment_stickiness_logit_bonus
                    * sticky_physical_targets[batch_indices, agent_indices]
                )
            row_mask = encoded.target_mask[batch_indices, agent_indices].clone()
            if target_count > 1:
                row_mask[:, 1:] &= counts < capacity
            row_active = active_agents[batch_indices, agent_indices]
            row_mask[~row_active] = False
            row_mask[~row_active, self.config.no_target_index] = True
            distribution, safe_mask = masked_categorical(
                row_logits,
                row_mask,
                default_index=self.config.no_target_index,
            )
            if provided_targets is None:
                selected = (
                    masked_argmax(row_logits, safe_mask, default_index=self.config.no_target_index)
                    if deterministic
                    else distribution.sample()
                )
            else:
                selected = provided_targets[batch_indices, agent_indices]
                valid_selected = safe_mask.gather(-1, selected.unsqueeze(-1)).squeeze(-1)
                if not bool(valid_selected.all()):
                    raise ValueError("target action violates detection or four-missile capacity mask")
            selected = selected.masked_fill(~row_active, self.config.no_target_index)
            targets[batch_indices, agent_indices] = selected
            log_prob[batch_indices, agent_indices] = distribution.log_prob(selected).masked_fill(~row_active, 0.0)
            entropy[batch_indices, agent_indices] = distribution.entropy().masked_fill(~row_active, 0.0)
            probabilities[batch_indices, agent_indices] = distribution.probs
            if target_count > 1:
                assigned = row_active & (selected > 0)
                if assigned.any():
                    counts[batch_indices[assigned], selected[assigned] - 1] += 1
        return targets, order, log_prob, entropy, probabilities

    def _target_outputs(
        self,
        probabilities: Tensor,
        target_indices: Tensor,
        agent_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        active = agent_mask.to(device=probabilities.device).bool().unsqueeze(-1)
        no_target = torch.zeros_like(probabilities)
        no_target[..., self.config.no_target_index] = 1.0
        probabilities = torch.where(active, probabilities, no_target)
        one_hot = torch.nn.functional.one_hot(
            target_indices,
            num_classes=probabilities.shape[-1],
        ).to(dtype=probabilities.dtype)
        assignment_matrix = one_hot[..., 1:]
        assignment_matrix = assignment_matrix * active.to(dtype=assignment_matrix.dtype)
        return probabilities, assignment_matrix

    def forward(
        self,
        inputs: AssignmentActorInputs,
        deterministic: bool = False,
    ) -> AssignmentPolicyOutput:
        encoded = self._encode(inputs)
        target, order, log_prob, entropy, probabilities = self._capacity_constrained_actions(
            encoded,
            inputs.agent_mask,
            deterministic=deterministic,
        )
        target_probabilities, assignment_matrix = self._target_outputs(
            probabilities,
            target,
            inputs.agent_mask,
        )
        return AssignmentPolicyOutput(
            actions=AssignmentActions(target=target, order=order),
            next_hidden=encoded.next_hidden,
            log_prob=log_prob,
            entropy=entropy,
            joint_log_prob=log_prob.sum(dim=-1),
            joint_entropy=entropy.sum(dim=-1),
            target_probabilities=target_probabilities,
            assignment_matrix=assignment_matrix,
        )

    def evaluate_actions(
        self,
        inputs: AssignmentActorInputs,
        actions: AssignmentActions,
    ) -> _AssignmentEvaluation:
        encoded = self._encode(inputs)
        _, _, log_prob, entropy, _ = self._capacity_constrained_actions(
            encoded,
            inputs.agent_mask,
            deterministic=False,
            target_actions=actions.target,
            order=actions.order,
        )
        return _AssignmentEvaluation(
            log_prob=log_prob,
            entropy=entropy,
            joint_log_prob=log_prob.sum(dim=-1),
            joint_entropy=entropy.sum(dim=-1),
            next_hidden=encoded.next_hidden,
        )


class OverloadBiasActor(nn.Module):
    def __init__(self, config: SwarmModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.self_encoder = build_mlp(config.d_execution_self, d, d, depth=2)
        self.friend_encoder = build_mlp(config.d_execution_friend, d, d, depth=2)
        self.target_encoder = build_mlp(config.d_execution_target, d, d, depth=2)
        self.context_encoder = build_mlp(config.d_execution_context, d, d, depth=2)
        self.friend_attention = MaskedSelfAttentionBlock(d, config.num_heads)
        self.friend_pool = AttentionPool(d)
        self.target_cross_attention = MaskedCrossAttentionBlock(d, config.num_heads)
        self.fusion = build_mlp(4 * d, d, d, depth=3)
        self.gru = nn.GRUCell(d, d)
        self.bias_mu = nn.Linear(d, config.d_bias)
        nn.init.zeros_(self.bias_mu.weight)
        nn.init.zeros_(self.bias_mu.bias)
        self.bias_log_std = nn.Parameter(torch.full((config.d_bias,), -2.5))

    def _validate_inputs(self, inputs: OverloadBiasActorInputs) -> tuple[int, int, int]:
        if inputs.self_state.dim() != 3 or inputs.self_state.shape[-1] != self.config.d_execution_self:
            raise ValueError(f"self_state must have shape [B, R, {self.config.d_execution_self}]")
        batch, agents = inputs.self_state.shape[:2]
        expected_prefix = (batch, agents)
        if inputs.same_target_friends.dim() != 4 or inputs.same_target_friends.shape[:2] != expected_prefix:
            raise ValueError("same_target_friends must have shape [B, R, F, d_friend]")
        if inputs.same_target_friends.shape[-1] != self.config.d_execution_friend:
            raise ValueError(
                f"same_target_friends feature width must be {self.config.d_execution_friend}"
            )
        friend_count = inputs.same_target_friends.shape[-2]
        if friend_count <= 0:
            raise ValueError("same_target_friends must provide at least one padded friend slot")
        if tuple(inputs.friend_mask.shape) != (batch, agents, friend_count):
            raise ValueError("friend_mask must have shape [B, R, F]")
        if tuple(inputs.assigned_target.shape) != (
            batch,
            agents,
            1,
            self.config.d_execution_target,
        ):
            raise ValueError(
                f"assigned_target must have shape [B, R, 1, {self.config.d_execution_target}]"
            )
        if tuple(inputs.target_mask.shape) != (batch, agents, 1):
            raise ValueError("target_mask must have shape [B, R, 1]")
        if tuple(inputs.environment_context.shape) != (
            batch,
            agents,
            self.config.d_execution_context,
        ):
            raise ValueError(
                "environment_context must have shape "
                f"[B, R, {self.config.d_execution_context}]"
            )
        if tuple(inputs.agent_mask.shape) != expected_prefix:
            raise ValueError("agent_mask must have shape [B, R]")
        return batch, agents, friend_count

    def _encode(self, inputs: OverloadBiasActorInputs) -> _EncodedExecutionObservation:
        batch, agents, friend_count = self._validate_inputs(inputs)
        d = self.config.d_model
        device = inputs.self_state.device
        dtype = inputs.self_state.dtype
        agent_mask = inputs.agent_mask.to(device=device).bool()
        target_mask = inputs.target_mask.to(device=device).bool()
        active = agent_mask & target_mask.any(dim=-1)

        self_token = self.self_encoder(inputs.self_state)
        friend_tokens = self.friend_encoder(
            inputs.same_target_friends.reshape(
                batch * agents,
                friend_count,
                self.config.d_execution_friend,
            )
        )
        flat_friend_mask = inputs.friend_mask.reshape(batch * agents, friend_count).to(device=device).bool()
        friend_tokens = self.friend_attention(friend_tokens, flat_friend_mask)
        friend_context = self.friend_pool(friend_tokens, flat_friend_mask).reshape(batch, agents, d)

        target_tokens = self.target_encoder(
            inputs.assigned_target.reshape(batch * agents, 1, self.config.d_execution_target)
        )
        cross_target = self.target_cross_attention(
            self_token.reshape(batch * agents, 1, d),
            target_tokens,
            agent_mask.reshape(batch * agents, 1),
            target_mask.reshape(batch * agents, 1),
        ).reshape(batch, agents, d)
        context_token = self.context_encoder(
            inputs.environment_context.to(device=device, dtype=dtype)
        )
        fused = self.fusion(
            torch.cat([self_token, friend_context, cross_target, context_token], dim=-1)
        )
        hidden = torch.zeros_like(fused) if inputs.hidden is None else inputs.hidden
        if hidden.shape != fused.shape:
            raise ValueError(f"hidden shape {tuple(hidden.shape)} does not match {tuple(fused.shape)}")
        hidden = hidden.to(device=device, dtype=dtype)
        next_hidden = self.gru(
            fused.reshape(batch * agents, d),
            hidden.reshape(batch * agents, d),
        ).reshape(batch, agents, d)
        next_hidden = next_hidden.masked_fill(~active.unsqueeze(-1), 0.0)
        return _EncodedExecutionObservation(next_hidden=next_hidden, active=active)

    def _distribution_parameters(self, hidden: Tensor) -> tuple[Tensor, Tensor]:
        mu = self.bias_mu(hidden)
        log_std = self.bias_log_std.expand_as(mu).clamp(
            self.config.log_std_min,
            self.config.log_std_max,
        )
        return mu, log_std

    def forward(
        self,
        inputs: OverloadBiasActorInputs,
        deterministic: bool = False,
    ) -> OverloadBiasOutput:
        encoded = self._encode(inputs)
        mu, log_std = self._distribution_parameters(encoded.next_hidden)
        if self.config.execution_action_distribution == "radial_tanh_disk":
            bias_matrix, log_prob, entropy = _radial_tanh_sample(
                mu,
                log_std,
                deterministic,
            )
        else:
            bias_matrix, log_prob, entropy = _tanh_normal_sample(
                mu,
                log_std,
                deterministic,
            )
        bias_matrix = bias_matrix.masked_fill(~encoded.active.unsqueeze(-1), 0.0)
        log_prob = log_prob.masked_fill(~encoded.active, 0.0)
        entropy = entropy.masked_fill(~encoded.active, 0.0)
        return OverloadBiasOutput(
            bias_matrix=bias_matrix,
            next_hidden=encoded.next_hidden,
            log_prob=log_prob,
            entropy=entropy,
            action_distribution=self.config.execution_action_distribution,
        )

    def evaluate_actions(
        self,
        inputs: OverloadBiasActorInputs,
        bias_matrix: Tensor,
    ) -> _OverloadBiasEvaluation:
        encoded = self._encode(inputs)
        expected_shape = (*encoded.active.shape, self.config.d_bias)
        if tuple(bias_matrix.shape) != expected_shape:
            raise ValueError(
                f"bias_matrix shape {tuple(bias_matrix.shape)} does not match {expected_shape}"
            )
        bias_matrix = bias_matrix.to(device=encoded.next_hidden.device, dtype=encoded.next_hidden.dtype)
        mu, log_std = self._distribution_parameters(encoded.next_hidden)
        log_prob = (
            _radial_tanh_log_prob(bias_matrix, mu, log_std)
            if self.config.execution_action_distribution == "radial_tanh_disk"
            else _tanh_normal_log_prob(bias_matrix, mu, log_std)
        )
        entropy = Normal(mu, torch.exp(log_std)).entropy().sum(dim=-1)
        return _OverloadBiasEvaluation(
            log_prob=log_prob.masked_fill(~encoded.active, 0.0),
            entropy=entropy.masked_fill(~encoded.active, 0.0),
            next_hidden=encoded.next_hidden,
        )
