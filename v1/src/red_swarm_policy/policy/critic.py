from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn

from ..core.config import SwarmModelConfig
from ..core.networks import (
    AttentionPool,
    MaskedCrossAttentionBlock,
    MaskedSelfAttentionBlock,
    build_mlp,
)


@dataclass
class AssignmentCriticInputs:
    global_red: Tensor
    red_mask: Tensor
    global_blue: Tensor
    blue_mask: Tensor
    global_context: Tensor
    target_assignment_counts: Tensor
    pair_state: Tensor
    current_assignment: Tensor

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> "AssignmentCriticInputs":
        return AssignmentCriticInputs(
            global_red=transform(self.global_red),
            red_mask=transform(self.red_mask),
            global_blue=transform(self.global_blue),
            blue_mask=transform(self.blue_mask),
            global_context=transform(self.global_context),
            target_assignment_counts=transform(self.target_assignment_counts),
            pair_state=transform(self.pair_state),
            current_assignment=transform(self.current_assignment),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "AssignmentCriticInputs":
        return self._map_tensors(
            lambda value: value.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "AssignmentCriticInputs":
        return self._map_tensors(lambda value: value.pin_memory())

    def detached(self) -> "AssignmentCriticInputs":
        return self._map_tensors(lambda value: value.detach())


@dataclass
class AssignmentCriticOutput:
    value: Tensor
    value_components: Tensor


@dataclass
class OverloadBiasCriticInputs:
    global_red: Tensor
    red_mask: Tensor
    global_blue: Tensor
    blue_mask: Tensor
    applied_bias: Tensor
    global_context: Tensor
    pair_state: Tensor
    current_assignment: Tensor
    hidden: Optional[Tensor] = None

    def _map_tensors(self, transform: Callable[[Tensor], Tensor]) -> "OverloadBiasCriticInputs":
        return OverloadBiasCriticInputs(
            global_red=transform(self.global_red),
            red_mask=transform(self.red_mask),
            global_blue=transform(self.global_blue),
            blue_mask=transform(self.blue_mask),
            applied_bias=transform(self.applied_bias),
            global_context=transform(self.global_context),
            pair_state=transform(self.pair_state),
            current_assignment=transform(self.current_assignment),
            hidden=None if self.hidden is None else transform(self.hidden),
        )

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "OverloadBiasCriticInputs":
        return self._map_tensors(
            lambda value: value.to(device, non_blocking=non_blocking)
        )

    def pin_memory(self) -> "OverloadBiasCriticInputs":
        return self._map_tensors(lambda value: value.pin_memory())

    def detached(self) -> "OverloadBiasCriticInputs":
        return self._map_tensors(lambda value: value.detach())


@dataclass
class OverloadBiasCriticOutput:
    value: Tensor
    value_components: Tensor
    next_hidden: Tensor


class TargetAssignmentCritic(nn.Module):
    def __init__(self, config: SwarmModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.red_encoder = build_mlp(config.d_global_red, d, d, depth=3)
        self.blue_encoder = build_mlp(config.d_global_blue, d, d, depth=3)
        self.assignment_encoder = build_mlp(1, d, d, depth=2)
        self.pair_encoder = build_mlp(config.d_pair + 1, d, d, depth=2)
        self.red_attention = MaskedSelfAttentionBlock(d, config.num_heads)
        self.blue_attention = MaskedSelfAttentionBlock(d, config.num_heads)
        self.red_from_blue = MaskedCrossAttentionBlock(d, config.num_heads)
        self.blue_from_red = MaskedCrossAttentionBlock(d, config.num_heads)
        self.red_pool = AttentionPool(d)
        self.blue_pool = AttentionPool(d)
        self.assignment_pool = AttentionPool(d)
        self.pair_pool = AttentionPool(d)
        self.context_encoder = build_mlp(config.d_global_context, d, d, depth=2)
        self.fusion = build_mlp(5 * d, d, d, depth=3)
        self.value_head = nn.Linear(d, config.assignment_value_components)

    def _validate_inputs(self, inputs: AssignmentCriticInputs) -> tuple[int, int, int]:
        if inputs.global_red.dim() != 3 or inputs.global_red.shape[-1] != self.config.d_global_red:
            raise ValueError(f"global_red must have shape [B, R, {self.config.d_global_red}]")
        batch, red_count = inputs.global_red.shape[:2]
        if red_count <= 0:
            raise ValueError("global_red must contain at least one entity slot")
        if tuple(inputs.red_mask.shape) != (batch, red_count):
            raise ValueError("red_mask must have shape [B, R]")
        if inputs.global_blue.dim() != 3 or inputs.global_blue.shape[0] != batch:
            raise ValueError("global_blue must have shape [B, U, d_global_blue]")
        if inputs.global_blue.shape[-1] != self.config.d_global_blue:
            raise ValueError(f"global_blue feature width must be {self.config.d_global_blue}")
        blue_count = inputs.global_blue.shape[1]
        if blue_count <= 0:
            raise ValueError("global_blue must contain at least one entity slot")
        if tuple(inputs.blue_mask.shape) != (batch, blue_count):
            raise ValueError("blue_mask must have shape [B, U]")
        if tuple(inputs.target_assignment_counts.shape) != (batch, blue_count):
            raise ValueError("target_assignment_counts must have shape [B, U]")
        if tuple(inputs.pair_state.shape) != (
            batch,
            red_count,
            blue_count,
            self.config.d_pair,
        ):
            raise ValueError(
                f"pair_state must have shape [B, R, U, {self.config.d_pair}]"
            )
        if tuple(inputs.current_assignment.shape) != (batch, red_count, blue_count):
            raise ValueError("current_assignment must have shape [B, R, U]")
        if tuple(inputs.global_context.shape) != (batch, self.config.d_global_context):
            raise ValueError(
                f"global_context must have shape [B, {self.config.d_global_context}]"
            )
        return batch, red_count, blue_count

    def forward(self, inputs: AssignmentCriticInputs) -> AssignmentCriticOutput:
        self._validate_inputs(inputs)
        device = inputs.global_red.device
        dtype = inputs.global_red.dtype
        red_mask = inputs.red_mask.to(device=device).bool()
        blue_mask = inputs.blue_mask.to(device=device).bool()
        red_tokens = self.red_attention(self.red_encoder(inputs.global_red), red_mask)
        blue_tokens = self.blue_attention(self.blue_encoder(inputs.global_blue), blue_mask)
        red_cross = self.red_from_blue(red_tokens, blue_tokens, red_mask, blue_mask)
        blue_cross = self.blue_from_red(blue_tokens, red_tokens, blue_mask, red_mask)
        has_blue = blue_mask.any(dim=-1, keepdim=True).unsqueeze(-1)
        has_red = red_mask.any(dim=-1, keepdim=True).unsqueeze(-1)
        red_tokens = torch.where(has_blue, red_cross, red_tokens)
        blue_tokens = torch.where(has_red, blue_cross, blue_tokens)
        assignment_counts = inputs.target_assignment_counts.to(device=device, dtype=dtype)
        assignment_tokens = self.assignment_encoder(assignment_counts.unsqueeze(-1))
        assignment_tokens = assignment_tokens.masked_fill(~blue_mask.unsqueeze(-1), 0.0)
        red_set = self.red_pool(red_tokens, red_mask)
        blue_set = self.blue_pool(blue_tokens + assignment_tokens, blue_mask)
        assignment_set = self.assignment_pool(assignment_tokens, blue_mask)
        pair_features = torch.cat(
            [
                inputs.pair_state.to(device=device, dtype=dtype),
                inputs.current_assignment.to(device=device, dtype=dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        pair_tokens = self.pair_encoder(pair_features)
        pair_mask = red_mask.unsqueeze(-1) & blue_mask.unsqueeze(1)
        pair_set = self.pair_pool(
            pair_tokens.reshape(pair_tokens.shape[0], -1, pair_tokens.shape[-1]),
            pair_mask.reshape(pair_mask.shape[0], -1),
        )
        context = self.context_encoder(inputs.global_context.to(device=device, dtype=dtype))
        state = self.fusion(
            torch.cat([red_set, blue_set, assignment_set, pair_set, context], dim=-1)
        )
        components = self.value_head(state)
        return AssignmentCriticOutput(
            value=components.sum(dim=-1),
            value_components=components,
        )


class OverloadBiasCritic(nn.Module):
    def __init__(self, config: SwarmModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        d = config.d_model
        self.red_encoder = build_mlp(config.d_global_red, d, d, depth=3)
        self.blue_encoder = build_mlp(config.d_global_blue, d, d, depth=3)
        self.bias_encoder = build_mlp(config.d_bias, d, d, depth=2)
        self.pair_encoder = build_mlp(config.d_pair + 1, d, d, depth=2)
        self.red_attention = MaskedSelfAttentionBlock(d, config.num_heads)
        self.blue_attention = MaskedSelfAttentionBlock(d, config.num_heads)
        self.red_from_blue = MaskedCrossAttentionBlock(d, config.num_heads)
        self.blue_from_red = MaskedCrossAttentionBlock(d, config.num_heads)
        self.red_pool = AttentionPool(d)
        self.blue_pool = AttentionPool(d)
        self.pair_pool = AttentionPool(d)
        self.context_encoder = build_mlp(config.d_global_context, d, d, depth=2)
        self.fusion = build_mlp(5 * d, d, d, depth=3)
        self.gru = nn.GRUCell(d, d)
        self.value_head = nn.Linear(d, config.d_value_components)

    def _validate_inputs(self, inputs: OverloadBiasCriticInputs) -> tuple[int, int, int]:
        if inputs.global_red.dim() != 3 or inputs.global_red.shape[-1] != self.config.d_global_red:
            raise ValueError(f"global_red must have shape [B, R, {self.config.d_global_red}]")
        batch, red_count = inputs.global_red.shape[:2]
        if red_count <= 0:
            raise ValueError("global_red must contain at least one entity slot")
        if tuple(inputs.red_mask.shape) != (batch, red_count):
            raise ValueError("red_mask must have shape [B, R]")
        if inputs.global_blue.dim() != 3 or inputs.global_blue.shape[0] != batch:
            raise ValueError("global_blue must have shape [B, U, d_global_blue]")
        if inputs.global_blue.shape[-1] != self.config.d_global_blue:
            raise ValueError(f"global_blue feature width must be {self.config.d_global_blue}")
        blue_count = inputs.global_blue.shape[1]
        if blue_count <= 0:
            raise ValueError("global_blue must contain at least one entity slot")
        if tuple(inputs.blue_mask.shape) != (batch, blue_count):
            raise ValueError("blue_mask must have shape [B, U]")
        if tuple(inputs.applied_bias.shape) != (batch, red_count, self.config.d_bias):
            raise ValueError(
                f"applied_bias must have shape [B, R, {self.config.d_bias}]"
            )
        if tuple(inputs.pair_state.shape) != (
            batch,
            red_count,
            blue_count,
            self.config.d_pair,
        ):
            raise ValueError(
                f"pair_state must have shape [B, R, U, {self.config.d_pair}]"
            )
        if tuple(inputs.current_assignment.shape) != (batch, red_count, blue_count):
            raise ValueError("current_assignment must have shape [B, R, U]")
        if tuple(inputs.global_context.shape) != (batch, self.config.d_global_context):
            raise ValueError(
                f"global_context must have shape [B, {self.config.d_global_context}]"
            )
        return batch, red_count, blue_count

    def forward(self, inputs: OverloadBiasCriticInputs) -> OverloadBiasCriticOutput:
        batch, red_count, _ = self._validate_inputs(inputs)
        d = self.config.d_model
        device = inputs.global_red.device
        dtype = inputs.global_red.dtype
        red_mask = inputs.red_mask.to(device=device).bool()
        blue_mask = inputs.blue_mask.to(device=device).bool()

        bias_tokens = self.bias_encoder(inputs.applied_bias.to(device=device, dtype=dtype))
        bias_tokens = bias_tokens.masked_fill(~red_mask.unsqueeze(-1), 0.0)
        red_tokens = self.red_attention(
            self.red_encoder(inputs.global_red) + bias_tokens,
            red_mask,
        )
        blue_tokens = self.blue_attention(self.blue_encoder(inputs.global_blue), blue_mask)

        red_cross = self.red_from_blue(red_tokens, blue_tokens, red_mask, blue_mask)
        blue_cross = self.blue_from_red(blue_tokens, red_tokens, blue_mask, red_mask)
        has_blue = blue_mask.any(dim=-1, keepdim=True).unsqueeze(-1)
        has_red = red_mask.any(dim=-1, keepdim=True).unsqueeze(-1)
        red_tokens = torch.where(has_blue, red_cross, red_tokens)
        blue_tokens = torch.where(has_red, blue_cross, blue_tokens)

        red_set = self.red_pool(red_tokens, red_mask)
        blue_set = self.blue_pool(blue_tokens, blue_mask)
        pair_features = torch.cat(
            [
                inputs.pair_state.to(device=device, dtype=dtype),
                inputs.current_assignment.to(device=device, dtype=dtype).unsqueeze(-1),
            ],
            dim=-1,
        )
        pair_tokens = self.pair_encoder(pair_features)
        pair_mask = red_mask.unsqueeze(-1) & blue_mask.unsqueeze(1)
        pair_context = self.pair_pool(
            pair_tokens.reshape(batch * red_count, -1, d),
            pair_mask.reshape(batch * red_count, -1),
        ).reshape(batch, red_count, d)
        context = self.context_encoder(inputs.global_context.to(device=device, dtype=dtype))
        global_tokens = torch.cat([red_set, blue_set, context], dim=-1).unsqueeze(1)
        global_tokens = global_tokens.expand(-1, red_count, -1)
        fused = self.fusion(torch.cat([red_tokens, pair_context, global_tokens], dim=-1))
        hidden = torch.zeros_like(fused) if inputs.hidden is None else inputs.hidden
        expected_hidden_shape = (batch, red_count, d)
        if tuple(hidden.shape) != expected_hidden_shape:
            raise ValueError(
                f"hidden shape {tuple(hidden.shape)} does not match {expected_hidden_shape}"
            )
        hidden = hidden.to(device=device, dtype=dtype)
        next_hidden = self.gru(
            fused.reshape(batch * red_count, d),
            hidden.reshape(batch * red_count, d),
        ).reshape(batch, red_count, d)
        next_hidden = next_hidden.masked_fill(~red_mask.unsqueeze(-1), 0.0)
        components = self.value_head(next_hidden).masked_fill(~red_mask.unsqueeze(-1), 0.0)
        return OverloadBiasCriticOutput(
            value=components.sum(dim=-1),
            value_components=components,
            next_hidden=next_hidden,
        )
