from __future__ import annotations

import torch
from torch import Tensor, nn

from .masks import apply_mask_to_logits, ensure_at_least_one_valid


def build_mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    depth: int = 2,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    if depth < 1:
        raise ValueError("depth must be at least 1")
    layers: list[nn.Module] = []
    last = in_dim
    for _ in range(depth - 1):
        layers.extend([nn.Linear(last, hidden_dim), activation()])
        last = hidden_dim
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class MaskedSelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        safe_mask, _ = ensure_at_least_one_valid(mask, default_index=0)
        x = x.masked_fill(~safe_mask.unsqueeze(-1), 0.0)
        attn_out, _ = self.attn(
            x,
            x,
            x,
            key_padding_mask=~safe_mask,
            need_weights=False,
        )
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x.masked_fill(~safe_mask.unsqueeze(-1), 0.0)


class MaskedCrossAttentionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.SiLU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(
        self,
        query: Tensor,
        key_value: Tensor,
        query_mask: Tensor,
        key_value_mask: Tensor,
    ) -> Tensor:
        if query.dim() != 3 or key_value.dim() != 3:
            raise ValueError("query and key_value must have shape [batch, entities, features]")
        if query.shape[0] != key_value.shape[0] or query.shape[-1] != key_value.shape[-1]:
            raise ValueError("query and key_value batch/feature dimensions must match")
        expected_query_mask = query.shape[:2]
        expected_key_value_mask = key_value.shape[:2]
        if tuple(query_mask.shape) != expected_query_mask:
            raise ValueError(
                f"query_mask shape {tuple(query_mask.shape)} does not match {expected_query_mask}"
            )
        if tuple(key_value_mask.shape) != expected_key_value_mask:
            raise ValueError(
                "key_value_mask shape "
                f"{tuple(key_value_mask.shape)} does not match {expected_key_value_mask}"
            )

        query_mask = query_mask.to(device=query.device).bool()
        original_key_mask = key_value_mask.to(device=key_value.device).bool()
        safe_key_mask, _ = ensure_at_least_one_valid(original_key_mask, default_index=0)
        query = query.masked_fill(~query_mask.unsqueeze(-1), 0.0)
        key_value = key_value.masked_fill(~safe_key_mask.unsqueeze(-1), 0.0)
        attn_out, _ = self.attn(
            query,
            key_value,
            key_value,
            key_padding_mask=~safe_key_mask,
            need_weights=False,
        )
        output = self.norm1(query + attn_out)
        output = self.norm2(output + self.ff(output))
        has_key = original_key_mask.any(dim=-1, keepdim=True)
        valid_query = query_mask & has_key
        return output.masked_fill(~valid_query.unsqueeze(-1), 0.0)


class AttentionPool(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.score = nn.Linear(d_model, 1)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        original_mask = mask.bool()
        logits = self.score(torch.tanh(x)).squeeze(-1)
        masked_logits, _ = apply_mask_to_logits(logits, mask, default_index=0)
        weights = torch.softmax(masked_logits, dim=-1)
        pooled = torch.sum(weights.unsqueeze(-1) * x, dim=-2)
        valid = original_mask.any(dim=-1, keepdim=True)
        return pooled.masked_fill(~valid, 0.0)
