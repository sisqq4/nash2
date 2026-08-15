from __future__ import annotations

from typing import Tuple

from torch import Tensor
from torch.distributions import Categorical

NEG_INF = -1.0e9


def as_bool_mask(mask: Tensor) -> Tensor:
    return mask.bool()


def ensure_at_least_one_valid(mask: Tensor, default_index: int = 0) -> Tuple[Tensor, Tensor]:
    """Return a mask with one valid fallback per row and a row-level all-invalid flag."""
    if mask.numel() == 0:
        raise ValueError("mask must have at least one element")
    mask_bool = as_bool_mask(mask).clone()
    if default_index >= mask_bool.shape[-1]:
        raise ValueError("default_index is outside the mask width")
    all_invalid = ~mask_bool.any(dim=-1)
    if all_invalid.any():
        flat = mask_bool.reshape(-1, mask_bool.shape[-1])
        flat_all_invalid = all_invalid.reshape(-1)
        flat[flat_all_invalid, default_index] = True
    return mask_bool, all_invalid


def apply_mask_to_logits(
    logits: Tensor,
    mask: Tensor,
    default_index: int = 0,
) -> Tuple[Tensor, Tensor]:
    safe_mask, _ = ensure_at_least_one_valid(mask, default_index=default_index)
    masked = logits.masked_fill(~safe_mask, NEG_INF)
    return masked, safe_mask


def masked_categorical(
    logits: Tensor,
    mask: Tensor,
    default_index: int = 0,
) -> Tuple[Categorical, Tensor]:
    masked_logits, safe_mask = apply_mask_to_logits(
        logits,
        mask,
        default_index=default_index,
    )
    return Categorical(logits=masked_logits), safe_mask


def masked_argmax(logits: Tensor, mask: Tensor, default_index: int = 0) -> Tensor:
    masked_logits, _ = apply_mask_to_logits(logits, mask, default_index=default_index)
    return masked_logits.argmax(dim=-1)
