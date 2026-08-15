from __future__ import annotations

import torch
from torch import Tensor


def _align_as(x: Tensor, target: Tensor) -> Tensor:
    while x.dim() < target.dim():
        x = x.unsqueeze(-1)
    return x


def generalized_advantage_estimation(
    rewards: Tensor,
    values: Tensor,
    dones: Tensor,
    gamma: float | Tensor,
    gae_lambda: float | Tensor,
) -> tuple[Tensor, Tensor]:
    """Compute TD(lambda)/GAE by reverse recursion.

    rewards: [T, ...]
    values: [T + 1, ...] or broadcastable to rewards
    dones: [T, ...] with 1 marking episode termination after the step
    """
    if values.shape[0] != rewards.shape[0] + 1:
        raise ValueError("values must have T + 1 entries")
    values_aligned = _align_as(values, rewards)
    dones_aligned = _align_as(dones, rewards).to(rewards.dtype)
    def align_parameter(value: float | Tensor) -> float | Tensor:
        if not isinstance(value, Tensor):
            return value
        tensor = value.to(device=rewards.device, dtype=rewards.dtype)
        return tensor if tensor.dim() == 0 else _align_as(tensor, rewards)

    gamma_aligned = align_parameter(gamma)
    lambda_aligned = align_parameter(gae_lambda)
    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        non_terminal = 1.0 - dones_aligned[t]
        gamma_t = (
            gamma_aligned
            if not isinstance(gamma_aligned, Tensor) or gamma_aligned.dim() == 0
            else gamma_aligned[t]
        )
        lambda_t = (
            lambda_aligned
            if not isinstance(lambda_aligned, Tensor) or lambda_aligned.dim() == 0
            else lambda_aligned[t]
        )
        delta = rewards[t] + gamma_t * non_terminal * values_aligned[t + 1] - values_aligned[t]
        last_advantage = delta + gamma_t * lambda_t * non_terminal * last_advantage
        advantages[t] = last_advantage
    returns = advantages + values_aligned[:-1]
    return advantages, returns
