from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..core.config import PPOConfig
from ..policy.actor import (
    AssignmentActions,
    AssignmentActorInputs,
    OverloadBiasActor,
    OverloadBiasActorInputs,
    TargetAssignmentActor,
)
from ..policy.critic import (
    AssignmentCriticInputs,
    OverloadBiasCritic,
    OverloadBiasCriticInputs,
    TargetAssignmentCritic,
)
from .gae import generalized_advantage_estimation


@dataclass
class MAPPOBatch:
    assignment_actor_inputs: AssignmentActorInputs
    execution_actor_inputs: OverloadBiasActorInputs
    assignment_actions: AssignmentActions
    bias_matrices: Tensor
    old_assignment_log_prob: Tensor
    old_execution_log_prob: Tensor
    rewards_high: Tensor
    durations_high_s: Tensor
    high_reference_interval_s: float
    rewards_low: Tensor
    durations_low_s: Tensor
    low_reference_interval_s: float
    dones_high: Tensor
    dones_low: Tensor
    episode_active_high: Tensor
    episode_active_low: Tensor
    assignment_critic_inputs: AssignmentCriticInputs
    execution_critic_inputs: OverloadBiasCriticInputs
    scenario_red_counts: Tensor | None = None
    scenario_blue_counts: Tensor | None = None

    def pin_memory(self) -> "MAPPOBatch":
        """Return a CPU-pinned rollout batch suitable for async CUDA transfer."""

        def pin_optional(value: Tensor | None) -> Tensor | None:
            return None if value is None else value.pin_memory()

        return replace(
            self,
            assignment_actor_inputs=self.assignment_actor_inputs.pin_memory(),
            execution_actor_inputs=self.execution_actor_inputs.pin_memory(),
            assignment_actions=self.assignment_actions.pin_memory(),
            bias_matrices=self.bias_matrices.pin_memory(),
            old_assignment_log_prob=self.old_assignment_log_prob.pin_memory(),
            old_execution_log_prob=self.old_execution_log_prob.pin_memory(),
            rewards_high=self.rewards_high.pin_memory(),
            durations_high_s=self.durations_high_s.pin_memory(),
            rewards_low=self.rewards_low.pin_memory(),
            durations_low_s=self.durations_low_s.pin_memory(),
            dones_high=self.dones_high.pin_memory(),
            dones_low=self.dones_low.pin_memory(),
            episode_active_high=self.episode_active_high.pin_memory(),
            episode_active_low=self.episode_active_low.pin_memory(),
            assignment_critic_inputs=self.assignment_critic_inputs.pin_memory(),
            execution_critic_inputs=self.execution_critic_inputs.pin_memory(),
            scenario_red_counts=pin_optional(self.scenario_red_counts),
            scenario_blue_counts=pin_optional(self.scenario_blue_counts),
        )

    def training_view(
        self,
        device: torch.device,
        *,
        assignment: bool,
        execution: bool,
    ) -> "MAPPOBatch":
        """Move compact training tensors while observations stay CPU-pinned."""

        def move(value: Tensor) -> Tensor:
            return value.to(device, non_blocking=True)

        def move_optional(value: Tensor | None) -> Tensor | None:
            return None if value is None else move(value)

        updates: dict[str, Any] = {
            "rewards_high": move(self.rewards_high),
            "durations_high_s": move(self.durations_high_s),
            "rewards_low": move(self.rewards_low),
            "durations_low_s": move(self.durations_low_s),
            "dones_high": move(self.dones_high),
            "dones_low": move(self.dones_low),
            "episode_active_high": move(self.episode_active_high),
            "episode_active_low": move(self.episode_active_low),
            "scenario_red_counts": move_optional(self.scenario_red_counts),
            "scenario_blue_counts": move_optional(self.scenario_blue_counts),
        }
        if assignment:
            updates["old_assignment_log_prob"] = move(
                self.old_assignment_log_prob
            )
        if execution:
            updates["old_execution_log_prob"] = move(
                self.old_execution_log_prob
            )
        return replace(self, **updates)


@dataclass(frozen=True)
class AdvantageTargets:
    assignment_raw_advantage: Tensor
    assignment_advantage: Tensor
    assignment_return: Tensor
    execution_raw_advantage: Tensor
    execution_advantage: Tensor
    execution_return: Tensor
    assignment_value_before_update: Tensor
    execution_value_before_update: Tensor


def _normalize_advantage(advantage: Tensor, active: Tensor) -> Tensor:
    active = active.to(device=advantage.device).bool()
    selected = advantage[active]
    if selected.numel() == 0:
        return torch.zeros_like(advantage)
    mean = selected.mean()
    std = selected.std(unbiased=False)
    return torch.where(active, (advantage - mean) / (std + 1.0e-8), torch.zeros_like(advantage))


def _execution_scenario_masks(
    batch: MAPPOBatch,
    active: Tensor,
) -> list[tuple[int, int, Tensor]]:
    if batch.scenario_red_counts is None or batch.scenario_blue_counts is None:
        raise ValueError("per-scenario execution semantics require scenario metadata")
    if active.dim() != 3:
        raise ValueError("execution active mask must have shape [T, B, A]")
    batch_size = active.shape[1]
    red_counts = batch.scenario_red_counts.to(device=active.device, dtype=torch.long)
    blue_counts = batch.scenario_blue_counts.to(device=active.device, dtype=torch.long)
    if tuple(red_counts.shape) != (batch_size,) or tuple(blue_counts.shape) != (
        batch_size,
    ):
        raise ValueError("scenario count metadata must have shape [B]")
    pairs = torch.stack((red_counts, blue_counts), dim=-1)
    masks: list[tuple[int, int, Tensor]] = []
    for red_count, blue_count in torch.unique(pairs, dim=0).detach().cpu().tolist():
        environment_mask = (red_counts == red_count) & (blue_counts == blue_count)
        scenario_mask = active & environment_mask.view(1, batch_size, 1)
        if bool(scenario_mask.any()):
            masks.append((int(red_count), int(blue_count), scenario_mask))
    if not masks and bool(active.any()):
        raise ValueError("active execution samples do not match scenario metadata")
    return masks


def _normalize_execution_advantage_per_scenario(
    batch: MAPPOBatch,
    advantage: Tensor,
    active: Tensor,
) -> Tensor:
    normalized = torch.zeros_like(advantage)
    for _, _, scenario_mask in _execution_scenario_masks(batch, active):
        scenario_normalized = _normalize_advantage(advantage, scenario_mask)
        normalized = torch.where(scenario_mask, scenario_normalized, normalized)
    return normalized


def _execution_actor_weights(
    batch: MAPPOBatch,
    active: Tensor,
    mode: str,
) -> Tensor:
    if mode == "active_step":
        return active.to(dtype=batch.rewards_low.dtype)
    if mode != "per_scenario":
        raise ValueError(f"unsupported execution actor loss weighting: {mode}")
    weights = torch.zeros_like(active, dtype=batch.rewards_low.dtype)
    for _, _, scenario_mask in _execution_scenario_masks(batch, active):
        count = scenario_mask.sum().to(dtype=weights.dtype).clamp_min(1.0)
        weights = torch.where(scenario_mask, torch.ones_like(weights) / count, weights)
    return weights


def _explained_variance(prediction: Tensor, target: Tensor, active: Tensor) -> Tensor:
    selected_prediction = prediction[active.bool()]
    selected_target = target[active.bool()]
    if selected_target.numel() < 2:
        return torch.zeros((), device=target.device, dtype=target.dtype)
    target_variance = selected_target.var(unbiased=False)
    if target_variance <= 1.0e-12:
        return torch.zeros((), device=target.device, dtype=target.dtype)
    residual_variance = (selected_target - selected_prediction).var(unbiased=False)
    return 1.0 - residual_variance / target_variance


def _masked_value_loss(
    prediction: Tensor,
    target: Tensor,
    active: Tensor,
    *,
    kind: str,
    huber_delta: float,
) -> tuple[Tensor, Tensor, Tensor]:
    mask = active.to(device=prediction.device).bool()
    count = mask.sum().to(dtype=prediction.dtype).clamp_min(1.0)
    squared_error = (prediction - target).pow(2)
    if kind == "mse":
        training_error = squared_error
    elif kind == "huber":
        training_error = F.huber_loss(
            prediction,
            target,
            reduction="none",
            delta=huber_delta,
        )
    else:
        raise ValueError(f"unsupported value loss: {kind}")
    mask_float = mask.to(dtype=prediction.dtype)
    return (
        (training_error * mask_float).sum(),
        (squared_error * mask_float).sum(),
        count,
    )


def _selected_summary(prefix: str, values: Tensor, active: Tensor) -> dict[str, float]:
    selected = values[active.bool()]
    if selected.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_p05": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p95": 0.0,
        }
    selected = selected.float()
    quantiles = torch.quantile(
        selected,
        torch.tensor((0.05, 0.50, 0.95), device=selected.device),
    )
    return {
        f"{prefix}_mean": float(selected.mean().cpu()),
        f"{prefix}_std": float(selected.std(unbiased=False).cpu()),
        f"{prefix}_p05": float(quantiles[0].cpu()),
        f"{prefix}_p50": float(quantiles[1].cpu()),
        f"{prefix}_p95": float(quantiles[2].cpu()),
    }


def _execution_scenario_diagnostics(
    batch: MAPPOBatch,
    active: Tensor,
    weights: Tensor,
    targets: AdvantageTargets,
    execution_value_after_update: Tensor,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    total_weight = weights.sum().clamp_min(1.0e-12)
    residual = targets.execution_return - targets.execution_value_before_update
    residual_after_update = targets.execution_return - execution_value_after_update
    for red_count, blue_count, scenario_mask in _execution_scenario_masks(batch, active):
        prefix = f"execution_scenario_r{red_count}_b{blue_count}"
        metrics[f"{prefix}_active_steps"] = float(scenario_mask.sum().cpu())
        metrics[f"{prefix}_actor_weight_share"] = float(
            (weights[scenario_mask].sum() / total_weight).cpu()
        )
        raw_advantage = targets.execution_raw_advantage[scenario_mask]
        raw_advantage_std = raw_advantage.std(unbiased=False)
        metrics[f"{prefix}_raw_advantage_snr"] = float(
            (
                raw_advantage.mean().abs()
                / (raw_advantage_std + 1.0e-8)
            ).cpu()
        )
        metrics[f"{prefix}_explained_variance_before_update"] = float(
            _explained_variance(
                targets.execution_value_before_update,
                targets.execution_return,
                scenario_mask,
            ).cpu()
        )
        metrics[f"{prefix}_explained_variance_after_update"] = float(
            _explained_variance(
                execution_value_after_update,
                targets.execution_return,
                scenario_mask,
            ).cpu()
        )
        metrics.update(
            _selected_summary(
                f"{prefix}_raw_advantage",
                targets.execution_raw_advantage,
                scenario_mask,
            )
        )
        metrics.update(
            _selected_summary(
                f"{prefix}_normalized_advantage",
                targets.execution_advantage,
                scenario_mask,
            )
        )
        metrics.update(
            _selected_summary(
                f"{prefix}_return",
                targets.execution_return,
                scenario_mask,
            )
        )
        metrics.update(
            _selected_summary(
                f"{prefix}_value_residual",
                residual,
                scenario_mask,
            )
        )
        metrics.update(
            _selected_summary(
                f"{prefix}_value_residual_after_update",
                residual_after_update,
                scenario_mask,
            )
        )
    return metrics


def _slice_assignment_inputs(
    inputs: AssignmentActorInputs,
    time_index: int,
    hidden: Tensor | None,
) -> AssignmentActorInputs:
    return AssignmentActorInputs(
        self_state=inputs.self_state[time_index],
        friend_entities=inputs.friend_entities[time_index],
        friend_mask=inputs.friend_mask[time_index],
        target_entities=inputs.target_entities[time_index],
        pair_state=inputs.pair_state[time_index],
        current_assignment=inputs.current_assignment[time_index],
        target_mask=inputs.target_mask[time_index],
        environment_context=inputs.environment_context[time_index],
        target_assignment_counts=inputs.target_assignment_counts[time_index],
        target_entity_mask=inputs.target_entity_mask[time_index],
        agent_mask=inputs.agent_mask[time_index],
        hidden=hidden,
    )


def _slice_execution_inputs(
    inputs: OverloadBiasActorInputs,
    time_index: int,
    hidden: Tensor | None,
) -> OverloadBiasActorInputs:
    return OverloadBiasActorInputs(
        self_state=inputs.self_state[time_index],
        same_target_friends=inputs.same_target_friends[time_index],
        friend_mask=inputs.friend_mask[time_index],
        assigned_target=inputs.assigned_target[time_index],
        target_mask=inputs.target_mask[time_index],
        environment_context=inputs.environment_context[time_index],
        agent_mask=inputs.agent_mask[time_index],
        hidden=hidden,
    )


def _slice_assignment_actions(actions: AssignmentActions, time_index: int) -> AssignmentActions:
    return AssignmentActions(
        target=actions.target[time_index],
        order=None if actions.order is None else actions.order[time_index],
    )


def _slice_assignment_critic_inputs(
    inputs: AssignmentCriticInputs,
    time_index: int,
) -> AssignmentCriticInputs:
    return AssignmentCriticInputs(
        global_red=inputs.global_red[time_index],
        red_mask=inputs.red_mask[time_index],
        global_blue=inputs.global_blue[time_index],
        blue_mask=inputs.blue_mask[time_index],
        global_context=inputs.global_context[time_index],
        target_assignment_counts=inputs.target_assignment_counts[time_index],
        pair_state=inputs.pair_state[time_index],
        current_assignment=inputs.current_assignment[time_index],
    )


def _slice_execution_critic_inputs(
    inputs: OverloadBiasCriticInputs,
    time_index: int,
    hidden: Tensor | None,
) -> OverloadBiasCriticInputs:
    return OverloadBiasCriticInputs(
        global_red=inputs.global_red[time_index],
        red_mask=inputs.red_mask[time_index],
        global_blue=inputs.global_blue[time_index],
        blue_mask=inputs.blue_mask[time_index],
        applied_bias=inputs.applied_bias[time_index],
        global_context=inputs.global_context[time_index],
        pair_state=inputs.pair_state[time_index],
        current_assignment=inputs.current_assignment[time_index],
        hidden=hidden,
    )


class MAPPOTrainer:
    """Two-level CTDE MAPPO with independent assignment and execution networks."""

    def __init__(
        self,
        assignment_actor: TargetAssignmentActor,
        execution_actor: OverloadBiasActor,
        assignment_critic: TargetAssignmentCritic,
        execution_critic: OverloadBiasCritic,
        config: PPOConfig = PPOConfig(),
    ) -> None:
        config.validate()
        self.assignment_actor = assignment_actor
        self.execution_actor = execution_actor
        self.assignment_critic = assignment_critic
        self.execution_critic = execution_critic
        self.config = config
        self.device = next(assignment_actor.parameters()).device
        modules = (execution_actor, assignment_critic, execution_critic)
        if any(next(module.parameters()).device != self.device for module in modules):
            raise ValueError("all MAPPO networks must be on the same device")
        assignment_actor_lr = (
            config.actor_learning_rate or config.assignment_actor_learning_rate
        )
        execution_actor_lr = (
            config.actor_learning_rate or config.execution_actor_learning_rate
        )
        assignment_critic_lr = (
            config.critic_learning_rate or config.assignment_critic_learning_rate
        )
        execution_critic_lr = (
            config.critic_learning_rate or config.execution_critic_learning_rate
        )
        self.assignment_actor_optimizer = torch.optim.Adam(
            assignment_actor.parameters(),
            lr=assignment_actor_lr,
        )
        self.execution_actor_optimizer = torch.optim.Adam(
            execution_actor.parameters(),
            lr=execution_actor_lr,
        )
        self.assignment_critic_optimizer = torch.optim.Adam(
            assignment_critic.parameters(),
            lr=assignment_critic_lr,
        )
        self.execution_critic_optimizer = torch.optim.Adam(
            execution_critic.parameters(),
            lr=execution_critic_lr,
        )
        self._update_step = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "assignment_actor_optimizer": self.assignment_actor_optimizer.state_dict(),
            "execution_actor_optimizer": self.execution_actor_optimizer.state_dict(),
            "assignment_critic_optimizer": self.assignment_critic_optimizer.state_dict(),
            "execution_critic_optimizer": self.execution_critic_optimizer.state_dict(),
            "update_step": self._update_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        optimizers = {
            "assignment_actor_optimizer": self.assignment_actor_optimizer,
            "execution_actor_optimizer": self.execution_actor_optimizer,
            "assignment_critic_optimizer": self.assignment_critic_optimizer,
            "execution_critic_optimizer": self.execution_critic_optimizer,
        }
        for key, optimizer in optimizers.items():
            if key in state:
                optimizer.load_state_dict(state[key])
        self._update_step = int(state.get("update_step", 0))

    @staticmethod
    def _initial_agent_hidden(
        inputs: AssignmentActorInputs | OverloadBiasActorInputs,
        time_index: int = 0,
    ) -> Tensor | None:
        return None if inputs.hidden is None else inputs.hidden[time_index]

    @staticmethod
    def _initial_critic_hidden(
        inputs: OverloadBiasCriticInputs,
        time_index: int = 0,
    ) -> Tensor | None:
        return None if inputs.hidden is None else inputs.hidden[time_index]

    def _assignment_policy_evaluation(
        self,
        batch: MAPPOBatch,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        time = batch.rewards_high.shape[0]
        end = time if end is None else min(end, time)
        hidden = self._initial_agent_hidden(batch.assignment_actor_inputs, start)
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        for time_index in range(start, end):
            inputs = _slice_assignment_inputs(
                batch.assignment_actor_inputs,
                time_index,
                hidden,
            ).to(self.device, non_blocking=True)
            evaluation = self.assignment_actor.evaluate_actions(
                inputs,
                _slice_assignment_actions(
                    batch.assignment_actions,
                    time_index,
                ).to(self.device, non_blocking=True),
            )
            log_probs.append(evaluation.joint_log_prob)
            entropies.append(evaluation.joint_entropy)
            keep = (1.0 - batch.dones_high[time_index]).bool().unsqueeze(-1).unsqueeze(-1)
            hidden = evaluation.next_hidden * keep
        return torch.stack(log_probs), torch.stack(entropies)

    def _execution_policy_evaluation(
        self,
        batch: MAPPOBatch,
        start: int = 0,
        end: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        time = batch.rewards_low.shape[0]
        end = time if end is None else min(end, time)
        hidden = self._initial_agent_hidden(batch.execution_actor_inputs, start)
        log_probs: list[Tensor] = []
        entropies: list[Tensor] = []
        for time_index in range(start, end):
            inputs = _slice_execution_inputs(
                batch.execution_actor_inputs,
                time_index,
                hidden,
            ).to(self.device, non_blocking=True)
            evaluation = self.execution_actor.evaluate_actions(
                inputs,
                batch.bias_matrices[time_index].to(
                    self.device,
                    non_blocking=True,
                ),
            )
            log_probs.append(evaluation.log_prob)
            entropies.append(evaluation.entropy)
            keep = (1.0 - batch.dones_low[time_index]).bool().unsqueeze(-1)
            hidden = evaluation.next_hidden * keep
        return torch.stack(log_probs), torch.stack(entropies)

    def _execution_approx_kl(self, batch: MAPPOBatch) -> Tensor:
        with torch.no_grad():
            log_prob, _ = self._execution_policy_evaluation(batch)
            log_ratio = log_prob - batch.old_execution_log_prob
            ratio = torch.exp(log_ratio)
            active = self._execution_active_mask(batch)
            weights = _execution_actor_weights(
                batch,
                active,
                self.config.execution_actor_loss_weighting,
            ).to(dtype=ratio.dtype)
            count = weights.sum().clamp_min(1.0)
            return (((ratio - 1.0) - log_ratio) * weights).sum() / count

    def _assignment_values(
        self,
        inputs: AssignmentCriticInputs,
        start: int = 0,
        end: int | None = None,
    ) -> Tensor:
        time_plus_one = inputs.global_red.shape[0]
        end = time_plus_one if end is None else min(end, time_plus_one)
        return torch.stack(
            [
                self.assignment_critic(
                    _slice_assignment_critic_inputs(inputs, time_index).to(
                        self.device,
                        non_blocking=True,
                    )
                ).value
                for time_index in range(start, end)
            ]
        )

    def _execution_values(
        self,
        inputs: OverloadBiasCriticInputs,
        dones: Tensor | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> Tensor:
        time_plus_one = inputs.global_red.shape[0]
        end = time_plus_one if end is None else min(end, time_plus_one)
        hidden = self._initial_critic_hidden(inputs, start)
        values: list[Tensor] = []
        for time_index in range(start, end):
            output = self.execution_critic(
                _slice_execution_critic_inputs(
                    inputs,
                    time_index,
                    hidden,
                ).to(self.device, non_blocking=True)
            )
            values.append(output.value)
            hidden = output.next_hidden
            if dones is not None and time_index < end - 1:
                hidden = hidden * (1.0 - dones[time_index]).bool().unsqueeze(-1)
        return torch.stack(values)

    @staticmethod
    def _execution_active_mask(batch: MAPPOBatch) -> Tensor:
        actor_active = (
            batch.execution_actor_inputs.agent_mask.bool()
            & batch.execution_actor_inputs.target_mask.any(dim=-1)
        ).to(batch.rewards_low.device, non_blocking=True)
        return actor_active & batch.episode_active_low.bool().unsqueeze(-1)

    def _advantages_and_returns(
        self,
        batch: MAPPOBatch,
        *,
        assignment: bool = True,
        execution: bool = True,
    ) -> AdvantageTargets:
        config = self.config
        if batch.high_reference_interval_s <= 0.0 or batch.low_reference_interval_s <= 0.0:
            raise ValueError("MAPPO time reference intervals must be positive")
        with torch.no_grad():
            if assignment:
                assignment_values = self._assignment_values(
                    batch.assignment_critic_inputs
                )
                learning_rewards_high = (
                    batch.rewards_high * config.assignment_reward_learning_scale
                )
                assignment_raw_advantage, assignment_return = (
                    generalized_advantage_estimation(
                        learning_rewards_high,
                        assignment_values,
                        batch.dones_high,
                        config.gamma_high
                        ** (
                            batch.durations_high_s
                            / batch.high_reference_interval_s
                        ),
                        config.lambda_high
                        ** (
                            batch.durations_high_s
                            / batch.high_reference_interval_s
                        ),
                    )
                )
            else:
                assignment_raw_advantage = torch.zeros_like(batch.rewards_high)
                assignment_return = torch.zeros_like(batch.rewards_high)
                assignment_values = torch.zeros(
                    batch.rewards_high.shape[0] + 1,
                    *batch.rewards_high.shape[1:],
                    dtype=batch.rewards_high.dtype,
                    device=batch.rewards_high.device,
                )
            if execution:
                execution_values = self._execution_values(
                    batch.execution_critic_inputs,
                    batch.dones_low,
                )
                execution_active = self._execution_active_mask(batch)
                learning_rewards_low = (
                    batch.rewards_low * config.execution_reward_learning_scale
                )
                execution_raw_advantage, execution_return = (
                    generalized_advantage_estimation(
                        learning_rewards_low,
                        execution_values,
                        batch.dones_low,
                        config.gamma_low
                        ** (
                            batch.durations_low_s / batch.low_reference_interval_s
                        ),
                        config.lambda_low
                        ** (
                            batch.durations_low_s / batch.low_reference_interval_s
                        ),
                    )
                )
            else:
                execution_raw_advantage = torch.zeros_like(batch.rewards_low)
                execution_return = torch.zeros_like(batch.rewards_low)
                execution_values = torch.zeros(
                    batch.rewards_low.shape[0] + 1,
                    *batch.rewards_low.shape[1:],
                    dtype=batch.rewards_low.dtype,
                    device=batch.rewards_low.device,
                )
            assignment_advantage = assignment_raw_advantage
            execution_advantage = execution_raw_advantage
            if config.normalize_advantage:
                if assignment:
                    assignment_advantage = _normalize_advantage(
                        assignment_raw_advantage,
                        batch.episode_active_high,
                    )
                if execution:
                    if config.execution_advantage_normalization == "per_scenario":
                        execution_advantage = (
                            _normalize_execution_advantage_per_scenario(
                                batch,
                                execution_raw_advantage,
                                execution_active,
                            )
                        )
                    else:
                        execution_advantage = _normalize_advantage(
                            execution_raw_advantage,
                            execution_active,
                        )
        return AdvantageTargets(
            assignment_raw_advantage=assignment_raw_advantage,
            assignment_advantage=assignment_advantage,
            assignment_return=assignment_return,
            execution_raw_advantage=execution_raw_advantage,
            execution_advantage=execution_advantage,
            execution_return=execution_return,
            assignment_value_before_update=assignment_values[:-1],
            execution_value_before_update=execution_values[:-1],
        )

    def _effort_finetune_advantage(self, batch: MAPPOBatch) -> Tensor:
        action = batch.bias_matrices.to(self.device, non_blocking=True)
        action_norm = torch.linalg.vector_norm(action, dim=-1, keepdim=True)
        projected = action / action_norm.clamp_min(1.0)
        load_cost = projected.pow(2).sum(dim=-1)
        previous_action = batch.execution_critic_inputs.applied_bias[:-1].to(
            self.device,
            non_blocking=True,
        )
        smooth_cost = (action - previous_action).pow(2).sum(dim=-1) / 8.0
        effort_reward = -(0.8 * load_cost + 0.2 * smooth_cost)
        zero_values = torch.zeros(
            (effort_reward.shape[0] + 1, *effort_reward.shape[1:]),
            dtype=effort_reward.dtype,
            device=effort_reward.device,
        )
        advantage, _ = generalized_advantage_estimation(
            effort_reward,
            zero_values,
            batch.dones_low,
            self.config.gamma_low ** (
                batch.durations_low_s / batch.low_reference_interval_s
            ),
            self.config.lambda_low ** (
                batch.durations_low_s / batch.low_reference_interval_s
            ),
        )
        active = self._execution_active_mask(batch)
        if self.config.normalize_advantage:
            advantage = _normalize_advantage(advantage, active)
        return advantage * self.config.effort_finetune_scale

    def _time_chunks(self, time: int, level: str) -> list[tuple[int, int]]:
        length = (
            self.config.high_sequence_length
            if level == "assignment"
            else self.config.low_sequence_length
        )
        return [(start, min(start + length, time)) for start in range(0, time, length)]

    def _backward_critic_chunks(
        self,
        batch: MAPPOBatch,
        assignment_return: Tensor,
        execution_return: Tensor,
        *,
        update_assignment: bool,
        update_execution: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        config = self.config
        device = self.device
        high_active = batch.episode_active_high.to(dtype=batch.rewards_high.dtype)
        high_count = high_active.sum().clamp_min(1.0)
        low_active = torch.zeros_like(batch.rewards_low, dtype=torch.bool)
        low_count = torch.tensor(1.0, device=device)
        if update_execution:
            low_active = self._execution_active_mask(batch).to(
                dtype=batch.rewards_low.dtype
            )
            low_count = low_active.sum().clamp_min(1.0)
        assignment_loss_sum = torch.tensor(0.0, device=device)
        execution_loss_sum = torch.tensor(0.0, device=device)
        execution_mse_sum = torch.tensor(0.0, device=device)
        assignment_grad_norm = torch.tensor(0.0, device=device)
        execution_grad_norm = torch.tensor(0.0, device=device)
        if update_assignment:
            self.assignment_critic_optimizer.zero_grad(set_to_none=True)
            for start, end in self._time_chunks(
                assignment_return.shape[0],
                "assignment",
            ):
                values = self._assignment_values(
                    batch.assignment_critic_inputs,
                    start=start,
                    end=end,
                )
                loss_sum, _, _ = _masked_value_loss(
                    values,
                    assignment_return[start:end],
                    high_active[start:end],
                    kind="mse",
                    huber_delta=1.0,
                )
                (config.value_coef * loss_sum / high_count).backward()
                assignment_loss_sum += loss_sum.detach()
            assignment_grad_norm = nn.utils.clip_grad_norm_(
                self.assignment_critic.parameters(),
                config.max_grad_norm,
            ).detach()
            self.assignment_critic_optimizer.step()

        if update_execution:
            self.execution_critic_optimizer.zero_grad(set_to_none=True)
            for start, end in self._time_chunks(
                execution_return.shape[0],
                "execution",
            ):
                values = self._execution_values(
                    batch.execution_critic_inputs,
                    batch.dones_low,
                    start=start,
                    end=end,
                )
                loss_sum, mse_sum, _ = _masked_value_loss(
                    values,
                    execution_return[start:end],
                    low_active[start:end],
                    kind=config.execution_value_loss,
                    huber_delta=config.execution_value_huber_delta,
                )
                (config.value_coef * loss_sum / low_count).backward()
                execution_loss_sum += loss_sum.detach()
                execution_mse_sum += mse_sum.detach()
            execution_grad_norm = nn.utils.clip_grad_norm_(
                self.execution_critic.parameters(),
                config.max_grad_norm,
            ).detach()
            self.execution_critic_optimizer.step()
        return (
            assignment_loss_sum / high_count,
            execution_loss_sum / low_count,
            execution_mse_sum / low_count,
            assignment_grad_norm,
            execution_grad_norm,
        )

    def _backward_actor_chunks(
        self,
        batch: MAPPOBatch,
        assignment_advantage: Tensor,
        execution_advantage: Tensor,
        *,
        update_assignment: bool,
        update_execution: bool,
    ) -> tuple[Tensor, ...]:
        config = self.config
        device = self.device
        assignment_active = batch.episode_active_high.bool()
        execution_active = torch.zeros_like(batch.rewards_low, dtype=torch.bool)
        if update_execution:
            execution_active = self._execution_active_mask(batch)
        assignment_count = assignment_active.sum().to(dtype=torch.float32).clamp_min(1.0)
        execution_weights = _execution_actor_weights(
            batch,
            execution_active,
            config.execution_actor_loss_weighting,
        )
        execution_count = execution_weights.sum().clamp_min(1.0)
        assignment_policy_sum = torch.tensor(0.0, device=device)
        execution_policy_sum = torch.tensor(0.0, device=device)
        assignment_entropy_sum = torch.tensor(0.0, device=device)
        execution_entropy_sum = torch.tensor(0.0, device=device)
        assignment_ratio_sum = torch.tensor(0.0, device=device)
        execution_ratio_sum = torch.tensor(0.0, device=device)
        assignment_kl_sum = torch.tensor(0.0, device=device)
        execution_kl_sum = torch.tensor(0.0, device=device)
        assignment_clip_sum = torch.tensor(0.0, device=device)
        execution_clip_sum = torch.tensor(0.0, device=device)
        assignment_grad_norm = torch.tensor(0.0, device=device)
        execution_grad_norm = torch.tensor(0.0, device=device)
        execution_post_step_kl = torch.tensor(0.0, device=device)
        execution_step_accepted = torch.tensor(0.0, device=device)

        if update_assignment:
            self.assignment_actor_optimizer.zero_grad(set_to_none=True)
            for start, end in self._time_chunks(
                assignment_advantage.shape[0],
                "assignment",
            ):
                log_prob, entropy = self._assignment_policy_evaluation(batch, start, end)
                log_ratio = log_prob - batch.old_assignment_log_prob[start:end]
                ratio = torch.exp(log_ratio)
                advantage = assignment_advantage[start:end]
                active = assignment_active[start:end]
                active_float = active.to(dtype=ratio.dtype)
                clipped = ratio.clamp(
                    1.0 - config.high_clip_epsilon,
                    1.0 + config.high_clip_epsilon,
                )
                surrogate = torch.minimum(ratio * advantage, clipped * advantage)
                policy_sum = -(surrogate * active_float).sum()
                entropy_sum = (entropy * active_float).sum()
                (
                    (policy_sum - config.high_entropy_coef * entropy_sum)
                    / assignment_count
                ).backward()
                assignment_policy_sum += policy_sum.detach()
                assignment_entropy_sum += entropy_sum.detach()
                assignment_ratio_sum += (ratio * active_float).sum().detach()
                assignment_kl_sum += (
                    ((ratio - 1.0) - log_ratio) * active_float
                ).sum().detach()
                assignment_clip_sum += (
                    (torch.abs(ratio - 1.0) > config.high_clip_epsilon).to(ratio.dtype)
                    * active_float
                ).sum().detach()
            assignment_grad_norm = nn.utils.clip_grad_norm_(
                self.assignment_actor.parameters(),
                config.max_grad_norm,
            ).detach()
            self.assignment_actor_optimizer.step()

        if update_execution:
            actor_snapshot = None
            optimizer_snapshot = None
            if config.execution_post_step_kl_rollback:
                actor_snapshot = copy.deepcopy(self.execution_actor.state_dict())
                optimizer_snapshot = copy.deepcopy(
                    self.execution_actor_optimizer.state_dict()
                )
            self.execution_actor_optimizer.zero_grad(set_to_none=True)
            for start, end in self._time_chunks(
                execution_advantage.shape[0],
                "execution",
            ):
                log_prob, entropy = self._execution_policy_evaluation(batch, start, end)
                log_ratio = log_prob - batch.old_execution_log_prob[start:end]
                ratio = torch.exp(log_ratio)
                advantage = execution_advantage[start:end]
                sample_weight = execution_weights[start:end].to(dtype=ratio.dtype)
                clipped = ratio.clamp(
                    1.0 - config.low_clip_epsilon,
                    1.0 + config.low_clip_epsilon,
                )
                surrogate = torch.minimum(ratio * advantage, clipped * advantage)
                policy_sum = -(surrogate * sample_weight).sum()
                entropy_sum = (entropy * sample_weight).sum()
                (
                    (policy_sum - config.low_entropy_coef * entropy_sum)
                    / execution_count
                ).backward()
                execution_policy_sum += policy_sum.detach()
                execution_entropy_sum += entropy_sum.detach()
                execution_ratio_sum += (ratio * sample_weight).sum().detach()
                execution_kl_sum += (
                    ((ratio - 1.0) - log_ratio) * sample_weight
                ).sum().detach()
                execution_clip_sum += (
                    (torch.abs(ratio - 1.0) > config.low_clip_epsilon).to(ratio.dtype)
                    * sample_weight
                ).sum().detach()
            execution_grad_norm = nn.utils.clip_grad_norm_(
                self.execution_actor.parameters(),
                config.max_grad_norm,
            ).detach()
            self.execution_actor_optimizer.step()
            execution_post_step_kl = self._execution_approx_kl(batch).detach()
            rejected = (
                config.execution_post_step_kl_rollback
                and config.execution_post_step_kl_limit is not None
                and execution_post_step_kl.item()
                > config.execution_post_step_kl_limit
            )
            if rejected:
                assert actor_snapshot is not None and optimizer_snapshot is not None
                self.execution_actor.load_state_dict(actor_snapshot)
                self.execution_actor_optimizer.load_state_dict(optimizer_snapshot)
            else:
                execution_step_accepted = torch.tensor(1.0, device=device)
        assignment_entropy = assignment_entropy_sum / assignment_count
        execution_entropy = execution_entropy_sum / execution_count
        return (
            assignment_policy_sum / assignment_count
            - config.high_entropy_coef * assignment_entropy,
            execution_policy_sum / execution_count
            - config.low_entropy_coef * execution_entropy,
            assignment_entropy,
            execution_entropy,
            assignment_ratio_sum / assignment_count,
            execution_ratio_sum / execution_count,
            assignment_kl_sum / assignment_count,
            execution_kl_sum / execution_count,
            assignment_clip_sum / assignment_count,
            execution_clip_sum / execution_count,
            assignment_grad_norm,
            execution_grad_norm,
            execution_post_step_kl,
            execution_step_accepted,
        )

    def update(
        self,
        batch: MAPPOBatch,
        mode: str = "joint",
        *,
        critic_steps_override: int | None = None,
    ) -> Dict[str, float]:
        if mode not in {
            "joint",
            "low_only",
            "low_critic_only",
            "high_only",
            "effort_finetune",
        }:
            raise ValueError(
                "mode must be one of joint, low_only, low_critic_only, high_only, "
                "effort_finetune"
            )
        if critic_steps_override is not None:
            if mode != "low_critic_only":
                raise ValueError(
                    "critic_steps_override is supported only for low_critic_only"
                )
            if critic_steps_override <= 0:
                raise ValueError("critic_steps_override must be positive")
        config = self.config
        self._update_step += 1
        should_update_actor = self._update_step % config.actor_update_interval == 0
        update_assignment = mode in {"joint", "high_only"}
        update_execution = mode in {"joint", "low_only", "effort_finetune"}
        update_execution_critic = mode in {"joint", "low_only", "low_critic_only"}
        needs_execution_data = update_execution or update_execution_critic
        batch = batch.training_view(
            self.device,
            assignment=update_assignment,
            execution=needs_execution_data,
        )
        targets = self._advantages_and_returns(
            batch,
            assignment=update_assignment,
            execution=needs_execution_data,
        )
        assignment_advantage = targets.assignment_advantage
        execution_advantage = targets.execution_advantage
        assignment_return = targets.assignment_return
        execution_return = targets.execution_return
        if mode == "effort_finetune":
            execution_advantage = self._effort_finetune_advantage(batch)

        device = self.device
        assignment_actor_loss_value = torch.tensor(0.0, device=device)
        execution_actor_loss_value = torch.tensor(0.0, device=device)
        assignment_critic_loss_value = torch.tensor(0.0, device=device)
        execution_critic_loss_value = torch.tensor(0.0, device=device)
        execution_critic_mse_value = torch.tensor(0.0, device=device)
        assignment_entropy_value = torch.tensor(0.0, device=device)
        execution_entropy_value = torch.tensor(0.0, device=device)
        assignment_ratio_value = torch.tensor(0.0, device=device)
        execution_ratio_value = torch.tensor(0.0, device=device)
        assignment_kl_value = torch.tensor(0.0, device=device)
        execution_kl_value = torch.tensor(0.0, device=device)
        assignment_clip_fraction_value = torch.tensor(0.0, device=device)
        execution_clip_fraction_value = torch.tensor(0.0, device=device)
        assignment_actor_grad_norm_value = torch.tensor(0.0, device=device)
        execution_actor_grad_norm_value = torch.tensor(0.0, device=device)
        assignment_critic_grad_norm_value = torch.tensor(0.0, device=device)
        execution_critic_grad_norm_value = torch.tensor(0.0, device=device)
        execution_post_step_kl_value = torch.tensor(0.0, device=device)
        execution_pre_step_kl_max = torch.tensor(0.0, device=device)
        execution_post_step_kl_max = torch.tensor(0.0, device=device)
        assignment_actor_updates = 0
        execution_actor_updates = 0
        execution_actor_steps_attempted = 0
        execution_actor_steps_rejected = 0
        assignment_critic_updates = 0
        execution_critic_updates = 0
        assignment_kl_stopped = False
        execution_kl_stopped = False

        update_epochs = 1 if critic_steps_override is not None else config.epochs
        critic_steps_per_epoch = (
            critic_steps_override
            if critic_steps_override is not None
            else config.critic_updates_per_actor
        )
        for _ in range(update_epochs):
            for _ in range(critic_steps_per_epoch):
                (
                    assignment_critic_loss_value,
                    execution_critic_loss_value,
                    execution_critic_mse_value,
                    assignment_critic_grad_norm_value,
                    execution_critic_grad_norm_value,
                ) = (
                    self._backward_critic_chunks(
                        batch,
                        assignment_return,
                        execution_return,
                        update_assignment=update_assignment,
                        update_execution=update_execution_critic,
                    )
                )
                assignment_critic_updates += int(update_assignment)
                execution_critic_updates += int(update_execution_critic)

            update_assignment_actor = (
                should_update_actor and update_assignment and not assignment_kl_stopped
            )
            update_execution_actor = (
                should_update_actor and update_execution and not execution_kl_stopped
            )
            if update_assignment_actor or update_execution_actor:
                (
                    assignment_actor_loss_value,
                    execution_actor_loss_value,
                    assignment_entropy_value,
                    execution_entropy_value,
                    assignment_ratio_value,
                    execution_ratio_value,
                    assignment_kl_value,
                    execution_kl_value,
                    assignment_clip_fraction_value,
                    execution_clip_fraction_value,
                    assignment_actor_grad_norm_value,
                    execution_actor_grad_norm_value,
                    execution_post_step_kl_value,
                    execution_step_accepted,
                ) = self._backward_actor_chunks(
                    batch,
                    assignment_advantage,
                    execution_advantage,
                    update_assignment=update_assignment_actor,
                    update_execution=update_execution_actor,
                )
                assignment_actor_updates += int(update_assignment_actor)
                execution_actor_steps_attempted += int(update_execution_actor)
                execution_actor_updates += int(
                    update_execution_actor and execution_step_accepted.item() > 0.5
                )
                execution_actor_steps_rejected += int(
                    update_execution_actor and execution_step_accepted.item() <= 0.5
                )
                if update_execution_actor:
                    execution_pre_step_kl_max = torch.maximum(
                        execution_pre_step_kl_max,
                        execution_kl_value.detach(),
                    )
                    execution_post_step_kl_max = torch.maximum(
                        execution_post_step_kl_max,
                        execution_post_step_kl_value.detach(),
                    )
                if (
                    update_assignment_actor
                    and assignment_kl_value.item() >= config.assignment_target_kl
                ):
                    assignment_kl_stopped = True
                if (
                    update_execution_actor
                    and (
                        execution_kl_value.item() >= config.execution_target_kl
                        or execution_step_accepted.item() <= 0.5
                    )
                ):
                    execution_kl_stopped = True

        actor_updates = max(assignment_actor_updates, execution_actor_updates)
        critic_updates = max(assignment_critic_updates, execution_critic_updates)
        zero_metric = torch.tensor(0.0, device=device)
        with torch.no_grad():
            if update_assignment:
                assignment_value = self._assignment_values(
                    batch.assignment_critic_inputs
                )[:-1]
                assignment_explained_variance = _explained_variance(
                    assignment_value,
                    assignment_return,
                    batch.episode_active_high,
                )
                assignment_explained_variance_before_update = _explained_variance(
                    targets.assignment_value_before_update,
                    assignment_return,
                    batch.episode_active_high,
                )
            else:
                assignment_value = torch.zeros_like(assignment_return)
                assignment_explained_variance = zero_metric
                assignment_explained_variance_before_update = zero_metric
            if needs_execution_data:
                execution_value = self._execution_values(
                    batch.execution_critic_inputs,
                    batch.dones_low,
                )[:-1]
                execution_active = self._execution_active_mask(batch)
                execution_explained_variance = _explained_variance(
                    execution_value,
                    execution_return,
                    execution_active,
                )
                execution_explained_variance_before_update = _explained_variance(
                    targets.execution_value_before_update,
                    execution_return,
                    execution_active,
                )
            else:
                execution_value = torch.zeros_like(execution_return)
                execution_active = torch.zeros_like(
                    batch.rewards_low,
                    dtype=torch.bool,
                )
                execution_explained_variance = zero_metric
                execution_explained_variance_before_update = zero_metric

        execution_residual = (
            targets.execution_return - targets.execution_value_before_update
        )
        diagnostic_metrics = {
            **_selected_summary(
                "execution_raw_advantage",
                targets.execution_raw_advantage,
                execution_active,
            ),
            **_selected_summary(
                "execution_return",
                targets.execution_return,
                execution_active,
            ),
            **_selected_summary(
                "execution_value",
                targets.execution_value_before_update,
                execution_active,
            ),
            **_selected_summary(
                "execution_value_residual",
                execution_residual,
                execution_active,
            ),
        }
        if needs_execution_data:
            execution_actor_weights = _execution_actor_weights(
                batch,
                execution_active,
                config.execution_actor_loss_weighting,
            )
            if (
                batch.scenario_red_counts is not None
                and batch.scenario_blue_counts is not None
            ):
                diagnostic_metrics.update(
                    _execution_scenario_diagnostics(
                        batch,
                        execution_active,
                        execution_actor_weights,
                        targets,
                        execution_value,
                    )
                )
            behavior_transformed_entropy_sample = -(
                batch.old_execution_log_prob * execution_actor_weights
            ).sum() / execution_actor_weights.sum().clamp_min(1.0)
        else:
            execution_actor_weights = torch.zeros_like(batch.rewards_low)
            behavior_transformed_entropy_sample = zero_metric
        normalized_advantage_std = (
            targets.execution_advantage[execution_active].std(unbiased=False)
            if execution_active.any()
            else torch.tensor(0.0, device=device)
        )

        return {
            "actor_loss": float(
                (assignment_actor_loss_value + execution_actor_loss_value).cpu()
            ),
            "assignment_actor_loss": float(assignment_actor_loss_value.cpu()),
            "execution_actor_loss": float(execution_actor_loss_value.cpu()),
            "critic_loss": float(
                (assignment_critic_loss_value + execution_critic_loss_value).cpu()
            ),
            "assignment_critic_loss": float(assignment_critic_loss_value.cpu()),
            "execution_critic_loss": float(execution_critic_loss_value.cpu()),
            "execution_critic_mse": float(execution_critic_mse_value.cpu()),
            "assignment_entropy": float(assignment_entropy_value.cpu()),
            "execution_entropy": float(execution_entropy_value.cpu()),
            "execution_pre_tanh_gaussian_entropy": float(
                execution_entropy_value.cpu()
            ),
            "execution_behavior_transformed_entropy_sample": float(
                behavior_transformed_entropy_sample.cpu()
            ),
            "assignment_ratio": float(assignment_ratio_value.cpu()),
            "execution_ratio": float(execution_ratio_value.cpu()),
            "assignment_approx_kl": float(assignment_kl_value.cpu()),
            "execution_approx_kl": float(execution_kl_value.cpu()),
            "assignment_clip_fraction": float(assignment_clip_fraction_value.cpu()),
            "execution_clip_fraction": float(execution_clip_fraction_value.cpu()),
            "assignment_kl_stopped": float(assignment_kl_stopped),
            "execution_kl_stopped": float(execution_kl_stopped),
            "actor_updates": float(actor_updates),
            "critic_updates": float(critic_updates),
            "assignment_actor_updates": float(assignment_actor_updates),
            "execution_actor_updates": float(execution_actor_updates),
            "execution_actor_steps_attempted": float(
                execution_actor_steps_attempted
            ),
            "execution_actor_steps_accepted": float(execution_actor_updates),
            "execution_actor_steps_rejected": float(execution_actor_steps_rejected),
            "execution_kl_rollback_triggered": float(
                execution_actor_steps_rejected > 0
            ),
            "assignment_critic_updates": float(assignment_critic_updates),
            "execution_critic_updates": float(execution_critic_updates),
            "execution_advantage_std": diagnostic_metrics[
                "execution_raw_advantage_std"
            ],
            "execution_normalized_advantage_std": float(
                normalized_advantage_std.cpu()
            ),
            "assignment_actor_grad_norm_preclip": float(
                assignment_actor_grad_norm_value.cpu()
            ),
            "execution_actor_grad_norm_preclip": float(
                execution_actor_grad_norm_value.cpu()
            ),
            "assignment_critic_grad_norm_preclip": float(
                assignment_critic_grad_norm_value.cpu()
            ),
            "execution_critic_grad_norm_preclip": float(
                execution_critic_grad_norm_value.cpu()
            ),
            "execution_pre_step_kl_last": float(execution_kl_value.cpu()),
            "execution_pre_step_kl_max": float(execution_pre_step_kl_max.cpu()),
            "execution_post_step_kl_last": float(
                execution_post_step_kl_value.cpu()
            ),
            "execution_post_step_kl_max": float(
                execution_post_step_kl_max.cpu()
            ),
            "assignment_explained_variance": float(
                assignment_explained_variance.cpu()
            ),
            "execution_explained_variance": float(
                execution_explained_variance.cpu()
            ),
            "assignment_explained_variance_before_update": float(
                assignment_explained_variance_before_update.cpu()
            ),
            "execution_explained_variance_before_update": float(
                execution_explained_variance_before_update.cpu()
            ),
            "optimizer_update_ratio": float(
                config.critic_updates_per_actor / config.actor_update_interval
            ),
            "environment_timescale_ratio": float(
                batch.high_reference_interval_s / batch.low_reference_interval_s
            ),
            "timescale_ratio": float(
                config.critic_updates_per_actor / config.actor_update_interval
            ),
            **diagnostic_metrics,
        }
