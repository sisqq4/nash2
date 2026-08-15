from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SwarmModelConfig:
    d_self: int = 13
    d_friend: int = 11
    d_target: int = 8
    d_pair: int = 11
    d_execution_self: int = 20
    d_execution_friend: int = 14
    d_execution_target: int = 17
    d_actor_context: int = 5
    d_execution_context: int = 4
    d_global_red: int = 15
    d_global_blue: int = 8
    d_global_context: int = 8
    d_bias: int = 2
    d_value_components: int = 5
    d_model: int = 128
    num_heads: int = 4
    max_missiles_per_target: int = 4
    no_target_index: int = 0
    log_std_min: float = -4.0
    log_std_max: float = -1.0
    execution_action_distribution: str = "tanh_box"
    critic_value_head_mode: str = "latent_sum"
    assignment_critic_value_head_mode: Optional[str] = None
    assignment_stickiness_logit_bonus: float = 0.0

    def validate(self) -> None:
        feature_dims = (
            self.d_self,
            self.d_friend,
            self.d_target,
            self.d_pair,
            self.d_execution_self,
            self.d_execution_friend,
            self.d_execution_target,
            self.d_actor_context,
            self.d_execution_context,
            self.d_global_red,
            self.d_global_blue,
            self.d_global_context,
        )
        expected_feature_dims = (13, 11, 8, 11, 20, 14, 17, 5, 4, 15, 8, 8)
        if feature_dims != expected_feature_dims:
            raise ValueError(
                "observation feature dimensions must be exactly "
                "(13, 11, 8, 11, 20, 14, 17, 5, 4, 15, 8, 8)"
            )
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if not isinstance(self.num_heads, int) or isinstance(self.num_heads, bool) or self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (
            not isinstance(self.max_missiles_per_target, int)
            or isinstance(self.max_missiles_per_target, bool)
            or self.max_missiles_per_target <= 0
        ):
            raise ValueError("max_missiles_per_target must be a positive integer")
        if self.d_bias != 2:
            raise ValueError("d_bias must be exactly 2 for the two lateral overload-bias components")
        if self.execution_action_distribution not in {
            "tanh_box",
            "radial_tanh_disk",
        }:
            raise ValueError(
                "execution_action_distribution must be tanh_box or radial_tanh_disk"
            )
        if self.critic_value_head_mode not in {"latent_sum", "scalar"}:
            raise ValueError("critic_value_head_mode must be latent_sum or scalar")
        if self.assignment_critic_value_head_mode not in {
            None,
            "latent_sum",
            "scalar",
        }:
            raise ValueError(
                "assignment_critic_value_head_mode must be latent_sum or scalar"
            )
        expected_value_components = (
            1 if self.critic_value_head_mode == "scalar" else 5
        )
        if self.d_value_components != expected_value_components:
            raise ValueError(
                f"{self.critic_value_head_mode} critic requires "
                f"d_value_components={expected_value_components}"
            )
        if self.no_target_index != 0:
            raise ValueError("no_target_index must be 0 because target slot 0 is reserved")
        if self.log_std_min >= self.log_std_max:
            raise ValueError("log_std_min must be smaller than log_std_max")
        if (
            not math.isfinite(self.assignment_stickiness_logit_bonus)
            or self.assignment_stickiness_logit_bonus < 0.0
        ):
            raise ValueError(
                "assignment_stickiness_logit_bonus must be non-negative and finite"
            )

    @property
    def effective_assignment_critic_value_head_mode(self) -> str:
        return self.assignment_critic_value_head_mode or self.critic_value_head_mode

    @property
    def assignment_value_components(self) -> int:
        return (
            1
            if self.effective_assignment_critic_value_head_mode == "scalar"
            else 5
        )


@dataclass(frozen=True)
class PPOConfig:
    gamma_high: float = 1.0
    gamma_low: float = 1.0
    lambda_high: float = 0.95
    lambda_low: float = 0.994883803
    clip_epsilon: Optional[float] = None
    assignment_clip_epsilon: float = 0.10
    execution_clip_epsilon: float = 0.20
    value_coef: float = 0.5
    entropy_coef: Optional[float] = None
    assignment_entropy_coef: float = 0.001
    execution_entropy_coef: float = 0.001
    learning_rate: float = 3.0e-4
    actor_learning_rate: Optional[float] = None
    critic_learning_rate: Optional[float] = None
    assignment_actor_learning_rate: float = 1.0e-4
    execution_actor_learning_rate: float = 5.0e-5
    assignment_critic_learning_rate: float = 3.0e-4
    execution_critic_learning_rate: float = 3.0e-4
    max_grad_norm: float = 0.5
    epochs: int = 4
    critic_updates_per_actor: int = 2
    actor_update_interval: int = 1
    sequence_length: Optional[int] = None
    assignment_sequence_length: int = 32
    execution_sequence_length: int = 128
    assignment_target_kl: float = 0.01
    execution_target_kl: float = 0.02
    assignment_reward_learning_scale: float = 1.0
    execution_reward_learning_scale: float = 1.0
    execution_value_loss: str = "mse"
    execution_value_huber_delta: float = 1.0
    execution_post_step_kl_rollback: bool = False
    execution_post_step_kl_limit: Optional[float] = None
    effort_finetune_scale: float = 1000.0
    normalize_advantage: bool = True
    execution_advantage_normalization: str = "global"
    execution_actor_loss_weighting: str = "active_step"

    def validate(self) -> None:
        if self.gamma_high != 1.0 or self.gamma_low != 1.0:
            raise ValueError("strict mission J requires gamma_high and gamma_low to equal 1")
        if not 0.0 <= self.lambda_high <= 1.0:
            raise ValueError("lambda_high must be in [0, 1]")
        if not 0.0 <= self.lambda_low <= 1.0:
            raise ValueError("lambda_low must be in [0, 1]")
        if self.clip_epsilon is not None and self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")
        if self.assignment_clip_epsilon <= 0.0 or self.execution_clip_epsilon <= 0.0:
            raise ValueError("assignment/execution clip epsilon must be positive")
        if self.entropy_coef is not None and self.entropy_coef < 0.0:
            raise ValueError("entropy_coef must be non-negative")
        if self.assignment_entropy_coef < 0.0 or self.execution_entropy_coef < 0.0:
            raise ValueError("assignment/execution entropy coefficient must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.actor_learning_rate is not None and self.actor_learning_rate <= 0.0:
            raise ValueError("actor_learning_rate must be positive")
        if self.critic_learning_rate is not None and self.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive")
        separate_rates = (
            self.assignment_actor_learning_rate,
            self.execution_actor_learning_rate,
            self.assignment_critic_learning_rate,
            self.execution_critic_learning_rate,
        )
        if any(rate <= 0.0 for rate in separate_rates):
            raise ValueError("separate actor/critic learning rates must be positive")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.critic_updates_per_actor <= 0:
            raise ValueError("critic_updates_per_actor must be positive")
        if self.actor_update_interval <= 0:
            raise ValueError("actor_update_interval must be positive")
        if self.sequence_length is not None and (
            not isinstance(self.sequence_length, int) or self.sequence_length <= 0
        ):
            raise ValueError("sequence_length must be a positive integer")
        if (
            not isinstance(self.assignment_sequence_length, int)
            or self.assignment_sequence_length <= 0
            or not isinstance(self.execution_sequence_length, int)
            or self.execution_sequence_length <= 0
        ):
            raise ValueError("assignment/execution sequence lengths must be positive integers")
        if self.assignment_target_kl <= 0.0 or self.execution_target_kl <= 0.0:
            raise ValueError("assignment/execution target KL must be positive")
        if (
            not math.isfinite(self.assignment_reward_learning_scale)
            or self.assignment_reward_learning_scale <= 0.0
        ):
            raise ValueError("assignment_reward_learning_scale must be positive and finite")
        if (
            not math.isfinite(self.execution_reward_learning_scale)
            or self.execution_reward_learning_scale <= 0.0
        ):
            raise ValueError("execution_reward_learning_scale must be positive and finite")
        if self.execution_value_loss not in {"mse", "huber"}:
            raise ValueError("execution_value_loss must be mse or huber")
        if (
            not math.isfinite(self.execution_value_huber_delta)
            or self.execution_value_huber_delta <= 0.0
        ):
            raise ValueError("execution_value_huber_delta must be positive and finite")
        if self.execution_post_step_kl_limit is not None and (
            not math.isfinite(self.execution_post_step_kl_limit)
            or self.execution_post_step_kl_limit <= 0.0
        ):
            raise ValueError("execution_post_step_kl_limit must be positive and finite")
        if (
            self.execution_post_step_kl_rollback
            and self.execution_post_step_kl_limit is None
        ):
            raise ValueError(
                "execution_post_step_kl_limit is required when rollback is enabled"
            )
        if self.effort_finetune_scale <= 0.0:
            raise ValueError("effort_finetune_scale must be positive")
        if self.execution_advantage_normalization not in {"global", "per_scenario"}:
            raise ValueError(
                "execution_advantage_normalization must be global or per_scenario"
            )
        if self.execution_actor_loss_weighting not in {"active_step", "per_scenario"}:
            raise ValueError(
                "execution_actor_loss_weighting must be active_step or per_scenario"
            )

    @property
    def high_sequence_length(self) -> int:
        return self.sequence_length or self.assignment_sequence_length

    @property
    def low_sequence_length(self) -> int:
        return self.sequence_length or self.execution_sequence_length

    @property
    def high_clip_epsilon(self) -> float:
        return self.clip_epsilon or self.assignment_clip_epsilon

    @property
    def low_clip_epsilon(self) -> float:
        return self.clip_epsilon or self.execution_clip_epsilon

    @property
    def high_entropy_coef(self) -> float:
        return self.assignment_entropy_coef if self.entropy_coef is None else self.entropy_coef

    @property
    def low_entropy_coef(self) -> float:
        return self.execution_entropy_coef if self.entropy_coef is None else self.entropy_coef
