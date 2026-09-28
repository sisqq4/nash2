from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .policy import PolicyRegistry
from .replay_buffer import NStepBuffer, PrioritizedReplayBuffer, Transition


def _require_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


class NoisyLinear(nn.Module):
    def __init__(self, inputs: int, outputs: int, std: float = 0.5) -> None:
        super().__init__()
        bound = inputs ** -0.5
        self.weight_mu = nn.Parameter(torch.empty(outputs, inputs).uniform_(-bound, bound))
        self.weight_sigma = nn.Parameter(torch.full((outputs, inputs), std / inputs ** 0.5))
        self.bias_mu = nn.Parameter(torch.empty(outputs).uniform_(-bound, bound))
        self.bias_sigma = nn.Parameter(torch.full((outputs,), std / outputs ** 0.5))
        self.register_buffer("weight_epsilon", torch.zeros(outputs, inputs))
        self.register_buffer("bias_epsilon", torch.zeros(outputs))
        self.reset_noise()

    def reset_noise(self) -> None:
        def noise(size: int) -> torch.Tensor:
            value = torch.randn(size, device=self.weight_mu.device)
            return value.sign() * value.abs().sqrt()
        output_noise, input_noise = noise(self.weight_mu.shape[0]), noise(self.weight_mu.shape[1])
        self.weight_epsilon.copy_(output_noise.outer(input_noise)); self.bias_epsilon.copy_(output_noise)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            return F.linear(value, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)
        return F.linear(value, self.weight_mu, self.bias_mu)


class RainbowNetwork(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, atoms: int, hidden: int, noisy_std: float) -> None:
        super().__init__(); self.action_dim, self.atoms = action_dim, atoms
        self.features = nn.Sequential(nn.Linear(observation_dim, hidden), nn.ReLU())
        self.value = nn.Sequential(NoisyLinear(hidden, hidden, noisy_std), nn.ReLU(), NoisyLinear(hidden, atoms, noisy_std))
        self.advantage = nn.Sequential(NoisyLinear(hidden, hidden, noisy_std), nn.ReLU(), NoisyLinear(hidden, action_dim * atoms, noisy_std))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        features = self.features(value)
        val = self.value(features).view(-1, 1, self.atoms)
        adv = self.advantage(features).view(-1, self.action_dim, self.atoms)
        return val + adv - adv.mean(1, keepdim=True)

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear): module.reset_noise()


@dataclass
class RainbowDQNConfig:
    observation_dim: int
    action_dim: int
    observation_schema: str = "legacy_v1"
    hidden_dim: int = 128
    learning_rate: float = 2.5e-4
    gamma: float = 0.999
    batch_size: int = 64
    replay_size: int = 50_000
    learning_starts: int = 1_000
    target_update_interval: int = 1_000
    n_step: int = 20
    atoms: int = 51
    # Covers the bounded tactical potential plus terminal/time bonuses.  Keep
    # these under review using projection clamp metrics on full training runs.
    value_min: float = -12.0
    value_max: float = 12.0
    noisy_std: float = 0.5
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000
    device: str = "cpu"
    # Serialized into every new checkpoint so evaluation reconstructs exactly
    # the same predictive constraint layer as behavior collection.
    flight_envelope_config: dict[str, object] | None = None
    # Versioned physical-state estimator used by normalized_v4 and its reward.
    mechanism_reward_config: dict[str, object] | None = None


class RainbowDQNAgent:
    """Rainbow DQN: dueling, noisy nets, C51, PER, n-step and Double DQN."""
    _TRAINING_STATE_FORMAT = "rainbow_dqn_training_state"
    _TRAINING_STATE_VERSION = 1

    def __init__(self, config: RainbowDQNConfig) -> None:
        self.config = config; self.device = torch.device(config.device)
        args = (config.observation_dim, config.action_dim, config.atoms, config.hidden_dim, config.noisy_std)
        self.online, self.target = RainbowNetwork(*args).to(self.device), RainbowNetwork(*args).to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = PrioritizedReplayBuffer(config.replay_size, config.observation_dim,
                                              config.per_alpha, config.action_dim)
        self.n_step = NStepBuffer(config.n_step, config.gamma)
        self._n_step_by_env: dict[int, NStepBuffer] = {0: self.n_step}
        self.total_steps = self.optimizer_updates = self.target_updates = 0
        self._last_target_sync_step = 0
        self.last_action_metrics: dict[str, float] = {}
        self.last_update_metrics: dict[str, float] = {}
        self.support = torch.linspace(config.value_min, config.value_max, config.atoms, device=self.device)
        self.delta = (config.value_max - config.value_min) / (config.atoms - 1)

    def select_action(self, observation: np.ndarray, *, evaluation: bool = False) -> int:
        return int(self.select_actions(np.asarray(observation)[None], evaluation=evaluation)[0])

    def expected_action_values(self, observations: np.ndarray, *, evaluation: bool = True) -> np.ndarray:
        """Return all C51 expectations for shared constraint arbitration."""
        array = np.asarray(observations, dtype=np.float32)
        if array.ndim == 1: array = array[None, :]
        if array.ndim != 2 or array.shape[1] != self.config.observation_dim:
            raise ValueError(f"observations must have shape [batch, {self.config.observation_dim}]")
        self.online.eval() if evaluation else self.online.train()
        if not evaluation:
            self.online.reset_noise()
        with torch.inference_mode():
            tensor = torch.as_tensor(array, dtype=torch.float32, device=self.device)
            return ((self.online(tensor).softmax(-1) * self.support).sum(-1)).cpu().numpy()

    def select_actions(self, observations: np.ndarray, *, evaluation: bool = False) -> np.ndarray:
        """Select a batch of actions in one device call for vector environments."""
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim != 2 or values.shape[1] != self.config.observation_dim:
            raise ValueError(
                f"observations must have shape [batch, {self.config.observation_dim}]"
            )
        self.online.eval() if evaluation else self.online.train()
        if not evaluation: self.online.reset_noise()
        with torch.no_grad():
            logits = self.online(torch.as_tensor(values, device=self.device))
            expected_values = (logits.softmax(-1) * self.support).sum(-1)
            actions = expected_values.argmax(1)
            selected_values = expected_values.gather(1, actions[:, None]).squeeze(1)
            self.last_action_metrics = {
                "selected_value_mean": float(selected_values.mean().item()),
                "selected_value_std": float(selected_values.std(unbiased=False).item()),
                "action_batch_size": float(values.shape[0]),
            }
        return actions.cpu().numpy().astype(np.int64, copy=False)

    def observe(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool) -> None:
        self.observe_for_env(0, observation, action, reward, next_observation, done)

    def observe_for_env(self, env_id: int, observation: np.ndarray, action: int, reward: float,
                        next_observation: np.ndarray, done: bool, *,
                        next_action_mask: np.ndarray | None = None,
                        next_action_penalty: np.ndarray | None = None) -> None:
        """Add one transition without mixing n-step sequences across environments."""
        buffer = self._n_step_by_env.setdefault(
            int(env_id), NStepBuffer(self.config.n_step, self.config.gamma)
        )
        transition = Transition(observation, action, reward, next_observation, done,
                                next_action_mask, next_action_penalty)
        for item in buffer.append(transition):
            self.replay.add(item.observation, item.action, item.reward, item.next_observation, item.done,
                            item.next_action_mask, item.next_action_penalty)
        self.total_steps += 1

    def update(self) -> float | None:
        c = self.config
        if self.replay.size < max(c.learning_starts, c.batch_size): return None
        beta = c.per_beta_start + min(1.0, self.total_steps / c.per_beta_frames) * (1.0 - c.per_beta_start)
        (obs, actions, rewards, next_obs, dones, next_masks, next_penalties,
         weights, indices) = self.replay.sample(c.batch_size, beta)
        obs_t = torch.as_tensor(obs, device=self.device); next_t = torch.as_tensor(next_obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device); rewards_t = torch.as_tensor(rewards, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device); weights_t = torch.as_tensor(weights, device=self.device)
        next_masks_t = torch.as_tensor(next_masks, dtype=torch.bool, device=self.device)
        next_penalties_t = torch.as_tensor(next_penalties, device=self.device)
        log_prob = self.online(obs_t).log_softmax(-1)[torch.arange(c.batch_size, device=self.device), actions_t]
        with torch.no_grad():
            next_values = (self.online(next_t).softmax(-1) * self.support).sum(-1) - next_penalties_t
            next_values = next_values.masked_fill(~next_masks_t, -torch.inf)
            next_action = next_values.argmax(1)
            next_prob = self.target(next_t).softmax(-1)[torch.arange(c.batch_size, device=self.device), next_action]
            target_z_unclamped = rewards_t[:, None] + (1 - dones_t[:, None]) * (c.gamma ** c.n_step) * self.support
            clamp_low_fraction = (target_z_unclamped < c.value_min).float().mean()
            clamp_high_fraction = (target_z_unclamped > c.value_max).float().mean()
            target_z = target_z_unclamped.clamp(c.value_min, c.value_max)
            b = (target_z - c.value_min) / self.delta; lower, upper = b.floor().long(), b.ceil().long()
            projected = torch.zeros_like(next_prob)
            offset = torch.arange(c.batch_size, device=self.device)[:, None] * c.atoms
            projected.view(-1).index_add_(0, (lower + offset).view(-1), (next_prob * (upper.float() - b + (lower == upper))).view(-1))
            projected.view(-1).index_add_(0, (upper + offset).view(-1), (next_prob * (b - lower.float())).view(-1))
        losses = -(projected * log_prob).sum(-1); loss = (losses * weights_t).mean()
        self.optimizer.zero_grad(); loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(self.online.parameters(), 10.0)
        self.optimizer.step(); self.optimizer_updates += 1
        priorities = losses.detach().cpu().numpy()
        self.replay.update_priorities(indices, priorities)
        target_synced = self.total_steps - self._last_target_sync_step >= c.target_update_interval
        if target_synced:
            self.target.load_state_dict(self.online.state_dict())
            self._last_target_sync_step = self.total_steps; self.target_updates += 1
        loss_value = float(loss.item())
        self.last_update_metrics = {
            "loss": loss_value,
            "unweighted_loss_mean": float(losses.mean().item()),
            "unweighted_loss_std": float(losses.std(unbiased=False).item()),
            "priority_mean": float(priorities.mean()),
            "priority_max": float(priorities.max()),
            "gradient_norm": float(gradient_norm.item()),
            "per_beta": float(beta),
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "replay_size": float(self.replay.size),
            "optimizer_updates": float(self.optimizer_updates),
            "target_updates": float(self.target_updates),
            "target_synced": float(target_synced),
            "c51_clamp_low_fraction": float(clamp_low_fraction.item()),
            "c51_clamp_high_fraction": float(clamp_high_fraction.item()),
        }
        return loss_value

    def _model_state_dict(self) -> dict[str, Any]:
        """State used by the established lightweight inference checkpoint."""
        return {
            "config": asdict(self.config),
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "optimizer_updates": self.optimizer_updates,
            "target_updates": self.target_updates,
            "last_target_sync_step": self._last_target_sync_step,
        }

    def save(self, path: str) -> None:
        """Save the existing compact model checkpoint (without replay data)."""
        torch.save(self._model_state_dict(), path)

    def assert_resume_safe(self) -> None:
        """Require a clean episode boundary before an outer resume snapshot."""
        pending = {
            int(env_id): len(buffer.items)
            for env_id, buffer in self._n_step_by_env.items()
            if buffer.items
        }
        if pending:
            details = ", ".join(
                f"env {env_id}: {count}" for env_id, count in sorted(pending.items())
            )
            raise RuntimeError(
                "resume snapshot requires empty n-step queues; pending transitions: " + details
            )

    def training_state_dict(self) -> dict[str, Any]:
        """Export all agent state needed to continue the same training run.

        Unlike :meth:`save`, this includes PER storage, every per-environment
        n-step queue, diagnostic metrics, modes, and learner counters.  RNG and
        outer-loop scheduling state belong to the training entry point because
        they are shared with environment/scenario generation.
        """
        if 0 not in self._n_step_by_env or self.n_step is not self._n_step_by_env[0]:
            raise RuntimeError("primary n-step buffer must alias environment 0")
        state = self._model_state_dict()
        state.update({
            "format": self._TRAINING_STATE_FORMAT,
            "version": self._TRAINING_STATE_VERSION,
            "replay": self.replay.state_dict(),
            "n_step_by_env": {
                int(env_id): buffer.state_dict()
                for env_id, buffer in sorted(self._n_step_by_env.items())
            },
            "last_action_metrics": dict(self.last_action_metrics),
            "last_update_metrics": dict(self.last_update_metrics),
            "online_training": bool(self.online.training),
        })
        return state

    @staticmethod
    def _validated_metrics(name: str, value: Any) -> dict[str, float]:
        if not isinstance(value, Mapping):
            raise TypeError(f"{name} must be a mapping")
        metrics: dict[str, float] = {}
        for key, metric in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{name} keys must be strings")
            try:
                number = float(metric)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"{name}[{key!r}] must be numeric") from exc
            if not np.isfinite(number):
                raise ValueError(f"{name}[{key!r}] must be finite")
            metrics[key] = number
        return metrics

    def load_training_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a complete state produced by :meth:`training_state_dict`.

        The current agent configuration must match the saved configuration,
        except for ``device`` so a run can be resumed on another supported
        accelerator.  Structural replay and n-step validation occurs before
        learner parameters are changed.
        """
        required = {
            "format", "version", "config", "online", "target", "optimizer",
            "total_steps", "optimizer_updates", "target_updates",
            "last_target_sync_step", "replay", "n_step_by_env",
            "last_action_metrics", "last_update_metrics", "online_training",
        }
        if not isinstance(state, Mapping):
            raise TypeError("Rainbow training state must be a mapping")
        missing, extra = required - set(state), set(state) - required
        if missing or extra:
            raise ValueError(
                f"invalid Rainbow training state keys: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        if state["format"] != self._TRAINING_STATE_FORMAT:
            raise ValueError(f"unsupported Rainbow training state format: {state['format']!r}")
        version = _require_integer("Rainbow training state version", state["version"])
        if version != self._TRAINING_STATE_VERSION:
            raise ValueError(f"unsupported Rainbow training state version: {state['version']!r}")
        saved_config = state["config"]
        if not isinstance(saved_config, Mapping):
            raise TypeError("Rainbow training config must be a mapping")
        expected_config = asdict(self.config)
        comparable_saved = dict(saved_config); comparable_saved.pop("device", None)
        comparable_expected = dict(expected_config); comparable_expected.pop("device", None)
        if comparable_saved != comparable_expected:
            differing = sorted(
                key for key in set(comparable_saved) | set(comparable_expected)
                if comparable_saved.get(key) != comparable_expected.get(key)
            )
            raise ValueError(f"Rainbow training config mismatch for: {', '.join(differing)}")

        counters: dict[str, int] = {}
        for name in ("total_steps", "optimizer_updates", "target_updates", "last_target_sync_step"):
            value = _require_integer(f"Rainbow {name}", state[name])
            if value < 0:
                raise ValueError(f"Rainbow {name} must be non-negative")
            counters[name] = value
        if counters["last_target_sync_step"] > counters["total_steps"]:
            raise ValueError("last_target_sync_step cannot exceed total_steps")

        # Validate queues and scalar metadata before mutating the large replay
        # allocation already created by __init__.  Reusing it avoids a second
        # full-capacity buffer during recovery.
        raw_buffers = state["n_step_by_env"]
        if not isinstance(raw_buffers, Mapping) or not raw_buffers:
            raise ValueError("n_step_by_env must be a non-empty mapping")
        buffers: dict[int, NStepBuffer] = {}
        for raw_env_id, raw_buffer in raw_buffers.items():
            env_id = _require_integer("n-step environment id", raw_env_id)
            if env_id < 0 or env_id in buffers:
                raise ValueError(f"invalid or duplicate n-step environment id: {raw_env_id!r}")
            buffer = NStepBuffer(self.config.n_step, self.config.gamma)
            buffer.load_state_dict(raw_buffer)
            for transition in buffer.items:
                if transition.observation.shape != (self.config.observation_dim,):
                    raise ValueError(
                        f"n-step observation for environment {env_id} must have shape "
                        f"({self.config.observation_dim},)"
                    )
                if not 0 <= transition.action < self.config.action_dim:
                    raise ValueError(f"n-step action for environment {env_id} is out of range")
                if transition.next_action_mask is not None and (
                    transition.next_action_mask.shape != (self.config.action_dim,)
                ):
                    raise ValueError(f"n-step action mask for environment {env_id} has wrong shape")
                if transition.next_action_penalty is not None and (
                    transition.next_action_penalty.shape != (self.config.action_dim,)
                ):
                    raise ValueError(f"n-step action penalty for environment {env_id} has wrong shape")
            buffers[env_id] = buffer
        if 0 not in buffers:
            raise ValueError("n_step_by_env must contain the primary environment 0")
        last_action_metrics = self._validated_metrics(
            "last_action_metrics", state["last_action_metrics"]
        )
        last_update_metrics = self._validated_metrics(
            "last_update_metrics", state["last_update_metrics"]
        )
        if not isinstance(state["online_training"], (bool, np.bool_)):
            raise TypeError("online_training must be a boolean")

        self.replay.load_state_dict(state["replay"])
        self.online.load_state_dict(state["online"])
        self.target.load_state_dict(state["target"])
        self.optimizer.load_state_dict(state["optimizer"])
        self._n_step_by_env = buffers
        self.n_step = buffers[0]
        self.total_steps = counters["total_steps"]
        self.optimizer_updates = counters["optimizer_updates"]
        self.target_updates = counters["target_updates"]
        self._last_target_sync_step = counters["last_target_sync_step"]
        self.last_action_metrics = last_action_metrics
        self.last_update_metrics = last_update_metrics
        self.online.train(bool(state["online_training"]))
        self.target.eval()

    @classmethod
    def from_training_state_dict(
        cls, state: Mapping[str, Any], device: str = "cpu",
    ) -> "RainbowDQNAgent":
        """Construct an agent and restore a complete training state."""
        if not isinstance(state, Mapping) or not isinstance(state.get("config"), Mapping):
            raise ValueError("Rainbow training state is missing its config")
        config = dict(state["config"]); config["device"] = device
        agent = cls(RainbowDQNConfig(**config))
        agent.load_training_state_dict(state)
        return agent

    def save_training_state(self, path: str) -> None:
        """Save only the complete agent state; outer-loop state is separate."""
        torch.save(self.training_state_dict(), path)

    @classmethod
    def load_training_state(cls, path: str, device: str = "cpu") -> "RainbowDQNAgent":
        state = torch.load(path, map_location=device, weights_only=False)
        return cls.from_training_state_dict(state, device)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RainbowDQNAgent":
        data = torch.load(path, map_location=device, weights_only=False); data["config"]["device"] = device
        agent = cls(RainbowDQNConfig(**data["config"])); agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"]); agent.optimizer.load_state_dict(data["optimizer"])
        agent.total_steps = _require_integer("Rainbow total_steps", data.get("total_steps", 0))
        agent.optimizer_updates = _require_integer(
            "Rainbow optimizer_updates", data.get("optimizer_updates", 0)
        )
        agent.target_updates = _require_integer(
            "Rainbow target_updates", data.get("target_updates", 0)
        )
        agent._last_target_sync_step = _require_integer(
            "Rainbow last_target_sync_step",
            data.get("last_target_sync_step", agent.total_steps),
        )
        return agent


PolicyRegistry.register("rainbow", RainbowDQNAgent)
