from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .policy import PolicyRegistry
from .replay_buffer import NStepBuffer, PrioritizedReplayBuffer, Transition


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
    hidden_dim: int = 128
    learning_rate: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_size: int = 50_000
    learning_starts: int = 1_000
    target_update_interval: int = 1_000
    n_step: int = 3
    atoms: int = 51
    value_min: float = -10.0
    value_max: float = 10.0
    noisy_std: float = 0.5
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000
    device: str = "cpu"


class RainbowDQNAgent:
    """Rainbow DQN: dueling, noisy nets, C51, PER, n-step and Double DQN."""
    def __init__(self, config: RainbowDQNConfig) -> None:
        self.config = config; self.device = torch.device(config.device)
        args = (config.observation_dim, config.action_dim, config.atoms, config.hidden_dim, config.noisy_std)
        self.online, self.target = RainbowNetwork(*args).to(self.device), RainbowNetwork(*args).to(self.device)
        self.target.load_state_dict(self.online.state_dict()); self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=config.learning_rate)
        self.replay = PrioritizedReplayBuffer(config.replay_size, config.observation_dim, config.per_alpha)
        self.n_step = NStepBuffer(config.n_step, config.gamma); self.total_steps = 0
        self.support = torch.linspace(config.value_min, config.value_max, config.atoms, device=self.device)
        self.delta = (config.value_max - config.value_min) / (config.atoms - 1)

    def select_action(self, observation: np.ndarray, *, evaluation: bool = False) -> int:
        self.online.eval() if evaluation else self.online.train()
        if not evaluation: self.online.reset_noise()
        with torch.no_grad():
            logits = self.online(torch.as_tensor(observation, dtype=torch.float32, device=self.device)[None])
            return int((logits.softmax(-1) * self.support).sum(-1).argmax(1).item())

    def observe(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool) -> None:
        for item in self.n_step.append(Transition(observation, action, reward, next_observation, done)):
            self.replay.add(item.observation, item.action, item.reward, item.next_observation, item.done)
        self.total_steps += 1

    def update(self) -> float | None:
        c = self.config
        if self.replay.size < max(c.learning_starts, c.batch_size): return None
        beta = c.per_beta_start + min(1.0, self.total_steps / c.per_beta_frames) * (1.0 - c.per_beta_start)
        obs, actions, rewards, next_obs, dones, weights, indices = self.replay.sample(c.batch_size, beta)
        obs_t = torch.as_tensor(obs, device=self.device); next_t = torch.as_tensor(next_obs, device=self.device)
        actions_t = torch.as_tensor(actions, device=self.device); rewards_t = torch.as_tensor(rewards, device=self.device)
        dones_t = torch.as_tensor(dones, device=self.device); weights_t = torch.as_tensor(weights, device=self.device)
        log_prob = self.online(obs_t).log_softmax(-1)[torch.arange(c.batch_size, device=self.device), actions_t]
        with torch.no_grad():
            next_action = (self.online(next_t).softmax(-1) * self.support).sum(-1).argmax(1)
            next_prob = self.target(next_t).softmax(-1)[torch.arange(c.batch_size, device=self.device), next_action]
            target_z = rewards_t[:, None] + (1 - dones_t[:, None]) * (c.gamma ** c.n_step) * self.support
            target_z = target_z.clamp(c.value_min, c.value_max)
            b = (target_z - c.value_min) / self.delta; lower, upper = b.floor().long(), b.ceil().long()
            projected = torch.zeros_like(next_prob)
            offset = torch.arange(c.batch_size, device=self.device)[:, None] * c.atoms
            projected.view(-1).index_add_(0, (lower + offset).view(-1), (next_prob * (upper.float() - b + (lower == upper))).view(-1))
            projected.view(-1).index_add_(0, (upper + offset).view(-1), (next_prob * (b - lower.float())).view(-1))
        losses = -(projected * log_prob).sum(-1); loss = (losses * weights_t).mean()
        self.optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(self.online.parameters(), 10.0); self.optimizer.step()
        self.replay.update_priorities(indices, losses.detach().cpu().numpy())
        if self.total_steps % c.target_update_interval == 0: self.target.load_state_dict(self.online.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        torch.save({"config": asdict(self.config), "online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "total_steps": self.total_steps}, path)

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RainbowDQNAgent":
        data = torch.load(path, map_location=device, weights_only=False); data["config"]["device"] = device
        agent = cls(RainbowDQNConfig(**data["config"])); agent.online.load_state_dict(data["online"])
        agent.target.load_state_dict(data["target"]); agent.optimizer.load_state_dict(data["optimizer"])
        agent.total_steps = int(data.get("total_steps", 0)); return agent


PolicyRegistry.register("rainbow", RainbowDQNAgent)
