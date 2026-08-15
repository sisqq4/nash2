
"""DQN agent for the blue escape policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class DQNConfig:
    obs_dim: int
    action_dim: int
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_size: int = 50_000
    start_learning: int = 1_000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 20_000
    target_update_interval: int = 1_000
    device: str = "cpu"


class DQNAgent:
    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.q_net = QNetwork(cfg.obs_dim, cfg.action_dim).to(self.device)
        self.target_q_net = QNetwork(cfg.obs_dim, cfg.action_dim).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay = ReplayBuffer(cfg.replay_size, cfg.obs_dim)

        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.start_learning = cfg.start_learning

        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.total_steps = 0
        self.loaded_from_checkpoint = False

        self.target_update_interval = cfg.target_update_interval

    def get_state(self) -> Dict[str, Any]:
        return {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "replay": self.replay.get_state(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        q_state = state["q_net"]
        first_weight = q_state.get("net.0.weight")
        last_weight = q_state.get("net.4.weight")
        if first_weight is not None and int(first_weight.shape[1]) != self.cfg.obs_dim:
            raise ValueError(
                "Checkpoint observation dimension does not match the current DQN model: "
                f"checkpoint obs_dim={int(first_weight.shape[1])}, current obs_dim={self.cfg.obs_dim}. "
                "Ensure EnvConfig.num_missiles matches the checkpoint, or use blue_eval_policy=bt "
                "when the DQN policy is not being evaluated."
            )
        if last_weight is not None and int(last_weight.shape[0]) != self.cfg.action_dim:
            raise ValueError(
                "Checkpoint action dimension does not match the current DQN model: "
                f"checkpoint action_dim={int(last_weight.shape[0])}, current action_dim={self.cfg.action_dim}."
            )

        self.q_net.load_state_dict(q_state)
        target_state = state.get("target_q_net")
        if target_state is None:
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        else:
            self.target_q_net.load_state_dict(target_state)
        optimizer_state = state.get("optimizer")
        if optimizer_state is not None:
            self.optimizer.load_state_dict(optimizer_state)
        self.total_steps = int(state.get("total_steps", 0))
        replay_state = state.get("replay")
        if replay_state is not None:
            self.replay.load_state(replay_state)
        self.loaded_from_checkpoint = True

    def _epsilon(self, eval_mode: bool) -> float:
        if eval_mode:
            return 0.0
        frac = min(1.0, self.total_steps / max(1, self.epsilon_decay))
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        self.total_steps += 1

        eps = self._epsilon(eval_mode)
        if not eval_mode and np.random.rand() < eps:
            return int(np.random.randint(0, self.cfg.action_dim))

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_net(obs_t)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    def store_transition(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.store(obs, act, rew, next_obs, done)

    def update(self) -> Optional[float]:
        if self.replay.size < self.start_learning:
            return None
        if not self.replay.can_sample(self.batch_size):
            return None

        obs, act, rew, next_obs, done = self.replay.sample(self.batch_size)

        obs_t = torch.from_numpy(obs).float().to(self.device)
        act_t = torch.from_numpy(act).long().to(self.device)
        rew_t = torch.from_numpy(rew).float().to(self.device)
        next_obs_t = torch.from_numpy(next_obs).float().to(self.device)
        done_t = torch.from_numpy(done).float().to(self.device)

        q_values = self.q_net(obs_t)
        q_sa = q_values.gather(1, act_t.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q_values = self.target_q_net(next_obs_t)
            next_q_max, _ = torch.max(next_q_values, dim=1)
            target = rew_t + self.gamma * (1.0 - done_t) * next_q_max

        loss = torch.nn.functional.mse_loss(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.total_steps % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
