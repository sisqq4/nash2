"""Rainbow DQN agent for the blue escape policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .replay_buffer import PrioritizedReplayBuffer, NStepBuffer, NStepTransition


class NoisyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1.0 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    @staticmethod
    def _scale_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class RainbowQNetwork(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        atom_size: int,
        hidden_dim: int = 128,
        noisy_std: float = 0.5,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.atom_size = atom_size

        self.feature = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, atom_size, noisy_std),
        )

        self.advantage_stream = nn.Sequential(
            NoisyLinear(hidden_dim, hidden_dim, noisy_std),
            nn.ReLU(),
            NoisyLinear(hidden_dim, action_dim * atom_size, noisy_std),
        )

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.feature(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)

        values = values.view(-1, 1, self.atom_size)
        advantages = advantages.view(-1, self.action_dim, self.atom_size)
        q_atoms = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_atoms


@dataclass
class RainbowDQNConfig:
    obs_dim: int
    action_dim: int
    lr: float = 1e-3
    gamma: float = 0.99
    batch_size: int = 64
    replay_size: int = 50_000
    start_learning: int = 1_000
    target_update_interval: int = 1_000
    device: str = "cpu"

    n_step: int = 3
    atom_size: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    noisy_std: float = 0.5

    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000


class RainbowDQNAgent:
    def __init__(self, cfg: RainbowDQNConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.q_net = RainbowQNetwork(
            cfg.obs_dim,
            cfg.action_dim,
            cfg.atom_size,
            noisy_std=cfg.noisy_std,
        ).to(self.device)
        self.target_q_net = RainbowQNetwork(
            cfg.obs_dim,
            cfg.action_dim,
            cfg.atom_size,
            noisy_std=cfg.noisy_std,
        ).to(self.device)
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.target_q_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.lr)
        self.replay = PrioritizedReplayBuffer(cfg.replay_size, cfg.obs_dim, cfg.per_alpha)
        self.n_step_buffer = NStepBuffer(cfg.n_step, cfg.gamma)

        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.start_learning = cfg.start_learning
        self.target_update_interval = cfg.target_update_interval

        self.atom_size = cfg.atom_size
        self.v_min = cfg.v_min
        self.v_max = cfg.v_max
        self.support = torch.linspace(cfg.v_min, cfg.v_max, cfg.atom_size).to(self.device)
        self.delta_z = (cfg.v_max - cfg.v_min) / (cfg.atom_size - 1)

        self.total_steps = 0

    def get_state(self) -> Dict[str, Any]:
        return {
            "q_net": self.q_net.state_dict(),
            "target_q_net": self.target_q_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "replay": self.replay.get_state(),
        }

    def load_state(self, state: Dict[str, Any]) -> None:
        self.q_net.load_state_dict(state["q_net"])
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

    def _beta(self) -> float:
        fraction = min(1.0, self.total_steps / max(1, self.cfg.per_beta_frames))
        return self.cfg.per_beta_start + fraction * (1.0 - self.cfg.per_beta_start)

    def select_action(self, obs: np.ndarray, eval_mode: bool = False) -> int:
        self.total_steps += 1

        obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        if eval_mode:
            self.q_net.eval()
        else:
            self.q_net.train()
            self.q_net.reset_noise()

        with torch.no_grad():
            dist = self.q_net(obs_t)
            prob = F.softmax(dist, dim=-1)
            q_values = (prob * self.support).sum(dim=-1)
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
        transition = NStepTransition(obs, int(act), float(rew), next_obs, bool(done))
        n_step_transition, ready = self.n_step_buffer.append(transition)
        if ready:
            self.replay.store(
                n_step_transition.obs,
                n_step_transition.action,
                n_step_transition.reward,
                n_step_transition.next_obs,
                n_step_transition.done,
            )
        if done:
            for remaining in self.n_step_buffer.flush():
                self.replay.store(
                    remaining.obs,
                    remaining.action,
                    remaining.reward,
                    remaining.next_obs,
                    remaining.done,
                )

    def update(self) -> Optional[float]:
        if self.replay.size < self.start_learning:
            return None
        if not self.replay.can_sample(self.batch_size):
            return None

        beta = self._beta()
        obs, act, rew, next_obs, done, weights, idxs = self.replay.sample(
            self.batch_size, beta
        )

        obs_t = torch.from_numpy(obs).float().to(self.device)
        act_t = torch.from_numpy(act).long().to(self.device)
        rew_t = torch.from_numpy(rew).float().to(self.device)
        next_obs_t = torch.from_numpy(next_obs).float().to(self.device)
        done_t = torch.from_numpy(done).float().to(self.device)
        weights_t = torch.from_numpy(weights).float().to(self.device)

        dist = self.q_net(obs_t)
        log_prob = F.log_softmax(dist, dim=-1)
        log_prob = log_prob[range(log_prob.size(0)), act_t]

        with torch.no_grad():
            next_dist = self.q_net(next_obs_t)
            next_prob = F.softmax(next_dist, dim=-1)
            next_q = (next_prob * self.support).sum(dim=-1)
            next_action = torch.argmax(next_q, dim=1)

            target_dist = self.target_q_net(next_obs_t)
            target_prob = F.softmax(target_dist, dim=-1)
            target_prob = target_prob[range(target_prob.size(0)), next_action]

            gamma_n = self.gamma ** self.cfg.n_step
            tz = rew_t.unsqueeze(1) + (1.0 - done_t.unsqueeze(1)) * gamma_n * self.support
            tz = tz.clamp(self.v_min, self.v_max)
            b = (tz - self.v_min) / self.delta_z
            l = b.floor().long()
            u = b.ceil().long()

            m = torch.zeros_like(target_prob)
            for atom in range(self.atom_size):
                l_idx = l[:, atom]
                u_idx = u[:, atom]
                prob = target_prob[:, atom]

                m[range(m.size(0)), l_idx] += prob * (u[:, atom].float() - b[:, atom])
                m[range(m.size(0)), u_idx] += prob * (b[:, atom] - l[:, atom].float())

                mask = l_idx == u_idx
                if mask.any():
                    m[range(m.size(0)), l_idx] += prob * mask.float()

        per_sample_loss = -(m * log_prob).sum(dim=1)
        loss = (per_sample_loss * weights_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        new_priorities = per_sample_loss.detach().cpu().numpy() + 1e-6
        self.replay.update_priorities(idxs, new_priorities)

        self.q_net.reset_noise()
        self.target_q_net.reset_noise()

        if self.total_steps % self.target_update_interval == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())

        return float(loss.item())
