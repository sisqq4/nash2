
"""Simple experience replay buffer for DQN-style agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)

        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)

        self.size = 0
        self.ptr = 0

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.act_buf[idx] = int(act)
        self.rew_buf[idx] = float(rew)
        self.done_buf[idx] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        assert self.size > 0, "Buffer is empty"
        batch_size = min(batch_size, self.size)
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
        )

    def get_state(self) -> dict:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "obs_buf": self.obs_buf.copy(),
            "next_obs_buf": self.next_obs_buf.copy(),
            "act_buf": self.act_buf.copy(),
            "rew_buf": self.rew_buf.copy(),
            "done_buf": self.done_buf.copy(),
            "size": self.size,
            "ptr": self.ptr,
        }

    def load_state(self, state: dict) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.obs_dim = int(state.get("obs_dim", self.obs_dim))
        self.obs_buf = np.array(state["obs_buf"], dtype=np.float32)
        self.next_obs_buf = np.array(state["next_obs_buf"], dtype=np.float32)
        self.act_buf = np.array(state["act_buf"], dtype=np.int64)
        self.rew_buf = np.array(state["rew_buf"], dtype=np.float32)
        self.done_buf = np.array(state["done_buf"], dtype=np.float32)
        self.size = int(state.get("size", 0))
        self.ptr = int(state.get("ptr", 0))

class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, alpha: float = 0.6) -> None:
        self.capacity = int(capacity)
        self.obs_dim = int(obs_dim)
        self.alpha = float(alpha)

        self.obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((self.capacity,), dtype=np.int64)
        self.rew_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.done_buf = np.zeros((self.capacity,), dtype=np.float32)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)

        self.size = 0
        self.ptr = 0
        self.max_priority = 1.0

    def store(
        self,
        obs: np.ndarray,
        act: int,
        rew: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        idx = self.ptr
        self.obs_buf[idx] = obs
        self.next_obs_buf[idx] = next_obs
        self.act_buf[idx] = int(act)
        self.rew_buf[idx] = float(rew)
        self.done_buf[idx] = float(done)
        self.priorities[idx] = self.max_priority

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch_size: int) -> bool:
        return self.size >= batch_size

    def sample(self, batch_size: int, beta: float) -> Tuple[np.ndarray, ...]:
        assert self.size > 0, "Buffer is empty"
        batch_size = min(batch_size, self.size)
        priorities = self.priorities[: self.size] ** self.alpha
        probs = priorities / priorities.sum()
        idxs = np.random.choice(self.size, batch_size, p=probs)

        weights = (self.size * probs[idxs]) ** (-beta)
        weights /= weights.max() if weights.size > 0 else 1.0

        return (
            self.obs_buf[idxs],
            self.act_buf[idxs],
            self.rew_buf[idxs],
            self.next_obs_buf[idxs],
            self.done_buf[idxs],
            weights.astype(np.float32),
            idxs,
        )

    def update_priorities(self, idxs: np.ndarray, priorities: np.ndarray) -> None:
        priorities = np.asarray(priorities, dtype=np.float32)
        self.priorities[idxs] = priorities
        self.max_priority = max(self.max_priority, float(priorities.max(initial=0.0)))

    def get_state(self) -> dict:
        return {
            "capacity": self.capacity,
            "obs_dim": self.obs_dim,
            "alpha": self.alpha,
            "obs_buf": self.obs_buf.copy(),
            "next_obs_buf": self.next_obs_buf.copy(),
            "act_buf": self.act_buf.copy(),
            "rew_buf": self.rew_buf.copy(),
            "done_buf": self.done_buf.copy(),
            "priorities": self.priorities.copy(),
            "size": self.size,
            "ptr": self.ptr,
            "max_priority": self.max_priority,
        }

    def load_state(self, state: dict) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.obs_dim = int(state.get("obs_dim", self.obs_dim))
        self.alpha = float(state.get("alpha", self.alpha))
        self.obs_buf = np.array(state["obs_buf"], dtype=np.float32)
        self.next_obs_buf = np.array(state["next_obs_buf"], dtype=np.float32)
        self.act_buf = np.array(state["act_buf"], dtype=np.int64)
        self.rew_buf = np.array(state["rew_buf"], dtype=np.float32)
        self.done_buf = np.array(state["done_buf"], dtype=np.float32)
        self.priorities = np.array(state.get("priorities", self.priorities), dtype=np.float32)
        self.size = int(state.get("size", 0))
        self.ptr = int(state.get("ptr", 0))
        self.max_priority = float(state.get("max_priority", 1.0))


@dataclass
class NStepTransition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool


class NStepBuffer:
    def __init__(self, n_step: int, gamma: float) -> None:
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.buffer: List[NStepTransition] = []

    def reset(self) -> None:
        self.buffer.clear()

    def append(self, transition: NStepTransition) -> Tuple[NStepTransition, bool]:
        self.buffer.append(transition)
        if len(self.buffer) < self.n_step:
            return transition, False
        return self._build_n_step(self.n_step), True

    def pop(self) -> Tuple[NStepTransition, bool]:
        if not self.buffer:
            return NStepTransition(np.array([]), 0, 0.0, np.array([]), False), False
        self.buffer.pop(0)
        if len(self.buffer) >= self.n_step:
            return self._build_n_step(self.n_step), True
        return NStepTransition(np.array([]), 0, 0.0, np.array([]), False), False

    def flush(self) -> List[NStepTransition]:
        transitions: List[NStepTransition] = []
        while self.buffer:
            steps = min(len(self.buffer), self.n_step)
            transitions.append(self._build_n_step(steps))
            self.buffer.pop(0)
        return transitions

    def _build_n_step(self, steps: int) -> NStepTransition:
        reward, next_obs, done = 0.0, self.buffer[-1].next_obs, self.buffer[-1].done
        for idx, item in enumerate(self.buffer[: steps]):
            reward += (self.gamma**idx) * item.reward
            next_obs = item.next_obs
            done = item.done
            if done:
                break
        first = self.buffer[0]
        return NStepTransition(first.obs, first.action, reward, next_obs, done)