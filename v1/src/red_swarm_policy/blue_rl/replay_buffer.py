from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


class PrioritizedReplayBuffer:
    def __init__(self, capacity: int, observation_dim: int, alpha: float = 0.6) -> None:
        self.capacity, self.observation_dim, self.alpha = int(capacity), int(observation_dim), float(alpha)
        self.observations = np.zeros((capacity, observation_dim), np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.priorities = np.zeros(capacity, np.float32)
        self.position = self.size = 0
        self.max_priority = 1.0

    def add(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray, done: bool) -> None:
        i = self.position
        self.observations[i], self.next_observations[i] = observation, next_observation
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        self.priorities[i] = self.max_priority
        self.position = (i + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float) -> tuple[np.ndarray, ...]:
        scaled = self.priorities[: self.size] ** self.alpha
        probabilities = scaled / scaled.sum()
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        weights = (self.size * probabilities[indices]) ** -beta
        weights /= weights.max()
        return (self.observations[indices], self.actions[indices], self.rewards[indices],
                self.next_observations[indices], self.dones[indices], weights.astype(np.float32), indices)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        values = np.maximum(np.asarray(priorities, np.float32), 1e-6)
        self.priorities[indices] = values
        self.max_priority = max(self.max_priority, float(values.max()))


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool


class NStepBuffer:
    def __init__(self, steps: int, gamma: float) -> None:
        self.steps, self.gamma = int(steps), float(gamma)
        self.items: deque[Transition] = deque()

    def append(self, transition: Transition) -> list[Transition]:
        self.items.append(transition)
        ready: list[Transition] = []
        if len(self.items) >= self.steps:
            ready.append(self._build(min(self.steps, len(self.items))))
            self.items.popleft()
        if transition.done:
            while self.items:
                ready.append(self._build(min(self.steps, len(self.items))))
                self.items.popleft()
        return ready

    def _build(self, count: int) -> Transition:
        chosen = list(self.items)[:count]
        reward = sum((self.gamma ** i) * item.reward for i, item in enumerate(chosen))
        return Transition(chosen[0].observation, chosen[0].action, reward,
                          chosen[-1].next_observation, chosen[-1].done)
