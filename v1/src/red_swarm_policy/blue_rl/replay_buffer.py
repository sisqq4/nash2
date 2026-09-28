from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


def _require_integer(name: str, value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


class PrioritizedReplayBuffer:
    _STATE_VERSION = 1

    def __init__(self, capacity: int, observation_dim: int, alpha: float = 0.6,
                 action_dim: int = 29) -> None:
        self.capacity, self.observation_dim, self.alpha = int(capacity), int(observation_dim), float(alpha)
        self.action_dim = int(action_dim)
        self.observations = np.zeros((capacity, observation_dim), np.float32)
        self.next_observations = np.zeros_like(self.observations)
        self.actions = np.zeros(capacity, np.int64)
        self.rewards = np.zeros(capacity, np.float32)
        self.dones = np.zeros(capacity, np.float32)
        self.next_action_masks = np.ones((capacity, self.action_dim), dtype=bool)
        self.next_action_penalties = np.zeros((capacity, self.action_dim), np.float32)
        self.priorities = np.zeros(capacity, np.float32)
        self.position = self.size = 0
        self.max_priority = 1.0

    def add(self, observation: np.ndarray, action: int, reward: float, next_observation: np.ndarray,
            done: bool, next_action_mask: np.ndarray | None = None,
            next_action_penalty: np.ndarray | None = None) -> None:
        i = self.position
        self.observations[i], self.next_observations[i] = observation, next_observation
        self.actions[i], self.rewards[i], self.dones[i] = action, reward, done
        if next_action_mask is None:
            self.next_action_masks[i] = True
        else:
            mask = np.asarray(next_action_mask, dtype=bool)
            if mask.shape != (self.action_dim,) or not mask.any():
                raise ValueError(f"next_action_mask must have shape ({self.action_dim},) and allow an action")
            self.next_action_masks[i] = mask
        if next_action_penalty is None:
            self.next_action_penalties[i] = 0.0
        else:
            penalty = np.asarray(next_action_penalty, dtype=np.float32)
            if penalty.shape != (self.action_dim,) or not np.all(np.isfinite(penalty)) or np.any(penalty < 0.0):
                raise ValueError(f"next_action_penalty must be a finite non-negative ({self.action_dim},) vector")
            self.next_action_penalties[i] = penalty
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
                self.next_observations[indices], self.dones[indices],
                self.next_action_masks[indices], self.next_action_penalties[indices],
                weights.astype(np.float32), indices)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        values = np.maximum(np.asarray(priorities, np.float32), 1e-6)
        self.priorities[indices] = values
        self.max_priority = max(self.max_priority, float(values.max()))

    def state_dict(self) -> dict[str, Any]:
        """Return the exact replay state while omitting unused capacity.

        Before the ring first fills, valid transitions always occupy the prefix
        ``[0:size]``.  Saving only that prefix keeps early resume snapshots much
        smaller.  Once full, physical array order and ``position`` are retained
        so that the next insertion overwrites the same transition after resume.
        """
        stored = self.capacity if self.size == self.capacity else self.size
        return {
            "version": self._STATE_VERSION,
            "capacity": self.capacity,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "alpha": self.alpha,
            "size": self.size,
            "position": self.position,
            "max_priority": self.max_priority,
            "storage": "full" if self.size == self.capacity else "prefix",
            # These are read-only views for immediate serialization, matching
            # PyTorch's state_dict reference semantics without doubling a
            # large replay buffer in memory at each safe boundary.
            "observations": self.observations[:stored],
            "next_observations": self.next_observations[:stored],
            "actions": self.actions[:stored],
            "rewards": self.rewards[:stored],
            "dones": self.dones[:stored],
            "next_action_masks": self.next_action_masks[:stored],
            "next_action_penalties": self.next_action_penalties[:stored],
            "priorities": self.priorities[:stored],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a replay state, rejecting incompatible or corrupt data."""
        required = {
            "version", "capacity", "observation_dim", "action_dim", "alpha",
            "size", "position", "max_priority", "storage", "observations",
            "next_observations", "actions", "rewards", "dones",
            "next_action_masks", "next_action_penalties", "priorities",
        }
        if not isinstance(state, Mapping):
            raise TypeError("replay state must be a mapping")
        missing, extra = required - set(state), set(state) - required
        if missing or extra:
            raise ValueError(f"invalid replay state keys: missing={sorted(missing)}, extra={sorted(extra)}")
        version = _require_integer("replay version", state["version"])
        if version != self._STATE_VERSION:
            raise ValueError(f"unsupported replay state version: {state['version']!r}")
        saved_capacity = _require_integer("replay capacity", state["capacity"])
        saved_observation_dim = _require_integer(
            "replay observation_dim", state["observation_dim"]
        )
        saved_action_dim = _require_integer("replay action_dim", state["action_dim"])
        saved_alpha = float(state["alpha"])
        if saved_capacity != self.capacity:
            raise ValueError(f"replay capacity mismatch: saved={saved_capacity}, current={self.capacity}")
        if saved_observation_dim != self.observation_dim:
            raise ValueError(
                f"replay observation_dim mismatch: saved={saved_observation_dim}, "
                f"current={self.observation_dim}"
            )
        if saved_action_dim != self.action_dim:
            raise ValueError(f"replay action_dim mismatch: saved={saved_action_dim}, current={self.action_dim}")
        if not np.isclose(saved_alpha, self.alpha, rtol=0.0, atol=1e-12):
            raise ValueError(f"replay alpha mismatch: saved={saved_alpha}, current={self.alpha}")

        size = _require_integer("replay size", state["size"])
        position = _require_integer("replay position", state["position"])
        if not 0 <= size <= self.capacity:
            raise ValueError(f"invalid replay size {size} for capacity {self.capacity}")
        storage = state["storage"]
        expected_storage = "full" if size == self.capacity else "prefix"
        if storage != expected_storage:
            raise ValueError(f"replay storage must be {expected_storage!r} for size {size}")
        if size < self.capacity and position != size:
            raise ValueError(f"unfilled replay position must equal size ({size}), got {position}")
        if size == self.capacity and not 0 <= position < self.capacity:
            raise ValueError(f"full replay position must be in [0, {self.capacity}), got {position}")
        stored = self.capacity if size == self.capacity else size

        def array(name: str, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
            value = state[name]
            if not isinstance(value, np.ndarray):
                raise TypeError(f"replay {name} must be a numpy array")
            if value.shape != shape:
                raise ValueError(f"replay {name} shape must be {shape}, got {value.shape}")
            if value.dtype != dtype:
                raise TypeError(f"replay {name} dtype must be {dtype}, got {value.dtype}")
            return value

        observations = array("observations", (stored, self.observation_dim), np.dtype(np.float32))
        next_observations = array(
            "next_observations", (stored, self.observation_dim), np.dtype(np.float32)
        )
        actions = array("actions", (stored,), np.dtype(np.int64))
        rewards = array("rewards", (stored,), np.dtype(np.float32))
        dones = array("dones", (stored,), np.dtype(np.float32))
        masks = array("next_action_masks", (stored, self.action_dim), np.dtype(bool))
        penalties = array(
            "next_action_penalties", (stored, self.action_dim), np.dtype(np.float32)
        )
        priorities = array("priorities", (stored,), np.dtype(np.float32))
        if not np.all(np.isfinite(observations)) or not np.all(np.isfinite(next_observations)):
            raise ValueError("replay observations must be finite")
        if not np.all(np.isfinite(rewards)):
            raise ValueError("replay rewards must be finite")
        if np.any((actions < 0) | (actions >= self.action_dim)):
            raise ValueError("replay actions contain an out-of-range action")
        if np.any((dones != 0.0) & (dones != 1.0)):
            raise ValueError("replay dones must contain only 0 or 1")
        if stored and np.any(~masks.any(axis=1)):
            raise ValueError("every replay next_action_mask must allow at least one action")
        if not np.all(np.isfinite(penalties)) or np.any(penalties < 0.0):
            raise ValueError("replay next_action_penalties must be finite and non-negative")
        if not np.all(np.isfinite(priorities)) or np.any(priorities <= 0.0):
            raise ValueError("valid replay priorities must be finite and positive")
        max_priority = float(state["max_priority"])
        if not np.isfinite(max_priority) or max_priority <= 0.0:
            raise ValueError("replay max_priority must be finite and positive")
        if stored and max_priority < float(priorities.max()):
            raise ValueError("replay max_priority cannot be below a stored priority")

        # Assign the valid prefix before clearing the unused tail.  State dicts
        # intentionally expose read-only views for low-memory serialization;
        # this ordering also makes ``buffer.load_state_dict(buffer.state_dict())``
        # safe instead of clearing its own source views first.
        self.observations[:stored] = observations
        self.next_observations[:stored] = next_observations
        self.actions[:stored] = actions
        self.rewards[:stored] = rewards
        self.dones[:stored] = dones
        self.next_action_masks[:stored] = masks
        self.next_action_penalties[:stored] = penalties
        self.priorities[:stored] = priorities
        self.observations[stored:] = 0.0
        self.next_observations[stored:] = 0.0
        self.actions[stored:] = 0
        self.rewards[stored:] = 0.0
        self.dones[stored:] = 0.0
        self.next_action_masks[stored:] = True
        self.next_action_penalties[stored:] = 0.0
        self.priorities[stored:] = 0.0
        self.size, self.position, self.max_priority = size, position, max_priority


@dataclass
class Transition:
    observation: np.ndarray
    action: int
    reward: float
    next_observation: np.ndarray
    done: bool
    next_action_mask: np.ndarray | None = None
    next_action_penalty: np.ndarray | None = None

    def state_dict(self) -> dict[str, Any]:
        action = _require_integer("transition action", self.action)
        if not isinstance(self.done, (bool, np.bool_)):
            raise TypeError("transition done must be a boolean")
        return {
            "observation": np.asarray(self.observation).copy(),
            "action": action,
            "reward": float(self.reward),
            "next_observation": np.asarray(self.next_observation).copy(),
            "done": bool(self.done),
            "next_action_mask": (
                None if self.next_action_mask is None else np.asarray(self.next_action_mask, dtype=bool).copy()
            ),
            "next_action_penalty": (
                None if self.next_action_penalty is None
                else np.asarray(self.next_action_penalty, dtype=np.float32).copy()
            ),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "Transition":
        required = {
            "observation", "action", "reward", "next_observation", "done",
            "next_action_mask", "next_action_penalty",
        }
        if not isinstance(state, Mapping):
            raise TypeError("transition state must be a mapping")
        missing, extra = required - set(state), set(state) - required
        if missing or extra:
            raise ValueError(
                f"invalid transition state keys: missing={sorted(missing)}, extra={sorted(extra)}"
            )
        observation = np.asarray(state["observation"]).copy()
        next_observation = np.asarray(state["next_observation"]).copy()
        if observation.shape != next_observation.shape:
            raise ValueError("transition observation shapes do not match")
        if not np.all(np.isfinite(observation)) or not np.all(np.isfinite(next_observation)):
            raise ValueError("transition observations must be finite")
        action = _require_integer("transition action", state["action"])
        if not isinstance(state["done"], (bool, np.bool_)):
            raise TypeError("transition done must be a boolean")
        reward = float(state["reward"])
        if not np.isfinite(reward):
            raise ValueError("transition reward must be finite")
        mask_value, penalty_value = state["next_action_mask"], state["next_action_penalty"]
        mask = None if mask_value is None else np.asarray(mask_value, dtype=bool).copy()
        penalty = None if penalty_value is None else np.asarray(penalty_value, dtype=np.float32).copy()
        if mask is not None and (mask.ndim != 1 or not mask.any()):
            raise ValueError("transition next_action_mask must be a vector that allows an action")
        if penalty is not None and (
            penalty.ndim != 1 or not np.all(np.isfinite(penalty)) or np.any(penalty < 0.0)
        ):
            raise ValueError("transition next_action_penalty must be a finite non-negative vector")
        if mask is not None and penalty is not None and mask.shape != penalty.shape:
            raise ValueError("transition action mask and penalty shapes do not match")
        return cls(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=bool(state["done"]),
            next_action_mask=mask,
            next_action_penalty=penalty,
        )


class NStepBuffer:
    _STATE_VERSION = 1

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
                          chosen[-1].next_observation, chosen[-1].done,
                          chosen[-1].next_action_mask, chosen[-1].next_action_penalty)

    def state_dict(self) -> dict[str, Any]:
        return {
            "version": self._STATE_VERSION,
            "steps": self.steps,
            "gamma": self.gamma,
            "items": [item.state_dict() for item in self.items],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        required = {"version", "steps", "gamma", "items"}
        if not isinstance(state, Mapping):
            raise TypeError("n-step state must be a mapping")
        missing, extra = required - set(state), set(state) - required
        if missing or extra:
            raise ValueError(f"invalid n-step state keys: missing={sorted(missing)}, extra={sorted(extra)}")
        version = _require_integer("n-step version", state["version"])
        if version != self._STATE_VERSION:
            raise ValueError(f"unsupported n-step state version: {state['version']!r}")
        saved_steps = _require_integer("n-step steps", state["steps"])
        saved_gamma = float(state["gamma"])
        if saved_steps != self.steps:
            raise ValueError(f"n-step length mismatch: saved={saved_steps}, current={self.steps}")
        if not np.isclose(saved_gamma, self.gamma, rtol=0.0, atol=1e-12):
            raise ValueError(f"n-step gamma mismatch: saved={saved_gamma}, current={self.gamma}")
        raw_items = state["items"]
        if not isinstance(raw_items, (list, tuple)):
            raise TypeError("n-step items must be a list or tuple")
        if len(raw_items) >= self.steps:
            raise ValueError(f"n-step queue must contain fewer than {self.steps} items")
        items = [Transition.from_state_dict(item) for item in raw_items]
        if any(item.done for item in items):
            raise ValueError("n-step queue cannot retain a terminal transition")
        self.items = deque(items)

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "NStepBuffer":
        if not isinstance(state, Mapping) or "steps" not in state or "gamma" not in state:
            raise ValueError("n-step state is missing steps or gamma")
        buffer = cls(_require_integer("n-step steps", state["steps"]), float(state["gamma"]))
        buffer.load_state_dict(state)
        return buffer
