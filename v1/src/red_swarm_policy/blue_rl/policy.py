from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DiscreteBluePolicy(Protocol):
    """Algorithm-neutral interface shared by Rainbow and a future discrete PPO."""

    def select_action(self, observation: np.ndarray, *, evaluation: bool = False) -> int: ...

    def save(self, path: str) -> None: ...


class PolicyRegistry:
    """Small policy factory registry; discrete PPO can be added without changing the env."""

    _factories: dict[str, Callable[..., DiscreteBluePolicy]] = {}

    @classmethod
    def register(cls, name: str, factory: Callable[..., DiscreteBluePolicy]) -> None:
        key = name.strip().lower()
        if not key:
            raise ValueError("policy name must not be empty")
        cls._factories[key] = factory

    @classmethod
    def create(cls, name: str, **kwargs: object) -> DiscreteBluePolicy:
        key = name.strip().lower()
        if key not in cls._factories:
            raise ValueError(f"unknown blue policy {name!r}; available: {sorted(cls._factories)}")
        return cls._factories[key](**kwargs)

    @classmethod
    def names(cls) -> tuple[str, ...]:
        return tuple(sorted(cls._factories))
