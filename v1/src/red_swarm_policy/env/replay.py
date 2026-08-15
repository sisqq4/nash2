from __future__ import annotations

from typing import Iterable

from .types import ReplayTransition


class ReplayBuffer:
    def __init__(self) -> None:
        self._items: list[ReplayTransition] = []

    def append(self, transition: ReplayTransition) -> None:
        self._items.append(transition)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterable[ReplayTransition]:
        return iter(self._items)

    def __getitem__(self, index: int) -> ReplayTransition:
        return self._items[index]
