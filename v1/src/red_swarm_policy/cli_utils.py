from __future__ import annotations

import math


def parse_float_sequence(
    value: str | tuple[float, ...] | list[float],
    name: str,
    *,
    minimum_length: int = 1,
) -> tuple[float, ...]:
    if isinstance(value, str):
        items = [float(part.strip()) for part in value.split(",") if part.strip()]
    else:
        items = [float(part) for part in value]
    if len(items) < minimum_length or not all(math.isfinite(item) for item in items):
        raise ValueError(f"{name} must contain at least {minimum_length} finite numbers")
    return tuple(items)


def parse_float_pair(
    value: str | tuple[float, float] | list[float],
    name: str,
) -> tuple[float, float]:
    items = parse_float_sequence(value, name, minimum_length=2)
    if len(items) != 2:
        raise ValueError(f"{name} must contain two finite numbers")
    return (items[0], items[1])


def parse_float_range(
    value: str | tuple[float, float] | list[float],
    name: str,
    *,
    positive: bool = True,
) -> tuple[float, float]:
    lower, upper = parse_float_pair(value, name)
    if lower > upper or (positive and lower <= 0.0):
        raise ValueError(f"{name} must contain an ascending range")
    return (lower, upper)
