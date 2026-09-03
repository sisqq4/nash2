from __future__ import annotations

import json
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any, TypeVar

from ..env.types import EnvironmentConfig

T = TypeVar("T")
BLUE_MISSION_DURATION_S = 200.0
BLUE_INITIAL_ALTITUDE_RANGE_M = (9000.0, 11000.0)


def _replace_dataclass(instance: T, values: dict[str, Any], path: str) -> T:
    known = {field.name for field in fields(instance)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown configuration keys at {path}: {unknown}")
    changes: dict[str, Any] = {}
    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise ValueError(f"{path}.{name} must be a JSON object")
            changes[name] = _replace_dataclass(current, value, f"{path}.{name}")
        else:
            changes[name] = value
    return replace(instance, **changes)


def load_environment_config(path: str | None) -> EnvironmentConfig:
    """Load validated, nested overrides while retaining every v1 default."""
    config = EnvironmentConfig()
    if path is not None:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("environment configuration root must be a JSON object")
        config = _replace_dataclass(config, raw, "environment")
    config.validate()
    return config


def configure_blue_mission_duration(
    config: EnvironmentConfig, duration_s: float = BLUE_MISSION_DURATION_S
) -> EnvironmentConfig:
    """Apply the shared blue train/evaluation mission and guidance horizon."""
    if duration_s <= config.policy_entry_time_s:
        raise ValueError("blue mission duration must exceed the post-boost policy entry time")
    configured = replace(
        config,
        max_steps=int(round(duration_s / config.time_step_s)),
        missile=replace(config.missile, max_guidance_time_s=duration_s),
        scenario=replace(config.scenario, blue_altitude_range_m=BLUE_INITIAL_ALTITUDE_RANGE_M),
    )
    configured.validate()
    return configured
