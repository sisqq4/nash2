from __future__ import annotations

import math

import numpy as np

from .math_utils import unit
from .types import MissileConfig, ThreeDoFState, los_kinematics


def seeker_boresight_angle_deg(missile: ThreeDoFState, target: ThreeDoFState) -> float:
    los = los_kinematics(missile, target)
    boresight = unit(missile.velocity_mps, np.array([1.0, 0.0, 0.0]))
    cosine = float(np.clip(np.dot(boresight, los.los_unit), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def seeker_fov_limit_deg(
    missile: ThreeDoFState,
    target_index: int,
    config: MissileConfig,
) -> float:
    estimate_target_index = (
        missile.target_estimate_target_index
        if missile.target_estimate_target_index >= 0
        else missile.current_target_index
    )
    locked_to_target = missile.seeker_locked and estimate_target_index == target_index
    return (
        config.seeker_tracking_fov_deg
        if locked_to_target
        else config.seeker_acquisition_fov_deg
    )


def seeker_target_visible(
    missile: ThreeDoFState,
    target: ThreeDoFState,
    target_index: int,
    config: MissileConfig,
    *,
    detection_range_m: float,
) -> bool:
    if not missile.alive or not target.alive:
        return False
    if los_kinematics(missile, target).range_m > detection_range_m:
        return False
    return seeker_boresight_angle_deg(missile, target) <= seeker_fov_limit_deg(
        missile,
        target_index,
        config,
    )
