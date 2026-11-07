from __future__ import annotations

import math

import numpy as np


BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G = np.array(
    [
        [0.0, 1.0, 0.0],
        [4.5, 1.0, 0.0],
        [9.0, 1.0, 0.0],
        [-4.5, 1.0, 0.0],
        [-9.0, 1.0, 0.0],
        [0.0, 3.0, math.acos(1.0 / 3.0)],
        [0.0, 6.0, math.acos(1.0 / 6.0)],
        [0.0, 9.0, math.acos(1.0 / 9.0)],
        [0.0, 3.0, -math.acos(1.0 / 3.0)],
        [0.0, 6.0, -math.acos(1.0 / 6.0)],
        [0.0, 9.0, -math.acos(1.0 / 9.0)],
        [0.0, 3.0, 0.0],
        [0.0, 6.0, 0.0],
        [0.0, 9.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, -3.0, 0.0],
        [0.0, -6.0, 0.0],
        [0.0, 3.0, 0.25 * math.pi],
        [0.0, 6.0, 0.25 * math.pi],
        [0.0, 9.0, 0.25 * math.pi],
        [0.0, 3.0, -0.25 * math.pi],
        [0.0, 6.0, -0.25 * math.pi],
        [0.0, 9.0, -0.25 * math.pi],
        [0.0, -3.0, -0.25 * math.pi],
        [0.0, -6.0, -0.25 * math.pi],
        [0.0, -9.0, -0.25 * math.pi],
        [0.0, -3.0, 0.25 * math.pi],
        [0.0, -6.0, 0.25 * math.pi],
        [0.0, -9.0, 0.25 * math.pi],
    ],
    dtype=np.float64,
)


def blue_aircraft_load_commands_body_g(action_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(action_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= len(BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G)):
        raise ValueError("blue action index must be in [0, 28]")
    return BLUE_AIRCRAFT_LOAD_COMMANDS_BODY_G[indices]
