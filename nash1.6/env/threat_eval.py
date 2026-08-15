from __future__ import annotations

import math as m
from dataclasses import dataclass

import numpy as np


def cal_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_heading(
    plane_pos: np.ndarray,
    missile_pos: np.ndarray,
    missile_vel: np.ndarray,
) -> float:
    """Horizontal (heading) attack angle between missile velocity and LOS."""
    rel = plane_pos - missile_pos
    rel_h = np.array([rel[0], rel[1], 0.0])
    vel_h = np.array([missile_vel[0], missile_vel[1], 0.0])

    rel_norm = np.linalg.norm(rel_h)
    vel_norm = np.linalg.norm(vel_h)
    if rel_norm < 1e-8 or vel_norm < 1e-8:
        return 0.0

    cos_val = float(np.clip(np.dot(rel_h, vel_h) / (rel_norm * vel_norm), -1.0, 1.0))
    return float(m.acos(cos_val))


def compute_pitch(
    plane_pos: np.ndarray,
    missile_pos: np.ndarray,
    missile_vel: np.ndarray,
) -> float:
    """Vertical (pitch) attack angle between missile velocity and LOS."""
    rel = plane_pos - missile_pos
    rel_h_norm = np.linalg.norm(rel[:2])
    vel_h_norm = np.linalg.norm(missile_vel[:2])

    pitch_rel = float(m.atan2(rel[2], rel_h_norm if rel_h_norm > 1e-8 else 1e-8))
    pitch_vel = float(m.atan2(missile_vel[2], vel_h_norm if vel_h_norm > 1e-8 else 1e-8))
    return pitch_rel - pitch_vel


class AngleThreat:
    def __init__(self, heading_max: float, pitch_max: float, omega: float) -> None:
        self.heading_max = heading_max
        self.pitch_max = pitch_max
        self.omega = omega

    def cal_ta(self, heading: float, pitch: float) -> float:
        ta = m.exp(
            max(
                1.0
                - self.omega * (abs(heading) / self.heading_max)
                - (1.0 - self.omega) * (abs(pitch) / self.pitch_max),
                0.0,
            )
        )
        return float(ta / m.e)


class DistanceThreat:
    def __init__(self, kd: float, sigma: float, dist_max: float) -> None:
        self.dist_max = dist_max
        self.kd = kd
        self.sigma = sigma

    def cal_td(self, dist: float) -> float:
        td = self.kd * (1.0 / (abs(dist) + self.sigma) - 1.0 / self.dist_max) ** 2
        td *= (abs(dist) + self.sigma) ** 2
        return float(td)


class SpeedThreat:
    def __init__(self, ve: float, v: float) -> None:
        self.ve = ve
        self.v = v

    def cal_ts(self) -> float:
        adv_speed = self.ve / max(self.v, 1e-8)
        if adv_speed <= 0.6:
            return 0.1
        if adv_speed > 1.5:
            return 1.0
        return float(adv_speed - 0.5)

    def set_v(self, ve: float, v: float) -> None:
        self.ve = ve
        self.v = v


def _calculate_mean_weights(criteria: np.ndarray) -> np.ndarray:
    criteria = np.asarray(criteria, dtype=float)
    if criteria.ndim != 2 or criteria.shape[0] != criteria.shape[1]:
        raise ValueError("criteria must be a square matrix")
    geometric_mean = np.prod(criteria, axis=1) ** (1.0 / criteria.shape[0])
    total = float(np.sum(geometric_mean))
    return geometric_mean / total if total > 0 else np.ones(criteria.shape[0]) / criteria.shape[0]


@dataclass
class ThreatParams:
    heading_max: float
    pitch_max: float
    omega: float
    dist_max: float
    kd: float
    sigma: float
    criteria: np.ndarray


class ThreatEvaluator:
    def __init__(self, params: ThreatParams) -> None:
        self.angle_threat = AngleThreat(params.heading_max, params.pitch_max, params.omega)
        self.distance_threat = DistanceThreat(params.kd, params.sigma, params.dist_max)
        self.speed_threat = SpeedThreat(1.0, 1.0)
        self.weights = _calculate_mean_weights(params.criteria)

    def evaluate(
        self,
        plane_pos: np.ndarray,
        plane_vel: np.ndarray,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
    ) -> float:
        heading = compute_heading(plane_pos, missile_pos, missile_vel)
        pitch = compute_pitch(plane_pos, missile_pos, missile_vel)
        treat_a = self.angle_threat.cal_ta(heading, pitch)

        dist = cal_distance(plane_pos, missile_pos)
        treat_d = self.distance_threat.cal_td(dist)

        v_m = float(np.linalg.norm(missile_vel))
        v_p = float(np.linalg.norm(plane_vel))
        self.speed_threat.set_v(v_m, v_p)
        treat_v = self.speed_threat.cal_ts()

        treat_tot = np.array([treat_a, treat_v, treat_d], dtype=float)
        return float(np.dot(treat_tot, self.weights))
