"""Behavior-tree blue agent used as a non-RL baseline policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PlaneSnapshot:
    """Minimal blue aircraft state required by the BT policy."""

    pos: np.ndarray
    vel: np.ndarray
    roll_rad: float

    @property
    def z(self) -> float:
        return float(self.pos[2])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.vel))

    @property
    def yaw_rad(self) -> float:
        return float(np.arctan2(self.vel[1], self.vel[0]))

    @property
    def pitch_rad(self) -> float:
        v_h = float(np.linalg.norm(self.vel[:2]))
        return float(np.arctan2(self.vel[2], max(v_h, 1e-9)))

    def get_velocity_vector(self) -> np.ndarray:
        return self.vel


@dataclass
class MissileSnapshot:
    """Minimal hostile missile state required by the BT policy."""

    pos: np.ndarray
    team: int
    is_active: bool


class BlueBTAgent:
    """蓝方行为树智能体（适配当前离散动作空间）。"""

    def __init__(self, uid: str, team: int, difficulty: float = 0.0):
        self.uid = uid
        self.team = team
        self.difficulty = difficulty
        self.cruise_velocity = None
        self.state = "CRUISE"
        self.state_timer = 0
        self.maneuver_sign = 1

    def reset(self, initial_state: np.ndarray) -> None:
        self.cruise_velocity = initial_state[3:6].copy()
        speed = np.linalg.norm(self.cruise_velocity)
        if speed > 1e-3:
            self.cruise_velocity /= speed
        else:
            self.cruise_velocity = np.array([1.0, 0.0, 0.0], dtype=float)
        self.state = "CRUISE"
        self.state_timer = 0
        self.maneuver_sign = np.random.choice([-1, 1])

    def get_action(self, my_plane: PlaneSnapshot, missiles: List[MissileSnapshot], enemies: List) -> int:
        del enemies

        safe_floor = 2.0
        _, _, vz = my_plane.get_velocity_vector()
        pred_z = my_plane.z + vz * 3.0

        if (my_plane.z < safe_floor and vz < -0.01) or (pred_z < 0.8) or (my_plane.z < 1.0):
            return 3

        if my_plane.speed < 0.18:
            if my_plane.z > 3.0:
                return 4
            return 1

        threat = self._assess_threats(my_plane, missiles)
        self._update_state(threat)
        return self._execute_state(my_plane, threat)

    def _assess_threats(self, my_plane: PlaneSnapshot, missiles: List[MissileSnapshot]) -> Dict:
        if not missiles:
            return {"level": 0, "missile": None}

        danger = []
        for missile in missiles:
            if not missile.is_active:
                continue
            if missile.team == self.team:
                continue
            dist = np.linalg.norm(missile.pos - my_plane.pos)
            if dist < 60.0:
                danger.append((dist, missile))

        if not danger:
            return {"level": 0, "missile": None}

        danger.sort(key=lambda x: x[0])
        nearest_dist, nearest_missile = danger[0]

        if nearest_dist < 12.0:
            level = 3
        elif nearest_dist < 40.0:
            level = 2
        else:
            level = 1
        return {"level": level, "missile": nearest_missile}

    def _update_state(self, threat: Dict) -> None:
        if self.state_timer > 0:
            self.state_timer -= 1
            return

        level = threat["level"]
        if level == 3:
            if self.state != "BREAK":
                self.state = "BREAK"
                self.state_timer = 30
                self.maneuver_sign = np.random.choice([-1, 1])
        elif level == 2:
            if self.state != "BEAM":
                self.state = "BEAM"
                self.state_timer = 50
        else:
            if self.state != "CRUISE":
                self.state = "CRUISE"
                self.state_timer = 0

    def _execute_state(self, plane: PlaneSnapshot, threat: Dict) -> int:
        if self.state == "CRUISE":
            return self._maintain_vector(plane)

        if self.state == "BEAM":
            missile = threat["missile"]
            if missile is None:
                return self._maintain_vector(plane)
            vec_to_missile = missile.pos - plane.pos
            angle_to_missile = np.arctan2(vec_to_missile[1], vec_to_missile[0])
            my_heading = plane.yaw_rad
            target_heading_1 = angle_to_missile + np.pi / 2
            target_heading_2 = angle_to_missile - np.pi / 2
            diff1 = abs(self._angle_diff(target_heading_1, my_heading))
            diff2 = abs(self._angle_diff(target_heading_2, my_heading))
            target_heading = target_heading_1 if diff1 < diff2 else target_heading_2
            diff = self._angle_diff(target_heading, my_heading)
            if abs(diff) > 0.1:
                return 9 if diff > 0 else 10
            return 1

        if self.state == "BREAK":
            if plane.z > 6.0:
                return 7 if self.maneuver_sign > 0 else 8
            if np.random.random() < 0.5:
                return 9 if self.maneuver_sign > 0 else 10
            return 5 if self.maneuver_sign > 0 else 6

        return 0

    def _maintain_vector(self, plane: PlaneSnapshot) -> int:
        if self.cruise_velocity is None:
            return 0

        if plane.speed < 0.25:
            return 1

        target_dir = self.cruise_velocity
        target_pitch = np.arcsin(target_dir[2])
        pitch_err = target_pitch - plane.pitch_rad
        if abs(pitch_err) > 0.08:
            return 3 if pitch_err > 0 else 4

        target_yaw = np.arctan2(target_dir[1], target_dir[0])
        yaw_err = self._angle_diff(target_yaw, plane.yaw_rad)
        if abs(yaw_err) > 0.08:
            return 9 if yaw_err > 0 else 10

        if abs(plane.roll_rad) > 0.1:
            return 0
        return 0

    def _angle_diff(self, target: float, current: float) -> float:
        d = target - current
        while d > np.pi:
            d -= 2 * np.pi
        while d < -np.pi:
            d += 2 * np.pi
        return float(d)