from __future__ import annotations

import math

import numpy as np

from .math_utils import EAST_AXIS, NORTH_AXIS, UP_AXIS, unit
from .types import AircraftConfig, EngagementState, MissileConfig, SCENARIO_STYLES, ScenarioConfig, ScenarioStyle, ThreeDoFState


class ScenarioGenerator:
    def __init__(self, config: ScenarioConfig, missile: MissileConfig, aircraft: AircraftConfig) -> None:
        config.validate()
        self.config = config
        self.missile = missile
        self.aircraft = aircraft

    def generate(
        self,
        seed: int | None = None,
        style: ScenarioStyle | None = None,
        red_count: int | None = None,
        blue_count: int | None = None,
    ) -> EngagementState:
        rng = np.random.default_rng(seed)
        scenario_style: ScenarioStyle = style or rng.choice(SCENARIO_STYLES).item()
        n_red = int(self.config.red_count if red_count is None else red_count)
        n_blue = int(self.config.blue_count if blue_count is None else blue_count)
        if n_red <= 0 or n_blue <= 0:
            raise ValueError("red_count and blue_count must be positive")
        blue = self._build_blue_targets(rng, n_blue)
        red = self._build_red_nodes(rng, n_red)
        return EngagementState(
            red=red,
            blue=blue,
            style=scenario_style,
            parameters={
                "seed": seed,
                "style": scenario_style,
                "red_count": n_red,
                "blue_count": n_blue,
                "red_launch_mach_range": self.config.red_launch_mach_range,
                "red_altitude_range_m": self.config.red_altitude_range_m,
                "blue_speed_range_mps": self.config.blue_speed_range_mps,
                "blue_altitude_range_m": self.config.blue_altitude_range_m,
                "blue_cluster_center_ne_m": self.config.blue_cluster_center_ne_m,
                "blue_cluster_radius_m": self.config.blue_cluster_radius_m,
                "red_cluster_radius_range_m": self.config.red_cluster_radius_range_m,
                "red_sector_center_azimuth_deg": self.config.red_sector_center_azimuth_deg,
                "red_sector_width_deg": self.config.red_sector_width_deg,
                "red_heading_bias_max_deg": self.config.red_heading_bias_max_deg,
                "position_perturb_m": self.config.position_perturb_m,
                "velocity_perturb_mps": self.config.velocity_perturb_mps,
            },
        )

    def _build_blue_targets(self, rng: np.random.Generator, n_blue: int) -> list[ThreeDoFState]:
        blue: list[ThreeDoFState] = []
        center_north_m, center_east_m = self.config.blue_cluster_center_ne_m
        for _ in range(n_blue):
            radius = self.config.blue_cluster_radius_m * math.sqrt(rng.uniform())
            azimuth = rng.uniform(-math.pi, math.pi)
            altitude = rng.uniform(*self.config.blue_altitude_range_m)
            position = np.array(
                [
                    center_north_m + radius * math.cos(azimuth),
                    altitude,
                    center_east_m + radius * math.sin(azimuth),
                ],
                dtype=np.float64,
            )
            if self.config.position_perturb_m > 0.0:
                position += rng.normal(0.0, self.config.position_perturb_m, 3)
                position = self._clip_blue_position(position)
            heading = math.radians(rng.uniform(*self.config.blue_heading_range_deg))
            speed = rng.uniform(*self.config.blue_speed_range_mps)
            velocity = speed * np.array([math.cos(heading), 0.0, math.sin(heading)], dtype=np.float64)
            if self.config.velocity_perturb_mps > 0.0:
                # Blue aircraft always start in level flight.  Random velocity
                # perturbations may change horizontal speed and heading, but
                # must never introduce an initial climb or dive component.
                horizontal_noise = rng.normal(0.0, self.config.velocity_perturb_mps, 2)
                velocity[[NORTH_AXIS, EAST_AXIS]] += horizontal_noise
                perturbed_speed = float(np.linalg.norm(velocity))
                bounded_speed = float(np.clip(perturbed_speed, *self.config.blue_speed_range_mps))
                velocity = unit(velocity, np.array([math.cos(heading), 0.0, math.sin(heading)])) * bounded_speed
            blue.append(
                ThreeDoFState(
                    position_m=position,
                    velocity_mps=velocity,
                    mass_kg=1.0,
                    bank_angle_rad=0.0,
                )
            )
        return blue

    def _build_red_nodes(
        self,
        rng: np.random.Generator,
        n_red: int,
    ) -> list[ThreeDoFState]:
        red: list[ThreeDoFState] = []
        center_north_m, center_east_m = self.config.blue_cluster_center_ne_m
        center_altitude_m = 0.5 * sum(self.config.blue_altitude_range_m)
        blue_center = np.array([center_north_m, center_altitude_m, center_east_m], dtype=np.float64)
        minimum_radius_m, maximum_radius_m = self.config.red_cluster_radius_range_m
        sector_center_rad = math.radians(self.config.red_sector_center_azimuth_deg)
        sector_half_width_rad = 0.5 * math.radians(self.config.red_sector_width_deg)
        heading_bias_max_rad = math.radians(self.config.red_heading_bias_max_deg)
        for _ in range(n_red):
            altitude = rng.uniform(*self.config.red_altitude_range_m)
            speed = rng.uniform(*self.config.red_launch_mach_range) * self.config.speed_of_sound_mps
            radius = math.sqrt(rng.uniform(minimum_radius_m**2, maximum_radius_m**2))
            azimuth = sector_center_rad + rng.uniform(-sector_half_width_rad, sector_half_width_rad)
            position = np.array(
                [
                    center_north_m + radius * math.cos(azimuth),
                    altitude,
                    center_east_m + radius * math.sin(azimuth),
                ],
                dtype=np.float64,
            )
            if self.config.position_perturb_m > 0.0:
                position += rng.normal(0.0, self.config.position_perturb_m, 3)
                position = self._clip_red_position(position)
            center_direction = unit(blue_center - position, np.array([1.0, 0.0, 0.0]))
            heading_bias_rad = rng.uniform(-heading_bias_max_rad, heading_bias_max_rad)
            direction = self._rotate_about_up(center_direction, heading_bias_rad)
            velocity = speed * direction
            if self.config.velocity_perturb_mps > 0.0:
                velocity += rng.normal(0.0, self.config.velocity_perturb_mps, 3)
                direction = self._limit_direction_offset(
                    center_direction,
                    unit(velocity, direction),
                    heading_bias_max_rad,
                )
                minimum_speed_mps = self.config.red_launch_mach_range[0] * self.config.speed_of_sound_mps
                maximum_speed_mps = self.config.red_launch_mach_range[1] * self.config.speed_of_sound_mps
                speed = float(np.clip(np.linalg.norm(velocity), minimum_speed_mps, maximum_speed_mps))
                velocity = speed * direction
            red.append(
                ThreeDoFState(
                    position_m=position,
                    velocity_mps=velocity,
                    mass_kg=self.missile.full_mass_kg,
                    fuel_mass_kg=self.missile.propellant_mass_kg,
                )
            )
        return red

    def _clip_blue_position(self, position_m: np.ndarray) -> np.ndarray:
        center_north_m, center_east_m = self.config.blue_cluster_center_ne_m
        relative_ne = np.array(
            [position_m[NORTH_AXIS] - center_north_m, position_m[EAST_AXIS] - center_east_m],
            dtype=np.float64,
        )
        radius = float(np.linalg.norm(relative_ne))
        if radius > self.config.blue_cluster_radius_m:
            relative_ne *= self.config.blue_cluster_radius_m / radius
        position_m[NORTH_AXIS] = center_north_m + relative_ne[0]
        position_m[EAST_AXIS] = center_east_m + relative_ne[1]
        position_m[UP_AXIS] = float(np.clip(position_m[UP_AXIS], *self.config.blue_altitude_range_m))
        return position_m

    def _clip_red_position(self, position_m: np.ndarray) -> np.ndarray:
        center_north_m, center_east_m = self.config.blue_cluster_center_ne_m
        relative_north_m = position_m[NORTH_AXIS] - center_north_m
        relative_east_m = position_m[EAST_AXIS] - center_east_m
        radius = float(np.clip(math.hypot(relative_north_m, relative_east_m), *self.config.red_cluster_radius_range_m))
        azimuth_deg = math.degrees(math.atan2(relative_east_m, relative_north_m))
        offset_deg = self._wrapped_angle_deg(azimuth_deg - self.config.red_sector_center_azimuth_deg)
        half_width_deg = 0.5 * self.config.red_sector_width_deg
        bounded_azimuth_rad = math.radians(
            self.config.red_sector_center_azimuth_deg + float(np.clip(offset_deg, -half_width_deg, half_width_deg))
        )
        position_m[NORTH_AXIS] = center_north_m + radius * math.cos(bounded_azimuth_rad)
        position_m[EAST_AXIS] = center_east_m + radius * math.sin(bounded_azimuth_rad)
        position_m[UP_AXIS] = float(np.clip(position_m[UP_AXIS], *self.config.red_altitude_range_m))
        return position_m

    @staticmethod
    def _rotate_about_up(vector: np.ndarray, angle_rad: float) -> np.ndarray:
        cosine = math.cos(angle_rad)
        sine = math.sin(angle_rad)
        north, up, east = np.asarray(vector, dtype=np.float64)
        return unit(np.array([cosine * north - sine * east, up, sine * north + cosine * east], dtype=np.float64))

    @staticmethod
    def _limit_direction_offset(reference: np.ndarray, candidate: np.ndarray, maximum_angle_rad: float) -> np.ndarray:
        reference_unit = unit(reference, np.array([1.0, 0.0, 0.0]))
        candidate_unit = unit(candidate, reference_unit)
        angle = math.acos(float(np.clip(np.dot(reference_unit, candidate_unit), -1.0, 1.0)))
        if angle <= maximum_angle_rad:
            return candidate_unit
        tangent = candidate_unit - np.dot(candidate_unit, reference_unit) * reference_unit
        tangent = unit(tangent, np.array([0.0, 0.0, 1.0]))
        return math.cos(maximum_angle_rad) * reference_unit + math.sin(maximum_angle_rad) * tangent

    @staticmethod
    def _wrapped_angle_deg(angle_deg: float) -> float:
        return (angle_deg + 180.0) % 360.0 - 180.0
