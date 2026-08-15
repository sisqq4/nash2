from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal

import numpy as np

from ..policy.actor import AssignmentActorInputs, OverloadBiasActorInputs
from ..policy.critic import AssignmentCriticInputs, OverloadBiasCriticInputs
from .math_utils import EAST_AXIS, EPS, NORTH_AXIS, UP_AXIS, norm

ScenarioStyle = Literal[
    "one_to_one",
    "many_to_one",
    "many_to_many",
    "same_direction_pursuit",
    "multi_direction_encirclement",
]
PolicyStartMode = Literal["post_boost", "launch"]
GuidanceMode = Literal["boost", "locked", "lock_hold", "inertial"]
TerminationReason = Literal["none", "success", "red_failure", "timeout"]

SCENARIO_STYLES: tuple[ScenarioStyle, ...] = (
    "one_to_one",
    "many_to_one",
    "many_to_many",
    "same_direction_pursuit",
    "multi_direction_encirclement",
)


def _all_finite(*values: float) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def _is_integer(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))


@dataclass(frozen=True)
class SensorConfig:
    detection_range_m: float = 200000.0
    communication_delay_steps: int = 0
    position_noise_m: float = 0.0
    velocity_noise_mps: float = 0.0

    def validate(self) -> None:
        if not _all_finite(
            self.detection_range_m,
            self.position_noise_m,
            self.velocity_noise_mps,
        ):
            raise ValueError("sensor scalar parameters must be finite")
        if self.detection_range_m <= 0.0:
            raise ValueError("detection_range_m must be positive")
        if not _is_integer(self.communication_delay_steps) or self.communication_delay_steps < 0:
            raise ValueError("communication_delay_steps must be non-negative")
        if self.position_noise_m < 0.0 or self.velocity_noise_mps < 0.0:
            raise ValueError("sensor noise values must be non-negative")


@dataclass(frozen=True)
class AircraftConfig:
    max_load_factor_g: float = 9.0
    min_speed_mps: float = 100.0
    max_speed_mps: float = 600.0
    min_altitude_m: float = 8000.0
    max_altitude_m: float = 12000.0

    def validate(self) -> None:
        if not _all_finite(
            self.max_load_factor_g,
            self.min_speed_mps,
            self.max_speed_mps,
            self.min_altitude_m,
            self.max_altitude_m,
        ):
            raise ValueError("aircraft scalar parameters must be finite")
        if self.max_load_factor_g <= 0.0:
            raise ValueError("aircraft physical parameters are invalid")
        if self.min_speed_mps <= 0.0 or self.min_speed_mps >= self.max_speed_mps:
            raise ValueError("aircraft speed limits are invalid")
        if self.min_altitude_m < 0.0 or self.min_altitude_m >= self.max_altitude_m:
            raise ValueError("aircraft altitude limits are invalid")


@dataclass(frozen=True)
class MissileConfig:
    dry_mass_kg: float = 120.0
    propellant_mass_kg: float = 45.0
    boost_duration_s: float = 7.0
    boost_target_mach_number: float = 6.0
    reference_speed_of_sound_mps: float = 295.0
    boost_climb_angle_deg: float = 20.0
    boost_pitch_transition_s: float = 2.0
    boost_pitch_tracking_gain: float = 2.0
    reference_area_m2: float = 0.028
    drag_coefficient: float | None = None
    drag_mach_breakpoints: tuple[float, ...] = (0.0, 0.8, 0.95, 1.05, 1.2, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
    zero_lift_drag_coefficients: tuple[float, ...] = (0.10, 0.11, 0.18, 0.34, 0.30, 0.22, 0.19, 0.17, 0.16, 0.15, 0.15)
    induced_drag_factor: float = 0.08
    max_load_factor_g: float = 35.0
    max_guidance_bias_g: float = 5.0
    min_speed_mps: float = 100.0
    proportional_navigation_gain: float = 3.5
    max_guidance_time_s: float = 180.0
    seeker_acquisition_fov_deg: float = 35.0
    seeker_tracking_fov_deg: float = 60.0
    fov_break_hold_s: float = 0.75
    post_closest_growth_m: float = 600.0
    post_closest_recede_speed_mps: float = 40.0
    lethal_radius_m: float = 5.0
    escape_range_m: float = 70000.0

    @property
    def full_mass_kg(self) -> float:
        return self.dry_mass_kg + self.propellant_mass_kg

    @property
    def mass_flow_rate_kg_s(self) -> float:
        return 0.0 if self.boost_duration_s <= 0.0 else self.propellant_mass_kg / self.boost_duration_s

    @property
    def max_speed_mps(self) -> float:
        return self.boost_target_mach_number * self.reference_speed_of_sound_mps

    def validate(self) -> None:
        if not _all_finite(
            self.dry_mass_kg,
            self.propellant_mass_kg,
            self.boost_duration_s,
            self.boost_target_mach_number,
            self.reference_speed_of_sound_mps,
            self.boost_climb_angle_deg,
            self.boost_pitch_transition_s,
            self.boost_pitch_tracking_gain,
            self.reference_area_m2,
            self.induced_drag_factor,
            self.max_load_factor_g,
            self.max_guidance_bias_g,
            self.min_speed_mps,
            self.proportional_navigation_gain,
            self.max_guidance_time_s,
            self.seeker_acquisition_fov_deg,
            self.seeker_tracking_fov_deg,
            self.fov_break_hold_s,
            self.post_closest_growth_m,
            self.post_closest_recede_speed_mps,
            self.lethal_radius_m,
            self.escape_range_m,
        ):
            raise ValueError("missile scalar parameters must be finite")
        if self.drag_coefficient is not None and not math.isfinite(self.drag_coefficient):
            raise ValueError("drag_coefficient must be finite when provided")
        if self.dry_mass_kg <= 0.0 or self.propellant_mass_kg < 0.0:
            raise ValueError("missile masses are invalid")
        if self.boost_duration_s <= 0.0 or self.boost_target_mach_number <= 0.0 or self.reference_speed_of_sound_mps <= 0.0:
            raise ValueError("missile propulsion parameters are invalid")
        if (
            not 0.0 <= self.boost_climb_angle_deg < 90.0
            or self.boost_pitch_transition_s <= 0.0
            or self.boost_pitch_tracking_gain <= 0.0
        ):
            raise ValueError("missile boost-climb parameters are invalid")
        if (
            self.reference_area_m2 <= 0.0
            or (self.drag_coefficient is not None and self.drag_coefficient < 0.0)
            or self.induced_drag_factor < 0.0
        ):
            raise ValueError("missile aerodynamic parameters are invalid")
        if (
            len(self.drag_mach_breakpoints) < 2
            or len(self.drag_mach_breakpoints) != len(self.zero_lift_drag_coefficients)
            or not all(math.isfinite(value) for value in self.drag_mach_breakpoints)
            or not all(math.isfinite(value) for value in self.zero_lift_drag_coefficients)
            or self.drag_mach_breakpoints[0] < 0.0
            or any(
                right <= left
                for left, right in zip(self.drag_mach_breakpoints, self.drag_mach_breakpoints[1:])
            )
            or any(value <= 0.0 for value in self.zero_lift_drag_coefficients)
        ):
            raise ValueError("missile Mach drag table is invalid")
        if (
            self.max_load_factor_g <= 0.0
            or not 0.0 <= self.max_guidance_bias_g <= self.max_load_factor_g
            or self.proportional_navigation_gain <= 0.0
        ):
            raise ValueError("missile guidance parameters are invalid")
        if self.min_speed_mps <= 0.0 or self.max_guidance_time_s <= 0.0:
            raise ValueError("missile timing and launch parameters are invalid")
        if (
            self.seeker_acquisition_fov_deg <= 0.0
            or self.seeker_tracking_fov_deg <= 0.0
            or self.fov_break_hold_s < 0.0
        ):
            raise ValueError("missile seeker parameters are invalid")
        if self.post_closest_growth_m < 0.0 or self.post_closest_recede_speed_mps < 0.0:
            raise ValueError("missile post-closest miss parameters are invalid")
        if self.lethal_radius_m <= 0.0 or self.escape_range_m <= self.lethal_radius_m:
            raise ValueError("missile adjudication ranges are invalid")


@dataclass(frozen=True)
class ScenarioConfig:
    red_count: int = 24
    blue_count: int = 4
    max_missiles_per_target: int = 4
    red_launch_mach_range: tuple[float, float] = (0.6, 0.9)
    red_altitude_range_m: tuple[float, float] = (8000.0, 10000.0)
    blue_speed_range_mps: tuple[float, float] = (300.0, 400.0)
    blue_altitude_range_m: tuple[float, float] = (8000.0, 12000.0)
    speed_of_sound_mps: float = 295.0
    blue_cluster_center_ne_m: tuple[float, float] = (0.0, 0.0)
    blue_cluster_radius_m: float = 20000.0
    blue_heading_range_deg: tuple[float, float] = (-180.0, 180.0)
    red_cluster_radius_range_m: tuple[float, float] = (140000.0, 160000.0)
    red_sector_center_azimuth_deg: float = 180.0
    red_sector_width_deg: float = 60.0
    red_heading_bias_max_deg: float = 15.0
    position_perturb_m: float = 0.0
    velocity_perturb_mps: float = 0.0

    def validate(self) -> None:
        def validate_range(name: str, values: tuple[float, float], *, positive: bool) -> None:
            if len(values) != 2 or not all(math.isfinite(float(value)) for value in values):
                raise ValueError(f"{name} must contain two finite values")
            lower, upper = float(values[0]), float(values[1])
            if lower > upper or (positive and lower <= 0.0):
                raise ValueError(f"{name} is invalid")

        if not _is_integer(self.red_count) or not _is_integer(self.blue_count) or self.red_count <= 0 or self.blue_count <= 0:
            raise ValueError("red_count and blue_count must be positive")
        if not _is_integer(self.max_missiles_per_target) or self.max_missiles_per_target <= 0:
            raise ValueError("max_missiles_per_target must be a positive integer")
        validate_range("red_launch_mach_range", self.red_launch_mach_range, positive=True)
        validate_range("red_altitude_range_m", self.red_altitude_range_m, positive=True)
        validate_range("blue_speed_range_mps", self.blue_speed_range_mps, positive=True)
        validate_range("blue_altitude_range_m", self.blue_altitude_range_m, positive=True)
        validate_range("blue_heading_range_deg", self.blue_heading_range_deg, positive=False)
        validate_range("red_cluster_radius_range_m", self.red_cluster_radius_range_m, positive=True)
        if not math.isfinite(self.speed_of_sound_mps) or self.speed_of_sound_mps <= 0.0:
            raise ValueError("speed_of_sound_mps must be positive")
        if len(self.blue_cluster_center_ne_m) != 2 or not all(
            math.isfinite(float(value)) for value in self.blue_cluster_center_ne_m
        ):
            raise ValueError("blue_cluster_center_ne_m must contain finite north/east coordinates")
        if not math.isfinite(self.blue_cluster_radius_m) or self.blue_cluster_radius_m <= 0.0:
            raise ValueError("blue_cluster_radius_m must be positive")
        if not math.isfinite(self.red_sector_center_azimuth_deg):
            raise ValueError("red_sector_center_azimuth_deg must be finite")
        if not math.isfinite(self.red_sector_width_deg) or not 0.0 < self.red_sector_width_deg <= 360.0:
            raise ValueError("red_sector_width_deg must be in (0, 360]")
        if not math.isfinite(self.red_heading_bias_max_deg) or not 0.0 <= self.red_heading_bias_max_deg <= 180.0:
            raise ValueError("red_heading_bias_max_deg must be in [0, 180]")
        if not _all_finite(self.position_perturb_m, self.velocity_perturb_mps):
            raise ValueError("scenario perturbations must be finite")
        if self.position_perturb_m < 0.0 or self.velocity_perturb_mps < 0.0:
            raise ValueError("scenario perturbations must be non-negative")


@dataclass(frozen=True)
class RewardConfig:
    high_damage_weight: float = 512.0
    high_waste_weight: float = 64.0
    high_potential_weight: float = 512.0
    high_potential_gamma: float = 1.0
    high_time_penalty_per_s: float = 2.0
    high_time_margin_scale_s: float = 10.0
    terminal_success_reward: float = 0.0
    terminal_failure_penalty: float = 0.0
    terminal_timeout_penalty: float = 0.0
    low_damage_weight: float = 512.0
    low_potential_weight: float = 1.0
    low_potential_gamma: float = 1.0
    low_missile_failure_penalty: float = 64.0
    low_load_penalty: float = 0.0008
    low_smooth_penalty: float = 0.0002
    low_time_credit_mode: str = "none"
    low_time_weight: float = 2.0
    low_option_boundary_potential: str = "exempt"
    zem_reference_range_m: float = 1000.0
    zem_floor_range_m: float = 5.0
    zem_weight: float = 0.6
    seeker_lock_weight: float = 0.2
    smooth_bias_denominator: float = 8.0
    zem_time_gate_scale_s: float = 1.0
    assignment_min_energy_fraction: float = 0.05
    assignment_min_available_load_fraction: float = 0.05
    assignment_correlation_weight: float = 0.5
    assignment_correlation_angle_scale_deg: float = 15.0
    assignment_correlation_time_scale_s: float = 5.0

    def validate(self) -> None:
        weights = [
            self.high_damage_weight,
            self.high_waste_weight,
            self.high_potential_weight,
            self.high_time_penalty_per_s,
            self.terminal_success_reward,
            self.terminal_failure_penalty,
            self.terminal_timeout_penalty,
            self.low_damage_weight,
            self.low_potential_weight,
            self.low_missile_failure_penalty,
            self.low_load_penalty,
            self.low_smooth_penalty,
            self.low_time_weight,
            self.zem_weight,
            self.seeker_lock_weight,
        ]
        if not all(math.isfinite(weight) for weight in weights) or any(weight < 0.0 for weight in weights):
            raise ValueError("reward weights must be non-negative")
        if self.low_time_credit_mode not in {"none", "terminal_active_share"}:
            raise ValueError(
                "low_time_credit_mode must be none or terminal_active_share"
            )
        if self.low_option_boundary_potential not in {"exempt", "terminal_zero"}:
            raise ValueError(
                "low_option_boundary_potential must be exempt or terminal_zero"
            )
        if (
            self.low_time_credit_mode == "terminal_active_share"
            and not math.isclose(
                self.low_time_weight,
                self.high_time_penalty_per_s,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            )
        ):
            raise ValueError(
                "terminal_active_share low_time_weight must equal the mission time weight"
            )
        if not _all_finite(
            self.high_potential_gamma,
            self.low_potential_gamma,
            self.high_time_margin_scale_s,
            self.zem_reference_range_m,
            self.zem_floor_range_m,
            self.smooth_bias_denominator,
            self.zem_time_gate_scale_s,
            self.assignment_min_energy_fraction,
            self.assignment_min_available_load_fraction,
            self.assignment_correlation_weight,
            self.assignment_correlation_angle_scale_deg,
            self.assignment_correlation_time_scale_s,
        ):
            raise ValueError("reward scalar parameters must be finite")
        if self.high_potential_gamma != 1.0 or self.low_potential_gamma != 1.0:
            raise ValueError("strict mission J requires high_potential_gamma and low_potential_gamma to equal 1")
        if self.high_time_margin_scale_s <= 0.0:
            raise ValueError("high_time_margin_scale_s must be positive")
        if self.zem_floor_range_m <= 0.0 or self.zem_reference_range_m <= self.zem_floor_range_m:
            raise ValueError("zem_reference_range_m must exceed positive zem_floor_range_m")
        if self.zem_weight + self.seeker_lock_weight > 1.0 + 1.0e-9:
            raise ValueError("zem_weight and seeker_lock_weight must sum to at most 1")
        if self.smooth_bias_denominator <= 0.0:
            raise ValueError("smooth_bias_denominator must be positive")
        if self.zem_time_gate_scale_s <= 0.0:
            raise ValueError("zem_time_gate_scale_s must be positive")
        if not 0.0 <= self.assignment_min_energy_fraction < 1.0:
            raise ValueError("assignment_min_energy_fraction must be in [0, 1)")
        if not 0.0 <= self.assignment_min_available_load_fraction < 1.0:
            raise ValueError("assignment_min_available_load_fraction must be in [0, 1)")
        if not 0.0 <= self.assignment_correlation_weight <= 1.0:
            raise ValueError("assignment_correlation_weight must be in [0, 1]")
        if self.assignment_correlation_angle_scale_deg <= 0.0:
            raise ValueError("assignment_correlation_angle_scale_deg must be positive")
        if self.assignment_correlation_time_scale_s <= 0.0:
            raise ValueError("assignment_correlation_time_scale_s must be positive")

    def validate_lexicographic_priority(self, red_count: int, blue_count: int) -> None:
        """Reject scenario scales for which the configured mission weights lose priority."""
        control_weight = self.low_load_penalty + self.low_smooth_penalty
        # Full mission success is an independent, highest-priority event.  Its
        # bonus must therefore not consume the lower-priority budget used to
        # prove damage and ineffective-loss ordering.  Failure and timeout
        # remain lower-priority outcome adjustments and stay in that budget.
        lower_terminal_span = (
            self.terminal_failure_penalty + self.terminal_timeout_penalty
        )
        time_weight = max(self.high_time_penalty_per_s, self.low_time_weight)
        lower_after_damage = (
            self.high_waste_weight
            + time_weight
            + control_weight
            + lower_terminal_span
        )
        lower_after_waste = time_weight + control_weight + lower_terminal_span
        if self.high_damage_weight / max(blue_count, 1) <= lower_after_damage:
            raise ValueError(
                "reward weights do not preserve damage priority for the requested blue_count"
            )
        if self.high_waste_weight / max(red_count, 1) <= lower_after_waste:
            raise ValueError(
                "reward weights do not preserve ineffective-loss priority for the requested red_count"
            )


@dataclass(frozen=True)
class EnvironmentConfig:
    time_step_s: float = 0.005
    bias_update_interval_s: float = 0.1
    assignment_update_interval_s: float = 5.0
    max_steps: int = 36000
    scenario: ScenarioConfig = field(default_factory=ScenarioConfig)
    aircraft: AircraftConfig = field(default_factory=AircraftConfig)
    missile: MissileConfig = field(default_factory=MissileConfig)
    sensor: SensorConfig = field(default_factory=SensorConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    policy_start_mode: PolicyStartMode = "post_boost"
    policy_entry_speed_tolerance_ratio: float = 1.0e-6
    policy_entry_flight_path_tolerance_deg: float = 0.5

    @property
    def bias_update_steps(self) -> int:
        return int(round(self.bias_update_interval_s / self.time_step_s))

    @property
    def assignment_update_steps(self) -> int:
        return int(round(self.assignment_update_interval_s / self.time_step_s))

    @property
    def bias_updates_per_assignment(self) -> int:
        return int(round(self.assignment_update_interval_s / self.bias_update_interval_s))

    @property
    def policy_entry_steps(self) -> int:
        return int(round(self.missile.boost_duration_s / self.time_step_s))

    @property
    def policy_entry_time_s(self) -> float:
        return float(self.missile.boost_duration_s)

    @property
    def policy_horizon_steps(self) -> int:
        return self.max_steps - self.policy_entry_steps

    @property
    def policy_horizon_s(self) -> float:
        return self.policy_horizon_steps * self.time_step_s

    @property
    def remaining_guidance_horizon_s(self) -> float:
        return self.missile.max_guidance_time_s - self.policy_entry_time_s

    @property
    def policy_entry_speed_tolerance_mps(self) -> float:
        return self.missile.max_speed_mps * self.policy_entry_speed_tolerance_ratio

    def validate(self) -> None:
        if not _all_finite(
            self.time_step_s,
            self.bias_update_interval_s,
            self.assignment_update_interval_s,
            self.policy_entry_speed_tolerance_ratio,
            self.policy_entry_flight_path_tolerance_deg,
        ):
            raise ValueError("environment scalar parameters must be finite")
        if (
            self.time_step_s <= 0.0
            or self.bias_update_interval_s <= 0.0
            or self.assignment_update_interval_s <= 0.0
            or not _is_integer(self.max_steps)
            or self.max_steps <= 0
            or self.policy_entry_speed_tolerance_ratio < 0.0
            or self.policy_entry_flight_path_tolerance_deg < 0.0
        ):
            raise ValueError("environment scalar parameters are invalid")
        if self.policy_start_mode not in ("post_boost", "launch"):
            raise ValueError("policy_start_mode must be 'post_boost' or 'launch'")
        self.scenario.validate()
        self.aircraft.validate()
        self.missile.validate()
        self.sensor.validate()
        self.reward.validate()
        self.reward.validate_lexicographic_priority(
            self.scenario.red_count,
            self.scenario.blue_count,
        )
        bias_ratio = self.bias_update_interval_s / self.time_step_s
        assignment_ratio = self.assignment_update_interval_s / self.time_step_s
        hierarchy_ratio = self.assignment_update_interval_s / self.bias_update_interval_s
        boost_ratio = self.missile.boost_duration_s / self.time_step_s
        if not math.isclose(bias_ratio, round(bias_ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("bias_update_interval_s must be an integer multiple of time_step_s")
        if not math.isclose(assignment_ratio, round(assignment_ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("assignment_update_interval_s must be an integer multiple of time_step_s")
        if not math.isclose(hierarchy_ratio, round(hierarchy_ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                "assignment_update_interval_s must be an integer multiple of bias_update_interval_s"
            )
        if not math.isclose(boost_ratio, round(boost_ratio), rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError("missile boost_duration_s must be an integer multiple of time_step_s")
        if self.policy_start_mode == "post_boost" and self.max_steps <= self.policy_entry_steps:
            raise ValueError("max_steps must exceed policy_entry_steps in post_boost mode")
        if self.remaining_guidance_horizon_s <= 0.0:
            raise ValueError("missile max_guidance_time_s must exceed policy entry time")
        if (
            self.scenario.blue_speed_range_mps[0] < self.aircraft.min_speed_mps
            or self.scenario.blue_speed_range_mps[1] > self.aircraft.max_speed_mps
        ):
            raise ValueError("blue initial speed range must be within aircraft speed limits")
        if (
            self.scenario.blue_altitude_range_m[0] < self.aircraft.min_altitude_m
            or self.scenario.blue_altitude_range_m[1] > self.aircraft.max_altitude_m
        ):
            raise ValueError("blue initial altitude range must be within aircraft altitude limits")


@dataclass
class ThreeDoFState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    mass_kg: float
    fuel_mass_kg: float = 0.0
    energy: float = 1.0
    alive: bool = True
    loss_reason: str = ""
    min_range_m: float = math.inf
    min_range_target_index: int = -1
    current_target_index: int = -1
    age_s: float = 0.0
    seeker_locked: bool = False
    fov_out_time_s: float = 0.0
    guidance_mode: GuidanceMode = "boost"
    target_estimate_valid: bool = False
    target_estimate_target_index: int = -1
    target_estimate_position_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    target_estimate_velocity_mps: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    target_estimate_age_s: float = 0.0
    boost_initial_flight_path_angle_rad: float = math.nan
    guidance_bias: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    pn_load_body_g: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    bias_load_body_g: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    gravity_load_body_g: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    final_load_body_g: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def copy(self) -> "ThreeDoFState":
        return ThreeDoFState(
            position_m=self.position_m.astype(np.float64).copy(),
            velocity_mps=self.velocity_mps.astype(np.float64).copy(),
            mass_kg=float(self.mass_kg),
            fuel_mass_kg=float(self.fuel_mass_kg),
            energy=float(self.energy),
            alive=bool(self.alive),
            loss_reason=str(self.loss_reason),
            min_range_m=float(self.min_range_m),
            min_range_target_index=int(self.min_range_target_index),
            current_target_index=int(self.current_target_index),
            age_s=float(self.age_s),
            seeker_locked=bool(self.seeker_locked),
            fov_out_time_s=float(self.fov_out_time_s),
            guidance_mode=self.guidance_mode,
            target_estimate_valid=bool(self.target_estimate_valid),
            target_estimate_target_index=int(self.target_estimate_target_index),
            target_estimate_position_m=np.asarray(self.target_estimate_position_m, dtype=np.float64).copy(),
            target_estimate_velocity_mps=np.asarray(self.target_estimate_velocity_mps, dtype=np.float64).copy(),
            target_estimate_age_s=float(self.target_estimate_age_s),
            boost_initial_flight_path_angle_rad=float(self.boost_initial_flight_path_angle_rad),
            guidance_bias=np.asarray(self.guidance_bias, dtype=np.float64).copy(),
            pn_load_body_g=np.asarray(self.pn_load_body_g, dtype=np.float64).copy(),
            bias_load_body_g=np.asarray(self.bias_load_body_g, dtype=np.float64).copy(),
            gravity_load_body_g=np.asarray(self.gravity_load_body_g, dtype=np.float64).copy(),
            final_load_body_g=np.asarray(self.final_load_body_g, dtype=np.float64).copy(),
        )


@dataclass
class EngagementState:
    red: list[ThreeDoFState]
    blue: list[ThreeDoFState]
    time_s: float = 0.0
    step_count: int = 0
    style: ScenarioStyle = "many_to_many"
    parameters: Dict[str, Any] = field(default_factory=dict)

    def copy(self) -> "EngagementState":
        return EngagementState(
            red=[x.copy() for x in self.red],
            blue=[x.copy() for x in self.blue],
            time_s=float(self.time_s),
            step_count=int(self.step_count),
            style=self.style,
            parameters=dict(self.parameters),
        )


@dataclass(frozen=True)
class RelativeKinematics:
    relative_position_m: np.ndarray
    relative_velocity_mps: np.ndarray
    range_m: float
    range_rate_mps: float
    closing_speed_mps: float
    los_unit: np.ndarray
    los_rate_radps: np.ndarray
    azimuth_rad: float
    elevation_rad: float


def los_kinematics(shooter: ThreeDoFState, target: ThreeDoFState) -> RelativeKinematics:
    rel_pos = target.position_m - shooter.position_m
    rel_vel = target.velocity_mps - shooter.velocity_mps
    distance = max(norm(rel_pos), EPS)
    los_unit = rel_pos / distance
    range_rate = float(np.dot(rel_vel, los_unit))
    los_rate = (rel_vel - range_rate * los_unit) / distance
    azimuth = math.atan2(float(rel_pos[EAST_AXIS]), float(rel_pos[NORTH_AXIS]))
    elevation = math.atan2(float(rel_pos[UP_AXIS]), math.hypot(float(rel_pos[NORTH_AXIS]), float(rel_pos[EAST_AXIS])))
    return RelativeKinematics(rel_pos, rel_vel, distance, range_rate, -range_rate, los_unit, los_rate, azimuth, elevation)


@dataclass
class RedAction:
    target_indices: np.ndarray
    guidance_bias: np.ndarray

    def __post_init__(self) -> None:
        targets = np.asarray(self.target_indices, dtype=np.int64)
        bias = np.asarray(self.guidance_bias, dtype=np.float64)
        if targets.ndim != 1:
            raise ValueError(f"target_indices shape {targets.shape} must be [R]")
        expected_bias_shape = (targets.shape[0], 2)
        if bias.shape != expected_bias_shape:
            raise ValueError(f"guidance_bias shape {bias.shape} must be {expected_bias_shape}")
        if not np.all(np.isfinite(bias)):
            raise ValueError("guidance_bias must contain only finite values")
        if np.any((bias < -1.0) | (bias > 1.0)):
            raise ValueError("guidance_bias values must be in [-1, 1]")
        self.target_indices = targets.copy()
        self.guidance_bias = bias.copy()


@dataclass
class BlueAction:
    load_command_body_g: np.ndarray


@dataclass
class JointAction:
    red: RedAction
    blue: BlueAction


@dataclass
class EnvironmentObservation:
    assignment_actor_inputs: AssignmentActorInputs
    execution_actor_inputs: OverloadBiasActorInputs
    assignment_critic_inputs: AssignmentCriticInputs
    execution_critic_inputs: OverloadBiasCriticInputs
    assignment_matrix: np.ndarray


@dataclass
class AdjudicationResult:
    reward_high: float
    reward_low: np.ndarray
    done: bool
    info: Dict[str, Any]
    terminated: bool = False
    truncated: bool = False
    termination_reason: TerminationReason = "none"


@dataclass
class ReplayTransition:
    state: EngagementState
    action: JointAction
    reward_high: float
    reward_low: np.ndarray
    next_state: EngagementState
    observation: EnvironmentObservation
    next_observation: EnvironmentObservation
    info: Dict[str, Any]


@dataclass
class EnvironmentStep:
    observation: EnvironmentObservation
    reward_high: float
    reward_low: np.ndarray
    done: bool
    info: Dict[str, Any]
    assignment_matrix: np.ndarray
    terminated: bool = False
    truncated: bool = False
    termination_reason: TerminationReason = "none"


@dataclass(frozen=True)
class ControlDecisionRequest:
    """The sole environment-owned schedule for hierarchical policy decisions."""

    assignment_due: bool
    bias_due: bool
    reason: Literal["initial", "periodic", "event", "held"]
