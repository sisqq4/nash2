
from dataclasses import dataclass
from typing import Optional, List

@dataclass
class EnvConfig:
    """Environment configuration for the missile–escape game.

    Units:
        - Position: kilometers (km)
        - Time: seconds (s)
        - Speed: kilometers per second (km/s)
    """

    # Overall scale (for reward shaping)
    region_span: float = 160.0  # km

    # Blue aircraft initial region (random inside this box)
    blue_x_min: float = 10.0
    blue_x_max: float = 30.0
    blue_y_min: float = 0.0
    blue_y_max: float = 0.0
    blue_z_min: float = 10.0
    blue_z_max: float = 10.0

    # Simulation
    dt: float = 0.1          # [s] physics & hit-judgement step
    missile_update_dt: float = 0.01  # [s] inner missile guidance/position update step
    max_steps: int = 1200    # episode length in steps (~120 s)

    # Blue aircraft dynamics
    blue_max_speed: float = 600.0 / 1000.0  # km/s (600 m/s)
    blue_min_speed: float = 100.0 / 1000.0  # km/s (100 m/s)
    blue_accel: float = 0.09                 # km/s^2 (~9 g)
    # Blue aircraft initial heading (degrees in xy-plane, 0 along +x)
    blue_heading_min: float = -180.0
    blue_heading_max: float = 180.0
    blue_fixed_start: bool = False
    blue_fixed_x: float = 0.0
    blue_fixed_y: float = 0.0
    blue_fixed_z: float = 10.0

    # Missile dynamics
    missile_speed: float = 4900.0 / 3600.0   # km/s
    missile_target_speed: float = 2100.0 / 1000.0  # km/s (2100 m/s)
    missile_max_speed: float = 2100.0 / 1000.0  # km/s (cap for boost speed)
    missile_boost_duration: float = 5.0      # [s]
    missile_speed_decay_interval: float = 1.0  # [s]
    missile_speed_decay_factor: float = 0.99
    missile_min_speed: float = 980.0 / 3600.0  # km/s
    num_missiles: int = 1
    missile_spawn_x: float = 0.0
    missile_spawn_y: float = 0.0
    missile_spawn_z: float = 10.0
    missile_spawn_mode: str = "fixed_point"  # fixed_point | annulus
    missile_spawn_radius_min: float = 6.0
    missile_spawn_radius_max: float = 30.0
    missile_spawn_alt_min: float = 7.0
    missile_spawn_alt_max: float = 13.0
    missile_launch_time_std: float = 0.5
    missile_launch_time_clip: float = 2.0
    # Optional deterministic per-missile evaluation profiles.  When left as
    # None, training and existing randomized tests keep their original launch
    # sampling and shared seeker/guidance parameters.
    missile_fixed_positions: Optional[List[List[float]]] = None
    missile_fixed_launch_times: Optional[List[float]] = None
    missile_nav_gains: Optional[List[float]] = None
    missile_seeker_fov_deg_by_missile: Optional[List[float]] = None
    missile_seeker_memory_time_by_missile: Optional[List[float]] = None
    missile_speed_decay_factor_by_missile: Optional[List[float]] = None
    nav_gain: float = 4.5
    missile_max_overload_g: float = 45.0  # max lateral load factor [g] (40-50g target)
    missile_cd: float = 0.28
    missile_ref_area_m2: float = 0.015
    missile_mass_kg: float = 157.0
    missile_boost_accel: float = 30.0 * 9.8 / 1000.0  # km/s^2 (30g boost)
    missile_stall_speed: float = 300.0 / 1000.0  # km/s (stall speed)
    missile_k_drag_base: float = 0.00005
    missile_k_induced: float = 30000.0
    missile_scale_height_m: float = 8500.0
    missile_seeker_fov_deg: float = 60.0
    missile_seeker_memory_time: float = 2.0  # [s]
    missile_terminal_blind_range_km: float = 0.8

    # Missile lifetime / energy
    missile_max_flight_time: float = 120.0   # [s]

    # Hit radius (warhead lethal radius, km)
    hit_radius: float = 0.015  # ~15 m

    # Game-theoretic launcher (position + launch time)
    candidate_launch_count: int = 32
    num_blue_strategies: int = 8
    fictitious_iters: int = 200
    blue_escape_distance: float = 10.0      # km (only for rough payoff shaping)
    max_launch_time: float = 8.0            # latest first-launch time [s]
    min_launch_interval: float = 1.0        # between launches [s]
    red_launch_x_min: float = 0.0
    red_launch_x_max: float = 5.0
    red_launch_y_min: float = -10.0
    red_launch_y_max: float = 10.0
    red_launch_z_min: float = 8.0
    red_launch_z_max: float = 12.0

    # Differential-game controller for PN gains
    use_diff_game: bool = False
    diff_step_size: float = 0.2
    diff_delta_gain: float = 0.2
    diff_gain_min: float = 0.5
    diff_gain_max: float = 8.0
    diff_w_dist: float = 1.0
    diff_w_gain: float = 0.01

    # Ground / terrain
    ground_crash_penalty: float = -5.0  # penalty when blue hits the ground
    ground_proximity_threshold: float = 2.0  # km, start penalizing below this altitude
    ground_proximity_penalty: float = 1.0  # max penalty applied at ground level

    # Reward shaping (units: km, km/s)
    safe_altitude_min: float = 8.0
    safe_altitude_max: float = 12.0
    safe_altitude_tolerance: float = 1.0
    height_reward_weight: float = 1.2
    distance_ratio_weight: float = 1.0
    danger_distance: float = 5.0
    engagement_range: float = 25.0
    danger_scale: float = 3.0
    climb_angle_limit_deg: float = 80.0
    climb_angle_penalty: float = 0.5
    max_sustained_pitch_deg: float = 30.0  # limit for sustained climb actions

    # Threat evaluation parameters
    threat_heading_max: float = 0.5 * 3.141592653589793
    threat_pitch_max: float = 0.5 * 3.141592653589793
    threat_omega: float = 0.2
    threat_dist_max: float = 160.0
    threat_kd: float = 1.0
    threat_sigma: float = 1e-8
    threat_reward_relief: float = 0.25
    threat_reward_increase: float = 1.0
    threat_aggressive_threshold: float = 0.6
    threat_aggressive_scale: float = 1.5

    # Threat-driven maneuver overrides
    threat_maneuver_start: float = 0.6
    threat_maneuver_stop: float = 0.4
    threat_maneuver_steps: int = 1

    # Reward selection
    reward_mode: str = "auto"  # auto | short_range | mid_small_azimuth | mid_large_azimuth
    short_range_distance: float = 8.0  # km
    small_azimuth_deg: float = 30.0
    short_range_buffer: float = 2.0  # km buffer before switching to short-range rules
    short_turn_roll_target_deg: float = 60.0

    short_range_distance_weight: float = 1.0
    short_range_turn_weight: float = 0.6
    short_range_roll_weight: float = 0.4
    short_range_speed_weight: float = 0.3
    short_range_height_weight: float = 0.4

    mid_small_azimuth_weight: float = 1.0
    mid_small_height_weight: float = 0.8
    mid_small_opposite_weight: float = 0.8
    mid_small_speed_weight: float = 0.4
    mid_small_level_weight: float = 0.4

    mid_large_distance_weight: float = 1.0
    mid_large_azimuth_weight: float = 0.8
    mid_large_height_weight: float = 0.8
    mid_large_speed_weight: float = 0.4
    mid_large_level_weight: float = 0.4
    mid_large_roll_zero_weight: float = 0.6
    # Multi-missile collaborative reward weights
    multi_distance_weight: float = 1.0
    multi_height_weight: float = 0.5
    multi_threat_relief_weight: float = 0.8
    multi_threat_increase_weight: float = 1.2
    multi_encirclement_penalty_weight: float = 1.0
    # Cenc = c1 * Cang + c2 * Csyn + c3 * Ccor
    coop_c1: float = 0.4
    coop_c2: float = 0.35
    coop_c3: float = 0.25
    coop_tau_t: float = 8.0
    coop_corridor_ref_width: float = 2.0

    # Threat score Ti = sigma(b1*(1/ri)+b2*max(0,-r_dot_i)+b3*(1/tgo_i)+b4*|q_i|+b5*xi_M_i)
    threat_b1: float = 1.0
    threat_b2: float = 1.2
    threat_b3: float = 1.1
    threat_b4: float = 0.6
    threat_b5: float = 0.8
    threat_softmax_gamma1: float = 2.0
    threat_softmax_gamma2: float = 0.8

    # Logging / Tacview export
    save_dir: str = "outputs"
    log_trajectories: bool = True


@dataclass
class TrainConfig:
    """Training hyperparameters for the blue RL agent."""

    episodes: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    replay_size: int = 50_000
    start_learning: int = 1_000

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 20_000

    n_step: int = 3
    atom_size: int = 51
    v_min: float = -10.0
    v_max: float = 10.0
    noisy_std: float = 0.5
    per_alpha: float = 0.6
    per_beta_start: float = 0.4
    per_beta_frames: int = 100_000

    target_update_interval: int = 1_000

    print_interval: int = 10
    checkpoint_dir: str = "outputs/checkpoints"
    checkpoint_interval: int = 50
    load_checkpoint_path: Optional[str] = None
    load_blue: bool = True
    load_red: bool = False
    results_dir: str = "outputs/results"
    reward_mode: str = "auto"
    blue_policy: str = "dqn"  # dqn | behavior_tree