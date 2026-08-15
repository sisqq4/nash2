
"""3D escape environment: blue aircraft vs. red missiles."""

from __future__ import annotations

from typing import Dict, Tuple, Any, List
import math
import numpy as np

from .game_theory_launcher import GameTheoreticLauncher, LaunchRegion
from .missile_dynamics import update_blue_state, update_missiles_pn
from .aircraft_missiles import Aircraft, Missiles
from .diff_game_controller import DifferentialGameController
from .acmi_io import write_csv, write_action_csv, write_table_csv
from . import action_space
from .threat_eval import ThreatEvaluator, ThreatParams
from config import EnvConfig



class EscapeEnv:
    def __init__(self, cfg: EnvConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        region = LaunchRegion(
            num_missiles=cfg.num_missiles,
            candidate_launch_count=cfg.candidate_launch_count,
            x_min=cfg.red_launch_x_min,
            x_max=cfg.red_launch_x_max,
            y_min=cfg.red_launch_y_min,
            y_max=cfg.red_launch_y_max,
            z_min=cfg.red_launch_z_min,
            z_max=cfg.red_launch_z_max,
            num_blue_strategies=cfg.num_blue_strategies,
            fictitious_iters=cfg.fictitious_iters,
            blue_escape_distance=cfg.blue_escape_distance,
            max_launch_time=cfg.max_launch_time,
            min_launch_interval=cfg.min_launch_interval,
        )
        self.launcher = GameTheoreticLauncher(region, rng=self.rng)
        self.diff_ctrl = DifferentialGameController(cfg) if cfg.use_diff_game else None
        # Dynamic models for blue aircraft and red missiles
        self.blue_model = Aircraft(
            dt=cfg.dt,
            accel_mag=cfg.blue_accel,
            v_max=cfg.blue_max_speed,
            v_min=cfg.blue_min_speed,
            max_sustained_pitch=math.radians(cfg.max_sustained_pitch_deg),
        )
        self.missile_model = Missiles(
            dt=cfg.dt,
            speed=cfg.missile_speed,
            max_overload_g=cfg.missile_max_overload_g,
        )

        self.blue_pos = np.zeros(3, dtype=float)
        self.blue_vel = np.zeros(3, dtype=float)

        M = cfg.num_missiles
        self.missile_pos = np.zeros((M, 3), dtype=float)
        self.missile_vel = np.zeros((M, 3), dtype=float)
        self.missile_speed = np.zeros(M, dtype=float)
        self.missile_initial_speed = np.zeros(M, dtype=float)
        self.nav_gains = np.full(M, cfg.nav_gain, dtype=float)
        self.initial_nav_gains = np.full(M, cfg.nav_gain, dtype=float)
        self.nav_gains = self.initial_nav_gains.copy()

        # Launch & lifetime
        self.missile_launch_times = np.zeros(M, dtype=float)
        self.missile_launched = np.zeros(M, dtype=bool)
        self.missile_alive = np.ones(M, dtype=bool)
        self.missile_time_alive = np.zeros(M, dtype=float)
        self.missile_is_boosting = np.zeros(M, dtype=bool)
        self.missile_fuel_depleted = np.zeros(M, dtype=bool)
        self.missile_seeker_lost_time = np.zeros(M, dtype=float)
        self.missile_fov_in_view = np.zeros(M, dtype=bool)
        self.missile_is_closing = np.zeros(M, dtype=bool)
        self.missile_closing_speed = np.zeros(M, dtype=float)
        self._prev_missile_fov_in_view = np.zeros(M, dtype=bool)
        self._prev_missile_is_closing = np.zeros(M, dtype=bool)
        self._guidance_events: List[Dict[str, Any]] = []

        self.step_count = 0
        self.time = 0.0
        self.done = False
        self.prev_threat = 1.0
        self.prev_min_dist = cfg.region_span
        self.episode_min_dist = cfg.region_span
        self.prev_blue_vel = np.zeros(3, dtype=float)
        self.initial_missile_distances = np.zeros(M, dtype=float)
        self.missile_target_distances = np.full(M, cfg.region_span, dtype=float)
        self.forced_maneuver_steps = 0
        self.threat_mode = False

        criteria = np.array(
            [
                [1, 1 / 2, 1 / 8],
                [2, 1, 1 / 6],
                [8, 6, 1],
            ],
            dtype=float,
        )
        self.threat_evaluator = ThreatEvaluator(
            ThreatParams(
                heading_max=cfg.threat_heading_max,
                pitch_max=cfg.threat_pitch_max,
                omega=cfg.threat_omega,
                dist_max=cfg.threat_dist_max,
                kd=cfg.threat_kd,
                sigma=cfg.threat_sigma,
                criteria=criteria,
            )
        )

        # Observation: blue pos (3) + blue vel (3) + rel missile pos (3*M)
        self.observation_dim = 3 + 3 + 3 * M
        self.action_dim = self.blue_model.num_strategies

        # Logging / episode indexing
        self.log_enabled = bool(cfg.log_trajectories)
        self.save_dir = cfg.save_dir
        self.episode_index = 0
        self.plane_global_id = 0
        self.missile_global_id = 0

        self._plane_track: List[List[float]] | None = None
        self._plane_name: str | None = None
        self._missile_tracks: List[List[List[float]]] | None = None
        self._missile_names: List[str] | None = None
        self._blue_action_log: List[List[float]] | None = None
        self._blue_action_name: str | None = None
        self._analysis_log: List[List[float | int | str]] | None = None
        self._analysis_name: str | None = None
        self._last_action: int | None = None
        self._episode_speed_sum = 0.0
        self._episode_speed_count = 0
        self._episode_min_speed = 0.0
        self._episode_altitude_sum = 0.0
        self._episode_altitude_count = 0
        self._episode_roll_abs_sum_deg = 0.0
        self._episode_roll_abs_count = 0
        self._episode_turn_rate_sum = 0.0
        self._episode_turn_rate_count = 0
    # ------------------------------------------------------------------
    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.time = 0.0
        self.done = False
        self._last_action = None
        self.blue_model.reset()

        # Blue initial position
        if self.cfg.blue_fixed_start:
            self.blue_pos = np.array(
                [self.cfg.blue_fixed_x, self.cfg.blue_fixed_y, self.cfg.blue_fixed_z],
                dtype=float,
            )
        else:
            self.blue_pos = np.array(
                [
                    self.rng.uniform(self.cfg.blue_x_min, self.cfg.blue_x_max),
                    self.rng.uniform(self.cfg.blue_y_min, self.cfg.blue_y_max),
                    self.rng.uniform(self.cfg.blue_z_min, self.cfg.blue_z_max),
                ],
                dtype=float,
            )

        # Random initial velocity direction
        # Random initial velocity direction in xy-plane
        heading_min = math.radians(float(self.cfg.blue_heading_min))
        heading_max = math.radians(float(self.cfg.blue_heading_max))
        if heading_max < heading_min:
            heading_min, heading_max = heading_max, heading_min
        heading = self.rng.uniform(heading_min, heading_max)
        v_dir = np.array([math.cos(heading), math.sin(heading), 0.0], dtype=float)
        self.blue_vel = v_dir * self.cfg.blue_max_speed

        # Red missile spawn and launch time profile.
        # In game-theory mode, recompute the red launch plan on every reset so
        # scenario-level red_launch_* overrides and the current blue initial
        # state actually affect the ACMI tracks. Fixed/annulus modes keep their
        # original sampling behaviour.
        self.missile_pos, self.missile_launch_times = self._sample_red_launch_profile()
        self.missile_launched[:] = False

        # Velocities start at zero (not yet launched)
        self.missile_vel.fill(0.0)
        self.missile_speed.fill(0.0)
        self.missile_initial_speed.fill(0.0)
        configured_nav_gains = getattr(self.cfg, "missile_nav_gains", None)
        if configured_nav_gains is not None:
            gains = np.asarray(configured_nav_gains, dtype=float)
            if gains.shape[0] != self.cfg.num_missiles:
                raise ValueError(
                    f"missile_nav_gains must contain {self.cfg.num_missiles} values, got {gains.shape[0]}"
                )
            self.initial_nav_gains = gains.copy()
        elif self.initial_nav_gains.shape[0] != self.cfg.num_missiles:
            self.initial_nav_gains = np.full(self.cfg.num_missiles, self.cfg.nav_gain, dtype=float)
        self.nav_gains = self.initial_nav_gains.copy()

        # Lifetime
        self.missile_alive[:] = True
        self.missile_time_alive[:] = 0.0
        self.missile_is_boosting[:] = False
        self.missile_fuel_depleted[:] = False
        self.missile_seeker_lost_time[:] = 0.0
        self.missile_fov_in_view[:] = False
        self.missile_is_closing[:] = False
        self.missile_closing_speed[:] = 0.0
        self._prev_missile_fov_in_view[:] = False
        self._prev_missile_is_closing[:] = False
        self._guidance_events = []
        self.prev_threat = 1.0
        self.forced_maneuver_steps = 0
        self.threat_mode = False
        self.initial_missile_distances = np.linalg.norm(
            self.missile_pos - self.blue_pos[None, :],
            axis=1,
        )
        self.missile_target_distances = self.initial_missile_distances.copy()
        if self.initial_missile_distances.size > 0:
            self.prev_min_dist = float(np.min(self.initial_missile_distances))
        else:
            self.prev_min_dist = float(self.cfg.region_span)
        self.episode_min_dist = float(self.prev_min_dist)
        self.prev_blue_vel = self.blue_vel.copy()
        self._episode_speed_sum = 0.0
        self._episode_speed_count = 0
        self._episode_min_speed = float(np.linalg.norm(self.blue_vel))
        self._episode_altitude_sum = 0.0
        self._episode_altitude_count = 0
        self._episode_roll_abs_sum_deg = 0.0
        self._episode_roll_abs_count = 0
        self._episode_turn_rate_sum = 0.0
        self._episode_turn_rate_count = 0
        self._update_episode_stats(turn_rate_deg=0.0)

        if self.log_enabled:
            self._init_logging()
            self._log_current_state()

        return self._get_obs()

    def get_red_params(self) -> Dict[str, Any]:
        return {
            "launcher_state": self.launcher.get_state(),
            "nav_gains": self.nav_gains.copy(),
        }

    def set_red_params(self, params: Dict[str, Any]) -> None:
        launcher_state = params.get("launcher_state")
        if launcher_state is not None:
            self.launcher.set_state(launcher_state)

        nav_gains = params.get("nav_gains")
        if nav_gains is not None:
            gains = np.asarray(nav_gains, dtype=float)
            if gains.shape[0] != self.cfg.num_missiles:
                gains = np.full(self.cfg.num_missiles, self.cfg.nav_gain, dtype=float)
            self.initial_nav_gains = gains.copy()

    # ------------------------------------------------------------------
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            raise RuntimeError("reset() must be called before stepping a finished episode")

        action = int(action)
        if not (0 <= action < self.action_dim):
            raise ValueError(f"Invalid action {action}")

        dt = self.cfg.dt

        prev_blue_pos = self.blue_pos.copy()
        prev_blue_vel = self.blue_vel.copy()
        prev_missile_pos = self.missile_pos.copy()
        prev_missile_vel = self.missile_vel.copy()
        self._guidance_events = []

        # 1) Update blue aircraft
        threat_pre = self._compute_threat()
        if threat_pre >= self.cfg.threat_maneuver_start:
            self.threat_mode = True
        elif threat_pre <= self.cfg.threat_maneuver_stop:
            self.threat_mode = False

        if self.threat_mode and not self.blue_model.has_forced_actions():
            self._schedule_evasive_maneuver(threat_pre)
        self.blue_pos, self.blue_vel = self.blue_model.step(
            self.blue_pos,
            self.blue_vel,
            action,
        )
        if self.forced_maneuver_steps > 0:
            self.forced_maneuver_steps -= 1

        # Enforce ground (terrain) for blue
        crashed = False
        if self.blue_pos[2] <= 0.0:
            self.blue_pos[2] = 0.0
            crashed = True

        # 2) Update time and possibly launch new missiles
        self.step_count += 1
        self.time += dt
        if self.log_enabled and self._blue_action_log is not None:
            self._last_action = int(action)
            self._blue_action_log.append([float(self.time), int(action)])

        for i in range(self.cfg.num_missiles):
            if (
                (not self.missile_launched[i])
                and self.missile_alive[i]
                and self.time >= self.missile_launch_times[i]
            ):
                # Launch missile i: set initial velocity toward current blue position
                direction = self.blue_pos - self.missile_pos[i]
                n = np.linalg.norm(direction)
                if n < 1e-6:
                    direction = np.array([1.0, 0.0, 0.0])
                    n = 1.0
                direction /= n
                initial_speed = float(np.linalg.norm(self.blue_vel))
                self.missile_initial_speed[i] = initial_speed
                self.missile_speed[i] = initial_speed
                self.missile_vel[i] = direction * initial_speed
                self.missile_launched[i] = True
                self.missile_time_alive[i] = 0.0
                self.missile_is_boosting[i] = True
                self.missile_fuel_depleted[i] = False
                self.missile_seeker_lost_time[i] = 0.0

        # 3) Seeker constraints (FOV, memory, terminal blind zone)
        guidance_active = np.zeros(self.cfg.num_missiles, dtype=bool)
        fov_in_view = np.zeros(self.cfg.num_missiles, dtype=bool)
        fov_deg = self._optional_cfg_array("missile_seeker_fov_deg_by_missile", self.cfg.missile_seeker_fov_deg)
        memory_time = self._optional_cfg_array(
            "missile_seeker_memory_time_by_missile", self.cfg.missile_seeker_memory_time
        )
        fov_cos = np.cos(np.radians(fov_deg))
        blind_range_km = self.cfg.missile_terminal_blind_range_km
        for i in range(self.cfg.num_missiles):
            if not (self.missile_launched[i] and self.missile_alive[i]):
                continue
            rel = self.blue_pos - self.missile_pos[i]
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm <= blind_range_km:
                self.missile_seeker_lost_time[i] = 0.0
                guidance_active[i] = True
                fov_in_view[i] = True
                continue
            vel_norm = float(np.linalg.norm(self.missile_vel[i]))
            if rel_norm < 1e-6 or vel_norm < 1e-6:
                self.missile_seeker_lost_time[i] = 0.0
                guidance_active[i] = True
                fov_in_view[i] = True
                continue
            cos_angle = float(np.dot(self.missile_vel[i], rel) / (vel_norm * rel_norm))
            if cos_angle >= fov_cos[i]:
                self.missile_seeker_lost_time[i] = 0.0
                guidance_active[i] = True
                fov_in_view[i] = True
            else:
                self.missile_seeker_lost_time[i] += dt
                if self.missile_seeker_lost_time[i] > memory_time[i]:
                    self.missile_alive[i] = False
                    self.nav_gains[i] = 0.0
                    self.missile_vel[i] = 0.0
                    self.missile_speed[i] = 0.0
                    self.missile_is_boosting[i] = False
                else:
                    guidance_active[i] = False

        nav_gains_effective = self.nav_gains.copy()
        nav_gains_effective[~guidance_active] = 0.0

        # 3.1) Geometry state: whether each launched/alive missile is in closing geometry.
        is_closing = np.zeros(self.cfg.num_missiles, dtype=bool)
        closing_speed = np.zeros(self.cfg.num_missiles, dtype=float)
        for i in range(self.cfg.num_missiles):
            if not (self.missile_launched[i] and self.missile_alive[i]):
                continue
            rel = self.blue_pos - self.missile_pos[i]
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm < 1e-6:
                continue
            los = rel / rel_norm
            rel_vel = self.blue_vel - self.missile_vel[i]
            vc = -float(np.dot(rel_vel, los))
            closing_speed[i] = vc
            is_closing[i] = vc > 0.0

        # 3.2) Record per-missile state transitions (FOV and closing geometry).
        for i in range(self.cfg.num_missiles):
            if not self.missile_launched[i]:
                continue
            prev_fov = bool(self._prev_missile_fov_in_view[i])
            curr_fov = bool(fov_in_view[i])
            if prev_fov != curr_fov:
                self._guidance_events.append(
                    {
                        "time": float(self.time),
                        "step": int(self.step_count),
                        "missile_id": int(i),
                        "event": "fov_state_change",
                        "from": prev_fov,
                        "to": curr_fov,
                    }
                )

            prev_closing = bool(self._prev_missile_is_closing[i])
            curr_closing = bool(is_closing[i])
            if prev_closing != curr_closing:
                self._guidance_events.append(
                    {
                        "time": float(self.time),
                        "step": int(self.step_count),
                        "missile_id": int(i),
                        "event": "closing_state_change",
                        "from": prev_closing,
                        "to": curr_closing,
                        "closing_speed": float(closing_speed[i]),
                    }
                )

        self.missile_fov_in_view = fov_in_view
        self.missile_is_closing = is_closing
        self.missile_closing_speed = closing_speed
        self._prev_missile_fov_in_view = fov_in_view.copy()
        self._prev_missile_is_closing = is_closing.copy()

        # 4) Update missile speed profiles for launched & alive missiles
        idx_active = np.where(self.missile_launched & self.missile_alive)[0]
        max_overload = np.full(self.cfg.num_missiles, self.cfg.missile_max_overload_g, dtype=float)
        decay_factor = self._optional_cfg_array("missile_speed_decay_factor_by_missile", 1.0)
        if idx_active.size > 0:
            for i in idx_active:
                speed = float(self.missile_speed[i])
                if speed <= 1e-6:
                    speed = float(np.linalg.norm(self.missile_vel[i]))
                if not self.missile_fuel_depleted[i]:
                    speed = min(speed + self.cfg.missile_boost_accel * dt, self.cfg.missile_target_speed)
                    if speed >= self.cfg.missile_target_speed - 1e-9:
                        self.missile_fuel_depleted[i] = True
                        self.missile_is_boosting[i] = False
                else:
                    self.missile_is_boosting[i] = False

                speed_m_s = speed * 1000.0
                max_g_aero = (speed_m_s ** 2) / 20000.0 if speed_m_s > 1e-6 else 0.0
                limit_g = min(max_g_aero, self.cfg.missile_max_overload_g)
                max_overload[i] = limit_g

                total_g = self._estimate_missile_load_g(
                    self.missile_pos[i],
                    self.missile_vel[i],
                    self.blue_pos,
                    self.blue_vel,
                    nav_gains_effective[i],
                    speed,
                    dt,
                    limit_g,
                )
                drag = self._missile_drag_decel(self.missile_pos[i, 2], speed, total_g)
                speed = max(speed - drag * dt, 0.0)
                if decay_factor[i] > 0.0 and decay_factor[i] != 1.0:
                    interval = max(float(self.cfg.missile_speed_decay_interval), 1e-12)
                    speed *= float(decay_factor[i]) ** (dt / interval)

                if speed < self.cfg.missile_stall_speed:
                    self.missile_alive[i] = False
                    self.nav_gains[i] = 0.0
                    self.missile_vel[i] = 0.0
                    self.missile_speed[i] = 0.0
                    self.missile_is_boosting[i] = False
                else:
                    self.missile_speed[i] = speed

        # 5) Differential-game update of nav_gains for launched & alive missiles
        if self.diff_ctrl is not None:
            idx = np.where(self.missile_launched & self.missile_alive)[0]
            if idx.size > 0:
                nav_sub = self.nav_gains[idx].copy()
                new_nav = self.diff_ctrl.update_nav_gains(
                    self.blue_pos,
                    self.blue_vel,
                    self.missile_pos[idx],
                    self.missile_vel[idx],
                    nav_sub,
                    dt,
                )
                self.nav_gains[idx] = new_nav
                self.nav_gains[~self.missile_alive] = 0.0

        nav_gains_effective = self.nav_gains.copy()
        nav_gains_effective[~guidance_active] = 0.0

        # 6) PN update for launched missiles (inner integration at missile_update_dt)
        #    and finer hit judgement on each missile_update_dt segment.
        idx_launched = np.where(self.missile_launched & self.missile_alive)[0]
        hit = False
        step_min_dist = float("inf")
        if idx_launched.size > 0:
            blue_pos_delta = self.blue_pos - prev_blue_pos
            blue_vel_delta = self.blue_vel - prev_blue_vel
            original_missile_dt = self.missile_model.dt
            try:
                fixed_substep_dt = 0.01
                full_substeps = int(dt // fixed_substep_dt)
                substep_dts = [fixed_substep_dt] * full_substeps
                remaining_dt = dt - full_substeps * fixed_substep_dt
                if remaining_dt > 1e-12:
                    substep_dts.append(float(remaining_dt))
                if not substep_dts:
                    substep_dts = [float(dt)]

                elapsed_time = 0.0
                for sub_dt in substep_dts:
                    frac0 = elapsed_time / max(dt, 1e-12)
                    elapsed_time += sub_dt
                    frac1 = elapsed_time / max(dt, 1e-12)
                    sub_prev_missile_pos = self.missile_pos.copy()
                    blue_sub_end = prev_blue_pos + frac1 * blue_pos_delta
                    blue_vel_sub_end = prev_blue_vel + frac1 * blue_vel_delta
                    self.missile_model.dt = float(sub_dt)
                    sub_pos, sub_vel = self.missile_model.step(
                        self.missile_pos[idx_launched],
                        self.missile_vel[idx_launched],
                        self.missile_speed[idx_launched],
                        blue_sub_end,
                        blue_vel_sub_end,
                        nav_gains_effective[idx_launched],
                        max_overload_g=max_overload[idx_launched],
                    )
                    self.missile_pos[idx_launched] = sub_pos
                    self.missile_vel[idx_launched] = sub_vel
                    blue_sub_start = prev_blue_pos + frac0 * blue_pos_delta
                    sub_hit, sub_min_dist, sub_missile_distances = self._check_hits_between_states(
                        blue_start=blue_sub_start,
                        blue_end=blue_sub_end,
                        missile_start=sub_prev_missile_pos,
                        missile_end=self.missile_pos,
                        return_per_missile_distances=True,
                    )
                    self.missile_target_distances = sub_missile_distances
                    step_min_dist = min(step_min_dist, sub_min_dist)
                    if sub_hit:
                        hit = True
                        break
            finally:
                self.missile_model.dt = original_missile_dt

        # 7) Update missile lifetime / energy
        self.missile_time_alive[self.missile_launched & self.missile_alive] += dt
        expired = self.missile_time_alive >= self.cfg.missile_max_flight_time
        self.missile_alive[expired] = False
        self.nav_gains[expired] = 0.0
        self.missile_speed[expired] = 0.0
        self.missile_is_boosting[expired] = False

        # 8) Enforce ground for missiles: z <= 0 destroys the missile
        for i in range(self.cfg.num_missiles):
            if self.missile_launched[i] and self.missile_alive[i] and self.missile_pos[i, 2] <= 0.0:
                self.missile_pos[i, 2] = 0.0
                self.missile_alive[i] = False
                self.nav_gains[i] = 0.0
                self.missile_vel[i] = 0.0
                self.missile_speed[i] = 0.0
                self.missile_is_boosting[i] = False

        if self.log_enabled:
            self._log_current_state(prev_blue_vel=prev_blue_vel, prev_missile_vel=prev_missile_vel)

        # 9) Hit detection (line-segment / sphere)
        if not np.isfinite(step_min_dist):
            hit, step_min_dist = self._check_hits(prev_blue_pos, prev_missile_pos)
        # Episode-level global minimum missile-target distance:
        # per step, compute all missile distances and keep the smallest seen so far.
        all_missile_dists = np.linalg.norm(self.missile_pos - self.blue_pos[None, :], axis=1)
        if idx_launched.size == 0:
            self.missile_target_distances = all_missile_dists.astype(float)
        if all_missile_dists.size > 0:
            step_min_dist = float(min(step_min_dist, float(np.min(all_missile_dists))))
        self.episode_min_dist = float(min(self.episode_min_dist, step_min_dist))
        min_dist = float(self.episode_min_dist)
        turn_rate_deg = self._compute_turn_rate_deg(prev_blue_vel)
        self._update_episode_stats(turn_rate_deg=turn_rate_deg)

        timeout = self.step_count >= self.cfg.max_steps
        missiles_exhausted = not np.any(self.missile_alive)

        # 10) Terminal conditions and reward
        if crashed:
            reward = self.cfg.ground_crash_penalty
            self.done = True
        elif hit:
            reward = -100.0
            self.done = True
        elif missiles_exhausted:
            reward = 100.0
            self.done = True
        elif timeout:
            reward = 100.0
            self.done = True
        else:
            # Keep reward shaping based on current-step tactical distance,
            # while reporting min_dist as episode-global minimum for statistics.
            reward = self._compute_reward(step_min_dist, prev_blue_vel)

        if self.done and self.log_enabled:
            self._flush_logs_to_csv()

        obs = self._get_obs()
        info: Dict[str, Any] = {
            "time": float(self.time),
            "step": int(self.step_count),
            "min_dist": float(min_dist),
            "step_min_dist": float(step_min_dist),
            "final_dist": self._compute_final_dist(),
            "final_speed": float(np.linalg.norm(self.blue_vel)),
            "avg_speed": self._episode_speed_sum / max(self._episode_speed_count, 1),
            "min_speed": float(self._episode_min_speed),
            "avg_altitude": self._episode_altitude_sum / max(self._episode_altitude_count, 1),
            "avg_roll_abs_deg": self._episode_roll_abs_sum_deg / max(self._episode_roll_abs_count, 1),
            "avg_turn_rate_deg": self._episode_turn_rate_sum / max(self._episode_turn_rate_count, 1),
            "hit": bool(hit),
            "timeout": bool(timeout),
            "crashed": bool(crashed),
            "missiles_exhausted": bool(missiles_exhausted),
            "threat": float(self.prev_threat),
            "nav_gains": self.nav_gains.copy(),
            "missile_alive": self.missile_alive.copy(),
            "missile_launched": self.missile_launched.copy(),
            "missile_time_alive": self.missile_time_alive.copy(),
            "launch_times": self.missile_launch_times.copy(),
            "missile_fov_in_view": self.missile_fov_in_view.copy(),
            "missile_is_closing": self.missile_is_closing.copy(),
            "missile_closing_speed": self.missile_closing_speed.copy(),
            "missile_target_distances": self.missile_target_distances.copy(),
            "guidance_events": list(self._guidance_events),
        }
        return obs, float(reward), bool(self.done), info

    # ------------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        rel_missiles = self.missile_pos - self.blue_pos[None, :]
        obs = np.concatenate(
            [self.blue_pos, self.blue_vel, rel_missiles.reshape(-1)],
            axis=0,
        )
        return obs.astype(np.float32)

    def _height_reward(self, altitude: float) -> float:
        safe_min = self.cfg.safe_altitude_min
        safe_max = self.cfg.safe_altitude_max
        tolerance = self.cfg.safe_altitude_tolerance
        hard_min = safe_min - tolerance
        hard_max = safe_max + tolerance

        if altitude < hard_min or altitude > hard_max:
            return -1.5
        if altitude < safe_min:
            ratio = (altitude - hard_min) / (safe_min - hard_min)
            return -1.0 + 2.0 * ratio
        if altitude > safe_max:
            ratio = (hard_max - altitude) / (hard_max - safe_max)
            return -1.0 + 2.0 * ratio

        center = (safe_min + safe_max) / 2.0
        span = (safe_max - safe_min) / 2.0
        offset = (altitude - center) / span
        return 1.0 - offset ** 2

    def _distance_reward(self) -> float:
        rd = 0.0
        rd_min = float("inf")
        danger_distance = max(self.cfg.danger_distance, 1e-6)
        engagement_range = self.cfg.engagement_range
        scale = engagement_range / danger_distance

        for i in range(self.cfg.num_missiles):
            if not (self.missile_launched[i] and self.missile_alive[i]):
                continue
            dist = float(np.linalg.norm(self.missile_pos[i] - self.blue_pos))
            dist = max(dist, 1e-6)
            if scale <= 1.0:
                rd = -1.5
            else:
                rd = math.log(dist / danger_distance) / math.log(scale)
            rd = max(min(rd, 1.0), -1.5)
            if rd < rd_min:
                rd_min = rd

        return 0.0 if rd_min == float("inf") else rd_min

    def _ground_proximity_penalty(self, altitude: float) -> float:
        threshold = self.cfg.ground_proximity_threshold
        if threshold <= 0.0 or altitude >= threshold:
            return 0.0
        ratio = (threshold - max(altitude, 0.0)) / threshold
        return -self.cfg.ground_proximity_penalty * ratio

    def _compute_threat(self) -> float:
        threat = 0.0
        for i in range(self.cfg.num_missiles):
            if not (self.missile_launched[i] and self.missile_alive[i]):
                continue
            treat = self.threat_evaluator.evaluate(
                self.blue_pos,
                self.blue_vel,
                self.missile_pos[i],
                self.missile_vel[i],
            )
            if treat > threat:
                threat = treat
        return float(threat)

    def _compute_reward(self, min_dist: float, prev_blue_vel: np.ndarray) -> float:
        if self.cfg.reward_mode == "multi_coop":
            reward = self._reward_multi_coop(prev_blue_vel)
            self.prev_min_dist = float(min_dist)
            self.prev_blue_vel = self.blue_vel.copy()
            return float(reward)
        primary_idx = self._select_primary_missile()
        azimuth_deg = (
            self._compute_missile_azimuth_deg(primary_idx) if primary_idx is not None else 0.0
        )

        mode = self.cfg.reward_mode
        if mode == "short_range":
            reward = self._reward_short_range(min_dist, prev_blue_vel)
        elif mode == "mid_small_azimuth":
            reward = self._reward_mid_small_azimuth(min_dist, azimuth_deg, primary_idx)
        elif mode == "mid_large_azimuth":
            reward = self._reward_mid_large_azimuth(min_dist, azimuth_deg, primary_idx)
        else:
            if min_dist <= self.cfg.short_range_distance:
                reward = self._reward_short_range(min_dist, prev_blue_vel)
            elif azimuth_deg <= self.cfg.small_azimuth_deg:
                reward = self._reward_mid_small_azimuth(min_dist, azimuth_deg, primary_idx)
            else:
                reward = self._reward_mid_large_azimuth(min_dist, azimuth_deg, primary_idx)

        threat = self._compute_threat()
        if threat <= self.prev_threat:
            reward += self.cfg.threat_reward_relief * threat
        else:
            reward -= self.cfg.threat_reward_increase * threat
        if threat > self.cfg.threat_aggressive_threshold:
            reward -= self.cfg.threat_aggressive_scale * (threat - self.cfg.threat_aggressive_threshold)

        reward += self._ground_proximity_penalty(self.blue_pos[2])

        self.prev_threat = threat
        self.prev_min_dist = float(min_dist)
        self.prev_blue_vel = self.blue_vel.copy()
        return float(reward)

    def _optional_cfg_array(self, name: str, default: float) -> np.ndarray:
        values = getattr(self.cfg, name, None)
        if values is None:
            return np.full(self.cfg.num_missiles, default, dtype=float)
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] != self.cfg.num_missiles:
            raise ValueError(f"{name} must contain {self.cfg.num_missiles} values, got {arr.shape[0]}")
        return arr.astype(float)

    def _fixed_launch_positions(self) -> np.ndarray | None:
        fixed_positions = getattr(self.cfg, "missile_fixed_positions", None)
        if fixed_positions is None:
            return None
        arr = np.asarray(fixed_positions, dtype=float)
        expected = (self.cfg.num_missiles, 3)
        if arr.shape != expected:
            raise ValueError(f"missile_fixed_positions must have shape {expected}, got {arr.shape}")
        return arr.astype(float)

    def _fixed_launch_times(self) -> np.ndarray | None:
        fixed_times = getattr(self.cfg, "missile_fixed_launch_times", None)
        if fixed_times is None:
            return None
        arr = np.asarray(fixed_times, dtype=float)
        expected = (self.cfg.num_missiles,)
        if arr.shape != expected:
            raise ValueError(f"missile_fixed_launch_times must have shape {expected}, got {arr.shape}")
        return arr.astype(float)

    def _sample_red_launch_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        fixed_pos = self._fixed_launch_positions()
        fixed_times = self._fixed_launch_times()
        if fixed_pos is not None or fixed_times is not None:
            launch_pos = fixed_pos if fixed_pos is not None else self._sample_missile_spawn_positions()
            launch_times = fixed_times if fixed_times is not None else self._sample_launch_times(self.cfg.num_missiles)
            return launch_pos.astype(float), launch_times.astype(float)

        mode = str(self.cfg.missile_spawn_mode).strip().lower()
        if mode == "game_theory":
            self._sync_launcher_region()
            launch_pos, launch_times = self.launcher.compute_launch_plan(
                blue_initial_pos=self.blue_pos,
                blue_speed=float(np.linalg.norm(self.blue_vel)),
                missile_speed=float(self.cfg.missile_speed),
            )
            return launch_pos.astype(float), launch_times.astype(float)

        return self._sample_missile_spawn_positions(), self._sample_launch_times(self.cfg.num_missiles)

    def _sync_launcher_region(self) -> None:
        """Keep the launcher region aligned with mutable EnvConfig overrides."""
        region = self.launcher.region
        region.num_missiles = self.cfg.num_missiles
        region.candidate_launch_count = self.cfg.candidate_launch_count
        region.x_min = self.cfg.red_launch_x_min
        region.x_max = self.cfg.red_launch_x_max
        region.y_min = self.cfg.red_launch_y_min
        region.y_max = self.cfg.red_launch_y_max
        region.z_min = self.cfg.red_launch_z_min
        region.z_max = self.cfg.red_launch_z_max
        region.num_blue_strategies = self.cfg.num_blue_strategies
        region.fictitious_iters = self.cfg.fictitious_iters
        region.blue_escape_distance = self.cfg.blue_escape_distance
        region.max_launch_time = self.cfg.max_launch_time
        region.min_launch_interval = self.cfg.min_launch_interval

    def _sample_missile_spawn_positions(self) -> np.ndarray:
        mode = str(self.cfg.missile_spawn_mode).strip().lower()
        if mode == "fixed_point":
            fixed_spawn = np.array(
                [self.cfg.missile_spawn_x, self.cfg.missile_spawn_y, self.cfg.missile_spawn_z],
                dtype=float,
            )
            return np.repeat(fixed_spawn[None, :], self.cfg.num_missiles, axis=0)

        if mode != "annulus":
            raise ValueError(
                f"Unsupported missile_spawn_mode={self.cfg.missile_spawn_mode!r}; "
                "expected 'fixed_point', 'annulus', or 'game_theory'."
            )
        radii = self.rng.uniform(self.cfg.missile_spawn_radius_min, self.cfg.missile_spawn_radius_max, self.cfg.num_missiles)
        bearings = self.rng.uniform(-math.pi, math.pi, self.cfg.num_missiles)
        alts = self.rng.uniform(self.cfg.missile_spawn_alt_min, self.cfg.missile_spawn_alt_max, self.cfg.num_missiles)
        x = radii * np.cos(bearings)
        y = radii * np.sin(bearings)
        return np.stack([x, y, alts], axis=1).astype(float)

    def _sample_launch_times(self, num_missiles: int) -> np.ndarray:
        if num_missiles <= 1:
            return np.zeros(1, dtype=float)
        launch_jitter = self.rng.normal(0.0, self.cfg.missile_launch_time_std, size=num_missiles)
        launch_jitter = np.clip(launch_jitter, -self.cfg.missile_launch_time_clip, self.cfg.missile_launch_time_clip)
        launch_times = launch_jitter - float(np.min(launch_jitter))
        return launch_times.astype(float)

    def _active_missile_indices(self) -> np.ndarray:
        return np.where(self.missile_launched & self.missile_alive)[0]

    def _compute_threat_terms(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx_active = self._active_missile_indices()
        if idx_active.size == 0:
            return idx_active, np.zeros(0), np.zeros(0), np.zeros(0)

        r_vec = self.blue_pos[None, :] - self.missile_pos[idx_active]
        r = np.linalg.norm(r_vec, axis=1)
        rel_vel = self.blue_vel[None, :] - self.missile_vel[idx_active]
        los = r_vec / np.maximum(r[:, None], 1e-6)
        r_dot = np.sum(rel_vel * los, axis=1)
        closing = np.maximum(-r_dot, 1e-6)
        tgo = r / closing
        q = np.linalg.norm(np.cross(r_vec, rel_vel), axis=1) / np.maximum(r ** 2, 1e-6)

        # Normalized residual maneuver capability proxy from speed margin.
        xi = np.clip(
            (self.missile_speed[idx_active] - self.cfg.missile_min_speed)
            / max(self.cfg.missile_max_speed - self.cfg.missile_min_speed, 1e-6),
            0.0,
            1.0,
        )
        return idx_active, r, tgo, q + 0.3 * xi

    def _compute_threat_scores(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx_active, r, tgo, q_plus_xi = self._compute_threat_terms()
        if idx_active.size == 0:
            return idx_active, np.zeros(0), np.zeros(0)

        r_vec = self.blue_pos[None, :] - self.missile_pos[idx_active]
        rel_vel = self.blue_vel[None, :] - self.missile_vel[idx_active]
        los = r_vec / np.maximum(r[:, None], 1e-6)
        r_dot = np.sum(rel_vel * los, axis=1)
        closing = np.maximum(0.0, -r_dot)
        xi = np.clip(
            (self.missile_speed[idx_active] - self.cfg.missile_min_speed)
            / max(self.cfg.missile_max_speed - self.cfg.missile_min_speed, 1e-6),
            0.0,
            1.0,
        )
        raw = (
            self.cfg.threat_b1 * (1.0 / np.maximum(r, 1e-6))
            + self.cfg.threat_b2 * closing
            + self.cfg.threat_b3 * (1.0 / np.maximum(tgo, 1e-6))
            + self.cfg.threat_b4 * np.abs(q_plus_xi - 0.3 * xi)
            + self.cfg.threat_b5 * xi
        )
        ti = 1.0 / (1.0 + np.exp(-raw))
        return idx_active, ti, tgo

    def _compute_collaborative_encirclement(self, idx_active: np.ndarray, tgo: np.ndarray) -> float:
        if idx_active.size <= 1:
            return 0.0
        rel = self.missile_pos[idx_active, :2] - self.blue_pos[None, :2]
        beta = np.sort(np.mod(np.arctan2(rel[:, 1], rel[:, 0]), 2.0 * math.pi))
        if beta.size <= 1:
            c_ang = 0.0
        else:
            gaps = np.diff(np.concatenate([beta, beta[:1] + 2.0 * math.pi]))
            delta_beta_max = float(np.max(gaps))
            c_ang = 1.0 - delta_beta_max / (2.0 * math.pi)

        tgo_var = float(np.var(tgo))
        c_syn = math.exp(-tgo_var / max(self.cfg.coop_tau_t ** 2, 1e-6))

        heading = self.blue_vel[:2]
        h_norm = float(np.linalg.norm(heading))
        if h_norm < 1e-6:
            heading = np.array([1.0, 0.0], dtype=float)
            h_norm = 1.0
        h_hat = heading / h_norm
        side = np.array([-h_hat[1], h_hat[0]], dtype=float)
        lateral = np.abs(np.dot(rel, side))
        w_safe = float(np.min(lateral)) if lateral.size > 0 else self.cfg.coop_corridor_ref_width
        w0 = max(self.cfg.coop_corridor_ref_width, 1e-6)
        c_cor = 1.0 - np.clip(w_safe / w0, 0.0, 1.0)

        c_enc = self.cfg.coop_c1 * c_ang + self.cfg.coop_c2 * c_syn + self.cfg.coop_c3 * c_cor
        return float(np.clip(c_enc, 0.0, 1.0))

    def _reward_multi_coop(self, prev_blue_vel: np.ndarray) -> float:
        idx_active = self._active_missile_indices()
        if idx_active.size == 0:
            return self._height_reward(self.blue_pos[2])

        dists = np.linalg.norm(self.missile_pos[idx_active] - self.blue_pos[None, :], axis=1)
        inv_mean_dist = 1.0 / max(float(np.mean(dists)), 1e-6)
        distance_reward = self.cfg.multi_distance_weight * inv_mean_dist
        height_reward = self.cfg.multi_height_weight * self._height_reward(self.blue_pos[2])

        idx_t, ti, tgo = self._compute_threat_scores()
        threat_mean = float(np.mean(ti)) if ti.size > 0 else 0.0
        if threat_mean <= self.prev_threat:
            threat_term = self.cfg.multi_threat_relief_weight * (self.prev_threat - threat_mean)
        else:
            threat_term = -self.cfg.multi_threat_increase_weight * (threat_mean - self.prev_threat)

        c_enc = self._compute_collaborative_encirclement(idx_t, tgo)
        reward = distance_reward + height_reward + threat_term - self.cfg.multi_encirclement_penalty_weight * c_enc
        reward += self._ground_proximity_penalty(self.blue_pos[2])
        self.prev_threat = threat_mean
        return float(reward)

    def _select_primary_missile(self) -> int | None:
        active = np.where(self.missile_launched & self.missile_alive)[0]
        if active.size == 0:
            return None
        dists = np.linalg.norm(self.missile_pos[active] - self.blue_pos[None, :], axis=1)
        return int(active[int(np.argmin(dists))])

    def _compute_missile_azimuth_deg(self, idx: int) -> float:
        rel = self.blue_pos - self.missile_pos[idx]
        rel_xy = np.array([rel[0], rel[1]], dtype=float)
        vel_xy = np.array([self.missile_vel[idx][0], self.missile_vel[idx][1]], dtype=float)
        rel_norm = float(np.linalg.norm(rel_xy))
        vel_norm = float(np.linalg.norm(vel_xy))
        if rel_norm < 1e-6 or vel_norm < 1e-6:
            return 0.0
        cos_angle = float(np.dot(rel_xy, vel_xy) / (rel_norm * vel_norm))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        return float(math.degrees(math.acos(cos_angle)))

    def _compute_turn_rate_deg(self, prev_vel: np.ndarray) -> float:
        prev_xy = np.array([prev_vel[0], prev_vel[1]], dtype=float)
        curr_xy = np.array([self.blue_vel[0], self.blue_vel[1]], dtype=float)
        prev_norm = float(np.linalg.norm(prev_xy))
        curr_norm = float(np.linalg.norm(curr_xy))
        if prev_norm < 1e-6 or curr_norm < 1e-6:
            return 0.0
        cos_angle = float(np.dot(prev_xy, curr_xy) / (prev_norm * curr_norm))
        cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
        angle_deg = float(math.degrees(math.acos(cos_angle)))
        return angle_deg / max(self.cfg.dt, 1e-6)

    def _compute_final_dist(self) -> float:
        launched = self.missile_launched
        if np.any(launched):
            dists = np.linalg.norm(self.missile_pos[launched] - self.blue_pos[None, :], axis=1)
            return float(np.min(dists))
        if self.initial_missile_distances.size > 0:
            return float(np.min(self.initial_missile_distances))
        return 0.0

    def _update_episode_stats(self, turn_rate_deg: float) -> None:
        speed = float(np.linalg.norm(self.blue_vel))
        altitude = float(self.blue_pos[2])
        roll_abs_deg = float(math.degrees(abs(self.blue_model.roll_rad or 0.0)))

        self._episode_speed_sum += speed
        self._episode_speed_count += 1
        self._episode_min_speed = min(self._episode_min_speed, speed)

        self._episode_altitude_sum += altitude
        self._episode_altitude_count += 1

        self._episode_roll_abs_sum_deg += roll_abs_deg
        self._episode_roll_abs_count += 1

        self._episode_turn_rate_sum += float(turn_rate_deg)
        self._episode_turn_rate_count += 1

    def _reward_short_range(self, min_dist: float, prev_blue_vel: np.ndarray) -> float:
        reward = 0.0
        distance_gain = (min_dist - self.prev_min_dist) / max(self.cfg.short_range_distance, 1e-6)
        reward += self.cfg.short_range_distance_weight * distance_gain

        roll_rad = self.blue_model.roll_rad or 0.0
        roll_target = math.radians(self.cfg.short_turn_roll_target_deg)
        roll_score = min(abs(roll_rad) / max(roll_target, 1e-6), 1.0)
        reward += self.cfg.short_range_roll_weight * roll_score

        turn_rate = self._compute_turn_rate_deg(prev_blue_vel)
        turn_score = min(turn_rate / 180.0, 1.0)
        reward += self.cfg.short_range_turn_weight * turn_score

        speed = float(np.linalg.norm(self.blue_vel))
        speed_norm = (speed - self.cfg.blue_min_speed) / max(
            self.cfg.blue_max_speed - self.cfg.blue_min_speed, 1e-6
        )
        reward += self.cfg.short_range_speed_weight * speed_norm

        reward += self.cfg.short_range_height_weight * self._height_reward(self.blue_pos[2])
        return reward

    def _reward_mid_small_azimuth(
        self,
        min_dist: float,
        azimuth_deg: float,
        primary_idx: int | None,
    ) -> float:
        reward = 0.0
        azimuth_score = 1.0 - min(azimuth_deg / max(self.cfg.small_azimuth_deg, 1e-6), 1.0)
        reward += self.cfg.mid_small_azimuth_weight * azimuth_score

        reward += self.cfg.mid_small_height_weight * self._height_reward(self.blue_pos[2])

        if primary_idx is not None:
            rel = self.blue_pos - self.missile_pos[primary_idx]
            rel_norm = float(np.linalg.norm(rel))
            if rel_norm > 1e-6:
                opposite = float(np.dot(self.blue_vel, rel) / (rel_norm * max(np.linalg.norm(self.blue_vel), 1e-6)))
                reward += self.cfg.mid_small_opposite_weight * max(opposite, 0.0)

        speed = float(np.linalg.norm(self.blue_vel))
        speed_norm = (speed - self.cfg.blue_min_speed) / max(
            self.cfg.blue_max_speed - self.cfg.blue_min_speed, 1e-6
        )
        reward += self.cfg.mid_small_speed_weight * speed_norm

        roll_rad = self.blue_model.roll_rad or 0.0
        roll_penalty = abs(roll_rad) / math.pi
        climb_angle = math.degrees(math.atan2(self.blue_vel[2], max(np.linalg.norm(self.blue_vel[:2]), 1e-6)))
        level_penalty = (abs(climb_angle) / 90.0) + roll_penalty
        reward -= self.cfg.mid_small_level_weight * min(level_penalty, 1.0)
        return reward

    def _reward_mid_large_azimuth(
        self,
        min_dist: float,
        azimuth_deg: float,
        primary_idx: int | None,
    ) -> float:
        reward = 0.0
        distance_gain = (min_dist - self.prev_min_dist) / max(self.cfg.short_range_distance, 1e-6)
        reward += self.cfg.mid_large_distance_weight * distance_gain

        azimuth_score = 1.0 - min(azimuth_deg / 180.0, 1.0)
        reward += self.cfg.mid_large_azimuth_weight * azimuth_score

        reward += self.cfg.mid_large_height_weight * self._height_reward(self.blue_pos[2])

        speed = float(np.linalg.norm(self.blue_vel))
        speed_norm = (speed - self.cfg.blue_min_speed) / max(
            self.cfg.blue_max_speed - self.cfg.blue_min_speed, 1e-6
        )
        reward += self.cfg.mid_large_speed_weight * speed_norm

        roll_rad = self.blue_model.roll_rad or 0.0
        roll_penalty = abs(roll_rad) / math.pi
        climb_angle = math.degrees(math.atan2(self.blue_vel[2], max(np.linalg.norm(self.blue_vel[:2]), 1e-6)))
        level_penalty = (abs(climb_angle) / 90.0) + roll_penalty
        reward -= self.cfg.mid_large_level_weight * min(level_penalty, 1.0)

        if min_dist <= self.cfg.short_range_distance + self.cfg.short_range_buffer:
            roll_zero_score = 1.0 - min(abs(roll_rad) / math.radians(30.0), 1.0)
            reward += self.cfg.mid_large_roll_zero_weight * roll_zero_score
        return reward

    def _air_density(self, altitude_km: float) -> float:
        altitude_m = max(0.0, altitude_km * 1000.0)
        rho0 = 1.225
        t0 = 288.15
        if altitude_m <= 11000.0:
            t = t0 - 0.0065 * altitude_m
            return rho0 * (t / t0) ** 4.25588
        if altitude_m <= 20000.0:
            return 0.36392 * math.exp((-altitude_m + 11000.0) / 6341.62)
        t = 216.65 + 0.001 * (altitude_m - 20000.0)
        return 0.088035 * (t / 216.65) ** -35.1632

    def _estimate_missile_load_g(
        self,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
        blue_pos: np.ndarray,
        blue_vel: np.ndarray,
        nav_gain: float,
        speed_km_s: float,
        dt: float,
        max_overload_g: float,
    ) -> float:
        speed = float(np.linalg.norm(missile_vel))
        if speed < 1e-6 or nav_gain == 0.0:
            return 0.0

        u = missile_vel / speed
        r = blue_pos - missile_pos
        r_norm = float(np.linalg.norm(r))
        if r_norm < 1e-6:
            return 0.0

        rel_vel = blue_vel - missile_vel
        los = r / r_norm
        closing_speed = -float(np.dot(rel_vel, los))
        if closing_speed <= 0.0:
            return 0.0
        los_omega = np.cross(r, rel_vel) / max(r_norm ** 2, 1e-9)

        # Standard 3D PN: a_n = N * Vc * (omega_LOS x u_m)
        a_cmd = nav_gain * closing_speed * np.cross(los_omega, u)
        lateral_acc_km_s2 = float(np.linalg.norm(a_cmd))
        if max_overload_g > 0.0:
            max_acc_km_s2 = (max_overload_g * 9.80665) / 1000.0
            lateral_acc_km_s2 = min(lateral_acc_km_s2, max_acc_km_s2)
        return (lateral_acc_km_s2 * 1000.0) / 9.80665

    def _missile_drag_decel(self, altitude_km: float, speed_km_s: float, total_g: float) -> float:
        altitude_m = max(0.0, altitude_km * 1000.0)
        speed_m_s = max(0.0, speed_km_s * 1000.0)
        v_sq = speed_m_s ** 2
        density_factor = math.exp(-altitude_m / self.cfg.missile_scale_height_m)
        drag_parasitic = self.cfg.missile_k_drag_base * density_factor * v_sq
        drag_induced = self.cfg.missile_k_induced * (total_g ** 2) / (density_factor * v_sq + 1.0)
        accel_m_s2 = drag_parasitic + drag_induced
        return accel_m_s2 / 1000.0

    def _schedule_evasive_maneuver(self, threat: float) -> None:
        if self.forced_maneuver_steps > 0:
            return
        if threat <= 0.0:
            return

        primitives = action_space.get_simple()
        if primitives.shape[0] < 11:
            return

        idx_active = np.where(self.missile_launched & self.missile_alive)[0]
        if idx_active.size == 0:
            return
        idx_t, ti, tgo = self._compute_threat_scores()
        if idx_t.size == 0:
            return
        logits = self.cfg.threat_softmax_gamma1 * ti + self.cfg.threat_softmax_gamma2 / np.maximum(tgo, 1e-6)
        logits -= np.max(logits)
        weights = np.exp(logits)
        weights /= max(np.sum(weights), 1e-9)

        one_v_one_cmds = []
        for i in idx_t:
            rel = self.missile_pos[i] - self.blue_pos
            roll_sign = 1.0 if rel[1] >= 0 else -1.0
            climb = rel[2] >= 0
            base_turn = primitives[9] if roll_sign >= 0 else primitives[10]
            base_vert = primitives[5] if climb else primitives[7]
            one_v_one_cmds.append(0.6 * base_turn + 0.4 * base_vert)
        a_mix = np.sum(weights[:, None] * np.array(one_v_one_cmds), axis=0)
        # Project mixed continuous command to nearest primitive action.
        nearest_idx = int(np.argmin(np.linalg.norm(primitives - a_mix[None, :], axis=1)))
        selected = primitives[nearest_idx]
        steps = max(1, int(self.cfg.threat_maneuver_steps))
        sequence = [selected] * steps
        self.blue_model.force_actions([np.asarray(a, dtype=float) for a in sequence])
        self.forced_maneuver_steps = steps

    # ------------------------------------------------------------------
    # Hit detection helpers
    # ------------------------------------------------------------------
    def _segment_sphere_hit(
        self,
        r0: np.ndarray,
        r1: np.ndarray,
        radius: float,
    ) -> Tuple[bool, float]:
        d = r1 - r0
        a = float(np.dot(d, d))
        if a < 1e-12:
            dist0 = float(np.linalg.norm(r0))
            return dist0 <= radius, dist0
        t = -float(np.dot(r0, d)) / a
        t_clamped = max(0.0, min(1.0, t))
        closest = r0 + t_clamped * d
        dist = float(np.linalg.norm(closest))
        hit = dist <= radius
        return hit, dist

    def _check_hits(
        self,
        prev_blue_pos: np.ndarray,
        prev_missile_pos: np.ndarray,
    ) -> Tuple[bool, float]:
        return self._check_hits_between_states(
            blue_start=prev_blue_pos,
            blue_end=self.blue_pos,
            missile_start=prev_missile_pos,
            missile_end=self.missile_pos,
        )

    def _check_hits_between_states(
        self,
        blue_start: np.ndarray,
        blue_end: np.ndarray,
        missile_start: np.ndarray,
        missile_end: np.ndarray,
        return_per_missile_distances: bool = False,
    ) -> Tuple[bool, float] | Tuple[bool, float, np.ndarray]:
        hit_any = False
        min_dist = float("inf")
        per_missile_distances = np.full(self.cfg.num_missiles, self.cfg.region_span, dtype=float)

        for i in range(self.cfg.num_missiles):
            if not (self.missile_launched[i] and self.missile_alive[i]):
                continue
            r0 = missile_start[i] - blue_start
            r1 = missile_end[i] - blue_end
            hit, dist = self._segment_sphere_hit(r0, r1, self.cfg.hit_radius)
            per_missile_distances[i] = dist
            if dist < min_dist:
                min_dist = dist
            if hit:
                hit_any = True

        if not np.isfinite(min_dist):
            dists = np.linalg.norm(
                self.missile_pos[self.missile_launched & self.missile_alive] - self.blue_pos[None, :],
                axis=1,
            )
            if dists.size > 0:
                min_dist = float(np.min(dists))
                launched_idx = np.where(self.missile_launched & self.missile_alive)[0]
                per_missile_distances[launched_idx] = dists
            else:
                min_dist = self.cfg.region_span
        if return_per_missile_distances:
            return hit_any, min_dist, per_missile_distances
        return hit_any, min_dist

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------
    def _init_logging(self) -> None:
        self.episode_index += 1

        self.plane_global_id += 1
        plane_id = self.plane_global_id
        self._plane_name = f"plane_blue.{plane_id}.0"

        self._missile_tracks = []
        self._missile_names = []
        for i in range(self.cfg.num_missiles):
            self.missile_global_id += 1
            mid = self.missile_global_id
            # Encode launch step index (integer) into filename so ACMI can
            # start each missile at its launch time.
            launch_step = int(round(self.missile_launch_times[i] / self.cfg.dt))
            start_token = str(max(0, launch_step))
            name = f"missile_red.{mid}.{start_token}"
            self._missile_names.append(name)
            self._missile_tracks.append([])

        self._plane_track = []
        self._blue_action_log = []
        self._blue_action_name = f"plane_blue_actions.{plane_id}"
        self._analysis_log = []
        self._analysis_name = f"analysis_episode.{self.episode_index}"

    def _compute_orientation(self, vel: np.ndarray) -> Tuple[float, float, float]:
        vx, vy, vz = vel
        horiz = math.sqrt(vx * vx + vy * vy)
        yaw = math.degrees(math.atan2(vy, vx))
        pitch = math.degrees(math.atan2(vz, horiz))
        roll = 0.0
        return roll, pitch, yaw

    def _log_current_state(
        self,
        prev_blue_vel: np.ndarray | None = None,
        prev_missile_vel: np.ndarray | None = None,
    ) -> None:
        if self._plane_track is None or self._missile_tracks is None:
            return
        if prev_blue_vel is None:
            prev_blue_vel = self.blue_vel.copy()
        if prev_missile_vel is None:
            prev_missile_vel = self.missile_vel.copy()

        dt = max(self.cfg.dt, 1e-9)

        # Plane is always visible from t=0
        roll, pitch, yaw = self._compute_orientation(self.blue_vel)
        self._plane_track.append(
            [
                float(self.blue_pos[0]),
                float(self.blue_pos[1]),
                float(self.blue_pos[2]),
                roll,
                pitch,
                yaw,
            ]
        )

        if self._analysis_log is not None:
            blue_acc = (self.blue_vel - prev_blue_vel) / dt
            blue_speed = float(np.linalg.norm(self.blue_vel))
            blue_acc_norm = float(np.linalg.norm(blue_acc))
            self._analysis_log.append(
                [
                    float(self.time),
                    int(self.step_count),
                    "blue",
                    0,
                    1,
                    1,
                    float(self.blue_pos[0]),
                    float(self.blue_pos[1]),
                    float(self.blue_pos[2]),
                    float(self.blue_vel[0]),
                    float(self.blue_vel[1]),
                    float(self.blue_vel[2]),
                    blue_speed,
                    float(blue_acc[0]),
                    float(blue_acc[1]),
                    float(blue_acc[2]),
                    blue_acc_norm,
                    0.0,
                    float(self.prev_threat),
                    -1 if self._last_action is None else int(self._last_action),
                    -1,
                    -1,
                    0.0,
                ]
            )

        # Missiles are only logged *after* they have been launched,
        # so they are invisible in Tacview before launch.
        for i in range(self.cfg.num_missiles):
            # if not self.missile_launched[i]:
            #     continue
            m_pos = self.missile_pos[i]
            m_vel = self.missile_vel[i]
            if self.missile_launched[i]:
                roll_m, pitch_m, yaw_m = self._compute_orientation(m_vel)
                self._missile_tracks[i].append(
                    [
                        float(m_pos[0]),
                        float(m_pos[1]),
                        float(m_pos[2]),
                        roll_m,
                        pitch_m,
                        yaw_m,
                    ]
                )

            if self._analysis_log is not None:
                m_acc = (m_vel - prev_missile_vel[i]) / dt
                m_speed = float(np.linalg.norm(m_vel))
                m_acc_norm = float(np.linalg.norm(m_acc))
                dist_to_blue = float(np.linalg.norm(m_pos - self.blue_pos))
                self._analysis_log.append(
                    [
                        float(self.time),
                        int(self.step_count),
                        "missile",
                        int(i),
                        int(self.missile_launched[i]),
                        int(self.missile_alive[i]),
                        float(m_pos[0]),
                        float(m_pos[1]),
                        float(m_pos[2]),
                        float(m_vel[0]),
                        float(m_vel[1]),
                        float(m_vel[2]),
                        m_speed,
                        float(m_acc[0]),
                        float(m_acc[1]),
                        float(m_acc[2]),
                        m_acc_norm,
                        dist_to_blue,
                        float(self.prev_threat),
                        -1,
                        int(self.missile_fov_in_view[i]),
                        int(self.missile_is_closing[i]),
                        float(self.missile_closing_speed[i]),
                    ]
                )

    def _flush_logs_to_csv(self) -> None:
        if (
            self._plane_track is None
            or self._missile_tracks is None
            or self._plane_name is None
            or self._missile_names is None
        ):
            return

        ep_idx = self.episode_index

        if self._plane_track:
            write_csv(self.cfg.save_dir, self._plane_name, self._plane_track, episode_index=ep_idx)

        for track, name in zip(self._missile_tracks, self._missile_names):
            if track:
                write_csv(self.cfg.save_dir, name, track, episode_index=ep_idx)

        if self._blue_action_log and self._blue_action_name is not None:
            write_action_csv(
                self.cfg.save_dir,
                self._blue_action_name,
                self._blue_action_log,
                episode_index=ep_idx,
            )

        if self._analysis_log and self._analysis_name is not None:
            write_table_csv(
                self.cfg.save_dir,
                self._analysis_name,
                [
                    "time", "step", "entity_type", "entity_id", "launched", "alive",
                    "x", "y", "z", "vx", "vy", "vz", "speed",
                    "ax", "ay", "az", "accel_norm", "distance_to_blue", "threat", "action",
                    "fov_in_view", "is_closing", "closing_speed",
                ],
                self._analysis_log,
                episode_index=ep_idx,
            )

        self._plane_track = None
        self._missile_tracks = None
        self._plane_name = None
        self._missile_names = None
        self._blue_action_log = None
        self._blue_action_name = None
        self._analysis_log = None
        self._analysis_name = None
