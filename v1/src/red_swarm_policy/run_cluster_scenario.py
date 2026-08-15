from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .cli_utils import parse_float_pair, parse_float_range, parse_float_sequence
from .core.config import SwarmModelConfig
from .env import EnvironmentConfig, MissileConfig, RedAction, RedBlueEngagementEnv, RewardConfig, ScenarioConfig
from .policy.actor import OverloadBiasActor, TargetAssignmentActor
from .train_env import (
    CHECKPOINT_SCHEMA_VERSION,
    SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS,
    _load_torch_checkpoint,
    _select_device,
)
from .training.rollout import HierarchicalPolicyRuntime


@dataclass(frozen=True)
class ClusterTrajectory:
    seed: int
    time_step_s: float
    bias_update_interval_s: float
    assignment_update_interval_s: float
    blue_action_library_entry: int
    times_s: np.ndarray
    red_positions_m: np.ndarray
    blue_positions_m: np.ndarray
    red_alive: np.ndarray
    blue_alive: np.ndarray
    assignment_source: str
    initial_target_indices: np.ndarray
    assignment_matrices: np.ndarray
    guidance_bias_matrices: np.ndarray
    pn_load_body_g: np.ndarray
    bias_load_body_g: np.ndarray
    gravity_load_body_g: np.ndarray
    final_load_body_g: np.ndarray
    bias_update_times_s: np.ndarray
    assignment_update_times_s: np.ndarray
    boost_speeds_mps: np.ndarray
    physics_steps: int
    done: bool
    final_info: dict[str, Any]


def _integer_step_count(interval_s: float, time_step_s: float, name: str) -> int:
    if not math.isfinite(time_step_s) or time_step_s <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    if not math.isfinite(interval_s) or interval_s <= 0.0:
        raise ValueError(f"{name} must be positive")
    ratio = interval_s / time_step_s
    steps = int(round(ratio))
    if steps <= 0 or not math.isclose(ratio, steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError(f"{name} must be an integer multiple of time_step_s")
    return steps


def _capacity_aware_baseline_targets(
    red_count: int,
    blue_count: int,
    capacity: int,
    rng: np.random.Generator,
) -> np.ndarray:
    targets = np.full(red_count, -1, dtype=np.int64)
    assignable_count = min(red_count, blue_count * capacity)
    if assignable_count == 0:
        return targets
    red_order = rng.permutation(red_count)[:assignable_count]
    target_slots = np.repeat(np.arange(blue_count, dtype=np.int64), capacity)
    rng.shuffle(target_slots)
    targets[red_order] = target_slots[:assignable_count]
    return targets


def _snapshot(
    state,
) -> tuple[np.ndarray, ...]:
    return (
        np.stack([entity.position_m.copy() for entity in state.red]),
        np.stack([entity.position_m.copy() for entity in state.blue]),
        np.asarray([entity.alive for entity in state.red], dtype=bool),
        np.asarray([entity.alive for entity in state.blue], dtype=bool),
        np.stack([entity.guidance_bias.copy() for entity in state.red]),
        np.stack([entity.pn_load_body_g.copy() for entity in state.red]),
        np.stack([entity.bias_load_body_g.copy() for entity in state.red]),
        np.stack([entity.gravity_load_body_g.copy() for entity in state.red]),
        np.stack([entity.final_load_body_g.copy() for entity in state.red]),
    )


def run_cluster_scenario(
    config: EnvironmentConfig,
    *,
    seed: int,
    duration_s: float,
    bias_update_interval_s: float | None = None,
    assignment_update_interval_s: float | None = None,
    trajectory_sample_interval_s: float = 0.1,
    blue_action_library_entry: int = 1,
    assignment_actor: TargetAssignmentActor | None = None,
    execution_actor: OverloadBiasActor | None = None,
    deterministic: bool = True,
) -> ClusterTrajectory:
    config.validate()
    if bias_update_interval_s is None:
        bias_update_interval_s = config.bias_update_interval_s
    if assignment_update_interval_s is None:
        assignment_update_interval_s = config.assignment_update_interval_s
    if not math.isclose(
        bias_update_interval_s,
        config.bias_update_interval_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError("bias_update_interval_s must match config.bias_update_interval_s")
    if not math.isclose(
        assignment_update_interval_s,
        config.assignment_update_interval_s,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise ValueError(
            "assignment_update_interval_s must match config.assignment_update_interval_s"
        )
    duration_steps = _integer_step_count(duration_s, config.time_step_s, "duration_s")
    bias_update_steps = _integer_step_count(
        bias_update_interval_s,
        config.time_step_s,
        "bias_update_interval_s",
    )
    assignment_update_steps = _integer_step_count(
        assignment_update_interval_s,
        config.time_step_s,
        "assignment_update_interval_s",
    )
    trajectory_sample_steps = _integer_step_count(
        trajectory_sample_interval_s,
        config.time_step_s,
        "trajectory_sample_interval_s",
    )
    if not 1 <= blue_action_library_entry <= 29:
        raise ValueError("blue_action_library_entry must be in [1, 29]")
    if (assignment_actor is None) != (execution_actor is None):
        raise ValueError("assignment_actor and execution_actor must be provided together")

    policy_enabled = assignment_actor is not None
    if policy_enabled:
        assert assignment_actor is not None and execution_actor is not None
        if assignment_actor.config.max_missiles_per_target != config.scenario.max_missiles_per_target:
            raise ValueError("model and environment max_missiles_per_target must match")
        actor_device = next(assignment_actor.parameters()).device
        if next(execution_actor.parameters()).device != actor_device:
            raise ValueError("assignment_actor and execution_actor must use the same device")
        assignment_actor.eval()
        execution_actor.eval()
    else:
        actor_device = torch.device("cpu")

    env = RedBlueEngagementEnv(config, device=actor_device, record_replay=False)
    observation = env.reset(seed=seed, style="many_to_many")
    assert env.state is not None
    state = env.state
    if policy_enabled:
        initial_target_indices = np.full(len(state.red), -1, dtype=np.int64)
        assignment_source = "schema{}_actor".format(
            getattr(
                assignment_actor,
                "checkpoint_schema_version",
                CHECKPOINT_SCHEMA_VERSION,
            )
        )
        controller = HierarchicalPolicyRuntime(
            env,
            assignment_actor,
            execution_actor,
            deterministic=deterministic,
        )
        controller.reset(observation)
        red_action: RedAction | None = None
    else:
        target_rng = np.random.default_rng(np.random.SeedSequence([seed, 0xB1A5]))
        initial_target_indices = _capacity_aware_baseline_targets(
            len(state.red),
            len(state.blue),
            config.scenario.max_missiles_per_target,
            target_rng,
        )
        controller = None
        assignment_source = "seeded_zero_bias_baseline"
        red_action = RedAction(
            target_indices=initial_target_indices.copy(),
            guidance_bias=np.zeros((len(state.red), 2), dtype=np.float64),
        )

    red_positions: list[np.ndarray] = []
    blue_positions: list[np.ndarray] = []
    red_alive: list[np.ndarray] = []
    blue_alive: list[np.ndarray] = []
    guidance_bias_matrices: list[np.ndarray] = []
    pn_load_body_g: list[np.ndarray] = []
    bias_load_body_g: list[np.ndarray] = []
    gravity_load_body_g: list[np.ndarray] = []
    final_load_body_g: list[np.ndarray] = []
    assignment_matrices = [observation.assignment_matrix.copy()]
    times_s = [state.time_s]
    initial_snapshot = _snapshot(state)
    red_positions.append(initial_snapshot[0])
    blue_positions.append(initial_snapshot[1])
    red_alive.append(initial_snapshot[2])
    blue_alive.append(initial_snapshot[3])
    guidance_bias_matrices.append(initial_snapshot[4])
    pn_load_body_g.append(initial_snapshot[5])
    bias_load_body_g.append(initial_snapshot[6])
    gravity_load_body_g.append(initial_snapshot[7])
    final_load_body_g.append(initial_snapshot[8])

    bias_update_times_s: list[float] = []
    assignment_update_times_s: list[float] = []
    boost_speeds_mps = np.asarray(
        [np.linalg.norm(entity.velocity_mps) for entity in state.red],
        dtype=np.float64,
    )
    maximum_steps = min(
        max(duration_steps - config.policy_entry_steps, 0),
        config.policy_horizon_steps,
    )
    step = None

    for physics_step in range(maximum_steps):
        decision_request = env.next_decision_request()
        if decision_request.assignment_due:
            assignment_update_times_s.append(state.time_s)
            if policy_enabled:
                assert controller is not None
        if decision_request.bias_due:
            bias_update_times_s.append(state.time_s)
            if policy_enabled:
                assert controller is not None
                policy, _ = controller.action(observation)
                if (initial_target_indices < 0).all():
                    initial_target_indices = (
                        policy.assignment.actions.target[0]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.int64)
                        - 1
                    )
                red_action = policy

        if policy_enabled and red_action is None:
            raise RuntimeError("policy action was not initialized")

        if red_action is None:
            raise RuntimeError("red action was not initialized")
        step = env.step(red_action=red_action, blue_action=blue_action_library_entry - 1)
        if policy_enabled:
            assert controller is not None
            controller.observe(step)
        assert env.state is not None
        state = env.state
        observation = step.observation

        should_sample = state.step_count % trajectory_sample_steps == 0 or step.done or physics_step + 1 == maximum_steps
        if should_sample and not math.isclose(times_s[-1], state.time_s, rel_tol=0.0, abs_tol=1.0e-12):
            snapshot = _snapshot(state)
            times_s.append(state.time_s)
            red_positions.append(snapshot[0])
            blue_positions.append(snapshot[1])
            red_alive.append(snapshot[2])
            blue_alive.append(snapshot[3])
            guidance_bias_matrices.append(snapshot[4])
            pn_load_body_g.append(snapshot[5])
            bias_load_body_g.append(snapshot[6])
            gravity_load_body_g.append(snapshot[7])
            final_load_body_g.append(snapshot[8])
            assignment_matrices.append(step.assignment_matrix.copy())
        if step.done:
            break

    final_done = False if step is None else step.done
    final_info = {
        "boost_warmup": False,
        "network_entry_reached": True,
        **env.policy_status(),
    } if step is None else {**dict(step.info), "policy_status": env.policy_status()}
    return ClusterTrajectory(
        seed=seed,
        time_step_s=config.time_step_s,
        bias_update_interval_s=bias_update_interval_s,
        assignment_update_interval_s=assignment_update_interval_s,
        blue_action_library_entry=blue_action_library_entry,
        times_s=np.asarray(times_s, dtype=np.float64),
        red_positions_m=np.stack(red_positions),
        blue_positions_m=np.stack(blue_positions),
        red_alive=np.stack(red_alive),
        blue_alive=np.stack(blue_alive),
        assignment_source=assignment_source,
        initial_target_indices=initial_target_indices,
        assignment_matrices=np.stack(assignment_matrices),
        guidance_bias_matrices=np.stack(guidance_bias_matrices),
        pn_load_body_g=np.stack(pn_load_body_g),
        bias_load_body_g=np.stack(bias_load_body_g),
        gravity_load_body_g=np.stack(gravity_load_body_g),
        final_load_body_g=np.stack(final_load_body_g),
        bias_update_times_s=np.asarray(bias_update_times_s, dtype=np.float64),
        assignment_update_times_s=np.asarray(
            assignment_update_times_s,
            dtype=np.float64,
        ),
        boost_speeds_mps=boost_speeds_mps,
        physics_steps=state.step_count,
        done=final_done,
        final_info=final_info,
    )


def _animation_frame_indices(times_s: np.ndarray, frame_interval_s: float) -> list[int]:
    if frame_interval_s <= 0.0:
        raise ValueError("frame_interval_s must be positive")
    indices = [0]
    next_time_s = frame_interval_s
    for index, time_s in enumerate(times_s[1:], start=1):
        if time_s + 1.0e-12 >= next_time_s:
            indices.append(index)
            while time_s + 1.0e-12 >= next_time_s:
                next_time_s += frame_interval_s
    if indices[-1] != len(times_s) - 1:
        indices.append(len(times_s) - 1)
    return indices


def render_trajectory_gif(
    trajectory: ClusterTrajectory,
    output_path: Path,
    *,
    frame_interval_s: float = 1.0,
    fps: int = 10,
) -> int:
    if fps <= 0:
        raise ValueError("fps must be positive")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    frame_indices = _animation_frame_indices(trajectory.times_s, frame_interval_s)
    red_km = trajectory.red_positions_m / 1000.0
    blue_km = trajectory.blue_positions_m / 1000.0
    all_positions = np.concatenate([red_km, blue_km], axis=1)
    north = all_positions[..., 0]
    east = all_positions[..., 2]
    up = all_positions[..., 1]

    def limits(values: np.ndarray, minimum_padding: float) -> tuple[float, float]:
        lower = float(np.min(values))
        upper = float(np.max(values))
        padding = max(minimum_padding, 0.05 * max(upper - lower, minimum_padding))
        return lower - padding, upper + padding

    north_limits = limits(north, 5.0)
    east_limits = limits(east, 5.0)
    up_limits = limits(up, 1.0)
    figure = plt.figure(figsize=(13.0, 7.0), dpi=100)
    axis = figure.add_subplot(121, projection="3d")
    load_axis = figure.add_subplot(122)
    axis.set_xlim(*north_limits)
    axis.set_ylim(*east_limits)
    axis.set_zlim(*up_limits)
    axis.set_xlabel("North (km)")
    axis.set_ylabel("East (km)")
    axis.set_zlabel("Up (km)")
    axis.view_init(elev=27.0, azim=-58.0)
    axis.grid(True, alpha=0.25)
    horizontal_aspect = max((north_limits[1] - north_limits[0]) / (east_limits[1] - east_limits[0]), 1.0)
    axis.set_box_aspect((horizontal_aspect, 1.0, 0.45))
    axis.xaxis.set_major_locator(MaxNLocator(7))
    axis.yaxis.set_major_locator(MaxNLocator(7))
    axis.zaxis.set_major_locator(MaxNLocator(5))

    red_colors = plt.cm.autumn(np.linspace(0.05, 0.85, red_km.shape[1]))
    blue_colors = plt.cm.winter(np.linspace(0.15, 0.85, blue_km.shape[1]))
    red_lines = [axis.plot([], [], [], color=red_colors[index], linewidth=1.1, alpha=0.7)[0] for index in range(red_km.shape[1])]
    red_markers = [axis.plot([], [], [], marker="o", color=red_colors[index], markersize=3.0)[0] for index in range(red_km.shape[1])]
    blue_lines = [axis.plot([], [], [], color=blue_colors[index], linewidth=2.0, alpha=0.9)[0] for index in range(blue_km.shape[1])]
    blue_markers = [axis.plot([], [], [], marker="^", color=blue_colors[index], markersize=7.0)[0] for index in range(blue_km.shape[1])]
    axis.scatter(red_km[0, :, 0], red_km[0, :, 2], red_km[0, :, 1], c=red_colors, marker=".", s=12, alpha=0.6)
    axis.scatter(blue_km[0, :, 0], blue_km[0, :, 2], blue_km[0, :, 1], c=blue_colors, marker="^", s=36, alpha=0.8)
    axis.legend(
        handles=[
            Line2D([0], [0], color="#d62728", label=f"Red missiles ({red_km.shape[1]})"),
            Line2D([0], [0], color="#1565c0", label=f"Blue aircraft ({blue_km.shape[1]})"),
        ],
        loc="upper left",
    )
    load_axis.set_xlim(float(trajectory.times_s[0]), max(float(trajectory.times_s[-1]), 1.0e-6))

    def active_mean_lateral_load(load_body_g: np.ndarray) -> np.ndarray:
        lateral_norm = np.linalg.norm(load_body_g[..., 1:], axis=-1)
        active = trajectory.red_alive.astype(np.float64)
        return (lateral_norm * active).sum(axis=-1) / active.sum(axis=-1).clip(min=1.0)

    pn_norm = active_mean_lateral_load(trajectory.pn_load_body_g)
    bias_norm = active_mean_lateral_load(trajectory.bias_load_body_g)
    gravity_norm = active_mean_lateral_load(trajectory.gravity_load_body_g)
    final_norm = active_mean_lateral_load(trajectory.final_load_body_g)
    maximum_load = max(
        1.0,
        float(np.max(np.stack([pn_norm, bias_norm, gravity_norm, final_norm]))),
    )
    load_axis.set_ylim(0.0, 1.05 * maximum_load)
    load_axis.set_xlabel("Time (s)")
    load_axis.set_ylabel("Mean lateral load (g)")
    load_axis.grid(True, alpha=0.25)
    pn_line = load_axis.plot([], [], color="#1f77b4", linewidth=1.8, label="PN")[0]
    bias_line = load_axis.plot([], [], color="#ff7f0e", linewidth=1.8, label="Execution bias")[0]
    gravity_line = load_axis.plot([], [], color="#9467bd", linewidth=1.8, label="Gravity compensation")[0]
    final_line = load_axis.plot([], [], color="#2ca02c", linewidth=2.2, label="Final command")[0]
    load_axis.legend(loc="upper right")

    def update(frame_number: int):
        sample_index = frame_indices[frame_number]
        for entity_index, (line, marker) in enumerate(zip(red_lines, red_markers)):
            path = red_km[: sample_index + 1, entity_index]
            line.set_data_3d(path[:, 0], path[:, 2], path[:, 1])
            marker.set_data_3d([path[-1, 0]], [path[-1, 2]], [path[-1, 1]])
            marker.set_alpha(1.0 if trajectory.red_alive[sample_index, entity_index] else 0.2)
        for entity_index, (line, marker) in enumerate(zip(blue_lines, blue_markers)):
            path = blue_km[: sample_index + 1, entity_index]
            line.set_data_3d(path[:, 0], path[:, 2], path[:, 1])
            marker.set_data_3d([path[-1, 0]], [path[-1, 2]], [path[-1, 1]])
            marker.set_alpha(1.0 if trajectory.blue_alive[sample_index, entity_index] else 0.2)
        axis.set_title(
            f"3-DoF engagement  t={trajectory.times_s[sample_index]:.1f} s  "
            f"active red={int(trajectory.red_alive[sample_index].sum())}  "
            f"active blue={int(trajectory.blue_alive[sample_index].sum())}"
        )
        shown = slice(0, sample_index + 1)
        pn_line.set_data(trajectory.times_s[shown], pn_norm[shown])
        bias_line.set_data(trajectory.times_s[shown], bias_norm[shown])
        gravity_line.set_data(trajectory.times_s[shown], gravity_norm[shown])
        final_line.set_data(trajectory.times_s[shown], final_norm[shown])
        load_axis.set_title(
            f"Execution-layer control  mean final={final_norm[sample_index]:.2f} g"
        )
        return [
            *red_lines,
            *red_markers,
            *blue_lines,
            *blue_markers,
            pn_line,
            bias_line,
            gravity_line,
            final_line,
        ]

    animation = FuncAnimation(figure, update, frames=len(frame_indices), interval=1000.0 / fps, blit=False)
    figure.subplots_adjust(left=0.02, right=0.98, bottom=0.10, top=0.91, wspace=0.20)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(figure)
    return len(frame_indices)


def _build_environment_config(args: argparse.Namespace) -> EnvironmentConfig:
    maximum_steps = _integer_step_count(args.duration_s, args.time_step_s, "duration_s")
    scenario = ScenarioConfig(
        red_count=args.red_count,
        blue_count=args.blue_count,
        max_missiles_per_target=args.max_missiles_per_target,
        red_launch_mach_range=parse_float_range(args.red_launch_mach_range, "red_launch_mach_range"),
        red_altitude_range_m=parse_float_range(args.red_altitude_range_m, "red_altitude_range_m"),
        blue_speed_range_mps=parse_float_range(args.blue_speed_range_mps, "blue_speed_range_mps"),
        blue_altitude_range_m=parse_float_range(args.blue_altitude_range_m, "blue_altitude_range_m"),
        speed_of_sound_mps=args.speed_of_sound_mps,
        blue_cluster_center_ne_m=parse_float_pair(args.blue_cluster_center_ne_m, "blue_cluster_center_ne_m"),
        blue_cluster_radius_m=args.blue_cluster_radius_m,
        blue_heading_range_deg=parse_float_range(
            args.blue_heading_range_deg,
            "blue_heading_range_deg",
            positive=False,
        ),
        red_cluster_radius_range_m=parse_float_range(
            args.red_cluster_radius_range_m,
            "red_cluster_radius_range_m",
        ),
        red_sector_center_azimuth_deg=args.red_sector_center_azimuth_deg,
        red_sector_width_deg=args.red_sector_width_deg,
        red_heading_bias_max_deg=args.red_heading_bias_max_deg,
        position_perturb_m=args.position_perturb_m,
        velocity_perturb_mps=args.velocity_perturb_mps,
    )
    missile = MissileConfig(
        boost_duration_s=args.missile_boost_duration_s,
        boost_target_mach_number=args.missile_max_mach,
        reference_speed_of_sound_mps=args.speed_of_sound_mps,
        boost_climb_angle_deg=args.missile_boost_climb_angle_deg,
        boost_pitch_transition_s=args.missile_boost_pitch_transition_s,
        boost_pitch_tracking_gain=args.missile_boost_pitch_tracking_gain,
        drag_coefficient=args.missile_drag_coefficient,
        drag_mach_breakpoints=parse_float_sequence(
            args.missile_drag_mach_breakpoints,
            "missile_drag_mach_breakpoints",
            minimum_length=2,
        ),
        zero_lift_drag_coefficients=parse_float_sequence(
            args.missile_zero_lift_drag_coefficients,
            "missile_zero_lift_drag_coefficients",
            minimum_length=2,
        ),
        induced_drag_factor=args.missile_induced_drag_factor,
        max_guidance_bias_g=args.missile_max_guidance_bias_g,
        proportional_navigation_gain=args.proportional_navigation_gain,
        max_guidance_time_s=args.missile_max_guidance_time_s,
    )
    return EnvironmentConfig(
        time_step_s=args.time_step_s,
        bias_update_interval_s=args.bias_update_interval_s,
        assignment_update_interval_s=args.assignment_update_interval_s,
        max_steps=maximum_steps,
        policy_start_mode="post_boost",
        policy_entry_speed_tolerance_ratio=args.policy_entry_speed_tolerance_ratio,
        policy_entry_flight_path_tolerance_deg=args.policy_entry_flight_path_tolerance_deg,
        scenario=scenario,
        missile=missile,
        reward=RewardConfig(
            high_damage_weight=args.high_damage_weight,
            high_waste_weight=args.high_waste_weight,
            high_potential_weight=args.high_potential_weight,
            high_potential_gamma=args.high_potential_gamma,
            high_time_penalty_per_s=args.high_time_penalty_per_s,
            high_time_margin_scale_s=args.high_time_margin_scale_s,
            terminal_success_reward=args.terminal_success_reward,
            terminal_failure_penalty=args.terminal_failure_penalty,
            terminal_timeout_penalty=args.terminal_timeout_penalty,
            low_damage_weight=args.low_damage_weight,
            low_potential_weight=args.low_potential_weight,
            low_potential_gamma=args.low_potential_gamma,
            low_missile_failure_penalty=args.low_missile_failure_penalty,
            low_load_penalty=args.low_load_penalty,
            low_smooth_penalty=args.low_smooth_penalty,
            zem_reference_range_m=args.zem_reference_range_m,
            zem_floor_range_m=args.zem_floor_range_m,
            zem_weight=args.zem_weight,
            seeker_lock_weight=args.seeker_lock_weight,
            smooth_bias_denominator=args.smooth_bias_denominator,
            zem_time_gate_scale_s=args.zem_time_gate_scale_s,
            assignment_min_energy_fraction=args.assignment_min_energy_fraction,
            assignment_min_available_load_fraction=args.assignment_min_available_load_fraction,
            assignment_correlation_weight=args.assignment_correlation_weight,
            assignment_correlation_angle_scale_deg=args.assignment_correlation_angle_scale_deg,
            assignment_correlation_time_scale_s=args.assignment_correlation_time_scale_s,
        ),
    )


def _load_policy_actors(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[TargetAssignmentActor, OverloadBiasActor]:
    checkpoint = _load_torch_checkpoint(checkpoint_path)
    schema_version = checkpoint.get("schema_version")
    if (
        not isinstance(schema_version, int)
        or isinstance(schema_version, bool)
        or schema_version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"checkpoint schema {schema_version} is incompatible with supported "
            f"schemas {SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS}"
        )
    model_data = checkpoint.get("model_config")
    if not isinstance(model_data, dict):
        raise ValueError("checkpoint model_config must be a dict")
    model_config = SwarmModelConfig(**model_data)
    assignment_actor = TargetAssignmentActor(model_config).to(device)
    execution_actor = OverloadBiasActor(model_config).to(device)
    assignment_actor.load_state_dict(checkpoint["assignment_actor"])
    execution_actor.load_state_dict(checkpoint["execution_actor"])
    assignment_actor.checkpoint_schema_version = schema_version
    execution_actor.checkpoint_schema_version = schema_version
    assignment_actor.eval()
    execution_actor.eval()
    return assignment_actor, execution_actor


def _write_metrics(
    path: Path,
    config: EnvironmentConfig,
    trajectory: ClusterTrajectory,
    gif_path: Path,
    gif_frames: int,
    checkpoint_path: Path | None,
) -> None:
    boost_speeds = trajectory.boost_speeds_mps
    metrics = {
        "seed": trajectory.seed,
        "environment_config": asdict(config),
        "physics_steps": trajectory.physics_steps,
        "simulation_time_s": float(trajectory.times_s[-1]),
        "launch_time_s": float(trajectory.times_s[-1]),
        "policy_control_time_s": float(
            max(trajectory.times_s[-1] - config.policy_entry_time_s, 0.0)
        ),
        "network_entry_time_s": config.policy_entry_time_s,
        "network_entry_step": config.policy_entry_steps,
        "pn_update_interval_s": trajectory.time_step_s,
        "bias_update_interval_s": trajectory.bias_update_interval_s,
        "bias_update_count": int(trajectory.bias_update_times_s.size),
        "assignment_update_interval_s": trajectory.assignment_update_interval_s,
        "assignment_update_count": int(trajectory.assignment_update_times_s.size),
        "blue_action_library_entry": trajectory.blue_action_library_entry,
        "blue_action_api_index": trajectory.blue_action_library_entry - 1,
        "assignment_source": trajectory.assignment_source,
        "policy_checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "initial_target_indices": trajectory.initial_target_indices.tolist(),
        "assignment_matrices": trajectory.assignment_matrices.astype(np.uint8).tolist(),
        "final_assignment_matrix": trajectory.assignment_matrices[-1].astype(int).tolist(),
        "assignment_matrix_shape": list(trajectory.assignment_matrices.shape),
        "guidance_bias_matrix_shape": list(trajectory.guidance_bias_matrices.shape),
        "pn_load_body_g_shape": list(trajectory.pn_load_body_g.shape),
        "bias_load_body_g_shape": list(trajectory.bias_load_body_g.shape),
        "gravity_load_body_g_shape": list(trajectory.gravity_load_body_g.shape),
        "final_load_body_g_shape": list(trajectory.final_load_body_g.shape),
        "guidance_bias_matrices": trajectory.guidance_bias_matrices.tolist(),
        "pn_load_body_g": trajectory.pn_load_body_g.tolist(),
        "bias_load_body_g": trajectory.bias_load_body_g.tolist(),
        "gravity_load_body_g": trajectory.gravity_load_body_g.tolist(),
        "final_load_body_g": trajectory.final_load_body_g.tolist(),
        "boost_speed_min_mps": float(np.min(boost_speeds)) if boost_speeds.size else None,
        "boost_speed_max_mps": float(np.max(boost_speeds)) if boost_speeds.size else None,
        "boost_target_speed_mps": config.missile.max_speed_mps,
        "done": trajectory.done,
        "final_info": trajectory.final_info,
        "gif_path": str(gif_path),
        "gif_frames": gif_frames,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, ensure_ascii=True, separators=(",", ":")) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the 3-blue/24-red cluster engagement and render all trajectories.")
    parser.add_argument("--output", default="outputs/trajectory_3blue_24red.gif")
    parser.add_argument("--metrics-path", default="outputs/trajectory_3blue_24red_metrics.json")
    parser.add_argument("--checkpoint", default=None, help="Schema 10 checkpoint used by both actor networks.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--stochastic", action="store_true", help="Sample actor outputs instead of using deterministic modes.")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--duration-s", type=float, default=180.0)
    parser.add_argument("--time-step-s", type=float, default=0.005)
    parser.add_argument("--bias-update-interval-s", type=float, default=0.1)
    parser.add_argument("--assignment-update-interval-s", type=float, default=5.0)
    parser.add_argument("--trajectory-sample-interval-s", type=float, default=0.1)
    parser.add_argument("--gif-frame-interval-s", type=float, default=1.0)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--blue-action-library-entry", type=int, default=1)
    parser.add_argument("--red-count", type=int, default=24)
    parser.add_argument("--blue-count", type=int, default=3)
    parser.add_argument("--max-missiles-per-target", type=int, default=4)
    parser.add_argument("--red-launch-mach-range", default="0.6,0.9")
    parser.add_argument("--red-altitude-range-m", default="8000,10000")
    parser.add_argument("--blue-speed-range-mps", default="300,400")
    parser.add_argument("--blue-altitude-range-m", default="8000,12000")
    parser.add_argument("--speed-of-sound-mps", type=float, default=295.0)
    parser.add_argument("--missile-boost-duration-s", type=float, default=7.0)
    parser.add_argument("--missile-max-mach", type=float, default=6.0)
    parser.add_argument("--policy-entry-speed-tolerance-ratio", type=float, default=1.0e-6)
    parser.add_argument("--policy-entry-flight-path-tolerance-deg", type=float, default=0.5)
    parser.add_argument("--missile-boost-climb-angle-deg", type=float, default=20.0)
    parser.add_argument("--missile-boost-pitch-transition-s", type=float, default=2.0)
    parser.add_argument("--missile-boost-pitch-tracking-gain", type=float, default=2.0)
    parser.add_argument("--missile-drag-coefficient", type=float, default=None)
    parser.add_argument(
        "--missile-drag-mach-breakpoints",
        default="0,0.8,0.95,1.05,1.2,2,3,4,5,6,8",
    )
    parser.add_argument(
        "--missile-zero-lift-drag-coefficients",
        default="0.10,0.11,0.18,0.34,0.30,0.22,0.19,0.17,0.16,0.15,0.15",
    )
    parser.add_argument("--missile-induced-drag-factor", type=float, default=0.08)
    parser.add_argument("--missile-max-guidance-bias-g", type=float, default=5.0)
    parser.add_argument("--proportional-navigation-gain", type=float, default=3.5)
    parser.add_argument("--missile-max-guidance-time-s", type=float, default=180.0)
    parser.add_argument("--blue-cluster-center-ne-m", default="0,0")
    parser.add_argument("--blue-cluster-radius-m", type=float, default=20000.0)
    parser.add_argument("--blue-heading-range-deg", default="-180,180")
    parser.add_argument("--red-cluster-radius-range-m", default="140000,160000")
    parser.add_argument("--red-sector-center-azimuth-deg", type=float, default=180.0)
    parser.add_argument("--red-sector-width-deg", type=float, default=60.0)
    parser.add_argument("--red-heading-bias-max-deg", type=float, default=15.0)
    parser.add_argument("--position-perturb-m", type=float, default=0.0)
    parser.add_argument("--velocity-perturb-mps", type=float, default=0.0)
    parser.add_argument("--high-damage-weight", type=float, default=512.0)
    parser.add_argument("--high-waste-weight", type=float, default=64.0)
    parser.add_argument("--high-potential-weight", type=float, default=1.0)
    parser.add_argument("--high-potential-gamma", type=float, default=1.0)
    parser.add_argument("--high-time-penalty-per-s", type=float, default=2.0)
    parser.add_argument("--high-time-margin-scale-s", type=float, default=10.0)
    parser.add_argument("--terminal-success-reward", type=float, default=0.0)
    parser.add_argument("--terminal-failure-penalty", type=float, default=0.0)
    parser.add_argument("--terminal-timeout-penalty", type=float, default=0.0)
    parser.add_argument("--low-damage-weight", type=float, default=512.0)
    parser.add_argument("--low-potential-weight", type=float, default=1.0)
    parser.add_argument("--low-potential-gamma", type=float, default=1.0)
    parser.add_argument("--low-missile-failure-penalty", type=float, default=64.0)
    parser.add_argument("--low-load-penalty", type=float, default=0.0008)
    parser.add_argument("--low-smooth-penalty", type=float, default=0.0002)
    parser.add_argument("--zem-reference-range-m", type=float, default=1000.0)
    parser.add_argument("--zem-floor-range-m", type=float, default=5.0)
    parser.add_argument("--zem-weight", type=float, default=0.6)
    parser.add_argument("--seeker-lock-weight", type=float, default=0.2)
    parser.add_argument("--smooth-bias-denominator", type=float, default=8.0)
    parser.add_argument("--zem-time-gate-scale-s", type=float, default=1.0)
    parser.add_argument("--assignment-min-energy-fraction", type=float, default=0.05)
    parser.add_argument("--assignment-min-available-load-fraction", type=float, default=0.05)
    parser.add_argument("--assignment-correlation-weight", type=float, default=0.5)
    parser.add_argument("--assignment-correlation-angle-scale-deg", type=float, default=15.0)
    parser.add_argument("--assignment-correlation-time-scale-s", type=float, default=5.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stochastic and args.checkpoint is None:
        parser.error("--stochastic requires --checkpoint")
    config = _build_environment_config(args)
    checkpoint_path = None if args.checkpoint is None else Path(args.checkpoint)
    assignment_actor = None
    execution_actor = None
    device_label = "cpu (seeded zero-bias baseline)"
    if checkpoint_path is not None:
        device, device_label = _select_device(args.device)
        assignment_actor, execution_actor = _load_policy_actors(checkpoint_path, device)
    trajectory = run_cluster_scenario(
        config,
        seed=args.seed,
        duration_s=args.duration_s,
        bias_update_interval_s=args.bias_update_interval_s,
        assignment_update_interval_s=args.assignment_update_interval_s,
        trajectory_sample_interval_s=args.trajectory_sample_interval_s,
        blue_action_library_entry=args.blue_action_library_entry,
        assignment_actor=assignment_actor,
        execution_actor=execution_actor,
        deterministic=not args.stochastic,
    )
    output_path = Path(args.output)
    gif_frames = render_trajectory_gif(
        trajectory,
        output_path,
        frame_interval_s=args.gif_frame_interval_s,
        fps=args.gif_fps,
    )
    _write_metrics(
        Path(args.metrics_path),
        config,
        trajectory,
        output_path,
        gif_frames,
        checkpoint_path,
    )
    print(
        json.dumps(
            {
                "event": "trajectory_gif",
                "path": str(output_path),
                "frames": gif_frames,
                "physics_steps": trajectory.physics_steps,
                "simulation_time_s": float(trajectory.times_s[-1]),
                "assignment_source": trajectory.assignment_source,
                "device": device_label,
            },
            ensure_ascii=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
