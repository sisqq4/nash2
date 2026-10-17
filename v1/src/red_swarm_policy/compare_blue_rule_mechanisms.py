"""Search and record paired Blue rule/mechanism/Rainbow comparisons.

Every policy variant is reset from the same ``(seed, missile_count)`` pair and
the serialized initialization is fingerprinted before the comparison is
accepted.  The four evaluation-only mechanism rules are applied one at a time
to the rule machine's native 29-action score vector.  The final Rainbow case
uses the checkpoint exactly as deployed: its learned policy and saved flight
envelope, without re-applying evaluation-only mechanism shaping.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

from .blue_rl import (
    BlueEscapeEnv,
    BlueEscapeEnvConfig,
    EvaluationActionShaper,
    EvaluationShapingConfig,
    FlightEnvelopeConfig,
    FlightEnvelopeConstraintLayer,
    FlightQualityTracker,
    MechanismRewardConfig,
    RainbowDQNAgent,
    append_flight_quality_episode,
    blue_observation_dim,
)
from .blue_rl.config_io import configure_blue_mission_duration, load_environment_config
from .cli_utils import BLUE_MAX_MISSILE_SCENARIO, parse_missile_scenarios
from .env import BlueEvasionConfig, BlueEvasionRuleMachine, GUIDANCE_CONTRACT


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT
    / "outputs"
    / "blue_rl"
    / "train_1v1-1v3_train_1v1-1v3_missle_around_60_ep2000"
    / "best_checkpoints"
    / "blue_rainbow_ep001660.pt"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "blue_rl" / "rule_mechanism_rl_comparison_ep1660"
MECHANISMS = ("threat", "timing", "direction", "overload")
VARIANT_ORDER = (
    "rule_raw",
    "rule_threat",
    "rule_timing",
    "rule_direction",
    "rule_overload",
    "rainbow_full",
)


@dataclass(frozen=True)
class VariantSpec:
    name: str
    policy: str
    mechanism: str | None = None


@dataclass
class EpisodeRun:
    summary: dict[str, Any]
    trajectory: dict[str, Any] | None
    diagnostics: list[dict[str, Any]]


def variant_specs() -> tuple[VariantSpec, ...]:
    return (
        VariantSpec("rule_raw", "rule"),
        *(VariantSpec(f"rule_{name}", "rule", name) for name in MECHANISMS),
        VariantSpec("rainbow_full", "rainbow"),
    )


def build_candidate_plan(
    seed_start: int,
    max_seeds: int,
    missile_counts: tuple[int, ...],
) -> list[tuple[int, int]]:
    if seed_start < 0 or max_seeds < 1:
        raise ValueError("seed-start must be non-negative and max-seeds must be positive")
    return [
        (seed, missile_count)
        for seed in range(seed_start, seed_start + max_seeds)
        for missile_count in missile_counts
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find identical multi-missile scenarios where the native Rainbow policy "
            "survives and the unmodified Blue rule machine does not, then record the "
            "raw rule, four single-mechanism rule variants, and Rainbow trajectories."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed-start", type=int, default=10089)
    parser.add_argument("--max-seeds", type=int, default=100)
    parser.add_argument("--matches", type=int, default=1)
    parser.add_argument(
        "--missiles",
        default="2,3",
        help="Comma-separated multi-missile scenarios; the checkpoint supports 2 and 3",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--red-sector-width-deg", type=float, default=None)
    parser.add_argument(
        "--blue-rule-execution-backend",
        choices=("reference", "vectorized_guarded", "shadow"),
        default="vectorized_guarded",
    )
    parser.add_argument("--mechanism-weight", type=float, default=0.35)
    for name in MECHANISMS:
        parser.add_argument(f"--mechanism-{name}-weight", type=float, default=1.0)
    parser.add_argument(
        "--no-acmi",
        action="store_true",
        help="Do not save the optional physics-step ACMI files (JSON/CSV traces are always saved)",
    )
    return parser


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_json(value: object) -> str:
    return json.dumps(
        _jsonable(value), ensure_ascii=True, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    )


def initialization_fingerprint(initialization: dict[str, object]) -> str:
    return hashlib.sha256(_canonical_json(initialization).encode("ascii")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_missile_slots(schema: str, observation_dim: int) -> int:
    matches = [
        slots for slots in range(1, BLUE_MAX_MISSILE_SCENARIO + 1)
        if blue_observation_dim(schema, slots) == observation_dim
    ]
    if len(matches) != 1:
        raise ValueError(
            f"checkpoint schema/dimension ({schema}, {observation_dim}) does not identify "
            "exactly one supported missile-slot count"
        )
    return matches[0]


def is_target_outcome(rule_summary: dict[str, Any], rl_summary: dict[str, Any]) -> bool:
    return not bool(rule_summary["blue_survived"]) and bool(rl_summary["blue_survived"])


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(_jsonable(value), ensure_ascii=True, allow_nan=False) + "\n")
        stream.flush()


def _mechanism_config(args: argparse.Namespace, name: str) -> EvaluationShapingConfig:
    switches = {mechanism: mechanism == name for mechanism in MECHANISMS}
    return EvaluationShapingConfig(
        **switches,
        weight=args.mechanism_weight,
        threat_weight=args.mechanism_threat_weight,
        timing_weight=args.mechanism_timing_weight,
        direction_weight=args.mechanism_direction_weight,
        overload_weight=args.mechanism_overload_weight,
    )


def _merge_reasons(*groups: Iterable[object]) -> list[str]:
    return list(dict.fromkeys(str(item) for group in groups for item in group))


def _environment_adapter(
    *,
    slots: int,
    observation_schema: str,
    decision_interval_s: float,
    mechanism_reward: MechanismRewardConfig,
    record_acmi: bool,
    acmi_directory: Path,
) -> BlueEscapeEnvConfig:
    return BlueEscapeEnvConfig(
        missile_count=2,
        max_missiles=slots,
        pad_observation_to_max_missiles=True,
        observation_schema=observation_schema,
        decision_interval_s=decision_interval_s,
        expose_evaluation_mechanism_state=True,
        record_red_telemetry=True,
        record_acmi=record_acmi,
        acmi_episode_interval=1 if record_acmi else 0,
        acmi_directory=str(acmi_directory),
        mechanism_reward=mechanism_reward,
    )


def _run_episode(
    *,
    spec: VariantSpec,
    seed: int,
    missile_count: int,
    environment_config: Any,
    adapter_config: BlueEscapeEnvConfig,
    agent: RainbowDQNAgent,
    envelope_config: FlightEnvelopeConfig,
    rule_backend: str,
    shaping_config_factory: Callable[[str], EvaluationShapingConfig],
    capture_trajectory: bool,
) -> EpisodeRun:
    env = BlueEscapeEnv(environment_config, adapter_config)
    observation, reset_info = env.reset(
        seed=seed, episode_index=1, missile_count=missile_count
    )
    initialization = dict(reset_info["initialization"])
    fingerprint = initialization_fingerprint(initialization)
    current_info: dict[str, Any] = dict(reset_info)
    rule = (
        BlueEvasionRuleMachine(
            environment_config,
            BlueEvasionConfig(decision_interval_s=adapter_config.decision_interval_s),
            execution_backend=rule_backend,
        )
        if spec.policy == "rule" else None
    )
    shaper = (
        EvaluationActionShaper(shaping_config_factory(spec.mechanism))
        if spec.mechanism is not None else None
    )
    # The rule ablations isolate EvaluationActionShaper itself.  They must not
    # silently inherit a checkpoint-specific envelope.  The native Rainbow
    # deployment remains the only case that restores that saved constraint.
    constraint = (
        FlightEnvelopeConstraintLayer(envelope_config)
        if spec.policy == "rainbow" else None
    )
    tracker = FlightQualityTracker() if capture_trajectory else None
    if tracker is not None:
        tracker.add(dict(reset_info["flight_quality_state"]))

    total_reward = 0.0
    decision_steps = 0
    raw_actions: Counter[int] = Counter()
    executed_actions: Counter[int] = Counter()
    mode_counts: Counter[str] = Counter()
    reward_components: Counter[str] = Counter()
    diagnostics: list[dict[str, Any]] = []
    intervention_count = 0
    mechanism_intervention_count = 0
    envelope_intervention_count = 0
    last_info: dict[str, Any] = current_info

    while True:
        learning_active = bool(current_info["learning_active"])
        snapshot = dict(current_info["mechanism_state"])
        action = 0
        policy_action = 0
        action_mask: np.ndarray | None = None
        fallback_required = False
        diagnostic: dict[str, Any] = {
            "phase": "pre_detection",
            "raw_action": 0,
            "executed_action": 0,
            "intervened": False,
        }

        if learning_active and spec.policy == "rainbow":
            q_values = agent.expected_action_values(observation, evaluation=True)[0]
            policy_action = int(np.argmax(q_values))
            assert constraint is not None
            action, envelope = constraint.select(q_values, snapshot)
            action_mask = np.asarray(envelope["effective_action_mask"], dtype=bool)
            fallback_required = bool(envelope["fallback_required"])
            diagnostic = {
                "policy": "rainbow_checkpoint_native",
                "evaluation_mechanisms": [],
                "raw_action": policy_action,
                "executed_action": int(action),
                "intervened": int(action) != policy_action,
                "envelope": envelope,
            }
            envelope_intervention_count += int(int(action) != policy_action)

        elif learning_active and spec.policy == "rule":
            assert rule is not None
            if shaper is None:
                decision = rule.decide(env.inner.state)  # type: ignore[arg-type]
                action = policy_action = int(decision.action_indices[0])
                mode_counts.update(decision.modes)
                diagnostic = {
                    "policy": "rule_raw",
                    "evaluation_mechanisms": [],
                    "raw_action": policy_action,
                    "executed_action": action,
                    "intervened": False,
                }
            else:
                selection: dict[str, Any] = {}

                def action_selector(scores: np.ndarray, blue_index: int) -> int:
                    if blue_index != 0:
                        raise RuntimeError("paired Blue comparison supports one Blue aircraft")
                    raw_rule_action = int(np.argmax(scores))
                    mechanism_action, mechanism = shaper.select(scores, snapshot)
                    selection.update({
                        "raw_rule_action": raw_rule_action,
                        "mechanism_action": int(mechanism_action),
                        "mechanism": mechanism,
                    })
                    return int(mechanism_action)

                decision = rule.decide(  # type: ignore[arg-type]
                    env.inner.state, action_selector=action_selector
                )
                mode_counts.update(decision.modes)
                if not selection:
                    # BlueEscapeEnv stays active after first detection, whereas
                    # the rule machine deliberately ignores a receding,
                    # unlocked missile.  In that state there is no 29-score
                    # vector to shape, so preserve its recovery/cruise action.
                    action = policy_action = int(decision.action_indices[0])
                    diagnostic = {
                        "policy": "rule_plus_single_evaluation_mechanism",
                        "evaluation_mechanisms": [spec.mechanism],
                        "raw_action": policy_action,
                        "executed_action": action,
                        "intervened": False,
                        "mechanism_skipped_reason": "no_rule_threat_scores",
                    }
                else:
                    policy_action = int(selection["raw_rule_action"])
                    action = int(selection["mechanism_action"])
                    mechanism = dict(selection["mechanism"])
                    mechanism_changed = action != policy_action
                    mechanism_intervention_count += int(mechanism_changed)
                    fallback_required = mechanism.get("fallback_reason") == "empty_safe_set"
                    diagnostic = {
                        "policy": "rule_plus_single_evaluation_mechanism",
                        "evaluation_mechanisms": [spec.mechanism],
                        "raw_action": policy_action,
                        "mechanism_action": action,
                        "executed_action": action,
                        "mechanism_intervened": mechanism_changed,
                        "intervened": mechanism_changed,
                        "mechanism": mechanism,
                    }

        observation, reward, terminated, truncated, info = env.step(
            int(action),
            policy_action=int(policy_action),
            action_mask=action_mask,
            fallback_required=fallback_required,
        )
        total_reward += float(reward)
        learning_transition = bool(info["learning_transition"])
        executed_action = int(info["executed_action_index"])
        if learning_transition:
            decision_steps += 1
            raw_actions[int(policy_action)] += 1
            executed_actions[executed_action] += 1
            intervention_count += int(executed_action != int(policy_action))
            for name, value in dict(info.get("reward_components", {})).items():
                reward_components[str(name)] += float(value)
            diagnostic.update({
                "time_s": float(info["time_s"]),
                "step_count": int(info["step_count"]),
                "executed_action": executed_action,
            })
            diagnostics.append(_jsonable(diagnostic))
        if tracker is not None:
            mechanism_diagnostic = dict(diagnostic.get("mechanism", {}))
            envelope_diagnostic = dict(diagnostic.get("envelope", {}))
            safety_reasons = _merge_reasons(
                mechanism_diagnostic.get("hard_mask_reasons", []),
                envelope_diagnostic.get("raw_action_hard_violation_reasons", []),
            )
            tracker.add(
                dict(info["flight_quality_state"]),
                policy_action=int(policy_action) if learning_transition else None,
                executed_action=executed_action if learning_transition else None,
                safety_intervened=bool(
                    learning_transition and (
                        mechanism_diagnostic.get("safety_filter_intervened", False)
                        or envelope_diagnostic.get("intervened", False)
                    )
                ),
                safety_reasons=safety_reasons,
            )
        current_info = dict(info)
        last_info = current_info
        if terminated or truncated:
            break

    summary: dict[str, Any] = {
        "variant": spec.name,
        "policy": spec.policy,
        "evaluation_mechanism": spec.mechanism,
        "evaluation_mechanisms": [] if spec.mechanism is None else [spec.mechanism],
        "checkpoint_flight_envelope_applied": spec.policy == "rainbow",
        "seed": seed,
        "missile_count": missile_count,
        "initialization_fingerprint": fingerprint,
        "initialization": initialization,
        "blue_survived": bool(last_info["blue_survived"]),
        "termination_reason": str(last_info["termination_reason"]),
        "hit_count": int(last_info["hit_count"]),
        "miss_distance_m": float(last_info["miss_distance_m"]),
        "simulation_time_s": float(last_info["time_s"]),
        "physics_steps": int(last_info["step_count"]),
        "decision_steps": decision_steps,
        "reward": total_reward,
        "raw_action_histogram": {
            str(key): value for key, value in sorted(raw_actions.items())
        },
        "executed_action_histogram": {
            str(key): value for key, value in sorted(executed_actions.items())
        },
        "blue_mode_counts": dict(sorted(mode_counts.items())),
        "intervention_count": intervention_count,
        "mechanism_intervention_count": mechanism_intervention_count,
        "envelope_intervention_count": envelope_intervention_count,
        "reward_component_sums": dict(reward_components),
        "red_loss_reasons": list(last_info.get("red_loss_reasons", [])),
        "red_termination_events": list(
            dict(last_info["flight_quality_state"]).get("red_termination_events", [])
        ),
        "acmi_path": last_info.get("acmi_path"),
        "guidance_contract": GUIDANCE_CONTRACT,
    }
    if rule is not None:
        summary["blue_rule_runtime_statistics"] = rule.runtime_statistics()
    trajectory = None
    if tracker is not None:
        trajectory = tracker.finish(
            episode=1,
            survived=bool(summary["blue_survived"]),
            metadata={
                "variant": spec.name,
                "seed": seed,
                "missile_count": missile_count,
                "initialization_fingerprint": fingerprint,
                "initialization": initialization,
                "termination_reason": summary["termination_reason"],
            },
        )
        summary["flight_quality"] = {
            key: trajectory[key] for key in ("metrics", "verdicts", "events")
        }
        summary["red_termination_events"] = list(
            trajectory.get("red_termination_events", summary["red_termination_events"])
        )
    return EpisodeRun(summary=summary, trajectory=trajectory, diagnostics=diagnostics)


def compare_with_baseline(
    baseline: EpisodeRun,
    candidate: EpisodeRun,
) -> dict[str, Any]:
    base_summary, candidate_summary = baseline.summary, candidate.summary
    result: dict[str, Any] = {
        "variant": candidate_summary["variant"],
        "outcome_changed": (
            bool(base_summary["blue_survived"]) != bool(candidate_summary["blue_survived"])
            or str(base_summary["termination_reason"])
            != str(candidate_summary["termination_reason"])
        ),
        "survival_changed": (
            bool(base_summary["blue_survived"]) != bool(candidate_summary["blue_survived"])
        ),
        "termination_reason_changed": (
            str(base_summary["termination_reason"])
            != str(candidate_summary["termination_reason"])
        ),
        "miss_distance_delta_m": (
            float(candidate_summary["miss_distance_m"])
            - float(base_summary["miss_distance_m"])
        ),
        "simulation_time_delta_s": (
            float(candidate_summary["simulation_time_s"])
            - float(base_summary["simulation_time_s"])
        ),
    }
    if baseline.trajectory is None or candidate.trajectory is None:
        return result
    base_trace = dict(baseline.trajectory["trace"])
    candidate_trace = dict(candidate.trajectory["trace"])
    base_positions = np.asarray(base_trace["position_m"], dtype=np.float64)
    candidate_positions = np.asarray(candidate_trace["position_m"], dtype=np.float64)
    shared = min(len(base_positions), len(candidate_positions))
    deltas = np.linalg.norm(candidate_positions[:shared] - base_positions[:shared], axis=1)
    base_actions = list(base_trace["executed_action"])
    candidate_actions = list(candidate_trace["executed_action"])
    shared_actions = min(len(base_actions), len(candidate_actions))
    different = [
        index for index in range(shared_actions)
        if base_actions[index] != candidate_actions[index]
    ]
    candidate_times = list(candidate_trace["time_s"])
    result.update({
        "trajectory_changed": (
            len(base_positions) != len(candidate_positions)
            or bool(np.any(deltas > 1.0e-9))
        ),
        "action_sequence_changed": (
            len(base_actions) != len(candidate_actions) or bool(different)
        ),
        "first_action_difference_time_s": (
            float(candidate_times[different[0]]) if different else None
        ),
        "shared_trace_samples": shared,
        "maximum_shared_position_delta_m": float(np.max(deltas)) if shared else None,
        "final_shared_position_delta_m": float(deltas[-1]) if shared else None,
    })
    return result


def _write_trajectory_csv(path: Path, trajectory: dict[str, Any]) -> None:
    trace = dict(trajectory["trace"])
    count = len(trace["time_s"])
    fieldnames = (
        "time_s", "sample_interval_s", "step_count",
        "blue_x_north_m", "blue_y_up_m", "blue_z_east_m",
        "blue_vx_mps", "blue_vy_mps", "blue_vz_mps", "blue_speed_mps",
        "policy_action", "executed_action", "estimated_actual_load_g",
        "safety_intervened", "safety_reasons", "red_positions_m",
        "red_velocities_mps", "red_alive", "red_loss_reasons",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(count):
            position = trace["position_m"][index]
            velocity = trace["velocity_mps"][index]
            writer.writerow({
                "time_s": trace["time_s"][index],
                "sample_interval_s": trace.get("sample_interval_s", [None] * count)[index],
                "step_count": trace.get("step_count", [None] * count)[index],
                "blue_x_north_m": position[0],
                "blue_y_up_m": position[1],
                "blue_z_east_m": position[2],
                "blue_vx_mps": velocity[0],
                "blue_vy_mps": velocity[1],
                "blue_vz_mps": velocity[2],
                "blue_speed_mps": trace["speed_mps"][index],
                "policy_action": trace["policy_action"][index],
                "executed_action": trace["executed_action"][index],
                "estimated_actual_load_g": trace["estimated_actual_load_g"][index],
                "safety_intervened": trace["safety_intervened"][index],
                "safety_reasons": json.dumps(
                    trace["safety_reasons"][index], ensure_ascii=False
                ),
                "red_positions_m": json.dumps(
                    trace.get("red_positions_m", [[]] * count)[index],
                    ensure_ascii=False,
                ),
                "red_velocities_mps": json.dumps(
                    trace.get("red_velocities_mps", [[]] * count)[index],
                    ensure_ascii=False,
                ),
                "red_alive": json.dumps(
                    trace.get("red_alive", [[]] * count)[index], ensure_ascii=False
                ),
                "red_loss_reasons": json.dumps(
                    trace.get("red_loss_reasons", [[]] * count)[index],
                    ensure_ascii=False,
                ),
            })


def _write_episode_artifacts(directory: Path, run: EpisodeRun) -> dict[str, str]:
    if run.trajectory is None:
        raise ValueError("trajectory capture is required for comparison artifacts")
    directory.mkdir(parents=True, exist_ok=True)
    summary_path = directory / "result.json"
    trajectory_path = directory / "trajectory.json"
    trace_jsonl_path = directory / "flight_quality_episodes.jsonl"
    trajectory_csv_path = directory / "trajectory.csv"
    diagnostics_path = directory / "decision_diagnostics.jsonl"
    _write_json(summary_path, run.summary)
    _write_json(trajectory_path, run.trajectory)
    trace_jsonl_path.write_text("", encoding="utf-8")
    append_flight_quality_episode(run.trajectory, trace_jsonl_path)
    _write_trajectory_csv(trajectory_csv_path, run.trajectory)
    diagnostics_path.write_text("", encoding="utf-8")
    for row in run.diagnostics:
        _append_jsonl(diagnostics_path, row)
    return {
        "result_json": str(summary_path),
        "trajectory_json": str(trajectory_path),
        "trajectory_csv": str(trajectory_csv_path),
        "flight_quality_jsonl": str(trace_jsonl_path),
        "decision_diagnostics_jsonl": str(diagnostics_path),
        **({"acmi": str(run.summary["acmi_path"])} if run.summary.get("acmi_path") else {}),
    }


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = (
        "variant", "policy", "evaluation_mechanism", "blue_survived",
        "termination_reason", "miss_distance_m", "simulation_time_s",
        "decision_steps", "intervention_count", "mechanism_intervention_count",
        "envelope_intervention_count", "outcome_changed", "survival_changed",
        "trajectory_changed", "action_sequence_changed",
        "first_action_difference_time_s", "maximum_shared_position_delta_m",
    )
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _learner_state(agent: RainbowDQNAgent) -> tuple[object, ...]:
    return (
        agent.total_steps,
        agent.optimizer_updates,
        agent.target_updates,
        agent.replay.size,
        tuple(parameter._version for parameter in agent.online.parameters()),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    if args.seed_start < 0 or args.max_seeds < 1 or args.matches < 1:
        raise SystemExit("seed-start must be non-negative; max-seeds and matches must be positive")
    if args.decision_interval <= 0.0:
        raise SystemExit("decision-interval must be positive")
    if args.red_sector_width_deg is not None and not 0.0 < args.red_sector_width_deg <= 360.0:
        raise SystemExit("red-sector-width-deg must be in (0, 360]")
    manifest_path = output / "manifest.json"
    if output.exists() and any(output.iterdir()):
        raise SystemExit(
            f"output directory is not empty; choose a new folder: {output}"
        )
    weights = (
        args.mechanism_weight,
        args.mechanism_threat_weight,
        args.mechanism_timing_weight,
        args.mechanism_direction_weight,
        args.mechanism_overload_weight,
    )
    if not all(math.isfinite(value) and value >= 0.0 for value in weights):
        raise SystemExit("mechanism weights must be finite and non-negative")
    try:
        for name in MECHANISMS:
            EvaluationActionShaper(_mechanism_config(args, name))
    except ValueError as error:
        raise SystemExit(str(error)) from error

    agent = RainbowDQNAgent.load(str(checkpoint), args.device)
    agent.online.eval()
    for parameter in agent.online.parameters():
        parameter.requires_grad_(False)
    immutable_learner_state = _learner_state(agent)
    slots = infer_missile_slots(
        agent.config.observation_schema, agent.config.observation_dim
    )
    if agent.config.action_dim != 29:
        raise SystemExit(
            f"checkpoint action dimension is {agent.config.action_dim}; expected 29"
        )
    try:
        missile_counts = parse_missile_scenarios(args.missiles)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if any(count < 2 for count in missile_counts):
        raise SystemExit("this comparison requires multi-missile scenarios (2 or more)")
    if any(count > slots for count in missile_counts):
        raise SystemExit(
            f"checkpoint has {slots} missile slots; requested scenarios are {missile_counts}"
        )
    environment_config = configure_blue_mission_duration(
        load_environment_config(args.env_config)
    )
    if args.red_sector_width_deg is not None:
        environment_config = replace(
            environment_config,
            scenario=replace(
                environment_config.scenario,
                red_sector_width_deg=args.red_sector_width_deg,
            ),
        )
        environment_config.validate()
    mechanism_reward = (
        MechanismRewardConfig(**agent.config.mechanism_reward_config)
        if agent.config.mechanism_reward_config
        else MechanismRewardConfig(enabled=False)
    )
    envelope_config = (
        FlightEnvelopeConfig(**agent.config.flight_envelope_config)
        if agent.config.flight_envelope_config
        else FlightEnvelopeConfig(action_prediction_s=args.decision_interval)
    )
    if not math.isclose(
        envelope_config.action_prediction_s,
        args.decision_interval,
        rel_tol=0.0,
        abs_tol=1.0e-12,
    ):
        raise SystemExit(
            "decision-interval must match the flight-envelope interval saved in the checkpoint"
        )

    def shaping_config_factory(name: str) -> EvaluationShapingConfig:
        return _mechanism_config(args, name)

    base_adapter = _environment_adapter(
        slots=slots,
        observation_schema=agent.config.observation_schema,
        decision_interval_s=args.decision_interval,
        mechanism_reward=mechanism_reward,
        record_acmi=False,
        acmi_directory=output / "screening_acmi_disabled",
    )
    candidates = build_candidate_plan(args.seed_start, args.max_seeds, missile_counts)
    output.mkdir(parents=True, exist_ok=True)
    search_log_path = output / "search_log.jsonl"
    search_log_path.write_text("", encoding="utf-8")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "checkpoint": {
            "path": str(checkpoint),
            "sha256": _sha256_file(checkpoint),
            "observation_schema": agent.config.observation_schema,
            "observation_dim": agent.config.observation_dim,
            "action_dim": agent.config.action_dim,
            "missile_slots": slots,
            "optimizer_updates": agent.optimizer_updates,
            "target_updates": agent.target_updates,
        },
        "search": {
            "seed_start": args.seed_start,
            "max_seeds": args.max_seeds,
            "missile_counts": list(missile_counts),
            "candidate_count": len(candidates),
            "required_matches": args.matches,
            "condition": "rule_raw.blue_survived == false and rainbow_full.blue_survived == true",
        },
        "variant_order": list(VARIANT_ORDER),
        "four_rules_are_evaluation_only": True,
        "rainbow_full": {
            "checkpoint_native_policy": True,
            "saved_flight_envelope": True,
            "evaluation_only_mechanisms_reapplied": False,
            "active_training_reward_mechanisms": list(mechanism_reward.active_mechanisms()),
        },
        "mechanism_shaping": {
            "weight": args.mechanism_weight,
            "rule_checkpoint_flight_envelope_applied": False,
            "shared_shaper_behavior": [
                "action_switch_penalty", "speed_altitude_load_safety_filter"
            ],
            **{
                f"{name}_weight": getattr(args, f"mechanism_{name}_weight")
                for name in MECHANISMS
            },
        },
        "guidance_contract": GUIDANCE_CONTRACT,
        "environment_config": asdict(environment_config),
        "adapter_config": asdict(base_adapter),
        "attempts_completed": 0,
        "matches": [],
        "complete": False,
    }
    _write_json(manifest_path, manifest)

    raw_spec = variant_specs()[0]
    rl_spec = variant_specs()[-1]
    matches: list[dict[str, Any]] = []
    for attempt_index, (seed, missile_count) in enumerate(candidates, 1):
        rule_screen = _run_episode(
            spec=raw_spec,
            seed=seed,
            missile_count=missile_count,
            environment_config=environment_config,
            adapter_config=base_adapter,
            agent=agent,
            envelope_config=envelope_config,
            rule_backend=args.blue_rule_execution_backend,
            shaping_config_factory=shaping_config_factory,
            capture_trajectory=False,
        )
        screen_record: dict[str, Any] = {
            "event": "candidate_screened",
            "attempt": attempt_index,
            "seed": seed,
            "missile_count": missile_count,
            "rule_raw": {
                key: rule_screen.summary[key]
                for key in (
                    "blue_survived", "termination_reason", "miss_distance_m",
                    "simulation_time_s", "initialization_fingerprint",
                )
            },
            "rainbow_full": None,
            "matched": False,
        }
        if not bool(rule_screen.summary["blue_survived"]):
            rl_screen = _run_episode(
                spec=rl_spec,
                seed=seed,
                missile_count=missile_count,
                environment_config=environment_config,
                adapter_config=base_adapter,
                agent=agent,
                envelope_config=envelope_config,
                rule_backend=args.blue_rule_execution_backend,
                shaping_config_factory=shaping_config_factory,
                capture_trajectory=False,
            )
            if (
                rule_screen.summary["initialization_fingerprint"]
                != rl_screen.summary["initialization_fingerprint"]
            ):
                raise RuntimeError("screening policies did not receive identical initialization")
            screen_record["rainbow_full"] = {
                key: rl_screen.summary[key]
                for key in (
                    "blue_survived", "termination_reason", "miss_distance_m",
                    "simulation_time_s", "initialization_fingerprint",
                )
            }
            screen_record["matched"] = is_target_outcome(
                rule_screen.summary, rl_screen.summary
            )
        _append_jsonl(search_log_path, screen_record)
        manifest["attempts_completed"] = attempt_index
        _write_json(manifest_path, manifest)
        print(json.dumps(_jsonable(screen_record), ensure_ascii=False), flush=True)
        if not screen_record["matched"]:
            continue

        match_index = len(matches) + 1
        match_directory = (
            output / "matches"
            / f"match_{match_index:03d}_seed_{seed}_missiles_{missile_count}"
        )
        final_runs: dict[str, EpisodeRun] = {}
        for spec in variant_specs():
            variant_directory = match_directory / spec.name
            adapter = replace(
                base_adapter,
                record_acmi=not args.no_acmi,
                acmi_episode_interval=0 if args.no_acmi else 1,
                acmi_directory=str(variant_directory / "acmi"),
            )
            run = _run_episode(
                spec=spec,
                seed=seed,
                missile_count=missile_count,
                environment_config=environment_config,
                adapter_config=adapter,
                agent=agent,
                envelope_config=envelope_config,
                rule_backend=args.blue_rule_execution_backend,
                shaping_config_factory=shaping_config_factory,
                capture_trajectory=True,
            )
            run.summary["artifacts"] = _write_episode_artifacts(
                variant_directory, run
            )
            _write_json(variant_directory / "result.json", run.summary)
            final_runs[spec.name] = run

        fingerprints = {
            str(run.summary["initialization_fingerprint"])
            for run in final_runs.values()
        }
        if len(fingerprints) != 1:
            raise RuntimeError("final policy variants did not receive identical initialization")
        baseline = final_runs["rule_raw"]
        rainbow = final_runs["rainbow_full"]
        if not is_target_outcome(baseline.summary, rainbow.summary):
            raise RuntimeError("screening match was not deterministic when trajectories were recorded")
        comparisons = {
            name: compare_with_baseline(baseline, run)
            for name, run in final_runs.items()
        }
        comparison_rows = []
        for name in VARIANT_ORDER:
            row = {**final_runs[name].summary, **comparisons[name]}
            comparison_rows.append(row)
        comparison = {
            "match_index": match_index,
            "seed": seed,
            "missile_count": missile_count,
            "initialization_fingerprint": next(iter(fingerprints)),
            "identical_initialization_verified": True,
            "target_condition_satisfied": True,
            "baseline_variant": "rule_raw",
            "variant_order": list(VARIANT_ORDER),
            "variants": [final_runs[name].summary for name in VARIANT_ORDER],
            "changes_from_rule_raw": [comparisons[name] for name in VARIANT_ORDER],
        }
        _write_json(match_directory / "comparison.json", comparison)
        _write_comparison_csv(match_directory / "comparison.csv", comparison_rows)
        match_record = {
            "match_index": match_index,
            "seed": seed,
            "missile_count": missile_count,
            "initialization_fingerprint": next(iter(fingerprints)),
            "directory": str(match_directory),
            "comparison_json": str(match_directory / "comparison.json"),
            "comparison_csv": str(match_directory / "comparison.csv"),
            "rule_raw_survived": False,
            "rainbow_full_survived": True,
        }
        matches.append(match_record)
        manifest["matches"] = matches
        _write_json(manifest_path, manifest)
        if len(matches) >= args.matches:
            break

    if _learner_state(agent) != immutable_learner_state:
        raise RuntimeError("comparison modified the checkpoint learner state")
    manifest["complete"] = len(matches) >= args.matches
    manifest["matches_found"] = len(matches)
    manifest["learner_state_unchanged"] = True
    _write_json(manifest_path, manifest)
    final_event = {
        "event": "comparison_complete",
        "complete": manifest["complete"],
        "attempts_completed": manifest["attempts_completed"],
        "matches_found": len(matches),
        "manifest": str(manifest_path),
    }
    _append_jsonl(search_log_path, final_event)
    print(json.dumps(final_event, ensure_ascii=False), flush=True)
    return 0 if bool(manifest["complete"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
