from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from dataclasses import replace

import pytest

from red_swarm_policy.blue_rl import (
    BlueEscapeEnvConfig,
    FlightEnvelopeConfig,
    MechanismRewardConfig,
)
from red_swarm_policy.blue_rl.config_io import default_blue_environment_config
from red_swarm_policy.compare_blue_rule_mechanisms import (
    EpisodeRun,
    VARIANT_ORDER,
    _mechanism_config,
    _run_episode,
    build_candidate_plan,
    build_parser,
    compare_with_baseline,
    infer_missile_slots,
    initialization_fingerprint,
    is_target_outcome,
    variant_specs,
)


def test_comparison_defaults_target_checkpoint_and_multimissile_scenarios() -> None:
    args = build_parser().parse_args([])
    assert args.checkpoint.name == "blue_rainbow_ep001660.pt"
    assert args.missiles == "2,3"
    assert args.seed_start == 10089
    assert args.matches == 1


def test_variant_suite_is_raw_rule_four_isolated_rules_and_native_rainbow() -> None:
    specs = variant_specs()
    assert tuple(spec.name for spec in specs) == VARIANT_ORDER
    assert specs[0].policy == "rule" and specs[0].mechanism is None
    assert [spec.mechanism for spec in specs[1:5]] == [
        "threat", "timing", "direction", "overload"
    ]
    assert specs[-1].policy == "rainbow" and specs[-1].mechanism is None


@pytest.mark.parametrize("selected", ["threat", "timing", "direction", "overload"])
def test_single_mechanism_configuration_enables_exactly_one_score(selected: str) -> None:
    args = build_parser().parse_args([])
    config = _mechanism_config(args, selected)
    assert {
        name: getattr(config, name) for name in ("threat", "timing", "direction", "overload")
    } == {
        name: name == selected
        for name in ("threat", "timing", "direction", "overload")
    }


def test_checkpoint_dimension_resolves_supported_slots_and_rejects_unknown_contract() -> None:
    assert infer_missile_slots("normalized_v4", 104) == 3
    assert infer_missile_slots("normalized_v4", 149) == 6
    with pytest.raises(ValueError, match="does not identify"):
        infer_missile_slots("normalized_v4", 999)


def test_candidate_plan_pairs_each_seed_with_each_multimissile_count() -> None:
    assert build_candidate_plan(10, 2, (2, 3)) == [
        (10, 2), (10, 3), (11, 2), (11, 3)
    ]


def test_initialization_fingerprint_is_canonical_and_detects_physical_change() -> None:
    first = {"blue": {"velocity": [1.0, 2.0, 3.0]}, "red": [2, 1]}
    reordered = {"red": [2, 1], "blue": {"velocity": [1.0, 2.0, 3.0]}}
    changed = deepcopy(first)
    changed["blue"]["velocity"][0] = 1.1
    assert initialization_fingerprint(first) == initialization_fingerprint(reordered)
    assert initialization_fingerprint(first) != initialization_fingerprint(changed)


def _run(name: str, survived: bool, reason: str, positions, actions) -> EpisodeRun:
    return EpisodeRun(
        summary={
            "variant": name,
            "blue_survived": survived,
            "termination_reason": reason,
            "miss_distance_m": 10.0 if survived else 1.0,
            "simulation_time_s": 20.0 if survived else 10.0,
        },
        trajectory={
            "trace": {
                "position_m": positions,
                "executed_action": actions,
                "time_s": [float(index) for index in range(len(positions))],
            }
        },
        diagnostics=[],
    )


def test_comparison_marks_action_trajectory_and_outcome_changes() -> None:
    baseline = _run(
        "rule_raw", False, "success",
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], [None, 1],
    )
    candidate = _run(
        "rainbow_full", True, "red_failure",
        [[0.0, 0.0, 0.0], [1.0, 2.0, 0.0]], [None, 2],
    )
    comparison = compare_with_baseline(baseline, candidate)
    assert comparison["outcome_changed"] is True
    assert comparison["survival_changed"] is True
    assert comparison["trajectory_changed"] is True
    assert comparison["action_sequence_changed"] is True
    assert comparison["first_action_difference_time_s"] == 1.0
    assert comparison["maximum_shared_position_delta_m"] == 2.0
    assert is_target_outcome(baseline.summary, candidate.summary) is True


def test_target_outcome_requires_rule_failure_and_rainbow_survival() -> None:
    assert not is_target_outcome({"blue_survived": True}, {"blue_survived": True})
    assert not is_target_outcome({"blue_survived": False}, {"blue_survived": False})


def test_short_paired_rule_runs_share_initialization_and_record_mechanism_trace() -> None:
    base = default_blue_environment_config()
    environment_config = replace(
        base,
        max_steps=1460,
        scenario=replace(
            base.scenario,
            red_cluster_radius_range_m=(55000.0, 55000.0),
        ),
    )
    environment_config.validate()
    adapter = BlueEscapeEnvConfig(
        missile_count=2,
        max_missiles=2,
        observation_schema="normalized_v4",
        decision_interval_s=0.1,
        expose_evaluation_mechanism_state=True,
        record_red_telemetry=True,
        record_acmi=False,
        acmi_episode_interval=0,
        mechanism_reward=MechanismRewardConfig(),
    )
    mechanism_args = Namespace(
        mechanism_weight=0.35,
        mechanism_threat_weight=1.0,
        mechanism_timing_weight=1.0,
        mechanism_direction_weight=1.0,
        mechanism_overload_weight=1.0,
    )
    common = dict(
        seed=7,
        missile_count=2,
        environment_config=environment_config,
        adapter_config=adapter,
        agent=None,  # Rule variants never access the Rainbow agent.
        envelope_config=FlightEnvelopeConfig(action_prediction_s=0.1),
        rule_backend="vectorized_guarded",
        shaping_config_factory=lambda name: _mechanism_config(mechanism_args, name),
        capture_trajectory=True,
    )
    raw = _run_episode(spec=variant_specs()[0], **common)  # type: ignore[arg-type]
    threat = _run_episode(spec=variant_specs()[1], **common)  # type: ignore[arg-type]

    assert raw.summary["initialization_fingerprint"] == threat.summary[
        "initialization_fingerprint"
    ]
    assert raw.summary["checkpoint_flight_envelope_applied"] is False
    assert threat.summary["checkpoint_flight_envelope_applied"] is False
    assert raw.trajectory is not None and threat.trajectory is not None
    assert len(raw.trajectory["trace"]["time_s"]) == 4
    assert len(threat.diagnostics) == threat.summary["decision_steps"] == 3
    mechanism_rows = [row for row in threat.diagnostics if "mechanism" in row]
    assert mechanism_rows
    assert all(row["mechanism"]["active_scores"] == ["relief"] for row in mechanism_rows)
