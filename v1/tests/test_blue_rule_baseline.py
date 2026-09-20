from __future__ import annotations

import csv

from red_swarm_policy.evaluate_blue_rule_baseline import (

    _build_episode_plan,
    _emit,

    _mark_as_blue_rule_baseline,
    _write_trials,
    build_parser,
)
from red_swarm_policy.cli_utils import parse_missile_scenarios


def test_blue_rule_baseline_defaults_match_blue_test_scenarios() -> None:
    args = build_parser().parse_args([])
    assert args.missiles == "1,2,3,4"
    assert args.episodes_per_scenario == 100
    assert args.decision_interval == 0.1
    assert args.reward_mechanisms is None
    assert args.blue_rule_execution_backend == "vectorized_guarded"

    assert args.log_interval == 1

    assert args.output.as_posix() == "outputs/blue_rl/rule_baseline"


def test_blue_rule_baseline_can_report_base_reward_only() -> None:
    args = build_parser().parse_args([
        "--reward-mechanisms", "none", "--episodes", "25"
    ])
    assert args.reward_mechanisms == ()
    assert args.episodes_per_scenario == 25


def test_blue_rule_plan_matches_rainbow_paired_evaluation_schedule() -> None:
    plan, sampling, seed_schedule = _build_episode_plan(
        seed_start=2000, episodes_per_scenario=3, missile_counts=(1, 2, 3),
    )
    assert [episode for episode, _, _, _ in plan] == list(range(1, 10))
    assert [scenario_episode for _, scenario_episode, _, _ in plan] == [1, 2, 3] * 3
    assert [seed for _, _, seed, _ in plan] == [2001, 2002, 2003] * 3
    assert [missiles for _, _, _, missiles in plan] == [1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert sampling == "balanced_paired_by_scenario"
    assert seed_schedule == "base_seed_plus_one_based_scenario_episode_v1"


def test_blue_rule_baseline_uses_shared_missile_validation() -> None:
    assert parse_missile_scenarios("1,4,1") == (1, 4)


def test_blue_rule_baseline_metadata_explicitly_disables_learning() -> None:
    summary: dict[str, object] = {"configuration": {}}
    _mark_as_blue_rule_baseline(summary)
    configuration = summary["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["baseline"] is True
    assert configuration["blue_policy"] == "BlueEvasionRuleMachine"
    assert configuration["red_policy"] == "fixed_target_zero_residual_pure_proportional_navigation"
    assert configuration["guidance_contract"] == "pure_proportional_navigation_vm_v1"
    assert configuration["blue_learning_enabled"] is False
    assert configuration["red_learning_enabled"] is False
    assert configuration["blue_checkpoint"] is None
    assert configuration["red_checkpoint"] is None


def test_blue_rule_baseline_progress_is_flushed_and_archived(tmp_path, capsys) -> None:
    progress_path = tmp_path / "progress.jsonl"
    _emit({"event": "baseline_start", "total_episodes": 4}, progress_path)
    assert '"event": "baseline_start"' in capsys.readouterr().out
    assert '"total_episodes": 4' in progress_path.read_text(encoding="utf-8")


def test_blue_rule_baseline_trials_support_sparse_optional_fields(tmp_path) -> None:
    trials_path = tmp_path / "trials.csv"
    _write_trials(trials_path, [
        {"episode": 1, "reward": 2.0},
        {"episode": 2, "reward": 3.0, "acmi_path": "acmi/episode_000002.acmi"},
    ])

    with trials_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert list(rows[0]) == ["episode", "reward", "acmi_path"]
    assert rows[0]["acmi_path"] == ""
    assert rows[1]["acmi_path"] == "acmi/episode_000002.acmi"
