from __future__ import annotations

from red_swarm_policy.evaluate_blue_rule_baseline import (
    _mark_as_blue_rule_baseline,
    build_parser,
)
from red_swarm_policy.cli_utils import parse_missile_scenarios


def test_blue_rule_baseline_defaults_match_blue_test_scenarios() -> None:
    args = build_parser().parse_args([])
    assert args.missiles == "1,2,3,4"
    assert args.episodes_per_scenario == 100
    assert args.decision_interval == 0.1
    assert args.output.as_posix() == "outputs/blue_rl/rule_baseline"


def test_blue_rule_baseline_uses_shared_missile_validation() -> None:
    assert parse_missile_scenarios("1,4,1") == (1, 4)


def test_blue_rule_baseline_metadata_explicitly_disables_learning() -> None:
    summary: dict[str, object] = {"configuration": {}}
    _mark_as_blue_rule_baseline(summary)
    configuration = summary["configuration"]
    assert isinstance(configuration, dict)
    assert configuration["baseline"] is True
    assert configuration["blue_policy"] == "BlueEvasionRuleMachine"
    assert configuration["red_policy"] == "fixed_target_zero_residual_proportional_navigation"
    assert configuration["blue_learning_enabled"] is False
    assert configuration["red_learning_enabled"] is False
    assert configuration["blue_checkpoint"] is None
    assert configuration["red_checkpoint"] is None
