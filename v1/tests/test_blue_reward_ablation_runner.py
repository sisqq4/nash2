from __future__ import annotations

import json
from pathlib import Path

from red_swarm_policy.run_blue_rl_reward_ablations import (
    CORE_CASES,
    _collect_artifacts,
    _snapshot_artifacts,
    build_parser,
    cases_for_suite,
    evaluation_command,
    main,
    rule_evaluation_command,
    training_command,
)


def test_reward_ablation_suites_cover_incremental_core_and_full_factorial() -> None:
    assert tuple((case.blue_policy, case.mechanisms) for case in CORE_CASES) == (
        ("rule", ()),
        ("rl", ()),
        ("rl", ("threat", "timing", "direction", "overload")),
        ("rl", ("threat",)),
        ("rl", ("threat", "timing")),
        ("rl", ("threat", "timing", "direction")),
    )
    assert tuple(case.mechanisms for case in CORE_CASES[1:]) == (
        (),
        ("threat", "timing", "direction", "overload"),
        ("threat",),
        ("threat", "timing"),
        ("threat", "timing", "direction"),
    )
    factorial = cases_for_suite("full-factorial")
    assert len(factorial) == 16
    assert len({case.mechanisms for case in factorial}) == 16


def test_reward_ablation_commands_use_the_same_selection_for_train_and_test(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args([])
    for case in CORE_CASES[1:]:
        case_root = tmp_path / case.name
        train = training_command(args, case, case_root / "train")
        checkpoint = case_root / "train" / "blue_rainbow.pt"
        evaluation = evaluation_command(
            args, case, checkpoint, case_root / "evaluation"
        )

        assert train[train.index("--reward-mechanisms") + 1] == case.selection
        assert evaluation[evaluation.index("--reward-mechanisms") + 1] == case.selection
        assert str(checkpoint) in evaluation
        assert all(
            f"--mechanism-{name}" not in evaluation
            for name in ("threat", "timing", "direction", "overload")
        )
        assert "--no-training-plots" not in train
        assert "--no-result-plots" not in evaluation
        assert "--flight-quality-plot-limit" not in train
        assert "--flight-quality-plot-limit" not in evaluation

    rule = rule_evaluation_command(args, tmp_path / "rule")
    assert "red_swarm_policy.evaluate_blue_rule_baseline" in rule
    assert rule[rule.index("--reward-mechanisms") + 1] == "none"
    assert rule[rule.index("--episodes") + 1] == str(args.eval_episodes)
    assert rule[rule.index("--seed-start") + 1] == str(args.eval_seed)


def test_reward_ablation_no_plots_disables_all_existing_plot_hooks(tmp_path: Path) -> None:
    args = build_parser().parse_args(["--no-plots"])
    case = CORE_CASES[1]
    train = training_command(args, case, tmp_path / "train")
    evaluation = evaluation_command(
        args, case, tmp_path / "train" / "blue_rainbow.pt", tmp_path / "evaluation"
    )
    rule = rule_evaluation_command(args, tmp_path / "rule")

    assert "--no-training-plots" in train
    assert train[train.index("--flight-quality-plot-limit") + 1] == "0"
    assert "--no-result-plots" in evaluation
    assert evaluation[evaluation.index("--flight-quality-plot-limit") + 1] == "0"
    assert "--no-result-plots" in rule


def test_reward_ablation_dry_run_writes_a_reviewable_manifest(tmp_path: Path) -> None:
    assert main([
        "--dry-run", "--output", str(tmp_path), "--suite", "core",
        "--train-episodes", "2", "--eval-episodes", "3",
    ]) == 0
    manifest = json.loads(
        (tmp_path / "reward_ablation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_count"] == 6
    assert manifest["execution_order"] == [
        "00_rule_baseline",
        "01_rl_baseline",
        "05_rl_all_mechanisms",
        "02_rl_threat",
        "03_rl_threat_timing",
        "04_rl_threat_timing_direction",
    ]
    assert manifest["base_reward_enabled_in_every_case"] is True
    assert manifest["runs"][0]["case"]["blue_policy"] == "rule"
    assert manifest["runs"][0]["train"]["status"] == "not_applicable"
    assert manifest["runs"][0]["evaluation"]["status"] == "dry_run"
    assert manifest["runs"][1]["case"]["mechanisms"] == []
    assert manifest["runs"][2]["case"]["mechanisms"] == [
        "threat", "timing", "direction", "overload"
    ]
    assert all(
        run["evaluation"]["status"] == "dry_run" for run in manifest["runs"]
    )
    assert all(
        run["train"]["status"] == "dry_run" for run in manifest["runs"][1:]
    )


def test_reward_ablation_artifact_audit_records_results_and_images(tmp_path: Path) -> None:
    for name in ("evaluation.json", "evaluation.jsonl"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    analysis = tmp_path / "evaluation_analysis"
    analysis.mkdir()
    (analysis / "survival.png").write_bytes(b"png")

    artifacts = _collect_artifacts(tmp_path, "rl_evaluation", plots_expected=True)

    assert artifacts["missing_required"] == []
    assert artifacts["stale_required"] == []
    assert artifacts["plot_status"] == "present"
    assert artifacts["images"] == [str(analysis / "survival.png")]
    assert artifacts["complete"] is True


def test_reward_ablation_artifact_audit_rejects_stale_results(tmp_path: Path) -> None:
    for name in ("evaluation.json", "evaluation.jsonl"):
        (tmp_path / name).write_text("{}", encoding="utf-8")
    analysis = tmp_path / "evaluation_analysis"
    analysis.mkdir()
    (analysis / "survival.png").write_bytes(b"png")
    previous = _snapshot_artifacts(tmp_path, "rl_evaluation")

    artifacts = _collect_artifacts(
        tmp_path, "rl_evaluation", plots_expected=True, previous=previous
    )

    assert artifacts["missing_required"] == []
    assert artifacts["stale_required"] == ["evaluation.json", "evaluation.jsonl"]
    assert artifacts["plot_status"] == "missing"
    assert artifacts["images"] == []
    assert artifacts["complete"] is False
