from __future__ import annotations

from pathlib import Path

from red_swarm_policy.run_blue_rl_ablations import (
    CORE_CASES,
    build_parser,
    cases_for_suite,
    evaluation_command,
    main,
    parse_seeds,
)


def test_ablation_suites_cover_core_and_all_factorial_combinations() -> None:
    assert len(CORE_CASES) == 8
    assert CORE_CASES[0].mechanisms == ()
    assert CORE_CASES[-1].mechanisms == ("threat", "timing", "direction", "overload")
    factorial = cases_for_suite("full-factorial")
    assert len(factorial) == 16
    assert len({case.mechanisms for case in factorial}) == 16


def test_ablation_command_only_enables_requested_mechanisms(tmp_path: Path) -> None:
    args = build_parser().parse_args(["checkpoint.pt"])
    command = evaluation_command(args, CORE_CASES[1], 7, tmp_path / "run")
    assert "--mechanism-threat" in command
    assert "--mechanism-timing" not in command
    assert command[command.index("--seed") + 1] == "7"
    assert command[command.index("--output") + 1] == str(tmp_path / "run")


def test_ablation_dry_run_writes_manifest_without_checkpoint(tmp_path: Path) -> None:
    assert main(["missing.pt", "--dry-run", "--output", str(tmp_path),
                 "--suite", "core", "--seeds", "3,5"]) == 0
    manifest = (tmp_path / "ablation_manifest.json").read_text(encoding="utf-8")
    assert '"run_count": 16' in manifest
    assert manifest.count('"status": "dry_run"') == 16
    assert parse_seeds("3,5,3") == (3, 5)
