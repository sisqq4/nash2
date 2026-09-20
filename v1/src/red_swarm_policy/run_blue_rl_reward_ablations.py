"""Sequentially train and evaluate blue reward-mechanism ablations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

from .blue_rl import MECHANISM_REWARD_NAMES
from .cli_utils import parse_missile_scenarios


@dataclass(frozen=True)
class RewardAblationCase:
    name: str
    mechanisms: tuple[str, ...]
    blue_policy: str = "rl"

    @property
    def selection(self) -> str:
        return ",".join(self.mechanisms) if self.mechanisms else "none"


CORE_CASES = (
    RewardAblationCase("01_rl_baseline", ()),
    RewardAblationCase("05_rl_all_mechanisms", MECHANISM_REWARD_NAMES),
    RewardAblationCase("02_rl_threat", ("threat",)),
    RewardAblationCase("03_rl_threat_timing", ("threat", "timing")),
    RewardAblationCase(
        "04_rl_threat_timing_direction", ("threat", "timing", "direction")
    ),
    RewardAblationCase("00_rule_baseline", (), blue_policy="rule"),
)


def cases_for_suite(suite: str) -> tuple[RewardAblationCase, ...]:
    if suite == "core":
        return CORE_CASES
    if suite == "full-factorial":
        cases: list[RewardAblationCase] = []
        for count in range(len(MECHANISM_REWARD_NAMES) + 1):
            for combination in itertools.combinations(MECHANISM_REWARD_NAMES, count):
                label = "baseline" if not combination else "_".join(combination)
                cases.append(RewardAblationCase(f"{len(cases):02d}_{label}", combination))
        return tuple(cases)
    raise ValueError(f"unknown ablation suite {suite!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sequentially train and evaluate blue reward-mechanism ablations"
    )
    parser.add_argument("--suite", choices=("core", "full-factorial"), default="core")
    parser.add_argument("--missiles", default="1,2,3,4")
    parser.add_argument("--train-episodes", type=int, default=1000)
    parser.add_argument(
        "--eval-episodes-per-scenario", "--eval-episodes", dest="eval_episodes",
        type=int, default=100,
        help="Evaluation rounds per selected missile-count scenario",
    )
    parser.add_argument(
        "--rule-episodes-per-scenario", type=int, default=None,
        help=("Optional rule-baseline rounds per missile count. If omitted, use "
              "--eval-episodes; both policies use the same paired seed schedule."),
    )
    parser.add_argument("--train-seed", type=int, default=42)
    parser.add_argument("--eval-seed", type=int, default=10042)
    parser.add_argument("--output", default="outputs/blue_rl/reward_ablations_six_cases")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parallel-envs", type=int, default=4)
    parser.add_argument("--env-worker-threads", type=int, default=1)
    parser.add_argument("--env-worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--updates-per-transition", type=float, default=1.0)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--acmi-interval", type=int, default=0)
    parser.add_argument("--env-config", default=None)
    parser.add_argument(
        "--blue-rule-execution-backend",
        choices=("reference", "vectorized_guarded", "shadow"),
        default="vectorized_guarded",
    )
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def training_command(args: argparse.Namespace, case: RewardAblationCase,
                     destination: Path) -> list[str]:
    if case.blue_policy != "rl":
        raise ValueError("the rule baseline has no training command")
    command = [
        sys.executable, "-m", "red_swarm_policy.train_blue_rl",
        "--reward-mechanisms", case.selection,
        "--missiles", args.missiles,
        "--episodes", str(args.train_episodes),
        "--seed", str(args.train_seed),
        "--output", str(destination),
        "--device", args.device,
        "--parallel-envs", str(args.parallel_envs),
        "--env-worker-threads", str(args.env_worker_threads),
        "--env-worker-timeout-s", str(args.env_worker_timeout_s),
        "--decision-interval", str(args.decision_interval),
        "--batch-size", str(args.batch_size),
        "--replay-size", str(args.replay_size),
        "--updates-per-transition", str(args.updates_per_transition),
        "--checkpoint-interval", str(args.checkpoint_interval),
        "--log-interval", str(args.log_interval),
        "--acmi-interval", str(args.acmi_interval),
    ]
    if args.env_config:
        command.extend(("--env-config", str(Path(args.env_config).resolve())))
    if args.curriculum:
        command.append("--curriculum")
    if args.no_plots:
        command.extend(("--no-training-plots", "--flight-quality-plot-limit", "0"))
    return command


def evaluation_command(args: argparse.Namespace, case: RewardAblationCase,
                       checkpoint: Path, destination: Path) -> list[str]:
    if case.blue_policy != "rl":
        raise ValueError("use rule_evaluation_command for the rule baseline")
    command = [
        sys.executable, "-m", "red_swarm_policy.evaluate_blue_rl", str(checkpoint),
        "--reward-mechanisms", case.selection,
        "--missiles", args.missiles,
        "--episodes-per-scenario", str(args.eval_episodes),
        "--seed", str(args.eval_seed),
        "--output", str(destination),
        "--device", args.device,
        "--parallel-envs", str(args.parallel_envs),
        "--env-worker-threads", str(args.env_worker_threads),
        "--env-worker-timeout-s", str(args.env_worker_timeout_s),
        "--decision-interval", str(args.decision_interval),
        "--log-interval", str(args.log_interval),
        "--acmi-interval", str(args.acmi_interval),
    ]
    if args.env_config:
        command.extend(("--env-config", str(Path(args.env_config).resolve())))
    if args.no_plots:
        command.extend(("--no-result-plots", "--flight-quality-plot-limit", "0"))
    return command


def rule_evaluation_command(args: argparse.Namespace, destination: Path) -> list[str]:
    """Build the existing no-learning rule-machine evaluation command."""
    command = [
        sys.executable, "-m", "red_swarm_policy.evaluate_blue_rule_baseline",
        "--reward-mechanisms", "none",
        "--missiles", args.missiles,
        "--seed-start", str(args.eval_seed),
        "--output", str(destination),
        "--decision-interval", str(args.decision_interval),
        "--blue-rule-execution-backend", args.blue_rule_execution_backend,
        "--log-interval", str(args.log_interval),
        "--acmi-interval", str(args.acmi_interval),
    ]
    command.extend((
        "--episodes-per-scenario",
        str(args.eval_episodes if args.rule_episodes_per_scenario is None
            else args.rule_episodes_per_scenario),
    ))
    if args.env_config:
        command.extend(("--env-config", str(Path(args.env_config).resolve())))
    if args.no_plots:
        command.append("--no-result-plots")
    return command


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


_REQUIRED_ARTIFACT_NAMES = {
    "rl_training": (
        "blue_rainbow.pt", "training_metrics.json", "training.jsonl", "episodes.json",
    ),
    "rl_evaluation": ("evaluation.json", "evaluation.jsonl"),
    "rule_evaluation": (
        "blue_rule_baseline_summary.json", "blue_rule_baseline_trials.csv",
        "blue_rule_baseline_progress.jsonl",
    ),
}


def _file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _snapshot_artifacts(output: Path, kind: str) -> dict[str, tuple[int, int]]:
    """Capture artifacts that existed before a subprocess starts."""
    if kind not in _REQUIRED_ARTIFACT_NAMES:
        raise ValueError(f"unknown artifact kind {kind!r}")
    paths = [output / name for name in _REQUIRED_ARTIFACT_NAMES[kind]]
    if output.exists():
        paths.extend(output.rglob("*.png"))
    return {
        str(path.resolve()): _file_signature(path)
        for path in paths if path.is_file()
    }


def _collect_artifacts(
    output: Path,
    kind: str,
    *,
    plots_expected: bool,
    previous: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, object]:
    if kind not in _REQUIRED_ARTIFACT_NAMES:
        raise ValueError(f"unknown artifact kind {kind!r}")
    required_paths = {name: output / name for name in _REQUIRED_ARTIFACT_NAMES[kind]}
    required = {name: str(path) for name, path in required_paths.items()}
    missing = [name for name, path in required_paths.items() if not path.is_file()]
    stale = [
        name for name, path in required_paths.items()
        if path.is_file() and previous is not None
        and previous.get(str(path.resolve())) == _file_signature(path)
    ]
    all_images = sorted(output.rglob("*.png")) if output.exists() else []
    current_images = [
        path for path in all_images
        if previous is None
        or previous.get(str(path.resolve())) != _file_signature(path)
    ]
    plot_status = (
        "disabled" if not plots_expected
        else ("present" if current_images else "missing")
    )
    warnings = []
    if stale:
        warnings.append("required files were not updated by this run")
    if plots_expected and not current_images:
        warnings.append("no PNG files were produced by the existing plotting hooks")
    complete = not missing and not stale and plot_status != "missing"
    return {
        "required": required,
        "missing_required": missing,
        "stale_required": stale,
        "images": [str(path) for path in current_images],
        "plot_status": plot_status,
        "warnings": warnings,
        "complete": complete,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    positive_counts = (
        args.train_episodes, args.eval_episodes, args.parallel_envs,
        args.env_worker_threads, args.batch_size, args.replay_size,
        args.checkpoint_interval, args.log_interval,
    )
    if min(positive_counts) < 1:
        raise SystemExit("episode, worker, batch, replay, checkpoint, and log counts must be positive")
    if args.env_worker_timeout_s <= 0 or args.updates_per_transition < 0:
        raise SystemExit("worker timeout must be positive and update ratio non-negative")
    if args.acmi_interval < 0 or min(args.train_seed, args.eval_seed) < 0:
        raise SystemExit("ACMI interval and seeds must be non-negative")
    try:
        missile_scenarios = parse_missile_scenarios(args.missiles)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if args.rule_episodes_per_scenario is not None and args.rule_episodes_per_scenario < 1:
        raise SystemExit("rule episodes per scenario must be positive")

    project_root = Path(__file__).resolve().parents[2]
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, object]] = []
    for case in cases_for_suite(args.suite):
        case_root = output_root / case.name
        eval_output = case_root / "evaluation"
        if case.blue_policy == "rule":
            rule_episodes_per_scenario = (
                args.eval_episodes if args.rule_episodes_per_scenario is None
                else args.rule_episodes_per_scenario
            )
            train: dict[str, object] = {
                "output": None, "command": None, "status": "not_applicable",
                "returncode": None,
                "reason": "BlueEvasionRuleMachine has no learner or checkpoint",
            }
            evaluation: dict[str, object] = {
                "output": str(eval_output), "checkpoint": None,
                "command": rule_evaluation_command(args, eval_output),
                "status": "planned", "returncode": None,
                "episodes_per_scenario": rule_episodes_per_scenario,
                "total_episodes": rule_episodes_per_scenario * len(missile_scenarios),
            }
        else:
            train_output = case_root / "train"
            checkpoint = train_output / "blue_rainbow.pt"
            train = {
                "output": str(train_output),
                "command": training_command(args, case, train_output),
                "status": "planned", "returncode": None,
            }
            evaluation = {
                "output": str(eval_output), "checkpoint": str(checkpoint),
                "command": evaluation_command(args, case, checkpoint, eval_output),
                "status": "planned", "returncode": None,
                "episodes_per_scenario": args.eval_episodes,
                "total_episodes": args.eval_episodes * len(missile_scenarios),
            }
        runs.append({
            "case": asdict(case),
            "base_reward_enabled": True,
            "train": train,
            "evaluation": evaluation,
        })
    manifest: dict[str, object] = {
        "suite": args.suite,
        "mechanism_order": list(MECHANISM_REWARD_NAMES),
        "base_reward_enabled_in_every_case": True,
        "train_seed": args.train_seed, "eval_seed": args.eval_seed,
        "eval_episodes_per_scenario": args.eval_episodes,
        "missiles": list(missile_scenarios),
        "rule_episodes_per_scenario": args.rule_episodes_per_scenario,
        "plots_enabled": not args.no_plots,
        "execution_order": [dict(run["case"])["name"] for run in runs],
        "run_count": len(runs), "runs": runs,
    }
    manifest_path = output_root / "reward_ablation_manifest.json"
    _write_manifest(manifest_path, manifest)

    child_environment = os.environ.copy()
    source_root = str(project_root / "src")
    existing_pythonpath = child_environment.get("PYTHONPATH")
    child_environment["PYTHONPATH"] = (
        source_root if not existing_pythonpath
        else source_root + os.pathsep + existing_pythonpath
    )
    for index, run in enumerate(runs, 1):
        case = dict(run["case"])
        print(f"[{index}/{len(runs)}] {case['name']}", flush=True)
        train = dict(run["train"])
        evaluation = dict(run["evaluation"])
        if args.dry_run:
            if train["status"] != "not_applicable":
                train["status"] = "dry_run"
            evaluation["status"] = "dry_run"
            run["train"] = train; run["evaluation"] = evaluation
            continue

        if train["status"] != "not_applicable":
            train_output_path = Path(str(train["output"]))
            train_output_path.mkdir(parents=True, exist_ok=True)
            previous_artifacts = _snapshot_artifacts(train_output_path, "rl_training")
            completed = subprocess.run(
                list(train["command"]), check=False, cwd=project_root,
                env=child_environment,
            )
            train["returncode"] = completed.returncode
            train["status"] = "passed" if completed.returncode == 0 else "failed"
            train["artifacts"] = _collect_artifacts(
                train_output_path, "rl_training",
                plots_expected=not args.no_plots,
                previous=previous_artifacts,
            )
            if completed.returncode == 0 and not train["artifacts"]["complete"]:
                train["status"] = "failed"
            run["train"] = train
            _write_manifest(manifest_path, manifest)
            if train["status"] == "failed":
                evaluation["status"] = "skipped"
                run["evaluation"] = evaluation
                _write_manifest(manifest_path, manifest)
                if not args.continue_on_error:
                    return completed.returncode or 1
                continue

        evaluation_output_path = Path(str(evaluation["output"]))
        evaluation_output_path.mkdir(parents=True, exist_ok=True)
        artifact_kind = (
            "rule_evaluation" if case["blue_policy"] == "rule" else "rl_evaluation"
        )
        previous_artifacts = _snapshot_artifacts(evaluation_output_path, artifact_kind)
        completed = subprocess.run(
            list(evaluation["command"]), check=False, cwd=project_root, env=child_environment
        )
        evaluation["returncode"] = completed.returncode
        evaluation["status"] = "passed" if completed.returncode == 0 else "failed"
        evaluation["artifacts"] = _collect_artifacts(
            evaluation_output_path, artifact_kind,
            plots_expected=not args.no_plots,
            previous=previous_artifacts,
        )
        if completed.returncode == 0 and not evaluation["artifacts"]["complete"]:
            evaluation["status"] = "failed"
        run["evaluation"] = evaluation
        _write_manifest(manifest_path, manifest)
        if evaluation["status"] == "failed" and not args.continue_on_error:
            return completed.returncode or 1

    _write_manifest(manifest_path, manifest)
    successful = {"passed", "dry_run", "not_applicable"}
    return 0 if all(
        dict(run[stage])["status"] in successful
        for run in runs for stage in ("train", "evaluation")
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
