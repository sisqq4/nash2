"""Batch runner for reproducible blue evaluation-mechanism ablations."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import itertools
import json
from pathlib import Path
import subprocess
import sys
from typing import Sequence

from .cli_utils import parse_float_sequence, parse_missile_scenarios

MECHANISMS = ("threat", "timing", "direction", "overload")


@dataclass(frozen=True)
class AblationCase:
    name: str
    mechanisms: tuple[str, ...]


CORE_CASES = (
    AblationCase("00_rainbow_only", ()),
    AblationCase("01_threat_only", ("threat",)),
    AblationCase("02_timing_only", ("timing",)),
    AblationCase("03_direction_only", ("direction",)),
    AblationCase("04_overload_only", ("overload",)),
    AblationCase("05_threat_timing", ("threat", "timing")),
    AblationCase("06_threat_timing_direction", ("threat", "timing", "direction")),
    AblationCase("07_full_fusion", MECHANISMS),
)


def cases_for_suite(suite: str) -> tuple[AblationCase, ...]:
    if suite == "core":
        return CORE_CASES
    if suite == "full-factorial":
        cases = []
        for enabled_count in range(len(MECHANISMS) + 1):
            for combination in itertools.combinations(MECHANISMS, enabled_count):
                label = "rainbow_only" if not combination else "_".join(combination)
                cases.append(AblationCase(f"{len(cases):02d}_{label}", combination))
        return tuple(cases)
    raise ValueError(f"unknown ablation suite {suite!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run reproducible batches of blue-RL evaluation ablations")
    parser.add_argument("checkpoint")
    parser.add_argument("--suite", choices=("core", "full-factorial"), default="core")
    parser.add_argument("--seeds", default="10042", help="Comma-separated evaluation seeds")
    parser.add_argument("--missiles", default="1,2,3,4")
    parser.add_argument("--episodes", type=int, default=400)
    parser.add_argument("--output", default="outputs/blue_rl/ablations")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--parallel-envs", type=int, default=4)
    parser.add_argument("--env-worker-threads", type=int, default=1)
    parser.add_argument("--env-worker-timeout-s", type=float, default=300.0)
    parser.add_argument("--decision-interval", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--acmi-interval", type=int, default=0)
    parser.add_argument("--env-config", default=None)
    parser.add_argument("--mechanism-weight", type=float, default=0.35)
    for name in MECHANISMS:
        parser.add_argument(f"--mechanism-{name}-weight", type=float, default=1.0)
    parser.add_argument("--mechanism-detail-log", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def parse_seeds(value: str) -> tuple[int, ...]:
    values = parse_float_sequence(value, "seeds")
    if any(not item.is_integer() or item < 0 for item in values):
        raise ValueError("seeds must be comma-separated non-negative integers")
    return tuple(dict.fromkeys(int(item) for item in values))


def evaluation_command(args: argparse.Namespace, case: AblationCase, seed: int,
                       destination: Path) -> list[str]:
    command = [
        sys.executable, "-m", "red_swarm_policy.evaluate_blue_rl", str(Path(args.checkpoint)),
        "--missiles", args.missiles, "--episodes", str(args.episodes), "--seed", str(seed),
        "--output", str(destination), "--device", args.device,
        "--parallel-envs", str(args.parallel_envs),
        "--env-worker-threads", str(args.env_worker_threads),
        "--env-worker-timeout-s", str(args.env_worker_timeout_s),
        "--decision-interval", str(args.decision_interval),
        "--log-interval", str(args.log_interval), "--acmi-interval", str(args.acmi_interval),
        "--mechanism-weight", str(args.mechanism_weight),
    ]
    if args.env_config:
        command.extend(("--env-config", args.env_config))
    for name in MECHANISMS:
        command.extend((f"--mechanism-{name}-weight", str(getattr(args, f"mechanism_{name}_weight"))))
    command.extend(f"--mechanism-{name}" for name in case.mechanisms)
    if args.mechanism_detail_log:
        command.append("--mechanism-detail-log")
    return command


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1 or args.parallel_envs < 1 or args.env_worker_threads < 1:
        raise SystemExit("episodes and worker counts must be positive")
    try:
        seeds = parse_seeds(args.seeds)
        parse_missile_scenarios(args.missiles)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if not args.dry_run and not Path(args.checkpoint).is_file():
        raise SystemExit(f"checkpoint does not exist: {args.checkpoint}")
    cases = cases_for_suite(args.suite)
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    runs = []
    for seed in seeds:
        for case in cases:
            destination = root / f"seed_{seed}" / case.name
            command = evaluation_command(args, case, seed, destination)
            runs.append({"seed": seed, "case": asdict(case), "output": str(destination),
                         "command": command, "status": "planned", "returncode": None})
    manifest: dict[str, object] = {
        "checkpoint": str(Path(args.checkpoint)), "suite": args.suite,
        "seeds": list(seeds), "missiles": list(parse_missile_scenarios(args.missiles)),
        "episodes_per_run": args.episodes, "run_count": len(runs), "runs": runs,
    }
    manifest_path = root / "ablation_manifest.json"; _write_manifest(manifest_path, manifest)
    for index, run in enumerate(runs, 1):
        command = list(run["command"])
        print(f"[{index}/{len(runs)}]", subprocess.list2cmdline(command), flush=True)
        if args.dry_run:
            run["status"] = "dry_run"
        else:
            Path(str(run["output"])).mkdir(parents=True, exist_ok=True)
            completed = subprocess.run(command, check=False)
            run["returncode"] = completed.returncode
            run["status"] = "passed" if completed.returncode == 0 else "failed"
            _write_manifest(manifest_path, manifest)
            if completed.returncode and not args.continue_on_error:
                return completed.returncode
    _write_manifest(manifest_path, manifest)
    return 0 if all(run["status"] in {"passed", "dry_run"} for run in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
