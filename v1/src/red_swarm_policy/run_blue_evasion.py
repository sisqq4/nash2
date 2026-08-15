from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .env import (
    BlueEvasionConfig,
    BlueEvasionController,
    BlueEvasionRuleMachine,
    EnvironmentConfig,
    RedBlueEngagementEnv,
    SCENARIO_STYLES,
    ScenarioConfig,
    ScenarioStyle,
    SensorConfig,
)


@dataclass(frozen=True)
class BlueEvasionRunSummary:
    seed: int
    start_mode: str
    physics_steps: int
    decisions: int
    final_time_s: float
    done: bool
    blue_survivors: int
    red_survivors: int
    final_info: dict[str, object]

    def output_record(self) -> dict[str, object]:
        return {
            "event": "blue_evasion_summary",
            "seed": self.seed,
            "start_mode": self.start_mode,
            "physics_steps": self.physics_steps,
            "decisions": self.decisions,
            "final_time_s": self.final_time_s,
            "done": self.done,
            "blue_survivors": self.blue_survivors,
            "red_survivors": self.red_survivors,
            "final_info": self.final_info,
        }


def run_blue_evasion_episode(
    environment: RedBlueEngagementEnv,
    controller: BlueEvasionController,
    *,
    seed: int,
    duration_s: float,
    style: ScenarioStyle = "many_to_many",
    start_mode: str = "launch",
    emit: Callable[[dict[str, object]], None] | None = None,
) -> BlueEvasionRunSummary:
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    duration_ratio = duration_s / environment.config.time_step_s
    duration_steps = int(round(duration_ratio))
    if not math.isclose(duration_ratio, duration_steps, rel_tol=0.0, abs_tol=1.0e-9):
        raise ValueError("duration_s must be an integer multiple of environment time_step_s")
    if start_mode not in ("launch", "post_boost"):
        raise ValueError("start_mode must be 'launch' or 'post_boost'")

    controller.reset()
    environment.reset(seed=seed, style=style, start_mode=start_mode)
    assert environment.state is not None
    initial_step = int(environment.state.step_count)
    final_step = initial_step + duration_steps
    decision_count = 0
    done = False
    final_info: dict[str, object] = {
        "time_s": float(environment.state.time_s),
        "step_count": int(environment.state.step_count),
    }

    while environment.state.step_count < final_step and not done:
        action, decision = controller.action_for(environment.state)
        if decision is not None:
            decision_count += 1
            if emit is not None:
                emit(decision.output_record())
        step = environment.step(blue_action=action)
        done = bool(step.done)
        final_info = dict(step.info)

    state = environment.state
    return BlueEvasionRunSummary(
        seed=seed,
        start_mode=start_mode,
        physics_steps=int(state.step_count - initial_step),
        decisions=decision_count,
        final_time_s=float(state.time_s),
        done=done,
        blue_survivors=sum(blue.alive for blue in state.blue),
        red_survivors=sum(red.alive for red in state.red),
        final_info=final_info,
    )


def _select_device(value: str) -> torch.device:
    normalized = value.strip().lower()
    if normalized == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the blue evasion rule machine at 0.1 s over a 0.005 s engagement environment."
    )
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--start-mode", choices=("launch", "post_boost"), default="launch")
    parser.add_argument("--style", choices=SCENARIO_STYLES, default="many_to_many")
    parser.add_argument("--red-count", type=int, default=24)
    parser.add_argument("--blue-count", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--time-step-s", type=float, default=0.005)
    parser.add_argument("--decision-interval-s", type=float, default=0.1)
    parser.add_argument("--detection-range-m", type=float, default=60000.0)
    parser.add_argument("--critical-range-m", type=float, default=30000.0)
    parser.add_argument("--lookahead-s", type=float, default=6.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.red_count <= 0 or args.blue_count <= 0:
        parser.error("--red-count and --blue-count must be positive")
    if args.duration_s <= 0.0 or args.time_step_s <= 0.0:
        parser.error("--duration-s and --time-step-s must be positive")
    duration_ratio = args.duration_s / args.time_step_s
    duration_steps = int(round(duration_ratio))
    if not math.isclose(duration_ratio, duration_steps, rel_tol=0.0, abs_tol=1.0e-9):
        parser.error("--duration-s must be an integer multiple of --time-step-s")

    base_config = EnvironmentConfig()
    max_steps = duration_steps
    if args.start_mode == "post_boost":
        max_steps += int(round(base_config.missile.boost_duration_s / args.time_step_s))
    environment_config = EnvironmentConfig(
        time_step_s=args.time_step_s,
        bias_update_interval_s=0.1,
        assignment_update_interval_s=1.0,
        max_steps=max_steps,
        policy_start_mode=args.start_mode,
        scenario=ScenarioConfig(red_count=args.red_count, blue_count=args.blue_count),
        sensor=SensorConfig(detection_range_m=args.detection_range_m),
    )
    evasion_config = BlueEvasionConfig(
        decision_interval_s=args.decision_interval_s,
        detection_range_m=args.detection_range_m,
        critical_range_m=args.critical_range_m,
        lookahead_s=args.lookahead_s,
    )
    device = _select_device(args.device)
    environment = RedBlueEngagementEnv(
        environment_config,
        device=device,
        record_replay=False,
    )
    controller = BlueEvasionController(
        BlueEvasionRuleMachine(environment_config, evasion_config)
    )

    def emit(record: dict[str, object]) -> None:
        print(json.dumps(record, ensure_ascii=True, separators=(",", ":")), flush=True)

    summary = run_blue_evasion_episode(
        environment,
        controller,
        seed=args.seed,
        duration_s=args.duration_s,
        style=args.style,
        start_mode=args.start_mode,
        emit=emit,
    )
    emit(summary.output_record())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
