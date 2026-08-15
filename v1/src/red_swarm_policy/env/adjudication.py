from __future__ import annotations

import math

import numpy as np

from .math_utils import norm
from .types import AdjudicationResult, EngagementState, EnvironmentConfig, JointAction, los_kinematics


class AdjudicationLayer:
    """Resolve physical hit/loss/terminal events; reward calculation is separate."""

    def __init__(self, config: EnvironmentConfig) -> None:
        self.config = config

    def evaluate(
        self,
        previous: EngagementState,
        current: EngagementState,
        action: JointAction,
        previous_action: JointAction | None = None,
        policy_updated: bool = True,
        assignment_updated: bool | None = None,
    ) -> AdjudicationResult:
        del previous_action, policy_updated, assignment_updated
        n_red = len(current.red)
        target_indices = np.asarray(action.red.target_indices, dtype=np.int64)
        if target_indices.shape != (n_red,):
            raise ValueError(f"target_indices shape {target_indices.shape} must be {(n_red,)}")

        hit_candidates: list[tuple[float, int, int]] = []
        miss_distance = math.inf
        loss_reasons: dict[int, str] = {}
        for red_index, red in enumerate(current.red):
            previous_red = previous.red[red_index]
            target_index = int(target_indices[red_index])
            if not previous_red.alive:
                continue
            if not (0 <= target_index < len(current.blue)) or not previous.blue[target_index].alive:
                if not red.alive:
                    loss_reasons[red_index] = self._physical_loss_reason(red)
                continue
            previous_blue = previous.blue[target_index]
            blue = current.blue[target_index]
            previous_range = los_kinematics(previous_red, previous_blue).range_m
            current_los = los_kinematics(red, blue)
            closest = self._segment_closest_distance_m(previous_red, red, previous_blue, blue)
            red.min_range_m = min(red.min_range_m, previous_range, current_los.range_m, closest)
            miss_distance = min(miss_distance, red.min_range_m)
            if closest <= self.config.missile.lethal_radius_m:
                hit_candidates.append((closest, red_index, target_index))
            elif red.alive and blue.alive and self._post_closest_miss(red, current_los):
                red.alive = False
                loss_reasons[red_index] = "post_closest_miss"
            elif not red.alive:
                loss_reasons[red_index] = self._physical_loss_reason(red)

        hit_pairs: list[tuple[int, int]] = []
        killed_blue: set[int] = set()
        for _, red_index, blue_index in sorted(hit_candidates):
            if blue_index in killed_blue or not current.blue[blue_index].alive:
                continue
            current.red[red_index].alive = False
            current.red[red_index].loss_reason = "valid_hit"
            current.blue[blue_index].alive = False
            killed_blue.add(blue_index)
            hit_pairs.append((red_index, blue_index))
            loss_reasons[red_index] = "valid_hit"

        all_blue_done = not any(target.alive for target in current.blue)
        timeout = current.step_count >= self.config.max_steps
        # The environment time limit is a task failure, not a rollout truncation.
        # Resolve still-flying missiles exactly once so W remains monotonic.
        if timeout and not all_blue_done:
            for red_index, red in enumerate(current.red):
                if red.alive:
                    red.alive = False
                    red.loss_reason = "mission_timeout"
                    loss_reasons[red_index] = "mission_timeout"

        red_loss_events: list[dict[str, object]] = []
        for red_index, (before, after) in enumerate(zip(previous.red, current.red)):
            if before.alive and not after.alive:
                after.loss_reason = loss_reasons.get(red_index, self._physical_loss_reason(after))
                red_loss_events.append(
                    {
                        "red_index": red_index,
                        "blue_index": int(target_indices[red_index]),
                        "loss_reason": after.loss_reason,
                        "time_s": float(current.time_s),
                        "actual_bias_load_body_g": np.asarray(after.bias_load_body_g).tolist(),
                    }
                )
        hit_events = [
            {
                "red_index": red_index,
                "blue_index": blue_index,
                "loss_reason": "valid_hit",
                "time_s": float(current.time_s),
                "actual_bias_load_body_g": np.asarray(current.red[red_index].bias_load_body_g).tolist(),
            }
            for red_index, blue_index in hit_pairs
        ]
        all_red_done = not any(missile.alive for missile in current.red)
        done = all_blue_done or all_red_done or timeout
        termination_reason = (
            "success"
            if all_blue_done
            else "timeout"
            if timeout
            else "red_failure"
            if all_red_done
            else "none"
        )
        if not math.isfinite(miss_distance):
            ranges = [
                los_kinematics(red, blue).range_m
                for red in current.red
                for blue in current.blue
                if red.alive and blue.alive
            ]
            miss_distance = min(ranges) if ranges else 0.0
        return AdjudicationResult(
            reward_high=0.0,
            reward_low=np.zeros(n_red, dtype=np.float32),
            done=done,
            terminated=done,
            truncated=False,
            termination_reason=termination_reason,
            info={
                "hit_count": len(hit_pairs),
                "hit_pairs": [list(pair) for pair in hit_pairs],
                "hit_red_indices": [pair[0] for pair in hit_pairs],
                "hit_events": hit_events,
                "red_loss_events": red_loss_events,
                "miss_distance_m": float(miss_distance),
                "missile_expenditure": sum(not missile.alive for missile in current.red) / max(n_red, 1),
                "all_blue_done": all_blue_done,
                "all_red_done": all_red_done,
                "timeout": timeout,
                "terminated": done,
                "truncated": False,
                "termination_reason": termination_reason,
                "style": current.style,
                "time_s": current.time_s,
                "step_count": current.step_count,
            },
        )

    def _post_closest_miss(self, red, los) -> bool:
        return (
            los.range_m - red.min_range_m >= self.config.missile.post_closest_growth_m
            and max(0.0, -los.closing_speed_mps) >= self.config.missile.post_closest_recede_speed_mps
        )

    def _physical_loss_reason(self, missile) -> str:
        if missile.position_m[1] <= 0.0:
            return "ground_impact"
        if missile.age_s >= self.config.missile.max_guidance_time_s:
            return "guidance_timeout"
        if (
            missile.age_s >= self.config.missile.boost_duration_s
            and norm(missile.velocity_mps) < self.config.missile.min_speed_mps
        ):
            return "low_speed"
        return "physical_failure"

    @staticmethod
    def _segment_closest_distance_m(previous_red, current_red, previous_blue, current_blue) -> float:
        start = previous_blue.position_m - previous_red.position_m
        end = current_blue.position_m - current_red.position_m
        delta = end - start
        denominator = float(np.dot(delta, delta))
        if denominator <= 0.0:
            return float(norm(start))
        fraction = float(np.clip(-np.dot(start, delta) / denominator, 0.0, 1.0))
        return float(norm(start + fraction * delta))
