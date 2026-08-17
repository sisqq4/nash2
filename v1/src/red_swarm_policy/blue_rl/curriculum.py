from __future__ import annotations

from dataclasses import asdict, dataclass
import random


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    episodes: int
    probabilities: tuple[float, float, float, float]
    score_weights: tuple[float, float, float, float]


DEFAULT_CURRICULUM: tuple[CurriculumStage, ...] = (
    CurriculumStage("A_1v1_foundation", 1000, (1., 0., 0., 0.), (.25, .25, .25, .25)),
    CurriculumStage("B_1v2_entry", 500, (.70, .30, 0., 0.), (.25, .25, .25, .25)),
    CurriculumStage("B_1v2_focus", 1000, (.40, .60, 0., 0.), (.25, .25, .25, .25)),
    CurriculumStage("C_1v3_entry", 500, (.25, .45, .30, 0.), (.25, .25, .25, .25)),
    CurriculumStage("C_1v3_focus", 1000, (.15, .30, .55, 0.), (.25, .25, .25, .25)),
    CurriculumStage("D_1v4_entry", 500, (.15, .25, .35, .25), (.10, .20, .30, .40)),
    CurriculumStage("D_1v4_focus", 1000, (.10, .20, .30, .40), (.10, .20, .30, .40)),
    CurriculumStage("E_balanced", 1000, (.25, .25, .25, .25), (.25, .25, .25, .25)),
)


class CurriculumSchedule:
    """Deterministic episode curriculum with a linear ramp at stage entry."""

    def __init__(self, stages: tuple[CurriculumStage, ...] = DEFAULT_CURRICULUM,
                 transition_episodes: int = 500) -> None:
        if not stages or transition_episodes < 0:
            raise ValueError("curriculum requires stages and a non-negative transition length")
        for stage in stages:
            if stage.episodes < 1 or any(value < 0 for value in stage.probabilities):
                raise ValueError("curriculum episodes and probabilities must be positive")
            if abs(sum(stage.probabilities) - 1.0) > 1e-9:
                raise ValueError("curriculum probabilities must sum to one")
        self.stages, self.transition_episodes = stages, transition_episodes
        self.total_episodes = sum(stage.episodes for stage in stages)

    def stage_at(self, episode: int) -> tuple[int, CurriculumStage, int]:
        if not 1 <= episode <= self.total_episodes:
            raise ValueError(f"episode must be in [1, {self.total_episodes}]")
        offset = episode
        for index, stage in enumerate(self.stages):
            if offset <= stage.episodes:
                return index, stage, offset
            offset -= stage.episodes
        raise AssertionError("unreachable")

    def probabilities_at(self, episode: int) -> tuple[float, float, float, float]:
        index, stage, within = self.stage_at(episode)
        if index == 0 or self.transition_episodes == 0:
            return stage.probabilities
        fraction = min(within / min(self.transition_episodes, stage.episodes), 1.0)
        previous = self.stages[index - 1].probabilities
        return tuple(old + fraction * (new - old)
                     for old, new in zip(previous, stage.probabilities))  # type: ignore[return-value]

    def sample(self, episode: int, rng: random.Random) -> int:
        return rng.choices((1, 2, 3, 4), weights=self.probabilities_at(episode), k=1)[0]

    def describe(self) -> dict[str, object]:
        return {"total_episodes": self.total_episodes, "transition_episodes": self.transition_episodes,
                "stages": [asdict(stage) for stage in self.stages]}


def balanced_score(rates: dict[int, float], weights: tuple[float, float, float, float]) -> float:
    return sum(weights[index - 1] * rates.get(index, 0.0) for index in range(1, 5))


def within_forgetting_limit(rates: dict[int, float], historical_best: dict[int, float],
                            limit: float = 0.05) -> bool:
    return all(rates.get(scenario, 0.0) >= best - limit
               for scenario, best in historical_best.items())
