"""Independent blue-side reinforcement-learning subsystem."""

from .environment import (BLUE_ACTION_CONTEXT_DIM, BlueEscapeEnv, BlueEscapeEnvConfig,
                          blue_action_context, blue_observation_dim)
from .controller import BlueRLController, BlueRLDecision
from .policy import DiscreteBluePolicy, PolicyRegistry
from .rainbow import RainbowDQNAgent, RainbowDQNConfig
from .env_pool import BlueProcessEnvironmentPool, BlueStepResult
from .curriculum import CurriculumSchedule, CurriculumStage, DEFAULT_CURRICULUM
from .evaluation_shaping import EvaluationActionShaper, EvaluationShapingConfig
from .flight_envelope import FlightEnvelopeConfig, FlightEnvelopeConstraintLayer
from .mechanism_reward import (BlueMechanismStateEstimator, MechanismRewardConfig,
                               MECHANISM_REWARD_NAMES, encode_normalized_v4,
                               mechanism_observation_dim, parse_mechanism_rewards)
from .minimal_input import (BlueMinimalAdapterOutput, BlueMinimalInput,
                            BlueMinimalInputAdapter, BlueMissileTrack,
                            blue_minimal_input_from_simulator)

from .flight_quality import (FlightQualityTracker, append_flight_quality_episode,
                             replot_flight_quality_episodes, write_flight_quality_report)


__all__ = [
    "BlueEscapeEnv",
    "BlueEscapeEnvConfig",
    "BLUE_ACTION_CONTEXT_DIM",
    "blue_action_context",
    "blue_observation_dim",
    "BlueRLController",
    "BlueRLDecision",
    "DiscreteBluePolicy",
    "PolicyRegistry",
    "RainbowDQNAgent",
    "RainbowDQNConfig",
    "BlueProcessEnvironmentPool",
    "BlueStepResult",
    "CurriculumSchedule",
    "CurriculumStage",
    "DEFAULT_CURRICULUM",
    "EvaluationActionShaper",
    "EvaluationShapingConfig",
    "FlightEnvelopeConfig",
    "FlightEnvelopeConstraintLayer",
    "BlueMechanismStateEstimator",
    "MechanismRewardConfig",
    "MECHANISM_REWARD_NAMES",
    "parse_mechanism_rewards",
    "encode_normalized_v4",
    "mechanism_observation_dim",
    "BlueMissileTrack",
    "BlueMinimalInput",
    "BlueMinimalAdapterOutput",
    "BlueMinimalInputAdapter",
    "blue_minimal_input_from_simulator",
    "FlightQualityTracker",
    "append_flight_quality_episode",
    "replot_flight_quality_episodes",
    "write_flight_quality_report",
]
