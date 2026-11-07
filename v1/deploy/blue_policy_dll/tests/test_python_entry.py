from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRY_DIR = ROOT / "python_app"
BUNDLE_DIR = ROOT.parent / "blue_policy_exe"
sys.path.insert(0, str(ENTRY_DIR))

import blue_policy_entry as entry  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def initialized_policy() -> None:
    info = entry.initialize(str(BUNDLE_DIR))
    assert info["observation_dim"] == 104
    assert info["action_dim"] == 29
    yield
    entry.shutdown()


def test_direct_prediction_matches_golden_reference() -> None:
    with np.load(BUNDLE_DIR / "model" / "golden_network_reference.npz") as golden:
        observation = golden["observations"][0]
        expected_action = int(golden["actions"][0])
    assert entry.predict(observation.tolist()) == expected_action


def test_stateful_exact_session_preserves_full_runtime_contract() -> None:
    handle = entry.create_session(0, 12)
    try:
        result = entry.step(
            handle,
            0,
            0.0,
            (0.0, 10_000.0, 0.0),
            (300.0, 0.0, 0.0),
            True,
            (0.0, 1.0, 0.0),
            [
                (
                    1001,
                    (-50_000.0, 10_000.0, 0.0),
                    (1_000.0, 0.0, 0.0),
                    True,
                    0.6,
                    1,
                )
            ],
        )
        assert result[0] == 2  # rainbow
        assert result[1] is True
        assert result[2] is True
        assert 0 <= result[4] < 29
        assert result[12] == 1
        assert result[13] == (1001,)
    finally:
        entry.destroy_session(handle)


def test_session_requires_strictly_increasing_frames() -> None:
    handle = entry.create_session(1, 3)
    try:
        arguments = (
            handle,
            1401,
            7.005,
            (0.0, 10_000.0, 0.0),
            (300.0, 0.0, 0.0),
            True,
            (0.0, 1.0, 0.0),
            [
                (
                    99,
                    (-50_000.0, 10_000.0, 0.0),
                    (1_000.0, 0.0, 0.0),
                    True,
                    0.0,
                    0,
                )
            ],
        )
        entry.step(*arguments)
        with pytest.raises(ValueError, match="step_count must increase strictly"):
            entry.step(*arguments)
    finally:
        entry.destroy_session(handle)
