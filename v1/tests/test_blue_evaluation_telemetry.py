from __future__ import annotations

import copy
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from red_swarm_policy.blue_rl import BlueEscapeEnv, BlueEscapeEnvConfig, RainbowDQNAgent, RainbowDQNConfig
from red_swarm_policy.blue_rl.flight_quality import (
    FlightQualityTracker, _build_episode_figure, append_flight_quality_episode, write_flight_quality_report,
)
from red_swarm_policy.env import EnvironmentConfig


def _sample(time_s: float, step: int) -> dict:
    return {
        "time_s": time_s, "step_count": step,
        "blue_position_m": [300.0 * time_s, 10_000.0, 0.0],
        "blue_velocity_mps": [300.0, 0.0, 0.0],
        "min_altitude_m": 8000.0, "max_altitude_m": 12000.0,
        "red_ids": [0, 1, 2],
        "red_positions_m": [[time_s, 9000.0, 0.0], [2 * time_s, 9000.0, 0.0], None],
        "red_velocities_mps": [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], None],
        "red_alive": [True, True, None], "red_loss_reasons": [None, None, None],
        "red_termination_events": [],
    }


def test_episode_archive_aligns_states_masks_missing_data_and_retains_terminal_frame(tmp_path: Path) -> None:
    tracker = FlightQualityTracker()
    initial = _sample(7.0, 1400)
    tracker.add(initial)
    initial["red_positions_m"][0][0] = 999.0  # Source mutations must not rewrite archived samples.
    event = {"red_id": 1, "time_s": 7.055, "step_count": 1411, "reason": "test_removed",
             "alive": False, "position_m": [14.11, 9000.0, 0.0], "velocity_mps": [2.0, 0.0, 0.0],
             "blue_position_m": [2116.5, 10_000.0, 0.0], "blue_velocity_mps": [300.0, 0.0, 0.0]}
    for now, step in [(7.1, 1420), (7.105, 1421)]:
        sample = _sample(now, step)
        sample["red_alive"][1] = False
        sample["red_loss_reasons"][1] = "test_removed"
        sample["red_positions_m"][1] = event["position_m"].copy()
        sample["red_termination_events"] = [event]  # Repeated delivery must not duplicate the event.
        sample["red_velocities_mps"][2] = [float("nan"), float("inf"), 0.0]
        tracker.add(sample)
    episode = tracker.finish(episode=3, survived=True, metadata={"seed": 123})
    trace = episode["trace"]
    assert trace["red_ids"] == [0, 1, 2]
    assert trace["time_s"] == [7.0, 7.1, 7.105]
    assert trace["step_count"] == [1400, 1420, 1421]
    assert trace["sample_interval_s"] == pytest.approx([0.0, .1, .005])
    assert trace["red_positions_m"][0][0][0] == 7.0
    assert trace["red_positions_m"][2][2] == [None, None, None]
    assert trace["red_velocities_mps"][2][2] == [None, None, 0.0]
    assert trace["red_position_valid"] == [[True, True, False]] * 3
    assert trace["red_state_valid"] == [[True, True, False], [True, False, False], [True, False, False]]
    assert trace["red_alive"][2] == [True, False, None]
    assert len(episode["red_termination_events"]) == 1
    assert episode["red_termination_events"][0]["time_s"] == 7.055
    event["position_m"][0] = -999.0
    assert episode["red_termination_events"][0]["position_m"][0] == 14.11
    original = json.dumps(episode, allow_nan=False)
    append_flight_quality_episode(episode, tmp_path / "episodes.jsonl")
    write_flight_quality_report([episode], tmp_path, plot_limit=0)
    assert json.loads((tmp_path / "episodes.jsonl").read_text()) == episode
    assert json.loads((tmp_path / "flight_quality.json").read_text())["episodes"] == [episode]
    figure = _build_episode_figure(episode)
    try:
        # Horizontal plot: blue, red 0, red 1, red 2.  Red 1 ends at the event itself.
        assert figure.axes[1].lines[2].get_xdata().tolist() == [14.0, 14.11]
        figure.canvas.draw()
    finally:
        figure.clear()
    assert json.dumps(episode, allow_nan=False) == original


def test_telemetry_rejects_reordered_ids_and_nonincreasing_time() -> None:
    tracker = FlightQualityTracker()
    tracker.add(_sample(7.0, 1400))
    reordered = _sample(7.1, 1420)
    reordered["red_ids"] = [1, 0, 2]
    with pytest.raises(ValueError, match="slots must remain fixed"):
        tracker.add(reordered)
    with pytest.raises(ValueError, match="strictly increasing"):
        tracker.add(_sample(7.0, 1400))


def test_physics_step_termination_is_recorded_before_decision_endpoint_and_reset_clears_it(monkeypatch) -> None:
    base = EnvironmentConfig()
    env = BlueEscapeEnv(replace(base, max_steps=base.policy_entry_steps + 60), BlueEscapeEnvConfig(
        missile_count=2, record_acmi=False, record_red_telemetry=True,
    ))
    _, info = env.reset(seed=5)
    initial = info["flight_quality_state"]
    tracker = FlightQualityTracker()
    tracker.add(initial)
    original_step = env.inner.step
    calls = 0
    terminal_state = {}

    def step_with_removal(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        calls += 1
        if calls == 2:
            env.inner.state.red[0].alive = False
            env.inner.state.red[0].loss_reason = "test_removed"
            terminal_state.update({"time_s": env.inner.state.time_s,
                                   "position_m": env.inner.state.red[0].position_m.tolist(),
                                   "blue_position_m": env.inner.state.blue[0].position_m.tolist()})
        return result

    monkeypatch.setattr(env.inner, "step", step_with_removal)
    _, _, _, _, info = env.step(0)
    snapshot = info["flight_quality_state"]
    tracker.add(snapshot)
    event = snapshot["red_termination_events"][0]
    assert event["red_id"] == 0 and event["reason"] == "test_removed"
    assert event["time_s"] == pytest.approx(initial["time_s"] + 2 * base.time_step_s)
    assert event["time_s"] < snapshot["time_s"]
    assert event["step_count"] == initial["step_count"] + 2
    assert event["position_m"] == terminal_state["position_m"]
    assert event["blue_position_m"] == terminal_state["blue_position_m"]
    assert snapshot["red_alive"] == [False, True]
    assert snapshot["red_ids"] == initial["red_ids"] == [0, 1]
    _, _, _, _, next_info = env.step(0)
    assert next_info["flight_quality_state"]["red_termination_events"] == []
    _, reset_info = env.reset(seed=6, episode_index=2, missile_count=1)
    assert reset_info["flight_quality_state"]["red_ids"] == [0]
    assert reset_info["flight_quality_state"]["red_termination_events"] == []
    assert tracker.finish(episode=1, survived=True)["red_termination_events"][0]["time_s"] == event["time_s"]


def test_passive_recording_does_not_change_simulation_or_observations() -> None:
    base = EnvironmentConfig()
    cfg = replace(base, max_steps=base.policy_entry_steps + 23)
    plain = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(record_acmi=False))
    recorded = BlueEscapeEnv(cfg, BlueEscapeEnvConfig(record_acmi=False, record_red_telemetry=True))
    original, _ = plain.reset(seed=9)
    observed, _ = recorded.reset(seed=9)
    np.testing.assert_array_equal(original, observed)
    for action in [0, 1]:
        original_result = plain.step(action)
        recorded_result = recorded.step(action)
        np.testing.assert_array_equal(original_result[0], recorded_result[0])
        assert original_result[1:4] == recorded_result[1:4]
        original_info, recorded_info = copy.deepcopy(original_result[4]), copy.deepcopy(recorded_result[4])
        original_info.pop("flight_quality_state")
        recorded_info.pop("flight_quality_state")
        assert original_info == recorded_info


def test_model_evaluation_writes_all_episode_traces_and_reproducibility_manifest(tmp_path: Path, monkeypatch) -> None:
    from red_swarm_policy import evaluate_blue_rl

    # Exercise real spawned workers and the CLI save path with short test episodes.
    base = EnvironmentConfig()
    monkeypatch.setattr(evaluate_blue_rl, "configure_blue_mission_duration",
                        lambda config: replace(config, max_steps=base.policy_entry_steps + 23))
    agent = RainbowDQNAgent(RainbowDQNConfig(observation_dim=12, action_dim=29,
                                           observation_schema="legacy_v1", device="cpu",
                                           hidden_dim=32, replay_size=100))
    checkpoint = tmp_path / "blue.pt"
    agent.save(checkpoint)
    output = tmp_path / "evaluation"
    monkeypatch.setattr(sys, "argv", [
        "evaluate_blue_rl", str(checkpoint), "--device", "cpu", "--missiles", "1,2", "--episodes", "3", "--seed", "18",
        "--parallel-envs", "2", "--acmi-interval", "0", "--flight-quality-plot-limit", "0",
        "--no-result-plots", "--output", str(output),
    ])
    assert evaluate_blue_rl.main() == 0
    manifest = json.loads((output / "flight_quality" / "evaluation_metadata.json").read_text())
    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert len(manifest["code"]["source_sha256"]) == 64
    assert "evaluate_blue_rl.py" in manifest["code"]["files_sha256"]
    assert manifest["environment_config"]["max_steps"] == base.policy_entry_steps + 23
    assert manifest["adapter_config"]["record_red_telemetry"] is True
    episodes = [json.loads(line) for line in
                (output / "flight_quality" / "flight_quality_episodes.jsonl").read_text().splitlines()]
    consolidated = json.loads((output / "flight_quality" / "flight_quality.json").read_text())
    summary = json.loads((output / "evaluation.json").read_text())
    assert sorted(episodes, key=lambda row: row["episode"]) == consolidated["episodes"]
    assert sorted(row["episode"] for row in episodes) == [1, 2, 3]
    for episode in episodes:
        metadata, trace = episode["metadata"], episode["trace"]
        assert metadata["run_id"] == manifest["run_id"] == summary["run_id"]
        assert metadata["seed"] == 18 + episode["episode"]
        assert metadata["manifest"] == "evaluation_metadata.json"
        assert trace["red_ids"] == list(range(metadata["missile_count"]))
        assert trace["sample_interval_s"] == pytest.approx([0, .1, .015])
        for key in ["red_positions_m", "red_velocities_mps", "red_alive", "red_loss_reasons",
                    "red_position_valid", "red_velocity_valid", "red_state_valid"]:
            assert len(trace[key]) == len(trace["time_s"]) == 3
            assert all(len(frame) == metadata["missile_count"] for frame in trace[key])
    assert summary["learner_state_unchanged"] is True
    assert not (output / "acmi").exists()
