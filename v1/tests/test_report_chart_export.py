"""Small offline checks for report statistics; no model or simulation imports."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest


SCRIPT_DIRECTORY = Path(__file__).resolve().parents[1] / "src" / "red_swarm_policy"
sys.path.insert(0, str(SCRIPT_DIRECTORY))
import export_blue_report_charts as charts


def training_fixture():
    # Parallel workers finish high-numbered episodes before earlier assignments.
    return {
        "iterations": [
            {
                "episode_ids": [7, 3],
                "completed_episodes": 2,
                "reward_mean": -8.0,
                "survival_rate": 0.0,
                "simulation_time_mean_s": 100.0,
                "optimizer_updates_in_window": 10,
            },
            {
                "episode_ids": [2, 1],
                "completed_episodes": 4,
                "reward_mean": 2.0,
                "survival_rate": 0.5,
                "simulation_time_mean_s": 150.0,
                "optimizer_updates_in_window": 30,
            },
        ]
    }


def evaluation_fixture():
    return {
        "episodes": 2,
        "results": [
            {
                "episode": 7,
                "blue_survived": True,
                "blue_orientation": "toward_missile_swarm",
                "missile_count": 1,
                "reward": 10.0,
                "miss_distance_m": 20.0,
                "simulation_time_s": 180.0,
                "decision_steps": 1_000,
            },
            {
                "episode": 2,
                "blue_survived": False,
                "blue_orientation": "away_from_missile_swarm",
                "missile_count": 2,
                "reward": -10.0,
                "miss_distance_m": 2.0,
                "simulation_time_s": 90.0,
                "decision_steps": 300,
            },
        ],
    }


class WilsonIntervalTests(unittest.TestCase):
    def test_known_balanced_sample(self):
        low, high = charts.wilson(50, 100)
        self.assertAlmostEqual(low, 0.4038315303659957, places=11)
        self.assertAlmostEqual(high, 0.5961684696340044, places=11)

    def test_zero_successes_still_has_positive_upper_bound(self):
        low, high = charts.wilson(0, 10)
        self.assertAlmostEqual(low, 0.0, places=14)
        self.assertAlmostEqual(high, 0.2775327998628892, places=11)

    def test_no_observations_has_no_interval(self):
        self.assertEqual(charts.wilson(0, 0), (None, None))


class SmoothingTests(unittest.TestCase):
    def test_trailing_average_uses_no_future_observations(self):
        values = [1.0, 3.0, 100.0, 200.0]
        self.assertEqual(charts.smooth(values, 2), [1.0, 2.0, 51.5, 150.0])
        before_future_arrival = charts.smooth(values[:3], 2)
        after_future_arrival = charts.smooth(values[:3] + [100_000.0], 2)
        self.assertEqual(before_future_arrival, after_future_arrival[:3])

    def test_missing_point_remains_gap_and_is_not_zero(self):
        self.assertEqual(
            charts.smooth([1.0, None, 5.0, 9.0, None, 13.0], 2),
            [1.0, None, 5.0, 7.0, None, 13.0],
        )
        self.assertEqual(charts.smooth([None, None, 6.0], 3), [None, None, 6.0])


class RecordValidationTests(unittest.TestCase):
    def test_parallel_training_windows_keep_completion_order(self):
        training = training_fixture()
        original = deepcopy(training)
        windows, evaluation_rows = charts.validate_data(training, evaluation_fixture())
        self.assertEqual([w["completed_episodes"] for w in windows], [2, 4])
        self.assertEqual([w["episode_ids"] for w in windows], [[7, 3], [2, 1]])
        self.assertEqual([w["reward_mean"] for w in windows], [-8.0, 2.0])
        self.assertEqual(training, original)
        self.assertEqual([r["episode"] for r in evaluation_rows], [2, 7])

    def test_duplicate_training_ids_are_rejected_within_and_across_windows(self):
        for duplicate_ids in ([2, 2], [7, 1]):
            with self.subTest(duplicate_ids=duplicate_ids):
                training = training_fixture()
                training["iterations"][1]["episode_ids"] = duplicate_ids
                with self.assertRaises(ValueError):
                    charts.validate_data(training, evaluation_fixture())

    def test_duplicate_evaluation_ids_are_rejected(self):
        evaluation = evaluation_fixture()
        evaluation["results"][1]["episode"] = 7
        with self.assertRaises(ValueError):
            charts.validate_data(training_fixture(), evaluation)

    def test_completion_count_must_match_actual_window_membership(self):
        training = training_fixture()
        training["iterations"][1]["completed_episodes"] = 7
        with self.assertRaises(ValueError):
            charts.validate_data(training, evaluation_fixture())

    def test_missing_survival_is_not_silently_treated_as_failure(self):
        evaluation = evaluation_fixture()
        del evaluation["results"][0]["blue_survived"]
        with self.assertRaises(ValueError):
            charts.validate_data(training_fixture(), evaluation)

    def test_survival_requires_boolean_not_null_number_or_string(self):
        for invalid in (None, 0, 1, "False"):
            with self.subTest(invalid=invalid):
                evaluation = evaluation_fixture()
                evaluation["results"][0]["blue_survived"] = invalid
                with self.assertRaises(ValueError):
                    charts.validate_data(training_fixture(), evaluation)


class SeparateInputDirectoryTests(unittest.TestCase):
    def test_training_and_evaluation_can_be_read_from_independent_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            training_dir = root / "saved_training"
            evaluation_dir = root / "new_evaluation"
            quality_dir = evaluation_dir / "flight_quality"
            training_dir.mkdir()
            quality_dir.mkdir(parents=True)

            training_payload = {"iterations": [], "episodes": []}
            evaluation_payload = {"run_id": "new-run", "results": []}
            metadata_payload = {"run_id": "new-run"}
            (training_dir / "training_metrics.json").write_text(
                json.dumps(training_payload), encoding="utf-8")
            (evaluation_dir / "evaluation.json").write_text(
                json.dumps(evaluation_payload), encoding="utf-8")
            (quality_dir / "evaluation_metadata.json").write_text(
                json.dumps(metadata_payload), encoding="utf-8")
            (quality_dir / "flight_quality_episodes.jsonl").write_text("", encoding="utf-8")

            training, evaluation, metadata = charts.read_run(
                root, training_dir=training_dir, evaluation_dir=evaluation_dir)
            trace = charts.load_trace_summary(
                root, evaluation, evaluation_dir=evaluation_dir)

            self.assertEqual(Path(training["source"]),
                             training_dir / "training_metrics.json")
            self.assertEqual(Path(evaluation["source"]),
                             evaluation_dir / "evaluation.json")
            self.assertEqual(Path(metadata["source"]),
                             quality_dir / "evaluation_metadata.json")
            self.assertEqual(Path(trace["source"]),
                             quality_dir / "flight_quality_episodes.jsonl")
            self.assertEqual(trace["metadata"]["run_id"], "new-run")


class GroupStatisticsTests(unittest.TestCase):
    def test_empty_group_keeps_missing_statistics(self):
        self.assertEqual(
            charts.group_stats("无样本", []),
            ["无样本", 0, 0, None, None, None, None, None],
        )

    def test_group_counts_episodes_and_exports_percent_not_fraction(self):
        group = charts.group_stats("示例", evaluation_fixture()["results"])
        self.assertEqual(group[:4], ["示例", 2, 1, 50.0])
        self.assertLess(group[4], 50.0)
        self.assertGreater(group[5], 50.0)
        self.assertEqual(group[6:], [11.0, 11.0])


if __name__ == "__main__":
    unittest.main()
