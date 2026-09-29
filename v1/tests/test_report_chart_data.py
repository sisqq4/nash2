"""Checks for report-source association and missing-value semantics."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


_SPEC = importlib.util.spec_from_file_location(
    "report_chart_data", Path(__file__).parents[1] / "src/red_swarm_policy/report_chart_data.py")
data = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(data)


def write_trace(root, rows):
    folder = root / "evaluation" / "flight_quality"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evaluation_metadata.json").write_text(json.dumps({"run_id": "test-run"}), encoding="utf-8")
    (folder / "flight_quality_episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def trace_row(episode, loads, run_id="test-run"):
    return {"episode": episode, "blue_survived": False,
            "metadata": {"run_id": run_id, "seed": 100 + episode},
            "trace": {"time_s": [7.0, 7.1, 7.2],
                      "position_m": [[0, 8001, 0], [0, 8000, 0], [0, 8000, 0]],
                      "velocity_mps": [[100, -10, 0], [100, 0, 0], [100, 0, 0]],
                      "speed_mps": [100.5, 100, 100], "commanded_load_g": [None, 9, 9],
                      "estimated_actual_load_g": loads,
                      "min_altitude_m": 8000, "max_altitude_m": 12000}}


def evaluation():
    return {"run_id": "test-run", "results": [
        {"episode": 1, "seed": 101, "blue_survived": False},
        {"episode": 2, "seed": 102, "blue_survived": False}]}


class ReportChartDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def test_out_of_order_trace_matches_episode_and_peak(self):
        write_trace(self.root, [trace_row(2, [1, 40, 1]), trace_row(1, [1, 2, 1])])
        result = data.load_trace_summary(self.root, evaluation())
        self.assertEqual([row["episode"] for row in result["episodes"]], [1, 2])
        self.assertEqual(result["peak"]["episode"], 2)
        self.assertEqual(result["peak"]["previous_vertical_speed_mps"], -10)
        self.assertEqual(result["peak"]["vertical_speed_mps"], 0)
        self.assertEqual(result["global"]["above_20g_samples"], 1)
        self.assertEqual(result["global"]["lower_boundary_contact_episodes"], 2)
        self.assertEqual(len(result["peak_window"]), 3)
        self.assertIsNone(result["peak_window"][0]["commanded_load_g"])

    def test_rejects_mixed_runs(self):
        write_trace(self.root, [trace_row(1, [1, 2, 1], "other-run")])
        with self.assertRaisesRegex(ValueError, "运行编号"):
            data.load_trace_summary(self.root, evaluation())

    def test_rejects_duplicate_episodes(self):
        write_trace(self.root, [trace_row(1, [1, 2, 1]), trace_row(1, [1, 2, 1])])
        with self.assertRaisesRegex(ValueError, "重复"):
            data.load_trace_summary(self.root, evaluation())

    def test_missing_values_and_missing_episodes_stay_visible(self):
        row = trace_row(2, [None, 21])
        row["trace"].pop("commanded_load_g")
        row["trace"].pop("max_altitude_m")
        write_trace(self.root, [row])
        result = data.load_trace_summary(self.root, evaluation())
        self.assertEqual(result["missing"]["episodes"], [1])
        self.assertIsNone(result["episodes"][0]["max_commanded_load_g"])
        self.assertIsNone(result["episodes"][0]["upper_boundary_contact"])
        self.assertEqual(result["episodes"][0]["missing_fields"]["estimated_actual_load_g"], 2)
        self.assertEqual(result["global"]["valid_estimated_load_samples"], 1)
        self.assertEqual(result["global"]["above_20g_samples"], 1)
        self.assertTrue(result["warnings"])

    def test_read_run_checks_sidecar_hash_before_joining(self):
        folder = self.root / "train"
        folder.mkdir()
        (folder / "training_metrics.json").write_text(json.dumps({"episodes": [{"episode": 1}]}), encoding="utf-8")
        (folder / "training_metrics_episode_metadata.json").write_text(json.dumps({
            "metrics_sha256": "stale", "episodes": [{"episode": 1, "blue_orientation": "away"}]}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            data.read_run(self.root)


if __name__ == "__main__":
    unittest.main()
