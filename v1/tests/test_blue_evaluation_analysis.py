from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from red_swarm_policy.analyze_blue_evaluations import (
    extract,
    inspect_rows,
    load_input,
    main,
    summarize,
    validate_rows,
)


def _episodes(survived: tuple[bool, ...], *, orientation: str = "toward") -> list[dict[str, object]]:
    return [{"episode": index, "missile_count": 2, "blue_orientation": orientation,
             "blue_survived": value, "miss_distance_m": float(index * 10), "reward": float(index),
             "simulation_time_s": 5.0, "decision_steps": 4, "hit_count": int(not value),
             "termination_reason": "escaped" if value else "hit",
             "mechanism": {"main_threat_switches": index - 1}}
            for index, value in enumerate(survived, 1)]


def test_summary_includes_requested_statistics_and_switches() -> None:
    result = summarize(_episodes((True, False, True, True)))
    assert result["escape_rate"] == pytest.approx(.75)
    assert result["miss_distance_m"]["mean"] == pytest.approx(25)
    assert result["miss_distance_m"]["p25"] == pytest.approx(17.5)
    assert result["miss_distance_m"]["p75"] == pytest.approx(32.5)
    assert result["main_threat_switches"]["mean"] == pytest.approx(1.5)
    assert 0 < result["escape_rate_ci95_low"] < result["escape_rate_ci95_high"] < 1


def test_extract_computes_paired_group_deltas(tmp_path: Path) -> None:
    base, policy = _episodes((False, True)), _episodes((True, True))
    report, rows = extract([("base", tmp_path / "base.json", base),
                            ("rl", tmp_path / "rl.json", policy)],
                           ("missile_count", "blue_orientation"), "base")
    rl_overall = next(row for row in rows if row["policy"] == "rl" and row["level"] == "overall")
    assert rl_overall["escape_rate_delta"] == pytest.approx(.5)
    assert len(report["groups"]) == 4


def test_cli_reads_evaluation_and_writes_json_csv_without_plots(tmp_path: Path) -> None:
    source = tmp_path / "run"; source.mkdir()
    (source / "evaluation.json").write_text(json.dumps({"results": _episodes((True, False))}),
                                             encoding="utf-8")
    label, path, rows = load_input(f"rainbow={source}")
    assert (label, path.name, len(rows)) == ("rainbow", "evaluation.json", 2)
    output = tmp_path / "report"
    assert main([f"rainbow={source}", "--output", str(output), "--no-plots"]) == 0
    assert json.loads((output / "analysis.json").read_text(encoding="utf-8"))["baseline"] == "rainbow"
    with (output / "metrics.csv").open(encoding="utf-8") as stream:
        table = list(csv.DictReader(stream))
    assert {row["level"] for row in table} == {"overall", "stratified"}
    with (output / "episodes.csv").open(encoding="utf-8") as stream:
        episodes = list(csv.DictReader(stream))
    assert len(episodes) == 2
    assert episodes[0]["miss_distance_m"] == "10.0"


def test_data_quality_rejects_missing_survival_instead_of_counting_failure(tmp_path: Path) -> None:
    rows = _episodes((True,))
    del rows[0]["blue_survived"]
    with pytest.raises(ValueError, match="blue_survived"):
        validate_rows(rows, tmp_path / "bad.json")


def test_disabled_mechanism_empty_trace_is_not_reported_as_zero_switches() -> None:
    rows = _episodes((True,))
    rows[0]["mechanism"] = {"enabled": False, "main_threat_switches": 0}
    quality = inspect_rows(rows)
    assert quality["main_threat_switches_available"] == 0
    assert summarize(rows)["main_threat_switches"]["count"] == 0


def test_cli_can_skip_invalid_non_baseline_input(tmp_path: Path) -> None:
    source = tmp_path / "valid"; source.mkdir()
    (source / "evaluation.json").write_text(json.dumps({"results": _episodes((True,))}),
                                             encoding="utf-8")
    output = tmp_path / "report"
    assert main([f"rainbow={source}", f"broken={tmp_path / 'missing'}",
                 "--skip-invalid-inputs", "--output", str(output), "--no-plots"]) == 0
    report = json.loads((output / "analysis.json").read_text(encoding="utf-8"))
    assert report["skipped_inputs"][0]["input"].startswith("broken=")
