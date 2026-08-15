
"""CSV and Tacview ACMI export utilities."""

from __future__ import annotations

import os
import glob
import csv
from typing import List, Dict

import numpy as np



def write_csv(
    save_dir: str,
    fname: str,
    data: List[List[float]],
    episode_index: int | None = None,
) -> None:
    """Write trajectory data to CSV.

    Args:
        save_dir: base directory (e.g. EnvConfig.save_dir)
        fname: name without extension, e.g. "plane_blue.1.0"
        data: list of [x, y, z, roll, pitch, yaw]
        episode_index: if set, files go to save_dir/csv/{episode_index}/
    """

    csv_root = os.path.join(save_dir, "csv")
    if episode_index is not None:
        csv_dir = os.path.join(csv_root, str(episode_index))
    else:
        csv_dir = csv_root

    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, fname + ".csv")

    rows = [["x", "y", "z", "roll", "pitch", "yaw"]] + data
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def write_action_csv(
    save_dir: str,
    fname: str,
    data: List[List[float]],
    episode_index: int | None = None,
) -> None:
    """Write blue action data to CSV.

    Args:
        save_dir: base directory (e.g. EnvConfig.save_dir)
        fname: name without extension, e.g. "plane_blue_actions.1"
        data: list of [time, action]
        episode_index: if set, files go to save_dir/csv/{episode_index}/
    """

    csv_root = os.path.join(save_dir, "csv")
    if episode_index is not None:
        csv_dir = os.path.join(csv_root, str(episode_index))
    else:
        csv_dir = csv_root

    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, fname + ".csv")

    rows = [["time", "action"]] + data
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def write_table_csv(
    save_dir: str,
    fname: str,
    headers: List[str],
    data: List[List[float | int | str]],
    episode_index: int | None = None,
) -> None:
    """Write generic tabular data to CSV with custom headers."""

    csv_root = os.path.join(save_dir, "csv")
    if episode_index is not None:
        csv_dir = os.path.join(csv_root, str(episode_index))
    else:
        csv_dir = csv_root

    os.makedirs(csv_dir, exist_ok=True)
    path = os.path.join(csv_dir, fname + ".csv")

    rows = [headers] + data
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

def _read_csv_no_header(path: str) -> List[List[float]]:
    rows: List[List[float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        _ = next(reader, None)
        for r in reader:
            if not r:
                continue
            rows.append([float(v) for v in r])
    return rows

def _insert_in_dict(d: Dict[str, List[str]], key: str, line: str) -> None:
    if key not in d:
        d[key] = []
    d[key].append(line)

def write_acmi(
    target_name: str,
    source_dir: str,
    time_unit: float,
    explode_time: int = 10,
    add_plane_explosion: bool = True,
) -> None:
    """Aggregate CSV files from one episode into a Tacview .acmi file.

    Args:
        target_name: name without extension, e.g. "session_ep0010"
        source_dir: directory containing CSV files for one episode
        time_unit: time interval between samples (seconds)
        explode_time: explosion duration in units of time_unit
    """

    base_dir = os.path.dirname(os.path.dirname(source_dir))
    acmi_dir = os.path.join(base_dir, "acmi")
    os.makedirs(acmi_dir, exist_ok=True)

    target_path = os.path.join(acmi_dir, target_name + ".acmi")

    data_dict: Dict[str, List[str]] = {}
    plane_counts = 0
    missile_counts = 0

    content = (
        "FileType=text/acmi/tacview\n"
        "FileVersion=2.1\n"
        "0,ReferenceTime=2023-12-09T00:00:00Z\n"
    )

    csv_files = glob.glob(os.path.join(source_dir, "*.csv"))

    for file_path in csv_files:
        base = os.path.basename(file_path)
        stem, _ = os.path.splitext(base)

        parts = stem.split(".")
        if len(parts) != 3:
            continue

        type_side = parts[0]
        # parts[1] is an index we do not use directly here
        start_token = parts[2]

        # If start_token is an integer index, interpret it as step index and multiply by time_unit.
        if start_token.isdigit():
            start_time = int(start_token) * time_unit
        else:
            try:
                start_time = float(start_token)
            except ValueError:
                start_time = 0.0

        ts_parts = type_side.split("_")
        if len(ts_parts) != 2:
            continue
        obj_type, side = ts_parts[0], ts_parts[1]

        colors = {"blue": "Blue", "red": "Red"}
        color = colors.get(side, "Blue")

        if obj_type == "plane":
            obj_name = "F16"
            plane_counts += 1
            obj_no = "a" + str(plane_counts)
        else:
            obj_name = "AIM-120"
            missile_counts += 1
            obj_no = "b" + str(missile_counts)

        apdata = _read_csv_no_header(file_path)
        # If there is no data (e.g., missile never launched), skip this object entirely.
        if not apdata:
            continue

        # Coordinate and heading conversion
        for i in range(len(apdata)):
            apdata[i][0] /= 111.3195   # km -> deg (approx)
            apdata[i][1] = -apdata[i][1] / 111.3195
            apdata[i][2] *= 1000
            apdata[i][5] += 90.0       # yaw offset

        for i, line in enumerate(apdata):
            line_string = (
                obj_no
                + ",T="
                + "|".join(str(elem) for elem in line)
                + ",Name="
                + obj_name
                + ",Color="
                + color
                + "\n"
            )
            t = start_time + i * time_unit
            _insert_in_dict(data_dict, str(t), line_string)

        end_time = start_time + len(apdata) * time_unit
        _insert_in_dict(data_dict, str(end_time), "-" + obj_no + "\n")

        if obj_type == "plane" and apdata and add_plane_explosion:
            last = apdata[-1]
            expl_line = (
                obj_no
                + "F,T="
                + "|".join(str(elem) for elem in last)
                + ",Type=Misc+Explosion,Color="
                + color
                + ",Radius=300\n"
            )
            for i in range(explode_time):
                t = end_time + i * time_unit
                _insert_in_dict(data_dict, str(t), expl_line)

    for t in sorted(data_dict.keys(), key=lambda x: float(x)):
        content += "#" + t + "\n"
        for line_string in data_dict[t]:
            content += line_string

    with open(target_path, "w", encoding="utf-8") as f:
        f.write(content)
