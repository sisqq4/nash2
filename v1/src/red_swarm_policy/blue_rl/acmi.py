from __future__ import annotations

import math
from pathlib import Path

from ..env.types import EngagementState


class AcmiRecorder:
    """Tacview 2.2 text recorder used identically by training and evaluation."""

    def __init__(self) -> None:
        self.frames: list[EngagementState] = []

    def record(self, state: EngagementState) -> None:
        self.frames.append(state.copy())

    def save(self, path: str | Path) -> Path:
        destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
        lines = ["FileType=text/acmi/tacview", "FileVersion=2.2", "0,ReferenceTime=2026-01-01T00:00:00Z"]
        for frame in self.frames:
            lines.append(f"#{frame.time_s:.3f}")
            for index, entity in enumerate(frame.blue):
                x, altitude, east = entity.position_m
                north_speed, vertical_speed, east_speed = entity.velocity_mps
                horizontal_speed = math.hypot(north_speed, east_speed)
                pitch_deg = math.degrees(math.atan2(vertical_speed, horizontal_speed))
                yaw_deg = math.degrees(math.atan2(east_speed, north_speed))
                roll_deg = math.degrees(entity.bank_angle_rad)
                transform = (
                    f"{east / 111320:.8f}|{x / 111320:.8f}|{altitude:.2f}|"
                    f"{roll_deg:.4f}|{pitch_deg:.4f}|{yaw_deg:.4f}"
                )
                lines.append(
                    f"{100 + index},T={transform},Name=Blue-{index + 1},"
                    "Type=Air+FixedWing,Coalition=Blue"
                )
            for index, entity in enumerate(frame.red):
                x, altitude, east = entity.position_m
                line = f"{200 + index},T={east / 111320:.8f}|{x / 111320:.8f}|{altitude:.2f},Name=Red-Missile-{index + 1},Type=Weapon+Missile,Coalition=Red"
                if not entity.alive: line += ",Destroyed=1"
                lines.append(line)
        destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return destination
