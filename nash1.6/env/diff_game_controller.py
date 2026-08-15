
"""Differential-game-style controller for missile PN gains."""

from dataclasses import dataclass
import numpy as np

from .missile_dynamics import update_missiles_pn
from config import EnvConfig


@dataclass
class DiffGameConfig:
    step_size: float
    delta_gain: float
    gain_min: float
    gain_max: float
    w_dist: float
    w_gain: float


class DifferentialGameController:
    def __init__(self, cfg: EnvConfig) -> None:
        self.cfg = cfg
        self.dg_cfg = DiffGameConfig(
            step_size=cfg.diff_step_size,
            delta_gain=cfg.diff_delta_gain,
            gain_min=cfg.diff_gain_min,
            gain_max=cfg.diff_gain_max,
            w_dist=cfg.diff_w_dist,
            w_gain=cfg.diff_w_gain,
        )

    def update_nav_gains(
        self,
        blue_pos: np.ndarray,
        blue_vel: np.ndarray,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
        nav_gains: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """One gradient-descent step on nav_gains using finite differences."""
        nav_gains = np.asarray(nav_gains, dtype=float)
        M = nav_gains.shape[0]
        grad = np.zeros_like(nav_gains)
        delta = self.dg_cfg.delta_gain

        for i in range(M):
            nav_plus = nav_gains.copy()
            nav_plus[i] += delta
            cost_plus = self._predict_cost(
                blue_pos, blue_vel, missile_pos, missile_vel, nav_plus, dt
            )

            nav_minus = nav_gains.copy()
            nav_minus[i] -= delta
            cost_minus = self._predict_cost(
                blue_pos, blue_vel, missile_pos, missile_vel, nav_minus, dt
            )

            grad[i] = (cost_plus - cost_minus) / (2.0 * delta)

        new_nav = nav_gains - self.dg_cfg.step_size * grad
        new_nav = np.clip(new_nav, self.dg_cfg.gain_min, self.dg_cfg.gain_max)
        return new_nav

    def _predict_cost(
        self,
        blue_pos: np.ndarray,
        blue_vel: np.ndarray,
        missile_pos: np.ndarray,
        missile_vel: np.ndarray,
        nav_gains: np.ndarray,
        dt: float,
    ) -> float:
        nav_gains = np.asarray(nav_gains, dtype=float)

        m_pos_next, _ = update_missiles_pn(
            missile_pos,
            missile_vel,
            blue_pos,
            blue_vel,
            self.cfg.missile_speed,
            dt,
            nav_gains,
        )

        rel = m_pos_next - blue_pos.reshape(1, 3)
        dist_sq = np.sum(rel ** 2, axis=1)

        J_dist = self.dg_cfg.w_dist * float(np.sum(dist_sq))
        J_gain = self.dg_cfg.w_gain * float(np.sum(nav_gains ** 2))
        return J_dist + J_gain
