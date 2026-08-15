
"""Game-theoretic launcher for red missiles (position + launch time)."""

from dataclasses import dataclass
from typing import Tuple, List
import numpy as np



@dataclass
class LaunchRegion:
    num_missiles: int
    candidate_launch_count: int
    x_min: float = 0.0
    x_max: float = 5.0
    y_min: float = -10.0
    y_max: float = 10.0
    z_min: float = 8.0
    z_max: float = 12.0
    num_blue_strategies: int = 8
    fictitious_iters: int = 200
    blue_escape_distance: float = 10.0  # km, only for shaping
    max_launch_time: float = 8.0        # [s]
    min_launch_interval: float = 1.0    # [s]



class GameTheoreticLauncher:
    def __init__(self, region: LaunchRegion, rng: np.random.Generator | None = None) -> None:
        self.region = region
        self.rng = rng or np.random.default_rng()
        self._last_blue_initial_pos = np.zeros(3, dtype=float)

    def get_state(self) -> dict:
        return {"rng_state": self.rng.bit_generator.state}

    def set_state(self, state: dict) -> None:
        rng_state = state.get("rng_state")
        if rng_state is not None:
            self.rng.bit_generator.state = rng_state

    # ------------------------------------------------------------------
    def compute_launch_plan(
        self,
        blue_initial_pos: np.ndarray,
        blue_speed: float,
        missile_speed: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return heuristic launch positions and times for all missiles.

        Returns:
            launch_pos: (M, 3)
            launch_times: (M,) in seconds, strictly increasing with
                at least min_launch_interval separation (if possible).
        """

        M = self.region.num_missiles
        launch_pos = np.zeros((M, 3), dtype=float)
        launch_times = np.zeros(M, dtype=float)
        self._last_blue_initial_pos = np.asarray(blue_initial_pos, dtype=float).copy()

        chosen_times: List[float] = []

        time_grid = np.arange(0.0, self.region.max_launch_time + 1e-6, 1.0)

        for m in range(M):
            allowed_times = self._filter_times(time_grid, chosen_times)
            if len(allowed_times) == 0:
                allowed_times = list(time_grid)

            K = self.region.candidate_launch_count
            cand_pos = self._sample_positions(K)
            cand_times = self._sample_times(K, allowed_times)

            A = self._build_payoff_matrix_for_missile(
                cand_pos,
                cand_times,
                blue_initial_pos,
                blue_speed,
                missile_speed,
            )
            red_mixed = self._fictitious_play_minimax(A)
            idx = self._sample_row_from_mixed(red_mixed)
            launch_pos[m] = cand_pos[idx]
            t_sel = float(cand_times[idx])
            launch_times[m] = t_sel
            chosen_times.append(t_sel)

        # Ensure non-decreasing times
        order = np.argsort(launch_times)
        launch_times = launch_times[order]
        launch_pos = launch_pos[order]
        return launch_pos, launch_times

    # ------------------------------------------------------------------
    # 红弹初始位置设置
    # 之前的设置(红弹在0点附近的一个立方体内发射)
    def _sample_positions(self, K: int) -> np.ndarray:
        x = self.rng.uniform(self.region.x_min, self.region.x_max, size=(K,))
        y = self.rng.uniform(self.region.y_min, self.region.y_max, size=(K,))
        z = self.rng.uniform(self.region.z_min, self.region.z_max, size=(K,))
        return np.stack([x, y, z], axis=1)
    # 新的设置(红弹在蓝弹周围两个立方体之间的区域内发射)
    # def _sample_positions(
    #         self,
    #         K: int,
    #         blue_initial_pos: np.ndarray | None = None,
    # ) -> np.ndarray:
    #     if blue_initial_pos is None:
    #         blue_pos = self._last_blue_initial_pos
    #     else:
    #         blue_pos = np.asarray(blue_initial_pos, dtype=float)
    #     half_outer = 50.0
    #     half_inner = 15.0
    #
    #     offsets = np.zeros((K, 3), dtype=float)
    #     for i in range(K):
    #         while True:
    #             candidate = self.rng.uniform(-half_outer, half_outer, size=3)
    #             if candidate[2] < 1.0:
    #                 continue
    #             if np.all(np.abs(candidate) <= half_inner):
    #                 continue
    #             offsets[i] = candidate
    #             break
    #
    #     return blue_pos[None, :] + offsets

    def _sample_times(self, K: int, allowed_times: List[float]) -> np.ndarray:
        idx = self.rng.integers(0, len(allowed_times), size=K)
        return np.array([allowed_times[i] for i in idx], dtype=float)

    def _filter_times(self, time_grid: np.ndarray, chosen_times: List[float]) -> List[float]:
        if not chosen_times:
            return list(time_grid)
        out: List[float] = []
        for t in time_grid:
            ok = True
            for t_prev in chosen_times:
                if abs(t - t_prev) < self.region.min_launch_interval:
                    ok = False
                    break
            if ok:
                out.append(float(t))
        return out

    def _sample_blue_headings(self, blue_initial_pos: np.ndarray) -> np.ndarray:
        base_dirs = [
            np.array([1.0, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -1.0]),
        ]

        bp = np.asarray(blue_initial_pos, dtype=float)
        norm_bp = np.linalg.norm(bp)
        if norm_bp > 1e-6:
            radial = bp / norm_bp
        else:
            radial = np.array([1.0, 0.0, 0.0])
        base_dirs.append(radial)
        base_dirs.append(-radial)

        dirs = []
        for d in base_dirs:
            n = np.linalg.norm(d)
            if n < 1e-6:
                continue
            dirs.append(d / n)
        if not dirs:
            dirs = [np.array([1.0, 0.0, 0.0])]
        dirs = np.stack(dirs, axis=0)

        D = dirs.shape[0]
        B = self.region.num_blue_strategies
        if B <= D:
            return dirs[:B]
        idx = self.rng.integers(0, D, size=B)
        return dirs[idx]

    def _build_payoff_matrix_for_missile(
        self,
        cand_pos: np.ndarray,
        cand_times: np.ndarray,
        blue_initial_pos: np.ndarray,
        blue_speed: float,
        missile_speed: float,
    ) -> np.ndarray:
        """Build K x B payoff matrix for one missile.

        Payoff is approximate distance between missile launch point and
        blue predicted position at an estimated intercept time.
        """

        K = cand_pos.shape[0]
        headings = self._sample_blue_headings(blue_initial_pos)
        B = headings.shape[0]

        A = np.zeros((K, B), dtype=float)
        blue_initial_pos = np.asarray(blue_initial_pos, dtype=float)

        for i in range(K):
            p = cand_pos[i]
            t_launch = float(cand_times[i])

            r0 = p - blue_initial_pos
            d0 = float(np.linalg.norm(r0))
            if missile_speed <= 1e-6:
                tof = 0.0
            else:
                tof = d0 / missile_speed
            t_eval = t_launch + tof

            for j in range(B):
                h = headings[j]
                blue_future = blue_initial_pos + h * blue_speed * t_eval
                d = np.linalg.norm(p - blue_future)
                A[i, j] = d
        return A

    def _fictitious_play_minimax(self, A: np.ndarray) -> np.ndarray:
        K, B = A.shape
        iters = max(1, self.region.fictitious_iters)

        row_counts = np.zeros(K, dtype=float)
        col_counts = np.zeros(B, dtype=float)

        row_counts[self.rng.integers(0, K)] += 1.0
        col_counts[self.rng.integers(0, B)] += 1.0

        for _ in range(iters):
            row_mixed = row_counts / row_counts.sum()
            col_mixed = col_counts / col_counts.sum()

            row_payoffs = A @ col_mixed
            best_row = int(np.argmin(row_payoffs))
            row_counts[best_row] += 1.0

            col_payoffs = row_mixed @ A
            best_col = int(np.argmax(col_payoffs))
            col_counts[best_col] += 1.0

        red_mixed = row_counts / row_counts.sum()
        return red_mixed

    def _sample_row_from_mixed(self, probs: np.ndarray) -> int:
        p = np.maximum(probs, 0.0)
        s = p.sum()
        if s <= 0.0:
            p = np.ones_like(p) / len(p)
        else:
            p = p / s

        nonzero = int((p > 1e-12).sum())
        replace = nonzero < 1
        idx = self.rng.choice(len(p), size=1, replace=replace, p=p)[0]
        return int(idx)
