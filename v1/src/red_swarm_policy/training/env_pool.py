from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from ..env import EnvironmentObservation, RedBlueEngagementEnv, ScenarioStyle
from ..policy.actor import OverloadBiasActorInputs


@dataclass(frozen=True)
class EnvironmentAdvanceResult:
    observation: EnvironmentObservation
    reward_high: float
    reward_low: np.ndarray
    done: bool
    info: dict[str, Any]
    frames_executed: int
    hit_count: int
    control_effort: float
    low_reward_component_sums: dict[str, float]
    time_credit_unassigned_count: int


@dataclass(frozen=True)
class EnvironmentWorkerInfo:
    index: int
    pid: int
    torch_threads: int
    torch_interop_threads: int


class EnvironmentWorkerError(RuntimeError):
    pass


def _advance_environment(
    env: RedBlueEngagementEnv,
    target_indices: np.ndarray,
    guidance_bias: np.ndarray,
    frame_count: int,
    collect_metrics: bool = False,
) -> EnvironmentAdvanceResult:
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if env.state is None:
        raise RuntimeError("environment state is unavailable")
    reward_high = 0.0
    reward_low = np.zeros(len(env.state.red), dtype=np.float64)
    final_result = None
    frames_executed = 0
    hit_count = 0
    low_reward_component_sums = {
        key: 0.0
        for key in (
            "hit_event",
            "miss_event",
            "potential_delta",
            "control_effort_increment",
            "time_credit",
            "reward_low",
        )
    }
    time_credit_unassigned_count = 0
    red_action = {
        "target_indices": np.asarray(target_indices, dtype=np.int64),
        "guidance_bias": np.asarray(guidance_bias, dtype=np.float64),
    }
    for _ in range(frame_count):
        assert env.state is not None
        final_result = env.step(red_action=red_action)
        frames_executed += 1
        reward_high += float(final_result.reward_high)
        reward_low += np.asarray(final_result.reward_low, dtype=np.float64)
        if collect_metrics:
            hit_count += int(final_result.info.get("hit_count", 0))
            components = final_result.info.get("reward_low_components", {})
            if (
                final_result.info.get("low_reward_settled", False)
                and isinstance(components, dict)
            ):
                for key in low_reward_component_sums:
                    raw = components.get(key, 0.0)
                    values = np.asarray(raw, dtype=np.float64)
                    if values.size and np.isfinite(values).all():
                        low_reward_component_sums[key] += float(values.sum())
                time_credit_unassigned_count += int(
                    bool(components.get("time_credit_unassigned", False))
                )
        if final_result.done or final_result.info.get("high_reward_settled", False):
            break
    assert final_result is not None
    return EnvironmentAdvanceResult(
        observation=final_result.observation,
        reward_high=reward_high,
        reward_low=reward_low,
        done=bool(final_result.done),
        info=dict(final_result.info),
        frames_executed=frames_executed,
        hit_count=hit_count,
        control_effort=float(final_result.info.get("control_effort", env.control_effort)),
        low_reward_component_sums=low_reward_component_sums,
        time_credit_unassigned_count=time_credit_unassigned_count,
    )


class ThreadEnvironmentPool:
    """In-process compatibility backend used by direct rollout callers."""

    def __init__(self, envs: Sequence[RedBlueEngagementEnv]) -> None:
        if not envs:
            raise ValueError("envs must contain at least one environment")
        self.envs = list(envs)
        self.size = len(self.envs)
        self._executor = ThreadPoolExecutor(max_workers=self.size)
        self._closed = False

    def reset(
        self,
        *,
        seed: int | None,
        style: ScenarioStyle | None,
        red_count: int | None,
        blue_count: int | None,
        red_counts: Sequence[int | None] | None = None,
        blue_counts: Sequence[int | None] | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> list[EnvironmentObservation]:
        base_seed = 0 if seed is None else seed
        if red_counts is not None and len(red_counts) != self.size:
            raise ValueError("red_counts must match environment count")
        if blue_counts is not None and len(blue_counts) != self.size:
            raise ValueError("blue_counts must match environment count")
        if seeds is not None and len(seeds) != self.size:
            raise ValueError("seeds must match environment count")

        def reset_one(item: tuple[int, RedBlueEngagementEnv]) -> EnvironmentObservation:
            index, env = item
            return env.reset(
                seed=(
                    seeds[index]
                    if seeds is not None
                    else None if seed is None else base_seed + index * 1_000_003
                ),
                style=style,
                red_count=(red_counts[index] if red_counts is not None else red_count),
                blue_count=(
                    blue_counts[index]
                    if blue_counts is not None
                    else blue_count
                ),
                start_mode="post_boost",
            )

        return list(self._executor.map(reset_one, enumerate(self.envs)))

    def execution_inputs(
        self,
        target_slots: np.ndarray,
        record_assignment: np.ndarray,
    ) -> list[OverloadBiasActorInputs]:
        slots = np.asarray(target_slots, dtype=np.int64)
        active = np.asarray(record_assignment, dtype=bool)
        if slots.shape[0] != self.size or active.shape != (self.size,):
            raise ValueError("execution input batch does not match environment count")

        def build(index: int) -> OverloadBiasActorInputs:
            env = self.envs[index]
            if env.state is None:
                raise RuntimeError("parallel environment state is unavailable")
            n_blue = len(env.state.blue)
            n_red = len(env.state.red)
            safe_slots = np.asarray(slots[index, :n_red], dtype=np.int64).copy()
            safe_slots[(safe_slots < 0) | (safe_slots > n_blue)] = 0
            if active[index]:
                env.record_network_call("assignment_actor")
            return env.observation_layer.execution_inputs(env.state, safe_slots)

        return list(self._executor.map(build, range(self.size)))

    def advance(
        self,
        active_indices: Sequence[int],
        target_indices: np.ndarray,
        guidance_bias: np.ndarray,
        frame_count: int,
        record_critic: bool = True,
        collect_metrics: bool = False,
    ) -> dict[int, EnvironmentAdvanceResult]:
        indices = [int(index) for index in active_indices]
        targets = np.asarray(target_indices, dtype=np.int64)
        biases = np.asarray(guidance_bias, dtype=np.float64)

        def advance_one(index: int) -> EnvironmentAdvanceResult:
            env = self.envs[index]
            if env.state is None:
                raise RuntimeError("parallel environment state is unavailable")
            n_red = len(env.state.red)
            safe_targets = np.asarray(targets[index, :n_red], dtype=np.int64).copy()
            n_blue = len(env.state.blue)
            safe_targets[(safe_targets < -1) | (safe_targets >= n_blue)] = -1
            env.record_network_call("execution_actor")
            if record_critic:
                env.record_network_call("execution_critic")
            return _advance_environment(
                env,
                safe_targets,
                biases[index, :n_red],
                frame_count,
                collect_metrics=collect_metrics,
            )

        results = self._executor.map(advance_one, indices)
        return dict(zip(indices, results))

    def sync_states(self, envs: Sequence[RedBlueEngagementEnv]) -> None:
        if len(envs) != self.size:
            raise ValueError("environment count does not match pool size")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except TypeError:
            # Python 3.8's ThreadPoolExecutor has no cancel_futures parameter.
            self._executor.shutdown(wait=True)

    def __enter__(self) -> "ThreadEnvironmentPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _send_bytes(connection: Connection, value: Any) -> None:
    connection.send_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _receive_bytes(connection: Connection) -> Any:
    return pickle.loads(connection.recv_bytes())


def _worker_error() -> dict[str, str]:
    error = traceback.format_exc()
    exception = error.strip().splitlines()[-1] if error.strip() else "unknown worker error"
    return {"exception": exception, "traceback": error}


def _environment_worker(
    connection: Connection,
    env: RedBlueEngagementEnv,
    worker_index: int,
    native_threads: int,
) -> None:
    try:
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        ):
            os.environ[name] = str(native_threads)
        os.environ["MKL_THREADING_LAYER"] = "GNU"
        torch.set_num_threads(native_threads)
        try:
            torch.set_num_interop_threads(native_threads)
        except RuntimeError:
            pass
        _send_bytes(
            connection,
            (
                True,
                EnvironmentWorkerInfo(
                    index=worker_index,
                    pid=os.getpid(),
                    torch_threads=torch.get_num_threads(),
                    torch_interop_threads=torch.get_num_interop_threads(),
                ),
            ),
        )
    except Exception:
        _send_bytes(connection, (False, _worker_error()))
        connection.close()
        return

    try:
        while True:
            try:
                operation, payload = _receive_bytes(connection)
            except EOFError:
                break
            if operation == "close":
                _send_bytes(connection, (True, None))
                break
            try:
                if operation == "reset":
                    result = env.reset(**payload, start_mode="post_boost")
                elif operation == "execution_inputs":
                    if env.state is None:
                        raise RuntimeError("parallel environment state is unavailable")
                    n_blue = len(env.state.blue)
                    n_red = len(env.state.red)
                    target_slots = np.asarray(
                        payload["target_slots"], dtype=np.int64
                    )[:n_red].copy()
                    target_slots[(target_slots < 0) | (target_slots > n_blue)] = 0
                    if payload["record_assignment"]:
                        env.record_network_call("assignment_actor")
                    result = env.observation_layer.execution_inputs(
                        env.state,
                        target_slots,
                    )
                elif operation == "advance":
                    if env.state is None:
                        raise RuntimeError(
                            "parallel environment state is unavailable before post-boost"
                        )
                    n_red = len(env.state.red)
                    target_indices = np.asarray(
                        payload["target_indices"], dtype=np.int64
                    )[:n_red].copy()
                    guidance_bias = np.asarray(
                        payload["guidance_bias"], dtype=np.float64
                    )[:n_red].copy()
                    n_blue = len(env.state.blue)
                    target_indices[(target_indices < -1) | (target_indices >= n_blue)] = -1
                    env.record_network_call("execution_actor")
                    if payload["record_critic"]:
                        env.record_network_call("execution_critic")
                    result = _advance_environment(
                        env,
                        target_indices,
                        guidance_bias,
                        payload["frame_count"],
                        collect_metrics=payload["collect_metrics"],
                    )
                elif operation == "state":
                    result = env.state
                else:
                    raise ValueError(f"unsupported worker operation: {operation}")
                _send_bytes(connection, (True, result))
            except Exception:
                _send_bytes(connection, (False, _worker_error()))
    finally:
        connection.close()


@contextmanager
def _worker_start_environment(native_threads: int) -> Iterator[None]:
    settings = {
        "OMP_NUM_THREADS": str(native_threads),
        "MKL_NUM_THREADS": str(native_threads),
        "OPENBLAS_NUM_THREADS": str(native_threads),
        "NUMEXPR_NUM_THREADS": str(native_threads),
        "VECLIB_MAXIMUM_THREADS": str(native_threads),
        "MKL_THREADING_LAYER": "GNU",
    }
    previous = {name: os.environ.get(name) for name in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class ProcessEnvironmentPool:
    """Persistent spawned workers, one CPU environment per process."""

    def __init__(
        self,
        envs: Sequence[RedBlueEngagementEnv],
        *,
        native_threads: int = 1,
        timeout_s: float = 300.0,
        start_method: str = "spawn",
    ) -> None:
        if not envs:
            raise ValueError("envs must contain at least one environment")
        if native_threads <= 0:
            raise ValueError("native_threads must be positive")
        if not np.isfinite(timeout_s) or timeout_s <= 0.0:
            raise ValueError("timeout_s must be positive and finite")
        self.size = len(envs)
        self.native_threads = int(native_threads)
        self.timeout_s = float(timeout_s)
        self._context = mp.get_context(start_method)
        self._connections: list[Connection] = []
        self._processes: list[mp.Process] = []
        self._closed = False
        self.worker_info: tuple[EnvironmentWorkerInfo, ...] = ()

        try:
            with _worker_start_environment(self.native_threads):
                for index, env in enumerate(envs):
                    parent_connection, child_connection = self._context.Pipe(duplex=True)
                    process = self._context.Process(
                        target=_environment_worker,
                        args=(child_connection, env, index, self.native_threads),
                        name=f"red-env-{index:02d}",
                        daemon=True,
                    )
                    process.start()
                    child_connection.close()
                    self._connections.append(parent_connection)
                    self._processes.append(process)
            ready = self._receive_many(range(self.size), "worker startup")
            self.worker_info = tuple(ready[index] for index in range(self.size))
        except Exception:
            self.close()
            raise

    def _send_many(self, requests: dict[int, tuple[str, Any]]) -> None:
        for index, request in requests.items():
            process = self._processes[index]
            if not process.is_alive():
                raise EnvironmentWorkerError(
                    f"environment worker {index} exited with code {process.exitcode}"
                )
            _send_bytes(self._connections[index], request)

    def _receive_many(self, indices: Sequence[int], phase: str) -> dict[int, Any]:
        connection_indices = {self._connections[index]: int(index) for index in indices}
        pending = set(connection_indices)
        results: dict[int, Any] = {}
        deadline = time.monotonic() + self.timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                stalled = sorted(connection_indices[connection] for connection in pending)
                raise TimeoutError(
                    f"{phase} exceeded {self.timeout_s:.3f}s for workers {stalled}"
                )
            ready = wait(pending, timeout=remaining)
            if not ready:
                continue
            for connection in ready:
                index = connection_indices[connection]
                pending.remove(connection)
                try:
                    ok, payload = _receive_bytes(connection)
                except (EOFError, OSError) as error:
                    process = self._processes[index]
                    raise EnvironmentWorkerError(
                        f"environment worker {index} disconnected during {phase}; "
                        f"exitcode={process.exitcode}"
                    ) from error
                if not ok:
                    raise EnvironmentWorkerError(
                        f"environment worker {index} failed during {phase}:\n"
                        f"{payload['traceback']}"
                    )
                results[index] = payload
        return results

    def _request(self, requests: dict[int, tuple[str, Any]], phase: str) -> dict[int, Any]:
        self._send_many(requests)
        return self._receive_many(list(requests), phase)

    def reset(
        self,
        *,
        seed: int | None,
        style: ScenarioStyle | None,
        red_count: int | None,
        blue_count: int | None,
        red_counts: Sequence[int | None] | None = None,
        blue_counts: Sequence[int | None] | None = None,
        seeds: Sequence[int | None] | None = None,
    ) -> list[EnvironmentObservation]:
        base_seed = 0 if seed is None else seed
        if red_counts is not None and len(red_counts) != self.size:
            raise ValueError("red_counts must match environment count")
        if blue_counts is not None and len(blue_counts) != self.size:
            raise ValueError("blue_counts must match environment count")
        if seeds is not None and len(seeds) != self.size:
            raise ValueError("seeds must match environment count")
        requests = {
            index: (
                "reset",
                {
                    "seed": (
                        seeds[index]
                        if seeds is not None
                        else None if seed is None else base_seed + index * 1_000_003
                    ),
                    "style": style,
                    "red_count": (
                        red_counts[index] if red_counts is not None else red_count
                    ),
                    "blue_count": (
                        blue_counts[index]
                        if blue_counts is not None
                        else blue_count
                    ),
                },
            )
            for index in range(self.size)
        }
        results = self._request(requests, "environment reset")
        return [results[index] for index in range(self.size)]

    def execution_inputs(
        self,
        target_slots: np.ndarray,
        record_assignment: np.ndarray,
    ) -> list[OverloadBiasActorInputs]:
        slots = np.asarray(target_slots, dtype=np.int64)
        active = np.asarray(record_assignment, dtype=bool)
        if slots.shape[0] != self.size or active.shape != (self.size,):
            raise ValueError("execution input batch does not match environment count")
        requests = {
            index: (
                "execution_inputs",
                {
                    "target_slots": slots[index],
                    "record_assignment": bool(active[index]),
                },
            )
            for index in range(self.size)
        }
        results = self._request(requests, "execution observation")
        return [results[index] for index in range(self.size)]

    def advance(
        self,
        active_indices: Sequence[int],
        target_indices: np.ndarray,
        guidance_bias: np.ndarray,
        frame_count: int,
        record_critic: bool = True,
        collect_metrics: bool = False,
    ) -> dict[int, EnvironmentAdvanceResult]:
        indices = [int(index) for index in active_indices]
        targets = np.asarray(target_indices, dtype=np.int64)
        biases = np.asarray(guidance_bias, dtype=np.float64)
        requests = {
            index: (
                "advance",
                {
                    "target_indices": targets[index],
                    "guidance_bias": biases[index],
                    "frame_count": int(frame_count),
                    "record_critic": bool(record_critic),
                    "collect_metrics": bool(collect_metrics),
                },
            )
            for index in indices
        }
        return self._request(requests, "environment advance")

    def sync_states(self, envs: Sequence[RedBlueEngagementEnv]) -> None:
        if len(envs) != self.size:
            raise ValueError("environment count does not match pool size")
        requests = {index: ("state", None) for index in range(self.size)}
        states = self._request(requests, "environment state synchronization")
        for index, env in enumerate(envs):
            env.state = states[index]

    @property
    def alive_worker_count(self) -> int:
        return sum(process.is_alive() for process in self._processes)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for index, connection in enumerate(self._connections):
            if index < len(self._processes) and self._processes[index].is_alive():
                try:
                    _send_bytes(connection, ("close", None))
                except (BrokenPipeError, EOFError, OSError):
                    pass
        deadline = time.monotonic() + 10.0
        for process in self._processes:
            process.join(timeout=max(0.0, deadline - time.monotonic()))
        for process in self._processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
        for connection in self._connections:
            connection.close()

    def __enter__(self) -> "ProcessEnvironmentPool":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
