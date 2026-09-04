from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from ..env.types import EnvironmentConfig
from .environment import BlueEscapeEnv, BlueEscapeEnvConfig


@dataclass(frozen=True)
class BlueStepResult:
    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, object]


def _send(connection: Connection, value: Any) -> None:
    connection.send_bytes(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))


def _receive(connection: Connection) -> Any:
    return pickle.loads(connection.recv_bytes())


def _worker(connection: Connection, environment: EnvironmentConfig,
            config: BlueEscapeEnvConfig, worker_index: int, native_threads: int) -> None:
    try:
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = str(native_threads)
        torch.set_num_threads(native_threads)
        env = BlueEscapeEnv(environment, config)
        _send(connection, (True, {"index": worker_index, "pid": os.getpid()}))
        while True:
            operation, payload = _receive(connection)
            if operation == "close":
                _send(connection, (True, None)); break
            try:
                if operation == "reset":
                    result = env.reset(payload["seed"], episode_index=payload["episode_index"],
                                       missile_count=payload.get("missile_count"))
                elif operation == "step":
                    if isinstance(payload, dict):
                        observation, reward, terminated, truncated, info = env.step(
                            payload["action"], policy_action=payload.get("policy_action")
                        )
                    else:
                        observation, reward, terminated, truncated, info = env.step(payload)
                    result = BlueStepResult(observation, reward, terminated, truncated, info)
                else:
                    raise ValueError(f"unknown blue environment operation: {operation}")
                _send(connection, (True, result))
            except Exception:
                _send(connection, (False, traceback.format_exc()))
    except EOFError:
        pass
    except Exception:
        try: _send(connection, (False, traceback.format_exc()))
        except Exception: pass
    finally:
        connection.close()


@contextmanager
def _worker_start_environment(native_threads: int) -> Iterator[None]:
    """Limit native libraries while spawned children import their modules."""
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update({name: str(native_threads) for name in names})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class BlueProcessEnvironmentPool:
    """Persistent spawned CPU workers used by the batched blue learner."""

    def __init__(self, environment: EnvironmentConfig, config: BlueEscapeEnvConfig,
                 size: int, *, native_threads: int = 1, timeout_s: float = 300.0) -> None:
        if size < 1 or native_threads < 1 or timeout_s <= 0:
            raise ValueError("size, native_threads and timeout_s must be positive")
        self.size, self.timeout_s = int(size), float(timeout_s)
        self._context = mp.get_context("spawn")
        self._connections: list[Connection] = []; self._processes: list[mp.Process] = []
        self._closed = False
        try:
            with _worker_start_environment(native_threads):
                for index in range(self.size):
                    parent, child = self._context.Pipe(duplex=True)
                    process = self._context.Process(
                        target=_worker, args=(child, environment, config, index, native_threads),
                        name=f"blue-env-{index:02d}", daemon=True,
                    )
                    process.start(); child.close()
                    self._connections.append(parent); self._processes.append(process)
            ready = self._receive_many(range(self.size), "startup")
            self.worker_info = tuple(ready[i] for i in range(self.size))
        except Exception:
            self.close()
            raise

    def _receive_many(self, indices: Sequence[int], phase: str) -> dict[int, Any]:
        mapping = {self._connections[i]: i for i in indices}; pending = set(mapping); results = {}
        deadline = time.monotonic() + self.timeout_s
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"{phase} timed out for workers {[mapping[c] for c in pending]}")
            for connection in wait(pending, timeout=remaining):
                pending.remove(connection); index = mapping[connection]
                try:
                    ok, payload = _receive(connection)
                except (EOFError, OSError) as error:
                    process = self._processes[index]
                    raise RuntimeError(
                        f"blue worker {index} disconnected during {phase}; "
                        f"exitcode={process.exitcode}"
                    ) from error
                if not ok: raise RuntimeError(f"blue worker {index} failed during {phase}:\n{payload}")
                results[index] = payload
        return results

    def _request(self, requests: dict[int, tuple[str, Any]], phase: str) -> dict[int, Any]:
        for index, request in requests.items():
            if not self._processes[index].is_alive():
                raise RuntimeError(f"blue worker {index} exited with {self._processes[index].exitcode}")
            _send(self._connections[index], request)
        return self._receive_many(list(requests), phase)

    def reset(self, assignments: dict[int, tuple[int, int] | tuple[int, int, int]]) -> dict[int, tuple[np.ndarray, dict[str, object]]]:
        requests = {}
        for index, assignment in assignments.items():
            seed, episode, *scenario = assignment
            requests[index] = ("reset", {"seed": seed, "episode_index": episode,
                                          "missile_count": scenario[0] if scenario else None})
        return self._request(requests, "reset")

    def step(self, actions: dict[int, int], *,
             policy_actions: dict[int, int] | None = None) -> dict[int, BlueStepResult]:
        if policy_actions is None:
            payloads = {i: int(action) for i, action in actions.items()}
        else:
            if set(policy_actions) != set(actions):
                raise ValueError("policy_actions must contain exactly the action worker ids")
            payloads = {i: {"action": int(action), "policy_action": int(policy_actions[i])}
                        for i, action in actions.items()}
        return self._request({i: ("step", payload) for i, payload in payloads.items()}, "step")

    def close(self) -> None:
        if self._closed: return
        self._closed = True
        for index, connection in enumerate(self._connections):
            if self._processes[index].is_alive():
                try: _send(connection, ("close", None))
                except OSError: pass
        for process in self._processes:
            process.join(timeout=5)
            if process.is_alive(): process.terminate(); process.join(timeout=5)
        for connection in self._connections: connection.close()

    def __enter__(self) -> "BlueProcessEnvironmentPool": return self
    def __exit__(self, *_: object) -> None: self.close()
    def __del__(self) -> None:
        try: self.close()
        except Exception: pass
