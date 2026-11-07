from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


class FrozenRainbowNumpy:
    """Inference-only v1 Rainbow network implemented with NumPy.

    The exported artifact contains only the online network parameters used in
    evaluation mode. No target network, optimizer, replay buffer, gradient or
    training entry point is present in the executable runtime.
    """

    _ARRAY_NAMES = (
        "features_weight",
        "features_bias",
        "value_hidden_weight",
        "value_hidden_bias",
        "value_output_weight",
        "value_output_bias",
        "advantage_hidden_weight",
        "advantage_hidden_bias",
        "advantage_output_weight",
        "advantage_output_bias",
        "support",
    )

    def __init__(self, model_path: str | Path, manifest_path: str | Path) -> None:
        self.model_path = Path(model_path).resolve(strict=True)
        self.manifest_path = Path(manifest_path).resolve(strict=True)
        self.manifest: dict[str, Any] = json.loads(
            self.manifest_path.read_text(encoding="utf-8")
        )
        self._validate_manifest()
        expected_sha = str(self.manifest["inference_model_sha256"])
        actual_sha = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(
                "inference model SHA-256 mismatch: "
                f"expected {expected_sha}, got {actual_sha}"
            )

        with np.load(self.model_path, allow_pickle=False) as archive:
            missing = sorted(set(self._ARRAY_NAMES) - set(archive.files))
            extra = sorted(set(archive.files) - set(self._ARRAY_NAMES))
            if missing or extra:
                raise ValueError(
                    f"invalid inference model arrays: missing={missing}, extra={extra}"
                )
            for name in self._ARRAY_NAMES:
                value = np.asarray(archive[name], dtype=np.float32).copy()
                if not np.all(np.isfinite(value)):
                    raise ValueError(f"model array {name} contains non-finite values")
                setattr(self, name, value)
        self._validate_shapes()

    @property
    def observation_dim(self) -> int:
        return int(self.manifest["model_contract"]["observation_dim"])

    @property
    def action_dim(self) -> int:
        return int(self.manifest["model_contract"]["action_dim"])

    @property
    def atoms(self) -> int:
        return int(self.manifest["model_contract"]["atoms"])

    def _validate_manifest(self) -> None:
        if int(self.manifest.get("bundle_format_version", -1)) != 1:
            raise ValueError("unsupported model bundle format")
        contract = self.manifest.get("model_contract")
        if not isinstance(contract, dict):
            raise ValueError("model manifest is missing model_contract")
        expected = {
            "observation_schema": "normalized_v4",
            "observation_dim": 104,
            "missile_slots": 3,
            "action_dim": 29,
            "atoms": 61,
        }
        for key, value in expected.items():
            if contract.get(key) != value:
                raise ValueError(
                    f"unsupported model contract {key}: {contract.get(key)!r}"
                )

    def _validate_shapes(self) -> None:
        hidden = int(self.manifest["model_contract"]["hidden_dim"])
        expected_shapes = {
            "features_weight": (hidden, self.observation_dim),
            "features_bias": (hidden,),
            "value_hidden_weight": (hidden, hidden),
            "value_hidden_bias": (hidden,),
            "value_output_weight": (self.atoms, hidden),
            "value_output_bias": (self.atoms,),
            "advantage_hidden_weight": (hidden, hidden),
            "advantage_hidden_bias": (hidden,),
            "advantage_output_weight": (self.action_dim * self.atoms, hidden),
            "advantage_output_bias": (self.action_dim * self.atoms,),
            "support": (self.atoms,),
        }
        for name, expected in expected_shapes.items():
            actual = tuple(getattr(self, name).shape)
            if actual != expected:
                raise ValueError(
                    f"model array {name} has shape {actual}, expected {expected}"
                )

    @staticmethod
    def _linear(
        values: np.ndarray, weight: np.ndarray, bias: np.ndarray
    ) -> np.ndarray:
        return values @ weight.T + bias

    @staticmethod
    def _relu(values: np.ndarray) -> np.ndarray:
        return np.maximum(values, np.float32(0.0))

    def logits(self, observations: np.ndarray) -> np.ndarray:
        values = np.asarray(observations, dtype=np.float32)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != self.observation_dim:
            raise ValueError(
                f"observations must have shape [batch, {self.observation_dim}]"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("observations must contain only finite values")

        features = self._relu(
            self._linear(values, self.features_weight, self.features_bias)
        )
        value_hidden = self._relu(
            self._linear(
                features, self.value_hidden_weight, self.value_hidden_bias
            )
        )
        value = self._linear(
            value_hidden, self.value_output_weight, self.value_output_bias
        ).reshape(-1, 1, self.atoms)
        advantage_hidden = self._relu(
            self._linear(
                features,
                self.advantage_hidden_weight,
                self.advantage_hidden_bias,
            )
        )
        advantage = self._linear(
            advantage_hidden,
            self.advantage_output_weight,
            self.advantage_output_bias,
        ).reshape(-1, self.action_dim, self.atoms)
        return value + advantage - advantage.mean(axis=1, keepdims=True)

    def expected_action_values(self, observations: np.ndarray) -> np.ndarray:
        logits = self.logits(observations)
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        numerator = np.exp(shifted)
        probabilities = numerator / numerator.sum(axis=-1, keepdims=True)
        return (probabilities * self.support).sum(axis=-1)
