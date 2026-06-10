# Copyright 2026 Alexandre Cardoso
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared base for sequence-aware autoencoder detectors (lstm_ae, tcn_ae).

Sequence-aware detectors score windows of consecutive events instead of single
events. The use-case layer (process_message) maintains a rolling window of the
last sequence_length event dicts in DetectorState.sequence_buffer and passes a
dict with a "_sequence" key to score(). See IDetector.requires_sequence.

This base implements everything network-agnostic (Template Method over the
IDetector strategy):
  - per-timestep feature extraction:
      [processing_latency_ms, input_size_bytes, output_size_bytes]
    (anomaly_score is deliberately NOT a feature: during this detector's own
    TRAINING phase no reliable score exists for it — using scores from sibling
    detectors would couple their lifecycles)
  - sliding-window slicing of the training buffer (stride 1)
  - per-feature normalisation (mean/std over all timesteps)
  - P95 × 1.5 auto-threshold calibration
  - score()/batch_score() plumbing including left-zero-padding of short windows

Subclasses implement only the network: _new_network() and _network_from_state_dict().
The network contract: forward(x) where x is (batch, seq_len, features), returning
a reconstruction of the same shape.

Score semantics: higher = more anomalous (reconstruction MSE).
Anomaly when score > threshold (auto-calibrated, embedded in the model dict).

Requires PyTorch.  Import is deferred to fit()/score().
"""

from abc import abstractmethod
from typing import Any

from sentinel.config.schema import DetectorConfig
from sentinel.domain.ports.detector import IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

_torch_cache: tuple | None = None

_FEATURES_PER_STEP = 3


def _get_torch():
    global _torch_cache
    if _torch_cache is None:
        try:
            import torch
            import torch.nn as nn
            _torch_cache = (torch, nn)
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for sequence detectors (lstm_ae, tcn_ae). "
                "Install it with: pip install torch"
            ) from e
    return _torch_cache


def _event_features(event_dict: dict) -> list[float]:
    return [
        float(event_dict.get("processing_latency_ms", 0.0)),
        float(event_dict.get("input_size_bytes", 0)),
        float(event_dict.get("output_size_bytes", 0)),
    ]


class SequenceAutoencoderDetector(IDetector):
    """Network-agnostic base for sequence autoencoder detectors."""

    # Per-model instance cache keyed by id(model dict), FIFO-bounded.
    # Class-level but keyed per concrete subclass via the model dicts themselves
    # (a model dict is only ever scored by the detector class that produced it).
    _instance_cache: dict[int, tuple]
    _cache_maxsize = 4

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls._instance_cache = {}

    # ── IDetector properties ──────────────────────────────────────────

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def training_window(self) -> int:
        return self._config.training_window

    @property
    def test_buffer_size(self) -> int:
        return self._config.test_buffer_size

    @property
    def requires_sequence(self) -> bool:
        return True

    @property
    def sequence_length(self) -> int:
        return self._config.sequence_length

    # ── Network hooks (implemented by subclasses) ─────────────────────

    @abstractmethod
    def _new_network(self) -> Any:
        """Build a fresh untrained network for (sequence_length, features) input."""

    @abstractmethod
    def _network_from_state_dict(self, sd: dict) -> Any:
        """Rebuild a trained network from its serialised state dict."""

    # ── Data plumbing ─────────────────────────────────────────────────

    def _window_matrix(self, events: list[dict]) -> list[list[float]]:
        """Convert a list of event dicts into a (seq_len, features) matrix.

        Windows shorter than sequence_length are left-padded with zero rows so
        cold-start scoring degrades gracefully instead of failing.
        """
        seq_len = self.sequence_length
        rows = [_event_features(e) for e in events[-seq_len:]]
        pad = seq_len - len(rows)
        if pad > 0:
            rows = [[0.0] * _FEATURES_PER_STEP] * pad + rows
        return rows

    def _sample_to_window(self, sample: dict) -> list[list[float]]:
        """Accept either a window dict ({"_sequence": [...]}) or a bare event."""
        sequence = sample.get("_sequence")
        if sequence is not None:
            return self._window_matrix(sequence)
        return self._window_matrix([sample])

    def extract_features(self, event_dict: dict) -> list[float]:
        """Flattened (seq_len × features) vector, oldest timestep first."""
        return [v for row in self._sample_to_window(event_dict) for v in row]

    def _get_cached(self, model: dict) -> tuple:
        cache = type(self)._instance_cache
        key = id(model)
        if key not in cache:
            if len(cache) >= self._cache_maxsize:
                cache.pop(next(iter(cache)))
            torch, _ = _get_torch()
            net = self._network_from_state_dict(model["state_dict"])
            net.eval_mode()
            mean = torch.tensor(model["mean"], dtype=torch.float32)
            std = torch.tensor(model["std"], dtype=torch.float32)
            cache[key] = (net, mean, std)
        return cache[key]

    # ── Training ──────────────────────────────────────────────────────

    def fit(self, samples: list[dict]) -> dict:
        """Train on sliding windows (stride 1) sliced from the ordered buffer.

        The training buffer holds individual event dicts in arrival order;
        contiguity is the use-case layer's responsibility (gating and denial
        hits clear the sequence state rather than punch holes in it).
        """
        torch, _ = _get_torch()

        seq_len = self.sequence_length
        if len(samples) < seq_len:
            raise ValueError(
                f"Sequence detector '{self.name}' needs at least sequence_length="
                f"{seq_len} samples to train, got {len(samples)}. "
                f"Increase training_window or decrease sequence_length."
            )

        windows = [
            self._window_matrix(samples[i: i + seq_len])
            for i in range(len(samples) - seq_len + 1)
        ]
        X = torch.tensor(windows, dtype=torch.float32)   # (N, L, F)

        # Per-feature normalisation across all windows and timesteps.
        flat = X.reshape(-1, _FEATURES_PER_STEP)
        mean = flat.mean(dim=0)
        std = flat.std(dim=0).clamp(min=1e-6)
        X_n = (X - mean) / std

        net = self._new_network()
        net.train_mode()
        optimiser = torch.optim.Adam(net.parameters(), lr=self._config.learning_rate)

        loss = None
        for _ in range(self._config.epochs):
            optimiser.zero_grad()
            x_hat = net.forward(X_n)
            loss = ((x_hat - X_n) ** 2).mean()
            loss.backward()
            optimiser.step()

        net.eval_mode()

        # Calibrate anomaly threshold from training distribution (P95 × margin).
        with torch.no_grad():
            x_hat_all = net.forward(X_n)
            per_window_errors = ((x_hat_all - X_n) ** 2).mean(dim=(1, 2))
            p95_error = float(torch.quantile(per_window_errors, 0.95).item())
            auto_threshold = p95_error * 1.5

        logger.info(
            f"{self.algorithm}_training_complete",
            detector=self.name,
            samples=len(samples),
            windows=len(windows),
            sequence_length=seq_len,
            final_loss=round(float(loss.item()), 6),
            p95_error=round(p95_error, 6),
            auto_threshold=round(auto_threshold, 6),
        )

        return {
            "state_dict": net.state_dict(),
            "mean": mean.tolist(),
            "std": std.tolist(),
            "sequence_length": seq_len,
            "auto_threshold": auto_threshold,
        }

    # ── Scoring ───────────────────────────────────────────────────────

    def _errors(self, model: dict, samples: list[dict]):
        torch, _ = _get_torch()
        net, mean, std = self._get_cached(model)
        windows = [self._sample_to_window(s) for s in samples]
        x = torch.tensor(windows, dtype=torch.float32)
        x_n = (x - mean) / std
        with torch.no_grad():
            x_hat = net.forward(x_n)
            return ((x_hat - x_n) ** 2).mean(dim=(1, 2))

    def score(self, model: Any, event_dict: dict) -> float:
        return float(self._errors(model, [event_dict])[0].item())

    def batch_score(self, model: Any, samples: list[dict]) -> list[float]:
        """Score all windows in a single forward pass (champion evaluation)."""
        return [float(e) for e in self._errors(model, samples)]

    def is_anomaly(self, score: float, model: Any = None) -> bool:
        if model is not None and "auto_threshold" in model:
            return score > model["auto_threshold"]
        return score > self._config.anomaly_threshold
