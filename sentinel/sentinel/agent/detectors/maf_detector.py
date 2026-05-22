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

"""Masked Autoencoder for Distribution Estimation (MADE) anomaly detector.

Models the joint distribution P(x_1, ..., x_D) using an autoregressive masked MLP:

    P(x) = ∏_i P(x_i | x_{<i})

Each P(x_i | x_{<i}) is modelled as a Gaussian with (μ_i, σ_i) predicted by the
masked network, which enforces that output i depends only on inputs {x_1,...,x_{i-1}}.

Anomaly score: -log P(x) = -Σ_i log N(x_i; μ_i(x_{<i}), σ_i(x_{<i}))
Higher score = more anomalous.  Anomaly when score > threshold (e.g. 5.0).

Unlike cVAE, MAF requires no field_map — it automatically discovers which numeric
fields are present in the payload and learns their joint distribution.  Field order
is learned and stored during the first fit() call; it must be consistent across
all calls to score() with the same trained model.

Optional field_map: if provided, restricts the detector to those fields instead of
using all numeric fields found in the payload.

Requires PyTorch.
"""

import math
from typing import Any

from sentinel.config.schema import DetectorConfig
from sentinel.domain.ports.detector import IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

# Cached torch modules — populated on first use.
_torch_cache: tuple | None = None

# Per-model instance cache keyed by id(model dict).
# Stores (made_instance, mean_tensor, std_tensor).
_MAF_INSTANCE_CACHE: dict[int, tuple] = {}
_MAF_CACHE_MAXSIZE = 4


def _get_torch():
    global _torch_cache
    if _torch_cache is None:
        try:
            import torch
            import torch.nn as nn
            _torch_cache = (torch, nn)
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for the MAF detector. "
                "Install it with: pip install torch"
            ) from e
    return _torch_cache


def _get_cached_maf(model: dict) -> tuple:
    """Return cached (made, mean, std) for a model dict.

    On first access, rebuilds the MADE from state_dict with mask application
    skipped (weights are already masked from training), then caches tensors.
    Evicts the oldest entry (FIFO) when the cache is full.
    """
    key = id(model)
    if key not in _MAF_INSTANCE_CACHE:
        if len(_MAF_INSTANCE_CACHE) >= _MAF_CACHE_MAXSIZE:
            _MAF_INSTANCE_CACHE.pop(next(iter(_MAF_INSTANCE_CACHE)))
        torch, _ = _get_torch()
        made = _MADEModel.from_state_dict(model["state_dict"])
        mean = torch.tensor(model["mean"], dtype=torch.float32)
        std = torch.tensor(model["std"], dtype=torch.float32)
        _MAF_INSTANCE_CACHE[key] = (made, mean, std)
    return _MAF_INSTANCE_CACHE[key]


def _build_mask(dim: int, hidden_dim: int) -> "list[torch.Tensor]":
    """Build MADE masks for a 2-layer network.

    Returns a list of masks (input→h1, h1→h2, h2→output×2).
    Each output unit i models p(x_i | x_{<i}), so its mask must only connect
    to hidden units that themselves only depend on x_{<i}.
    """
    torch, _ = _get_torch()

    # Assign order indices to input units: m_0[k] = k (identity ordering)
    m0 = list(range(dim))

    # Hidden units get uniformly spread order indices in [0, dim-1]
    m1 = [i % dim for i in range(hidden_dim)]
    m2 = [i % dim for i in range(hidden_dim)]

    def _mask(m_prev, m_next, strict=False):
        rows = torch.tensor(m_prev).unsqueeze(1)  # (in_dim, 1)
        cols = torch.tensor(m_next).unsqueeze(0)  # (1, out_dim)
        if strict:
            return (cols > rows).float()
        return (cols >= rows).float()

    mask_0 = _mask(m0, m1)          # (dim, hidden)  input → h1
    mask_1 = _mask(m1, m2)          # (hidden, hidden) h1 → h2
    # Output: 2*dim units — first dim for mu, second dim for log_sigma
    mask_out = _mask(m2, m0, strict=True)  # (hidden, dim) for one output head

    return [mask_0, mask_1, mask_out]


class _MADEModel:
    """2-hidden-layer MADE with Gaussian outputs.

    Serialisable to a plain dict for joblib storage.
    """

    def __init__(self, dim: int, hidden_dim: int) -> None:
        torch, nn = _get_torch()

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, dim)
        self.fc_log_sigma = nn.Linear(hidden_dim, dim)

        masks = _build_mask(dim, hidden_dim)
        self._mask0 = masks[0].T   # (hidden, dim) — applied as fc1.weight mask
        self._mask1 = masks[1].T
        self._mask_out = masks[2].T

        # Apply masks immediately so weights are constrained from the start.
        # During training, masks must be re-applied after each optimizer step
        # (the optimizer update can restore masked-out weights).
        # At inference time (from_state_dict), weights are already masked — set
        # _apply_masks=False to skip the redundant multiplication in forward().
        self._apply_masks: bool = True

    def apply_masks(self) -> None:
        """Zero out weight entries that violate the autoregressive constraint."""
        torch, _ = _get_torch()
        with torch.no_grad():
            self.fc1.weight.data *= self._mask0
            self.fc2.weight.data *= self._mask1
            self.fc_mu.weight.data *= self._mask_out
            self.fc_log_sigma.weight.data *= self._mask_out

    def forward(self, x):
        torch, nn = _get_torch()

        if self._apply_masks:
            self.apply_masks()

        h1 = torch.relu(self.fc1(x))
        h2 = torch.relu(self.fc2(h1))
        mu = self.fc_mu(h2)
        log_sigma = self.fc_log_sigma(h2).clamp(-4, 4)
        return mu, log_sigma

    def parameters(self):
        for layer in (self.fc1, self.fc2, self.fc_mu, self.fc_log_sigma):
            yield from layer.parameters()

    def state_dict(self) -> dict:
        return {
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "fc1": self.fc1.state_dict(),
            "fc2": self.fc2.state_dict(),
            "fc_mu": self.fc_mu.state_dict(),
            "fc_log_sigma": self.fc_log_sigma.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "_MADEModel":
        obj = cls(dim=sd["dim"], hidden_dim=sd["hidden_dim"])
        obj.fc1.load_state_dict(sd["fc1"])
        obj.fc2.load_state_dict(sd["fc2"])
        obj.fc_mu.load_state_dict(sd["fc_mu"])
        obj.fc_log_sigma.load_state_dict(sd["fc_log_sigma"])
        # Weights loaded from disk are already masked from the training phase.
        # Skip redundant mask application on every inference forward pass.
        obj._apply_masks = False
        return obj


def _neg_log_likelihood(mu, log_sigma, x):
    """Gaussian NLL per sample: -Σ_i log N(x_i; μ_i, σ_i)."""
    torch, _ = _get_torch()
    sigma = torch.exp(log_sigma)
    nll = 0.5 * ((x - mu) / sigma).pow(2) + log_sigma + 0.5 * math.log(2 * math.pi)
    return nll.sum(dim=-1)   # sum over features, mean over batch in training


class MAFDetector(IDetector):
    """MADE-based density estimator for joint distribution anomaly detection.

    If field_map is provided, uses input + output fields from the payload bodies.
    Otherwise, uses all numeric fields found in _input_body and _output_body.
    Field discovery happens on the first fit() call and is stored in the model.

    Example sentinel.json detector entry (no field_map — auto-discover):
      {
        "name": "maf_e2e",
        "algorithm": "maf",
        "phase": "TRAINING",
        "training_window": 2000,
        "test_buffer_size": 200,
        "anomaly_threshold": 5.0,
        "hidden_dim": 64,
        "learning_rate": 0.001,
        "epochs": 50
      }

    Example with explicit field_map:
      {
        "name": "maf_batch",
        "algorithm": "maf",
        "field_map": { "input": ["field_a"], "output": ["field_b", "field_c"] }
      }
    """

    def __init__(self, config: DetectorConfig) -> None:
        self._config = config
        self._explicit_fields: list[str] | None = None
        if config.field_map is not None:
            self._explicit_fields = config.field_map.input + config.field_map.output

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def algorithm(self) -> str:
        return "maf"

    @property
    def training_window(self) -> int:
        return self._config.training_window

    @property
    def test_buffer_size(self) -> int:
        return self._config.test_buffer_size

    def _discover_fields(self, sample: dict) -> list[str]:
        """Return sorted numeric field names from the payload bodies."""
        fields: set[str] = set()
        for body_key in ("_input_body", "_output_body"):
            body = sample.get(body_key) or {}
            for k, v in body.items():
                if isinstance(v, (int, float)) and not k.startswith("_"):
                    fields.add(k)
        return sorted(fields)

    def _extract_from_bodies(self, event_dict: dict, fields: list[str]) -> list[float]:
        # Lazy-merge bodies and cache the result on the event_dict so that
        # repeated calls (champion evaluation, multiple MAF detectors) pay the
        # merge cost only once per event.
        merged = event_dict.get("_merged_body")
        if merged is None:
            merged = {}
            for body_key in ("_input_body", "_output_body"):
                body = event_dict.get(body_key)
                if body:
                    merged.update(body)
            event_dict["_merged_body"] = merged
        return [float(merged.get(f, 0.0)) for f in fields]

    def extract_features(self, event_dict: dict) -> list[float]:
        if self._explicit_fields is not None:
            return self._extract_from_bodies(event_dict, self._explicit_fields)
        # Fallback to operational features when no bodies are present
        return [
            float(event_dict.get("processing_latency_ms", 0.0)),
            float(event_dict.get("input_size_bytes", 0)),
            float(event_dict.get("output_size_bytes", 0)),
        ]

    def fit(self, samples: list[dict]) -> dict:
        torch, nn = _get_torch()

        # Determine feature fields
        if self._explicit_fields is not None:
            fields = self._explicit_fields
        elif samples:
            fields = self._discover_fields(samples[0])
        else:
            fields = []

        if not fields:
            # Fall back to operational features
            fields = ["_latency_ms", "_input_size", "_output_size"]
            data_rows = [
                [
                    float(s.get("processing_latency_ms", 0.0)),
                    float(s.get("input_size_bytes", 0)),
                    float(s.get("output_size_bytes", 0)),
                ]
                for s in samples
            ]
        else:
            data_rows = [self._extract_from_bodies(s, fields) for s in samples]

        X = torch.tensor(data_rows, dtype=torch.float32)
        dim = X.shape[1]

        # Normalise
        mean = X.mean(dim=0)
        std = X.std(dim=0).clamp(min=1e-6)
        X_n = (X - mean) / std

        model = _MADEModel(dim=dim, hidden_dim=self._config.hidden_dim)
        optimiser = torch.optim.Adam(model.parameters(), lr=self._config.learning_rate)

        for _ in range(self._config.epochs):
            optimiser.zero_grad()
            mu, log_sigma = model.forward(X_n)
            loss = _neg_log_likelihood(mu, log_sigma, X_n).mean()
            loss.backward()
            optimiser.step()
            # Re-apply masks after optimizer step to ensure autoregressive constraint.
            model.apply_masks()

        logger.info(
            "maf_training_complete",
            detector=self.name,
            samples=len(samples),
            dim=dim,
            fields=fields,
            final_loss=round(float(loss.item()), 6),
        )

        return {
            "state_dict": model.state_dict(),
            "fields": fields,
            "mean": mean.tolist(),
            "std": std.tolist(),
        }

    def score(self, model: Any, event_dict: dict) -> float:
        torch, _ = _get_torch()

        # Retrieve cached MADE instance and pre-converted normalization tensors.
        made, mean, std = _get_cached_maf(model)

        fields = model["fields"]
        if fields and not fields[0].startswith("_"):
            values = self._extract_from_bodies(event_dict, fields)
        else:
            values = [
                float(event_dict.get("processing_latency_ms", 0.0)),
                float(event_dict.get("input_size_bytes", 0)),
                float(event_dict.get("output_size_bytes", 0)),
            ]

        x = torch.tensor(values, dtype=torch.float32).unsqueeze(0)
        x_n = (x - mean) / std

        with torch.no_grad():
            mu, log_sigma = made.forward(x_n)
            nll = _neg_log_likelihood(mu, log_sigma, x_n)

        # Normalise by number of features so threshold is dimension-independent
        dim = len(fields)
        return float(nll.item()) / max(dim, 1)

    def is_anomaly(self, score: float, model: Any = None) -> bool:
        return score > self._config.anomaly_threshold
