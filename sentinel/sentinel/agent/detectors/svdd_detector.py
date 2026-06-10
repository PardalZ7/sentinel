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

"""One-Class Deep SVDD anomaly detector.

Learns a minimal hypersphere around normal data in latent space: an encoder
network phi() maps events to a latent representation, and training minimises
the mean squared distance of all (normal) training points to a fixed center c.

  Center c: mean of phi(training set), computed once after a no-grad forward
            pass with the freshly initialised network, then frozen.
  Loss:     (1/N) * sum ||phi(x_i) - c||^2
  Score:    ||phi(x) - c||^2   (higher = more anomalous)

Hypersphere-collapse mitigation: all Linear layers are bias-free, the standard
practice from the Deep SVDD literature — with biases the network can trivially
map every input onto c regardless of its value.

Note on retraining: each training cycle computes its own center, so scores from
different model versions are NOT on the same scale. This is safe because each
model carries its own auto_threshold (the (1-nu) percentile of training-set
distances) and champion/challenger comparison uses FP rates, not raw scores.

Field discovery follows the MAF/VAE strategy: numeric payload fields are
auto-discovered on fit(); an optional field_map restricts them; operational
features are the fallback when payloads have no numeric fields.

Score semantics: higher = more anomalous.  Anomaly when score > threshold.

Requires PyTorch.  Import is deferred to fit()/score().
"""

from typing import Any

from sentinel.config.schema import DetectorConfig
from sentinel.domain.ports.detector import IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

_torch_cache: tuple | None = None

# Per-model instance cache keyed by id(model dict).
# Stores (network, center_tensor, mean_tensor, std_tensor).
_SVDD_INSTANCE_CACHE: dict[int, tuple] = {}
_SVDD_CACHE_MAXSIZE = 4


def _get_torch():
    global _torch_cache
    if _torch_cache is None:
        try:
            import torch
            import torch.nn as nn
            _torch_cache = (torch, nn)
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for the SVDD detector. "
                "Install it with: pip install torch"
            ) from e
    return _torch_cache


def _get_cached_svdd(model: dict) -> tuple:
    """Return cached (net, center, mean, std) for a model dict; FIFO-evict when full."""
    key = id(model)
    if key not in _SVDD_INSTANCE_CACHE:
        if len(_SVDD_INSTANCE_CACHE) >= _SVDD_CACHE_MAXSIZE:
            _SVDD_INSTANCE_CACHE.pop(next(iter(_SVDD_INSTANCE_CACHE)))
        torch, _ = _get_torch()
        net = _SVDDNetwork.from_state_dict(model["state_dict"])
        net.eval_mode()
        center = torch.tensor(model["center"], dtype=torch.float32)
        mean = torch.tensor(model["mean"], dtype=torch.float32)
        std = torch.tensor(model["std"], dtype=torch.float32)
        _SVDD_INSTANCE_CACHE[key] = (net, center, mean, std)
    return _SVDD_INSTANCE_CACHE[key]


class _SVDDNetwork:
    """Bias-free encoder mapping events into the SVDD latent space.

    Plain class (not nn.Module subclass) so it serialises to a plain dict.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        _, nn = _get_torch()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # bias=False on every layer: see module docstring (collapse mitigation).
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim, bias=False),
        )

    def forward(self, x):
        return self.net(x)

    def parameters(self):
        yield from self.net.parameters()

    def state_dict(self) -> dict:
        return {
            "net": self.net.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "_SVDDNetwork":
        obj = cls(
            input_dim=sd["input_dim"],
            hidden_dim=sd["hidden_dim"],
            latent_dim=sd["latent_dim"],
        )
        obj.net.load_state_dict(sd["net"])
        return obj

    def eval_mode(self):
        self.net.eval()
        return self

    def train_mode(self):
        self.net.train()
        return self


class SVDDDetector(IDetector):
    """Deep SVDD detector: distance-to-center anomaly scoring in latent space.

    Example sentinel.json detector entry:
      {
        "name": "svdd_compact",
        "algorithm": "svdd",
        "phase": "TRAINING",
        "training_window": 1000,
        "test_buffer_size": 150,
        "hidden_dim": 32,
        "latent_dim": 8,
        "learning_rate": 0.001,
        "epochs": 30,
        "nu": 0.1
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
        return "svdd"

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
        merged = event_dict.get("_merged_body")
        if merged is None:
            merged = {}
            for body_key in ("_input_body", "_output_body"):
                body = event_dict.get(body_key)
                if body:
                    merged.update(body)
            event_dict["_merged_body"] = merged
        return [float(merged.get(f, 0.0)) for f in fields]

    @staticmethod
    def _operational_features(event_dict: dict) -> list[float]:
        return [
            float(event_dict.get("processing_latency_ms", 0.0)),
            float(event_dict.get("input_size_bytes", 0)),
            float(event_dict.get("output_size_bytes", 0)),
        ]

    def extract_features(self, event_dict: dict) -> list[float]:
        if self._explicit_fields is not None:
            return self._extract_from_bodies(event_dict, self._explicit_fields)
        return self._operational_features(event_dict)

    def _rows_for(self, samples: list[dict], fields: list[str]) -> list[list[float]]:
        if fields and not fields[0].startswith("_"):
            return [self._extract_from_bodies(s, fields) for s in samples]
        return [self._operational_features(s) for s in samples]

    def fit(self, samples: list[dict]) -> dict:
        torch, _ = _get_torch()

        if self._explicit_fields is not None:
            fields = self._explicit_fields
        elif samples:
            fields = self._discover_fields(samples[0])
        else:
            fields = []

        if not fields:
            fields = ["_latency_ms", "_input_size", "_output_size"]

        data_rows = self._rows_for(samples, fields)
        X = torch.tensor(data_rows, dtype=torch.float32)

        mean = X.mean(dim=0)
        std = X.std(dim=0).clamp(min=1e-6)
        X_n = (X - mean) / std

        net = _SVDDNetwork(
            input_dim=X.shape[1],
            hidden_dim=self._config.hidden_dim,
            latent_dim=self._config.latent_dim,
        )

        # Fix the center as the mean latent representation of the training set,
        # computed with the freshly initialised network and frozen thereafter.
        net.eval_mode()
        with torch.no_grad():
            center = net.forward(X_n).mean(dim=0)
        # Nudge center components away from zero — exact zeros combined with
        # bias-free layers admit the trivial all-zero solution.
        eps = 0.1
        center = torch.where(center.abs() < eps, torch.full_like(center, eps), center)

        net.train_mode()
        optimiser = torch.optim.Adam(net.parameters(), lr=self._config.learning_rate)

        for _ in range(self._config.epochs):
            optimiser.zero_grad()
            latent = net.forward(X_n)
            loss = ((latent - center) ** 2).sum(dim=1).mean()
            loss.backward()
            optimiser.step()

        net.eval_mode()

        # Threshold: (1-nu) percentile of training-set distances to the center.
        with torch.no_grad():
            distances = ((net.forward(X_n) - center) ** 2).sum(dim=1)
            quantile = min(max(1.0 - self._config.nu, 0.0), 1.0)
            auto_threshold = float(torch.quantile(distances, quantile).item())

        logger.info(
            "svdd_training_complete",
            detector=self.name,
            samples=len(samples),
            fields=fields,
            final_loss=round(float(loss.item()), 6),
            auto_threshold=round(auto_threshold, 6),
        )

        return {
            "state_dict": net.state_dict(),
            "center": center.tolist(),
            "fields": fields,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "auto_threshold": auto_threshold,
        }

    def _distances(self, model: dict, samples: list[dict]):
        torch, _ = _get_torch()
        net, center, mean, std = _get_cached_svdd(model)
        rows = self._rows_for(samples, model["fields"])
        x = torch.tensor(rows, dtype=torch.float32)
        x_n = (x - mean) / std
        with torch.no_grad():
            return ((net.forward(x_n) - center) ** 2).sum(dim=1)

    def score(self, model: Any, event_dict: dict) -> float:
        return float(self._distances(model, [event_dict])[0].item())

    def batch_score(self, model: Any, samples: list[dict]) -> list[float]:
        """Score all samples in a single forward pass (champion evaluation)."""
        return [float(d) for d in self._distances(model, samples)]

    def is_anomaly(self, score: float, model: Any = None) -> bool:
        if model is not None and "auto_threshold" in model:
            return score > model["auto_threshold"]
        return score > self._config.anomaly_threshold
