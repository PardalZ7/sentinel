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

"""Unconditional Variational Autoencoder (VAE) anomaly detector.

Learns the unconditional distribution P(x) over all numeric payload fields and
flags events with high reconstruction error — i.e. events that do not belong to
the distribution seen during training.

Unlike cVAE, no field_map is required: fields are auto-discovered from the
payload bodies on the first fit() call (same strategy as MAF). An optional
field_map restricts the detector to specific fields.

Design:
  Encoder: q(z | x)        → mu, log_var
  Decoder: p(x | z)        → x_hat
  Training loss: ELBO = reconstruction_MSE + kl_weight * KL(q || N(0,1))
  Anomaly score: MSE(decoder(mu), x)  (no sampling at inference)

Score semantics: higher = more anomalous.  Anomaly when score > threshold.
The threshold is auto-calibrated at fit() time (P95 of training errors × 1.5)
and embedded in the model dict as auto_threshold.

Requires PyTorch.  Import is deferred to fit()/score().
"""

from typing import Any

from sentinel.config.schema import DetectorConfig
from sentinel.domain.ports.detector import IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

_HIDDEN_SCALE = 2

_torch_cache: tuple | None = None

# Per-model instance cache keyed by id(model dict).
# Stores (vae_instance, mean_tensor, std_tensor).
_VAE_INSTANCE_CACHE: dict[int, tuple] = {}
_VAE_CACHE_MAXSIZE = 4


def _get_torch():
    global _torch_cache
    if _torch_cache is None:
        try:
            import torch
            import torch.nn as nn
            _torch_cache = (torch, nn)
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for the VAE detector. "
                "Install it with: pip install torch"
            ) from e
    return _torch_cache


def _get_cached_vae(model: dict) -> tuple:
    """Return cached (vae, mean, std) for a model dict; FIFO-evict when full."""
    key = id(model)
    if key not in _VAE_INSTANCE_CACHE:
        if len(_VAE_INSTANCE_CACHE) >= _VAE_CACHE_MAXSIZE:
            _VAE_INSTANCE_CACHE.pop(next(iter(_VAE_INSTANCE_CACHE)))
        torch, _ = _get_torch()
        vae = _VAE.from_state_dict(model["state_dict"])
        vae.eval_mode()
        mean = torch.tensor(model["mean"], dtype=torch.float32)
        std = torch.tensor(model["std"], dtype=torch.float32)
        _VAE_INSTANCE_CACHE[key] = (vae, mean, std)
    return _VAE_INSTANCE_CACHE[key]


class _VAE:
    """Minimal unconditional VAE.

    Plain class (not nn.Module subclass) so it serialises to a plain dict
    without requiring the module to be importable at load time.
    """

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        torch, nn = _get_torch()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * _HIDDEN_SCALE),
            nn.ReLU(),
            nn.Linear(hidden_dim * _HIDDEN_SCALE, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * _HIDDEN_SCALE),
            nn.ReLU(),
            nn.Linear(hidden_dim * _HIDDEN_SCALE, input_dim),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(self, mu, log_var):
        torch, _ = _get_torch()
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def parameters(self):
        for module in (self.encoder, self.fc_mu, self.fc_log_var, self.decoder):
            yield from module.parameters()

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "fc_mu": self.fc_mu.state_dict(),
            "fc_log_var": self.fc_log_var.state_dict(),
            "decoder": self.decoder.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "_VAE":
        model = cls(
            input_dim=sd["input_dim"],
            hidden_dim=sd["hidden_dim"],
            latent_dim=sd["latent_dim"],
        )
        model.encoder.load_state_dict(sd["encoder"])
        model.fc_mu.load_state_dict(sd["fc_mu"])
        model.fc_log_var.load_state_dict(sd["fc_log_var"])
        model.decoder.load_state_dict(sd["decoder"])
        return model

    def eval_mode(self):
        for m in (self.encoder, self.fc_mu, self.fc_log_var, self.decoder):
            m.eval()
        return self

    def train_mode(self):
        for m in (self.encoder, self.fc_mu, self.fc_log_var, self.decoder):
            m.train()
        return self


class VAEDetector(IDetector):
    """Unconditional VAE detector with automatic field discovery.

    If field_map is provided, uses input + output fields from the payload bodies.
    Otherwise, uses all numeric fields found in _input_body and _output_body,
    falling back to operational features when payloads have no numeric fields.

    Example sentinel.json detector entry:
      {
        "name": "vae_semantic",
        "algorithm": "vae",
        "phase": "TRAINING",
        "training_window": 1000,
        "test_buffer_size": 150,
        "hidden_dim": 32,
        "latent_dim": 8,
        "learning_rate": 0.001,
        "epochs": 30,
        "kl_weight": 1.0
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
        return "vae"

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
        torch, nn = _get_torch()

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

        model = _VAE(
            input_dim=X.shape[1],
            hidden_dim=self._config.hidden_dim,
            latent_dim=self._config.latent_dim,
        )
        model.train_mode()

        optimiser = torch.optim.Adam(model.parameters(), lr=self._config.learning_rate)

        for _ in range(self._config.epochs):
            optimiser.zero_grad()
            mu, log_var = model.encode(X_n)
            z = model.reparameterise(mu, log_var)
            x_hat = model.decode(z)
            recon_loss = nn.functional.mse_loss(x_hat, X_n)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + self._config.kl_weight * kl_loss
            loss.backward()
            optimiser.step()

        model.eval_mode()

        # Calibrate anomaly threshold from training distribution (P95 × margin).
        with torch.no_grad():
            mu_all, _ = model.encode(X_n)
            x_hat_all = model.decode(mu_all)
            per_sample_errors = ((x_hat_all - X_n) ** 2).mean(dim=1)
            p95_error = float(torch.quantile(per_sample_errors, 0.95).item())
            auto_threshold = p95_error * 1.5

        logger.info(
            "vae_training_complete",
            detector=self.name,
            samples=len(samples),
            fields=fields,
            final_loss=round(float(loss.item()), 6),
            p95_error=round(p95_error, 6),
            auto_threshold=round(auto_threshold, 6),
        )

        return {
            "state_dict": model.state_dict(),
            "fields": fields,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "auto_threshold": auto_threshold,
        }

    def _normalised_tensor(self, model: dict, samples: list[dict]):
        torch, _ = _get_torch()
        vae, mean, std = _get_cached_vae(model)
        rows = self._rows_for(samples, model["fields"])
        x = torch.tensor(rows, dtype=torch.float32)
        return vae, (x - mean) / std

    def score(self, model: Any, event_dict: dict) -> float:
        torch, nn = _get_torch()
        vae, x_n = self._normalised_tensor(model, [event_dict])
        with torch.no_grad():
            mu, _ = vae.encode(x_n)
            x_hat = vae.decode(mu)
            error = float(nn.functional.mse_loss(x_hat, x_n).item())
        return error

    def batch_score(self, model: Any, samples: list[dict]) -> list[float]:
        """Score all samples in a single forward pass (champion evaluation)."""
        torch, _ = _get_torch()
        vae, x_n = self._normalised_tensor(model, samples)
        with torch.no_grad():
            mu, _ = vae.encode(x_n)
            x_hat = vae.decode(mu)
            errors = ((x_hat - x_n) ** 2).mean(dim=1)
        return [float(e) for e in errors]

    def is_anomaly(self, score: float, model: Any = None) -> bool:
        if model is not None and "auto_threshold" in model:
            return score > model["auto_threshold"]
        return score > self._config.anomaly_threshold
