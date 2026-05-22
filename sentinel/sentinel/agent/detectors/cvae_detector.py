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

"""Conditional Variational Autoencoder (cVAE) anomaly detector.

Detects semantic anomalies: cases where the output field values are inconsistent
with the corresponding input field values, according to the distribution learned
during training.

Design:
  Encoder: q(z | input_fields, output_fields)  → mu, log_var
  Decoder: p(output_fields | z, input_fields)  → output_hat

  Training loss: ELBO = reconstruction_MSE + β * KL_divergence(q || N(0,1))
  Anomaly score: MSE(decoder(mu, input), actual_output)  (no sampling at inference)

Score semantics: higher = more anomalous.  Anomaly when score > threshold (e.g. 0.05).

The field_map in DetectorConfig specifies which payload fields are "input" (conditioning)
and which are "output" (reconstructed).  This is usecase-specific knowledge and must be
configured in sentinel.json — the Sentinel library does not interpret field names.

Requires PyTorch.  Import is deferred to fit()/score() to avoid a hard dependency
when only IsolationForest detectors are used.
"""

from typing import Any

from sentinel.config.schema import DetectorConfig
from sentinel.domain.ports.detector import IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

_HIDDEN_SCALE = 2   # hidden_dim multiplier for encoder/decoder width

# Cached torch modules — populated on first use, avoids repeated import lookups.
_torch_cache: tuple | None = None

# Per-model instance cache keyed by id(model dict).
# Stores (cvae_instance, in_mean_tensor, in_std_tensor, out_mean_tensor, out_std_tensor).
# Avoids rebuilding the full neural network from state_dict on every score() call.
_CVAE_INSTANCE_CACHE: dict[int, tuple] = {}
_CVAE_CACHE_MAXSIZE = 4


def _get_torch():
    global _torch_cache
    if _torch_cache is None:
        try:
            import torch
            import torch.nn as nn
            _torch_cache = (torch, nn)
        except ImportError as e:
            raise ImportError(
                "PyTorch is required for the cVAE detector. "
                "Install it with: pip install torch"
            ) from e
    return _torch_cache


def _get_cached_cvae(model: dict) -> tuple:
    """Return cached (cvae, in_mean, in_std, out_mean, out_std) for a model dict.

    Rebuilds and caches on first access per model identity. The cache is bounded
    at _CVAE_CACHE_MAXSIZE entries; evicts the oldest entry (FIFO) when full so
    that the remaining cached models are not discarded needlessly.
    """
    key = id(model)
    if key not in _CVAE_INSTANCE_CACHE:
        if len(_CVAE_INSTANCE_CACHE) >= _CVAE_CACHE_MAXSIZE:
            _CVAE_INSTANCE_CACHE.pop(next(iter(_CVAE_INSTANCE_CACHE)))
        torch, _ = _get_torch()
        cvae = _ConditionalVAE.from_state_dict(model["state_dict"])
        cvae.eval_mode()
        in_mean = torch.tensor(model["in_mean"], dtype=torch.float32)
        in_std = torch.tensor(model["in_std"], dtype=torch.float32)
        out_mean = torch.tensor(model["out_mean"], dtype=torch.float32)
        out_std = torch.tensor(model["out_std"], dtype=torch.float32)
        _CVAE_INSTANCE_CACHE[key] = (cvae, in_mean, in_std, out_mean, out_std)
    return _CVAE_INSTANCE_CACHE[key]


class _ConditionalVAE:
    """Minimal cVAE implemented with PyTorch.

    Kept as a plain class (not nn.Module subclass) so it can be serialised
    to a plain dict without requiring the module to be importable at load time.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        latent_dim: int,
    ) -> None:
        torch, nn = _get_torch()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        enc_in = input_dim + output_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_dim * _HIDDEN_SCALE),
            nn.ReLU(),
            nn.Linear(hidden_dim * _HIDDEN_SCALE, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)

        dec_in = latent_dim + input_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * _HIDDEN_SCALE),
            nn.ReLU(),
            nn.Linear(hidden_dim * _HIDDEN_SCALE, output_dim),
        )

    def encode(self, x_in, x_out):
        torch, _ = _get_torch()
        x = torch.cat([x_in, x_out], dim=-1)
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)

    def reparameterise(self, mu, log_var):
        torch, _ = _get_torch()
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, x_in):
        torch, _ = _get_torch()
        return self.decoder(torch.cat([z, x_in], dim=-1))

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
            "output_dim": self.output_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "_ConditionalVAE":
        model = cls(
            input_dim=sd["input_dim"],
            output_dim=sd["output_dim"],
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


class ConditionalVAEDetector(IDetector):
    """cVAE-based semantic anomaly detector.

    Requires field_map to be configured in DetectorConfig.  The field_map
    specifies which input and output payload fields to monitor.

    Example sentinel.json detector entry:
      {
        "name": "cvae_enrichment",
        "algorithm": "cvae",
        "phase": "TRAINING",
        "training_window": 1000,
        "test_buffer_size": 150,
        "anomaly_threshold": 0.05,
        "hidden_dim": 32,
        "latent_dim": 8,
        "learning_rate": 0.001,
        "epochs": 30,
        "field_map": {
          "input": ["field_a", "field_b"],
          "output": ["result_field"]
        }
      }
    """

    def __init__(self, config: DetectorConfig) -> None:
        if config.field_map is None or (
            not config.field_map.input and not config.field_map.output
        ):
            raise ValueError(
                f"cVAE detector '{config.name}' requires a field_map with at least "
                "one input or output field. Configure field_map in sentinel.json."
            )
        self._config = config
        self._input_fields: list[str] = config.field_map.input
        self._output_fields: list[str] = config.field_map.output

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def algorithm(self) -> str:
        return "cvae"

    @property
    def training_window(self) -> int:
        return self._config.training_window

    @property
    def test_buffer_size(self) -> int:
        return self._config.test_buffer_size

    @staticmethod
    def _extract_value(body: dict, field: str) -> float:
        """Extract a single field value. Supports len(field) syntax for string length."""
        if field.startswith("len(") and field.endswith(")"):
            inner = field[4:-1]
            val = body.get(inner, "")
            return float(len(str(val)) if val is not None else 0)
        val = body.get(field, 0.0)
        try:
            return float(val)
        except (TypeError, ValueError):
            return 0.0

    def _extract_field_values(self, body: dict | None, fields: list[str]) -> list[float]:
        if body is None:
            return [0.0] * len(fields)
        return [self._extract_value(body, f) for f in fields]

    def extract_features(self, event_dict: dict) -> list[float]:
        input_body = event_dict.get("_input_body") or {}
        output_body = event_dict.get("_output_body") or {}
        return (
            self._extract_field_values(input_body, self._input_fields)
            + self._extract_field_values(output_body, self._output_fields)
        )

    def fit(self, samples: list[dict]) -> dict:
        torch, nn = _get_torch()

        input_dim = len(self._input_fields)
        output_dim = len(self._output_fields)

        if input_dim == 0 and output_dim == 0:
            raise ValueError(f"cVAE detector '{self.name}': field_map has no fields configured.")

        # Extract input/output tensors from samples
        x_in_list, x_out_list = [], []
        for s in samples:
            inp = self._extract_field_values(s.get("_input_body"), self._input_fields)
            out = self._extract_field_values(s.get("_output_body"), self._output_fields)
            x_in_list.append(inp)
            x_out_list.append(out)

        x_in = torch.tensor(x_in_list, dtype=torch.float32)
        x_out = torch.tensor(x_out_list, dtype=torch.float32)

        # Normalise to zero mean unit variance per field (fit on training set)
        in_mean = x_in.mean(dim=0)
        in_std = x_in.std(dim=0).clamp(min=1e-6)
        out_mean = x_out.mean(dim=0)
        out_std = x_out.std(dim=0).clamp(min=1e-6)

        x_in_n = (x_in - in_mean) / in_std
        x_out_n = (x_out - out_mean) / out_std

        model = _ConditionalVAE(
            input_dim=max(input_dim, 1),
            output_dim=max(output_dim, 1),
            hidden_dim=self._config.hidden_dim,
            latent_dim=self._config.latent_dim,
        )
        model.train_mode()

        kl_weight = getattr(self._config, "kl_weight", 1.0)
        optimiser = torch.optim.Adam(model.parameters(), lr=self._config.learning_rate)

        for _ in range(self._config.epochs):
            optimiser.zero_grad()
            mu, log_var = model.encode(x_in_n, x_out_n)
            z = model.reparameterise(mu, log_var)
            x_out_hat = model.decode(z, x_in_n)
            recon_loss = nn.functional.mse_loss(x_out_hat, x_out_n)
            kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
            loss = recon_loss + kl_weight * kl_loss
            loss.backward()
            optimiser.step()

        model.eval_mode()

        # Calibrate anomaly threshold from training distribution (P95 × margin).
        with torch.no_grad():
            mu_all, _ = model.encode(x_in_n, x_out_n)
            x_out_hat_all = model.decode(mu_all, x_in_n)
            per_sample_errors = ((x_out_hat_all - x_out_n) ** 2).mean(dim=1)
            p95_error = float(torch.quantile(per_sample_errors, 0.95).item())
            auto_threshold = p95_error * 1.5

        logger.info(
            "cvae_training_complete",
            detector=self.name,
            samples=len(samples),
            final_loss=round(float(loss.item()), 6),
            p95_error=round(p95_error, 6),
            auto_threshold=round(auto_threshold, 6),
        )

        return {
            "state_dict": model.state_dict(),
            "in_mean": in_mean.tolist(),
            "in_std": in_std.tolist(),
            "out_mean": out_mean.tolist(),
            "out_std": out_std.tolist(),
            "auto_threshold": auto_threshold,
        }

    def score(self, model: Any, event_dict: dict) -> float:
        torch, nn = _get_torch()

        # Retrieve cached model instance and pre-converted tensors.
        # Avoids rebuilding the neural network and re-allocating tensors on every call.
        cvae, in_mean, in_std, out_mean, out_std = _get_cached_cvae(model)

        input_body = event_dict.get("_input_body") or {}
        output_body = event_dict.get("_output_body") or {}

        x_in = torch.tensor(
            self._extract_field_values(input_body, self._input_fields),
            dtype=torch.float32,
        ).unsqueeze(0)
        x_out = torch.tensor(
            self._extract_field_values(output_body, self._output_fields),
            dtype=torch.float32,
        ).unsqueeze(0)

        x_in_n = (x_in - in_mean) / in_std
        x_out_n = (x_out - out_mean) / out_std

        with torch.no_grad():
            mu, _ = cvae.encode(x_in_n, x_out_n)
            x_out_hat = cvae.decode(mu, x_in_n)   # use mu directly — no sampling at inference
            error = float(nn.functional.mse_loss(x_out_hat, x_out_n).item())

        return error

    def is_anomaly(self, score: float, model: Any = None) -> bool:
        if model is not None and "auto_threshold" in model:
            return score > model["auto_threshold"]
        return score > self._config.anomaly_threshold
