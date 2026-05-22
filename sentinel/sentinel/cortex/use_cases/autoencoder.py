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

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sentinel.config.schema import ModelAutoencoderConfig
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class SentinelAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def create_autoencoder(input_dim: int, hidden_dim: int) -> SentinelAutoencoder:
    """Instantiate a new SentinelAutoencoder."""
    return SentinelAutoencoder(input_dim=input_dim, hidden_dim=hidden_dim)


def train_autoencoder(
    model: SentinelAutoencoder,
    samples: np.ndarray,
    config: ModelAutoencoderConfig,
) -> SentinelAutoencoder:
    """Train the autoencoder on the provided samples."""
    model.train()
    tensor = torch.tensor(samples, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = loss_fn(output, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            logger.info("autoencoder_epoch", epoch=epoch + 1, loss=epoch_loss)

    model.eval()
    return model


def compute_reconstruction_error(
    model: SentinelAutoencoder, sample: np.ndarray
) -> float:
    """Compute mean squared reconstruction error for a single sample."""
    model.eval()
    with torch.no_grad():
        tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
        output = model(tensor)
        error = nn.functional.mse_loss(output, tensor)
        return float(error.item())


def is_systemic_anomaly(
    error: float, baseline_error: float, multiplier: float = 2.0
) -> bool:
    """Return True if the reconstruction error exceeds baseline * multiplier."""
    return error > baseline_error * multiplier
