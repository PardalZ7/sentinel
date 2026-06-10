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

"""Temporal Convolutional Network (TCN) Autoencoder anomaly detector.

Same detection target as lstm_ae — anomalous temporal patterns across event
sequences — implemented with dilated causal convolutions instead of recurrence:

  - Training processes all timesteps in parallel (faster than LSTM)
  - Dilated stacking gives long receptive fields without vanishing gradients

Causality invariant: convolutions are padded on the LEFT only, so timestep t
only ever sees t-1, t-2, ... — never the future.

Architecture (encoder/decoder mirror, dilations 1→2→4 and 4→2→1):
  Encoder: CausalConv(F→H, d=1) → CausalConv(H→H, d=2) → CausalConv(H→latent, d=4)
  Decoder: CausalConv(latent→H, d=4) → CausalConv(H→H, d=2) → CausalConv(H→F, d=1)
  Each block: causal Conv1d + ReLU (no activation on the final output block).
  Loss/score: per-window reconstruction MSE.

All sequence plumbing lives in SequenceAutoencoderDetector — this module only
defines the convolutional network.

Requires PyTorch.
"""

from sentinel.agent.detectors.sequence_base import (
    _FEATURES_PER_STEP,
    SequenceAutoencoderDetector,
    _get_torch,
)
from sentinel.config.schema import DetectorConfig


class _CausalConvBlock:
    """Conv1d with left-only padding + optional ReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, relu: bool) -> None:
        _, nn = _get_torch()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
        self.left_pad = (kernel_size - 1) * dilation
        self.relu = relu

    def forward(self, x):
        torch, _ = _get_torch()
        # Left-only padding preserves causality: timestep t sees only the past.
        x = torch.nn.functional.pad(x, (self.left_pad, 0))
        out = self.conv(x)
        if self.relu:
            out = torch.relu(out)
        return out


class _TCNAutoencoder:
    """Causal convolutional autoencoder. Plain class for dict serialisation."""

    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, kernel_size: int) -> None:
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.kernel_size = kernel_size

        k = kernel_size
        self.blocks = [
            # Encoder
            _CausalConvBlock(input_dim, hidden_dim, k, dilation=1, relu=True),
            _CausalConvBlock(hidden_dim, hidden_dim, k, dilation=2, relu=True),
            _CausalConvBlock(hidden_dim, latent_dim, k, dilation=4, relu=True),
            # Decoder (mirror)
            _CausalConvBlock(latent_dim, hidden_dim, k, dilation=4, relu=True),
            _CausalConvBlock(hidden_dim, hidden_dim, k, dilation=2, relu=True),
            _CausalConvBlock(hidden_dim, input_dim, k, dilation=1, relu=False),
        ]

    def forward(self, x):
        """x: (batch, seq_len, features) → reconstruction of the same shape.

        Conv1d expects (batch, channels, seq_len); transpose at the boundaries.
        """
        out = x.transpose(1, 2)
        for block in self.blocks:
            out = block.forward(out)
        return out.transpose(1, 2)

    def parameters(self):
        for block in self.blocks:
            yield from block.conv.parameters()

    def state_dict(self) -> dict:
        return {
            "blocks": [b.conv.state_dict() for b in self.blocks],
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "latent_dim": self.latent_dim,
            "kernel_size": self.kernel_size,
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "_TCNAutoencoder":
        obj = cls(
            input_dim=sd["input_dim"],
            hidden_dim=sd["hidden_dim"],
            latent_dim=sd["latent_dim"],
            kernel_size=sd["kernel_size"],
        )
        for block, block_sd in zip(obj.blocks, sd["blocks"]):
            block.conv.load_state_dict(block_sd)
        return obj

    def eval_mode(self):
        for b in self.blocks:
            b.conv.eval()
        return self

    def train_mode(self):
        for b in self.blocks:
            b.conv.train()
        return self


class TCNAEDetector(SequenceAutoencoderDetector):
    """TCN Autoencoder detector (sequence-aware).

    Example sentinel.json detector entry:
      {
        "name": "tcn_sequence",
        "algorithm": "tcn_ae",
        "phase": "TRAINING",
        "training_window": 500,
        "test_buffer_size": 100,
        "sequence_length": 10,
        "hidden_dim": 32,
        "latent_dim": 8,
        "kernel_size": 3,
        "learning_rate": 0.001,
        "epochs": 30
      }
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)

    @property
    def algorithm(self) -> str:
        return "tcn_ae"

    def _new_network(self) -> _TCNAutoencoder:
        return _TCNAutoencoder(
            input_dim=_FEATURES_PER_STEP,
            hidden_dim=self._config.hidden_dim,
            latent_dim=self._config.latent_dim,
            kernel_size=self._config.kernel_size,
        )

    def _network_from_state_dict(self, sd: dict) -> _TCNAutoencoder:
        return _TCNAutoencoder.from_state_dict(sd)
