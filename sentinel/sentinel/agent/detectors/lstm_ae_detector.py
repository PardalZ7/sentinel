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

"""LSTM Autoencoder anomaly detector for event sequences.

Detects anomalous temporal patterns across consecutive events of an agent:
inverted operation ordering, progressively degrading latency, atypical
oscillations — patterns invisible to per-event detectors.

Design (seq2seq autoencoder):
  Encoder: LSTM over the window → final hidden state h (the sequence embedding)
  Decoder: h repeated across all timesteps → LSTM → Linear back to feature space
  Loss:    MSE(reconstructed window, original window)
  Score:   per-window reconstruction MSE (higher = more anomalous)

All sequence plumbing (window slicing, padding, normalisation, threshold
calibration) lives in SequenceAutoencoderDetector — this module only defines
the recurrent network.

Requires PyTorch.
"""

from sentinel.agent.detectors.sequence_base import (
    _FEATURES_PER_STEP,
    SequenceAutoencoderDetector,
    _get_torch,
)
from sentinel.config.schema import DetectorConfig


class _LSTMAutoencoder:
    """Seq2seq LSTM autoencoder. Plain class for dict serialisation."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        _, nn = _get_torch()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        """x: (batch, seq_len, features) → reconstruction of the same shape."""
        torch, _ = _get_torch()
        seq_len = x.shape[1]
        _, (h, _) = self.encoder(x)                       # h: (1, batch, hidden)
        embedding = h[-1].unsqueeze(1)                    # (batch, 1, hidden)
        repeated = embedding.repeat(1, seq_len, 1)        # (batch, seq_len, hidden)
        decoded, _ = self.decoder(repeated)
        return self.output_layer(decoded)

    def parameters(self):
        for module in (self.encoder, self.decoder, self.output_layer):
            yield from module.parameters()

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "output_layer": self.output_layer.state_dict(),
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }

    @classmethod
    def from_state_dict(cls, sd: dict) -> "_LSTMAutoencoder":
        obj = cls(input_dim=sd["input_dim"], hidden_dim=sd["hidden_dim"])
        obj.encoder.load_state_dict(sd["encoder"])
        obj.decoder.load_state_dict(sd["decoder"])
        obj.output_layer.load_state_dict(sd["output_layer"])
        return obj

    def eval_mode(self):
        for m in (self.encoder, self.decoder, self.output_layer):
            m.eval()
        return self

    def train_mode(self):
        for m in (self.encoder, self.decoder, self.output_layer):
            m.train()
        return self


class LSTMAEDetector(SequenceAutoencoderDetector):
    """LSTM Autoencoder detector (sequence-aware).

    Example sentinel.json detector entry:
      {
        "name": "lstm_sequence",
        "algorithm": "lstm_ae",
        "phase": "TRAINING",
        "training_window": 500,
        "test_buffer_size": 100,
        "sequence_length": 10,
        "hidden_dim": 32,
        "learning_rate": 0.001,
        "epochs": 30
      }
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)

    @property
    def algorithm(self) -> str:
        return "lstm_ae"

    def _new_network(self) -> _LSTMAutoencoder:
        return _LSTMAutoencoder(
            input_dim=_FEATURES_PER_STEP,
            hidden_dim=self._config.hidden_dim,
        )

    def _network_from_state_dict(self, sd: dict) -> _LSTMAutoencoder:
        return _LSTMAutoencoder.from_state_dict(sd)
