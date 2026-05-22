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

"""EMA strategy — adapts only the detection threshold, never the model weights."""

import numpy as np
import torch
import torch.nn as nn

from sentinel.cortex.use_cases.adaptation.base import AdaptationContext, IAdaptationStrategy
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class EMAAdaptationStrategy(IAdaptationStrategy):
    """Updates baseline_error as an exponential moving average of recent reconstruction errors.

    The autoencoder weights are never modified. Only the threshold drifts,
    allowing the detector to follow slow changes in system load or latency
    without losing the structural patterns encoded in the model.

    Convergence: with alpha=0.01, ~460 samples to incorporate 99% of a level change.
    """

    def __init__(self, alpha: float = 0.01) -> None:
        self._alpha = alpha

    @property
    def name(self) -> str:
        return "ema"

    def on_phase_entered(self, ctx: AdaptationContext) -> AdaptationContext:
        ctx.ema_alpha = self._alpha
        return ctx

    def on_inference_sample(
        self,
        sample: np.ndarray,
        ctx: AdaptationContext,
    ) -> AdaptationContext:
        current_error = self._reconstruction_error(ctx.model, sample)
        ctx.baseline_error = (
            self._alpha * current_error + (1.0 - self._alpha) * ctx.baseline_error
        )
        return ctx

    @staticmethod
    def _reconstruction_error(model: "SentinelAutoencoder", sample: np.ndarray) -> float:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            output = model(tensor)
            return float(nn.functional.mse_loss(output, tensor).item())
