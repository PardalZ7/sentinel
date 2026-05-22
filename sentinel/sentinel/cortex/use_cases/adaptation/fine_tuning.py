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

"""Fine-tuning strategy — periodic incremental updates with replay to prevent forgetting."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn

from sentinel.cortex.use_cases.adaptation.base import AdaptationContext, IAdaptationStrategy
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class FineTuningStrategy(IAdaptationStrategy):
    """Incrementally updates the autoencoder every N inference samples.

    Mixes recent samples with a replay buffer of older ones to prevent
    catastrophic forgetting. The baseline_error is updated via EMA after
    each fine-tuning pass so the detection threshold stays calibrated.

    Design invariants:
    - replay_ratio controls what fraction of the training batch comes from
      the replay buffer; the remainder comes from the current inference window.
    - Fine-tuning uses a much lower learning rate than initial training to
      make small corrections without overwriting past knowledge.
    - The replay buffer uses FIFO eviction (deque with maxlen), so the oldest
      system states are naturally forgotten over time.
    """

    def __init__(
        self,
        interval: int = 100,
        epochs: int = 8,
        lr: float = 0.0001,
        replay_ratio: float = 0.3,
        replay_buffer_size: int = 500,
        ema_alpha: float = 0.1,
    ) -> None:
        self._interval = interval
        self._epochs = epochs
        self._lr = lr
        self._replay_ratio = replay_ratio
        self._replay_buffer_size = replay_buffer_size
        self._ema_alpha = ema_alpha

    @property
    def name(self) -> str:
        return "fine_tuning"

    def on_phase_entered(self, ctx: AdaptationContext) -> AdaptationContext:
        ctx.inference_buffer = []
        ctx.replay_buffer = []
        ctx.fine_tuning_counter = 0
        return ctx

    def on_inference_sample(
        self,
        sample: np.ndarray,
        ctx: AdaptationContext,
    ) -> AdaptationContext:
        ctx.inference_buffer.append(sample)
        ctx.fine_tuning_counter += 1

        if ctx.fine_tuning_counter >= self._interval:
            ctx = self._run_fine_tuning(ctx)

        return ctx

    def _run_fine_tuning(self, ctx: AdaptationContext) -> AdaptationContext:
        batch = self._build_batch(ctx.inference_buffer, ctx.replay_buffer)
        if len(batch) < 2:
            return ctx

        samples = np.array(batch, dtype=np.float32)
        self._fine_tune(ctx.model, samples)

        new_error = self._mean_reconstruction_error(ctx.model, samples)
        ctx.baseline_error = (
            self._ema_alpha * new_error + (1.0 - self._ema_alpha) * ctx.baseline_error
        )

        replay = deque(ctx.replay_buffer, maxlen=self._replay_buffer_size)
        replay.extend(ctx.inference_buffer)
        ctx.replay_buffer = list(replay)

        logger.info(
            "cortex_fine_tuning_complete",
            new_samples=len(ctx.inference_buffer),
            replay_samples=len(ctx.replay_buffer),
            baseline_error=round(ctx.baseline_error, 6),
        )

        ctx.inference_buffer = []
        ctx.fine_tuning_counter = 0
        return ctx

    def _build_batch(
        self,
        inference_buffer: list,
        replay_buffer: list,
    ) -> list:
        if not replay_buffer:
            return inference_buffer

        n_replay = min(
            int(len(inference_buffer) * self._replay_ratio / (1.0 - self._replay_ratio)),
            len(replay_buffer),
        )
        replay_sample = random.sample(replay_buffer, n_replay) if n_replay > 0 else []
        return inference_buffer + replay_sample

    def _fine_tune(self, model: "SentinelAutoencoder", samples: np.ndarray) -> None:
        model.train()
        tensor = torch.tensor(samples, dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._lr)
        loss_fn = nn.MSELoss()

        for _ in range(self._epochs):
            optimizer.zero_grad()
            output = model(tensor)
            loss = loss_fn(output, tensor)
            loss.backward()
            optimizer.step()

        model.eval()

    @staticmethod
    def _mean_reconstruction_error(
        model: "SentinelAutoencoder", samples: np.ndarray
    ) -> float:
        model.eval()
        with torch.no_grad():
            tensor = torch.tensor(samples, dtype=torch.float32)
            output = model(tensor)
            return float(nn.functional.mse_loss(output, tensor).item())
