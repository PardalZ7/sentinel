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

"""Sliding-window strategy — full retrain over a rolling buffer of recent samples."""

import asyncio
from collections import deque

import numpy as np

from sentinel.cortex.use_cases.adaptation.base import AdaptationContext, IAdaptationStrategy
from sentinel.cortex.use_cases.autoencoder import (
    SentinelAutoencoder,
    create_autoencoder,
    train_autoencoder,
)
from sentinel.config.schema import ModelAutoencoderConfig
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


class SlidingWindowStrategy(IAdaptationStrategy):
    """Periodically retrains the autoencoder from scratch on a rolling buffer.

    The sliding buffer is a fixed-capacity FIFO deque — oldest samples are
    automatically evicted as new ones arrive, so the model always reflects
    the most recent system behaviour.

    Trade-off: if the system is in an anomalous state long enough to fill the
    buffer, that anomalous state becomes the new "normal". This is the correct
    behaviour for systems with intentional regime changes (e.g. traffic ramp-up,
    new deployment), and the wrong behaviour for persistent failures.

    Retraining is executed in a thread-pool executor to avoid blocking the
    asyncio event loop during the PyTorch training loop.
    """

    def __init__(
        self,
        window_size: int = 1000,
        interval: int = 200,
        ae_config: ModelAutoencoderConfig | None = None,
    ) -> None:
        self._window_size = window_size
        self._interval = interval
        self._ae_config = ae_config or ModelAutoencoderConfig()
        self._retraining: bool = False

    @property
    def name(self) -> str:
        return "sliding_window"

    def on_phase_entered(self, ctx: AdaptationContext) -> AdaptationContext:
        ctx.sliding_buffer = []
        ctx.sliding_counter = 0
        return ctx

    def on_inference_sample(
        self,
        sample: np.ndarray,
        ctx: AdaptationContext,
    ) -> AdaptationContext:
        buf = deque(ctx.sliding_buffer, maxlen=self._window_size)
        buf.append(sample)
        ctx.sliding_buffer = list(buf)
        ctx.sliding_counter += 1

        if (
            ctx.sliding_counter >= self._interval
            and len(ctx.sliding_buffer) >= self._window_size
            and not self._retraining
        ):
            asyncio.ensure_future(self._retrain(ctx))

        return ctx

    async def _retrain(self, ctx: AdaptationContext) -> None:
        """Full retrain in a thread-pool executor to avoid blocking the event loop."""
        self._retraining = True
        try:
            samples = np.array(ctx.sliding_buffer, dtype=np.float32)
            input_dim = samples.shape[1]
            hidden_dim = self._ae_config.hidden_dim

            loop = asyncio.get_event_loop()
            new_model, new_baseline = await loop.run_in_executor(
                None,
                self._train_sync,
                samples,
                input_dim,
                hidden_dim,
            )

            ctx.model = new_model
            ctx.baseline_error = new_baseline
            ctx.sliding_counter = 0

            logger.info(
                "cortex_sliding_window_retrain_complete",
                samples=len(samples),
                baseline_error=round(new_baseline, 6),
            )
        except Exception as exc:
            logger.warning("cortex_sliding_window_retrain_failed", error=str(exc))
        finally:
            self._retraining = False

    def _train_sync(
        self,
        samples: np.ndarray,
        input_dim: int,
        hidden_dim: int,
    ) -> tuple[SentinelAutoencoder, float]:
        import torch
        import torch.nn as nn

        model = create_autoencoder(input_dim=input_dim, hidden_dim=hidden_dim)
        model = train_autoencoder(model, samples, self._ae_config)

        tensor = torch.tensor(samples, dtype=torch.float32)
        with torch.no_grad():
            output = model(tensor)
            baseline = float(nn.functional.mse_loss(output, tensor).item())

        return model, baseline
