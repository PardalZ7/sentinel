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

"""Base interface and shared context for cortex adaptation strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from sentinel.cortex.use_cases.autoencoder import SentinelAutoencoder


@dataclass
class AdaptationContext:
    """Mutable state owned by the active strategy.

    CortexRunner holds one instance and passes it to every strategy call.
    Strategies may read and write any field they need; fields irrelevant
    to the active strategy are simply ignored.
    """
    model: SentinelAutoencoder
    baseline_error: float

    # EMA
    ema_alpha: float = 0.01

    # Fine-tuning
    inference_buffer: list = field(default_factory=list)
    replay_buffer: list = field(default_factory=list)
    fine_tuning_counter: int = 0

    # Sliding window
    sliding_buffer: list = field(default_factory=list)
    sliding_counter: int = 0


class IAdaptationStrategy(ABC):
    """Contract for all cortex adaptation strategies.

    Each strategy is stateless — all mutable state lives in AdaptationContext,
    which is passed in on every call. This makes strategies trivially
    replaceable at runtime and independently testable.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy identifier."""

    @abstractmethod
    def on_inference_sample(
        self,
        sample: np.ndarray,
        ctx: AdaptationContext,
    ) -> AdaptationContext:
        """Called for every system-state sample received while in INFERENCE phase.

        Must return the (possibly mutated) context. Returning the same object
        is fine — the contract only requires the caller to use the returned value.
        """

    @abstractmethod
    def on_phase_entered(self, ctx: AdaptationContext) -> AdaptationContext:
        """Called once when the cortex transitions INTO inference mode.

        Use this hook to initialise strategy-specific buffers from the
        context's current model/baseline_error.
        """
