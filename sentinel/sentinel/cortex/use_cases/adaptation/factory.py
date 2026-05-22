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

"""Factory that maps adaptation_mode strings to concrete strategy instances."""

from sentinel.config.schema import AdaptationConfig, ModelAutoencoderConfig
from sentinel.cortex.use_cases.adaptation.base import IAdaptationStrategy
from sentinel.cortex.use_cases.adaptation.ema import EMAAdaptationStrategy
from sentinel.cortex.use_cases.adaptation.fine_tuning import FineTuningStrategy
from sentinel.cortex.use_cases.adaptation.none import NoAdaptationStrategy
from sentinel.cortex.use_cases.adaptation.sliding_window import SlidingWindowStrategy


def create_strategy(
    mode: str,
    adaptation: AdaptationConfig,
    ae_config: ModelAutoencoderConfig | None = None,
) -> IAdaptationStrategy:
    """Instantiate the correct strategy for the given mode.

    Raises ValueError for unknown modes so misconfiguration fails loudly at
    startup rather than silently falling back to a default.
    """
    match mode:
        case "none":
            return NoAdaptationStrategy()

        case "ema":
            return EMAAdaptationStrategy(alpha=adaptation.ema_alpha)

        case "fine_tuning":
            return FineTuningStrategy(
                interval=adaptation.fine_tuning_interval,
                epochs=adaptation.fine_tuning_epochs,
                lr=adaptation.fine_tuning_lr,
                replay_ratio=adaptation.replay_ratio,
                replay_buffer_size=adaptation.replay_buffer_size,
            )

        case "sliding_window":
            return SlidingWindowStrategy(
                window_size=adaptation.sliding_window_size,
                interval=adaptation.sliding_window_interval,
                ae_config=ae_config,
            )

        case _:
            raise ValueError(
                f"Unknown adaptation_mode '{mode}'. "
                "Valid options: none, ema, fine_tuning, sliding_window"
            )
