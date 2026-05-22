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

"""Training use cases for individual detectors.

Functions are pure and stateless — they receive state and return new state.
Persistence is intentionally NOT performed here; the champion/challenger
selection in process_message decides whether a trained model deserves to be saved.
"""

import asyncio
from functools import partial
from typing import Any

from sentinel.domain.ports.detector import IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


def accumulate_sample(buffer: list, sample: dict) -> list:
    """Append a sample dict to the training buffer and return the updated buffer."""
    return buffer + [sample]


def should_train(buffer: list, window_size: int) -> bool:
    """Return True if the buffer has reached the training window size."""
    return len(buffer) >= window_size


async def run_training(
    detector: IDetector,
    buffer: list[dict],
    agent_name: str,
) -> tuple[Any, list]:
    """Fit a new model (challenger) on the buffer contents.

    Returns (challenger_model, empty_buffer).  The buffer is cleared after
    training regardless of whether the challenger is promoted.

    Raises any exception from detector.fit() — caller handles persistence decisions.
    """
    logger.info(
        "training_started",
        agent=agent_name,
        detector=detector.name,
        algorithm=detector.algorithm,
        samples=len(buffer),
    )
    loop = asyncio.get_event_loop()
    challenger = await loop.run_in_executor(None, partial(detector.fit, buffer))
    logger.info(
        "challenger_trained",
        agent=agent_name,
        detector=detector.name,
    )
    return challenger, []
