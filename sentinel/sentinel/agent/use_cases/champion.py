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

"""Champion/challenger model selection for detectors.

After each training cycle the new model (challenger) is evaluated against the
held-out test buffer.  It replaces the current champion only if it achieves a
lower false-positive rate — ensuring only improvements are promoted to inference.

Invariants:
- If the test buffer is empty the challenger is accepted unconditionally,
  preserving the original behaviour for detectors with test_sample_rate=0.
- The champion is never None after the first training cycle; subsequent
  challengers must strictly beat it (strict inequality) to be promoted.
- Evaluation is purely in-memory; disk persistence is the caller's responsibility
  and should only happen when a new champion is crowned.
- The IDetector strategy is used for scoring so champion selection works with
  any algorithm without algorithm-specific branches.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from sentinel.domain.ports.detector import IDetector
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)

_NO_CHAMPION_FP_RATE = 1.0  # sentinel value: no champion yet


@dataclass(frozen=True)
class ChampionResult:
    model: Any          # trained model object (opaque, algorithm-specific)
    fp_rate: float
    is_new_champion: bool
    denial_recall: float | None = None  # recall on denial_buffer; None if buffer was empty


def _evaluate_fp_rate_sync(
    detector: IDetector,
    model: Any,
    test_buffer: list[dict],
) -> float:
    if not test_buffer:
        return _NO_CHAMPION_FP_RATE

    # Use batch_score when available (e.g. IsolationForest) to score all samples
    # in a single vectorized call instead of N individual calls.
    if hasattr(detector, "batch_score"):
        try:
            scores = detector.batch_score(model, list(test_buffer))
            flagged = sum(1 for s in scores if detector.is_anomaly(s, model))
            return flagged / len(test_buffer)
        except Exception as exc:
            logger.warning(
                "champion_batch_score_failed",
                detector=detector.name,
                error=str(exc),
            )
            # Fall through to per-sample path

    flagged = 0
    errors = 0
    for sample in test_buffer:
        try:
            score = detector.score(model, sample)
            if detector.is_anomaly(score, model):
                flagged += 1
        except Exception as exc:
            errors += 1
            flagged += 1  # conservative: scoring failures count as FP

    if errors:
        logger.warning(
            "champion_score_errors",
            detector=detector.name,
            errors=errors,
            total=len(test_buffer),
        )

    return flagged / len(test_buffer)


def evaluate_fp_rate(
    detector: IDetector,
    model: Any,
    test_buffer: list[dict],
) -> float:
    """Score every sample in the test buffer and return the false-positive rate.

    A 'false positive' is a normal training sample that the model incorrectly
    flags as anomalous.  Lower is better.

    Returns 1.0 if the buffer is empty (worst possible rate — the caller treats
    this as 'no information, accept challenger unconditionally').
    """
    return _evaluate_fp_rate_sync(detector, model, test_buffer)


def evaluate_denial_recall(
    detector: IDetector,
    model: Any,
    denial_buffer: list[dict],
) -> float:
    """Score every sample in the denial_buffer and return the recall rate.

    A denial_buffer sample is a known-bad event (flagged by a business rule).
    Recall measures how many of these the model correctly identifies as anomalous.
    Higher is better.

    Returns 1.0 if the buffer is empty (no information — do not penalise).
    """
    if not denial_buffer:
        return 1.0

    recalled = 0
    for sample in denial_buffer:
        try:
            score = detector.score(model, sample)
            if detector.is_anomaly(score, model):
                recalled += 1
        except Exception as exc:
            logger.warning(
                "denial_recall_score_error",
                detector=detector.name,
                error=str(exc),
            )

    return recalled / len(denial_buffer)


def select_champion(
    challenger: Any,
    challenger_fp_rate: float,
    current_champion: Any | None,
    current_champion_fp_rate: float,
    detector_name: str,
) -> ChampionResult:
    """Return the better model between champion and challenger.

    The challenger wins on strict improvement (fp_rate < current).
    Ties keep the existing champion for stability.
    """
    no_champion = current_champion is None
    challenger_wins = challenger_fp_rate < current_champion_fp_rate

    if no_champion or challenger_wins:
        logger.info(
            "new_champion_crowned",
            detector=detector_name,
            challenger_fp_rate=round(challenger_fp_rate, 4),
            previous_fp_rate=round(current_champion_fp_rate, 4),
        )
        return ChampionResult(
            model=challenger,
            fp_rate=challenger_fp_rate,
            is_new_champion=True,
        )

    logger.info(
        "challenger_rejected",
        detector=detector_name,
        challenger_fp_rate=round(challenger_fp_rate, 4),
        champion_fp_rate=round(current_champion_fp_rate, 4),
    )
    return ChampionResult(
        model=current_champion,
        fp_rate=current_champion_fp_rate,
        is_new_champion=False,
    )


async def run_selection(
    detector: IDetector,
    challenger: Any,
    current_champion: Any | None,
    current_champion_fp_rate: float,
    test_buffer: list[dict],
    denial_buffer: list[dict] | None = None,
    min_denial_recall: float = 0.0,
) -> ChampionResult:
    """Convenience facade: evaluate challenger then select champion.

    If the test buffer is empty, the challenger is accepted unconditionally
    (fp_rate stored as 0.0 for display purposes).

    When denial_buffer is non-empty and min_denial_recall > 0, the challenger
    must also achieve at least min_denial_recall recall on the denial_buffer
    to be promoted. A challenger that passes FP rate but fails recall is rejected.
    """
    denial_buffer = denial_buffer or []

    if not test_buffer:
        denial_recall = evaluate_denial_recall(detector, challenger, denial_buffer)
        if min_denial_recall > 0.0 and denial_recall < min_denial_recall:
            logger.info(
                "challenger_rejected_low_recall",
                detector=detector.name,
                denial_recall=round(denial_recall, 4),
                min_denial_recall=min_denial_recall,
            )
            return ChampionResult(
                model=current_champion,
                fp_rate=current_champion_fp_rate,
                is_new_champion=False,
                denial_recall=denial_recall,
            )
        return ChampionResult(
            model=challenger,
            fp_rate=0.0,
            is_new_champion=True,
            denial_recall=denial_recall if denial_buffer else None,
        )

    loop = asyncio.get_event_loop()
    challenger_fp_rate = await loop.run_in_executor(
        None, _evaluate_fp_rate_sync, detector, challenger, test_buffer
    )

    denial_recall = evaluate_denial_recall(detector, challenger, denial_buffer)

    if min_denial_recall > 0.0 and denial_recall < min_denial_recall:
        logger.info(
            "challenger_rejected_low_recall",
            detector=detector.name,
            challenger_fp_rate=round(challenger_fp_rate, 4),
            denial_recall=round(denial_recall, 4),
            min_denial_recall=min_denial_recall,
        )
        return ChampionResult(
            model=current_champion,
            fp_rate=current_champion_fp_rate,
            is_new_champion=False,
            denial_recall=denial_recall,
        )

    result = select_champion(
        challenger,
        challenger_fp_rate,
        current_champion,
        current_champion_fp_rate,
        detector.name,
    )
    return ChampionResult(
        model=result.model,
        fp_rate=result.fp_rate,
        is_new_champion=result.is_new_champion,
        denial_recall=denial_recall if denial_buffer else None,
    )
