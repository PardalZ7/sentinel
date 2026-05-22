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

"""Core agent message processing use case.

Orchestrates the per-event detection pipeline starting from an already-correlated pair:
  1. Build event_dict with operational features + raw payload bodies
  2. For each detector in INFERENCE: score the event concurrently
  3. For each detector in TRAINING: accumulate sample, trigger training if window reached
  4. Publish ProcessedEvent to reporter
  5. Return updated AgentState (immutable — new instance on each call)

The Agent has no knowledge of correlation logic. It receives CorrelatedPair objects
produced by the CorrelationEngine and focuses exclusively on detection and training.

Multi-detector design:
  - Each detector has its own DetectorState (phase, model, buffers, champion)
  - An event is marked is_anomaly if ANY detector in INFERENCE flags it
  - The reported anomaly_score is the worst (most anomalous) score across detectors
  - Detectors in TRAINING never contribute to anomaly scoring but still accumulate samples

AgentState is immutable in the functional sense: process_pair returns a new AgentState
rather than mutating the input. This makes the state transition explicit and testable.
"""

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from sentinel.agent.use_cases.train import should_train
from sentinel.config.schema import AgentConfig
from sentinel.domain.models import (
    AnomalyType,
    CorrelatedPair,
    ModelPhase,
    ProcessedEvent,
)
from sentinel.domain.ports.detector import DetectorState, IDetector
from sentinel.domain.ports.model_store import IModelStore
from sentinel.domain.ports.reporter import IReporter
from sentinel.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AgentState:
    """Runtime state for an agent with multiple detectors.

    Each detector maintains its own independent lifecycle (phase, model, buffers,
    champion stats).  The top-level properties provide backward-compatible access
    to aggregate values for heartbeats and dashboard display.
    """

    agent_name: str
    detectors: dict[str, DetectorState]   # detector_name → DetectorState
    message_count: int = 0
    error_count: int = 0

    # ------------------------------------------------------------------
    # Backward-compat aggregate properties (used by heartbeat loop and dashboard)
    # ------------------------------------------------------------------

    @property
    def phase(self) -> ModelPhase:
        """Aggregate phase: INFERENCE only when all detectors are in INFERENCE, else TRAINING."""
        if not self.detectors:
            return ModelPhase.COLD
        phases = {d.phase for d in self.detectors.values()}
        if phases == {ModelPhase.INFERENCE}:
            return ModelPhase.INFERENCE
        if ModelPhase.COLD in phases:
            return ModelPhase.COLD
        return ModelPhase.TRAINING

    @property
    def champion_fp_rate(self) -> float:
        """Best (lowest) champion FP rate across all detectors."""
        if not self.detectors:
            return 1.0
        return min(d.champion_fp_rate for d in self.detectors.values())

    @property
    def challengers_rejected(self) -> int:
        """Total rejected challengers across all detectors."""
        return sum(d.challengers_rejected for d in self.detectors.values())

    @property
    def test_sample_rate(self) -> float:
        """Test sample rate of the first detector."""
        if not self.detectors:
            return 0.0
        return next(iter(self.detectors.values())).test_sample_rate

    @property
    def test_buffer(self) -> list:
        """Test buffer of the first detector."""
        if not self.detectors:
            return []
        return list(next(iter(self.detectors.values())).test_buffer)

    @property
    def last_test_result(self) -> dict:
        """Last test result of the first detector."""
        if not self.detectors:
            return {}
        return next(iter(self.detectors.values())).last_test_result


@lru_cache(maxsize=256)
def _compute_schema_hash(keys_tuple: tuple) -> str:
    """Compute a stable schema fingerprint from a sorted tuple of payload keys.

    Cached because payloads from a given service have a fixed schema — the same
    key set repeats for every message, making recomputation pure overhead.
    """
    schema_repr = json.dumps(list(keys_tuple))
    return hashlib.sha256(schema_repr.encode()).hexdigest()[:16]


def _build_event_dict(pair: CorrelatedPair) -> dict[str, Any]:
    """Convert a CorrelatedPair into the feature dict consumed by all detectors."""
    schema_hash = _compute_schema_hash(tuple(sorted(pair.output_body.keys())))
    return {
        "processing_latency_ms": pair.processing_latency_ms,
        "input_size_bytes": pair.input_size_bytes,
        "output_size_bytes": pair.output_size_bytes,
        "payload_schema_hash": schema_hash,
        "_input_body": pair.input_body,
        "_output_body": pair.output_body,
    }


async def process_pair(
    pair: CorrelatedPair,
    agent_config: AgentConfig,
    state: AgentState,
    detectors: dict[str, IDetector],
    model_store: IModelStore,
    reporter: IReporter,
) -> AgentState:
    """Score and train all detectors against a correlated pair, then report.

    Timed-out pairs are forwarded to the reporter as TIMEOUT events and do not
    contribute to training buffers.
    """
    if pair.timed_out:
        return await _handle_timeout(pair, agent_config, state, reporter)

    event_dict = _build_event_dict(pair)
    return await _score_and_train(
        pair=pair,
        event_dict=event_dict,
        agent_config=agent_config,
        state=state,
        detectors=detectors,
        reporter=reporter,
    )


async def _handle_timeout(
    pair: CorrelatedPair,
    agent_config: AgentConfig,
    state: AgentState,
    reporter: IReporter,
) -> AgentState:
    schema_hash = _compute_schema_hash(tuple(sorted(pair.output_body.keys())))
    event = ProcessedEvent(
        agent_name=agent_config.name,
        correlation_id=pair.correlation_id,
        input_received_at=pair.output_received_at,
        output_sent_at=pair.output_received_at,
        processing_latency_ms=0.0,
        anomaly_score=0.0,
        is_anomaly=agent_config.timeout_is_anomaly,
        anomaly_type=AnomalyType.TIMEOUT,
        payload_schema_hash=schema_hash,
        input_size_bytes=0,
        output_size_bytes=pair.output_size_bytes,
        output_body=pair.output_body,
    )
    try:
        await reporter.publish_event(event)
    except Exception as exc:
        logger.warning("reporter_unavailable", error=str(exc))
    return state


async def _score_and_train(
    pair: CorrelatedPair,
    event_dict: dict[str, Any],
    agent_config: AgentConfig,
    state: AgentState,
    detectors: dict[str, IDetector],
    reporter: IReporter,
) -> AgentState:
    """Score via all INFERENCE detectors, accumulate TRAINING samples, and publish."""
    worst_anomaly_score = 0.0
    detected_anomaly = False
    new_detectors = dict(state.detectors)
    loop = asyncio.get_running_loop()

    # ── Phase 1: Score all INFERENCE detectors concurrently ──────────
    _inference_jobs: list[tuple] = []
    for det_name, det_state in state.detectors.items():
        detector = detectors.get(det_name)
        if (
            detector is not None
            and det_state.phase == ModelPhase.INFERENCE
            and det_state.model is not None
        ):
            fut = loop.run_in_executor(None, detector.score, det_state.model, event_dict)
            _inference_jobs.append((det_name, det_state, detector, fut))

    if _inference_jobs:
        _scores = await asyncio.gather(
            *[f for _, _, _, f in _inference_jobs], return_exceptions=True
        )
        for (det_name, det_state, detector, _), score_or_exc in zip(_inference_jobs, _scores):
            if isinstance(score_or_exc, Exception):
                logger.warning(
                    "scoring_failed",
                    agent=agent_config.name,
                    detector=det_name,
                    error=str(score_or_exc),
                )
            else:
                score: float = score_or_exc
                if detector.is_anomaly(score, det_state.model):
                    detected_anomaly = True
                    if abs(score) > abs(worst_anomaly_score):
                        worst_anomaly_score = score
                    logger.info(
                        "anomaly_detected",
                        agent=agent_config.name,
                        detector=det_name,
                        algorithm=detector.algorithm,
                        score=round(score, 4),
                        threshold=detector._config.anomaly_threshold
                        if hasattr(detector, "_config") else None,
                    )

    # ── Phase 2: Training accumulation ───────────────────────────────
    for det_name, det_state in state.detectors.items():
        detector = detectors.get(det_name)
        if detector is None or det_state.phase != ModelPhase.TRAINING:
            continue

        if det_state.is_degraded:
            continue

        # Apply gating: skip contaminated samples using a trusted detector as filter.
        if det_state.gating_detector:
            gate_state = state.detectors.get(det_state.gating_detector)
            gate_detector = detectors.get(det_state.gating_detector)
            if gate_state and gate_detector and gate_state.model is not None:
                try:
                    gate_score = await loop.run_in_executor(
                        None, gate_detector.score, gate_state.model, event_dict
                    )
                    if gate_detector.is_anomaly(gate_score):
                        continue
                except Exception:
                    pass  # gate unavailable — allow sample through

        max_test_size = detector.test_buffer_size
        test_buf = det_state.test_buffer
        divert_to_test = False

        if len(test_buf) < max_test_size:
            test_buf.append(event_dict)
            divert_to_test = True
        elif det_state.test_sample_rate > 0.0 and random.random() < det_state.test_sample_rate:
            test_buf.popleft()
            test_buf.append(event_dict)
            divert_to_test = True

        if not divert_to_test:
            det_state.training_buffer.append(event_dict)

            if should_train(det_state.training_buffer, detector.training_window) and not det_state.training_in_progress:
                new_detectors[det_name] = replace(det_state, training_in_progress=True)
                continue

    # ── Build and publish event ───────────────────────────────────────
    schema_hash = event_dict["payload_schema_hash"]
    event = ProcessedEvent(
        agent_name=agent_config.name,
        correlation_id=pair.correlation_id,
        input_received_at=pair.input_received_at,
        output_sent_at=pair.output_received_at,
        processing_latency_ms=pair.processing_latency_ms,
        anomaly_score=worst_anomaly_score,
        is_anomaly=detected_anomaly,
        anomaly_type=AnomalyType.SCORE if detected_anomaly else None,
        payload_schema_hash=schema_hash,
        input_size_bytes=pair.input_size_bytes,
        output_size_bytes=pair.output_size_bytes,
        input_body=pair.input_body if detected_anomaly else None,
        output_body=pair.output_body if detected_anomaly else None,
    )

    any_in_training = any(d.phase == ModelPhase.TRAINING for d in new_detectors.values())
    send = (
        agent_config.reporting.send_all_events
        or detected_anomaly
        or any_in_training
        or bool(agent_config.cortex)
    )
    if send:
        try:
            await reporter.publish_event(event)
        except Exception as exc:
            logger.warning("reporter_unavailable_standalone", error=str(exc))

    return replace(
        state,
        detectors=new_detectors,
        message_count=state.message_count + 1,
        error_count=state.error_count + (1 if detected_anomaly else 0),
    )
