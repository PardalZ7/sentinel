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

from __future__ import annotations

from typing import TYPE_CHECKING

import asyncio
import random
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sentinel.adapters.store.redis_store import RedisCorrelationStore
from sentinel.adapters.transport.sqs_sns import SqsSnsTransport
from sentinel.config.schema import AgentConfig, CorrelationEngineConfig, CortexConfig, DetectorConfig, SentinelConfig
from sentinel.domain.models import Alert, AlertSeverity, AlertType, CorrelatedPair, HeartbeatEvent, ModelPhase
from sentinel.domain.ports.detector import DetectorState, IDetector
from sentinel.domain.ports.reporter import IReporter
from sentinel.logging.logger import get_logger

if TYPE_CHECKING:
    from sentinel.adapters.grpc.client import GrpcReporter
    from sentinel.adapters.model_store.disk_store import DiskModelStore
    from sentinel.agent.use_cases.process_message import AgentState
    from sentinel.cortex.use_cases.aggregate import AgentStateVector
    from sentinel.cortex.use_cases.adaptation import AdaptationContext, IAdaptationStrategy
    from sentinel.cortex.use_cases.autoencoder import SentinelAutoencoder
    from sentinel.dashboard.state import DashboardState

logger = get_logger(__name__)


def _synthesise_detector_configs(
    agent_config: AgentConfig,
    sentinel_config: SentinelConfig,
) -> list[DetectorConfig]:
    """Build a single IsolationForest DetectorConfig from the global model config.

    Called when an agent has no explicit detectors list, ensuring backward
    compatibility with existing sentinel.json files.
    """
    ifc = sentinel_config.model.isolation_forest
    return [
        DetectorConfig(
            name="isolation_forest",
            algorithm="isolation_forest",
            phase=agent_config.phase,
            training_window=ifc.training_window,
            test_buffer_size=agent_config.test_buffer_size or ifc.test_buffer_size,
            anomaly_threshold=ifc.anomaly_threshold,
            n_estimators=ifc.n_estimators,
            contamination=ifc.contamination,
        )
    ]


def _proto_to_pair(proto) -> CorrelatedPair:
    """Deserialize a CorrelatedPairProto into a domain CorrelatedPair."""
    import json as _json
    from datetime import timezone as _tz
    return CorrelatedPair(
        engine_name=proto.engine_name,
        correlation_id=proto.correlation_id,
        input_body=_json.loads(proto.input_body) if proto.input_body else {},
        output_body=_json.loads(proto.output_body) if proto.output_body else {},
        input_received_at=datetime.fromtimestamp(proto.input_received_at_ms / 1000, tz=_tz.utc),
        output_received_at=datetime.fromtimestamp(proto.output_received_at_ms / 1000, tz=_tz.utc),
        input_size_bytes=proto.input_size_bytes,
        output_size_bytes=proto.output_size_bytes,
        processing_latency_ms=proto.processing_latency_ms,
        timed_out=proto.timed_out,
    )


@dataclass
class AgentRunner:
    agent_config: AgentConfig
    sentinel_config: SentinelConfig
    state: AgentState
    detectors: dict[str, IDetector]
    model_store: DiskModelStore
    reporter: IReporter
    pairs_transport: SqsSnsTransport
    dashboard_state: DashboardState | None = field(default=None, repr=False)

    async def run(self) -> None:
        """Main loop: load saved models, then consume the pairs queue."""
        for det_name in list(self.state.detectors.keys()):
            try:
                model = await self.model_store.load_detector(
                    self.agent_config.name, det_name
                )
                det = self.state.detectors[det_name]
                from dataclasses import replace
                self.state.detectors[det_name] = replace(det, model=model)
                logger.info(
                    "detector_model_loaded",
                    agent=self.agent_config.name,
                    detector=det_name,
                )
            except Exception:
                pass  # no saved model yet — trains from scratch

        logger.info(
            "agent_started",
            agent=self.agent_config.name,
            detectors=list(self.state.detectors.keys()),
        )
        await asyncio.gather(
            self._run_pairs_loop(),
            self._run_heartbeat_loop(),
        )

    async def _run_heartbeat_loop(self) -> None:
        interval = self.agent_config.reporting.heartbeat_interval_s
        prev_messages = 0
        prev_errors = 0
        while True:
            await asyncio.sleep(interval)
            now = datetime.now(tz=timezone.utc)
            delta_msg = self.state.message_count - prev_messages
            delta_err = self.state.error_count - prev_errors
            prev_messages = self.state.message_count
            prev_errors = self.state.error_count
            heartbeat = HeartbeatEvent(
                agent_name=self.agent_config.name,
                timestamp=now,
                message_rate_per_sec=delta_msg / max(interval, 1),
                error_rate_last_window=delta_err / max(delta_msg, 1),
                last_message_timestamp=now,
                model_phase=self.state.phase,
            )
            try:
                await self.reporter.publish_heartbeat(heartbeat)
            except Exception as exc:
                logger.warning(
                    "agent_heartbeat_failed",
                    agent=self.agent_config.name,
                    error=str(exc),
                )

    async def _run_background_training(self, det_name: str) -> None:
        """Train a challenger in background, then apply champion selection atomically."""
        from dataclasses import replace as dc_replace
        from sentinel.agent.use_cases.train import run_training
        from sentinel.agent.use_cases.champion import run_selection

        det_state = self.state.detectors.get(det_name)
        detector = self.detectors.get(det_name)
        if det_state is None or detector is None:
            return

        buffer_snapshot = list(det_state.training_buffer)
        test_buffer_snapshot = list(det_state.test_buffer)

        try:
            challenger, _ = await run_training(detector, buffer_snapshot, self.agent_config.name)
            new_training_count = det_state.training_count + 1

            result = await run_selection(
                detector=detector,
                challenger=challenger,
                current_champion=det_state.model,
                current_champion_fp_rate=det_state.champion_fp_rate,
                test_buffer=test_buffer_snapshot,
            )

            new_model = result.model
            new_fp_rate = result.fp_rate
            new_challengers_rejected = det_state.challengers_rejected

            if result.is_new_champion:
                await self.model_store.save_detector_version(
                    self.agent_config.name, det_name, new_model
                )
            else:
                new_challengers_rejected += 1
                if det_state.max_challengers_rejected > 0 and new_challengers_rejected >= det_state.max_challengers_rejected:
                    logger.warning(
                        "detector_degraded",
                        agent=self.agent_config.name,
                        detector=det_name,
                        rejected=new_challengers_rejected,
                    )

        except NotImplementedError:
            logger.warning(
                "detector_training_not_implemented",
                agent=self.agent_config.name,
                detector=det_name,
            )
            new_model = det_state.model
            new_fp_rate = det_state.champion_fp_rate
            new_challengers_rejected = det_state.challengers_rejected
            new_training_count = det_state.training_count
        except Exception as exc:
            logger.error(
                "detector_training_failed",
                agent=self.agent_config.name,
                detector=det_name,
                error=str(exc),
            )
            new_model = det_state.model
            new_fp_rate = det_state.champion_fp_rate
            new_challengers_rejected = det_state.challengers_rejected
            new_training_count = det_state.training_count

        # Apply result atomically — preserve samples accumulated during training.
        # current_det.training_buffer contains samples that arrived while the fit
        # was running in background (output loop kept consuming). We keep those
        # and discard only the snapshot that was used for training.
        is_new_champion = new_model is not det_state.model
        current_det = self.state.detectors.get(det_name, det_state)
        post_training_samples = current_det.training_buffer[len(buffer_snapshot):]
        updated_det = dc_replace(
            current_det,
            model=new_model,
            champion_fp_rate=new_fp_rate,
            challengers_rejected=new_challengers_rejected,
            training_count=new_training_count,
            training_buffer=post_training_samples,
            training_in_progress=False,
        )
        new_detectors = dict(self.state.detectors)
        new_detectors[det_name] = updated_det
        from dataclasses import replace as dr
        self.state = dr(self.state, detectors=new_detectors)

        # Dashboard notifications
        if self.dashboard_state is not None:
            if is_new_champion:
                self.dashboard_state.update_detector_champion(
                    self.agent_config.name, det_name, new_fp_rate,
                    new_challengers_rejected, new_training_count,
                )
            else:
                self.dashboard_state.update_detector_training_count(
                    self.agent_config.name, det_name,
                    new_training_count, new_challengers_rejected,
                )
            # Auto-infer check
            threshold = updated_det.auto_infer_fp_threshold
            if (
                threshold is not None
                and updated_det.phase == ModelPhase.TRAINING
                and new_fp_rate < threshold
            ):
                logger.info(
                    "auto_infer_triggered",
                    agent=self.agent_config.name,
                    detector=det_name,
                    fp_rate=new_fp_rate,
                    threshold=threshold,
                )
                await self.dashboard_state.change_agent_phase(
                    self.agent_config.name, "INFERENCE", det_name
                )

    def _decode_pair(self, raw_body: dict) -> CorrelatedPair | None:
        """Decode a CorrelatedPairProto from the pairs queue message body."""
        hex_data = raw_body.get("_proto") or raw_body.get("raw")
        if not hex_data:
            logger.warning("pairs_queue_invalid_body", agent=self.agent_config.name)
            return None
        from sentinel.adapters.grpc.generated.correlated_pair_pb2 import CorrelatedPairProto
        proto = CorrelatedPairProto()
        proto.ParseFromString(bytes.fromhex(hex_data))
        return _proto_to_pair(proto)

    async def _run_pairs_loop(self) -> None:
        _training_tasks: dict[str, asyncio.Task] = {}

        from sentinel.agent.use_cases.process_message import process_pair
        async for raw_message in self.pairs_transport.receive():
            try:
                pair = self._decode_pair(raw_message.body)
                if pair is None:
                    await self.pairs_transport.nack(raw_message.id)
                    continue

                prev_buf_counts = (
                    {name: len(det.test_buffer) for name, det in self.state.detectors.items()}
                    if self.dashboard_state is not None
                    else {}
                )

                self.state = await process_pair(
                    pair=pair,
                    agent_config=self.agent_config,
                    state=self.state,
                    detectors=self.detectors,
                    model_store=self.model_store,
                    reporter=self.reporter,
                )

                # Single pass: spawn background training + notify dashboard.
                for det_name, det_state in self.state.detectors.items():
                    if det_state.training_in_progress:
                        existing = _training_tasks.get(det_name)
                        if existing is None or existing.done():
                            task = asyncio.create_task(
                                self._run_background_training(det_name),
                                name=f"train-{self.agent_config.name}-{det_name}",
                            )
                            _training_tasks[det_name] = task

                    if self.dashboard_state is not None:
                        new_buf = len(det_state.test_buffer)
                        if new_buf != prev_buf_counts.get(det_name, 0):
                            self.dashboard_state.update_detector_buffer_count(
                                self.agent_config.name, det_name, new_buf,
                            )

                await self.pairs_transport.ack(raw_message.id)
                await self.pairs_transport.flush_acks()
            except Exception as exc:
                logger.error(
                    "agent_processing_error",
                    agent=self.agent_config.name,
                    error=str(exc),
                )
                await self.pairs_transport.nack(raw_message.id)

    async def change_phase(
        self, new_phase: str, detector_name: str | None = None
    ) -> None:
        """Change phase for a specific detector or all detectors if detector_name is None."""
        from sentinel.agent.use_cases.set_phase import set_phase
        phase = ModelPhase(new_phase)
        self.state = set_phase(self.state, phase, detector_name)
        # Force send_all_events when entering TRAINING so Cortex receives all data
        if phase == ModelPhase.TRAINING:
            self.agent_config.reporting.send_all_events = True
        logger.info(
            "agent_phase_changed",
            agent=self.agent_config.name,
            detector=detector_name or "all",
            phase=new_phase,
        )

    async def set_test_rate(
        self, rate: float, detector_name: str | None = None
    ) -> None:
        """Set the test sample rate for a specific detector or all detectors."""
        from dataclasses import replace

        clamped = max(0.0, min(1.0, rate))
        new_detectors = dict(self.state.detectors)

        target = [detector_name] if detector_name else list(new_detectors.keys())
        for dname in target:
            if dname in new_detectors:
                new_detectors[dname] = replace(new_detectors[dname], test_sample_rate=clamped)

        self.state = replace(self.state, detectors=new_detectors)

        if self.dashboard_state is not None:
            for dname in target:
                if dname in self.state.detectors:
                    det = self.state.detectors[dname]
                    self.dashboard_state.update_detector_test_rate(
                        self.agent_config.name, dname, clamped, len(det.test_buffer)
                    )

    async def run_test(self, detector_name: str | None = None) -> dict:
        """Run test buffer evaluation for a specific detector or all detectors."""
        from sentinel.agent.use_cases.run_test import run_detector_test
        from dataclasses import replace

        results: dict[str, dict] = {}
        new_detectors = dict(self.state.detectors)

        target = [detector_name] if detector_name else list(new_detectors.keys())
        for dname in target:
            det_state = new_detectors.get(dname)
            detector = self.detectors.get(dname)
            if det_state is None or detector is None:
                continue
            result = run_detector_test(detector, det_state)
            new_detectors[dname] = replace(det_state, last_test_result=result)
            results[dname] = result
            if self.dashboard_state is not None:
                self.dashboard_state.update_detector_test_result(
                    self.agent_config.name, dname, result
                )

        self.state = replace(self.state, detectors=new_detectors)
        return results

    async def set_auto_infer(
        self, detector_name: str, threshold: float | None
    ) -> None:
        """Enable (threshold=float) or disable (threshold=None) auto-transition to INFERENCE."""
        from dataclasses import replace

        new_detectors = dict(self.state.detectors)
        if detector_name in new_detectors:
            new_detectors[detector_name] = replace(
                new_detectors[detector_name], auto_infer_fp_threshold=threshold
            )
            self.state = replace(self.state, detectors=new_detectors)


@dataclass
class CortexRunner:
    cortex_config: CortexConfig
    sentinel_config: SentinelConfig
    dashboard_state: DashboardState | None = None
    state_vectors: dict[str, AgentStateVector] = field(default_factory=dict)
    causal_window: deque = field(default_factory=lambda: deque(maxlen=100))
    phase: ModelPhase = field(default=ModelPhase.TRAINING)
    autoencoder_model: SentinelAutoencoder | None = None
    training_buffer: list = field(default_factory=list)
    baseline_error: float = 0.01
    test_sample_rate: float = 0.0
    test_buffer: deque = field(default_factory=deque)
    last_test_result: dict = field(default_factory=dict)
    _strategy: IAdaptationStrategy | None = field(default=None, repr=False)
    _adaptation_ctx: AdaptationContext | None = field(default=None, repr=False)
    _parent_reporters: list[GrpcReporter] = field(default_factory=list, repr=False)
    _model_store: DiskModelStore | None = field(default=None, repr=False)
    _correlation_store: RedisCorrelationStore | None = field(default=None, repr=False)
    _grouping_ttl_s: int = field(default=300, repr=False)
    _last_training_sample_at: datetime | None = field(default=None, repr=False)

    def _all_inputs_in_inference(self) -> bool:
        from sentinel.domain.models import ModelPhase as _MP
        for name in self.cortex_config.inputs:
            sv = self.state_vectors.get(name)
            if sv is None or sv.model_phase != _MP.INFERENCE:
                return False
        return True

    async def _accumulate_training_sample(self):
        from sentinel.cortex.use_cases.aggregate import build_feature_matrix

        # INFERENCE: always build and return the current sample for scoring.
        if self.phase == ModelPhase.INFERENCE:
            matrix = build_feature_matrix(self.state_vectors)
            if matrix.shape[0] == 0:
                return None
            return matrix.mean(axis=0)

        # TRAINING: check cheap conditions before building the feature matrix.
        # build_feature_matrix + mean are skipped on the vast majority of calls
        # where the sampling interval has not elapsed yet.
        if not self._all_inputs_in_inference():
            return None

        now = datetime.now(tz=timezone.utc)
        interval = self.cortex_config.training_sample_interval_s
        if self._last_training_sample_at is not None:
            elapsed = (now - self._last_training_sample_at).total_seconds()
            if elapsed < interval:
                return None

        matrix = build_feature_matrix(self.state_vectors)
        if matrix.shape[0] == 0:
            return None
        sample = matrix.mean(axis=0)
        self._last_training_sample_at = now

        divert_to_test = False
        max_size = (
            self.cortex_config.test_buffer_size
            or self.sentinel_config.model.autoencoder.test_buffer_size
        )
        if len(self.test_buffer) < max_size:
            self.test_buffer.append(sample)
            divert_to_test = True
        elif self.test_sample_rate > 0.0 and random.random() < self.test_sample_rate:
            self.test_buffer.popleft()
            self.test_buffer.append(sample)
            divert_to_test = True

        if divert_to_test:
            if self.dashboard_state:
                self.dashboard_state.update_cortex_test_buffer(
                    self.cortex_config.name, len(self.test_buffer)
                )
        else:
            self.training_buffer.append(sample)
            if self.dashboard_state:
                self.dashboard_state.record_cortex_training_sample(self.cortex_config.name)

        return None

    async def change_phase(self, new_phase: str) -> None:
        phase = ModelPhase(new_phase)

        if phase == ModelPhase.INFERENCE:
            await self._ensure_model_trained()
            if self.autoencoder_model is None:
                return
            self._activate_strategy()

        self.phase = phase
        logger.info("cortex_phase_changed", cortex=self.cortex_config.name, phase=new_phase)
        if self.dashboard_state:
            self.dashboard_state.update_cortex_phase(self.cortex_config.name, new_phase)

        await self._send_heartbeat_to_parent()

    async def _ensure_model_trained(self) -> None:
        if self.autoencoder_model is not None:
            return

        if len(self.training_buffer) == 0:
            logger.warning("cortex_no_training_data", cortex=self.cortex_config.name)
            return

        import torch

        ae_config = self.sentinel_config.model.autoencoder
        logger.info(
            "cortex_training_autoencoder",
            cortex=self.cortex_config.name,
            samples=len(self.training_buffer),
        )
        import numpy as np
        samples = np.array(self.training_buffer, dtype=np.float32)
        input_dim = samples.shape[1]
        from sentinel.cortex.use_cases.autoencoder import create_autoencoder, train_autoencoder
        model = create_autoencoder(input_dim=input_dim, hidden_dim=ae_config.hidden_dim)
        loop = asyncio.get_event_loop()
        self.autoencoder_model = await loop.run_in_executor(
            None, train_autoencoder, model, samples, ae_config
        )

        with torch.no_grad():
            tensor = torch.tensor(samples, dtype=torch.float32)
            output = self.autoencoder_model(tensor)
            self.baseline_error = float(
                torch.nn.functional.mse_loss(output, tensor).item()
            )

        self.training_buffer.clear()

        if self._model_store is not None:
            try:
                await self._model_store.save_cortex_version(
                    self.cortex_config.name,
                    self.autoencoder_model,
                    {"baseline_error": self.baseline_error, "input_dim": input_dim,
                     "hidden_dim": ae_config.hidden_dim},
                )
            except Exception as exc:
                logger.warning(
                    "cortex_model_save_failed",
                    cortex=self.cortex_config.name,
                    error=str(exc),
                )

    def _activate_strategy(self) -> None:
        from sentinel.cortex.use_cases.adaptation import AdaptationContext, create_strategy
        ae_config = self.sentinel_config.model.autoencoder
        self._strategy = create_strategy(
            mode=self.cortex_config.adaptation_mode,
            adaptation=self.cortex_config.adaptation,
            ae_config=ae_config,
        )
        ctx = AdaptationContext(model=self.autoencoder_model, baseline_error=self.baseline_error)
        self._adaptation_ctx = self._strategy.on_phase_entered(ctx)
        logger.info(
            "cortex_strategy_activated",
            cortex=self.cortex_config.name,
            strategy=self._strategy.name,
        )

    def set_test_rate(self, rate: float) -> None:
        self.test_sample_rate = max(0.0, min(1.0, rate))
        if self.dashboard_state:
            self.dashboard_state.update_cortex_test_rate(
                self.cortex_config.name, self.test_sample_rate
            )

    def run_test(self) -> dict:
        import torch
        import torch.nn as nn

        if self.autoencoder_model is None:
            return {"error": "no_model", "samples_run": 0, "failed": 0}
        if not self.test_buffer:
            return {"samples_run": 0, "failed": 0}

        try:
            import numpy as np
            samples = np.array(self.test_buffer, dtype=np.float32)
            tensor = torch.tensor(samples, dtype=torch.float32)
            with torch.no_grad():
                output = self.autoencoder_model(tensor)
                errors = nn.functional.mse_loss(output, tensor, reduction="none").mean(dim=1)
            threshold = self.baseline_error * 2.0
            failed = int((errors > threshold).sum().item())
            result = {"samples_run": len(self.test_buffer), "failed": failed}
        except Exception as exc:
            logger.warning("cortex_test_failed", cortex=self.cortex_config.name, error=str(exc))
            result = {"error": str(exc), "samples_run": 0, "failed": 0}

        self.last_test_result = result
        if self.dashboard_state:
            self.dashboard_state.update_cortex_test_result(self.cortex_config.name, result)
        logger.info("cortex_test_complete", cortex=self.cortex_config.name, **result)
        return result

    async def _forward_event_to_parents(self, event) -> None:
        if not self._parent_reporters:
            return
        from sentinel.domain.models import ProcessedEvent as PE
        forwarded = PE(
            agent_name=self.cortex_config.name,
            correlation_id=event.correlation_id,
            input_received_at=event.input_received_at,
            output_sent_at=event.output_sent_at,
            processing_latency_ms=event.processing_latency_ms,
            anomaly_score=event.anomaly_score,
            is_anomaly=event.is_anomaly,
            anomaly_type=event.anomaly_type,
            payload_schema_hash=event.payload_schema_hash,
            input_size_bytes=event.input_size_bytes,
            output_size_bytes=event.output_size_bytes,
        )
        for reporter in self._parent_reporters:
            try:
                await reporter.publish_event(forwarded)
            except Exception as exc:
                logger.warning(
                    "cortex_parent_forward_failed",
                    cortex=self.cortex_config.name,
                    error=str(exc),
                )

    async def _send_heartbeat_to_parent(self) -> None:
        if not self._parent_reporters:
            return
        now = datetime.now(tz=timezone.utc)
        heartbeat = HeartbeatEvent(
            agent_name=self.cortex_config.name,
            timestamp=now,
            message_rate_per_sec=0.0,
            error_rate_last_window=0.0,
            last_message_timestamp=now,
            model_phase=self.phase,
        )
        for reporter in self._parent_reporters:
            try:
                await reporter.publish_heartbeat(heartbeat)
            except Exception as exc:
                logger.warning(
                    "cortex_parent_heartbeat_failed",
                    cortex=self.cortex_config.name,
                    error=str(exc),
                )

    async def handle_event(self, proto_event: object) -> None:
        from sentinel.adapters.grpc.generated import sentinel_pb2
        from sentinel.domain.models import ProcessedEvent

        if isinstance(proto_event, sentinel_pb2.ProcessedEventProto):
            event = ProcessedEvent(
                agent_name=proto_event.agent_name,
                correlation_id=proto_event.correlation_id,
                input_received_at=datetime.fromtimestamp(
                    proto_event.input_received_at_ms / 1000, tz=timezone.utc
                ),
                output_sent_at=datetime.fromtimestamp(
                    proto_event.output_sent_at_ms / 1000, tz=timezone.utc
                ),
                processing_latency_ms=proto_event.processing_latency_ms,
                anomaly_score=proto_event.anomaly_score,
                is_anomaly=proto_event.is_anomaly,
                anomaly_type=None,
                payload_schema_hash=proto_event.payload_schema_hash,
                input_size_bytes=proto_event.input_size_bytes,
                output_size_bytes=proto_event.output_size_bytes,
            )
            from sentinel.cortex.use_cases.aggregate import update_state, check_silence
            from sentinel.cortex.use_cases.causal_chain import add_event
            self.state_vectors = update_state(self.state_vectors, event)
            if self.phase == ModelPhase.INFERENCE:
                add_event(self.causal_window, event)
            await self._handle_grouping(event)

        elif isinstance(proto_event, sentinel_pb2.HeartbeatProto):
            from sentinel.cortex.use_cases.aggregate import update_state, check_silence
            input_phase = (
                ModelPhase(proto_event.model_phase)
                if proto_event.model_phase
                else ModelPhase.COLD
            )
            event = HeartbeatEvent(
                agent_name=proto_event.agent_name,
                timestamp=datetime.fromtimestamp(
                    proto_event.timestamp_ms / 1000, tz=timezone.utc
                ),
                message_rate_per_sec=proto_event.message_rate_per_sec,
                error_rate_last_window=proto_event.error_rate_last_window,
                last_message_timestamp=datetime.fromtimestamp(
                    proto_event.last_message_timestamp_ms / 1000, tz=timezone.utc
                ),
                model_phase=input_phase,
            )
            self.state_vectors = update_state(self.state_vectors, event)
            now = datetime.now(tz=timezone.utc)
            for agent_name in self.cortex_config.inputs:
                if check_silence(
                    self.state_vectors, agent_name,
                    self.cortex_config.silence_threshold_s, now
                ):
                    logger.warning("agent_silence_detected", agent=agent_name)
                    if self.dashboard_state:
                        self.dashboard_state.mark_silent(agent_name)

        current_sample = await self._accumulate_training_sample()
        if self.phase == ModelPhase.INFERENCE:
            await self._run_inference(sample=current_sample)

    async def _handle_grouping(self, event) -> None:
        if not self._parent_reporters or self._correlation_store is None:
            self._record_dashboard_event()
            await self._forward_event_to_parents(event)
            return

        expected = self.cortex_config.inputs
        if not expected:
            self._record_dashboard_event()
            await self._forward_event_to_parents(event)
            return

        cid = event.correlation_id
        if not cid:
            self._record_dashboard_event()
            await self._forward_event_to_parents(event)
            return

        try:
            from sentinel.cortex.use_cases.group_buffer import (
                record_source_event, is_complete, build_grouped_event, delete_group,
            )
            group = await record_source_event(
                store=self._correlation_store,
                cortex_name=self.cortex_config.name,
                correlation_id=cid,
                source_name=event.agent_name,
                event=event,
                expected_sources=expected,
                ttl_s=self._grouping_ttl_s,
            )
            if is_complete(group, expected):
                grouped = build_grouped_event(group, self.cortex_config.name, cid, expected)
                self._record_dashboard_event()
                await self._forward_event_to_parents(grouped)
                await delete_group(self._correlation_store, self.cortex_config.name, cid)
        except Exception as exc:
            logger.warning(
                "cortex_grouping_failed",
                cortex=self.cortex_config.name,
                error=str(exc),
            )
            self._record_dashboard_event()
            await self._forward_event_to_parents(event)

    def _record_dashboard_event(self) -> None:
        if self.dashboard_state:
            self.dashboard_state.record_cortex_event(self.cortex_config.name)

    async def _run_inference(self, sample=None) -> None:
        from sentinel.cortex.dual_network import run_dual_network

        if (
            sample is not None
            and self._strategy is not None
            and self._adaptation_ctx is not None
        ):
            loop = asyncio.get_event_loop()
            self._adaptation_ctx = await loop.run_in_executor(
                None, self._strategy.on_inference_sample, sample, self._adaptation_ctx
            )
            self.autoencoder_model = self._adaptation_ctx.model
            self.baseline_error = self._adaptation_ctx.baseline_error

        result = run_dual_network(
            causal_window=self.causal_window,
            autoencoder_model=self.autoencoder_model,
            state_vectors=self.state_vectors,
            config=self.cortex_config,
            baseline_error=self.baseline_error,
            source_name=self.cortex_config.name,
        )
        if result.final_alert is not None:
            await self._emit_alert(result.final_alert)

    async def _emit_alert(self, alert: Alert) -> None:
        logger.warning(
            "cortex_alert",
            cortex=self.cortex_config.name,
            alert_type=alert.alert_type.value,
            severity=alert.severity.value,
            description=alert.description,
        )
        if self.dashboard_state:
            self.dashboard_state.record_alert(alert)

    async def _group_expiry_task(self) -> None:
        import time

        while True:
            await asyncio.sleep(self._grouping_ttl_s)
            if self._correlation_store is None or not self._parent_reporters:
                continue
            expected = self.cortex_config.inputs
            if not expected:
                continue
            try:
                client = self._correlation_store._get_client()
                pattern = f"cortex:{self.cortex_config.name}:group:*"
                cursor = 0
                now_ms = int(time.time() * 1000)
                while True:
                    cursor, keys = await client.scan(cursor, match=pattern, count=200)
                    for key in keys:
                        try:
                            raw = await client.get(key)
                            if raw is None:
                                continue
                            import json as _json
                            group = _json.loads(raw)
                            created = group.get("_created_at_ms", now_ms)
                            age_s = (now_ms - created) / 1000
                            if age_s >= self._grouping_ttl_s:
                                from sentinel.cortex.use_cases.group_buffer import build_grouped_event
                                cid = key.split(":", 3)[-1]
                                grouped = build_grouped_event(
                                    group, self.cortex_config.name, cid, expected
                                )
                                from sentinel.domain.models import AnomalyType
                                grouped.is_anomaly = True
                                grouped.anomaly_type = AnomalyType.TIMEOUT
                                await self._forward_event_to_parents(grouped)
                                await self._correlation_store.delete(key)
                                logger.warning(
                                    "cortex_group_expired",
                                    cortex=self.cortex_config.name,
                                    correlation_id=cid,
                                    present_sources=[s for s in expected if group.get(s)],
                                )
                        except Exception as exc:
                            logger.warning("cortex_expiry_key_error", key=key, error=str(exc))
                    if cursor == 0:
                        break
            except Exception as exc:
                logger.warning(
                    "cortex_expiry_scan_failed",
                    cortex=self.cortex_config.name,
                    error=str(exc),
                )

    async def run(self) -> None:
        if self._model_store is not None:
            try:
                from sentinel.cortex.use_cases.autoencoder import create_autoencoder
                ae_config = self.sentinel_config.model.autoencoder

                def _factory():
                    return create_autoencoder(
                        input_dim=self._model_store.cortex_manifest(self.cortex_config.name)
                            .get("versions", [{}])[-1].get("input_dim", ae_config.hidden_dim),
                        hidden_dim=ae_config.hidden_dim,
                    )

                model, meta = await self._model_store.load_cortex_version(
                    self.cortex_config.name, _factory
                )
                self.autoencoder_model = model
                self.baseline_error = meta.get("baseline_error", self.baseline_error)
                self.phase = ModelPhase.INFERENCE
                self._activate_strategy()
                logger.info("cortex_model_loaded_from_disk", cortex=self.cortex_config.name)
            except Exception:
                pass

        if self.dashboard_state:
            self.dashboard_state.register_cortex(
                self.cortex_config.name, self.cortex_config.inputs, self.phase.value
            )
            self.dashboard_state.register_cortex_callbacks(
                self.cortex_config.name,
                change_phase_cb=self.change_phase,
                set_rate_cb=self.set_test_rate,
                run_test_cb=self.run_test,
            )

        from sentinel.adapters.grpc.server import create_server, start_server
        server = create_server(
            host="0.0.0.0",
            port=self.cortex_config.grpc_port,
            event_handler=self.handle_event,
        )
        await asyncio.gather(
            start_server(server),
            self._group_expiry_task(),
        )


class _MultiReporter(IReporter):
    def __init__(self, reporters: list[GrpcReporter]) -> None:
        self._reporters = reporters

    async def publish_event(self, event) -> None:
        await asyncio.gather(*(r.publish_event(event) for r in self._reporters))

    async def publish_heartbeat(self, heartbeat) -> None:
        await asyncio.gather(*(r.publish_heartbeat(heartbeat) for r in self._reporters))


def _build_base_reporter(agent_config: AgentConfig) -> IReporter:
    from sentinel.adapters.grpc.client import GrpcReporter

    if not agent_config.cortex:
        class _StandaloneReporter(IReporter):
            async def publish_event(self, event):
                logger.info("standalone_event", agent=event.agent_name)

            async def publish_heartbeat(self, heartbeat):
                logger.info("standalone_heartbeat", agent=heartbeat.agent_name)

        return _StandaloneReporter()

    reporters = [GrpcReporter(host=ref.host, port=ref.port) for ref in agent_config.cortex]
    if len(reporters) == 1:
        return reporters[0]
    return _MultiReporter(reporters)


def build_correlation_engine(
    engine_config: CorrelationEngineConfig,
    sentinel_config: SentinelConfig,
) -> "CorrelationEngineRunner":
    """Wire up all dependencies for a CorrelationEngineRunner."""
    from sentinel.runtime.correlation_engine_runner import CorrelationEngineRunner
    from sentinel.adapters.queue.sqs_pair_publisher import SqsPairPublisher
    from sentinel.adapters.alerting.sns_timeout_notifier import SnsTimeoutNotifier

    endpoint = sentinel_config.aws.endpoint_url
    region = sentinel_config.aws.region

    input_transport = SqsSnsTransport(
        endpoint_url=endpoint,
        input_queue_url=engine_config.input.resource,
        output_topic_arn="",
        region=region,
        wait_time_seconds=1,
    )
    output_transport = SqsSnsTransport(
        endpoint_url=endpoint,
        input_queue_url=engine_config.output.resource,
        output_topic_arn="",
        region=region,
        wait_time_seconds=1,
    )
    correlation_store = RedisCorrelationStore(
        host=sentinel_config.redis.host,
        port=sentinel_config.redis.port,
        db=sentinel_config.redis.db,
        password=sentinel_config.redis.password,
    )
    pair_publishers = [
        SqsPairPublisher(
            queue_url=dest.resource,
            region=region,
            endpoint_url=endpoint,
        )
        for dest in engine_config.destinations
    ]
    timeout_notifier = (
        SnsTimeoutNotifier(
            topic_arn=engine_config.timeout_topic_arn,
            region=region,
            endpoint_url=endpoint,
        )
        if engine_config.timeout_topic_arn
        else None
    )
    return CorrelationEngineRunner(
        engine_config=engine_config,
        input_transport=input_transport,
        output_transport=output_transport,
        correlation_store=correlation_store,
        pair_publishers=pair_publishers,
        timeout_notifier=timeout_notifier,
    )


def build_agent(
    agent_config: AgentConfig,
    sentinel_config: SentinelConfig,
    dashboard_state: DashboardState | None = None,
) -> AgentRunner:
    """Wire up all dependencies for an AgentRunner."""
    from sentinel.adapters.model_store.disk_store import DiskModelStore
    from sentinel.adapters.reporting.dashboard_reporter import DashboardReporter
    from sentinel.agent.detectors.factory import create_detector
    from sentinel.agent.use_cases.process_message import AgentState
    from sentinel.agent.use_cases.set_phase import set_phase  # noqa: F401

    storage_path = Path(sentinel_config.storage_path).expanduser()

    model_store = DiskModelStore(
        base_path=str(storage_path),
        max_versions=sentinel_config.model.max_versions,
    )
    base_reporter = _build_base_reporter(agent_config)
    reporter: IReporter = DashboardReporter(base_reporter)

    endpoint = sentinel_config.aws.endpoint_url
    region = sentinel_config.aws.region

    pairs_transport = SqsSnsTransport(
        endpoint_url=endpoint,
        input_queue_url=agent_config.pairs_queue.resource,
        output_topic_arn="",
        region=region,
        wait_time_seconds=1,
    )

    # Determine detector configs — synthesise from global IF config if not explicitly defined
    detector_configs = agent_config.detectors or _synthesise_detector_configs(
        agent_config, sentinel_config
    )

    # Build detector strategy objects and initial DetectorState instances
    detectors: dict[str, IDetector] = {}
    initial_detector_states: dict[str, DetectorState] = {}

    for dc in detector_configs:
        detector = create_detector(dc)
        detectors[dc.name] = detector
        initial_detector_states[dc.name] = DetectorState(
            name=dc.name,
            algorithm=dc.algorithm,
            phase=ModelPhase(dc.phase),
            model=None,
            training_buffer=[],
            auto_infer_fp_threshold=dc.auto_infer_fp_threshold,
            gating_detector=dc.gating_detector,
            max_challengers_rejected=dc.max_challengers_rejected,
        )

    state = AgentState(
        agent_name=agent_config.name,
        detectors=initial_detector_states,
    )

    runner = AgentRunner(
        agent_config=agent_config,
        sentinel_config=sentinel_config,
        state=state,
        detectors=detectors,
        model_store=model_store,
        reporter=reporter,
        pairs_transport=pairs_transport,
        dashboard_state=dashboard_state,
    )

    if dashboard_state is not None:
        dashboard_state.register_agent(
            agent_config.name,
            initial_detector_states=initial_detector_states,
        )
        dashboard_state.register_phase_callback(
            agent_config.name, runner.change_phase
        )
        dashboard_state.register_agent_test_callbacks(
            agent_config.name,
            set_rate_cb=runner.set_test_rate,
            run_test_cb=runner.run_test,
        )
        dashboard_state.register_auto_infer_callback(
            agent_config.name, runner.set_auto_infer
        )

    return runner


def build_cortex(
    cortex_config: CortexConfig,
    sentinel_config: SentinelConfig,
    dashboard_state: DashboardState | None = None,
) -> CortexRunner:
    from sentinel.adapters.grpc.client import GrpcReporter
    from sentinel.adapters.model_store.disk_store import DiskModelStore

    parent_reporters = [
        GrpcReporter(host=ref.host, port=ref.port)
        for ref in cortex_config.parent_cortex
    ]

    storage_path = Path(sentinel_config.storage_path).expanduser()
    model_store = DiskModelStore(
        base_path=str(storage_path),
        max_versions=sentinel_config.model.max_versions,
    )

    grouping_ttl_s = cortex_config.grouping_ttl_s
    if grouping_ttl_s <= 0:
        # Derive from the correlation engines whose destinations feed these agents
        engine_ttls = [
            e.correlation_ttl_s
            for e in sentinel_config.correlation_engines
            if any(a.name in cortex_config.inputs for a in sentinel_config.agents)
        ]
        grouping_ttl_s = max(engine_ttls, default=300)

    correlation_store = (
        RedisCorrelationStore(
            host=sentinel_config.redis.host,
            port=sentinel_config.redis.port,
            db=sentinel_config.redis.db,
            password=sentinel_config.redis.password,
        )
        if parent_reporters
        else None
    )

    return CortexRunner(
        cortex_config=cortex_config,
        sentinel_config=sentinel_config,
        dashboard_state=dashboard_state,
        _parent_reporters=parent_reporters,
        _model_store=model_store,
        _correlation_store=correlation_store,
        _grouping_ttl_s=grouping_ttl_s,
    )


async def launch(
    config: SentinelConfig,
    mode: str = "all",
) -> None:
    """Start Sentinel components according to *mode*.

    Modes:
      all     — start everything: engines, agents, cortex, dashboard (default)
      engine  — start only CorrelationEngines (use-case side)
      agent   — start only Agents and Cortex nodes (Sentinel cloud side)
    """
    from sentinel.logging.logger import configure_logging

    configure_logging()

    run_engines = mode in ("all", "engine")
    run_agents = mode in ("all", "agent")

    from sentinel.dashboard.state import get_state
    ds = get_state() if (config.dashboard.enabled and run_agents) else None
    if ds is not None:
        ds.set_config(config.model_dump(mode="json"))

    tasks = []

    # When control_plane is enabled, all topology components (engines, agents,
    # cortex) are managed exclusively by TopologyManager — either restored from
    # disk on restart or deployed via PUT /cp/topologies/{id}. Starting them
    # statically here would duplicate runners and cause port conflicts on cortex.
    cp_enabled = config.control_plane.enabled

    if run_engines and (not cp_enabled or mode == "engine"):
        for engine_config in config.correlation_engines:
            runner = build_correlation_engine(engine_config, config)
            tasks.append(asyncio.create_task(runner.run(), name=f"engine-{engine_config.name}"))

    if run_agents:
        from sentinel.runtime.topology_manager import TopologyManager
        from sentinel.runtime.topology_store import TopologyStore

        topology_store = TopologyStore(config.storage_path)
        topology_manager = TopologyManager(config, ds)
        await topology_manager.restore_from_store(topology_store)

        if not cp_enabled:
            for agent_config in config.agents:
                runner = build_agent(agent_config, config, ds)
                tasks.append(asyncio.create_task(runner.run(), name=f"agent-{agent_config.name}"))

            for cortex_config in config.cortex:
                runner = build_cortex(cortex_config, config, ds)
                tasks.append(asyncio.create_task(runner.run(), name=f"cortex-{cortex_config.name}"))

        if config.dashboard.enabled and ds is not None:
            from sentinel.dashboard.server import start_dashboard
            cp_manager = topology_manager if config.control_plane.enabled else None
            cp_store = topology_store if config.control_plane.enabled else None
            tasks.append(
                asyncio.create_task(
                    start_dashboard(
                        ds,
                        host=config.dashboard.host,
                        port=config.dashboard.port,
                        topology_manager=cp_manager,
                        topology_store=cp_store,
                    ),
                    name="dashboard",
                )
            )

    if not tasks:
        logger.warning("no_components_for_mode", mode=mode)
        return

    logger.info(
        "sentinel_launched",
        mode=mode,
        engines=len(config.correlation_engines) if run_engines else 0,
        agents=len(config.agents) if run_agents else 0,
        cortex=len(config.cortex) if run_agents else 0,
        dashboard=(
            f"http://{config.dashboard.host}:{config.dashboard.port}"
            if config.dashboard.enabled
            else "disabled"
        ),
        control_plane=(
            f"http://{config.dashboard.host}:{config.dashboard.port}/cp"
            if (run_agents and config.control_plane.enabled)
            else "disabled"
        ),
    )
    await asyncio.gather(*tasks)
