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
from sentinel.config.schema import AgentConfig, CorrelationEngineConfig, CortexConfig, DetectorConfig, SentinelConfig, TemporalLayerConfig
from sentinel.domain.models import Alert, AlertSeverity, AlertType, CorrelatedPair, HeartbeatEvent, ModelPhase
from sentinel.domain.ports.detector import DetectorState, IDetector
from sentinel.domain.ports.reporter import IReporter
from sentinel.logging.logger import get_logger

if TYPE_CHECKING:
    from sentinel.adapters.grpc.client import GrpcReporter
    from sentinel.adapters.model_store.disk_store import DiskModelStore
    from sentinel.agent.use_cases.process_message import AgentState
    from sentinel.cortex.use_cases.aggregate import LayerStateVector
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
    from sentinel.domain.models import InputCapture
    inputs = [
        InputCapture(
            name=inp.name,
            body=_json.loads(inp.body) if inp.body else {},
            received_at=datetime.fromtimestamp(inp.received_at_ms / 1000, tz=_tz.utc),
            size_bytes=inp.size_bytes,
        )
        for inp in proto.inputs
    ]
    return CorrelatedPair(
        engine_name=proto.engine_name,
        correlation_id=proto.correlation_id,
        inputs=inputs,
        output_body=_json.loads(proto.output_body) if proto.output_body else {},
        output_received_at=datetime.fromtimestamp(proto.output_received_at_ms / 1000, tz=_tz.utc),
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
        async def _guarded_pairs_loop() -> None:
            try:
                await self._run_pairs_loop()
            except Exception as exc:
                logger.error("agent_pairs_loop_failed", agent=self.agent_config.name, error=str(exc), exc_info=True)
                raise

        await asyncio.gather(
            _guarded_pairs_loop(),
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
        from sentinel.agent.use_cases.train import run_training
        from sentinel.agent.use_cases.champion import run_selection

        det_state = self.state.detectors.get(det_name)
        detector = self.detectors.get(det_name)
        if det_state is None or detector is None:
            return

        buffer_snapshot = list(det_state.training_buffer)
        test_buffer_snapshot = list(det_state.test_buffer)
        denial_buffer_snapshot = list(det_state.denial_buffer)

        try:
            challenger, _ = await run_training(detector, buffer_snapshot, self.agent_config.name)
            new_training_count = det_state.training_count + 1

            result = await run_selection(
                detector=detector,
                challenger=challenger,
                current_champion=det_state.model,
                current_champion_fp_rate=det_state.champion_fp_rate,
                test_buffer=test_buffer_snapshot,
                denial_buffer=denial_buffer_snapshot,
                min_denial_recall=det_state.min_denial_recall,
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

        # Apply result — preserve samples accumulated during training.
        # current_det.training_buffer contains samples that arrived while the fit
        # was running in background (output loop kept consuming). We keep those
        # and discard only the snapshot that was used for training.
        is_new_champion = new_model is not det_state.model
        current_det = self.state.detectors.get(det_name, det_state)
        post_training_samples = current_det.training_buffer[len(buffer_snapshot):]
        current_det.model = new_model
        current_det.champion_fp_rate = new_fp_rate
        current_det.challengers_rejected = new_challengers_rejected
        current_det.training_count = new_training_count
        current_det.training_buffer = post_training_samples
        current_det.training_in_progress = False

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
            threshold = current_det.auto_infer_fp_threshold
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

    async def _handle_one_pair(
        self,
        raw_message,
        training_tasks: dict[str, asyncio.Task],
        prev_buf_counts: dict[str, int],
    ) -> None:
        """Process a single pair message: decode, score/train, ack, spawn training if needed."""
        from sentinel.agent.use_cases.process_message import process_pair
        try:
            pair = self._decode_pair(raw_message.body)
            if pair is None:
                await self.pairs_transport.nack(raw_message.id)
                return

            await process_pair(
                pair=pair,
                agent_config=self.agent_config,
                state=self.state,
                detectors=self.detectors,
                model_store=self.model_store,
                reporter=self.reporter,
            )

            for det_name, det_state in self.state.detectors.items():
                if det_state.training_in_progress:
                    existing = training_tasks.get(det_name)
                    if existing is None or existing.done():
                        task = asyncio.create_task(
                            self._run_background_training(det_name),
                            name=f"train-{self.agent_config.name}-{det_name}",
                        )
                        training_tasks[det_name] = task

                if self.dashboard_state is not None:
                    new_buf = len(det_state.test_buffer)
                    if new_buf != prev_buf_counts.get(det_name, 0):
                        self.dashboard_state.update_detector_buffer_count(
                            self.agent_config.name, det_name, new_buf,
                        )
                        prev_buf_counts[det_name] = new_buf

            await self.pairs_transport.ack(raw_message.id)
        except Exception as exc:
            logger.error(
                "agent_processing_error",
                agent=self.agent_config.name,
                error=str(exc),
            )
            await self.pairs_transport.nack(raw_message.id)

    async def _run_pairs_loop(self) -> None:
        _training_tasks: dict[str, asyncio.Task] = {}
        _sem = asyncio.Semaphore(10)  # matches SQS max batch size
        logger.info("agent_pairs_polling_start", agent=self.agent_config.name, queue=self.agent_config.pairs_queue.resource)

        while True:
            batch = await self.pairs_transport.receive_batch()
            if not batch:
                continue

            logger.info("agent_pairs_batch_received", agent=self.agent_config.name, count=len(batch))

            prev_buf_counts = (
                {name: len(det.test_buffer) for name, det in self.state.detectors.items()}
                if self.dashboard_state is not None
                else {}
            )

            async def _bounded(msg):
                async with _sem:
                    await self._handle_one_pair(msg, _training_tasks, prev_buf_counts)

            await asyncio.gather(*[_bounded(msg) for msg in batch], return_exceptions=True)

    async def change_phase(
        self, new_phase: str, detector_name: str | None = None
    ) -> None:
        """Change phase for a specific detector or all detectors if detector_name is None."""
        phase = ModelPhase(new_phase)
        target = [detector_name] if detector_name else list(self.state.detectors.keys())
        for dname in target:
            det = self.state.detectors.get(dname)
            if det is None:
                logger.warning("phase_transition_detector_not_found", agent=self.agent_config.name, detector=dname)
                continue
            old_phase = det.phase
            det.phase = phase
            logger.info("phase_transition", agent=self.agent_config.name, detector=dname, from_phase=old_phase, to_phase=phase)
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
        clamped = max(0.0, min(1.0, rate))
        target = [detector_name] if detector_name else list(self.state.detectors.keys())
        for dname in target:
            det = self.state.detectors.get(dname)
            if det is not None:
                det.test_sample_rate = clamped

        if self.dashboard_state is not None:
            for dname in target:
                det = self.state.detectors.get(dname)
                if det is not None:
                    self.dashboard_state.update_detector_test_rate(
                        self.agent_config.name, dname, clamped, len(det.test_buffer)
                    )

    async def run_test(self, detector_name: str | None = None) -> dict:
        """Run test buffer evaluation for a specific detector or all detectors."""
        from sentinel.agent.use_cases.run_test import run_detector_test

        results: dict[str, dict] = {}
        target = [detector_name] if detector_name else list(self.state.detectors.keys())
        for dname in target:
            det_state = self.state.detectors.get(dname)
            detector = self.detectors.get(dname)
            if det_state is None or detector is None:
                continue
            result = run_detector_test(detector, det_state)
            det_state.last_test_result = result
            results[dname] = result
            if self.dashboard_state is not None:
                self.dashboard_state.update_detector_test_result(
                    self.agent_config.name, dname, result
                )

        return results

    async def set_auto_infer(
        self, detector_name: str, threshold: float | None
    ) -> None:
        """Enable (threshold=float) or disable (threshold=None) auto-transition to INFERENCE."""
        det = self.state.detectors.get(detector_name)
        if det is not None:
            det.auto_infer_fp_threshold = threshold

    async def reset(self) -> None:
        """Reset all detector states to initial — clears models, buffers, counters."""
        for det_state in self.state.detectors.values():
            det_state.phase = ModelPhase.TRAINING
            det_state.model = None
            det_state.training_buffer = []
            det_state.test_buffer = deque()
            det_state.denial_buffer = deque(
                maxlen=det_state.denial_buffer_size if det_state.denial_buffer_size > 0 else None
            )
            det_state.last_test_result = {}
            det_state.champion_fp_rate = 1.0
            det_state.challengers_rejected = 0
            det_state.training_count = 0
            det_state.training_in_progress = False
        self.state.message_count = 0
        self.state.error_count = 0

        if self.dashboard_state is not None:
            snap = self.dashboard_state.agents.get(self.agent_config.name)
            if snap:
                snap.phase = ModelPhase.TRAINING.value
                snap.total_messages = 0
                snap.total_anomalies = 0
                snap.last_score = 0.0
                snap.champion_fp_rate = 1.0
                snap.challengers_rejected = 0
                snap.test_buffer_count = 0
                snap.last_test_result = {}
                for det in snap.detectors:
                    det.phase = ModelPhase.TRAINING.value
                    det.champion_fp_rate = 1.0
                    det.challengers_rejected = 0
                    det.training_count = 0
                    det.last_score = 0.0
                    det.test_buffer_count = 0
                    det.last_test_result = {}
            self.dashboard_state._push_sse("agent_reset", {"name": self.agent_config.name})

        logger.info("agent_reset", agent=self.agent_config.name)


@dataclass
class CortexRunner:
    cortex_config: CortexConfig
    sentinel_config: SentinelConfig
    dashboard_state: DashboardState | None = None
    layer_vectors: dict[str, LayerStateVector] = field(default_factory=dict)
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
    _last_training_sample_at: datetime | None = field(default=None, repr=False)

    def _all_inputs_in_inference(self) -> bool:
        for name in self.cortex_config.inputs:
            lv = self.layer_vectors.get(name)
            if lv is None or lv.model_phase != ModelPhase.INFERENCE:
                return False
        return True

    async def _accumulate_training_sample(self):
        from sentinel.cortex.use_cases.aggregate import build_layer_feature_matrix

        if self.phase == ModelPhase.INFERENCE:
            matrix = build_layer_feature_matrix(self.layer_vectors)
            if matrix.shape[0] == 0:
                return None
            return matrix.mean(axis=0)

        if not self._all_inputs_in_inference():
            return None

        now = datetime.now(tz=timezone.utc)
        interval = self.cortex_config.training_sample_interval_s
        if self._last_training_sample_at is not None:
            elapsed = (now - self._last_training_sample_at).total_seconds()
            if elapsed < interval:
                return None

        matrix = build_layer_feature_matrix(self.layer_vectors)
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

    async def reset(self) -> None:
        """Reset cortex to initial state — clears model, buffers, layer vectors."""
        self.autoencoder_model = None
        self.training_buffer = []
        self.test_buffer = deque()
        self.last_test_result = {}
        self.baseline_error = 0.01
        self.layer_vectors = {}
        self.causal_window = deque(maxlen=100)
        self.phase = ModelPhase.TRAINING
        self._strategy = None
        self._adaptation_ctx = None
        self._last_training_sample_at = None

        if self.dashboard_state is not None:
            snap = self.dashboard_state.cortex.get(self.cortex_config.name)
            if snap:
                snap.phase = "TRAINING"
                snap.total_messages = 0
                snap.total_alerts = 0
                snap.training_samples = 0
                snap.last_alert_type = ""
                snap.last_alert_severity = ""
                snap.last_alert_at = ""
                snap.test_sample_rate = 0.0
                snap.test_buffer_count = 0
                snap.last_test_result = {}
            self.dashboard_state._push_sse("cortex_reset", {"name": self.cortex_config.name})

        logger.info("cortex_reset", cortex=self.cortex_config.name)

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
        from sentinel.cortex.use_cases.aggregate import (
            check_layer_silence,
            update_layer_state,
        )
        from sentinel.cortex.use_cases.causal_chain import add_temporal_signal
        from sentinel.domain.models import TemporalAnomalySignal, TemporalHeartbeatEvent

        if isinstance(proto_event, sentinel_pb2.TemporalHeartbeatProto):
            event = TemporalHeartbeatEvent(
                layer_name=proto_event.layer_name,
                timestamp=datetime.fromtimestamp(
                    proto_event.timestamp_ms / 1000, tz=timezone.utc
                ),
                window_start=datetime.fromtimestamp(
                    proto_event.window_start_ms / 1000, tz=timezone.utc
                ),
                window_end=datetime.fromtimestamp(
                    proto_event.window_end_ms / 1000, tz=timezone.utc
                ),
                agents_monitored=list(proto_event.agents_monitored),
                total_events_in_window=proto_event.total_events_in_window,
                anomaly_rate=proto_event.anomaly_rate,
                avg_score=proto_event.avg_score,
                avg_latency_ms=proto_event.avg_latency_ms,
                bucket_confidence=proto_event.bucket_confidence,
                model_phase=ModelPhase(proto_event.model_phase) if proto_event.model_phase else ModelPhase.WARMUP,
            )
            self.layer_vectors = update_layer_state(self.layer_vectors, event)
            now = datetime.now(tz=timezone.utc)
            for layer_name in self.cortex_config.inputs:
                if check_layer_silence(
                    self.layer_vectors, layer_name,
                    self.cortex_config.silence_threshold_s, now,
                ):
                    logger.warning("layer_silence_detected", layer=layer_name)
                    if self.dashboard_state:
                        self.dashboard_state.mark_silent(layer_name)

        elif isinstance(proto_event, sentinel_pb2.TemporalAnomalySignalProto):
            signal = TemporalAnomalySignal(
                layer_name=proto_event.layer_name,
                window_start=datetime.fromtimestamp(
                    proto_event.window_start_ms / 1000, tz=timezone.utc
                ),
                window_end=datetime.fromtimestamp(
                    proto_event.window_end_ms / 1000, tz=timezone.utc
                ),
                hour_of_week=proto_event.hour_of_week,
                anomaly_rate=proto_event.anomaly_rate,
                anomaly_rate_z=proto_event.anomaly_rate_z,
                volume=proto_event.volume,
                volume_z=proto_event.volume_z,
                avg_score_z=proto_event.avg_score_z,
                avg_latency_z=proto_event.avg_latency_z,
                is_rate_anomaly=proto_event.flags.is_rate_anomaly,
                is_volume_anomaly=proto_event.flags.is_volume_anomaly,
                is_score_anomaly=proto_event.flags.is_score_anomaly,
                is_latency_anomaly=proto_event.flags.is_latency_anomaly,
                contributing_agents=list(proto_event.contributing_agents),
                bucket_confidence=proto_event.bucket_confidence,
            )
            self.layer_vectors = update_layer_state(self.layer_vectors, signal)
            if self.phase == ModelPhase.INFERENCE:
                add_temporal_signal(self.causal_window, signal)

        current_sample = await self._accumulate_training_sample()
        if self.phase == ModelPhase.INFERENCE:
            await self._run_inference(sample=current_sample)

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
            layer_vectors=self.layer_vectors,
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
        await start_server(server)


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


@dataclass
class TemporalLayerRunner:
    config: TemporalLayerConfig
    sentinel_config: SentinelConfig
    dashboard_state: DashboardState | None = None
    phase: ModelPhase = field(default=ModelPhase.WARMUP)
    _accumulator: object = field(default=None, repr=False)
    _agent_phases: dict[str, ModelPhase] = field(default_factory=dict, repr=False)
    _bucket_store: object = field(default=None, repr=False)
    _reporters: list = field(default_factory=list, repr=False)
    _closing: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        from sentinel.temporal.use_cases.window_accumulator import WindowAccumulator
        self._accumulator = WindowAccumulator(
            layer_name=self.config.name,
            duration_seconds=self.config.bucket.duration_seconds,
            window_start=datetime.now(tz=timezone.utc),
        )

    async def _push_dashboard_update(self, bucket_idx: int, all_buckets: list, extra_anomaly: bool = False) -> None:
        if self.dashboard_state is None:
            return
        snap = self.dashboard_state.temporal_layers.get(self.config.name)
        total_anomalies = (snap.total_anomalies if snap else 0) + (1 if extra_anomaly else 0)
        total_windows = (snap.total_windows if snap else 0) + 1
        self.dashboard_state.update_temporal_layer({
            "name": self.config.name,
            "phase": self.phase.value,
            "current_bucket": bucket_idx,
            "bucket_observations": [b.n_observations for b in all_buckets],
            "total_windows": total_windows,
            "total_anomalies": total_anomalies,
        })

    async def handle_event(self, proto_event: object) -> None:
        from sentinel.adapters.grpc.generated import sentinel_pb2
        from sentinel.domain.models import ModelPhase as _MP
        from sentinel.temporal.use_cases.window_accumulator import add_event, should_close

        if isinstance(proto_event, sentinel_pb2.ProcessedEventProto):
            agent_phase = self._agent_phases.get(
                proto_event.agent_name, _MP.TRAINING
            )
            from sentinel.domain.models import ProcessedEvent, AnomalyType
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
                anomaly_type=AnomalyType(proto_event.anomaly_type) if proto_event.anomaly_type else None,
                payload_schema_hash=proto_event.payload_schema_hash,
                input_size_bytes=proto_event.input_size_bytes,
                output_size_bytes=proto_event.output_size_bytes,
            )
            self._accumulator = add_event(self._accumulator, event, agent_phase)
            now = datetime.now(tz=timezone.utc)
            if should_close(self._accumulator, now) and not self._closing:
                await self._close_and_process()

        elif isinstance(proto_event, sentinel_pb2.HeartbeatProto):
            if proto_event.model_phase:
                self._agent_phases[proto_event.agent_name] = _MP(proto_event.model_phase)

    async def _close_and_process(self) -> None:
        from sentinel.temporal.use_cases.anomaly_decider import decide
        from sentinel.temporal.use_cases.seasonal_normalizer import (
            bucket_index_for,
            compute_z_scores,
            hour_of_week,
            update_bucket,
        )
        from sentinel.temporal.use_cases.window_accumulator import close_window

        self._closing = True
        try:
            now = datetime.now(tz=timezone.utc)
            bucket_idx = bucket_index_for(
                now,
                self.config.bucket.count,
                self.config.bucket.duration_seconds,
            )

            snapshot, self._accumulator = close_window(self._accumulator, bucket_idx)

            # Skip bucket update for empty windows
            if snapshot.total_events == 0:
                return

            bucket = await self._bucket_store.get_bucket(bucket_idx)
            z_scores = compute_z_scores(snapshot, bucket)
            all_buckets = await self._bucket_store.get_all_buckets()
            confidence = self._bucket_store.bucket_confidence(all_buckets)
            decision = decide(z_scores, self.config, confidence)

            updated_bucket = update_bucket(bucket, snapshot, self.config.bucket.ema_alpha)
            await self._bucket_store.save_bucket(bucket_idx, updated_bucket)

            # Check WARMUP → INFERENCE transition
            refreshed = await self._bucket_store.get_all_buckets()
            if self.phase == ModelPhase.WARMUP and all(
                b.n_observations >= self.config.bucket.min_observations
                for b in refreshed
            ):
                self.phase = ModelPhase.INFERENCE
                logger.info(
                    "temporal_layer_inference",
                    layer=self.config.name,
                )

            await self._push_dashboard_update(
                bucket_idx, refreshed, extra_anomaly=decision.is_anomalous
            )

            if decision.is_anomalous and self.phase == ModelPhase.INFERENCE:
                from sentinel.domain.models import TemporalAnomalySignal
                from sentinel.adapters.grpc.client import GrpcReporter
                signal = TemporalAnomalySignal(
                    layer_name=self.config.name,
                    window_start=snapshot.window_start,
                    window_end=snapshot.window_end,
                    hour_of_week=hour_of_week(snapshot.window_end),
                    anomaly_rate=snapshot.anomaly_rate,
                    anomaly_rate_z=z_scores.get("anomaly_rate", 0.0),
                    volume=snapshot.total_events,
                    volume_z=z_scores.get("volume", 0.0),
                    avg_score_z=z_scores.get("avg_score", 0.0),
                    avg_latency_z=z_scores.get("avg_latency", 0.0),
                    is_rate_anomaly=decision.is_rate_anomaly,
                    is_volume_anomaly=decision.is_volume_anomaly,
                    is_score_anomaly=decision.is_score_anomaly,
                    is_latency_anomaly=decision.is_latency_anomaly,
                    contributing_agents=snapshot.contributing_agents,
                    bucket_confidence=confidence,
                )
                for reporter in self._reporters:
                    if isinstance(reporter, GrpcReporter):
                        try:
                            await reporter.publish_temporal_anomaly_signal(signal)
                        except Exception as exc:
                            logger.warning(
                                "temporal_anomaly_signal_failed",
                                layer=self.config.name,
                                error=str(exc),
                            )
        finally:
            self._closing = False

    async def _heartbeat_loop(self) -> None:
        from sentinel.adapters.grpc.client import GrpcReporter
        from sentinel.domain.models import TemporalHeartbeatEvent

        from sentinel.temporal.use_cases.seasonal_normalizer import bucket_index_for

        interval = self.config.heartbeat_interval_s
        while True:
            await asyncio.sleep(interval)
            now = datetime.now(tz=timezone.utc)
            all_buckets = await self._bucket_store.get_all_buckets() if self._bucket_store else []
            confidence = self._bucket_store.bucket_confidence(all_buckets) if self._bucket_store else 0

            if self.dashboard_state is not None and self._bucket_store is not None:
                current_idx = bucket_index_for(now, self.config.bucket.count, self.config.bucket.duration_seconds)
                snap = self.dashboard_state.temporal_layers.get(self.config.name)
                self.dashboard_state.update_temporal_layer({
                    "name": self.config.name,
                    "phase": self.phase.value,
                    "current_bucket": current_idx,
                    "bucket_observations": [b.n_observations for b in all_buckets],
                    "total_windows": snap.total_windows if snap else 0,
                    "total_anomalies": snap.total_anomalies if snap else 0,
                })

            acc = self._accumulator
            total = len(acc.events) if hasattr(acc, "events") else 0
            anomaly_count = sum(1 for e in acc.events if e.is_anomaly) if hasattr(acc, "events") else 0
            anomaly_rate = anomaly_count / total if total > 0 else 0.0
            avg_score = sum(e.anomaly_score for e in acc.events) / total if total > 0 else 0.0
            avg_latency = sum(e.processing_latency_ms for e in acc.events) / total if total > 0 else 0.0

            heartbeat = TemporalHeartbeatEvent(
                layer_name=self.config.name,
                timestamp=now,
                window_start=acc.window_start,
                window_end=now,
                agents_monitored=sorted(acc.active_agents) if hasattr(acc, "active_agents") else [],
                total_events_in_window=total,
                anomaly_rate=anomaly_rate,
                avg_score=avg_score,
                avg_latency_ms=avg_latency,
                bucket_confidence=confidence,
                model_phase=self.phase,
            )

            for reporter in self._reporters:
                if isinstance(reporter, GrpcReporter):
                    try:
                        await reporter.publish_temporal_heartbeat(heartbeat)
                    except Exception as exc:
                        logger.warning(
                            "temporal_heartbeat_failed",
                            layer=self.config.name,
                            error=str(exc),
                        )

    async def _window_clock_loop(self) -> None:
        from sentinel.temporal.use_cases.window_accumulator import should_close
        while True:
            await asyncio.sleep(1)
            if not self._closing and should_close(self._accumulator, datetime.now(tz=timezone.utc)):
                await self._close_and_process()

    async def run(self) -> None:
        from sentinel.adapters.grpc.server import create_server, start_server
        from sentinel.temporal.use_cases.bucket_store import BucketStore

        redis = RedisCorrelationStore(
            host=self.sentinel_config.redis.host,
            port=self.sentinel_config.redis.port,
            db=self.sentinel_config.redis.db,
            password=self.sentinel_config.redis.password,
        )
        redis_client = redis._get_client()
        self._bucket_store = BucketStore(
            redis_client=redis_client,
            layer_name=self.config.name,
            bucket_count=self.config.bucket.count,
        )

        # Resume INFERENCE if all buckets are already warm from a previous run
        all_buckets = await self._bucket_store.get_all_buckets()
        if all(
            b.n_observations >= self.config.bucket.min_observations
            for b in all_buckets
        ):
            self.phase = ModelPhase.INFERENCE
            logger.info(
                "temporal_layer_resumed_inference",
                layer=self.config.name,
            )

        if self.dashboard_state is not None:
            port_to_name = {cx.grpc_port: cx.name for cx in (self.sentinel_config.cortex or [])}
            cortex_targets = [
                port_to_name.get(ref.port, f"port:{ref.port}")
                for ref in self.config.cortex
            ]
            self.dashboard_state.register_temporal_layer(
                name=self.config.name,
                inputs=list(self.config.inputs),
                cortex_targets=cortex_targets,
                bucket_count=self.config.bucket.count,
            )
            self.dashboard_state.update_temporal_layer({
                "name": self.config.name,
                "phase": self.phase.value,
                "current_bucket": 0,
                "bucket_observations": [b.n_observations for b in all_buckets],
                "total_windows": 0,
                "total_anomalies": 0,
            })

        server = create_server(
            host="0.0.0.0",
            port=self.config.grpc_port,
            event_handler=self.handle_event,
        )
        await asyncio.gather(
            start_server(server),
            self._heartbeat_loop(),
            self._window_clock_loop(),
        )


def build_temporal_layer(
    tl_config: TemporalLayerConfig,
    sentinel_config: SentinelConfig,
    dashboard_state: DashboardState | None = None,
) -> TemporalLayerRunner:
    from sentinel.adapters.grpc.client import GrpcReporter

    reporters = [
        GrpcReporter(host=ref.host, port=ref.port)
        for ref in tl_config.cortex
    ]

    return TemporalLayerRunner(
        config=tl_config,
        sentinel_config=sentinel_config,
        dashboard_state=dashboard_state,
        _reporters=reporters,
    )


def build_correlation_engine(
    engine_config: CorrelationEngineConfig,
    sentinel_config: SentinelConfig,
) -> "CorrelationEngineRunner":
    """Wire up all dependencies for a CorrelationEngineRunner."""
    from sentinel.runtime.correlation_engine_runner import CorrelationEngineRunner
    from sentinel.adapters.queue.sqs_pair_publisher import SqsPairPublisher

    endpoint = sentinel_config.aws.endpoint_url
    region = sentinel_config.aws.region

    input_transports = [
        SqsSnsTransport(
            endpoint_url=endpoint,
            input_queue_url=inp_cfg.transport.resource,
            output_topic_arn="",
            region=region,
            wait_time_seconds=1,
        )
        for inp_cfg in engine_config.inputs
    ]
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
    return CorrelationEngineRunner(
        engine_config=engine_config,
        input_transports=input_transports,
        correlation_store=correlation_store,
        pair_publishers=pair_publishers,
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
            denial_buffer=deque(maxlen=dc.denial_buffer_size if dc.denial_buffer_size > 0 else None),
            denial_buffer_size=dc.denial_buffer_size,
            min_denial_recall=dc.min_denial_recall,
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
        dashboard_state.register_reset_callbacks(
            agent_config.name, runner.reset
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

    runner = CortexRunner(
        cortex_config=cortex_config,
        sentinel_config=sentinel_config,
        dashboard_state=dashboard_state,
        _parent_reporters=parent_reporters,
        _model_store=model_store,
    )

    if dashboard_state is not None:
        dashboard_state.register_cortex_reset_callback(cortex_config.name, runner.reset)

    return runner


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
        topology_manager = TopologyManager(config, ds, engines_enabled=run_engines)
        await topology_manager.restore_from_store(topology_store)

        if not cp_enabled:
            for agent_config in config.agents:
                runner = build_agent(agent_config, config, ds)
                tasks.append(asyncio.create_task(runner.run(), name=f"agent-{agent_config.name}"))

            for tl_config in config.temporal_layers:
                runner = build_temporal_layer(tl_config, config, ds)
                tasks.append(asyncio.create_task(runner.run(), name=f"temporal-{tl_config.name}"))

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
        temporal_layers=len(config.temporal_layers) if run_agents else 0,
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
