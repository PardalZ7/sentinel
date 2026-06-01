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

"""Shared in-memory state for the dashboard.

Design principles:
- Agents write via fire-and-forget helpers (never block the processing loop).
- Dashboard server reads without locks (asyncio is single-threaded; reads of
  plain Python dicts are safe).
- The SSE event queue is bounded; if the browser is slow the oldest events are
  dropped silently — the processing loop is never back-pressured.
"""

import asyncio
import json
from collections import deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from sentinel.domain.models import Alert, HeartbeatEvent, ModelPhase, ProcessedEvent

_SSE_QUEUE_MAXSIZE = 256
_MAX_RECENT_EVENTS = 100
_MAX_RECENT_ALERTS = 50
_MAX_PAYLOAD_STORE = 200


@dataclass
class DetectorSnapshot:
    name: str
    algorithm: str
    phase: str = ModelPhase.COLD.value
    champion_fp_rate: float = 1.0
    challengers_rejected: int = 0
    training_count: int = 0
    last_score: float = 0.0
    test_buffer_count: int = 0
    test_sample_rate: float = 0.0
    last_test_result: dict = field(default_factory=dict)
    auto_infer_fp_threshold: float | None = None


@dataclass
class AgentSnapshot:
    name: str
    phase: str = ModelPhase.COLD.value
    total_messages: int = 0
    total_anomalies: int = 0
    last_score: float = 0.0
    last_seen: str = ""
    message_rate: float = 0.0
    error_rate: float = 0.0
    is_silent: bool = False
    test_sample_rate: float = 0.0
    test_buffer_count: int = 0
    last_test_result: dict = field(default_factory=dict)
    champion_fp_rate: float = 1.0
    challengers_rejected: int = 0
    detectors: list = field(default_factory=list)  # list[DetectorSnapshot]


@dataclass
class CortexSnapshot:
    name: str
    inputs: list[str] = field(default_factory=list)
    phase: str = "TRAINING"
    total_messages: int = 0
    total_alerts: int = 0
    training_samples: int = 0
    last_alert_type: str = ""
    last_alert_severity: str = ""
    last_alert_at: str = ""
    test_sample_rate: float = 0.0
    test_buffer_count: int = 0
    last_test_result: dict = field(default_factory=dict)


@dataclass
class EventRecord:
    ts: str
    kind: str          # "event" | "heartbeat" | "alert"
    agent: str
    is_anomaly: bool
    score: float
    anomaly_type: str
    phase: str
    latency_ms: float
    extra: dict = field(default_factory=dict)
    correlation_id: str = ""
    input_received_at: str = ""
    input_size_bytes: int = 0
    output_size_bytes: int = 0
    payload_schema_hash: str = ""


@dataclass
class AlertRecord:
    ts: str
    source: str
    severity: str
    alert_type: str
    affected: list[str]
    description: str
    alert_id: str = ""  # timestamp-based ID for payload lookup


class DashboardState:
    """Central observation state.  Mutated only from the asyncio event loop."""

    def __init__(self) -> None:
        self.agents: dict[str, AgentSnapshot] = {}
        self.cortex: dict[str, CortexSnapshot] = {}
        self.recent_events: deque[EventRecord] = deque(maxlen=_MAX_RECENT_EVENTS)
        self.recent_errors: deque[EventRecord] = deque(maxlen=_MAX_RECENT_EVENTS)
        self.recent_alerts: deque[AlertRecord] = deque(maxlen=_MAX_RECENT_ALERTS)
        self._sse_queue: asyncio.Queue = asyncio.Queue(maxsize=_SSE_QUEUE_MAXSIZE)
        self._phase_change_callbacks: dict[str, Any] = {}       # agent_name → cb(phase, detector_name|None)
        self._agent_set_rate_callbacks: dict[str, Any] = {}     # agent_name → cb(rate, detector_name|None)
        self._agent_run_test_callbacks: dict[str, Any] = {}     # agent_name → cb(detector_name|None) → dict
        self._cortex_change_phase_callbacks: dict[str, Any] = {}  # cortex_name → cb(phase)
        self._cortex_set_rate_callbacks: dict[str, Any] = {}    # cortex_name → cb(rate)
        self._cortex_run_test_callbacks: dict[str, Any] = {}    # cortex_name → cb() → dict
        self._auto_infer_callbacks: dict[str, Any] = {}          # agent_name → cb(detector_name, threshold|None)
        self._reset_agent_callbacks: dict[str, Any] = {}         # agent_name → async cb()
        self._reset_cortex_callbacks: dict[str, Any] = {}        # cortex_name → async cb()
        self.config_data: dict = {}
        self._payload_store: dict[str, dict] = {}
        self._payload_order: deque[str] = deque(maxlen=_MAX_PAYLOAD_STORE)

    def set_config(self, config_data: dict) -> None:
        """Store the parsed sentinel config so the dashboard can serve it."""
        self.config_data = config_data

    # ------------------------------------------------------------------
    # Writers (called by DashboardReporter, never awaited from hot path)
    # ------------------------------------------------------------------

    def record_event(self, event: ProcessedEvent) -> None:
        snap = self.agents.setdefault(
            event.agent_name, AgentSnapshot(name=event.agent_name)
        )
        snap.total_messages += 1
        snap.last_score = round(event.anomaly_score, 4)
        snap.last_seen = event.output_sent_at.isoformat()
        snap.is_silent = False
        if event.is_anomaly:
            snap.total_anomalies += 1

        rec = EventRecord(
            ts=event.output_sent_at.isoformat(),
            kind="event",
            agent=event.agent_name,
            is_anomaly=event.is_anomaly,
            score=round(event.anomaly_score, 4),
            anomaly_type=event.anomaly_type.value if event.anomaly_type else "",
            phase=snap.phase,
            latency_ms=round(event.processing_latency_ms, 2),
            correlation_id=event.correlation_id,
            input_received_at=event.input_received_at.isoformat(),
            input_size_bytes=event.input_size_bytes,
            output_size_bytes=event.output_size_bytes,
            payload_schema_hash=event.payload_schema_hash,
        )
        self.recent_events.append(rec)
        if event.is_anomaly:
            self.recent_errors.append(rec)
        self._push_sse("event", asdict(rec))

        if event.is_anomaly and (event.input_body is not None or event.output_body is not None):
            self._store_payload(event.correlation_id, event.input_body, event.output_body)

    def record_heartbeat(self, hb: HeartbeatEvent) -> None:
        snap = self.agents.setdefault(hb.agent_name, AgentSnapshot(name=hb.agent_name))
        snap.phase = hb.model_phase.value
        snap.message_rate = round(hb.message_rate_per_sec, 3)
        snap.error_rate = round(hb.error_rate_last_window, 3)
        snap.last_seen = hb.timestamp.isoformat()
        snap.is_silent = False

        rec = EventRecord(
            ts=hb.timestamp.isoformat(),
            kind="heartbeat",
            agent=hb.agent_name,
            is_anomaly=False,
            score=0.0,
            anomaly_type="",
            phase=hb.model_phase.value,
            latency_ms=0.0,
            extra={"rate": snap.message_rate, "err_rate": snap.error_rate},
        )
        self.recent_events.append(rec)
        self._push_sse("heartbeat", asdict(rec))

    def record_alert(self, alert: Alert) -> None:
        snap = self.cortex.setdefault(alert.source, CortexSnapshot(name=alert.source))
        snap.total_alerts += 1
        snap.last_alert_type = alert.alert_type.value
        snap.last_alert_severity = alert.severity.value
        snap.last_alert_at = alert.timestamp.isoformat()

        alert_id = f"{alert.source}:{alert.timestamp.isoformat()}"
        rec = AlertRecord(
            ts=alert.timestamp.isoformat(),
            source=alert.source,
            severity=alert.severity.value,
            alert_type=alert.alert_type.value,
            affected=alert.affected_agents,
            description=alert.description,
            alert_id=alert_id,
        )
        self.recent_alerts.append(rec)
        self._push_sse("alert", asdict(rec))

        if alert.diagnosis:
            self._store_payload(
                alert_id,
                {"diagnosis": alert.diagnosis, "affected_agents": alert.affected_agents},
                None,
            )

    def mark_silent(self, agent_name: str) -> None:
        if agent_name in self.agents:
            self.agents[agent_name].is_silent = True
            self._push_sse("silence", {"agent": agent_name})

    def register_cortex(self, name: str, inputs: list[str], phase: str = "TRAINING") -> None:
        self.cortex.setdefault(name, CortexSnapshot(name=name, inputs=inputs))
        self.cortex[name].inputs = inputs
        self.cortex[name].phase = phase

    def record_cortex_event(self, name: str) -> None:
        snap = self.cortex.get(name)
        if snap:
            snap.total_messages += 1
            self._push_sse("cortex_update", {"name": name, "total_messages": snap.total_messages, "training_samples": snap.training_samples, "phase": snap.phase})

    def record_cortex_training_sample(self, name: str) -> None:
        snap = self.cortex.get(name)
        if snap:
            snap.training_samples += 1
            self._push_sse("cortex_update", {"name": name, "total_messages": snap.total_messages, "training_samples": snap.training_samples, "phase": snap.phase})

    def update_cortex_phase(self, name: str, phase: str) -> None:
        snap = self.cortex.get(name)
        if snap:
            snap.phase = phase
            self._push_sse("cortex_update", {"name": name, "total_messages": snap.total_messages, "training_samples": snap.training_samples, "phase": phase})

    def unregister_agent(self, agent_name: str) -> None:
        """Remove an agent from the dashboard (called when a dynamic topology is removed)."""
        self.agents.pop(agent_name, None)
        self._phase_change_callbacks.pop(agent_name, None)
        self._agent_set_rate_callbacks.pop(agent_name, None)
        self._agent_run_test_callbacks.pop(agent_name, None)
        self._auto_infer_callbacks.pop(agent_name, None)
        self._push_sse("agent_removed", {"name": agent_name})

    def unregister_cortex(self, cortex_name: str) -> None:
        """Remove a cortex from the dashboard (called when a dynamic topology is removed)."""
        self.cortex.pop(cortex_name, None)
        self._cortex_change_phase_callbacks.pop(cortex_name, None)
        self._cortex_set_rate_callbacks.pop(cortex_name, None)
        self._cortex_run_test_callbacks.pop(cortex_name, None)
        self._push_sse("cortex_removed", {"name": cortex_name})

    def register_agent(self, agent_name: str, initial_detector_states: dict) -> None:
        """Pre-register an agent with its detectors so it appears on the dashboard from startup.

        Args:
            initial_detector_states: dict[detector_name, DetectorState] from launcher.
        """
        snap = self.agents.setdefault(agent_name, AgentSnapshot(name=agent_name))
        det_snaps = []
        for det_name, det_state in initial_detector_states.items():
            det_snaps.append(DetectorSnapshot(
                name=det_name,
                algorithm=det_state.algorithm,
                phase=det_state.phase.value,
                test_sample_rate=det_state.test_sample_rate,
                auto_infer_fp_threshold=det_state.auto_infer_fp_threshold,
            ))
        snap.detectors = det_snaps
        if det_snaps:
            snap.phase = det_snaps[0].phase

    def register_phase_callback(self, agent_name: str, callback) -> None:
        """Register callback: cb(new_phase: str, detector_name: str | None)."""
        self._phase_change_callbacks[agent_name] = callback

    def register_agent_test_callbacks(self, agent_name: str, set_rate_cb, run_test_cb) -> None:
        """Register callbacks: set_rate_cb(rate, detector_name|None), run_test_cb(detector_name|None)."""
        self._agent_set_rate_callbacks[agent_name] = set_rate_cb
        self._agent_run_test_callbacks[agent_name] = run_test_cb

    def register_auto_infer_callback(self, agent_name: str, callback) -> None:
        """Register callback: cb(detector_name: str, threshold: float | None)."""
        self._auto_infer_callbacks[agent_name] = callback

    def register_reset_callbacks(self, agent_name: str, reset_cb) -> None:
        """Register async reset callback for an agent: async cb()."""
        self._reset_agent_callbacks[agent_name] = reset_cb

    def register_cortex_reset_callback(self, cortex_name: str, reset_cb) -> None:
        """Register async reset callback for a cortex: async cb()."""
        self._reset_cortex_callbacks[cortex_name] = reset_cb

    async def reset_topology(self) -> None:
        """Reset all agents and cortex to initial state in-memory (buffers, models, counters).

        Does NOT clear disk models or Redis — those are handled by the server endpoint
        which also calls this method.
        """
        for cb in self._reset_agent_callbacks.values():
            await cb()
        for cb in self._reset_cortex_callbacks.values():
            await cb()

        self.recent_events.clear()
        self.recent_errors.clear()
        self.recent_alerts.clear()
        self._payload_store.clear()
        self._payload_order.clear()

        self._push_sse("topology_reset", {})

    async def set_detector_auto_infer(
        self, agent_name: str, detector_name: str, threshold: float | None
    ) -> bool:
        cb = self._auto_infer_callbacks.get(agent_name)
        if cb is None:
            return False
        snap = self.agents.get(agent_name)
        if snap:
            for det in snap.detectors:
                if det.name == detector_name:
                    det.auto_infer_fp_threshold = threshold
                    break
        await cb(detector_name, threshold)
        self._push_sse("detector_update", {
            "agent": agent_name,
            "detector": detector_name,
            "auto_infer_fp_threshold": threshold,
        })
        return True

    def register_cortex_callbacks(self, cortex_name: str, change_phase_cb, set_rate_cb, run_test_cb) -> None:
        self._cortex_change_phase_callbacks[cortex_name] = change_phase_cb
        self._cortex_set_rate_callbacks[cortex_name] = set_rate_cb
        self._cortex_run_test_callbacks[cortex_name] = run_test_cb

    async def change_agent_phase(
        self, agent_name: str, new_phase: str, detector_name: str | None = None
    ) -> bool:
        cb = self._phase_change_callbacks.get(agent_name)
        if cb is None:
            return False
        # Update snapshot and push SSE before calling the callback so the UI
        # reflects the change even if the callback raises.
        snap = self.agents.get(agent_name)
        if snap:
            if detector_name:
                for det in snap.detectors:
                    if det.name == detector_name:
                        det.phase = new_phase
                # Recompute aggregate phase: INFERENCE only when ALL detectors are in INFERENCE
                all_phases = [det.phase for det in snap.detectors]
                snap.phase = new_phase if all(p == new_phase for p in all_phases) else snap.phase
            else:
                snap.phase = new_phase
                for det in snap.detectors:
                    det.phase = new_phase
        self._push_sse("phase_change", {
            "agent": agent_name, "phase": new_phase, "detector": detector_name
        })
        await cb(new_phase, detector_name)
        return True

    async def set_agent_test_rate(
        self, agent_name: str, rate: float, detector_name: str | None = None
    ) -> bool:
        cb = self._agent_set_rate_callbacks.get(agent_name)
        if cb is None:
            return False
        await cb(rate, detector_name)
        return True

    async def run_agent_test(
        self, agent_name: str, detector_name: str | None = None
    ) -> dict | None:
        cb = self._agent_run_test_callbacks.get(agent_name)
        if cb is None:
            return None
        return await cb(detector_name)

    async def change_cortex_phase(self, cortex_name: str, new_phase: str) -> bool:
        cb = self._cortex_change_phase_callbacks.get(cortex_name)
        if cb is None:
            return False
        await cb(new_phase)
        return True

    async def set_cortex_test_rate(self, cortex_name: str, rate: float) -> bool:
        cb = self._cortex_set_rate_callbacks.get(cortex_name)
        if cb is None:
            return False
        cb(rate)  # sync method on CortexRunner
        return True

    def run_cortex_test(self, cortex_name: str) -> dict | None:
        cb = self._cortex_run_test_callbacks.get(cortex_name)
        if cb is None:
            return None
        return cb()  # sync method on CortexRunner

    def update_agent_champion(self, agent_name: str, fp_rate: float, challengers_rejected: int) -> None:
        snap = self.agents.get(agent_name)
        if snap:
            snap.champion_fp_rate = fp_rate
            snap.challengers_rejected = challengers_rejected
            self._push_sse("agent_champion_update", {
                "name": agent_name,
                "champion_fp_rate": fp_rate,
                "challengers_rejected": challengers_rejected,
            })

    def update_agent_test_rate(self, agent_name: str, rate: float, buffer_count: int) -> None:
        snap = self.agents.get(agent_name)
        if snap:
            snap.test_sample_rate = rate
            snap.test_buffer_count = buffer_count
            self._push_sse("agent_test_update", {
                "name": agent_name,
                "test_sample_rate": rate,
                "test_buffer_count": buffer_count,
                "last_test_result": snap.last_test_result,
            })

    def update_agent_test_result(self, agent_name: str, result: dict) -> None:
        snap = self.agents.get(agent_name)
        if snap:
            snap.last_test_result = result
            self._push_sse("agent_test_update", {
                "name": agent_name,
                "test_sample_rate": snap.test_sample_rate,
                "test_buffer_count": snap.test_buffer_count,
                "last_test_result": result,
            })

    def update_detector_champion(
        self, agent_name: str, detector_name: str, fp_rate: float, challengers_rejected: int,
        training_count: int = 0,
    ) -> None:
        snap = self.agents.get(agent_name)
        if not snap:
            return
        for det in snap.detectors:
            if det.name == detector_name:
                det.champion_fp_rate = fp_rate
                det.challengers_rejected = challengers_rejected
                det.training_count = training_count
                break
        self._push_sse("detector_update", {
            "agent": agent_name,
            "detector": detector_name,
            "champion_fp_rate": fp_rate,
            "challengers_rejected": challengers_rejected,
            "training_count": training_count,
        })

    def update_detector_test_rate(
        self, agent_name: str, detector_name: str, rate: float, buffer_count: int
    ) -> None:
        snap = self.agents.get(agent_name)
        if not snap:
            return
        for det in snap.detectors:
            if det.name == detector_name:
                det.test_sample_rate = rate
                det.test_buffer_count = buffer_count
                break
        self._push_sse("detector_update", {
            "agent": agent_name,
            "detector": detector_name,
            "test_sample_rate": rate,
            "test_buffer_count": buffer_count,
        })

    def update_detector_training_count(
        self, agent_name: str, detector_name: str, training_count: int,
        challengers_rejected: int | None = None,
    ) -> None:
        snap = self.agents.get(agent_name)
        if not snap:
            return
        for det in snap.detectors:
            if det.name == detector_name:
                det.training_count = training_count
                if challengers_rejected is not None:
                    det.challengers_rejected = challengers_rejected
                break
        event: dict = {
            "agent": agent_name,
            "detector": detector_name,
            "training_count": training_count,
        }
        if challengers_rejected is not None:
            event["challengers_rejected"] = challengers_rejected
        self._push_sse("detector_update", event)

    def update_detector_buffer_count(
        self, agent_name: str, detector_name: str, buffer_count: int
    ) -> None:
        snap = self.agents.get(agent_name)
        if not snap:
            return
        for det in snap.detectors:
            if det.name == detector_name:
                det.test_buffer_count = buffer_count
                break
        self._push_sse("detector_update", {
            "agent": agent_name,
            "detector": detector_name,
            "test_buffer_count": buffer_count,
        })

    def update_detector_test_result(
        self, agent_name: str, detector_name: str, result: dict
    ) -> None:
        snap = self.agents.get(agent_name)
        if not snap:
            return
        for det in snap.detectors:
            if det.name == detector_name:
                det.last_test_result = result
                break
        self._push_sse("detector_update", {
            "agent": agent_name,
            "detector": detector_name,
            "last_test_result": result,
        })

    def update_detector_phase(
        self, agent_name: str, detector_name: str, new_phase: str
    ) -> None:
        snap = self.agents.get(agent_name)
        if not snap:
            return
        for det in snap.detectors:
            if det.name == detector_name:
                det.phase = new_phase
                break
        all_phases = [det.phase for det in snap.detectors]
        if all_phases and len(set(all_phases)) == 1:
            snap.phase = all_phases[0]

    def update_detector_score(
        self, agent_name: str, detector_name: str, score: float
    ) -> None:
        snap = self.agents.get(agent_name)
        if not snap:
            return
        for det in snap.detectors:
            if det.name == detector_name:
                det.last_score = round(score, 4)
                break

    def update_cortex_test_buffer(self, cortex_name: str, buffer_count: int) -> None:
        snap = self.cortex.get(cortex_name)
        if snap:
            snap.test_buffer_count = buffer_count
            self._push_sse("cortex_test_update", {
                "name": cortex_name,
                "test_sample_rate": snap.test_sample_rate,
                "test_buffer_count": buffer_count,
                "last_test_result": snap.last_test_result,
            })

    def update_cortex_test_rate(self, cortex_name: str, rate: float) -> None:
        snap = self.cortex.get(cortex_name)
        if snap:
            snap.test_sample_rate = rate
            self._push_sse("cortex_test_update", {
                "name": cortex_name,
                "test_sample_rate": rate,
                "test_buffer_count": snap.test_buffer_count,
                "last_test_result": snap.last_test_result,
            })

    def update_cortex_test_result(self, cortex_name: str, result: dict) -> None:
        snap = self.cortex.get(cortex_name)
        if snap:
            snap.last_test_result = result
            self._push_sse("cortex_test_update", {
                "name": cortex_name,
                "test_sample_rate": snap.test_sample_rate,
                "test_buffer_count": snap.test_buffer_count,
                "last_test_result": result,
            })

    def _store_payload(self, correlation_id: str, input_body: dict | None, output_body: dict | None) -> None:
        if correlation_id in self._payload_store:
            self._payload_store[correlation_id] = {"input": input_body, "output": output_body}
            return
        if len(self._payload_order) >= _MAX_PAYLOAD_STORE:
            oldest = self._payload_order[0]
            self._payload_store.pop(oldest, None)
        self._payload_store[correlation_id] = {"input": input_body, "output": output_body}
        self._payload_order.append(correlation_id)

    def get_payload(self, correlation_id: str) -> dict | None:
        return self._payload_store.get(correlation_id)

    # ------------------------------------------------------------------
    # SSE helpers
    # ------------------------------------------------------------------

    def _push_sse(self, event_type: str, data: dict) -> None:
        msg = {"type": event_type, "data": data}
        try:
            self._sse_queue.put_nowait(msg)
        except asyncio.QueueFull:
            # Drop oldest, push newest — never block the caller
            try:
                self._sse_queue.get_nowait()
                self._sse_queue.put_nowait(msg)
            except Exception:
                pass

    async def sse_stream(self):
        """Async generator that yields SSE-formatted strings."""
        while True:
            msg = await self._sse_queue.get()
            yield f"data: {json.dumps(msg)}\n\n"

    # ------------------------------------------------------------------
    # Snapshot for REST API
    # ------------------------------------------------------------------

    def to_status_dict(self) -> dict:
        return {
            "agents": {k: asdict(v) for k, v in self.agents.items()},
            "cortex": {k: asdict(v) for k, v in self.cortex.items()},
            "recent_events": [asdict(e) for e in list(self.recent_events)[-50:]],
            "recent_alerts": [asdict(a) for a in list(self.recent_alerts)],
        }


# Module-level singleton — created by launcher, shared across the process
dashboard_state: DashboardState | None = None


def get_state() -> DashboardState:
    global dashboard_state
    if dashboard_state is None:
        dashboard_state = DashboardState()
    return dashboard_state
