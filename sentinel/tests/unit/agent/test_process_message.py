from collections import deque
from datetime import datetime, timezone

import pytest

from sentinel.agent.use_cases import process_pair
from sentinel.agent.use_cases.correlate import store_input_capture, try_build_pair
from sentinel.config.schema import AgentConfig, CorrelationEngineConfig, InputConfig, ReportingConfig, TransportResourceConfig
from sentinel.domain.models import AnomalyType, CorrelatedPair, ModelPhase, ProcessedEvent, RawMessage
from sentinel.domain.ports.correlation_store import ICorrelationStore
from sentinel.domain.ports.detector import DetectorState, IDetector
from sentinel.domain.ports.model_store import IModelStore
from sentinel.domain.ports.reporter import IReporter
from sentinel.agent.use_cases.process_message import AgentState


# ── Mocks ─────────────────────────────────────────────────────────────────────

class MockCorrelationStore(ICorrelationStore):
    def __init__(self):
        self._store: dict = {}

    async def save(self, key, data, ttl_s):
        self._store[key] = data

    async def get(self, key):
        return self._store.get(key)

    async def delete(self, key):
        self._store.pop(key, None)

    async def record_group_source(self, key, source_name, event_snapshot, expected_sources, ttl_s):
        existing = self._store.get(key, {})
        if not existing:
            existing = {s: None for s in expected_sources}
        existing[source_name] = event_snapshot
        self._store[key] = existing
        return existing

    async def try_claim_complete_group(self, key, expected_sources):
        group = self._store.get(key)
        if group is None:
            return False
        if any(group.get(s) is None for s in expected_sources):
            return False
        self._store.pop(key)
        return True


class MockModelStore(IModelStore):
    def __init__(self):
        self.saved: list = []

    async def save_version(self, name, model):
        return "v_test"

    async def load(self, name, version=None):
        return None

    async def list_versions(self, name):
        return []

    async def rollback(self, name):
        pass

    async def save_detector_version(self, agent_name, detector_name, model):
        self.saved.append((agent_name, detector_name))
        return "v_test"

    async def load_detector(self, agent_name, detector_name, version=None):
        return None

    async def list_detector_versions(self, agent_name, detector_name):
        return []


class MockReporter(IReporter):
    def __init__(self):
        self.events: list[ProcessedEvent] = []

    async def publish_event(self, event):
        self.events.append(event)

    async def publish_heartbeat(self, heartbeat):
        pass


class MockDetector(IDetector):
    def __init__(self, name="mock", *, always_anomaly=False, score_value=-0.5):
        self._name = name
        self._always_anomaly = always_anomaly
        self._score_value = score_value
        self._fitted = []

    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return "mock"

    @property
    def training_window(self):
        return 5

    @property
    def test_buffer_size(self):
        return 10

    def extract_features(self, event_dict):
        return [event_dict.get("processing_latency_ms", 0.0)]

    def fit(self, samples):
        self._fitted = samples
        return {"mock_model": True, "samples": len(samples)}

    def score(self, model, event_dict):
        return self._score_value

    def is_anomaly(self, score, model=None):
        return self._always_anomaly


# ── Helpers ───────────────────────────────────────────────────────────────────

def _agent_config(send_all=True):
    return AgentConfig(
        name="agent01",
        reporting=ReportingConfig(heartbeat_interval_s=30, send_all_events=send_all),
    )


def _engine_config():
    return CorrelationEngineConfig(
        name="agent01",
        inputs=[
            InputConfig(name="request",  transport=TransportResourceConfig(type="sqs", resource="http://sqs/in"),  correlation_field="correlationId"),
            InputConfig(name="response", transport=TransportResourceConfig(type="sqs", resource="http://sqs/out"), correlation_field="correlationId"),
        ],
        correlation_ttl_s=300,
    )


def _input_cfg(config: CorrelationEngineConfig) -> InputConfig:
    return config.inputs[0]


def _make_state(phase=ModelPhase.TRAINING, *, always_anomaly=False):
    det = DetectorState(
        name="mock",
        algorithm="mock",
        phase=phase,
        model={"mock_model": True} if phase == ModelPhase.INFERENCE else None,
        training_buffer=[],
    )
    return (
        AgentState(agent_name="agent01", detectors={"mock": det}),
        {"mock": MockDetector("mock", always_anomaly=always_anomaly)},
    )


def _now():
    return datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


def _make_pair(
    correlation_id="corr-1",
    *,
    input_body=None,
    output_body=None,
    latency_ms=1000.0,
    timed_out=False,
):
    from sentinel.domain.models import InputCapture
    now = _now()
    input_at = datetime(2024, 1, 1, 11, 59, 59, tzinfo=timezone.utc)
    body = input_body or {"correlationId": correlation_id, "data": "input"}
    return CorrelatedPair(
        engine_name="agent01",
        correlation_id=correlation_id,
        inputs=[] if timed_out else [InputCapture(name="request", body=body, received_at=input_at, size_bytes=10)],
        output_body=output_body or {"correlationId": correlation_id, "data": "output"},
        output_received_at=now,
        output_size_bytes=10,
        processing_latency_ms=latency_ms,
        timed_out=timed_out,
    )


def _make_raw(body, *, ts=None):
    import hashlib
    msg_id = hashlib.md5(str(body).encode()).hexdigest()
    return RawMessage(id=msg_id, body=body, received_at=ts or _now(), size_bytes=len(str(body)))


# ── correlate use case ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_store_request_persists():
    store = MockCorrelationStore()
    config = _engine_config()
    body = {"correlationId": "corr-1", "data": "hello"}
    await store_input_capture(store, config, _input_cfg(config), _make_raw(body))
    stored = await store.get("agent01:corr-1")
    assert stored is not None
    assert stored["request"]["body"] == body


@pytest.mark.asyncio
async def test_store_request_missing_field():
    store = MockCorrelationStore()
    config = _engine_config()
    cid, group = await store_input_capture(store, config, _input_cfg(config), _make_raw({"no_correlation": "here"}))
    assert cid is None
    assert group is None
    assert len(store._store) == 0


@pytest.mark.asyncio
async def test_build_pair_normal_path():
    store = MockCorrelationStore()
    config = _engine_config()
    input_ts  = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    output_ts = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)

    input_body  = {"correlationId": "corr-2", "data": "input"}
    output_body = {"correlationId": "corr-2", "data": "output"}

    request_cfg  = config.inputs[0]
    response_cfg = config.inputs[1]

    cid, _ = await store_input_capture(store, config, request_cfg,  _make_raw(input_body,  ts=input_ts))
    cid, group = await store_input_capture(store, config, response_cfg, _make_raw(output_body, ts=output_ts))

    pair = await try_build_pair(store, config, cid, group)

    assert pair is not None
    assert pair.correlation_id == "corr-2"
    assert pair.processing_latency_ms == pytest.approx(1000.0, abs=10)
    assert pair.timed_out is False
    assert await store.get("agent01:corr-2") is None  # key deleted after claim


@pytest.mark.asyncio
async def test_build_pair_incomplete():
    store = MockCorrelationStore()
    config = _engine_config()
    body = {"correlationId": "corr-3", "data": "input"}
    # only request stored — response missing
    cid, group = await store_input_capture(store, config, config.inputs[0], _make_raw(body))
    pair = await try_build_pair(store, config, cid, group)
    assert pair is None  # group incomplete


# ── process_pair — normal scoring path ───────────────────────────────────────

@pytest.mark.asyncio
async def test_process_pair_normal():
    reporter = MockReporter()
    model_store = MockModelStore()
    config = _agent_config(send_all=True)
    state, detectors = _make_state(ModelPhase.TRAINING)

    pair = _make_pair("corr-2", latency_ms=1000.0)
    new_state = await process_pair(pair, config, state, detectors, model_store, reporter)

    assert new_state.message_count == 1
    assert len(reporter.events) == 1
    event = reporter.events[0]
    assert event.agent_name == "agent01"
    assert event.correlation_id == "corr-2"
    assert event.processing_latency_ms == pytest.approx(1000.0)
    assert event.is_anomaly is False


@pytest.mark.asyncio
async def test_process_pair_timeout_event():
    reporter = MockReporter()
    model_store = MockModelStore()
    config = _agent_config(send_all=True)
    state, detectors = _make_state(ModelPhase.TRAINING)

    pair = _make_pair("expired-corr", timed_out=True)
    new_state = await process_pair(pair, config, state, detectors, model_store, reporter)

    assert len(reporter.events) == 1
    event = reporter.events[0]
    assert event.is_anomaly is True
    assert event.anomaly_type == AnomalyType.TIMEOUT
    # state.message_count not incremented for timeouts
    assert new_state.message_count == 0


@pytest.mark.asyncio
async def test_process_pair_inference_anomaly():
    reporter = MockReporter()
    model_store = MockModelStore()
    config = _agent_config(send_all=False)
    state, detectors = _make_state(ModelPhase.INFERENCE, always_anomaly=True)

    pair = _make_pair("corr-anomaly")
    new_state = await process_pair(pair, config, state, detectors, model_store, reporter)

    assert len(reporter.events) == 1
    assert reporter.events[0].is_anomaly is True
    assert new_state.error_count == 1


@pytest.mark.asyncio
async def test_multi_detector_any_anomaly_propagates():
    reporter = MockReporter()
    model_store = MockModelStore()
    config = _agent_config(send_all=False)

    det_normal = DetectorState(
        name="normal_det", algorithm="mock", phase=ModelPhase.INFERENCE,
        model={"ok": True}, training_buffer=[],
    )
    det_anomaly = DetectorState(
        name="anomaly_det", algorithm="mock", phase=ModelPhase.INFERENCE,
        model={"ok": True}, training_buffer=[],
    )
    state = AgentState(
        agent_name="agent01",
        detectors={"normal_det": det_normal, "anomaly_det": det_anomaly},
    )
    detectors = {
        "normal_det": MockDetector("normal_det", always_anomaly=False),
        "anomaly_det": MockDetector("anomaly_det", always_anomaly=True),
    }

    pair = _make_pair("corr-multi")
    new_state = await process_pair(pair, config, state, detectors, model_store, reporter)

    assert reporter.events[0].is_anomaly is True


@pytest.mark.asyncio
async def test_training_triggers_at_window():
    """When training_window samples accumulate, training_in_progress is set True.

    Actual model saving happens in AgentRunner._run_background_training (not in
    process_pair itself), so we only verify the flag here.
    """
    reporter = MockReporter()
    model_store = MockModelStore()
    config = _agent_config(send_all=True)

    det_state = DetectorState(
        name="mock", algorithm="mock", phase=ModelPhase.TRAINING,
        model=None, training_buffer=[],
        test_buffer=deque([{"dummy": i} for i in range(10)]),
        test_sample_rate=0.0,  # disable random diversions for deterministic test
    )
    state = AgentState(agent_name="agent01", detectors={"mock": det_state})
    detectors = {"mock": MockDetector("mock")}

    for i in range(5):
        pair = _make_pair(f"corr-train-{i}")
        state = await process_pair(pair, config, state, detectors, model_store, reporter)

    assert state.detectors["mock"].training_in_progress is True


# ── AgentState backward-compat properties ────────────────────────────────────

def test_agent_state_phase_property():
    det = DetectorState(
        name="d1", algorithm="mock", phase=ModelPhase.INFERENCE, model=None, training_buffer=[]
    )
    state = AgentState(agent_name="a", detectors={"d1": det})
    assert state.phase == ModelPhase.INFERENCE


def test_agent_state_phase_cold_when_no_detectors():
    state = AgentState(agent_name="a", detectors={})
    assert state.phase == ModelPhase.COLD
