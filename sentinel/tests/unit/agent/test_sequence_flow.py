"""Integration-style tests for the sequence-aware detector flow in process_pair.

Uses a torch-free mock sequence detector so the use-case plumbing (rolling
window, cold start, whole-window test/denial buffers, contiguous training
buffer, denial clearing) is tested independently of any concrete network.
"""

import asyncio
from datetime import datetime, timezone

import pytest

from sentinel.agent.use_cases import process_pair
from sentinel.agent.use_cases.process_message import AgentState
from sentinel.config.schema import AgentConfig, DenialRuleConfig, ReportingConfig
from sentinel.domain.models import CorrelatedPair, InputCapture, ModelPhase, ProcessedEvent
from sentinel.domain.ports.detector import DetectorState, IDetector
from sentinel.domain.ports.model_store import IModelStore
from sentinel.domain.ports.reporter import IReporter


class MockReporter(IReporter):
    def __init__(self):
        self.events: list[ProcessedEvent] = []

    async def publish_event(self, event):
        self.events.append(event)

    async def publish_heartbeat(self, heartbeat):
        pass


class MockModelStore(IModelStore):
    async def save_version(self, name, model):
        return "v_test"

    async def load(self, name, version=None):
        return None

    async def list_versions(self, name):
        return []

    async def rollback(self, name):
        pass

    async def save_detector_version(self, agent_name, detector_name, model):
        return "v_test"

    async def load_detector(self, agent_name, detector_name, version=None):
        return None

    async def list_detector_versions(self, agent_name, detector_name):
        return []


class MockSequenceDetector(IDetector):
    """Sequence-aware mock: records every sample passed to score()."""

    def __init__(self, name="seq", seq_len=3, always_anomaly=False):
        self._name = name
        self._seq_len = seq_len
        self._always_anomaly = always_anomaly
        self.scored_samples: list[dict] = []

    @property
    def name(self):
        return self._name

    @property
    def algorithm(self):
        return "mock_seq"

    @property
    def training_window(self):
        return 100

    @property
    def test_buffer_size(self):
        return 5

    @property
    def requires_sequence(self):
        return True

    @property
    def sequence_length(self):
        return self._seq_len

    def extract_features(self, event_dict):
        return [0.0]

    def fit(self, samples):
        return {"mock": True}

    def score(self, model, event_dict):
        self.scored_samples.append(event_dict)
        return 1.0

    def is_anomaly(self, score, model=None):
        return self._always_anomaly


def _agent_config(denial_rules=None):
    return AgentConfig(
        name="agent01",
        reporting=ReportingConfig(heartbeat_interval_s=30, send_all_events=True),
        denial_rules=denial_rules or [],
    )


def _make_pair(correlation_id="c1", status="ok", latency=100.0):
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    input_at = datetime(2024, 1, 1, 11, 59, 59, tzinfo=timezone.utc)
    body = {"correlationId": correlation_id, "status": status}
    return CorrelatedPair(
        engine_name="agent01",
        correlation_id=correlation_id,
        inputs=[InputCapture(name="request", body=body, received_at=input_at, size_bytes=10)],
        output_body={"correlationId": correlation_id, "result": 1},
        output_received_at=now,
        output_size_bytes=10,
        processing_latency_ms=latency,
        timed_out=False,
    )


def _make_state(phase, detector):
    det_state = DetectorState(
        name=detector.name,
        algorithm=detector.algorithm,
        phase=phase,
        model={"mock": True} if phase == ModelPhase.INFERENCE else None,
        training_buffer=[],
    )
    state = AgentState(agent_name="agent01", detectors={detector.name: det_state})
    return state, det_state


async def _run_pairs(n, state, detector, config=None):
    config = config or _agent_config()
    reporter = MockReporter()
    store = MockModelStore()
    detectors = {detector.name: detector}
    for i in range(n):
        await process_pair(
            pair=_make_pair(correlation_id=f"c{i}"),
            agent_config=config,
            state=state,
            detectors=detectors,
            model_store=store,
            reporter=reporter,
        )
    await asyncio.sleep(0)   # let fire-and-forget publish tasks run
    return reporter


@pytest.mark.asyncio
async def test_sequence_buffer_fills_and_trims():
    detector = MockSequenceDetector(seq_len=3)
    state, det_state = _make_state(ModelPhase.TRAINING, detector)
    await _run_pairs(5, state, detector)
    assert len(det_state.sequence_buffer) == 3   # trimmed to sequence_length


@pytest.mark.asyncio
async def test_cold_start_skips_scoring_until_window_full():
    detector = MockSequenceDetector(seq_len=3)
    state, det_state = _make_state(ModelPhase.INFERENCE, detector)
    await _run_pairs(2, state, detector)
    assert detector.scored_samples == []          # window not full yet

    await _run_pairs(1, state, detector)
    assert len(detector.scored_samples) == 1      # third event fills the window


@pytest.mark.asyncio
async def test_score_receives_full_window():
    detector = MockSequenceDetector(seq_len=3)
    state, det_state = _make_state(ModelPhase.INFERENCE, detector)
    await _run_pairs(4, state, detector)

    sample = detector.scored_samples[-1]
    assert "_sequence" in sample
    assert len(sample["_sequence"]) == 3


@pytest.mark.asyncio
async def test_training_buffer_stays_contiguous_and_test_buffer_holds_windows():
    detector = MockSequenceDetector(seq_len=3)
    state, det_state = _make_state(ModelPhase.TRAINING, detector)
    await _run_pairs(6, state, detector)

    # Every event lands in the training buffer — none diverted away.
    assert len(det_state.training_buffer) == 6

    # Test buffer holds whole-window dicts, only produced once the window is full.
    assert len(det_state.test_buffer) == 4        # events 3..6 each snapshot a window
    assert all("_sequence" in s and len(s["_sequence"]) == 3 for s in det_state.test_buffer)


@pytest.mark.asyncio
async def test_denial_hit_stores_window_and_clears_sequence():
    rule = DenialRuleConfig(
        name="failed",
        condition="input.status == 'failed'",
        score=-1.0,
    )
    detector = MockSequenceDetector(seq_len=3)
    state, det_state = _make_state(ModelPhase.TRAINING, detector)
    config = _agent_config(denial_rules=[rule])

    reporter = MockReporter()
    store = MockModelStore()
    detectors = {detector.name: detector}

    # Two normal events, then a denial hit.
    for i in range(2):
        await process_pair(
            pair=_make_pair(correlation_id=f"c{i}"),
            agent_config=config, state=state,
            detectors=detectors, model_store=store, reporter=reporter,
        )
    await process_pair(
        pair=_make_pair(correlation_id="bad", status="failed"),
        agent_config=config, state=state,
        detectors=detectors, model_store=store, reporter=reporter,
    )
    await asyncio.sleep(0)

    # Window [n1, n2, bad] was full at the denial hit → stored for recall validation.
    assert len(det_state.denial_buffer) == 1
    assert len(det_state.denial_buffer[0]["_sequence"]) == 3
    # The contaminated window must not remain as context.
    assert len(det_state.sequence_buffer) == 0
