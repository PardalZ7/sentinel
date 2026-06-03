"""Unit tests for multi-detector use cases: champion, train, set_phase, run_test."""

import pytest

from sentinel.agent.use_cases.champion import ChampionResult, evaluate_fp_rate, run_selection
from sentinel.agent.use_cases.run_test import run_detector_test
from sentinel.agent.use_cases.set_phase import set_phase
from sentinel.agent.use_cases.train import accumulate_sample, run_training, should_train
from sentinel.agent.use_cases.process_message import AgentState
from sentinel.domain.models import ModelPhase
from sentinel.domain.ports.detector import DetectorState, IDetector


# ── Shared mock ───────────────────────────────────────────────────────────────

class StubDetector(IDetector):
    def __init__(self, *, window=5, buffer_size=10, score=0.0, anomaly=False):
        self._window = window
        self._buffer_size = buffer_size
        self._score = score
        self._anomaly = anomaly

    @property
    def name(self):
        return "stub"

    @property
    def algorithm(self):
        return "mock"

    @property
    def training_window(self):
        return self._window

    @property
    def test_buffer_size(self):
        return self._buffer_size

    def extract_features(self, event_dict):
        return [event_dict.get("processing_latency_ms", 0.0)]

    def fit(self, samples):
        return {"n": len(samples)}

    def score(self, model, event_dict):
        return self._score

    def is_anomaly(self, score, model=None):
        return self._anomaly


def _det_state(phase=ModelPhase.TRAINING, model=None, buffer=None, test_buffer=None):
    return DetectorState(
        name="stub",
        algorithm="mock",
        phase=phase,
        model=model,
        training_buffer=buffer or [],
        test_buffer=test_buffer or [],
    )


def _agent_state(det_state):
    return AgentState(agent_name="test_agent", detectors={"stub": det_state})


# ── accumulate_sample ────────────────────────────────────────────────────────

def test_accumulate_sample_appends():
    buf = [{"a": 1}]
    new_buf = accumulate_sample(buf, {"b": 2})
    assert len(new_buf) == 2
    assert new_buf[-1] == {"b": 2}


def test_accumulate_sample_does_not_mutate():
    original = [{"a": 1}]
    new_buf = accumulate_sample(original, {"b": 2})
    assert len(original) == 1  # original unchanged
    assert len(new_buf) == 2


# ── should_train ─────────────────────────────────────────────────────────────

def test_should_train_when_window_reached():
    det = StubDetector(window=3)
    assert should_train([{}, {}, {}], det.training_window) is True


def test_should_train_false_below_window():
    det = StubDetector(window=3)
    assert should_train([{}, {}], det.training_window) is False


# ── run_training ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_training_returns_model_and_clears_buffer():
    det = StubDetector(window=3)
    buffer = [{"processing_latency_ms": 10.0} for _ in range(3)]
    model, remaining = await run_training(det, buffer, "agent01")
    assert model is not None
    assert model == {"n": 3}
    assert remaining == []


# ── evaluate_fp_rate ─────────────────────────────────────────────────────────

def test_evaluate_fp_rate_no_anomalies():
    det = StubDetector(anomaly=False)
    test_buf = [{"processing_latency_ms": 10.0} for _ in range(5)]
    fp = evaluate_fp_rate(det, {"model": True}, test_buf)
    assert fp == 0.0


def test_evaluate_fp_rate_all_anomalies():
    det = StubDetector(anomaly=True)
    test_buf = [{"processing_latency_ms": 10.0} for _ in range(4)]
    fp = evaluate_fp_rate(det, {"model": True}, test_buf)
    assert fp == 1.0


def test_evaluate_fp_rate_empty_buffer_returns_worst():
    # Empty buffer → 1.0 (worst) so no unvalidated model is promoted
    det = StubDetector()
    assert evaluate_fp_rate(det, {}, []) == 1.0


# ── run_selection ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_selection_challenger_wins():
    det = StubDetector(anomaly=False)
    test_buf = [{"processing_latency_ms": 10.0} for _ in range(5)]
    result = await run_selection(
        detector=det,
        challenger={"new_model": True},
        current_champion={"old_model": True},
        current_champion_fp_rate=0.5,
        test_buffer=test_buf,
    )
    assert isinstance(result, ChampionResult)
    assert result.is_new_champion is True
    assert result.model == {"new_model": True}
    assert result.fp_rate == 0.0


@pytest.mark.asyncio
async def test_run_selection_champion_retained_when_challenger_worse():
    det = StubDetector(anomaly=True)  # challenger has 100% FP rate
    test_buf = [{"processing_latency_ms": 10.0} for _ in range(5)]
    result = await run_selection(
        detector=det,
        challenger={"new_model": True},
        current_champion={"old_model": True},
        current_champion_fp_rate=0.1,
        test_buffer=test_buf,
    )
    assert result.is_new_champion is False
    assert result.model == {"old_model": True}
    assert result.fp_rate == 0.1


@pytest.mark.asyncio
async def test_run_selection_no_champion_accepts_first_challenger():
    det = StubDetector(anomaly=False)
    test_buf = [{"processing_latency_ms": 10.0} for _ in range(3)]
    result = await run_selection(
        detector=det,
        challenger={"model": True},
        current_champion=None,
        current_champion_fp_rate=1.0,
        test_buffer=test_buf,
    )
    assert result.is_new_champion is True


# ── run_detector_test ─────────────────────────────────────────────────────────

def test_run_detector_test_no_model():
    det = StubDetector()
    state = _det_state(model=None, test_buffer=[{"x": 1}])
    result = run_detector_test(det, state)
    assert "error" in result
    assert result["samples_run"] == 0


def test_run_detector_test_empty_buffer():
    det = StubDetector()
    state = _det_state(model={"m": 1}, test_buffer=[])
    result = run_detector_test(det, state)
    assert result["samples_run"] == 0
    assert result["failed"] == 0


def test_run_detector_test_all_pass():
    det = StubDetector(anomaly=False)
    test_buf = [{"processing_latency_ms": 10.0} for _ in range(5)]
    state = _det_state(model={"m": 1}, test_buffer=test_buf)
    result = run_detector_test(det, state)
    assert result["samples_run"] == 5
    assert result["failed"] == 0
    assert result["fp_rate"] == 0.0


def test_run_detector_test_all_fail():
    det = StubDetector(anomaly=True)
    test_buf = [{"processing_latency_ms": 10.0} for _ in range(4)]
    state = _det_state(model={"m": 1}, test_buffer=test_buf)
    result = run_detector_test(det, state)
    assert result["failed"] == 4
    assert result["fp_rate"] == 1.0


# ── set_phase ────────────────────────────────────────────────────────────────

def test_set_phase_all_detectors():
    det1 = _det_state(phase=ModelPhase.TRAINING)
    det2 = _det_state(phase=ModelPhase.TRAINING)
    det2 = DetectorState(
        name="det2", algorithm="mock", phase=ModelPhase.TRAINING, model=None, training_buffer=[]
    )
    state = AgentState(agent_name="a", detectors={"stub": det1, "det2": det2})

    new_state = set_phase(state, ModelPhase.INFERENCE)
    for det in new_state.detectors.values():
        assert det.phase == ModelPhase.INFERENCE


def test_set_phase_single_detector():
    det1 = _det_state(phase=ModelPhase.TRAINING)
    det2 = DetectorState(
        name="det2", algorithm="mock", phase=ModelPhase.TRAINING, model=None, training_buffer=[]
    )
    state = AgentState(agent_name="a", detectors={"stub": det1, "det2": det2})

    new_state = set_phase(state, ModelPhase.INFERENCE, detector_name="stub")
    assert new_state.detectors["stub"].phase == ModelPhase.INFERENCE
    assert new_state.detectors["det2"].phase == ModelPhase.TRAINING


def test_set_phase_unknown_detector_is_noop():
    det = _det_state(phase=ModelPhase.TRAINING)
    state = AgentState(agent_name="a", detectors={"stub": det})
    new_state = set_phase(state, ModelPhase.INFERENCE, detector_name="nonexistent")
    assert new_state.detectors["stub"].phase == ModelPhase.TRAINING


def test_set_phase_returns_new_state():
    det = _det_state(phase=ModelPhase.TRAINING)
    state = AgentState(agent_name="a", detectors={"stub": det})
    new_state = set_phase(state, ModelPhase.INFERENCE)
    assert new_state is not state
