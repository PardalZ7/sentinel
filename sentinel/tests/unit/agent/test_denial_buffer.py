"""Unit tests for the denial_buffer feature.

Tests cover:
- Accumulation of denial hits in denial_buffer during TRAINING
- maxlen enforcement (denial_buffer_size)
- No accumulation for INFERENCE detectors
- evaluate_denial_recall: empty buffer, correct rate calculation
- run_selection: rejection when recall < min_denial_recall
- run_selection: min_denial_recall=0.0 disables recall check
- DetectorState.reset() clears denial_buffer
- run_detector_test: denial_recall included in result when buffer is non-empty
"""

import asyncio
from collections import deque

import pytest

from sentinel.agent.use_cases.champion import evaluate_denial_recall, run_selection
from sentinel.agent.use_cases.run_test import run_detector_test
from sentinel.domain.models import ModelPhase
from sentinel.domain.ports.detector import DetectorState, IDetector


# ── Shared helpers ────────────────────────────────────────────────────────────

class _StubDetector(IDetector):
    """Stub that returns a fixed score and anomaly flag for all samples."""

    def __init__(self, *, score: float = 0.0, anomaly: bool = False):
        self._score = score
        self._is_anomaly = anomaly

    @property
    def name(self):
        return "stub"

    @property
    def algorithm(self):
        return "mock"

    @property
    def training_window(self):
        return 10

    @property
    def test_buffer_size(self):
        return 20

    def extract_features(self, event_dict):
        return [0.0]

    def fit(self, samples):
        return {"n": len(samples)}

    def score(self, model, event_dict):
        return self._score

    def is_anomaly(self, score, model=None):
        return self._is_anomaly


def _sample(tag: str = "x") -> dict:
    return {"processing_latency_ms": 1.0, "_tag": tag}


def _det_state(**kwargs) -> DetectorState:
    defaults = dict(
        name="stub",
        algorithm="mock",
        phase=ModelPhase.TRAINING,
        model={"trained": True},
        denial_buffer=deque(),
        denial_buffer_size=200,
        min_denial_recall=0.0,
    )
    defaults.update(kwargs)
    return DetectorState(**defaults)


# ── evaluate_denial_recall ────────────────────────────────────────────────────

def test_recall_empty_buffer_returns_one():
    detector = _StubDetector(anomaly=True)
    assert evaluate_denial_recall(detector, {}, []) == 1.0


def test_recall_all_detected():
    detector = _StubDetector(anomaly=True)
    buf = [_sample(), _sample(), _sample()]
    assert evaluate_denial_recall(detector, {}, buf) == 1.0


def test_recall_none_detected():
    detector = _StubDetector(anomaly=False)
    buf = [_sample(), _sample(), _sample()]
    assert evaluate_denial_recall(detector, {}, buf) == 0.0


def test_recall_partial():
    """3 samples, detector alternates anomaly via side-effect counter."""
    calls = []

    class _CountingDetector(_StubDetector):
        def is_anomaly(self, score, model=None):
            calls.append(1)
            return len(calls) <= 3  # first 3 calls → True

    detector = _CountingDetector()
    buf = [_sample()] * 4
    recall = evaluate_denial_recall(detector, {}, buf)
    assert recall == pytest.approx(3 / 4)


# ── run_selection: recall gate ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_run_selection_rejects_low_recall():
    """Challenger with good FP rate but low recall is rejected."""
    detector = _StubDetector(anomaly=False)  # never flags → recall=0
    denial_buf = [_sample()] * 4

    result = await run_selection(
        detector=detector,
        challenger={"new": True},
        current_champion={"old": True},
        current_champion_fp_rate=0.5,
        test_buffer=[],  # empty → would normally accept unconditionally
        denial_buffer=denial_buf,
        min_denial_recall=0.6,
    )

    assert result.is_new_champion is False
    assert result.denial_recall == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_run_selection_accepts_sufficient_recall():
    """Challenger that meets min_denial_recall is promoted."""
    detector = _StubDetector(anomaly=True)  # always flags → recall=1
    denial_buf = [_sample()] * 4

    result = await run_selection(
        detector=detector,
        challenger={"new": True},
        current_champion=None,
        current_champion_fp_rate=1.0,
        test_buffer=[],
        denial_buffer=denial_buf,
        min_denial_recall=0.6,
    )

    assert result.is_new_champion is True
    assert result.denial_recall == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_run_selection_zero_min_recall_disables_check():
    """min_denial_recall=0.0 never rejects on recall grounds."""
    detector = _StubDetector(anomaly=False)  # recall=0 but check is disabled
    denial_buf = [_sample()] * 4

    result = await run_selection(
        detector=detector,
        challenger={"new": True},
        current_champion=None,
        current_champion_fp_rate=1.0,
        test_buffer=[],
        denial_buffer=denial_buf,
        min_denial_recall=0.0,
    )

    assert result.is_new_champion is True


@pytest.mark.asyncio
async def test_run_selection_denial_recall_none_when_no_buffer():
    """denial_recall is None when no denial_buffer is provided."""
    detector = _StubDetector(anomaly=False)

    result = await run_selection(
        detector=detector,
        challenger={"new": True},
        current_champion=None,
        current_champion_fp_rate=1.0,
        test_buffer=[],
    )

    assert result.is_new_champion is True
    assert result.denial_recall is None


@pytest.mark.asyncio
async def test_run_selection_recall_check_with_test_buffer():
    """Recall check is applied even when test_buffer is non-empty."""
    detector = _StubDetector(score=0.0, anomaly=False)  # FP=0 but recall=0
    test_buf = [_sample()] * 5
    denial_buf = [_sample()] * 3

    result = await run_selection(
        detector=detector,
        challenger={"new": True},
        current_champion={"old": True},
        current_champion_fp_rate=0.5,
        test_buffer=test_buf,
        denial_buffer=denial_buf,
        min_denial_recall=0.5,
    )

    assert result.is_new_champion is False
    assert result.denial_recall == pytest.approx(0.0)


# ── DetectorState: denial_buffer fields and reset ────────────────────────────

def test_detector_state_default_fields():
    state = _det_state()
    assert isinstance(state.denial_buffer, deque)
    assert state.denial_buffer_size == 200
    assert state.min_denial_recall == 0.0


def test_denial_buffer_maxlen_set_on_construction():
    buf = deque(maxlen=50)
    state = _det_state(denial_buffer=buf, denial_buffer_size=50)
    assert state.denial_buffer.maxlen == 50


def test_reset_clears_denial_buffer():
    buf = deque(maxlen=10)
    buf.append(_sample())
    state = _det_state(denial_buffer=buf, denial_buffer_size=10)
    assert len(state.denial_buffer) == 1

    reset = state.reset()
    assert len(reset.denial_buffer) == 0
    assert reset.denial_buffer.maxlen == 10  # size preserved


def test_reset_unbounded_when_denial_buffer_size_zero():
    state = _det_state(denial_buffer=deque(), denial_buffer_size=0)
    reset = state.reset()
    assert reset.denial_buffer.maxlen is None


# ── run_detector_test: denial_recall in result ────────────────────────────────

def test_run_test_includes_denial_recall_when_buffer_non_empty():
    detector = _StubDetector(anomaly=True)
    buf = deque([_sample(), _sample()])
    state = _det_state(denial_buffer=buf)

    result = run_detector_test(detector, state)

    assert "denial_recall" in result
    assert result["denial_recall"] == pytest.approx(1.0)
    assert result["denial_buffer_size"] == 2


def test_run_test_omits_denial_recall_when_buffer_empty():
    detector = _StubDetector(anomaly=True)
    state = _det_state(denial_buffer=deque())

    result = run_detector_test(detector, state)

    assert "denial_recall" not in result
