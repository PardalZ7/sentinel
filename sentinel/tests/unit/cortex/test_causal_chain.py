from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from sentinel.cortex.use_cases.causal_chain import (
    CausalDiagnosis,
    add_event,
    analyze_causality,
)
from sentinel.domain.models import AnomalyType, ProcessedEvent


def _make_event(
    agent_name: str,
    is_anomaly: bool = True,
    offset_s: float = 0.0,
    base_time: datetime | None = None,
) -> ProcessedEvent:
    if base_time is None:
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t = base_time + timedelta(seconds=offset_s)
    return ProcessedEvent(
        agent_name=agent_name,
        correlation_id=f"{agent_name}-{offset_s}",
        input_received_at=t,
        output_sent_at=t,
        processing_latency_ms=100.0,
        anomaly_score=-0.5,
        is_anomaly=is_anomaly,
        anomaly_type=AnomalyType.SCORE if is_anomaly else None,
        payload_schema_hash="abc",
        input_size_bytes=100,
        output_size_bytes=100,
    )


# ── add_event ─────────────────────────────────────────────────────────────────

def test_add_event_anomaly_is_added():
    window = deque()
    event = _make_event("agent01", is_anomaly=True)
    result = add_event(window, event)
    assert len(result) == 1


def test_add_event_non_anomaly_not_added():
    window = deque()
    event = _make_event("agent01", is_anomaly=False)
    result = add_event(window, event)
    assert len(result) == 0


def test_add_event_multiple():
    window = deque()
    for i in range(5):
        add_event(window, _make_event(f"agent{i:02d}", is_anomaly=True))
    assert len(window) == 5


# ── analyze_causality ─────────────────────────────────────────────────────────

def test_analyze_causality_empty_window():
    window = deque()
    result = analyze_causality(window)
    assert result.root_agent is None
    assert result.affected_agents == []
    assert result.confidence == 0.0


def test_analyze_causality_single_event():
    window = deque()
    add_event(window, _make_event("agent01", offset_s=0))
    result = analyze_causality(window)
    assert result.root_agent == "agent01"
    assert result.affected_agents == []
    assert result.sequence == ["agent01"]


def test_analyze_causality_chain():
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    window = deque()
    add_event(window, _make_event("agent01", offset_s=0.0, base_time=base))
    add_event(window, _make_event("agent02", offset_s=5.0, base_time=base))
    add_event(window, _make_event("agent03", offset_s=10.0, base_time=base))

    result = analyze_causality(window, window_s=30.0)

    assert result.root_agent == "agent01"
    assert "agent02" in result.affected_agents or "agent02" in result.sequence
    assert "agent03" in result.affected_agents or "agent03" in result.sequence
    assert result.sequence == ["agent01", "agent02", "agent03"]


def test_analyze_causality_filters_old_events():
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    window = deque()
    # Old event — outside 30s window relative to latest
    add_event(window, _make_event("agent_old", offset_s=0.0, base_time=base))
    # Recent events
    add_event(window, _make_event("agent01", offset_s=60.0, base_time=base))
    add_event(window, _make_event("agent02", offset_s=70.0, base_time=base))

    result = analyze_causality(window, window_s=30.0)

    assert "agent_old" not in result.sequence


def test_analyze_causality_confidence_range():
    window = deque()
    for i in range(3):
        add_event(window, _make_event(f"agent{i:02d}", offset_s=float(i * 2)))

    result = analyze_causality(window)
    assert 0.0 <= result.confidence <= 1.0
