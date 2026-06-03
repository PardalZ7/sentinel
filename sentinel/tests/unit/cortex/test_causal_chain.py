from collections import deque
from datetime import datetime, timedelta, timezone

import pytest

from sentinel.cortex.use_cases.causal_chain import (
    CausalDiagnosis,
    add_temporal_signal,
    analyze_causality,
)
from sentinel.domain.models import TemporalAnomalySignal


def _make_signal(
    layer_name: str,
    offset_s: float = 0.0,
    base_time: datetime | None = None,
) -> TemporalAnomalySignal:
    if base_time is None:
        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t = base_time + timedelta(seconds=offset_s)
    return TemporalAnomalySignal(
        layer_name=layer_name,
        window_start=t - timedelta(seconds=120),
        window_end=t,
        hour_of_week=t.weekday() * 24 + t.hour,
        anomaly_rate=0.3,
        anomaly_rate_z=2.5,
        volume=50,
        volume_z=1.0,
        avg_score_z=2.1,
        avg_latency_z=0.5,
        is_rate_anomaly=True,
        is_volume_anomaly=False,
        is_score_anomaly=True,
        is_latency_anomaly=False,
        contributing_agents=["agent01"],
        bucket_confidence=5,
    )


# ── add_temporal_signal ───────────────────────────────────────────────────────

def test_add_temporal_signal_adds_to_window():
    window = deque()
    signal = _make_signal("layer01")
    result = add_temporal_signal(window, signal)
    assert len(result) == 1


def test_add_temporal_signal_multiple():
    window = deque()
    for i in range(5):
        add_temporal_signal(window, _make_signal(f"layer{i:02d}"))
    assert len(window) == 5


# ── analyze_causality ─────────────────────────────────────────────────────────

def test_analyze_causality_empty_window():
    window = deque()
    result = analyze_causality(window)
    assert result.root_layer is None
    assert result.affected_layers == []
    assert result.confidence == 0.0


def test_analyze_causality_single_signal():
    window = deque()
    add_temporal_signal(window, _make_signal("layer01", offset_s=0))
    result = analyze_causality(window)
    assert result.root_layer == "layer01"
    assert result.affected_layers == []
    assert result.sequence == ["layer01"]


def test_analyze_causality_chain():
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    window = deque()
    add_temporal_signal(window, _make_signal("layer01", offset_s=0.0, base_time=base))
    add_temporal_signal(window, _make_signal("layer02", offset_s=5.0, base_time=base))
    add_temporal_signal(window, _make_signal("layer03", offset_s=10.0, base_time=base))

    result = analyze_causality(window, window_s=30.0)

    assert result.root_layer == "layer01"
    assert result.sequence == ["layer01", "layer02", "layer03"]


def test_analyze_causality_filters_old_signals():
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    window = deque()
    add_temporal_signal(window, _make_signal("layer_old", offset_s=0.0, base_time=base))
    add_temporal_signal(window, _make_signal("layer01", offset_s=60.0, base_time=base))
    add_temporal_signal(window, _make_signal("layer02", offset_s=70.0, base_time=base))

    result = analyze_causality(window, window_s=30.0)

    assert "layer_old" not in result.sequence


def test_analyze_causality_confidence_range():
    window = deque()
    for i in range(3):
        add_temporal_signal(window, _make_signal(f"layer{i:02d}", offset_s=float(i * 2)))

    result = analyze_causality(window)
    assert 0.0 <= result.confidence <= 1.0
