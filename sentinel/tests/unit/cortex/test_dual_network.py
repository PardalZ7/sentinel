from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import pytest

from sentinel.cortex.dual_network import DualNetworkResult, run_dual_network
from sentinel.cortex.use_cases.aggregate import LayerStateVector
from sentinel.cortex.use_cases.autoencoder import create_autoencoder
from sentinel.domain.models import ModelPhase, TemporalAnomalySignal
from sentinel.cortex.use_cases.causal_chain import add_temporal_signal


def _make_signal(layer_name: str, offset_s: float = 0.0) -> TemporalAnomalySignal:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t = base + timedelta(seconds=offset_s)
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


def _make_layer_vectors(layers: list[str]) -> dict[str, LayerStateVector]:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return {
        name: LayerStateVector(
            layer_name=name,
            last_seen=base,
            model_phase=ModelPhase.INFERENCE,
            anomaly_rate=0.1,
            avg_score=0.2,
            avg_latency_ms=100.0,
            anomaly_rate_z=0.5,
            volume_z=0.3,
            bucket_confidence=10,
        )
        for name in layers
    }


class MockCortexConfig:
    name = "cortex01"
    silence_threshold_s = 60.0


# ── run_dual_network — no anomalies ──────────────────────────────────────────

def test_run_dual_network_no_anomalies_returns_no_alert():
    window = deque()
    layer_vectors = _make_layer_vectors(["layer01"])
    config = MockCortexConfig()

    result = run_dual_network(
        causal_window=window,
        autoencoder_model=None,
        layer_vectors=layer_vectors,
        config=config,
    )

    assert isinstance(result, DualNetworkResult)
    assert result.final_alert is None


# ── run_dual_network — causal chain only ─────────────────────────────────────

def test_run_dual_network_causal_chain_triggers_alert():
    window = deque()
    for i, name in enumerate(["layer01", "layer02", "layer03"]):
        add_temporal_signal(window, _make_signal(name, offset_s=float(i * 2)))

    layer_vectors = _make_layer_vectors(["layer01", "layer02", "layer03"])
    config = MockCortexConfig()

    result = run_dual_network(
        causal_window=window,
        autoencoder_model=None,
        layer_vectors=layer_vectors,
        config=config,
    )

    assert result.final_alert is not None
    assert result.causal is not None
    assert result.causal.root_layer == "layer01"


# ── run_dual_network — systemic anomaly ──────────────────────────────────────

def test_run_dual_network_systemic_triggers_alert():
    window = deque()
    layer_vectors = _make_layer_vectors(["layer01"])
    config = MockCortexConfig()

    model = create_autoencoder(input_dim=5, hidden_dim=2)

    result = run_dual_network(
        causal_window=window,
        autoencoder_model=model,
        layer_vectors=layer_vectors,
        config=config,
        baseline_error=0.0,
    )

    assert isinstance(result, DualNetworkResult)
    assert isinstance(result.systemic_error, float)


# ── run_dual_network — handles autoencoder error gracefully ──────────────────

def test_run_dual_network_autoencoder_exception_is_handled():
    window = deque()
    layer_vectors = _make_layer_vectors(["layer01"])
    config = MockCortexConfig()

    broken_model = MagicMock()
    broken_model.side_effect = RuntimeError("model exploded")

    result = run_dual_network(
        causal_window=window,
        autoencoder_model=broken_model,
        layer_vectors=layer_vectors,
        config=config,
    )

    assert isinstance(result, DualNetworkResult)
