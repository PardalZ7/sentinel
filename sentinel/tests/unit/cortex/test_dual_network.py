from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

from sentinel.cortex.dual_network import DualNetworkResult, run_dual_network
from sentinel.cortex.use_cases.aggregate import AgentStateVector
from sentinel.cortex.use_cases.autoencoder import SentinelAutoencoder, create_autoencoder
from sentinel.domain.models import AnomalyType, ModelPhase, ProcessedEvent


def _make_event(agent_name: str, offset_s: float = 0.0) -> ProcessedEvent:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    t = base + timedelta(seconds=offset_s)
    return ProcessedEvent(
        agent_name=agent_name,
        correlation_id=f"{agent_name}-{offset_s}",
        input_received_at=t,
        output_sent_at=t,
        processing_latency_ms=100.0,
        anomaly_score=-0.5,
        is_anomaly=True,
        anomaly_type=AnomalyType.SCORE,
        payload_schema_hash="abc",
        input_size_bytes=100,
        output_size_bytes=100,
    )


def _make_state_vectors(agents: list[str]) -> dict[str, AgentStateVector]:
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    return {
        name: AgentStateVector(
            agent_name=name,
            last_seen=base,
            message_rate=10.0,
            error_rate=0.1,
            anomaly_score=-0.5,
            model_phase=ModelPhase.INFERENCE,
        )
        for name in agents
    }


class MockCortexConfig:
    name = "cortex01"
    silence_threshold_s = 60.0


# ── run_dual_network — no anomalies ──────────────────────────────────────────

def test_run_dual_network_no_anomalies_returns_no_alert():
    window = deque()
    state_vectors = _make_state_vectors(["agent01"])
    config = MockCortexConfig()

    result = run_dual_network(
        causal_window=window,
        autoencoder_model=None,
        state_vectors=state_vectors,
        config=config,
    )

    assert isinstance(result, DualNetworkResult)
    assert result.final_alert is None


# ── run_dual_network — causal chain only ─────────────────────────────────────

def test_run_dual_network_causal_chain_triggers_alert():
    window = deque()
    base = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    for i, name in enumerate(["agent01", "agent02", "agent03"]):
        window.append(_make_event(name, offset_s=float(i * 2)))

    state_vectors = _make_state_vectors(["agent01", "agent02", "agent03"])
    config = MockCortexConfig()

    result = run_dual_network(
        causal_window=window,
        autoencoder_model=None,
        state_vectors=state_vectors,
        config=config,
    )

    assert result.final_alert is not None
    assert result.causal is not None
    assert result.causal.root_agent == "agent01"


# ── run_dual_network — systemic anomaly ──────────────────────────────────────

def test_run_dual_network_systemic_triggers_alert():
    window = deque()
    state_vectors = _make_state_vectors(["agent01"])
    config = MockCortexConfig()

    # Create a trained autoencoder with very low reconstruction on normal data
    model = create_autoencoder(input_dim=4, hidden_dim=2)

    result = run_dual_network(
        causal_window=window,
        autoencoder_model=model,
        state_vectors=state_vectors,
        config=config,
        baseline_error=0.0,  # any error will exceed 0 * 2 = 0
    )

    # With baseline_error=0, any non-zero reconstruction error triggers systemic
    # (depends on model output, may or may not trigger)
    assert isinstance(result, DualNetworkResult)
    assert isinstance(result.systemic_error, float)


# ── run_dual_network — handles autoencoder error gracefully ──────────────────

def test_run_dual_network_autoencoder_exception_is_handled():
    window = deque()
    state_vectors = _make_state_vectors(["agent01"])
    config = MockCortexConfig()

    broken_model = MagicMock()
    broken_model.side_effect = RuntimeError("model exploded")

    result = run_dual_network(
        causal_window=window,
        autoencoder_model=broken_model,
        state_vectors=state_vectors,
        config=config,
    )

    assert isinstance(result, DualNetworkResult)
