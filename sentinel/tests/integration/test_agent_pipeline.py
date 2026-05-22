"""Integration tests for the agent pipeline.

Requires Redis and LocalStack running via docker-compose.test.yml.
Run with: pytest tests/integration
"""
import asyncio
import json
from datetime import datetime, timedelta, timezone

import pytest
import pytest_asyncio

from sentinel.adapters.store.redis_store import RedisCorrelationStore
from sentinel.agent.correlator import (
    make_correlation_key,
    retrieve_input,
    store_input,
)
from sentinel.agent.use_cases.process_message import (
    AgentState,
    process_input_message,
    process_output_message,
)
from sentinel.config.schema import AgentConfig, ModelIsolationForestConfig, ReportingConfig
from sentinel.domain.models import AnomalyType, ModelPhase, ProcessedEvent
from sentinel.domain.ports.model_store import IModelStore
from sentinel.domain.ports.reporter import IReporter


# ── Fixtures ──────────────────────────────────────────────────────────────────

class MockModelStore(IModelStore):
    async def save_version(self, name, model):
        return "v_test"

    async def load(self, name, version=None):
        return None

    async def list_versions(self, name):
        return []

    async def rollback(self, name):
        pass


class MockReporter(IReporter):
    def __init__(self):
        self.events: list[ProcessedEvent] = []

    async def publish_event(self, event):
        self.events.append(event)

    async def publish_heartbeat(self, heartbeat):
        pass


@pytest_asyncio.fixture
async def real_redis_store():
    store = RedisCorrelationStore(host="localhost", port=6379)
    # Flush test keys
    yield store
    await store.close()


def _agent_config():
    return AgentConfig(
        name="test_agent",
        correlation_field="correlationId",
        correlation_ttl_s=2,  # short TTL for timeout tests
        phase="TRAINING",
        reporting=ReportingConfig(heartbeat_interval_s=30, send_all_events=True),
    )


def _if_config():
    return ModelIsolationForestConfig(
        n_estimators=10,
        contamination="auto",
        anomaly_threshold=-0.1,
        training_window=20,
    )


def _state():
    return AgentState(
        agent_name="test_agent",
        phase=ModelPhase.TRAINING,
        model=None,
        training_buffer=[],
    )


# ── Test 1: Store and retrieve correlation ────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_stores_input_and_retrieves_on_output(real_redis_store):
    config = _agent_config()
    reporter = MockReporter()
    model_store = MockModelStore()
    state = _state()

    input_body = {"correlationId": "int-test-1", "payload": "hello"}
    output_body = {"correlationId": "int-test-1", "response": "world"}

    input_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    output_time = datetime(2024, 1, 1, 12, 0, 1, tzinfo=timezone.utc)

    await process_input_message(input_body, input_time, config, real_redis_store)

    # Verify it was stored
    key = make_correlation_key("test_agent", "int-test-1")
    stored = await retrieve_input(real_redis_store, key)
    assert stored is not None

    new_state = await process_output_message(
        raw_body=output_body,
        received_at=output_time,
        agent_config=config,
        if_config=_if_config(),
        state=state,
        correlation_store=real_redis_store,
        model_store=model_store,
        reporter=reporter,
    )

    assert new_state.message_count == 1
    assert len(reporter.events) == 1
    event = reporter.events[0]
    assert event.correlation_id == "int-test-1"
    assert event.is_anomaly is False


# ── Test 2: Timeout detection ─────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_detects_timeout_when_ttl_expires(real_redis_store):
    config = _agent_config()  # TTL = 2s
    reporter = MockReporter()
    model_store = MockModelStore()
    state = _state()

    input_body = {"correlationId": "int-timeout-1", "payload": "hello"}
    input_time = datetime.now(tz=timezone.utc)

    await process_input_message(input_body, input_time, config, real_redis_store)

    # Wait for TTL to expire
    await asyncio.sleep(3)

    output_body = {"correlationId": "int-timeout-1", "response": "late"}
    new_state = await process_output_message(
        raw_body=output_body,
        received_at=datetime.now(tz=timezone.utc),
        agent_config=config,
        if_config=_if_config(),
        state=state,
        correlation_store=real_redis_store,
        model_store=model_store,
        reporter=reporter,
    )

    assert len(reporter.events) == 1
    assert reporter.events[0].anomaly_type == AnomalyType.TIMEOUT
    assert reporter.events[0].is_anomaly is True


# ── Test 3: Standalone mode when gRPC unavailable ────────────────────────────

@pytest.mark.asyncio
@pytest.mark.integration
async def test_agent_standalone_mode_when_grpc_unavailable(real_redis_store):
    """Agent should continue processing even when gRPC reporter fails."""
    from sentinel.adapters.grpc.client import GrpcReporter

    config = _agent_config()
    # Use a GrpcReporter pointing to a non-existent server
    reporter = GrpcReporter(host="localhost", port=59999)
    model_store = MockModelStore()
    state = _state()

    input_body = {"correlationId": "int-standalone-1", "payload": "hello"}
    output_body = {"correlationId": "int-standalone-1", "response": "world"}

    input_time = datetime.now(tz=timezone.utc)
    output_time = datetime.now(tz=timezone.utc) + timedelta(seconds=0.1)

    await process_input_message(input_body, input_time, config, real_redis_store)

    # Should not raise even when gRPC fails
    new_state = await process_output_message(
        raw_body=output_body,
        received_at=output_time,
        agent_config=config,
        if_config=_if_config(),
        state=state,
        correlation_store=real_redis_store,
        model_store=model_store,
        reporter=reporter,
    )

    # Processing should complete despite gRPC failure
    assert new_state.message_count == 1


# ── Test 4: Full pipeline ─────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.integration
async def test_full_pipeline_input_correlate_score_publish(real_redis_store):
    """Full pipeline: input → correlate → accumulate → publish."""
    config = _agent_config()
    reporter = MockReporter()
    model_store = MockModelStore()
    state = _state()

    base_time = datetime(2024, 6, 1, 10, 0, 0, tzinfo=timezone.utc)

    # Process multiple pairs to accumulate training data
    for i in range(5):
        corr_id = f"pipeline-{i}"
        input_body = {"correlationId": corr_id, "seq": i}
        output_body = {"correlationId": corr_id, "result": i * 2}

        input_time = base_time + timedelta(seconds=i * 10)
        output_time = input_time + timedelta(milliseconds=100)

        await process_input_message(input_body, input_time, config, real_redis_store)

        state = await process_output_message(
            raw_body=output_body,
            received_at=output_time,
            agent_config=config,
            if_config=_if_config(),
            state=state,
            correlation_store=real_redis_store,
            model_store=model_store,
            reporter=reporter,
        )

    assert state.message_count == 5
    assert len(reporter.events) == 5
    assert len(state.training_buffer) == 5
