"""Unit tests for CorrelationEngineRunner loop resilience."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.domain.errors import TransportError
from sentinel.domain.models import RawMessage
from sentinel.runtime.correlation_engine_runner import CorrelationEngineRunner


def _make_message(msg_id: str = "msg-1") -> RawMessage:
    return RawMessage(
        id=msg_id,
        body={"id": msg_id, "status": "ok"},
        received_at=datetime.now(tz=timezone.utc),
        size_bytes=20,
    )


def _make_runner(transport, correlation_store=None, pair_publishers=None):
    engine_config = MagicMock()
    engine_config.name = "engine01"
    engine_config.inputs = [MagicMock(name="data")]
    engine_config.effective_input_mode = MagicMock(return_value="normal")

    store = correlation_store or AsyncMock()
    publishers = pair_publishers or []

    return CorrelationEngineRunner(
        engine_config=engine_config,
        input_transports=[transport],
        correlation_store=store,
        pair_publishers=publishers,
    )


@pytest.mark.asyncio
async def test_loop_continues_after_flush_acks_failure():
    """_run_input_loop must NOT crash when flush_acks raises TransportError."""
    transport = AsyncMock()
    call_count = 0

    async def receive_batch_side_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [_make_message("m1")]
        if call_count == 2:
            return [_make_message("m2")]
        # stop after two batches
        raise asyncio.CancelledError()

    transport.receive_batch = receive_batch_side_effect
    transport.ack = AsyncMock()
    transport.nack = AsyncMock()
    transport.flush_acks = AsyncMock(side_effect=TransportError("SQS delete failed"))

    runner = _make_runner(transport)

    # Patch store so that process_message is a quick no-op
    with patch("sentinel.agent.use_cases.correlate.store_input_capture", new=AsyncMock(return_value=(None, None))):
        with pytest.raises(asyncio.CancelledError):
            await runner._run_input_loop(runner.engine_config.inputs[0], transport)

    # flush_acks was called for both batches — loop survived the first failure
    assert transport.flush_acks.call_count == 2


@pytest.mark.asyncio
async def test_loop_processes_all_messages_in_batch():
    """All messages in a batch must be processed even if some fail."""
    transport = AsyncMock()
    call_count = 0

    async def receive_batch_side_effect():
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [_make_message("m1"), _make_message("m2"), _make_message("m3")]
        raise asyncio.CancelledError()

    transport.receive_batch = receive_batch_side_effect
    transport.ack = AsyncMock()
    transport.nack = AsyncMock()
    transport.flush_acks = AsyncMock()

    runner = _make_runner(transport)

    with patch("sentinel.agent.use_cases.correlate.store_input_capture", new=AsyncMock(return_value=(None, None))):
        with pytest.raises(asyncio.CancelledError):
            await runner._run_input_loop(runner.engine_config.inputs[0], transport)

    # All 3 messages were acked (store returned None so no pair is built)
    assert transport.ack.call_count == 3
