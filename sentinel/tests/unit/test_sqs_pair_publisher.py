"""Unit tests for SqsPairPublisher client pooling."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from sentinel.adapters.queue.sqs_pair_publisher import SqsPairPublisher
from sentinel.domain.models import CorrelatedPair, InputCapture


def _make_pair() -> CorrelatedPair:
    now = datetime.now(tz=timezone.utc)
    return CorrelatedPair(
        engine_name="engine01",
        correlation_id="abc-123",
        inputs=[InputCapture(name="data", body={"id": "abc-123"}, received_at=now, size_bytes=10)],
        output_body={},
        output_received_at=now,
        output_size_bytes=0,
        processing_latency_ms=5.0,
        timed_out=False,
    )


@pytest.fixture
def mock_sqs_client():
    client = AsyncMock()
    client.send_message = AsyncMock(return_value={})
    client.__aexit__ = AsyncMock(return_value=None)
    return client


@pytest.fixture
def mock_session(mock_sqs_client):
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=mock_sqs_client)
    cm.__aexit__ = AsyncMock(return_value=None)

    session = MagicMock()
    session.create_client = MagicMock(return_value=cm)
    return session


@pytest.mark.asyncio
async def test_publish_reuses_client(mock_session, mock_sqs_client):
    """publish() must reuse the same SQS client across multiple calls."""
    publisher = SqsPairPublisher(queue_url="http://localhost:4566/queue/test", endpoint_url="http://localhost:4566")

    with patch("aiobotocore.session.get_session", return_value=mock_session):
        pair = _make_pair()
        await publisher.publish(pair)
        await publisher.publish(pair)
        await publisher.publish(pair)

    # create_client should only be called once despite three publish() calls
    assert mock_session.create_client.call_count == 1
    assert mock_sqs_client.send_message.call_count == 3


@pytest.mark.asyncio
async def test_publish_sends_proto_payload(mock_session, mock_sqs_client):
    """publish() must send a hex-encoded protobuf body."""
    publisher = SqsPairPublisher(queue_url="http://localhost:4566/queue/test", endpoint_url="http://localhost:4566")

    with patch("aiobotocore.session.get_session", return_value=mock_session):
        await publisher.publish(_make_pair())

    call_kwargs = mock_sqs_client.send_message.call_args.kwargs
    body = json.loads(call_kwargs["MessageBody"])
    assert "_proto" in body
    assert isinstance(body["_proto"], str)
    assert len(body["_proto"]) > 0


@pytest.mark.asyncio
async def test_close_releases_client(mock_session, mock_sqs_client):
    """close() must call __aexit__ on the cached client."""
    publisher = SqsPairPublisher(queue_url="http://localhost:4566/queue/test", endpoint_url="http://localhost:4566")

    with patch("aiobotocore.session.get_session", return_value=mock_session):
        await publisher.publish(_make_pair())
        await publisher.close()

    mock_sqs_client.__aexit__.assert_awaited_once()
    assert publisher._sqs_client is None


@pytest.mark.asyncio
async def test_close_is_idempotent(mock_session):
    """close() on a publisher that never published must not raise."""
    publisher = SqsPairPublisher(queue_url="http://localhost:4566/queue/test")
    await publisher.close()  # no error
